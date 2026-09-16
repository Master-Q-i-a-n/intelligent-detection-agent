"""综合运行 RAG 检索与 Agent 端到端评测。"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import time
import uuid
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

import duckdb

from ..conversation_agent.agent import ConversationAgentService
from ..conversation_agent.schemas import ChatResumeRequest
from ..rag.evaluate import DEFAULT_CASES_PATH, evaluate as evaluate_rag, load_cases as load_rag_cases
from ..rag.pipeline import PROJECT_ROOT, RagConfig, RagPipeline
from ..safety_operations.env import load_project_env
from .grading import FRAMEWORK_TOOLS, distribution, grade_arguments, grade_tool_policy
from .judge import EvaluationJudge
from .models import AgentEvalCase, ArgumentAssertion, JudgeResult, ToolRequirement, DistillationConfig, JointJudgeResult
from .distillation import TrajectoryCollector, score_distillation
from .telemetry import TurnTelemetry


AGENT_CASES_PATH = Path(__file__).with_name("agent_eval_cases.json")
NEW_RAG_IDS = {
    "gas_system_06",
    "gas_system_07",
    "gas_system_08",
    "ultrasonic_06",
    "ultrasonic_07",
    "ultrasonic_08",
    "temperature_06",
    "temperature_07",
    "pressure_06",
    "pressure_07",
}
DOMAIN_TOOLS = {
    "get_current_time",
    "describe_data_source",
    "query_business_data",
    "query_diagnosis_data",
    "query_security_data",
    "search_technical_documents",
    "ask_user",
    "create_work_order",
    "build_report_artifact",
}
RAG_QUERY_TERMS = {
    "gas_system_06": ["接地", "电阻"],
    # 检索参数只校验用户问题中的概念，不能要求模型预先知道待查询的答案。
    "gas_system_07": ["数据采集处理装置"],
    "gas_system_08": ["发热量", "离线", "在线"],
    "ultrasonic_06": ["超声流量计", "组态", "记录"],
    "ultrasonic_07": ["超声流量计", "参数", "基础资料"],
    "ultrasonic_08": ["超声流量计", "实流校准", "脉动流"],
    "temperature_06": ["Loop Test", "温度变送器"],
    "temperature_07": ["X-Well", "Hot Backup"],
    "pressure_06": ["Rosemount 3051", "回路电阻"],
    "pressure_07": ["压力变送器", "导压管"],
}


def load_agent_cases(path: Path = AGENT_CASES_PATH) -> list[AgentEvalCase]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError("Agent 评测集必须是 JSON 数组")
    cases = [AgentEvalCase.model_validate(value) for value in values]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Agent 评测用例 ID 不能重复")
    return cases


def build_rag_agent_cases() -> list[AgentEvalCase]:
    """将新增的十条检索标注转换成端到端 Agent 用例。"""

    cases = []
    for value in load_rag_cases(DEFAULT_CASES_PATH):
        case_id = str(value["id"])
        if case_id not in NEW_RAG_IDS:
            continue
        cases.append(
            AgentEvalCase(
                id=f"agent_{case_id}",
                category="rag",
                description=f"通过技术文档回答：{value['query']}",
                turns=[{"message": value["query"]}],
                required_tools=[
                    ToolRequirement(name="search_technical_documents", min_calls=1, max_calls=2),
                    ToolRequirement(name="read_file", min_calls=1),
                ],
                allowed_tools=["search_technical_documents"],
                forbidden_tools=[
                    "create_work_order",
                ],
                argument_assertions=[
                    ArgumentAssertion(
                        tool="search_technical_documents",
                        argument="query",
                        kind="contains_all",
                        expected=RAG_QUERY_TERMS[case_id],
                    )
                ],
                artifact_types=["rag_retrieval"],
                judge=True,
                judge_criteria=["仅依据检索资料", "关键事实正确", "对应句段有资料引用", "末尾列出参考资料名称"],
                rag_relevant=value["relevant"],
                rag_expected_answer=value.get("expected_answer"),
                require_source_list=True,
                require_image_markdown=case_id == "pressure_06",
            )
        )
    if len(cases) != 10:
        raise ValueError(f"新增 RAG Agent 用例应为10条，实际为{len(cases)}条")
    return cases


def _query_rows(root: Path, source: str, sql: str) -> dict[str, Any]:
    if source == "security":
        with sqlite3.connect(root / "safety_operations" / "data" / "security.db") as connection:
            cursor = connection.execute(sql)
            columns = [item[0] for item in cursor.description or []]
            rows = cursor.fetchall()
    else:
        filename = "gas_ai_input.duckdb" if source == "business" else "gas_ai_results.duckdb"
        with duckdb.connect(str(root / "database" / filename), read_only=True) as connection:
            cursor = connection.execute(sql)
            columns = [item[0] for item in cursor.description]
            rows = cursor.fetchall()
    return {
        "source": source,
        "columns": columns,
        "rows": [list(row) for row in rows],
    }


def execute_oracles(root: Path, case: AgentEvalCase) -> list[dict[str, Any]]:
    return [_query_rows(root, oracle.source, oracle.sql) for oracle in case.oracles]


def dataset_fingerprint(root: Path) -> dict[str, Any]:
    """记录评测事实源的规模和日期边界，便于不同报告间追溯。"""

    return {
        "business": _query_rows(
            root,
            "business",
            "SELECT MIN(data_date) AS start_date,MAX(data_date) AS end_date,COUNT(*) AS row_count,COUNT(DISTINCT user_id) AS user_count FROM telemetry.scada_observation",
        ),
        "metering": _query_rows(
            root,
            "diagnosis",
            "SELECT MIN(diagnosis_date) AS start_date,MAX(diagnosis_date) AS end_date,COUNT(*) AS run_count,COUNT(DISTINCT user_id) AS user_count FROM metering.diagnosis_run",
        ),
        "equipment": _query_rows(
            root,
            "diagnosis",
            "SELECT MIN(diagnosis_date) AS start_date,MAX(diagnosis_date) AS end_date,COUNT(*) AS run_count,COUNT(DISTINCT user_id) AS user_count FROM equipment.health_diagnosis",
        ),
        "security": _query_rows(
            root,
            "security",
            "SELECT COUNT(*) AS event_count,SUM(CASE WHEN final_decision='CONFIRMED' THEN 1 ELSE 0 END) AS confirmed_count FROM security_events",
        ),
        "conversation_work_orders": _query_rows(
            root,
            "diagnosis",
            "SELECT COUNT(*) AS work_order_count FROM operations.work_order",
        ),
    }


def preflight(
    root: Path,
    cases: list[AgentEvalCase],
    *,
    check_rag: bool,
    check_agent: bool,
) -> dict[str, Any]:
    """不调用 LLM，检查数据、Qdrant、用例结构和参考查询。"""

    load_project_env(root / ".env")
    checks: list[dict[str, Any]] = []
    for path in (
        root / "database" / "gas_ai_input.duckdb",
        root / "database" / "gas_ai_results.duckdb",
        root / "safety_operations" / "data" / "security.db",
    ):
        checks.append({"name": str(path.relative_to(root)), "passed": path.is_file()})
    oracle_failures = []
    for case in cases:
        try:
            execute_oracles(root, case)
        except Exception as exc:
            oracle_failures.append({"case_id": case.id, "error": f"{type(exc).__name__}: {exc}"})
    checks.append({"name": "oracle_queries", "passed": not oracle_failures, "failures": oracle_failures})

    qdrant: dict[str, Any] | None = None
    if check_rag:
        try:
            pipeline = RagPipeline(RagConfig.from_env())
            pipeline.check_qdrant()
            collection = pipeline.qdrant.get_collection(pipeline.config.collection_name)
            qdrant = {
                "url": pipeline.config.qdrant_url,
                "collection": pipeline.config.collection_name,
                "points_count": getattr(collection, "points_count", None),
            }
            checks.append({"name": "qdrant", "passed": True})
        except Exception as exc:
            checks.append({"name": "qdrant", "passed": False, "error": f"{type(exc).__name__}: {exc}"})
        dashscope_configured = bool(os.getenv("DASHSCOPE_API_KEY") and os.getenv("DASHSCOPE_WORKSPACE_ID"))
        checks.append({"name": "dashscope", "passed": dashscope_configured})
    if check_agent:
        model_configured = bool(
            os.getenv("CHAT_LLM_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("LLM_API_KEY")
        )
        checks.append({"name": "chat_model", "passed": model_configured})
    return {
        "passed": all(bool(item["passed"]) for item in checks),
        "checks": checks,
        "qdrant": qdrant,
        "agent_case_count": len(cases),
        "rag_retrieval_case_count": len(load_rag_cases(DEFAULT_CASES_PATH)) if check_rag else 0,
        "dataset_fingerprint": dataset_fingerprint(root) if not oracle_failures else None,
    }


def _artifact_key(artifact: dict[str, Any]) -> tuple[str, str]:
    return str(artifact.get("type") or ""), str(artifact.get("id") or "")


def _unique_tool_calls(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """HITL resume 会重放历史消息，按 tool_call_id 只统计一次逻辑调用。"""

    calls: list[dict[str, Any]] = []
    seen: set[str] = set()
    for turn in turns:
        for call in turn["telemetry"]["tool_calls"]:
            call_id = str(call.get("tool_call_id") or "")
            if call_id and call_id in seen:
                continue
            if call_id:
                seen.add(call_id)
            calls.append(call)
    return calls


async def _run_turn(
    service: ConversationAgentService,
    user_id: str,
    thread_id: str,
    turn: Any,
    trajectory: TrajectoryCollector | None = None,
) -> dict[str, Any]:
    telemetry = TurnTelemetry(trajectory=trajectory)
    if turn.resume:
        request = ChatResumeRequest(thread_id=thread_id, **turn.resume)
        source = service.stream_resume(user_id, request, telemetry=telemetry)
    else:
        source = service.stream_turn(user_id, thread_id, str(turn.message), telemetry=telemetry)
    events = []
    try:
        async for event in source:
            events.append(event)
    except Exception as exc:
        # 保留失败前已收到的事件及真实指标，不记录可能含凭证的异常正文。
        telemetry.finish("error")
        events.append({"event": "error", "data": {"message": f"用例执行异常：{type(exc).__name__}"}})
    done = next((event["data"] for event in reversed(events) if event["event"] == "done"), None)
    error = next((event["data"] for event in reversed(events) if event["event"] == "error"), None)
    artifacts = [event["data"] for event in events if event["event"] == "artifact"]
    return {
        "expected_status": turn.expected_status,
        "actual_status": (done or {}).get("status") or ("error" if error else telemetry.status),
        "answer": str((done or {}).get("message") or ""),
        "interrupt": (done or {}).get("interrupt"),
        "artifacts": artifacts,
        "error": error,
        "telemetry": telemetry.as_dict(),
    }


def _grade_rag(case: AgentEvalCase, artifacts: list[dict[str, Any]], answer: str) -> list[str]:
    failures = []
    rag_artifacts = [item for item in artifacts if item.get("type") == "rag_retrieval"]
    if not rag_artifacts:
        return ["缺少 rag_retrieval 产物"]
    results = [
        result
        for artifact in rag_artifacts
        for result in ((artifact.get("payload") or {}).get("results") or [])
    ]
    retrieved = {
        (str(item.get("source") or ""), str(item.get("chunk_id") or ""))
        for item in results
    }
    # 端到端回答只要求取得正确来源并被语义评审验证；精确 chunk 排名由独立 RAG 指标负责。
    expected_sources = {item["source"] for item in case.rag_relevant}
    retrieved_sources = {source for source, _chunk_id in retrieved}
    missing_sources = expected_sources - retrieved_sources
    if missing_sources:
        failures.append(f"Agent 检索结果缺少相关来源：{sorted(missing_sources)}")
    if "[资料" not in answer:
        failures.append("回答缺少 [资料N] 引用")
    source_names = {name for source in expected_sources for name in {source, Path(source).stem}}
    if case.require_source_list and not any(source_name in answer for source_name in source_names):
        failures.append("回答末尾缺少参考资料名称")
    if case.require_image_markdown and not re.search(r"!\[[^\]]*\]\([^\)]+\)", answer):
        failures.append("应展示的资料图片没有使用 Markdown 内联")
    return failures


def _hard_grade(case: AgentEvalCase, turns: list[dict[str, Any]]) -> dict[str, Any]:
    answers = "\n".join(turn["answer"] for turn in turns if turn["answer"])
    artifacts_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for turn in turns:
        for artifact in turn["artifacts"]:
            artifacts_by_key[_artifact_key(artifact)] = artifact
    artifacts = list(artifacts_by_key.values())
    calls = _unique_tool_calls(turns)
    failures = []
    if len(turns) != len(case.turns):
        failures.append(f"对话轮次未完成：{len(turns)} / {len(case.turns)}")
    for index, turn in enumerate(turns):
        if turn["actual_status"] != turn["expected_status"]:
            failures.append(
                f"第{index + 1}轮状态错误：{turn['actual_status']} != {turn['expected_status']}"
            )
        if turn["error"]:
            failures.append(f"第{index + 1}轮执行错误：{turn['error']}")
    actual_types = Counter(str(item.get("type") or "") for item in artifacts)
    for artifact_type in case.artifact_types:
        if actual_types[artifact_type] < 1:
            failures.append(f"缺少产物：{artifact_type}")
    for text in case.answer_contains:
        expected_text = text.lower()
        answer_text = answers.lower()
        # 日期只比较完整年月日，允许中文、斜线和未补零的展示形式。
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", expected_text):
            dates = {
                f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
                for y, m, d in re.findall(r"(?<!\d)(\d{4})\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})(?!\d)", answers)
            }
            if expected_text not in dates:
                failures.append(f"答案缺少日期：{text}")
            continue
        # 展示层允许千位分隔符，不能让 127,965.56 与 127965.56 被判为不同事实。
        if expected_text not in answer_text and expected_text.replace(",", "") not in answer_text.replace(",", ""):
            failures.append(f"答案缺少：{text}")
    # 仅消除合法三位分组中的横向空白，不合并换行或任意相邻数字。
    numeric_text = re.sub(
        r"(?<![\w.])(-?\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:\.\d+)?)(?![\w.])",
        lambda match: re.sub(r"[ \u00a0\u202f]", "", match.group()), answers,
    )
    answer_values = [
        float(value.replace(",", ""))
        for value in re.findall(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?", numeric_text)
    ]
    for assertion in case.answer_numbers:
        if not any(abs(value - assertion.expected) <= assertion.absolute_tolerance for value in answer_values):
            failures.append(
                f"答案缺少数值：{assertion.expected}（允许误差 ±{assertion.absolute_tolerance}）"
            )
    for pattern in case.answer_regex:
        if re.search(pattern, answers, re.IGNORECASE | re.DOTALL) is None:
            failures.append(f"答案未匹配：{pattern}")

    tool_grade = grade_tool_policy(case, calls)
    argument_grade = grade_arguments(case, calls)
    failures.extend(tool_grade["failures"])
    failures.extend(
        f"参数错误 {item['tool']}.{item['argument']}：{item['reason']}"
        for item in argument_grade["assertions"]
        if not item["passed"]
    )
    if case.category == "rag":
        failures.extend(_grade_rag(case, artifacts, answers))
    return {
        "passed": not failures,
        "failures": failures,
        "tool": tool_grade,
        "arguments": argument_grade,
        "answers": answers,
        "artifacts": artifacts,
    }


def _judge_execution_context(turns: list[dict[str, Any]]) -> dict[str, Any]:
    """只给评审器必要的可观测执行事实，不传隐藏推理或完整工具结果。"""

    tool_counts = Counter(call["name"] for call in _unique_tool_calls(turns))
    artifact_types = Counter(
        str(artifact.get("type") or "")
        for turn in turns
        for artifact in turn["artifacts"]
    )
    return {
        "turns": [
            {
                "expected_status": turn["expected_status"],
                "actual_status": turn["actual_status"],
                "has_interrupt": bool(turn["interrupt"]),
                "error": turn["error"],
            }
            for turn in turns
        ],
        "tool_counts": dict(tool_counts),
        "artifact_types": dict(artifact_types),
    }


async def _run_sync(function, *args):
    """取消调用者时仍等待后台数据库操作结束，避免提前关闭共享连接。"""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


async def run_agent_case(
    service: ConversationAgentService,
    judge: EvaluationJudge | None,
    root: Path,
    case: AgentEvalCase,
    *,
    repeat_index: int,
    run_id: str,
    distillation: DistillationConfig | None = None,
    on_result: Callable[[dict[str, Any], dict[str, Any] | None], None] | None = None,
) -> dict[str, Any]:
    user_id = f"evaluation-{run_id}"
    thread_id = f"eval_{case.id}_{repeat_index}_{uuid.uuid4().hex[:8]}"
    turns = []
    trajectory = TrajectoryCollector() if distillation else None
    oracle_results = []
    sample = None
    started = time.monotonic()
    try:
        try:
            oracle_results = await _run_sync(execute_oracles, root, case)
            if case.rag_expected_answer:
                oracle_results.append(
                    {
                        "source": "rag_ground_truth",
                        "expected_answer": case.rag_expected_answer,
                        "relevant": case.rag_relevant,
                    }
                )
            for turn in case.turns:
                turns.append(await _run_turn(service, user_id, thread_id, turn, trajectory))
                if turns[-1]["error"]:
                    break
            hard = _hard_grade(case, turns)
            judge_result: JudgeResult | None = None
            sample = None
            distillation_result = None
            if distillation is not None:
                trace_error = None
                judge_error = None
                joint = None
                try:
                    sample = trajectory.sample(expected_interrupt=case.turns[-1].expected_status == "interrupted")
                except ValueError as exc:
                    trace_error = str(exc)
                if hard["passed"] and sample is not None:
                    try:
                        if judge is None:
                            raise ValueError("蒸馏模式需要联合 Judge")
                        joint = await judge.grade_trajectory(case, oracle_results, sample, {
                            **_judge_execution_context(turns),
                            "tool_calls": _unique_tool_calls(turns),
                        })
                        judge_result = JudgeResult.model_validate(joint.model_dump())
                    except Exception as exc:
                        # 不保存供应商异常正文，避免错误对象带入凭证或完整请求。
                        judge_error = str(exc) if isinstance(exc, ValueError) else f"联合评审调用失败：{type(exc).__name__}"
                distillation_result = score_distillation(
                    case, _unique_tool_calls(turns), hard["passed"], joint, distillation,
                    trace_error=trace_error, judge_error=judge_error,
                )
                distillation_result["models"] = list(dict.fromkeys(c["model"] for c in trajectory.calls.values() if c.get("model")))
                distillation_result["judge_model"] = getattr(judge, "model_name", None)
                distillation_result["model_config"] = {
                    name: getattr(service, name, None)
                    for name in ("model_name", "provider", "thinking_mode", "reasoning_effort", "chat_template_kwargs", "stream_usage")
                }
            elif judge is not None and case.judge:
                execution_error = any(turn["error"] for turn in turns)
                if execution_error or not hard["answers"].strip():
                    judge_result = JudgeResult(
                        score=1,
                        passed=False,
                        reason="执行报错或最终答案为空，未调用外部评审模型。",
                        usage={},
                        latency_ms=0,
                    )
                else:
                    judge_result = await judge.grade(
                        case,
                        hard["answers"],
                        oracle_results,
                        _judge_execution_context(turns),
                    )
            success = hard["passed"] and (judge_result is None or (judge_result.passed and judge_result.score >= 4))
            if distillation is not None:
                success = hard["passed"] and judge_result is not None and judge_result.passed and judge_result.score >= 4
            row = {
                "id": case.id,
                "category": case.category,
                "description": case.description,
                "repeat": repeat_index,
                "success": success,
                "hard_grade": hard,
                "judge": judge_result.model_dump(mode="json") if judge_result else None,
                "oracle_results": oracle_results,
                "turns": turns,
            }
            if distillation_result is not None:
                row["distillation"] = distillation_result
        except Exception as exc:
            error = f"用例执行异常：{type(exc).__name__}"
            hard = _hard_grade(case, turns)
            hard["passed"] = False
            hard["failures"].append(error)
            row = {
                "id": case.id, "category": case.category, "description": case.description,
                "repeat": repeat_index, "success": False, "hard_grade": hard,
                "judge": None, "oracle_results": oracle_results, "turns": turns,
                "execution_error": error,
            }
            if distillation is not None:
                row["distillation"] = score_distillation(
                    case, _unique_tool_calls(turns), False, None, distillation, judge_error=error,
                )
        row["execution_ms"] = round((time.monotonic() - started) * 1000, 2)
        row["thread_id"] = thread_id
        if on_result is not None:
            on_result(row, sample)
        return row
    finally:
        await _run_sync(service.delete_thread, user_id, thread_id)


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"case_count": 0}
    llm_counts = [sum(turn["telemetry"]["llm_call_count"] for turn in row["turns"]) for row in rows]
    tool_counts = [len(_unique_tool_calls(row["turns"])) for row in rows]
    domain_tool_counts = [
        sum(1 for call in _unique_tool_calls(row["turns"]) if call["name"] in DOMAIN_TOOLS)
        for row in rows
    ]
    scenario_latency = [
        sum(float(turn["telemetry"]["end_to_end_ms"] or 0) for turn in row["turns"])
        for row in rows
        if row["turns"] and all(turn["telemetry"]["end_to_end_ms"] is not None for turn in row["turns"])
    ]
    first_tokens = [
        float(turn["telemetry"]["first_token_ms"])
        for row in rows
        for turn in row["turns"]
        if turn["telemetry"]["first_token_ms"] is not None
    ]
    parameter_assertions = [
        assertion
        for row in rows
        for assertion in row["hard_grade"]["arguments"]["assertions"]
    ]
    model_usage = [
        model_call["usage"]
        for row in rows
        for turn in row["turns"]
        for model_call in turn["telemetry"]["model_calls"]
    ]
    usage_summary = {}
    for key in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens", "total_tokens"):
        known = [int(item[key]) for item in model_usage if item.get(key) is not None]
        usage_summary[key] = {
            "total": sum(known),
            "coverage": len(known) / len(model_usage) if model_usage else 0.0,
        }
    return {
        "case_count": len(rows),
        **({"distillation": _summarize_distillation(rows)} if any("distillation" in row for row in rows) else {}),
        "task_success_rate": sum(1 for row in rows if row["success"]) / len(rows),
        "tool_call_accuracy": sum(1 for row in rows if row["hard_grade"]["tool"]["passed"]) / len(rows),
        "tool_budget_compliance": (
            sum(1 for row in rows if row["hard_grade"]["tool"].get("budget_passed", True)) / len(rows)
        ),
        "tool_precision": fmean(float(row["hard_grade"]["tool"]["precision"]) for row in rows),
        "tool_recall": fmean(float(row["hard_grade"]["tool"]["recall"]) for row in rows),
        "parameter_accuracy": (
            sum(1 for item in parameter_assertions if item["passed"]) / len(parameter_assertions)
            if parameter_assertions
            else 1.0
        ),
        "llm_calls": {"total": sum(llm_counts), **distribution([float(value) for value in llm_counts])},
        "tool_calls": {"total": sum(tool_counts), **distribution([float(value) for value in tool_counts])},
        "domain_tool_calls": {
            "total": sum(domain_tool_counts),
            **distribution([float(value) for value in domain_tool_counts]),
        },
        "end_to_end_ms": distribution(scenario_latency),
        "first_token_ms": {**distribution(first_tokens), "coverage": len(first_tokens) / max(1, sum(len(row["turns"]) for row in rows))},
        "token_usage": usage_summary,
    }


def _summarize_distillation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [row["distillation"] for row in rows if "distillation" in row]
    return {
        "case_count": len(values),
        "selected_count": sum(value["selected"] for value in values),
        "selected_rate": sum(value["selected"] for value in values) / len(values) if values else 0,
        "exported_count": sum(value["exported"] for value in values),
        "rejection_reasons": dict(Counter(reason for value in values if not value["selected"] for reason in value["issues"])),
        "scores": {
            key: distribution([value[key] for value in values if value.get(key) is not None])
            for key in ("answer_score", "process_score", "efficiency_score", "total_score")
        },
    }


def summarize_agent(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    return {
        "overall": _summarize_rows(rows),
        "by_category": {name: _summarize_rows(items) for name, items in sorted(grouped.items())},
    }


def regrade_saved_run(output_dir: Path, distillation: DistillationConfig | None = None) -> dict[str, Any]:
    """使用当前确定性规则重判已保存轨迹，不重复调用 Agent 或评审模型。"""

    rows_path = output_dir / "agent_cases.jsonl"
    if not rows_path.is_file():
        raise FileNotFoundError(f"找不到 Agent 逐题结果：{rows_path}")
    cases = {case.id: case for case in build_rag_agent_cases() + load_agent_cases()}
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    regraded = []
    for row in rows:
        case = cases.get(str(row.get("id") or ""))
        if case is None:
            raise ValueError(f"当前评测集中不存在用例：{row.get('id')}")
        hard = _hard_grade(case, row["turns"])
        if row.get("execution_error"):
            hard["passed"] = False
            hard["failures"].append(row["execution_error"])
        judge_result = row.get("judge")
        expected_interrupt = case.turns[-1].expected_status == "interrupted"
        if case.judge and (any(turn.get("error") for turn in row["turns"]) or (not hard["answers"].strip() and not (row.get("distillation") and expected_interrupt))):
            judge_result = JudgeResult(
                score=1,
                passed=False,
                reason="执行报错或最终答案为空，确定性规则直接判定不通过。",
                usage={},
                latency_ms=0,
            ).model_dump(mode="json")
        success = hard["passed"] and (
            judge_result is None
            or (bool(judge_result.get("passed")) and int(judge_result.get("score") or 0) >= 4)
        )
        updated = {**row, "success": success, "hard_grade": hard, "judge": judge_result}
        if distillation is not None or row.get("distillation"):
            saved = row.get("distillation") or {}
            scoring_config = distillation or DistillationConfig(
                threshold=saved["threshold"], weights=tuple(saved["weights"]), export_sft=False,
            )
            raw_joint = saved.get("joint_judge")
            try:
                joint = JointJudgeResult.model_validate(raw_joint) if raw_joint else None
            except ValueError:
                joint = None
            updated["distillation"] = score_distillation(
                case, _unique_tool_calls(row["turns"]), hard["passed"], joint, scoring_config,
                trace_error=saved.get("trace_error") if saved else "旧记录缺少完整轨迹及过程评分",
                judge_error=saved.get("judge_error"),
            )
            updated["distillation"].update(
                exported=bool(saved.get("exported")), sft_line=saved.get("sft_line"),
                models=saved.get("models", []), judge_model=saved.get("judge_model"),
                model_config=saved.get("model_config"),
                regrade_note="只重算评分；未归档的样本不能补导出，已生成SFT文件不变。",
            )
            updated["success"] = hard["passed"] and joint is not None and joint.passed and joint.score >= 4
        regraded.append(updated)

    summary = summarize_agent(regraded)
    rows_output = output_dir / "agent_cases_regraded.jsonl"
    with rows_output.open("w", encoding="utf-8") as handle:
        for row in regraded:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    (output_dir / "agent_summary_regraded.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    run_id = f"{output_dir.name}_regraded"
    agent_markdown = _markdown_report(run_id, summary, regraded)
    rag_path = output_dir / "rag_retrieval.json"
    if rag_path.is_file():
        rag_result = json.loads(rag_path.read_text(encoding="utf-8"))
        report = _combined_markdown({"run_id": run_id, "rag": rag_result}, agent_markdown)
    else:
        report = agent_markdown
    (output_dir / "report_regraded.md").write_text(report, encoding="utf-8")
    return {"output_dir": str(output_dir), "summary": summary, "case_count": len(regraded)}


def _markdown_report(run_id: str, summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    overall = summary.get("overall") or {}
    lines = [
        f"# 智能检测 Agent 评测报告 {run_id}",
        "",
        "## 总览",
        "",
        f"- 端到端用例数：{overall.get('case_count', 0)}",
        f"- 任务成功率：{float(overall.get('task_success_rate') or 0):.2%}",
        f"- 工具调用准确率：{float(overall.get('tool_call_accuracy') or 0):.2%}",
        f"- 工具预算达标率：{float(overall.get('tool_budget_compliance') or 0):.2%}",
        f"- 参数准确率：{float(overall.get('parameter_accuracy') or 0):.2%}",
        f"- LLM 调用总轮数：{(overall.get('llm_calls') or {}).get('total', 0)}",
        f"- 工具调用总数：{(overall.get('tool_calls') or {}).get('total', 0)}",
        "",
        "## 分项结果",
        "",
        "| 用例 | 类别 | 成功 | LLM轮数 | 工具数 | 端到端ms | 首字ms | 失败原因 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    execution = summary.get("execution")
    if execution:
        lines[4:4] = [
            f"- 配置并发数：{execution['concurrency']}；实际 worker：{execution['worker_count']}",
            f"- 整批耗时：{execution['batch_elapsed_ms'] / 1000:.2f} 秒（含初始化和收尾）",
        ]
    for row in sorted(rows, key=lambda item: (item["id"], item["repeat"])):
        llm_count = sum(turn["telemetry"]["llm_call_count"] for turn in row["turns"])
        tool_count = len(_unique_tool_calls(row["turns"]))
        latency = sum(float(turn["telemetry"]["end_to_end_ms"] or 0) for turn in row["turns"])
        first = next(
            (
                turn["telemetry"]["first_token_ms"]
                for turn in row["turns"]
                if turn["telemetry"]["first_token_ms"] is not None
            ),
            None,
        )
        failures = list(row["hard_grade"]["failures"])
        budget_failures = row["hard_grade"]["tool"].get("budget_failures") or []
        if budget_failures:
            failures.append(f"效率提醒：{'；'.join(budget_failures)}")
        if row.get("judge") and not row["judge"]["passed"]:
            failures.append(f"LLM评审：{row['judge']['reason']}")
        reason = "；".join(failures).replace("|", "\\|")
        lines.append(
            f"| {row['id']} | {row['category']} | {'是' if row['success'] else '否'} | "
            f"{llm_count} | {tool_count} | {latency:.2f} | {first if first is not None else '-'} | {reason} |"
        )
    if "distillation" in overall:
        lines.extend(["", "## 蒸馏筛选", "", "任务成功、评分入选与实际导出分别统计；离线重算不会修改SFT文件。", ""])
        for category, section in [("全部", overall), *summary["by_category"].items()]:
            stats = section.get("distillation")
            if stats is None:
                continue
            lines.append(f"- {category}：入选 {stats['selected_count']}/{stats['case_count']}（{stats['selected_rate']:.2%}），实际导出 {stats['exported_count']}。")
            lines.append(f"  分数分布：{json.dumps(stats['scores'], ensure_ascii=False)}")
            lines.append(f"  淘汰原因：{json.dumps(stats['rejection_reasons'], ensure_ascii=False)}")
        lines.extend(["", "候选轨迹未归档；降低阈值后，新入选但未保存的样本需要重新执行任务才能导出。"])
    return "\n".join(lines) + "\n"


def _combined_markdown(result: dict[str, Any], agent_markdown: str | None) -> str:
    lines = [f"# 智能检测综合测评报告 {result['run_id']}", "",
             "工具白名单仅用于 Precision；黑名单与必要工具为硬规则。SQL 等价路径由 Judge 核对，关闭 Judge 时不包含这部分语义验证。", ""]
    rag = result.get("rag")
    if rag:
        rerank = rag["summary"]["qwen3_rerank"]
        lines.extend(
            [
                "## RAG 检索评测",
                "",
                f"- 用例数：{rag['case_count']}",
                f"- Rerank Recall@5：{float(rerank['recall@5']):.2%}",
                f"- Rerank Hit@5：{float(rerank['hit@5']):.2%}",
                f"- Rerank MRR@20：{float(rerank['mrr@20']):.4f}",
                f"- 平均检索耗时：{float(rag['summary']['mean_retrieval_latency_ms']):.2f} ms",
                "",
            ]
        )
    if agent_markdown:
        # 去掉子报告一级标题，合并后仍保持一个主标题。
        agent_lines = agent_markdown.splitlines()
        if agent_lines and agent_lines[0].startswith("# "):
            agent_lines = agent_lines[1:]
        lines.extend(agent_lines)
    return "\n".join(lines).rstrip() + "\n"


async def run_agent_suite(
    root: Path,
    output_dir: Path,
    cases: list[AgentEvalCase],
    *,
    repeat: int,
    judge_enabled: bool,
    run_id: str,
    distillation: DistillationConfig | None = None,
    concurrency: int = 1,
) -> dict[str, Any]:
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1 or repeat < 1:
        raise ValueError("concurrency 和 repeat 必须为正整数")
    if distillation is not None and not judge_enabled:
        raise ValueError("蒸馏模式不能关闭 Judge")
    service = ConversationAgentService(root)
    rows = []
    started = time.monotonic()
    total = len(cases) * repeat
    worker_count = min(concurrency, total)
    try:
        judge = EvaluationJudge(root) if judge_enabled else None
        # 先完成模型/工具及共享数据库初始化，再放行 worker。
        if total:
            await _run_sync(service._get_agent)
        output_dir.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            results_file = stack.enter_context((output_dir / "agent_cases.jsonl").open("w", encoding="utf-8"))
            sft_file = stack.enter_context((output_dir / "sft.jsonl").open("w", encoding="utf-8")) if distillation and distillation.export_sft else None
            sft_line = 0

            def persist(row: dict[str, Any], sample: dict[str, Any] | None) -> None:
                nonlocal sft_line
                selection = row.get("distillation")
                if selection and selection["selected"] and sft_file is not None:
                    if sample is None:
                        raise ValueError("入选轨迹缺少SFT样本")
                    sft_file.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
                    sft_file.flush()
                    sft_line += 1
                    selection.update(exported=True, sft_line=sft_line)
                results_file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                results_file.flush()

            queue: asyncio.Queue[tuple[AgentEvalCase, int]] = asyncio.Queue()
            stopped = False
            for repeat_index in range(1, repeat + 1):
                for case in cases:
                    queue.put_nowait((case, repeat_index))

            async def worker() -> None:
                nonlocal stopped
                while not stopped and not queue.empty():
                    case, repeat_index = queue.get_nowait()
                    print(f"[启动，已完成 {len(rows)}/{total}] {case.id} 第{repeat_index}次")
                    try:
                        row = await run_agent_case(
                            service, judge, root, case, repeat_index=repeat_index,
                            run_id=run_id, distillation=distillation, on_result=persist,
                        )
                        rows.append(row)
                        print(f"[完成 {len(rows)}/{total}] {case.id} 第{repeat_index}次：{'成功' if row['success'] else '失败'}")
                    except BaseException:
                        stopped = True
                        raise
                    finally:
                        queue.task_done()

            workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
            try:
                await asyncio.gather(*workers)
            finally:
                # 全局异常或取消：先收尾所有运行，再关闭文件和 checkpoint 连接。
                for task in workers:
                    if not task.done() and not task.cancelling():
                        task.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
    finally:
        service.close()
    summary = summarize_agent(rows)
    summary["execution"] = {
        "concurrency": concurrency, "worker_count": worker_count,
        "total_tasks": total, "batch_elapsed_ms": round((time.monotonic() - started) * 1000, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "agent_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(_markdown_report(run_id, summary, rows), encoding="utf-8")
    return {"summary": summary, "rows": rows}


async def run_evaluation(
    *,
    root: Path = PROJECT_ROOT,
    suite: str = "all",
    case_ids: set[str] | None = None,
    repeat: int = 1,
    judge_enabled: bool = True,
    output_root: Path | None = None,
    distillation: DistillationConfig | None = None,
    concurrency: int = 1,
) -> dict[str, Any]:
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("concurrency 必须为正整数")
    if distillation is not None and (not judge_enabled or suite == "rag"):
        raise ValueError("蒸馏模式需要 Agent 用例并开启 Judge")
    # 防止短时间内重复运行覆盖上一批已导出的训练数据。
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (output_root or root / "output" / "evaluation") / run_id
    all_agent_cases = build_rag_agent_cases() + load_agent_cases()
    rag_values = load_rag_cases(DEFAULT_CASES_PATH)
    raw_rag_ids = {str(item["id"]) for item in rag_values}
    known_ids = raw_rag_ids | {case.id for case in all_agent_cases}
    missing_ids = set(case_ids or []) - known_ids
    if missing_ids:
        raise ValueError(f"不存在的评测用例：{', '.join(sorted(missing_ids))}")
    if case_ids:
        agent_cases = [
            case
            for case in all_agent_cases
            if case.id in case_ids
            or (case.category == "rag" and case.id.removeprefix("agent_") in case_ids)
        ]
    else:
        agent_cases = all_agent_cases
    check_rag = suite in {"rag", "all"} and (not case_ids or bool(raw_rag_ids & {item.removeprefix('agent_') for item in case_ids}))
    check_agent = suite in {"agent", "all"} and bool(agent_cases)
    preflight_report = preflight(
        root,
        agent_cases if check_agent else [],
        check_rag=check_rag,
        check_agent=check_agent,
    )
    if not preflight_report["passed"]:
        failures = [item for item in preflight_report["checks"] if not item["passed"]]
        raise RuntimeError(f"评测预检失败：{json.dumps(failures, ensure_ascii=False)}")
    result: dict[str, Any] = {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "preflight": preflight_report,
    }
    if suite in {"rag", "all"}:
        rag_case_ids: set[str] | None = None
        if case_ids:
            rag_case_ids = {
                case_id.removeprefix("agent_")
                for case_id in case_ids
                if case_id.removeprefix("agent_") in raw_rag_ids
            }
        if not case_ids or rag_case_ids:
            result["rag"] = evaluate_rag(
                DEFAULT_CASES_PATH,
                output_dir / "rag_retrieval.json",
                case_ids=rag_case_ids,
            )
    if suite in {"agent", "all"}:
        if case_ids and not agent_cases:
            raise ValueError("指定用例没有 Agent 端到端版本")
        result["agent"] = await run_agent_suite(
            root,
            output_dir,
            agent_cases,
            repeat=repeat,
            judge_enabled=judge_enabled,
            run_id=run_id,
            distillation=distillation,
            concurrency=concurrency,
        )
        before_count = preflight_report["dataset_fingerprint"]["conversation_work_orders"]["rows"][0][0]
        after_count = dataset_fingerprint(root)["conversation_work_orders"]["rows"][0][0]
        if after_count != before_count:
            raise RuntimeError(f"评测期间工单数量发生变化：{before_count} -> {after_count}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "suite": suite,
        "repeat": repeat,
        "judge_enabled": judge_enabled,
        "concurrency": concurrency,
        "distillation": distillation.model_dump(mode="json") if distillation else None,
        "rag_case_count": (result.get("rag") or {}).get("case_count", 0),
        "agent_case_count": len((result.get("agent") or {}).get("rows", [])),
        "rag_summary": (result.get("rag") or {}).get("summary"),
        "agent_summary": (result.get("agent") or {}).get("summary"),
        "environment": preflight_report,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    agent_report_path = output_dir / "report.md"
    agent_markdown = agent_report_path.read_text(encoding="utf-8") if agent_report_path.is_file() else None
    agent_report_path.write_text(_combined_markdown(result, agent_markdown), encoding="utf-8")
    return result
