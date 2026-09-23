"""离线生成入口的模拟测试；不访问真实数据库或付费模型。"""
import asyncio
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import subprocess

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import pytest


spec = importlib.util.spec_from_file_location(
    "generate_agent_sft", Path(__file__).resolve().parents[1] / "scripts/generate_agent_sft.py"
)
generator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = generator
spec.loader.exec_module(generator)


def test_teacher_thinking_roundtrip_and_sft_does_not_mutate_history():
    from intelligent_detection_agent.evaluation.distillation import sft_message

    settings = generator.TeacherSettings("deepseek-flash", "deepseek", "https://api.deepseek.com", "test-key", 128000)
    model = settings.build(0.1, streaming=True)
    assert model.extra_body["thinking"]["type"] == "enabled"
    request_message = AIMessage(content="", additional_kwargs={"reasoning_content": "tool reasoning"},
                               tool_calls=[{"id": "call_1", "name": "query_business_data", "args": {"sql": "SELECT 1"}}])
    final_message = AIMessage(content="查询完成", additional_kwargs={"reasoning_content": "answer reasoning"})
    history = [HumanMessage(content="查询"), request_message,
               ToolMessage(content="1", tool_call_id="call_1"), final_message, HumanMessage(content="继续")]
    payload = model._get_request_payload(history, tools=[{"type": "function", "function": {
        "name": "query_business_data", "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}}
    }}])
    assert [message["reasoning_content"] for message in payload["messages"] if message["role"] == "assistant"] == [
        "tool reasoning", "answer reasoning"
    ]
    exported = [sft_message(message) for message in history]
    assert "reasoning_content" not in json.dumps(exported)
    assert request_message.additional_kwargs["reasoning_content"] == "tool reasoning"
    assert final_message.additional_kwargs["reasoning_content"] == "answer reasoning"


def test_quota_totals_and_small_batch():
    assert sum(generator.NORMAL.values()) == 750
    assert sum(generator.BOUNDARY.values()) == 250
    assert set(generator.BOUNDARY) == {"out_of_scope", "no_tool", "clarification", "empty"}
    assert sum(generator.allocate(7, generator.NORMAL).values()) == 7
    assert set(generator.allocate(0, generator.BOUNDARY).values()) == {0}


def test_task_uses_existing_user_date_pair():
    catalog = {"profiles": [{"user_id": "profile-only"}],
               "diagnoses": [{"user_id": "diagnosis-user", "diagnosis_date": "2026-01-02"}]}
    task = generator.make_task("report", catalog, random.Random(1))
    assert task["slots"] == {"user_id": "diagnosis-user", "date": "2026-01-02"}
    assert task["group"] == "entity:diagnosis-user"
    assert "2026-01-02" in task["case"]["oracles"][0]["sql"]
    assert "{{" not in task["case"]["turns"][0]["message"]


def test_offline_rewrites_require_matching_placeholder_and_template(tmp_path):
    task = {"id": "a", "template": "查询 {{user_id}}", "slots": {"user_id": "123"}}
    path = tmp_path / "rewrites.jsonl"
    row = {"id": "a", "template_sha256": generator.digest(task["template"]), "text": "看看 {{user_id}}"}
    path.write_text(json.dumps(row), encoding="utf-8")
    assert generator.load_rewrites(path, [task]) == {"a": "看看 123"}
    row["text"] = "看看 456"
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="占位符"):
        generator.load_rewrites(path, [task])
    row["text"], row["template_sha256"] = "看看 {{user_id}}", "stale"
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="模板"):
        generator.load_rewrites(path, [task])


def test_guard_keeps_real_results_and_limits_repeats():
    async def exercise():
        guard = generator.GenerationGuard(10, 10, 3)
        executed = []

        async def handler(request):
            executed.append(request.tool_call["id"])
            return ToolMessage(content="real result", tool_call_id=request.tool_call["id"])

        for index in range(3):
            request = SimpleNamespace(tool_call={"id": str(index), "name": "query_business_data", "args": {"sql": "SELECT 1"}})
            result = await guard.awrap_tool_call(request, handler)
            assert result.content == "real result"
            assert result.status == "success"
        with pytest.raises(RuntimeError, match="generation_tool_call_limit"):
            await guard.awrap_tool_call(SimpleNamespace(tool_call={
                "id": "3", "name": "query_business_data", "args": {"sql": "SELECT 1"}
            }), handler)
        assert executed == ["0", "1", "2"]
        guard.reset()
        assert guard.model_call_count == guard.tool_call_count == 0
        assert getattr(guard, "tools", []) == []
        assert not guard.error_ids

    asyncio.run(exercise())


def test_real_tool_failure_is_preserved_and_recorded():
    async def exercise():
        guard = generator.GenerationGuard(10, 10, 3)

        async def handler(request):
            return ToolMessage(content="real SQL failure", status="error", tool_call_id="1")

        result = await guard.awrap_tool_call(SimpleNamespace(tool_call={
            "id": "1", "name": "query_business_data", "args": {"sql": "bad SQL"}
        }), handler)
        assert result.content == "real SQL failure"
        assert guard.error_ids == {"1"}
        guard.reset()
        assert not guard.error_ids

    asyncio.run(exercise())


def test_teacher_real_deep_agent_initialization_and_guard_execution(tmp_path, monkeypatch):
    """保留 TeacherService、DeepAgents 与业务工具装配，只把模型替换为本地假模型。"""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from intelligent_detection_agent.conversation_agent import agent as agent_module

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setattr(agent_module, "load_project_env", lambda *a: None)
    (tmp_path / "database").mkdir()
    for name in ["gas_ai_input.duckdb", "gas_ai_results.duckdb"]:
        with generator.duckdb.connect(str(tmp_path / "database" / name)) as connection:
            connection.execute("CREATE TABLE smoke(n INTEGER)")
    model = Model(responses=[AIMessage(content="测试完成")])
    monkeypatch.setattr(generator.TeacherService, "_build_model", lambda self: model)
    guard = generator.GenerationGuard(1, 5, 2)
    settings = generator.TeacherSettings("test", "openai-compatible", "http://unused", "unused", 128000)
    service = generator.TeacherService(tmp_path, settings, guard, .1)
    try:
        graph = service._get_agent()
        assert graph is service._get_agent()
        assert getattr(guard, "tools", []) == []
        async def exercise():
            result = await graph.ainvoke({"messages": [HumanMessage(content="测试")]},
                config={"configurable": {"thread_id": "guard-smoke"}})
            assert result["messages"][-1].content == "测试完成"
            assert guard.model_call_count == 1 and guard.tool_call_count == 0
            with pytest.raises(RuntimeError, match="generation_model_call_limit"):
                await graph.ainvoke({"messages": [HumanMessage(content="超预算")]},
                    config={"configurable": {"thread_id": "guard-limit"}})
        asyncio.run(exercise())
    finally:
        service.close()


def test_cli_default_target_is_1000(monkeypatch):
    captured = []
    monkeypatch.setattr(sys, "argv", ["generate", "plan", "--plan", "unused.json"])
    monkeypatch.setattr(generator, "build_plan", lambda args: captured.append(args))
    assert generator.main() == 0
    assert (captured[0].normal, captured[0].boundary) == (750, 250)


def test_empty_tasks_cover_missing_user_and_existing_user_missing_date():
    catalog = {"profiles": [{"user_id": "123"}],
               "diagnoses": [{"user_id": "123", "diagnosis_date": "2025-01-12"}]}
    tasks = [generator.make_task("empty", catalog, random.Random(seed)) for seed in range(30)]
    assert {t["case"]["oracles"][0]["source"] for t in tasks} == {"business", "diagnosis"}
    for task in tasks:
        if "date" in task["slots"]:
            assert task["slots"]["user_id"] == "123"
            assert task["slots"]["date"] > "2025-01-12"
            assert "用户不存在" in task["case"]["judge_criteria"][-1]
        else:
            assert task["slots"]["user_id"] != "123"


def recovery_fixture():
    def action(call_id):
        return {"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function",
                "function": {"name": "lookup", "arguments": {"field": "bad" if call_id == "bad" else "good"}}}]}
    sample = {"messages": [{"role": "user", "content": "查询"}, action("bad"),
                           {"role": "tool", "tool_call_id": "bad", "content": "字段不存在"}, action("good"),
                           {"role": "tool", "tool_call_id": "good", "content": "42"},
                           {"role": "assistant", "content": "结果为42"}],
              "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]}
    calls = [{"tool_call_id": name, "name": "lookup", "status": status, "arguments": {}}
             for name, status in [("bad", "error"), ("good", "success")]]
    return sample, calls


def test_recovery_masks_error_history_without_changing_messages():
    sample, calls = recovery_fixture()
    original = json.dumps(sample)
    result, start = generator.recovery_sample(sample, calls, {"bad"})
    assert start == 3
    assert result["prompt"] + result["completion"] == sample["messages"]
    assert result["training"]["assistant_loss_mask"] == [False, False, False, True, False, True]
    assert "messages" not in result  # 旧训练器要求 messages，因此不能静默误学错误动作。
    assert json.dumps(sample) == original


def test_parallel_sibling_result_stays_in_prompt():
    sample, calls = recovery_fixture()
    sample["messages"][1]["tool_calls"].append({"id": "parallel", "type": "function",
        "function": {"name": "lookup", "arguments": {}}})
    sample["messages"].insert(3, {"role": "tool", "tool_call_id": "parallel", "content": "旁路结果"})
    calls.append({"tool_call_id": "parallel", "name": "lookup", "status": "success"})
    result, start = generator.recovery_sample(sample, calls, {"bad"})
    assert start == 4 and result["prompt"][-1]["tool_call_id"] == "parallel"
    # 同批成功结果不能冒充看见错误后的修正。
    with pytest.raises(ValueError, match="成功执行"):
        generator.recovery_sample(sample, [c for c in calls if c["tool_call_id"] != "good"], {"bad"})


@pytest.mark.parametrize("problem", ["missing", "unresolved"])
def test_recovery_rejects_unverifiable_or_unrecovered_error(problem):
    sample, calls = recovery_fixture()
    if problem == "missing":
        sample["messages"].pop(2)
    elif problem == "unresolved":
        calls[1]["status"] = "error"
    with pytest.raises(ValueError):
        generator.recovery_sample(sample, calls, {"bad"})


def test_recovery_allows_an_alternative_tool_without_supervising_error():
    sample, calls = recovery_fixture()
    calls[0]["name"] = "ls"
    sample["messages"][1]["tool_calls"][0]["function"]["name"] = "ls"
    result, start = generator.recovery_sample(sample, calls, {"bad"})
    assert start == 3
    assert result["prompt"][1]["tool_calls"][0]["function"]["name"] == "ls"
    assert result["training"]["assistant_loss_mask"] == [False, False, False, True, False, True]


@pytest.mark.parametrize("problem", [None, "no_empty", "wrong_source", "wrong_table", "missing_rows", "invalid_sql"])
def test_empty_allows_nonempty_auxiliary_queries_but_requires_target_evidence(problem):
    task = {"scenario": "empty", "case": {"oracles": [{"source": "diagnosis"}],
        "argument_assertions": [{"tables": ["metering.diagnosis_run"]}]}}
    payload = {"source": "diagnosis", "rows": [], "sql":
        "SELECT risk_level FROM metering.diagnosis_run WHERE user_id='123' AND diagnosis_date=DATE '2025-02-01'"}
    if problem == "no_empty":
        payload["rows"] = [{"risk_level": "低"}]
    elif problem == "wrong_source":
        payload["source"] = "business"
    elif problem == "wrong_table":
        payload["sql"] = "SELECT * FROM equipment.health_diagnosis"
    elif problem == "missing_rows":
        del payload["rows"]
    elif problem == "invalid_sql":
        payload["sql"] = "SELECT ("
    turns = [{"telemetry": {"tool_calls": []}, "artifacts": [
        {"type": "query_result", "payload": payload},
        {"type": "query_result", "payload": {"source": "diagnosis", "rows": [{"n": 20}],
            "sql": "SELECT COUNT(*) AS n FROM metering.diagnosis_run"}}]}]
    assert bool(generator.boundary_failures(task, turns)) == (problem is not None)


def joint(**changes):
    from intelligent_detection_agent.evaluation.models import JointJudgeResult
    return JointJudgeResult.model_validate(dict(score=5, passed=True, reason="正确", parameter_score=100,
        dependency_score=100, recovery_score=100, issues=[], redundant_call_ids=[], justified_repeats=[],
        unrecovered_failure=False) | changes)


@pytest.mark.parametrize("failure", [None, "hard", "answer", "unrecovered", "scoped", "judge_exception", "alternative"])
def test_rollout_keeps_raw_score_and_separately_selects_recovery(monkeypatch, tmp_path, failure):
    sample, calls = recovery_fixture()
    if failure in {"alternative", "unrecovered"}:
        calls[0]["name"] = "ls"
        sample["messages"][1]["tool_calls"][0]["function"]["name"] = "ls"
    case = generator.AgentEvalCase(id="test", category="query", description="查询", turns=[{"message": "查询"}])
    turn = {"error": None, "answer": "42", "actual_status": "completed", "expected_status": "completed",
            "interrupt": None, "artifacts": [], "telemetry": {"tool_calls": calls}}
    monkeypatch.setattr(generator, "TrajectoryCollector", lambda: SimpleNamespace(sample=lambda **kw: sample))
    monkeypatch.setattr(generator, "execute_oracles", lambda *a: [])
    async def run_turn(*args):
        return turn
    monkeypatch.setattr(generator, "_run_turn", run_turn)
    monkeypatch.setattr(generator, "_hard_grade", lambda *a: {"passed": failure != "hard",
        "failures": ["禁止工具"] if failure == "hard" else []})
    seen = []
    class Judge:
        async def grade_trajectory(self, *args, **kwargs):
            seen.append(kwargs)
            assert "不要求失败工具再次同名调用成功" in "".join(args[0].judge_criteria)
            if kwargs:
                if failure == "judge_exception":
                    raise ValueError("invalid judge")
                return joint(parameter_score=75 if failure == "scoped" else 100)
            return joint(parameter_score=0, score=3 if failure == "answer" else 5,
                         unrecovered_failure=failure == "unrecovered")
    row, exported = asyncio.run(generator.rollout({"id": "test", "scenario": "profile"}, case,
        SimpleNamespace(root=tmp_path), Judge(), generator.GenerationGuard(10, 10, 3), generator.DistillationConfig(), {}))
    assert not row["distillation"]["selected"]
    assert row["recovery_selection"]["selected"] == (failure in {None, "alternative"})
    if failure is None:
        assert seen == [{}, {"process_start_index": 3}]
        assert row["distillation"]["process_score"] == 40
        assert row["recovery_selection"]["process_score"] == 100
        assert exported["training"]["loss_policy"] == "completion_assistant_only"
    if failure in {"hard", "answer", "unrecovered"}:
        assert len(seen) == (0 if failure == "hard" else 1)


def test_stale_empty_candidate_stops_before_teacher(monkeypatch, tmp_path):
    monkeypatch.setattr(generator, "execute_oracles", lambda *a: [{"rows": [["now exists"]]}])
    async def forbidden(*args):
        pytest.fail("stale candidate must not call teacher")
    monkeypatch.setattr(generator, "_run_turn", forbidden)
    case = generator.AgentEvalCase(id="empty", category="query", description="查", turns=[{"message": "查"}])
    with pytest.raises(ValueError, match="过期"):
        asyncio.run(generator.rollout({"id": "empty", "scenario": "empty"}, case,
            SimpleNamespace(root=tmp_path), None, generator.GenerationGuard(10, 10, 3), generator.DistillationConfig(), {}))


def test_rollout_offline_returns_full_packet_without_judge(monkeypatch, tmp_path):
    sample, calls = recovery_fixture()
    task = {"id": "test", "scenario": "profile"}
    case = generator.AgentEvalCase(id="test", category="query", description="查", turns=[{"message": "查"}])
    turn = {"answer": "42", "error": None, "actual_status": "completed", "expected_status": "completed",
            "interrupt": None, "artifacts": [], "telemetry": {"tool_calls": calls}}
    monkeypatch.setattr(generator, "TrajectoryCollector", lambda: SimpleNamespace(sample=lambda **kw: sample))
    monkeypatch.setattr(generator, "execute_oracles", lambda *a: [{"rows": [[42]]}])
    monkeypatch.setattr(generator, "_hard_grade", lambda *a: {"passed": True, "failures": []})
    async def fake_turn(*args):
        return turn
    monkeypatch.setattr(generator, "_run_turn", fake_turn)
    row, output = asyncio.run(generator.rollout(task, case, SimpleNamespace(root=tmp_path), None,
        generator.GenerationGuard(10, 10, 3), generator.DistillationConfig(), {}))
    assert output is None and "distillation" not in row
    assert row["review_status"] == "pending"
    packet = row["_review_packet"]
    assert packet["trajectory"] == sample and packet["oracle_results"] == [{"rows": [[42]]}]
    assert packet["process_start_index"] == 3
    assert packet["execution"]["tool_calls"] == calls


def offline_batch_fixture(tmp_path):
    batch = tmp_path / "batch"
    (batch / "review_packets").mkdir(parents=True)
    (batch / "config.json").write_text(json.dumps({"judge_mode": "offline", "threshold": 85,
        "weights": [.4, .4, .2], "quotas": {"profile": 3}}), encoding="utf-8")
    rows, reviews = [], []
    for index in range(3):
        sample, calls = recovery_fixture()
        if index != 1:
            sample["messages"] = [sample["messages"][0], *sample["messages"][3:]]
            calls = calls[1:]
        sample["messages"][0]["content"] = str(index)
        case = generator.AgentEvalCase(id=str(index), category="query", description=str(index),
            turns=[{"message": str(index)}], required_tools=[{"name": "lookup", "min_calls": 1}])
        task = {"id": str(index), "scenario": "profile", "group": str(index), "split": "train",
                "case": case.model_dump(mode="json")}
        turn = {"answer": "42", "error": None, "actual_status": "completed", "expected_status": "completed",
                "interrupt": None, "artifacts": [], "telemetry": {"tool_calls": calls}}
        row = {**{key:task[key] for key in ["id", "scenario", "group", "split"]}, "turns": [turn],
               "hard_grade": generator._hard_grade(case, [turn]), "review_status": "pending", "exported": False,
               "error_call_ids": ["bad"] if index == 1 else [], "sample_format": "recovery" if index == 1 else "standard"}
        packet = {"version": 1, "task": task, "case": case.model_dump(mode="json"), "trajectory": sample,
            "record": dict(row), "process_start_index": 3 if index == 1 else None}
        filename = f"review_packets/{index}.json"
        (batch / filename).write_text(json.dumps(packet, ensure_ascii=False), encoding="utf-8")
        row.update(review_packet=filename, packet_sha256=generator.digest(packet))
        rows.append(row)
        full = joint(parameter_score=0 if index == 1 else 100).model_dump(exclude={"usage", "latency_ms"})
        reviews.append({"id": str(index), "packet_sha256": row["packet_sha256"], "reviewer": "test-reviewer",
            "full": full, "recovery": joint().model_dump(exclude={"usage", "latency_ms"}) if index == 1 else None})
    (batch / "results.jsonl").write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding="utf-8")
    return batch, reviews


@pytest.mark.parametrize("problem", [None, "duplicate", "hash", "unknown_call", "out_of_scope", "tampered_packet",
                                     "unrecovered", "missing_recovery", "bad_recovery", "invalid_score"])
def test_offline_export_validates_reviews_and_never_calls_models(tmp_path, monkeypatch, problem):
    batch, reviews = offline_batch_fixture(tmp_path)
    monkeypatch.setattr(generator, "EvaluationJudge", lambda *a: pytest.fail("must not call Judge"))
    monkeypatch.setattr(generator, "TeacherService", lambda *a: pytest.fail("must not create teacher"))
    monkeypatch.setattr(generator, "load_project_env", lambda *a: pytest.fail("export must not need credentials"))
    reviews = reviews[:2]  # 第三条故意不评，不能默认通过。
    if problem == "duplicate":
        reviews.append(reviews[0])
    elif problem == "hash":
        reviews[0]["packet_sha256"] = "wrong"
    elif problem == "unknown_call":
        reviews[0]["full"]["redundant_call_ids"] = ["unknown"]
    elif problem == "out_of_scope":
        reviews[1]["recovery"]["issues"] = [{"tool_call_id": "bad", "reason": "prefix"}]
    elif problem == "tampered_packet":
        path = batch / "review_packets/0.json"
        packet = json.loads(path.read_text(encoding="utf-8"))
        packet["trajectory"]["messages"][-1]["content"] = "changed"
        path.write_text(json.dumps(packet), encoding="utf-8")
    elif problem == "unrecovered":
        reviews[1]["full"]["unrecovered_failure"] = True
    elif problem == "missing_recovery":
        reviews[1]["recovery"] = None
    elif problem == "bad_recovery":
        reviews[1]["recovery"]["parameter_score"] = 75
    elif problem == "invalid_score":
        reviews[0]["full"]["parameter_score"] = 88
    path = tmp_path / "reviews.jsonl"
    path.write_text(''.join(json.dumps(r)+'\n' for r in reviews), encoding="utf-8")
    args = SimpleNamespace(batch=batch, reviews=path, output=tmp_path / "export", threshold=None, weights=None)
    if problem in {"duplicate", "hash", "unknown_call", "out_of_scope", "tampered_packet", "invalid_score"}:
        with pytest.raises(ValueError):
            generator.export_offline_reviews(args)
        return
    assert generator.export_offline_reviews(args) == 2
    results = [json.loads(line) for line in (args.output / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert results[0]["exported"] and results[0]["sft_line"] == 1
    assert results[2]["review_status"] == "pending" and not results[2]["exported"]
    assert results[1]["exported"] == (problem is None)
    if problem is None:
        assert results[1]["distillation"]["total_score"] == 76
        assert results[1]["recovery_selection"]["total_score"] == 100
        saved = json.loads((args.output / "sft.recovery.jsonl").read_text(encoding="utf-8"))
        assert saved["training"]["assistant_loss_mask"] == [False, False, False, True, False, True]
        assert "reviewer" not in saved and "full" not in saved
    if problem == "missing_recovery":
        assert results[1]["review_status"] == "pending_recovery"
    summary = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
    assert summary["pending_review"] == (2 if problem == "missing_recovery" else 1)
    assert summary["exported"] == (2 if problem is None else 1)


def test_multiple_errors_mask_through_last_error():
    sample, calls = recovery_fixture()
    sample["messages"][3:3] = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "bad2", "type": "function",
            "function": {"name": "lookup", "arguments": {}}}]},
        {"role": "tool", "tool_call_id": "bad2", "content": "仍然错误"}]
    calls.append({"tool_call_id": "bad2", "name": "lookup", "status": "error"})
    result, start = generator.recovery_sample(sample, calls, {"bad", "bad2"})
    assert start == 5
    assert result["training"]["assistant_loss_mask"] == [False] * 5 + [True, False, True]


def test_scoped_judge_sees_full_context_but_rejects_prefix_issue_ids(monkeypatch):
    sample, _ = recovery_fixture()
    requests = []
    response = joint()
    class Model:
        async def ainvoke(self, messages):
            requests.append(messages)
            return AIMessage(content=response.model_dump_json())
    judge = object.__new__(generator.EvaluationJudge)
    monkeypatch.setattr(judge, "_build_model", lambda: Model())
    case = generator.AgentEvalCase(id="test", category="query", description="查询", turns=[{"message": "查询"}])
    asyncio.run(judge.grade_trajectory(case, [], sample, {}, process_start_index=3))
    assert "起点为 3" in requests[0][0][1]
    assert json.loads(requests[0][1][1])["trajectory"] == sample
    response = joint(issues=[{"tool_call_id": "bad", "reason": "历史错误"}])
    with pytest.raises(ValueError, match="不存在"):
        asyncio.run(judge.grade_trajectory(case, [], sample, {}, process_start_index=3))
    assert len(requests) == 3  # 格式失败有限重试，仍不接受越过监督范围的判定。
    with pytest.raises(ValueError, match="起点"):
        asyncio.run(judge.grade_trajectory(case, [], sample, {}, process_start_index=2))
    assert len(requests) == 3


def test_v1_plan_rejected_before_model_setup(tmp_path, monkeypatch):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"version": 1}), encoding="utf-8")
    monkeypatch.setattr(generator, "load_project_env", lambda *a: pytest.fail("must reject before setup"))
    with pytest.raises(ValueError, match="v2"):
        asyncio.run(generator.run_generation(SimpleNamespace(plan=path)))


@pytest.mark.parametrize("leave_wal", [False, True])
def test_worker_snapshots_include_committed_wal_and_preserve_source(tmp_path, leave_wal):
    """子进程提交后直接退出以保留真实 WAL；仅操作临时测试库，不连接模型。"""
    root = tmp_path / "source"
    (root / "database").mkdir(parents=True)
    for name in ["gas_ai_input.duckdb", "gas_ai_results.duckdb"]:
        with generator.duckdb.connect(str(root / "database" / name)) as connection:
            connection.execute("CREATE TABLE records(id INTEGER PRIMARY KEY, value VARCHAR)")
            connection.execute("INSERT INTO records VALUES (1, '原数据')")
            connection.execute("CREATE VIEW totals AS SELECT count(*) AS n FROM records")
    results = root / "database/gas_ai_results.duckdb"
    if leave_wal:
        subprocess.run([sys.executable, "-c",
            "import duckdb,os,sys; c=duckdb.connect(sys.argv[1]); "
            "c.execute(\"INSERT INTO records VALUES (2, 'committed-in-wal')\"); os._exit(0)",
            str(results)], check=True, timeout=30)
        assert Path(str(results) + ".wal").is_file()
    before = {p.name: p.read_bytes() for p in (root / "database").iterdir()}
    targets = [tmp_path / f"worker_{i}" for i in range(3)]
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(lambda target: generator.prepare_worker_root(root, target), targets))
    assert {p.name: p.read_bytes() for p in (root / "database").iterdir()} == before
    for target in targets:
        db = target / "database/gas_ai_results.duckdb"
        with generator.duckdb.connect(str(db)) as connection:
            assert connection.execute("SELECT n FROM totals").fetchone()[0] == (2 if leave_wal else 1)
            if leave_wal:
                assert connection.execute("SELECT value FROM records WHERE id=2").fetchone()[0] == 'committed-in-wal'
            with pytest.raises(generator.duckdb.ConstraintException):
                connection.execute("INSERT INTO records VALUES (1, 'duplicate')")
            connection.execute("CHECKPOINT")
        assert not Path(str(db) + ".wal").exists()


@pytest.mark.parametrize("fail_write", [False, True])
@pytest.mark.parametrize("judge_mode", ["api", "offline"])
def test_generation_routes_formats_and_preserves_completed_output(tmp_path, monkeypatch, fail_write, judge_mode):
    """只使用假服务、假数据库和固定样本，覆盖真实持久化及 worker 清理路径。"""
    quotas = dict.fromkeys([*generator.NORMAL, *generator.BOUNDARY], 0)
    quotas["profile"] = 3
    tasks = []
    for index in range(3):
        tasks.append({"id": str(index), "scenario": "profile", "group": str(index), "split": "train",
            "case": {"id": str(index), "category": "query", "description": str(index),
                     "turns": [{"message": str(index)}]}})
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"version": 2, "quotas": quotas, "tasks": tasks}), encoding="utf-8")
    args = SimpleNamespace(plan=path, root=tmp_path, max_candidates=0, rewrites=tmp_path / "unused",
        threshold=85, weights=[.4, .4, .2], output=tmp_path / "out", rollout_temperature=.1,
        max_model_calls=16, max_tool_calls=24, max_identical_calls=3, case_timeout=10, concurrency=1, judge_mode=judge_mode)
    settings = SimpleNamespace(model="fake", provider="fake")
    monkeypatch.setattr(generator, "load_project_env", lambda *a: None)
    monkeypatch.setattr(generator.TeacherSettings, "from_env", lambda: settings)
    for key in ["MODEL", "PROVIDER", "BASE_URL", "API_KEY"]:
        monkeypatch.setenv("EVAL_JUDGE_" + key, "test")
    monkeypatch.setattr(generator, "EvaluationJudge", lambda *a: SimpleNamespace(model_name="fake"))
    if judge_mode == "offline":
        monkeypatch.setattr(generator, "EvaluationJudge", lambda *a: pytest.fail("offline must not instantiate Judge"))
        for key in ["MODEL", "PROVIDER", "BASE_URL", "API_KEY"]:
            monkeypatch.delenv("EVAL_JUDGE_" + key)
    monkeypatch.setattr(generator, "load_rewrites", lambda *a: {str(i): str(i) for i in range(3)})
    monkeypatch.setattr(generator, "prepare_worker_root", lambda *a: None)
    monkeypatch.setattr(generator.duckdb, "connect", lambda *a: nullcontext(SimpleNamespace(execute=lambda *a: None)))
    cleaned, closed = [], []
    class Service:
        def __init__(self, *args):
            pass
        def _get_agent(self):
            pass
        def delete_thread(self, user, task_id):
            cleaned.append(task_id)
        def close(self):
            closed.append(True)
    monkeypatch.setattr(generator, "TeacherService", Service)
    sample, calls = recovery_fixture()
    recovery, _ = generator.recovery_sample(sample, calls, {"bad"})
    async def fake_rollout(task, case, service, judge, guard, config, row):
        if task["id"] == "2":
            raise ValueError("later task failed")
        recovered = task["id"] == "1"
        row.update(sample_format="recovery" if recovered else "standard",
                   distillation={"selected": not recovered, "exported": False},
                   hard_grade={"failures": []})
        if recovered:
            row["recovery_selection"] = {"selected": True}
        if judge_mode == "offline":
            assert judge is None
            row.update(review_status="pending", _review_packet={"version": 1, "task": task,
                "case": case.model_dump(mode="json"), "trajectory": sample, "process_start_index": 3 if recovered else None})
            return row, None
        # 标准样本不带历史错误；两种格式绝不混写。
        return row, recovery if recovered else {"messages": [sample["messages"][0], sample["messages"][-1]], "tools": []}
    monkeypatch.setattr(generator, "rollout", fake_rollout)
    if fail_write:
        original_open = Path.open
        class FailingWriter:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
            def write(self, value):
                raise OSError("disk full")
        def patched_open(path, *a, **kw):
            if (judge_mode == "api" and path.name == "sft.recovery.jsonl") or (
                    judge_mode == "offline" and path.parent.name == "review_packets" and path.stem == generator.digest("1")):
                return FailingWriter()
            return original_open(path, *a, **kw)
        monkeypatch.setattr(Path, "open", patched_open)
        with pytest.raises(OSError, match="disk full"):
            asyncio.run(generator.run_generation(args))
    else:
        assert asyncio.run(generator.run_generation(args)) == (0 if judge_mode == "offline" else 2)
    rows = [json.loads(line) for line in (args.output / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == (1 if fail_write else 3)
    if judge_mode == "offline":
        assert not list(args.output.glob("sft*.jsonl"))
        summary = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
        assert summary["pending_review"] == (1 if fail_write else 2)
        assert summary["exported"] == 0
        assert summary["status"] == ("interrupted_or_failed" if fail_write else "awaiting_review")
        for row in rows[:1 if fail_write else 2]:
            packet = json.loads((args.output / row["review_packet"]).read_text(encoding="utf-8"))
            assert generator.digest(packet) == row["packet_sha256"]
            assert not row["exported"] and row["review_status"] == "pending"
        assert closed == [True]
        return
    assert rows[0]["sft_file"] == "sft.jsonl" and rows[0]["sft_line"] == 1
    if not fail_write:
        assert rows[1]["sft_file"] == "sft.recovery.jsonl" and rows[1]["sft_line"] == 1
        assert not rows[1]["distillation"]["exported"] and rows[1]["recovery_selection"]["exported"]
        assert rows[2]["stage_failed"] and not rows[2]["exported"]
        for row in rows[:2]:
            data = json.loads((args.output / row["sft_file"]).read_text(encoding="utf-8"))
            assert ("prompt" in data) == (row["sample_format"] == "recovery")
            assert (args.output / row["split_file"]).read_text(encoding="utf-8") == (args.output / row["sft_file"]).read_text(encoding="utf-8")
    summary = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
    assert summary["exported"] == (1 if fail_write else 2)
    assert summary["recovery_exported"] == (0 if fail_write else 1)
    assert summary["shortfall"]["profile"] == (2 if fail_write else 1)
    assert cleaned == (["0", "1"] if fail_write else ["0", "1", "2"])
    assert closed == [True]
