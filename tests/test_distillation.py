from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import uuid4

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from intelligent_detection_agent.conversation_agent.agent import ConversationAgentService
from intelligent_detection_agent.evaluation import runner
from intelligent_detection_agent.evaluation.distillation import TrajectoryCollector, score_distillation, sft_message
from intelligent_detection_agent.evaluation.judge import EvaluationJudge
from intelligent_detection_agent.evaluation.models import AgentEvalCase, DistillationConfig, JointJudgeResult
from intelligent_detection_agent.evaluation.telemetry import TurnTelemetry


TOOLS = [{"type": "function", "function": {"name": "lookup", "description": "查询", "parameters": {"type": "object", "properties": {}}}}]


@pytest.fixture(autouse=True)
def disable_remote_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


def case(**changes):
    data = dict(id="test", category="query", description="查询", turns=[{"message": "查询"}],
                required_tools=[{"name": "lookup", "min_calls": 1, "max_calls": 1}])
    data.update(changes)
    return AgentEvalCase.model_validate(data)


def joint(**changes):
    data = dict(score=5, passed=True, reason="正确", parameter_score=100, dependency_score=100,
                recovery_score=100, issues=[], redundant_call_ids=[], justified_repeats=[], unrecovered_failure=False)
    data.update(changes)
    return JointJudgeResult.model_validate(data)


def call(call_id="a"):
    return {"tool_call_id": call_id, "name": "lookup", "arguments": {}}


def capture(collector, messages, output, tools=TOOLS):
    run_id = uuid4()
    collector.on_chat_model_start({}, [messages], run_id=run_id, metadata={"langgraph_node": "model"},
                                  invocation_params={"tools": tools, "model": "fake", "api_key": "never-save"})
    collector.on_llm_end(LLMResult(generations=[[ChatGeneration(message=output)]]), run_id=run_id)


def trajectory():
    collector = TrajectoryCollector()
    inputs = [SystemMessage(content="系统"), HumanMessage(content="问题")]
    action = AIMessage(content="", tool_calls=[{"id": "a", "name": "lookup", "args": {}}],
                       additional_kwargs={"reasoning_content": "secret-reasoning"})
    result = ToolMessage(content="真实结果", tool_call_id="a")
    answer = AIMessage(content="答案")
    capture(collector, inputs, action)
    capture(collector, inputs + [action, result], answer)
    return collector, inputs + [action, result, answer]


@pytest.mark.parametrize("kwargs", [
    {"threshold": -1}, {"threshold": 101}, {"threshold": float("nan")},
    {"weights": (0.4, 0.4)}, {"weights": (0.4, 0.4, 0.4)},
    {"weights": (-0.1, 0.5, 0.6)}, {"weights": (float("nan"), 0, 1)},
])
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        DistillationConfig(**kwargs)


def test_score_threshold_and_hard_gates():
    config = DistillationConfig(threshold=100)
    assert score_distillation(case(), [call()], True, joint(), config)["selected"]
    assert not score_distillation(case(), [call()], False, joint(), config)["selected"]
    assert not score_distillation(case(), [call()], True, joint(passed=False), config)["selected"]
    assert not score_distillation(case(), [call()], True, joint(unrecovered_failure=True), config)["selected"]
    assert not score_distillation(case(), [call()], True, None, config)["selected"]
    low = score_distillation(case(), [call()], True, joint(parameter_score=25), config)
    assert low["process_score"] == 55
    assert low["eligible"] and not low["selected"]


def test_efficiency_dedup_and_no_double_penalty():
    scored = score_distillation(case(), [call("a"), call("a"), call("b"), call("c")], True,
                                joint(redundant_call_ids=["b"], justified_repeats=[{"tool_call_id": "c", "reason": "刷新"}]),
                                DistillationConfig())
    assert scored["penalties"] == {"b": 15, "c": 10}
    assert scored["efficiency_score"] == 75
    assert scored["process_score"] == 100
    no_budget = case(required_tools=[{"name": "lookup"}])
    assert score_distillation(no_budget, [call("a"), call("b")], True, joint(), DistillationConfig())["efficiency_score"] == 100


def test_multiturn_sft_fields_and_no_reasoning():
    collector, messages = trajectory()
    capture(collector, messages + [HumanMessage(content="追问")], AIMessage(content="下一轮答案"))
    sample = collector.sample()
    assert list(sample) == ["messages", "tools"]
    assert [m["role"] for m in sample["messages"]] == ["system", "user", "assistant", "tool", "assistant", "user", "assistant"]
    encoded = json.dumps(sample, ensure_ascii=False)
    assert "secret-reasoning" not in encoded and "never-save" not in encoded
    assert sample["messages"][2]["tool_calls"][0]["function"]["arguments"] == {}
    assert json.loads(encoded) == sample


def test_v3_tool_call_content_projection_is_not_multimodal():
    projected = {"type": "tool_call", "id": "a", "name": "lookup", "args": {}}
    message = AIMessage(content=[{"type": "reasoning", "reasoning": "hidden"}, projected], tool_calls=[projected])
    normalized = sft_message(message)
    assert normalized["content"] == ""
    assert len(normalized["tool_calls"]) == 1
    assert normalized["tool_calls"][0]["function"]["arguments"] == {}
    message.content[1] = {**projected, "args": {"wrong": True}}
    with pytest.raises(ValueError, match="不一致"):
        sft_message(message)


def test_parallel_tools_and_expected_interrupt():
    collector = TrajectoryCollector()
    action = AIMessage(content="", tool_calls=[{"id": i, "name": "lookup", "args": {}} for i in ["a", "b"]])
    capture(collector, [HumanMessage(content="查")], action)
    collector.observe_state([ToolMessage(content="结果", tool_call_id="b")])
    assert collector.sample(expected_interrupt=True)["messages"][-1]["tool_call_id"] == "b"
    with pytest.raises(ValueError, match="缺少工具返回"):
        collector.sample()
    # 人工恢复后只能使用实际到达模型的消息。
    capture(collector, [HumanMessage(content="查"), action, ToolMessage(content="结果", tool_call_id="b"),
                        ToolMessage(content="已批准", tool_call_id="a")], AIMessage(content="完成"))
    assert collector.sample()["messages"][-1]["content"] == "完成"


@pytest.mark.parametrize("change", ["reorder", "content", "duplicate", "missing", "cross_step", "assistant"])
def test_parallel_return_reordering_requires_identical_complete_batch(change):
    collector = TrajectoryCollector()
    user = HumanMessage(content="查询")
    action = AIMessage(content="", tool_calls=[
        {"id": name, "name": "lookup", "args": {}} for name in ("a", "b")
    ])
    a, b = ToolMessage(content="结果A", tool_call_id="a"), ToolMessage(content="结果B", tool_call_id="b")
    answer = AIMessage(content="完成第一轮")
    capture(collector, [user], action)
    capture(collector, [user, action, a, b], answer)
    history = [user, action, b, a, answer]
    if change == "content":
        history[2] = ToolMessage(content="被修改", tool_call_id="b")
    elif change == "duplicate":
        history[3] = b
    elif change == "missing":
        del history[3]
    elif change == "cross_step":
        history = [user, action, b, answer, a]
    elif change == "assistant":
        history[4] = AIMessage(content="改写后的回答")
    capture(collector, history + [HumanMessage(content="追问")], AIMessage(content="最终回答"))
    if change != "reorder":
        with pytest.raises(ValueError, match="压缩或改写"):
            collector.sample()
    else:
        sample = collector.sample()
        assert [m["tool_call_id"] for m in sample["messages"] if m["role"] == "tool"] == ["a", "b"]
        assert list(collector.calls.values())[-1]["messages"][2]["tool_call_id"] == "b"
        assert sample["messages"][-1]["content"] == "最终回答"


def test_changed_history_tools_and_multimodal_fail_closed():
    collector, _ = trajectory()
    capture(collector, [HumanMessage(content="摘要替换历史")], AIMessage(content="回答"))
    with pytest.raises(ValueError, match="压缩或改写"):
        collector.sample()
    collector, messages = trajectory()
    capture(collector, messages, AIMessage(content="回答"), tools=[])
    with pytest.raises(ValueError, match="工具定义变化"):
        collector.sample()
    collector = TrajectoryCollector()
    capture(collector, [HumanMessage(content=[{"type": "image_url", "image_url": {"url": "x"}}])], AIMessage(content="图"))
    with pytest.raises(ValueError, match="非文本"):
        collector.sample()


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)


def test_real_v3_graph_callbacks_and_service_sse(tmp_path):
    @tool
    def lookup() -> str:
        """Return the observed value."""
        return "真实结果"

    async def run():
        service = ConversationAgentService(tmp_path)
        try:
            model = ToolModel(responses=[
                AIMessage(content="", tool_calls=[{"id": "a", "name": "lookup", "args": {}}]),
                AIMessage(content="真实结果是42"),
            ])
            service._agent = create_agent(model, tools=[lookup], system_prompt="实际系统提示", checkpointer=service._checkpointer)
            collector = TrajectoryCollector()
            events = [event async for event in service.stream_turn("u", "t", "查询", telemetry=TurnTelemetry(trajectory=collector))]
            assert events[-1]["event"] == "done"
            sample = collector.sample()
            assert len(collector.calls) == 2
            assert sample["messages"][0] == {"role": "system", "content": "实际系统提示"}
            assert sample["tools"][0]["function"]["name"] == "lookup"
            assert sample["messages"][3]["content"] == "真实结果"
            assert sample["messages"][-1]["content"] == "真实结果是42"
        finally:
            service.close()
    asyncio.run(run())


def test_configured_deep_agent_collects_actual_context(tmp_path, monkeypatch):
    async def run():
        service = ConversationAgentService(tmp_path)
        model = ToolModel(responses=[
            AIMessage(content="", tool_calls=[{"id": "clock-1", "name": "get_current_time", "args": {}}]),
            AIMessage(content="已查询当前时间。"),
        ])
        monkeypatch.setattr(service, "_build_model", lambda: model)
        try:
            collector = TrajectoryCollector()
            events = [event async for event in service.stream_turn("u", "t", "现在几点", telemetry=TurnTelemetry(trajectory=collector))]
            assert events[-1]["event"] == "done", events
            sample = collector.sample()
            assert sample["messages"][0]["role"] == "system"
            assert any(t["function"]["name"] == "get_current_time" for t in sample["tools"])
            assert any(m["role"] == "tool" for m in sample["messages"])
        finally:
            service.close()
    asyncio.run(run())


class JudgeModel:
    def __init__(self, responses):
        self.responses = responses
        self.requests = []

    async def ainvoke(self, request):
        self.requests.append(request)
        value = self.responses[min(len(self.requests) - 1, len(self.responses) - 1)]
        if isinstance(value, Exception):
            raise value
        return AIMessage(content=value)


def test_joint_judge_once_and_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_JUDGE_API_KEY", "fake")
    judge = EvaluationJudge(tmp_path)
    model = JudgeModel([joint().model_dump_json()])
    monkeypatch.setattr(judge, "_build_model", lambda: model)
    sample = trajectory()[0].sample()
    result = asyncio.run(judge.grade_trajectory(case(), [], sample, {}))
    assert result.score == 5 and len(model.requests) == 1
    assert "真实结果" in model.requests[0][1][1]
    invalid = joint(redundant_call_ids=["nonexistent"]).model_dump_json()
    model.responses = [invalid]
    with pytest.raises(ValueError, match="不存在"):
        asyncio.run(judge.grade_trajectory(case(), [], sample, {}))
    assert len(model.requests) == 3
    with pytest.raises(ValueError, match="120000"):
        asyncio.run(judge.grade_trajectory(case(), [], {"messages": [{"content": "x" * 120001}], "tools": []}, {}))
    assert len(model.requests) == 3


@pytest.mark.parametrize("args", [
    ["--distill", "--no-judge"], ["--distill", "--suite", "rag"],
    ["--distill", "--distill-threshold", "nan"], ["--distill", "--distill-weights", "0.2,0.2"],
    ["--no-export-sft"],
])
def test_cli_validation(monkeypatch, args):
    from intelligent_detection_agent.evaluation.__main__ import main
    monkeypatch.setattr("sys.argv", ["evaluation", *args])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def suite_fakes(tmp_path, monkeypatch, *, judge_failure=False, later_failure=False):
    """使用真实 Agent/流协议，替换网络模型和业务数据库。"""
    @tool
    def lookup() -> str:
        """Return observed facts."""
        return "42"

    services = []
    class Service(ConversationAgentService):
        def __init__(self, root):
            super().__init__(root)
            self._agent = create_agent(ToolModel(responses=[
                AIMessage(content="", tool_calls=[{"id": "a", "name": "lookup", "args": {}}]),
                AIMessage(content="答案42"),
            ]), tools=[lookup], system_prompt="system", checkpointer=self._checkpointer)
            self.deleted = []
            services.append(self)

        def delete_thread(self, user_id, thread_id):
            # 导出成功的记录应在清理 checkpoint 前写入并刷新。
            self.deleted.append((tmp_path / "out" / "agent_cases.jsonl").read_text(encoding="utf-8"))
            super().delete_thread(user_id, thread_id)

    class Judge:
        model_name = "joint-fake"
        count = 0

        def __init__(self, root):
            pass

        async def grade_trajectory(self, *args):
            Judge.count += 1
            if judge_failure:
                raise RuntimeError("fake provider failure")
            return joint()

    oracle_count = 0
    def oracle(*args):
        nonlocal oracle_count
        oracle_count += 1
        if later_failure and oracle_count > 1:
            raise RuntimeError("later case failed")
        return []

    monkeypatch.setattr(runner, "ConversationAgentService", Service)
    monkeypatch.setattr(runner, "EvaluationJudge", Judge)
    monkeypatch.setattr(runner, "execute_oracles", oracle)
    return services, Judge


@pytest.mark.parametrize("export,threshold,expected", [(True, 85, 1), (False, 85, 0), (True, 100, 1)])
def test_suite_sft_incremental_and_reporting(tmp_path, monkeypatch, export, threshold, expected):
    services, judge = suite_fakes(tmp_path, monkeypatch)
    result = asyncio.run(runner.run_agent_suite(tmp_path, tmp_path / "out", [case()], repeat=1,
        judge_enabled=True, run_id="test", distillation=DistillationConfig(threshold=threshold, export_sft=export)))
    row = result["rows"][0]
    assert judge.count == 1  # 查询用例原本未开启 Judge，蒸馏模式也只联合评审一次。
    assert row["success"] and row["distillation"]["selected"]
    assert row["distillation"]["exported"] == bool(expected)
    assert result["summary"]["overall"]["distillation"]["exported_count"] == expected
    assert services[0].deleted[0].strip()
    path = tmp_path / "out" / "sft.jsonl"
    assert path.exists() == export
    if export:
        sample = json.loads(path.read_text(encoding="utf-8"))
        assert set(sample) == {"messages", "tools"}
        assert row["distillation"]["sft_line"] == 1
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "蒸馏筛选" in report and "实际导出" in report


def test_judge_failure_keeps_running_and_empty_sft(tmp_path, monkeypatch):
    _, judge = suite_fakes(tmp_path, monkeypatch, judge_failure=True)
    result = asyncio.run(runner.run_agent_suite(tmp_path, tmp_path / "out", [case()], repeat=2,
        judge_enabled=True, run_id="test", distillation=DistillationConfig()))
    assert judge.count == 2
    assert len(result["rows"]) == 2
    assert all(not r["distillation"]["selected"] for r in result["rows"])
    assert (tmp_path / "out" / "sft.jsonl").read_text(encoding="utf-8") == ""


def test_later_failure_preserves_completed_files(tmp_path, monkeypatch):
    suite_fakes(tmp_path, monkeypatch, later_failure=True)
    result = asyncio.run(runner.run_agent_suite(tmp_path, tmp_path / "out", [case()], repeat=2,
        judge_enabled=True, run_id="test", distillation=DistillationConfig()))
    assert [row["success"] for row in result["rows"]] == [True, False]
    assert result["rows"][1]["turns"] == []
    assert len((tmp_path / "out" / "agent_cases.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert len((tmp_path / "out" / "sft.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_sft_write_failure_is_fatal_not_reported_exported(tmp_path, monkeypatch):
    suite_fakes(tmp_path, monkeypatch)
    original_open = Path.open

    class FailingWriter:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def write(self, value):
            raise OSError("disk full")

    def patched_open(path, *args, **kwargs):
        if path.name == "sft.jsonl":
            return FailingWriter()
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", patched_open)
    with pytest.raises(OSError, match="disk full"):
        asyncio.run(runner.run_agent_suite(tmp_path, tmp_path / "out", [case()], repeat=1,
            judge_enabled=True, run_id="test", distillation=DistillationConfig()))
    assert (tmp_path / "out" / "agent_cases.jsonl").read_text(encoding="utf-8") == ""


def test_regrade_no_model_or_fabricated_export(tmp_path, monkeypatch):
    _, judge = suite_fakes(tmp_path, monkeypatch)
    asyncio.run(runner.run_agent_suite(tmp_path, tmp_path / "out", [case()], repeat=1,
        judge_enabled=True, run_id="test", distillation=DistillationConfig(export_sft=False)))
    monkeypatch.setattr(runner, "load_agent_cases", lambda: [case()])
    monkeypatch.setattr(runner, "build_rag_agent_cases", lambda: [])
    runner.regrade_saved_run(tmp_path / "out", distillation=DistillationConfig(threshold=0))
    row = json.loads((tmp_path / "out" / "agent_cases_regraded.jsonl").read_text(encoding="utf-8"))
    assert row["distillation"]["selected"] and not row["distillation"]["exported"]
    assert judge.count == 1
    assert not (tmp_path / "out" / "sft.jsonl").exists()
    original = json.loads((tmp_path / "out" / "agent_cases.jsonl").read_text(encoding="utf-8"))
    del original["distillation"]
    (tmp_path / "out" / "agent_cases.jsonl").write_text(json.dumps(original), encoding="utf-8")
    runner.regrade_saved_run(tmp_path / "out", distillation=DistillationConfig(threshold=0))
    row = json.loads((tmp_path / "out" / "agent_cases_regraded.jsonl").read_text(encoding="utf-8"))
    assert not row["distillation"]["selected"]
    assert row["distillation"]["total_score"] is None
