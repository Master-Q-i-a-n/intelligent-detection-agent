from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from uuid import uuid4

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from intelligent_detection_agent.conversation_agent.agent import ConversationAgentService
from intelligent_detection_agent.evaluation import runner
from intelligent_detection_agent.evaluation.models import AgentEvalCase, DistillationConfig, JointJudgeResult


def make_case(name):
    return AgentEvalCase(id=name, category="query", description=name, turns=[{"message": name}])


@pytest.mark.parametrize("concurrency", [1, 2, 30])
def test_unified_queue_limit_completion_order_and_no_omissions(tmp_path, monkeypatch, concurrency):
    active = 0
    peak = 0
    started = []
    completed = []
    overlaps = set()
    active_keys = set()

    class Service:
        def __init__(self, root):
            pass

        def _get_agent(self):
            pass

        def close(self):
            assert active == 0

    async def evaluate(service, judge, root, case, *, repeat_index, on_result, **kwargs):
        nonlocal active, peak
        key = (case.id, repeat_index)
        active += 1
        active_keys.add(key)
        peak = max(peak, active)
        if len({k[0] for k in active_keys}) > 1:
            overlaps.add("questions")
        if len({k[1] for k in active_keys}) > 1:
            overlaps.add("repeats")
        started.append(key)
        try:
            await asyncio.sleep(0.005 if case.id == "b" else 0.015)
            row = {"id": case.id, "repeat": repeat_index, "success": True}
            on_result(row, None)
            completed.append(key)
            return row
        finally:
            active_keys.remove(key)
            active -= 1

    monkeypatch.setattr(runner, "ConversationAgentService", Service)
    monkeypatch.setattr(runner, "run_agent_case", evaluate)
    monkeypatch.setattr(runner, "summarize_agent", lambda rows: {})
    monkeypatch.setattr(runner, "_markdown_report", lambda *args: "report")
    result = asyncio.run(runner.run_agent_suite(tmp_path, tmp_path / "out", [make_case("a"), make_case("b")],
        repeat=20, judge_enabled=False, run_id="test", concurrency=concurrency))
    assert len(set(started)) == len(started) == len(completed) == 40
    assert peak == concurrency
    if concurrency == 30:
        assert overlaps == {"questions", "repeats"}
    saved = [json.loads(line) for line in (tmp_path / "out" / "agent_cases.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["id"], r["repeat"]) for r in saved] == completed
    assert result["summary"]["execution"]["worker_count"] == concurrency


@pytest.mark.parametrize("external_cancel", [True, False])
def test_cancellation_and_global_write_failure_wait_for_cleanup(tmp_path, monkeypatch, external_cancel):
    active = 0
    cleaned = 0
    began = asyncio.Event()

    class Service:
        def __init__(self, root):
            pass

        def _get_agent(self):
            pass

        def close(self):
            assert active == 0 and cleaned == 3

    async def evaluate(*args, on_result, **kwargs):
        nonlocal active, cleaned
        active += 1
        if active == 3:
            began.set()
        try:
            await began.wait()
            if not external_cancel:
                # 模拟持久化失败：不能转成普通用例失败而继续队列。
                raise OSError("disk full")
            await asyncio.Event().wait()
        finally:
            try:
                await runner._run_sync(time.sleep, 0.02)
            finally:
                active -= 1
                cleaned += 1

    monkeypatch.setattr(runner, "ConversationAgentService", Service)
    monkeypatch.setattr(runner, "run_agent_case", evaluate)

    async def run():
        task = asyncio.create_task(runner.run_agent_suite(tmp_path, tmp_path / "out", [make_case("a")],
            repeat=10, judge_enabled=False, run_id="test", concurrency=3))
        await began.wait()
        if external_cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if external_cancel else OSError):
            await task
        assert active == 0
    asyncio.run(run())


class EchoModel(FakeMessagesListChatModel):
    """按输入作答，不依赖共享响应序号，用于验证并发会话隔离。"""
    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        time.sleep(0.01)
        if messages[-1].type == "tool":
            message = AIMessage(content=messages[-1].content)
        else:
            message = AIMessage(content="", tool_calls=[{
                "id": uuid4().hex, "name": "echo", "args": {"value": messages[-1].content},
            }])
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_real_graph_multiturn_shared_checkpoint_sft_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    @tool
    def echo(value: str) -> str:
        """Return the supplied observation."""
        return value

    class Service(ConversationAgentService):
        def __init__(self, root):
            super().__init__(root)
            self._agent = create_agent(EchoModel(responses=[]), tools=[echo], system_prompt="echo",
                                       checkpointer=self._checkpointer)

        def close(self):
            assert not self._active_runs
            assert self._checkpoint_connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 0
            super().close()

    class Judge:
        model_name = "fake"
        def __init__(self, root):
            pass

        async def grade_trajectory(self, case, oracle, sample, context):
            await asyncio.sleep(0.01)
            human = [m["content"] for m in sample["messages"] if m["role"] == "user"]
            observed = [m["content"] for m in sample["messages"] if m["role"] == "tool"]
            assert human == observed == [t.message for t in case.turns]
            return JointJudgeResult(score=5, passed=True, reason="ok", parameter_score=100,
                dependency_score=100, recovery_score=100, issues=[], redundant_call_ids=[],
                justified_repeats=[], unrecovered_failure=False)

    monkeypatch.setattr(runner, "ConversationAgentService", Service)
    monkeypatch.setattr(runner, "EvaluationJudge", Judge)
    monkeypatch.setattr(runner, "execute_oracles", lambda *args: [])
    cases = [AgentEvalCase(id=name, category="query", description=name,
        turns=[{"message": name}, {"message": name + " followup"}],
        required_tools=[{"name": "echo", "min_calls": 2}]) for name in ["a", "b"]]
    result = asyncio.run(runner.run_agent_suite(tmp_path, tmp_path / "out", cases,
        repeat=3, judge_enabled=True, run_id="test", concurrency=30, distillation=DistillationConfig()))
    assert len({row["thread_id"] for row in result["rows"]}) == 6
    samples = [json.loads(line) for line in (tmp_path / "out" / "sft.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(samples) == 6
    for row in result["rows"]:
        assert row["success"] and row["distillation"]["selected"]
        sample = samples[row["distillation"]["sft_line"] - 1]
        assert sample["messages"][1]["content"] == row["id"]


@pytest.mark.parametrize("value", [0, -1])
def test_cli_concurrency_validation(monkeypatch, value):
    from intelligent_detection_agent.evaluation.__main__ import main
    monkeypatch.setattr("sys.argv", ["evaluation", "--concurrency", str(value)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
