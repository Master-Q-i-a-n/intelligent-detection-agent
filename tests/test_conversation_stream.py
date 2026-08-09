from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from conversation_agent.agent import ConversationAgentService
from conversation_api import _sse_stream


class _FakeMessageStream:
    def __init__(self, message_id: str, text: str, output: AIMessage):
        self.node = "model"
        self.message_id = message_id
        self._text = text
        self._output = output

    @property
    def text(self):
        async def chunks():
            for character in self._text:
                yield character

        return chunks()

    @property
    def output(self):
        async def completed():
            return self._output

        return completed()


class _FakeRun:
    def __init__(self):
        self.tool_ai = AIMessage(
            content="",
            tool_calls=[{"id": "tool-1", "name": "get_current_time", "args": {}}],
        )
        self.tool_result = ToolMessage(
            content="当前时间已查询。",
            tool_call_id="tool-1",
            name="get_current_time",
        )
        self.final_ai = AIMessage(content="查询完成。")
        self.final_state = {
            "messages": [self.tool_ai, self.tool_result, self.final_ai],
            "todos": [{"content": "查询当前时间", "status": "completed"}],
        }
        self._interrupts: list[dict[str, Any]] = []
        self.aborted = False

    @property
    def messages(self):
        async def values():
            # 第一段临时文字最终变为工具调用，前端应通过 answer_reset 撤回。
            yield _FakeMessageStream("m1", "准备查询", self.tool_ai)
            yield _FakeMessageStream("m2", "查询完成。", self.final_ai)

        return values()

    @property
    def values(self):
        async def states():
            yield {"messages": [self.tool_ai], "todos": [{"content": "查询当前时间", "status": "in_progress"}]}
            yield {"messages": [self.tool_ai, self.tool_result], "todos": [{"content": "查询当前时间", "status": "completed"}]}
            yield self.final_state

        return states()

    async def output(self):
        return self.final_state

    async def interrupts(self):
        return self._interrupts

    async def abort(self):
        self.aborted = True


class _FakeAgent:
    def __init__(self):
        self.run = _FakeRun()

    async def astream_events(self, *_args: Any, **_kwargs: Any):
        return self.run


def test_stream_protocol_contains_safe_lifecycle(tmp_path: Path):
    async def collect():
        service = ConversationAgentService(tmp_path)
        fake_agent = _FakeAgent()
        service._agent = fake_agent
        events = [event async for event in service.stream_turn("thread-1", "现在几点")]
        return events, fake_agent.run

    events, run = asyncio.run(collect())
    names = [event["event"] for event in events]
    assert names[0:2] == ["meta", "status"]
    assert "answer_reset" in names
    assert "tool_start" in names
    assert "tool_end" in names
    assert names[-1] == "done"
    assert events[-1]["data"]["message"] == "查询完成。"
    assert events[-1]["data"]["todos"] == [{"content": "查询当前时间", "status": "completed"}]
    assert run.aborted is True


def test_sse_transport_serializes_events_and_heartbeats_are_not_required():
    async def source():
        yield {"event": "meta", "data": {"run_id": "r1"}}
        yield {"event": "done", "data": {"status": "completed"}}

    frames = asyncio.run(_collect_frames(source()))
    assert frames[0].startswith("event: meta\n")
    assert json.loads(frames[0].split("data: ", 1)[1]) == {"run_id": "r1"}
    assert frames[-1].startswith("event: done\n")


def test_stream_interrupt_is_a_json_object(tmp_path: Path):
    async def collect():
        service = ConversationAgentService(tmp_path)
        fake_agent = _FakeAgent()
        fake_agent.run._interrupts = [{"kind": "clarification", "question": "请补充阈值"}]
        service._agent = fake_agent
        return [event async for event in service.stream_turn("thread-hitl", "查询超标用户")]

    events = asyncio.run(collect())
    interrupt = next(event for event in events if event["event"] == "interrupt")
    assert interrupt["data"] == {"kind": "clarification", "question": "请补充阈值"}
    assert events[-1]["data"]["status"] == "interrupted"


def test_response_artifacts_only_include_latest_user_turn(tmp_path: Path):
    service = ConversationAgentService(tmp_path)
    old_query = ToolMessage(
        content="旧查询",
        tool_call_id="old-tool",
        artifact={"type": "query_result", "query_id": "query-old", "rows": []},
    )
    new_query = ToolMessage(
        content="新查询",
        tool_call_id="new-tool",
        artifact={"type": "query_result", "query_id": "query-new", "rows": []},
    )
    state = {
        "messages": [
            HumanMessage(content="第一轮"), old_query, AIMessage(content="第一轮完成"),
            HumanMessage(content="第二轮"), new_query, AIMessage(content="第二轮完成"),
        ],
        "todos": [],
    }

    response = service._response_from_state(state, [])

    assert response.message == "第二轮完成"
    assert [artifact.id for artifact in response.artifacts] == ["query-new"]


def test_hitl_resume_keeps_artifacts_from_original_turn(tmp_path: Path):
    service = ConversationAgentService(tmp_path)
    report = ToolMessage(
        content="报告",
        tool_call_id="report-tool",
        artifact={"type": "report", "report_id": "report-current", "datasets": []},
    )
    state = {
        "messages": [HumanMessage(content="生成报告并创建工单"), report],
        "todos": [],
    }

    response = service._response_from_state(
        state,
        [{"kind": "clarification", "question": "请补充报告对象"}],
    )

    assert response.status == "interrupted"
    assert [artifact.id for artifact in response.artifacts] == ["report-current"]


async def _collect_frames(events):
    return [frame async for frame in _sse_stream(events)]
