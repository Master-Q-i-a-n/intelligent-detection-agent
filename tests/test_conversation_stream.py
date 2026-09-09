from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from PIL import Image

from intelligent_detection_agent.conversation_agent.agent import (
    ConversationAgentService,
    ReasoningAwareChatDeepSeek,
    _tool_call_ids,
)
from intelligent_detection_agent.conversation_api import _inspect_chat_image, _sse_stream
from intelligent_detection_agent.evaluation.telemetry import TurnTelemetry


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
        self.final_ai = AIMessage(
            content="查询完成。",
            usage_metadata={"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
        )
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
        telemetry = TurnTelemetry()
        events = [
            event
            async for event in service.stream_turn(
                "test-user",
                "thread-1",
                "现在几点",
                telemetry=telemetry,
            )
        ]
        return events, fake_agent.run, telemetry

    events, run, telemetry = asyncio.run(collect())
    names = [event["event"] for event in events]
    assert names[0:2] == ["meta", "status"]
    assert "answer_reset" in names
    assert "tool_start" in names
    assert "tool_end" in names
    assert names[-1] == "done"
    assert events[-1]["data"]["message"] == "查询完成。"
    assert events[-1]["data"]["todos"] == [{"content": "查询当前时间", "status": "completed"}]
    assert run.aborted is True
    assert telemetry.as_dict()["llm_call_count"] == 2
    assert telemetry.as_dict()["tool_calls"][0]["arguments"] == {}
    assert telemetry.as_dict()["usage"]["total_tokens"] is None
    assert telemetry.first_token_ms == telemetry.model_calls[-1].first_text_ms
    assert all("telemetry" not in event["data"] for event in events)


def test_cancel_thread_aborts_registered_active_run(tmp_path: Path):
    async def cancel():
        service = ConversationAgentService(tmp_path)
        run = _FakeRun()
        scoped_thread_id = service._scoped_thread_id("test-user", "thread-running")
        with service._active_runs_guard:
            service._active_runs[scoped_thread_id] = run

        cancelled = await service.cancel_thread("test-user", "thread-running")
        return cancelled, run

    cancelled, run = asyncio.run(cancel())

    assert cancelled is True
    assert run.aborted is True


def test_sse_transport_serializes_events_and_heartbeats_are_not_required():
    async def source():
        yield {"event": "meta", "data": {"run_id": "r1"}}
        yield {"event": "done", "data": {"status": "completed"}}

    frames = asyncio.run(_collect_frames(source()))
    assert frames[0].startswith("event: meta\n")
    assert json.loads(frames[0].split("data: ", 1)[1]) == {"run_id": "r1"}
    assert frames[-1].startswith("event: done\n")


def test_uploaded_image_is_identified_from_its_content():
    buffer = io.BytesIO()
    Image.new("RGB", (16, 12), "white").save(buffer, format="PNG")

    assert _inspect_chat_image(buffer.getvalue()) == ("image/png", 16, 12)


def test_stream_interrupt_is_a_json_object(tmp_path: Path):
    async def collect():
        service = ConversationAgentService(tmp_path)
        fake_agent = _FakeAgent()
        fake_agent.run._interrupts = [{"kind": "clarification", "question": "请补充阈值"}]
        service._agent = fake_agent
        return [event async for event in service.stream_turn("test-user", "thread-hitl", "查询超标用户")]

    events = asyncio.run(collect())
    interrupt = next(event for event in events if event["event"] == "interrupt")
    assert interrupt["data"] == {"kind": "clarification", "question": "请补充阈值"}
    assert events[-1]["data"]["status"] == "interrupted"


def test_checkpoint_tool_ids_are_available_for_resume_deduplication() -> None:
    messages = [
        AIMessage(content="", tool_calls=[{"id": "ask-1", "name": "ask_user", "args": {}}]),
        ToolMessage(content="已回答", tool_call_id="ask-1"),
    ]

    assert _tool_call_ids(messages) == {"ask-1"}


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


def test_reasoning_content_is_written_back_for_tool_and_cross_turn_messages():
    model = ReasoningAwareChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        api_key="test-key",
        base_url="https://api.deepseek.com",
        extra_body={"thinking": {"type": "enabled"}},
        reasoning_effort="high",
    )
    messages = [
        HumanMessage(content="第一轮"),
        AIMessage(
            content="",
            additional_kwargs={"reasoning_content": "同轮工具推理"},
            tool_calls=[{"id": "call-1", "name": "search_technical_documents", "args": {"query": "测试"}}],
        ),
        ToolMessage(content="资料", tool_call_id="call-1"),
        AIMessage(content="第一轮完成", additional_kwargs={"reasoning_content": "跨轮推理"}),
        HumanMessage(content="继续"),
    ]

    payload = model._get_request_payload(messages)

    assistant_messages = [item for item in payload["messages"] if item["role"] == "assistant"]
    assert [item["reasoning_content"] for item in assistant_messages] == ["同轮工具推理", "跨轮推理"]


def test_user_image_is_encoded_only_in_deepseek_request(tmp_path: Path):
    attachment_id = "img_" + "a" * 32
    attachment_root = tmp_path / "chat_attachments"
    attachment_root.mkdir()
    (attachment_root / attachment_id).write_bytes(b"fake-png-content")
    model = ReasoningAwareChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        api_key="test-key",
        base_url="https://api.deepseek.com",
    )
    model.set_attachment_root(attachment_root)
    user_message = HumanMessage(
        content="识别表计读数",
        additional_kwargs={"image_attachments": [{"id": attachment_id, "mime_type": "image/png"}]},
    )

    payload = model._get_request_payload([user_message])

    assert user_message.content == "识别表计读数"
    assert user_message.additional_kwargs["image_attachments"][0]["id"] == attachment_id
    request_message = payload["messages"][0]
    assert "image_attachments" not in request_message
    assert request_message["content"][0] == {"type": "text", "text": "识别表计读数"}
    assert request_message["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_conversation_model_defaults_to_vision_with_high_thinking(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("CHAT_LLM_MODEL", raising=False)
    monkeypatch.setenv("CHAT_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("CHAT_LLM_API_KEY", "test-key")
    monkeypatch.setenv("CHAT_LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.delenv("CHAT_LLM_THINKING", raising=False)
    monkeypatch.delenv("CHAT_LLM_REASONING_EFFORT", raising=False)
    service = ConversationAgentService(tmp_path)

    model = service._build_model()

    assert service.model_name == "deepseek-v4-flash-vision-exp"
    assert isinstance(model, ReasoningAwareChatDeepSeek)
    assert model.extra_body == {"thinking": {"type": "enabled"}}
    assert model.reasoning_effort == "high"
    assert model.temperature is None


def test_rag_artifact_is_included_in_current_turn_response(tmp_path: Path):
    service = ConversationAgentService(tmp_path)
    rag_result = ToolMessage(
        content="资料",
        tool_call_id="rag-tool",
        artifact={"type": "rag_retrieval", "retrieval_id": "rag-1", "status": "ok", "results": []},
    )
    state = {
        "messages": [HumanMessage(content="查询标准"), rag_result, AIMessage(content="回答 [资料1]")],
        "todos": [],
    }

    response = service._response_from_state(state, [])

    assert [artifact.id for artifact in response.artifacts] == ["rag-1"]
    assert response.artifacts[0].type == "rag_retrieval"


async def _collect_frames(events):
    return [frame async for frame in _sse_stream(events)]
