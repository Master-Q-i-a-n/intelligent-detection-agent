from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import HumanMessage

from intelligent_detection_agent.conversation_agent.agent import ConversationAgentService


@pytest.fixture
def compatible_env(monkeypatch):
    monkeypatch.setenv("CHAT_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("CHAT_LLM_API_KEY", "local")
    monkeypatch.setenv("CHAT_LLM_BASE_URL", "http://127.0.0.1:18000/v1")
    monkeypatch.setenv("CHAT_LLM_MODEL", "agent-sft")
    monkeypatch.setenv("CHAT_LLM_CONTEXT_WINDOW", "20000")
    monkeypatch.delenv("CHAT_LLM_CHAT_TEMPLATE_KWARGS", raising=False)
    monkeypatch.delenv("CHAT_LLM_STREAM_USAGE", raising=False)


def test_template_kwargs_reach_http_body(tmp_path: Path, monkeypatch, compatible_env):
    monkeypatch.setenv("CHAT_LLM_CHAT_TEMPLATE_KWARGS", '{"enable_thinking":false}')
    requests = []

    def respond(request: httpx.Request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "test-completion", "object": "chat.completion", "created": 0,
            "model": "agent-sft",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "查询完成"}}],
        })

    service = ConversationAgentService(tmp_path)
    try:
        model = service._build_model()
        # 经过真实 SDK 序列化，验证扩展参数落在 HTTP 正文而不是 model_kwargs 中。
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            model = type(model)(
                model=model.model_name, api_key="local", base_url=model.openai_api_base,
                extra_body=model.extra_body, http_client=client, streaming=False,
            )
            assert model.invoke([HumanMessage(content="测试")]).content == "查询完成"
        assert requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
        assert "extra_body" not in requests[0]
        assert "thinking" not in requests[0]
        assert "reasoning_effort" not in requests[0]
        original = service._build_model()
        assert original.profile["max_input_tokens"] == 20000
        assert "max_input_tokens" not in requests[0]
    finally:
        service.close()


@pytest.mark.parametrize("value", ["{invalid", "[]", "null", "false", '"text"'])
def test_invalid_template_config_fails_before_checkpoint(
    tmp_path: Path, monkeypatch, compatible_env, value: str,
):
    monkeypatch.setenv("CHAT_LLM_CHAT_TEMPLATE_KWARGS", value)
    with pytest.raises(ValueError, match="CHAT_LLM_CHAT_TEMPLATE_KWARGS"):
        ConversationAgentService(tmp_path)
    assert not (tmp_path / "database").exists()


def test_unset_template_config_preserves_compatible_requests(tmp_path: Path, compatible_env):
    service = ConversationAgentService(tmp_path)
    try:
        assert service._build_model().extra_body is None
        assert service._build_model().stream_usage is None
    finally:
        service.close()


def test_deepseek_ignores_compatible_template_config(tmp_path: Path, monkeypatch, compatible_env):
    monkeypatch.setenv("CHAT_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("CHAT_LLM_MODEL", "deepseek-flash")
    monkeypatch.setenv("CHAT_LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("CHAT_LLM_THINKING", "enabled")
    monkeypatch.setenv("CHAT_LLM_REASONING_EFFORT", "high")
    monkeypatch.setenv("CHAT_LLM_CHAT_TEMPLATE_KWARGS", '{"enable_thinking":false}')
    monkeypatch.setenv("CHAT_LLM_STREAM_USAGE", "true")
    service = ConversationAgentService(tmp_path)
    try:
        model = service._build_model()
        assert model.extra_body == {"thinking": {"type": "enabled"}}
        assert model.reasoning_effort == "high"
        assert model.temperature is None
    finally:
        service.close()


@pytest.mark.parametrize("value", ["true", "false"])
def test_stream_usage_reaches_sdk(tmp_path: Path, monkeypatch, compatible_env, value: str):
    monkeypatch.setenv("CHAT_LLM_STREAM_USAGE", value)
    service = ConversationAgentService(tmp_path)
    try:
        assert service._build_model().stream_usage is (value == "true")
    finally:
        service.close()


def test_invalid_stream_usage_fails_before_checkpoint(tmp_path: Path, monkeypatch, compatible_env):
    monkeypatch.setenv("CHAT_LLM_STREAM_USAGE", "yes")
    with pytest.raises(ValueError, match="CHAT_LLM_STREAM_USAGE"):
        ConversationAgentService(tmp_path)
    assert not (tmp_path / "database").exists()


def test_project_env_preserves_quoted_template_json(tmp_path: Path, monkeypatch, compatible_env):
    (tmp_path / ".env").write_text(
        'CHAT_LLM_CHAT_TEMPLATE_KWARGS=\'{"enable_thinking":false}\'\n', encoding="utf-8",
    )
    # 项目加载器和 uv --env-file 都必须支持同一份带引号的配置。
    service = ConversationAgentService(tmp_path)
    try:
        assert service.chat_template_kwargs == {"enable_thinking": False}
    finally:
        service.close()
    monkeypatch.delenv("CHAT_LLM_CHAT_TEMPLATE_KWARGS")
