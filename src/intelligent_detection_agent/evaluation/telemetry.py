"""从 Agent 内部流式事件收集评测指标，不向前端暴露评测数据。"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


def _number(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def normalize_usage(message: Any) -> dict[str, int | None]:
    """统一 LangChain 和 OpenAI 兼容响应中的 token 字段。"""

    usage = dict(getattr(message, "usage_metadata", None) or {})
    response_metadata = dict(getattr(message, "response_metadata", None) or {})
    provider_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
    if isinstance(provider_usage, dict):
        usage = {**provider_usage, **usage}

    input_details = usage.get("input_token_details") or usage.get("prompt_tokens_details") or {}
    output_details = usage.get("output_token_details") or usage.get("completion_tokens_details") or {}
    if not isinstance(input_details, dict):
        input_details = {}
    if not isinstance(output_details, dict):
        output_details = {}

    return {
        "input_tokens": _number(_first(usage, "input_tokens", "prompt_tokens")),
        "output_tokens": _number(_first(usage, "output_tokens", "completion_tokens")),
        "reasoning_tokens": _number(
            _first(output_details, "reasoning", "reasoning_tokens")
            if _first(output_details, "reasoning", "reasoning_tokens") is not None
            else usage.get("reasoning_tokens")
        ),
        "cached_tokens": _number(
            _first(input_details, "cache_read", "cached_tokens")
            if _first(input_details, "cache_read", "cached_tokens") is not None
            else usage.get("cached_tokens")
        ),
        "total_tokens": _number(usage.get("total_tokens")),
    }


@dataclass
class ModelCallTrace:
    message_id: str
    node: str
    started_ms: float
    first_text_ms: float | None = None
    completed_ms: float | None = None
    has_tool_calls: bool = False
    usage: dict[str, int | None] = field(default_factory=dict)


@dataclass
class ToolCallTrace:
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    started_ms: float
    completed_ms: float | None = None
    elapsed_ms: float | None = None
    status: str = "running"


@dataclass
class TurnTelemetry:
    """保存一轮对话的模型、工具和延时指标。"""

    started_at: float = field(default_factory=time.monotonic)
    completed_at: float | None = None
    status: str | None = None
    model_calls: list[ModelCallTrace] = field(default_factory=list)
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    _model_by_id: dict[str, ModelCallTrace] = field(default_factory=dict, repr=False)
    _tool_by_id: dict[str, ToolCallTrace] = field(default_factory=dict, repr=False)

    def _elapsed_ms(self, at: float | None = None) -> float:
        return ((at or time.monotonic()) - self.started_at) * 1000

    def begin_model_call(self, message_id: str, node: str) -> None:
        if message_id in self._model_by_id:
            return
        trace = ModelCallTrace(message_id=message_id, node=node, started_ms=self._elapsed_ms())
        self._model_by_id[message_id] = trace
        self.model_calls.append(trace)

    def record_model_text(self, message_id: str) -> None:
        trace = self._model_by_id.get(message_id)
        if trace is not None and trace.first_text_ms is None:
            trace.first_text_ms = self._elapsed_ms()

    def complete_model_call(self, message_id: str, message: Any) -> None:
        trace = self._model_by_id.get(message_id)
        if trace is None:
            return
        trace.completed_ms = self._elapsed_ms()
        trace.has_tool_calls = bool(getattr(message, "tool_calls", None))
        trace.usage = normalize_usage(message)
        if trace.first_text_ms is None and not trace.has_tool_calls and getattr(message, "content", None):
            # 非流式供应商只能在完整消息返回时确认首字时间。
            trace.first_text_ms = trace.completed_ms

    def begin_tool_call(self, tool_call_id: str, name: str, arguments: Any) -> None:
        if tool_call_id in self._tool_by_id:
            return
        trace = ToolCallTrace(
            tool_call_id=tool_call_id,
            name=name,
            arguments=dict(arguments) if isinstance(arguments, dict) else {},
            started_ms=self._elapsed_ms(),
        )
        self._tool_by_id[tool_call_id] = trace
        self.tool_calls.append(trace)

    def complete_tool_call(self, tool_call_id: str, status: str) -> None:
        trace = self._tool_by_id.get(tool_call_id)
        if trace is None or trace.completed_ms is not None:
            return
        trace.completed_ms = self._elapsed_ms()
        trace.elapsed_ms = trace.completed_ms - trace.started_ms
        trace.status = status

    def finish(self, status: str) -> None:
        if self.completed_at is not None:
            return
        self.completed_at = time.monotonic()
        self.status = status
        for trace in self.tool_calls:
            if trace.completed_ms is None:
                trace.status = "interrupted" if status == "interrupted" else status

    @property
    def end_to_end_ms(self) -> float | None:
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at) * 1000

    @property
    def first_token_ms(self) -> float | None:
        # 只统计最终非工具响应，避免把随后 answer_reset 的临时文本当成首字。
        candidates = [
            trace.first_text_ms
            for trace in self.model_calls
            if trace.node == "model" and not trace.has_tool_calls and trace.first_text_ms is not None
        ]
        return min(candidates) if candidates else None

    def usage_totals(self) -> dict[str, int | None]:
        keys = ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens", "total_tokens")
        totals: dict[str, int | None] = {}
        for key in keys:
            values = [trace.usage.get(key) for trace in self.model_calls]
            known = [value for value in values if value is not None]
            totals[key] = sum(known) if known and len(known) == len(values) else None
        return totals

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "llm_call_count": len(self.model_calls),
            "tool_call_count": len(self.tool_calls),
            "end_to_end_ms": round(self.end_to_end_ms, 2) if self.end_to_end_ms is not None else None,
            "first_token_ms": round(self.first_token_ms, 2) if self.first_token_ms is not None else None,
            "usage": self.usage_totals(),
            "model_calls": [asdict(trace) for trace in self.model_calls],
            "tool_calls": [asdict(trace) for trace in self.tool_calls],
        }
