from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from conversation_agent import ConversationAgentService
from conversation_agent.schemas import ChatResumeRequest, ChatTurnRequest, ChatTurnResponse


def _encode_sse(event: dict[str, Any]) -> str:
    payload = json.dumps(event.get("data", {}), ensure_ascii=False, default=str, separators=(",", ":"))
    return f"event: {event.get('event', 'message')}\ndata: {payload}\n\n"


async def _sse_stream(events: AsyncIterator[dict[str, Any]]) -> AsyncIterator[str]:
    """在真实事件间发送注释心跳，避免长查询被代理误判为空闲连接。"""

    iterator = events.__aiter__()
    pending: asyncio.Task[dict[str, Any]] | None = asyncio.create_task(anext(iterator))
    try:
        while pending is not None:
            done, _ = await asyncio.wait({pending}, timeout=15)
            if not done:
                yield ": keep-alive\n\n"
                continue
            try:
                event = pending.result()
            except StopAsyncIteration:
                break
            yield _encode_sse(event)
            pending = asyncio.create_task(anext(iterator))
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        close = getattr(iterator, "aclose", None)
        if close is not None:
            await close()


def create_conversation_router(root: Path) -> APIRouter:
    """延迟初始化大模型，避免未配置对话密钥时影响原有检测接口。"""

    router = APIRouter(prefix="/chat", tags=["conversation-agent"])
    service = ConversationAgentService(root)

    @router.get("/status")
    def status():
        return {
            "configured": service.configured,
            "provider": service.provider,
            "model": service.model_name,
            "memory": "in-process-thread-only",
            "tracing_enabled": service.tracing_enabled,
        }

    @router.post("/turns", response_model=ChatTurnResponse)
    def create_turn(request: ChatTurnRequest):
        try:
            return service.turn(request.thread_id, request.message)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"对话 Agent 调用失败：{type(exc).__name__}: {exc}") from exc

    @router.post("/resume", response_model=ChatTurnResponse)
    def resume_turn(request: ChatResumeRequest):
        try:
            return service.resume(request)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"对话 Agent 恢复失败：{type(exc).__name__}: {exc}") from exc

    @router.post("/turns/stream")
    async def stream_turn(request: ChatTurnRequest):
        if not service.configured:
            raise HTTPException(status_code=503, detail="对话 Agent 未配置 API Key。")
        return StreamingResponse(
            _sse_stream(service.stream_turn(request.thread_id, request.message)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.post("/resume/stream")
    async def stream_resume(request: ChatResumeRequest):
        if not service.configured:
            raise HTTPException(status_code=503, detail="对话 Agent 未配置 API Key。")
        return StreamingResponse(
            _sse_stream(service.stream_resume(request)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router
