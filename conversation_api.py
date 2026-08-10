from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from conversation_agent import ConversationAgentService
from conversation_agent.schemas import ChatResumeRequest, ChatTurnRequest, ChatTurnResponse
from user_store import UserStore


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


def _current_user(request: Request) -> dict[str, str]:
    user = getattr(request.state, "user", None)
    if not isinstance(user, dict):
        raise HTTPException(status_code=401, detail="请先登录。")
    return user


def create_conversation_router(root: Path, store: UserStore) -> APIRouter:
    """延迟初始化大模型，避免未配置对话密钥时影响原有检测接口。"""

    router = APIRouter(prefix="/chat", tags=["conversation-agent"])
    service = ConversationAgentService(root)

    @router.on_event("shutdown")
    def close_service() -> None:
        service.close()

    @router.get("/status")
    def status():
        return {
            "configured": service.configured,
            "provider": service.provider,
            "model": service.model_name,
            "memory": "sqlite-user-thread",
            "tracing_enabled": service.tracing_enabled,
        }

    @router.get("/threads")
    def list_threads(http_request: Request):
        user = _current_user(http_request)
        return {"items": store.list_threads(user["user_id"])}

    @router.get("/threads/{thread_id}")
    def get_thread(thread_id: str, http_request: Request):
        user = _current_user(http_request)
        detail = store.thread_detail(user["user_id"], thread_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="对话不存在。")
        return detail

    @router.delete("/threads/{thread_id}", status_code=204)
    def delete_thread(thread_id: str, http_request: Request) -> Response:
        user = _current_user(http_request)
        if not store.owns_thread(user["user_id"], thread_id):
            raise HTTPException(status_code=404, detail="对话不存在。")
        service.delete_thread(user["user_id"], thread_id)
        store.delete_thread_records(user["user_id"], thread_id)
        return Response(status_code=204)

    @router.post("/turns", response_model=ChatTurnResponse)
    def create_turn(request: ChatTurnRequest, http_request: Request):
        user = _current_user(http_request)
        try:
            store.start_turn(user["user_id"], request.thread_id, request.message.strip())
            response = service.turn(user["user_id"], request.thread_id, request.message)
            store.record_response(user["user_id"], request.thread_id, response.model_dump(mode="json"))
            return response
        except PermissionError as exc:
            raise HTTPException(status_code=404, detail="对话不存在。") from exc
        except RuntimeError as exc:
            store.record_error(user["user_id"], request.thread_id, str(exc))
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            store.record_error(user["user_id"], request.thread_id, str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            store.record_error(user["user_id"], request.thread_id, f"{type(exc).__name__}: {exc}")
            raise HTTPException(status_code=502, detail=f"对话 Agent 调用失败：{type(exc).__name__}: {exc}") from exc

    @router.post("/resume", response_model=ChatTurnResponse)
    def resume_turn(request: ChatResumeRequest, http_request: Request):
        user = _current_user(http_request)
        try:
            store.prepare_resume(
                user["user_id"], request.thread_id,
                request.message if request.kind == "clarification" else None,
            )
            response = service.resume(user["user_id"], request)
            store.record_response(user["user_id"], request.thread_id, response.model_dump(mode="json"))
            return response
        except PermissionError as exc:
            raise HTTPException(status_code=404, detail="对话不存在。") from exc
        except RuntimeError as exc:
            store.record_error(user["user_id"], request.thread_id, str(exc), keep_pending=True)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            store.record_error(user["user_id"], request.thread_id, str(exc), keep_pending=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            store.record_error(user["user_id"], request.thread_id, f"{type(exc).__name__}: {exc}", keep_pending=True)
            raise HTTPException(status_code=502, detail=f"对话 Agent 恢复失败：{type(exc).__name__}: {exc}") from exc

    async def persisted_stream(
        events: AsyncIterator[dict[str, Any]], user_id: str, thread_id: str, *, keep_pending_on_error: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        completed = False
        try:
            async for event in events:
                if event.get("event") == "done":
                    store.record_response(user_id, thread_id, dict(event.get("data") or {}))
                    completed = True
                elif event.get("event") == "error":
                    store.record_error(user_id, thread_id, str((event.get("data") or {}).get("message") or "对话执行失败"), keep_pending=keep_pending_on_error)
                yield event
        except asyncio.CancelledError:
            store.record_error(user_id, thread_id, "请求已取消，可从历史对话继续。", keep_pending=keep_pending_on_error)
            raise
        except Exception as exc:
            store.record_error(user_id, thread_id, f"{type(exc).__name__}: {exc}", keep_pending=keep_pending_on_error)
            raise
        finally:
            if not completed:
                # 已有明确 error 时重复写入同类状态无副作用，且能覆盖异常断流。
                detail = store.thread_detail(user_id, thread_id)
                if detail is not None and not detail.get("last_error"):
                    store.record_error(user_id, thread_id, "流式响应未完成，可从历史对话继续。", keep_pending=keep_pending_on_error)

    @router.post("/turns/stream")
    async def stream_turn(request: ChatTurnRequest, http_request: Request):
        if not service.configured:
            raise HTTPException(status_code=503, detail="对话 Agent 未配置 API Key。")
        user = _current_user(http_request)
        try:
            store.start_turn(user["user_id"], request.thread_id, request.message.strip())
        except PermissionError as exc:
            raise HTTPException(status_code=404, detail="对话不存在。") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return StreamingResponse(
            _sse_stream(persisted_stream(
                service.stream_turn(user["user_id"], request.thread_id, request.message),
                user["user_id"], request.thread_id,
            )),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.post("/resume/stream")
    async def stream_resume(request: ChatResumeRequest, http_request: Request):
        if not service.configured:
            raise HTTPException(status_code=503, detail="对话 Agent 未配置 API Key。")
        user = _current_user(http_request)
        try:
            store.prepare_resume(
                user["user_id"], request.thread_id,
                request.message if request.kind == "clarification" else None,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=404, detail="对话不存在。") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return StreamingResponse(
            _sse_stream(persisted_stream(
                service.stream_resume(user["user_id"], request),
                user["user_id"], request.thread_id,
                keep_pending_on_error=True,
            )),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router
