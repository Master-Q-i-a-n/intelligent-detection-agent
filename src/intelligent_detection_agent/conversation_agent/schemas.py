from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ChatTurnRequest(BaseModel):
    thread_id: str = Field(min_length=8, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    message: str = Field(default="", max_length=8000)
    attachment_ids: list[str] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def require_message_or_attachment(self):
        if not self.message.strip() and not self.attachment_ids:
            raise ValueError("消息文字和图片不能同时为空。")
        if len(set(self.attachment_ids)) != len(self.attachment_ids):
            raise ValueError("同一张图片不能重复提交。")
        return self


class ChatResumeRequest(BaseModel):
    thread_id: str = Field(min_length=8, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    kind: Literal["clarification", "work_order_approval"]
    decision: Literal["answer", "approve", "edit", "reject"]
    message: str = Field(default="", max_length=4000)
    edited_action: dict[str, Any] | None = None


class ChatArtifact(BaseModel):
    type: Literal["query_result", "report", "work_order", "rag_retrieval"]
    id: str
    payload: dict[str, Any]


class ChatTodo(BaseModel):
    content: str
    status: Literal["pending", "in_progress", "completed"]


class ChatTurnResponse(BaseModel):
    status: Literal["completed", "interrupted"]
    message: str = ""
    generator: str
    artifacts: list[ChatArtifact] = Field(default_factory=list)
    interrupt: dict[str, Any] | None = None
    todos: list[ChatTodo] = Field(default_factory=list)
