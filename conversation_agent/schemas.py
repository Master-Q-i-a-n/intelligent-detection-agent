from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatTurnRequest(BaseModel):
    thread_id: str = Field(min_length=8, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    message: str = Field(min_length=1, max_length=8000)


class ChatResumeRequest(BaseModel):
    thread_id: str = Field(min_length=8, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    kind: Literal["clarification", "work_order_approval"]
    decision: Literal["answer", "approve", "edit", "reject"]
    message: str = Field(default="", max_length=4000)
    edited_action: dict[str, Any] | None = None


class ChatArtifact(BaseModel):
    type: Literal["query_result", "report", "work_order"]
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
