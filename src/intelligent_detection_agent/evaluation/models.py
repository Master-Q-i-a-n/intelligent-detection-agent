"""评测用例和运行结果的数据结构。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ToolRequirement(BaseModel):
    name: str
    min_calls: int = Field(default=1, ge=0)
    max_calls: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_range(self):
        if self.max_calls is not None and self.max_calls < self.min_calls:
            raise ValueError("max_calls 不能小于 min_calls")
        return self


class ArgumentAssertion(BaseModel):
    tool: str
    argument: str
    kind: Literal["exact", "contains_all", "regex", "sql"]
    expected: Any = None
    tables: list[str] = Field(default_factory=list)
    contains: list[str] = Field(default_factory=list)


class NumericAnswerAssertion(BaseModel):
    """允许展示值按业务精度四舍五入，避免用字符串小数位决定成败。"""

    expected: float
    absolute_tolerance: float = Field(default=0.01, ge=0)


class OracleQuery(BaseModel):
    source: Literal["business", "diagnosis", "security"]
    sql: str


class EvalTurn(BaseModel):
    message: str | None = None
    expected_status: Literal["completed", "interrupted"] = "completed"
    resume: dict[str, Any] | None = None

    @model_validator(mode="after")
    def require_action(self):
        if bool(self.message) == bool(self.resume):
            raise ValueError("每一轮必须且只能提供 message 或 resume")
        return self


class AgentEvalCase(BaseModel):
    id: str
    category: Literal["rag", "query", "analysis", "report", "boundary"]
    description: str
    turns: list[EvalTurn] = Field(min_length=1)
    required_tools: list[ToolRequirement] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    argument_assertions: list[ArgumentAssertion] = Field(default_factory=list)
    artifact_types: list[str] = Field(default_factory=list)
    answer_contains: list[str] = Field(default_factory=list)
    answer_numbers: list[NumericAnswerAssertion] = Field(default_factory=list)
    answer_regex: list[str] = Field(default_factory=list)
    oracles: list[OracleQuery] = Field(default_factory=list)
    judge: bool = False
    judge_criteria: list[str] = Field(default_factory=list)
    rag_relevant: list[dict[str, str]] = Field(default_factory=list)
    rag_expected_answer: str | None = None
    require_source_list: bool = False
    require_image_markdown: bool = False


class JudgeResult(BaseModel):
    score: int = Field(ge=1, le=5)
    passed: bool
    reason: str
    usage: dict[str, int | None] = Field(default_factory=dict)
    latency_ms: float | None = None
