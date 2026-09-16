"""评测用例和运行结果的数据结构。"""

from __future__ import annotations

import math
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


class DistillationConfig(BaseModel):
    threshold: float = Field(default=85, ge=0, le=100, allow_inf_nan=False)
    weights: tuple[float, float, float] = (0.4, 0.4, 0.2)
    export_sft: bool = True

    @model_validator(mode="after")
    def validate_weights(self):
        if any(not math.isfinite(v) or v < 0 for v in self.weights) or not math.isclose(sum(self.weights), 1, abs_tol=1e-9):
            raise ValueError("蒸馏权重必须非负、有限且总和为1")
        return self


class ProcessIssue(BaseModel):
    tool_call_id: str
    reason: str = Field(min_length=1)


class JointJudgeResult(JudgeResult):
    parameter_score: Literal[0, 25, 50, 75, 100]
    dependency_score: Literal[0, 25, 50, 75, 100]
    recovery_score: Literal[0, 25, 50, 75, 100]
    issues: list[ProcessIssue]
    redundant_call_ids: list[str]
    justified_repeats: list[ProcessIssue]
    unrecovered_failure: bool
