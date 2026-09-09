from __future__ import annotations

from langchain_core.messages import AIMessage

from intelligent_detection_agent.evaluation.grading import (
    grade_arguments,
    grade_tool_policy,
    percentile,
)
from intelligent_detection_agent.evaluation.judge import _parse_json
from intelligent_detection_agent.evaluation.models import (
    AgentEvalCase,
    ArgumentAssertion,
    NumericAnswerAssertion,
    ToolRequirement,
)
from intelligent_detection_agent.evaluation.runner import (
    _hard_grade,
    _judge_execution_context,
    _unique_tool_calls,
    build_rag_agent_cases,
    load_agent_cases,
)
from intelligent_detection_agent.evaluation.telemetry import TurnTelemetry, normalize_usage


def _case(**updates) -> AgentEvalCase:
    values = {
        "id": "case",
        "category": "query",
        "description": "测试",
        "turns": [{"message": "测试"}],
        "required_tools": [ToolRequirement(name="query_business_data", min_calls=1, max_calls=1)],
        "allowed_tools": ["query_business_data"],
    }
    values.update(updates)
    return AgentEvalCase.model_validate(values)


def test_agent_case_distribution_and_rag_double_run() -> None:
    cases = load_agent_cases()
    counts = {category: sum(case.category == category for case in cases) for category in {case.category for case in cases}}

    assert len(cases) == 20
    assert counts == {"query": 6, "analysis": 6, "report": 4, "boundary": 4}
    assert sum(len(case.turns) > 1 for case in cases) == 4
    assert len(build_rag_agent_cases()) == 10


def test_tool_policy_detects_missing_and_unexpected_tools() -> None:
    result = grade_tool_policy(
        _case(forbidden_tools=["query_security_data"]),
        [{"name": "query_security_data", "arguments": {}}],
    )

    assert result["passed"] is False
    assert result["recall"] == 0.0
    assert any("调用不足" in failure for failure in result["failures"])
    assert any("禁止" in failure for failure in result["failures"])


def test_tool_call_budget_is_reported_without_failing_correct_tool_choice() -> None:
    result = grade_tool_policy(
        _case(),
        [
            {"name": "query_business_data", "arguments": {}},
            {"name": "query_business_data", "arguments": {}},
        ],
    )

    assert result["passed"] is True
    assert result["budget_passed"] is False
    assert result["budget_failures"] == ["query_business_data 调用过多：2 > 1"]


def test_sql_parameter_grading_is_semantic_not_string_exact() -> None:
    case = _case(
        argument_assertions=[
            ArgumentAssertion(
                tool="query_business_data",
                argument="sql",
                kind="sql",
                tables=["telemetry.scada_observation"],
                contains=["2025-01-12", "time_bucket"],
            )
        ]
    )
    result = grade_arguments(
        case,
        [
            {
                "name": "query_business_data",
                "arguments": {
                    "sql": "SELECT time_bucket(INTERVAL '5 minutes', observed_at) FROM telemetry.scada_observation WHERE data_date=DATE '2025-01-12'"
                },
            }
        ],
    )

    assert result["passed"] is True
    assert result["accuracy"] == 1.0


def test_contains_all_parameter_can_span_multiple_retrieval_calls() -> None:
    case = _case(
        required_tools=[ToolRequirement(name="search_technical_documents")],
        allowed_tools=["search_technical_documents"],
        argument_assertions=[
            ArgumentAssertion(
                tool="search_technical_documents",
                argument="query",
                kind="contains_all",
                expected=["X-Well", "Hot Backup"],
            )
        ],
    )
    result = grade_arguments(
        case,
        [
            {"name": "search_technical_documents", "arguments": {"query": "X-Well 能力"}},
            {"name": "search_technical_documents", "arguments": {"query": "Hot Backup 能力"}},
        ],
    )

    assert result["passed"] is True


def test_numeric_answer_assertion_allows_business_rounding() -> None:
    case = _case(
        required_tools=[],
        allowed_tools=[],
        answer_numbers=[NumericAnswerAssertion(expected=9467.8323, absolute_tolerance=0.01)],
    )
    turn = {
        "expected_status": "completed",
        "actual_status": "completed",
        "answer": "实测量为 9,467.83 m³。",
        "interrupt": None,
        "artifacts": [],
        "error": None,
        "telemetry": {"tool_calls": []},
    }

    assert _hard_grade(case, [turn])["passed"] is True


def test_answer_fragment_ignores_numeric_thousands_separator() -> None:
    case = _case(required_tools=[], allowed_tools=[], answer_contains=["127965.56"])
    turn = {
        "expected_status": "completed",
        "actual_status": "completed",
        "answer": "日用气量为 127,965.56 m³。",
        "interrupt": None,
        "artifacts": [],
        "error": None,
        "telemetry": {"tool_calls": []},
    }

    assert _hard_grade(case, [turn])["passed"] is True


def test_hitl_replayed_tool_call_is_counted_once_and_judge_sees_interrupt() -> None:
    call = {"tool_call_id": "ask-1", "name": "ask_user", "arguments": {"question": "日期？"}}
    turns = [
        {
            "expected_status": "interrupted",
            "actual_status": "interrupted",
            "interrupt": {"kind": "clarification"},
            "artifacts": [],
            "error": None,
            "telemetry": {"tool_calls": [call]},
        },
        {
            "expected_status": "completed",
            "actual_status": "completed",
            "interrupt": None,
            "artifacts": [{"type": "report", "id": "report-1"}],
            "error": None,
            "telemetry": {"tool_calls": [call]},
        },
    ]

    assert len(_unique_tool_calls(turns)) == 1
    context = _judge_execution_context(turns)
    assert context["turns"][0]["has_interrupt"] is True
    assert context["tool_counts"] == {"ask_user": 1}
    assert context["artifact_types"] == {"report": 1}


def test_telemetry_ignores_tool_preamble_for_first_token() -> None:
    telemetry = TurnTelemetry()
    tool_message = AIMessage(content="准备查询", tool_calls=[{"id": "t1", "name": "x", "args": {}}])
    final_message = AIMessage(
        content="最终答案",
        usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    )
    telemetry.begin_model_call("m1", "model")
    telemetry.record_model_text("m1")
    telemetry.complete_model_call("m1", tool_message)
    telemetry.begin_model_call("m2", "model")
    telemetry.record_model_text("m2")
    telemetry.complete_model_call("m2", final_message)
    telemetry.finish("completed")

    assert telemetry.first_token_ms == telemetry.model_calls[1].first_text_ms
    assert telemetry.usage_totals()["total_tokens"] is None


def test_usage_keeps_zero_cached_and_reasoning_tokens() -> None:
    message = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": 3,
            "output_tokens": 1,
            "total_tokens": 4,
            "input_token_details": {"cache_read": 0},
            "output_token_details": {"reasoning": 0},
        },
    )

    assert normalize_usage(message)["cached_tokens"] == 0
    assert normalize_usage(message)["reasoning_tokens"] == 0


def test_percentile_and_judge_json_parser() -> None:
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert _parse_json("```json\n{\"score\": 4, \"passed\": true, \"reason\": \"ok\"}\n```")["score"] == 4
