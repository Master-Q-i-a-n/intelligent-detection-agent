import json

import pytest

from intelligent_detection_agent.evaluation.grading import grade_arguments, grade_tool_policy
from intelligent_detection_agent.evaluation.models import AgentEvalCase
from intelligent_detection_agent.evaluation.runner import _hard_grade, load_agent_cases


def grade_answer(answer, **fields):
    case = AgentEvalCase(id="format", category="query", description="格式", turns=[{"message": "查询"}], **fields)
    turn = dict(answer=answer, artifacts=[], actual_status="completed", expected_status="completed",
                error=None, telemetry={"tool_calls": []})
    return _hard_grade(case, [turn])["passed"]


@pytest.mark.parametrize("date", ["2025年1月12日", "2025/1/12", "2025-01-12"])
def test_equivalent_dates(date):
    assert grade_answer(date, answer_contains=["2025-01-12"])
    assert not grade_answer(date, answer_contains=["2026-01-12"])


@pytest.mark.parametrize("number", ["9 467.8323", "9\u202f467.8323", "9,467.8323", "9467.83"])
def test_numeric_format_and_tolerance(number):
    assert grade_answer(number, answer_numbers=[{"expected": 9467.8323}])
    assert not grade_answer("9467.80", answer_numbers=[{"expected": 9467.8323}])
    assert not grade_answer("9\n467.8323", answer_numbers=[{"expected": 9467.8323}])


def test_soft_tool_path_and_hard_prohibition():
    case = next(c for c in load_agent_cases() if c.id == "query_05_equipment_diagnosis")
    calls = [{"name": "query_diagnosis_data"}, {"name": "query_business_data"}]
    assert grade_tool_policy(case, calls)["passed"]
    assert not grade_tool_policy(case, calls + [{"name": "create_work_order"}])["passed"]
    assert not grade_tool_policy(case, [{"name": "query_business_data"}])["passed"]


def test_full_result_filter_is_reviewed_by_judge():
    case = next(c for c in load_agent_cases() if c.id == "query_06_confirmed_security")
    assert case.judge
    assert grade_arguments(case, [{"name": "query_security_data", "arguments": {
        "sql": "SELECT event_id, final_decision, handling_status FROM security_events"
    }}])["passed"]
    assert not grade_arguments(case, [{"name": "query_security_data", "arguments": {
        "sql": "DELETE FROM security_events"
    }}])["passed"]


def test_case_numeric_answers_allow_unrounded_values():
    case = next(c for c in load_agent_cases() if c.id == "analysis_02_metering_gas_followup")
    assert grade_answer("798.3951 834.3291 710.6405", answer_numbers=case.answer_numbers)


def test_context_diff_is_specific_without_archiving_content():
    from intelligent_detection_agent.evaluation.distillation import TrajectoryCollector

    collector = TrajectoryCollector()
    collector.calls = {
        "first": {"messages": [{"role": "user", "content": "private-input"}], "tools": [],
                  "output": {"role": "assistant", "content": "private-answer"}},
        "second": {"messages": [{"role": "user", "content": "changed-input"}], "tools": [],
                   "output": {"role": "assistant", "content": "answer"}},
    }
    with pytest.raises(ValueError) as error:
        collector.sample()
    assert "模型调用序号=2" in str(error.value)
    assert "消息序号=1" in str(error.value)
    assert "字段=content" in str(error.value)
    assert "private" not in str(error.value)
