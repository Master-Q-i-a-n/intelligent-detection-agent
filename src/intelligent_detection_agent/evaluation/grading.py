"""确定性评测规则与聚合指标。"""

from __future__ import annotations

import math
import re
from collections import Counter
from statistics import fmean
from typing import Any

from sqlglot import exp, parse_one

from .models import AgentEvalCase, ArgumentAssertion


FRAMEWORK_TOOLS = {
    "read_file",
    "ls",
    "glob",
    "grep",
    "write_todos",
}


def _tool_name(call: dict[str, Any]) -> str:
    return str(call.get("name") or "")


def grade_tool_policy(case: AgentEvalCase, calls: list[dict[str, Any]]) -> dict[str, Any]:
    """评估工具选择；最大调用次数只衡量效率，不否定任务正确性。"""

    counts = Counter(_tool_name(call) for call in calls)
    required = {item.name: item for item in case.required_tools}
    allowed = set(case.allowed_tools) | set(required) | FRAMEWORK_TOOLS
    forbidden = set(case.forbidden_tools)
    failures: list[str] = []
    budget_failures: list[str] = []
    matched_required = 0
    expected_required = sum(item.min_calls for item in case.required_tools)

    for item in case.required_tools:
        actual = counts[item.name]
        matched_required += min(actual, item.min_calls)
        if actual < item.min_calls:
            failures.append(f"{item.name} 调用不足：{actual} < {item.min_calls}")
        if item.max_calls is not None and actual > item.max_calls:
            budget_failures.append(f"{item.name} 调用过多：{actual} > {item.max_calls}")

    unexpected = []
    for name, count in counts.items():
        if name in forbidden or name not in allowed:
            unexpected.extend([name] * count)
    if unexpected:
        failures.append(f"出现禁止或未允许工具：{', '.join(unexpected)}")

    domain_actual = [name for name in counts.elements() if name not in FRAMEWORK_TOOLS]
    correct_actual = len(domain_actual) - len(unexpected)
    precision = correct_actual / len(domain_actual) if domain_actual else 1.0
    recall = matched_required / expected_required if expected_required else 1.0
    return {
        "passed": not failures,
        "failures": failures,
        "budget_passed": not budget_failures,
        "budget_failures": budget_failures,
        "counts": dict(counts),
        "precision": precision,
        "recall": recall,
    }


def _nested_argument(call: dict[str, Any], key: str) -> Any:
    value: Any = call.get("arguments") or {}
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _grade_sql(value: Any, assertion: ArgumentAssertion) -> tuple[bool, str]:
    if not isinstance(value, str) or not value.strip():
        return False, "SQL 参数为空"
    try:
        expression = parse_one(value, read="duckdb")
    except Exception as exc:
        return False, f"SQL 无法解析：{type(exc).__name__}"
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter, exp.Command)
    if any(expression.find(kind) is not None for kind in forbidden):
        return False, "SQL 不是只读查询"
    actual_tables = {
        f"{table.db}.{table.name}" if table.db else table.name
        for table in expression.find_all(exp.Table)
    }
    missing_tables = [table for table in assertion.tables if table.lower() not in {item.lower() for item in actual_tables}]
    if missing_tables:
        return False, f"SQL 缺少表：{', '.join(missing_tables)}"
    normalized = " ".join(value.lower().split())
    missing_fragments = [item for item in assertion.contains if item.lower() not in normalized]
    if missing_fragments:
        return False, f"SQL 缺少条件：{', '.join(missing_fragments)}"
    return True, ""


def _assert_argument(value: Any, assertion: ArgumentAssertion) -> tuple[bool, str]:
    if assertion.kind == "exact":
        return value == assertion.expected, f"实际值 {value!r} != {assertion.expected!r}"
    if assertion.kind == "contains_all":
        text = str(value or "").lower()
        missing = [str(item) for item in (assertion.expected or []) if str(item).lower() not in text]
        return not missing, f"缺少内容：{', '.join(missing)}"
    if assertion.kind == "regex":
        passed = re.search(str(assertion.expected), str(value or ""), re.IGNORECASE) is not None
        return passed, f"未匹配正则：{assertion.expected}"
    return _grade_sql(value, assertion)


def grade_arguments(case: AgentEvalCase, calls: list[dict[str, Any]]) -> dict[str, Any]:
    """参数可由一次调用满足；文本关键词也允许分布在多次检索中。"""

    results = []
    for assertion in case.argument_assertions:
        candidates = [call for call in calls if _tool_name(call) == assertion.tool]
        candidate_results = [
            _assert_argument(_nested_argument(call, assertion.argument), assertion)
            for call in candidates
        ]
        if assertion.kind == "contains_all" and len(candidates) > 1:
            combined = "\n".join(str(_nested_argument(call, assertion.argument) or "") for call in candidates)
            candidate_results.append(_assert_argument(combined, assertion))
        passed = any(item[0] for item in candidate_results)
        reasons = [item[1] for item in candidate_results if item[1]]
        results.append(
            {
                "tool": assertion.tool,
                "argument": assertion.argument,
                "kind": assertion.kind,
                "passed": passed,
                "reason": "；".join(reasons) if reasons else ("" if passed else "没有对应工具调用"),
            }
        )
    passed_count = sum(1 for item in results if item["passed"])
    return {
        "passed": passed_count == len(results),
        "accuracy": passed_count / len(results) if results else 1.0,
        "assertions": results,
    }


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }
