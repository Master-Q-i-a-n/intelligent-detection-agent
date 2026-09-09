from __future__ import annotations

import json
from pathlib import Path

import pytest

from intelligent_detection_agent.rag.evaluate import DEFAULT_CASES_PATH, load_cases, ranking_metrics


def _item(source: str, chunk_id: str) -> dict:
    return {"payload": {"source": source, "chunk_id": chunk_id}}


def test_ranking_metrics_find_relevant_at_rank_two() -> None:
    metrics = ranking_metrics(
        [_item("a.pdf", "wrong"), _item("a.pdf", "target")],
        {("a.pdf", "target")},
    )

    assert metrics["first_relevant_rank"] == 2
    assert metrics["recall@1"] == 0.0
    assert metrics["recall@5"] == 1.0
    assert metrics["mrr@20"] == 0.5
    assert metrics["ndcg@5"] == pytest.approx(1 / 1.584962500721156)


def test_ranking_metrics_support_multiple_relevant_chunks() -> None:
    metrics = ranking_metrics(
        [_item("a.pdf", "one"), _item("a.pdf", "wrong")],
        {("a.pdf", "one"), ("a.pdf", "two")},
    )

    assert metrics["recall@5"] == 0.5
    assert metrics["hit@5"] == 1.0


def test_ranking_metrics_require_ground_truth() -> None:
    with pytest.raises(ValueError, match="相关 chunk"):
        ranking_metrics([], set())


def test_rag_cases_keep_original_twenty_and_add_ten() -> None:
    cases = load_cases(DEFAULT_CASES_PATH)
    original_ids = [
        "gas_system_01", "gas_system_02", "gas_system_03", "gas_system_04", "gas_system_05",
        "ultrasonic_01", "ultrasonic_02", "ultrasonic_03", "ultrasonic_04", "ultrasonic_05",
        "temperature_01", "temperature_02", "temperature_03", "temperature_04", "temperature_05",
        "pressure_01", "pressure_02", "pressure_03", "pressure_04", "pressure_05",
    ]

    assert len(cases) == 30
    assert [case["id"] for case in cases[:20]] == original_ids
    assert all(case["relevant"] for case in cases[20:])
