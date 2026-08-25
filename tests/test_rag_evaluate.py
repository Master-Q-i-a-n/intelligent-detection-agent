from __future__ import annotations

import pytest

from rag.evaluate import ranking_metrics


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
