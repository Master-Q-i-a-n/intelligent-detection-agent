"""运行带人工相关性标注的 RAG 检索评测。"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import fmean
from typing import Any

from .pipeline import PROJECT_ROOT, RagConfig, RagPipeline


DEFAULT_CASES_PATH = Path(__file__).with_name("eval_cases.json")
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output" / "rag_eval_results.json"
METRIC_KS = (1, 5, 10, 20)
CASE_MAX_ATTEMPTS = 3


def _payload_key(payload: dict[str, Any]) -> tuple[str, str]:
    return str(payload.get("source", "")), str(payload.get("chunk_id", ""))


def ranking_metrics(
    items: list[dict[str, Any]],
    relevant: set[tuple[str, str]],
) -> dict[str, float | int | None]:
    """计算二元相关性下的 Recall、Hit、MRR 和 nDCG。"""

    if not relevant:
        raise ValueError("每条评测用例至少需要一个相关 chunk")

    relevance = [
        1 if _payload_key(item.get("payload") or {}) in relevant else 0
        for item in items
    ]
    first_rank = next(
        (rank for rank, value in enumerate(relevance, start=1) if value),
        None,
    )
    metrics: dict[str, float | int | None] = {
        "first_relevant_rank": first_rank,
        "mrr@20": 0.0 if first_rank is None or first_rank > 20 else 1 / first_rank,
    }

    for k in METRIC_KS:
        hits = sum(relevance[:k])
        metrics[f"recall@{k}"] = hits / len(relevant)
        metrics[f"hit@{k}"] = 1.0 if hits else 0.0

        dcg = sum(
            value / math.log2(rank + 1)
            for rank, value in enumerate(relevance[:k], start=1)
        )
        ideal_hits = min(len(relevant), k)
        idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        metrics[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0

    return metrics


def _source_rank(items: list[dict[str, Any]], expected_source: str) -> int | None:
    return next(
        (
            rank
            for rank, item in enumerate(items, start=1)
            if str((item.get("payload") or {}).get("source", ""))
            == expected_source
        ),
        None,
    )


def _mean_metrics(rows: list[dict[str, Any]], ranking: str) -> dict[str, float]:
    names = ["mrr@20"] + [
        f"{prefix}@{k}"
        for k in METRIC_KS
        for prefix in ("recall", "hit", "ndcg")
    ]
    return {
        name: fmean(float(row[ranking][name]) for row in rows)
        for name in names
    }


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("评测集必须是非空 JSON 数组")
    for case in cases:
        if not case.get("id") or not case.get("query") or not case.get("relevant"):
            raise ValueError("每条用例必须包含 id、query 和 relevant")
    return cases


def evaluate(
    cases_path: Path,
    output_path: Path,
    *,
    hybrid_limit: int = 20,
) -> dict[str, Any]:
    """执行检索与重排评测，并保存包含逐题排名的 JSON 报告。"""

    if hybrid_limit < max(METRIC_KS):
        raise ValueError(f"hybrid_limit 不能小于 {max(METRIC_KS)}")

    cases = load_cases(cases_path)
    pipeline = RagPipeline(RagConfig.from_env())
    rows = []

    for index, case in enumerate(cases, start=1):
        relevant = {
            (str(item["source"]), str(item["chunk_id"]))
            for item in case["relevant"]
        }
        result = None
        for attempt in range(CASE_MAX_ATTEMPTS):
            try:
                result = pipeline.search(
                    str(case["query"]),
                    hybrid_limit=hybrid_limit,
                    top_n=hybrid_limit,
                )
                break
            except Exception as exc:
                if attempt == CASE_MAX_ATTEMPTS - 1:
                    raise
                wait_seconds = 2**attempt
                print(
                    f"{case['id']} 第 {attempt + 1} 次执行失败，"
                    f"{wait_seconds} 秒后重试：{exc}"
                )
                time.sleep(wait_seconds)

        if result is None:
            raise RuntimeError(f"评测用例未返回结果：{case['id']}")
        hybrid_items = list(result["hybrid_candidates"])
        reranked_items = list(result["results"])
        expected_source = str(case["relevant"][0]["source"])

        row = {
            "id": case["id"],
            "document": case.get("document"),
            "query": case["query"],
            "expected_answer": case.get("expected_answer"),
            "relevant": case["relevant"],
            "hybrid": ranking_metrics(hybrid_items, relevant),
            "rerank": ranking_metrics(reranked_items, relevant),
            "hybrid_source_rank": _source_rank(hybrid_items, expected_source),
            "rerank_source_rank": _source_rank(reranked_items, expected_source),
            "hybrid_top5": [
                {
                    "source": item["payload"].get("source"),
                    "chunk_id": item["payload"].get("chunk_id"),
                    "score": item.get("rrf_score"),
                }
                for item in hybrid_items[:5]
            ],
            "rerank_top5": [
                {
                    "source": item["payload"].get("source"),
                    "chunk_id": item["payload"].get("chunk_id"),
                    "score": item.get("rerank_score"),
                }
                for item in reranked_items[:5]
            ],
        }
        rows.append(row)
        print(
            f"[{index:02d}/{len(cases)}] {case['id']} "
            f"hybrid_rank={row['hybrid']['first_relevant_rank']} "
            f"rerank_rank={row['rerank']['first_relevant_rank']}"
        )

    report = {
        "collection": pipeline.config.collection_name,
        "case_count": len(cases),
        "hybrid_limit": hybrid_limit,
        "summary": {
            "hybrid_rrf": _mean_metrics(rows, "hybrid"),
            "qwen3_rerank": _mean_metrics(rows, "rerank"),
            "hybrid_source_hit@1": fmean(
                1.0 if row["hybrid_source_rank"] == 1 else 0.0 for row in rows
            ),
            "rerank_source_hit@1": fmean(
                1.0 if row["rerank_source_rank"] == 1 else 0.0 for row in rows
            ),
        },
        "cases": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 20 条标准文档 RAG 检索评测")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--hybrid-limit", type=int, default=20)
    args = parser.parse_args()

    report = evaluate(
        args.cases,
        args.output,
        hybrid_limit=args.hybrid_limit,
    )
    print("=" * 72)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"评测报告已保存：{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
