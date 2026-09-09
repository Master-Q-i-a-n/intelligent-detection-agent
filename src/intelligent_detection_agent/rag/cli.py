"""独立 RAG 命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .pipeline import (
    DEFAULT_RECORDS_PATH,
    PROJECT_ROOT,
    RagConfig,
    RagPipeline,
)


DEMO_QUERIES = [
    "超声流量计单向测量时，上下游直管段和流动调整器应如何安装？",
    "出现超声噪声时，表 G.1 中哪些自诊断参数会表现异常？",
]


def _excerpt(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[:limit] + "…"


def _print_result(result: dict[str, Any]) -> None:
    print(f"检索问题：{result['retrieval_query']}")
    for index, item in enumerate(result["results"], start=1):
        payload = item["payload"]
        dense = item.get("dense") or {}
        bm25 = item.get("bm25") or {}
        print()
        print(
            f"[{index}] rerank={item['rerank_score']:.6f} "
            f"rrf={item['rrf_score']:.6f} "
            f"dense_rank={dense.get('rank', '-')} "
            f"bm25_rank={bm25.get('rank', '-')}"
        )
        print(
            f"chunk={payload.get('chunk_id')} "
            f"pages={payload.get('page_numbers')} "
            f"headings={payload.get('headings')}"
        )
        images = [
            image.get("image_path")
            for image in payload.get("images", [])
            if image.get("image_path")
        ]
        if images:
            print(f"images={images}")
        print(_excerpt(str(payload.get("text", ""))))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="独立标准文档 RAG 入库与混合检索"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Embedding 并写入 Qdrant")
    index_parser.add_argument(
        "--records",
        type=Path,
        default=DEFAULT_RECORDS_PATH,
        help=f"records.json 路径（默认：{DEFAULT_RECORDS_PATH}）",
    )
    index_parser.add_argument(
        "--recreate",
        action="store_true",
        help="删除并重建专用 Collection",
    )

    query_parser = subparsers.add_parser("query", help="运行一条完整混合检索")
    query_parser.add_argument("question", help="原始问题")
    query_parser.add_argument("--hybrid-limit", type=int, default=20)
    query_parser.add_argument("--top-n", type=int, default=5)

    demo_parser = subparsers.add_parser("demo", help="运行两条固定示例")
    demo_parser.add_argument("--hybrid-limit", type=int, default=20)
    demo_parser.add_argument("--top-n", type=int, default=5)
    demo_parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "rag_demo_results.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    pipeline = RagPipeline(RagConfig.from_env())

    if args.command == "index":
        count = pipeline.index_records(
            args.records,
            recreate=args.recreate,
        )
        print(
            f"入库完成：collection={pipeline.config.collection_name}，"
            f"points={count}"
        )
        return 0

    if args.command == "query":
        result = pipeline.search(
            args.question,
            hybrid_limit=args.hybrid_limit,
            top_n=args.top_n,
        )
        _print_result(result)
        return 0

    results = []
    for question in DEMO_QUERIES:
        print("=" * 72)
        result = pipeline.search(
            question,
            hybrid_limit=args.hybrid_limit,
            top_n=args.top_n,
        )
        _print_result(result)
        results.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"示例结果已保存：{args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"RAG 执行失败：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
