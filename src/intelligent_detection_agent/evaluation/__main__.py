"""综合测评命令行入口。"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from ..rag.pipeline import PROJECT_ROOT
from .models import DistillationConfig
from .runner import (
    build_rag_agent_cases,
    load_agent_cases,
    preflight,
    regrade_saved_run,
    run_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="运行智能检测 Agent 综合测评")
    parser.add_argument("--suite", choices=("all", "rag", "agent"), default="all")
    parser.add_argument("--case", action="append", dest="case_ids", help="只运行指定用例，可重复传入")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1, help="Agent用例并发上限，默认1；问题和重复次数统一调度")
    parser.add_argument("--no-judge", action="store_true", help="关闭开放式任务 LLM 评审")
    parser.add_argument("--preflight", action="store_true", help="只检查配置、数据和参考查询，不调用模型")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--regrade", type=Path, help="按当前规则重判已有输出目录，不调用模型")
    parser.add_argument("--distill", action="store_true", help="联合评分并直接导出入选SFT样本")
    parser.add_argument("--distill-threshold", type=float, default=85, help="蒸馏入选阈值，0至100（含边界）")
    parser.add_argument("--distill-weights", default="0.4,0.4,0.2", help="答案、过程、效率权重，逗号分隔且总和为1")
    parser.add_argument("--no-export-sft", action="store_true", help="蒸馏模式只评分，不生成SFT文件")
    args = parser.parse_args()
    distillation = None
    if args.distill:
        if args.no_judge or args.suite == "rag":
            parser.error("--distill 不能与 --no-judge 或 --suite rag 同时使用")
        try:
            distillation = DistillationConfig(
                threshold=args.distill_threshold,
                weights=tuple(float(value.strip()) for value in args.distill_weights.split(",")),
                export_sft=not args.no_export_sft,
            )
        except ValueError as exc:
            parser.error(str(exc))
    elif args.no_export_sft or args.distill_threshold != 85 or args.distill_weights != "0.4,0.4,0.2":
        parser.error("蒸馏参数需要 --distill")
    if args.repeat < 1:
        parser.error("--repeat 必须大于等于1")
    if args.concurrency < 1:
        parser.error("--concurrency 必须大于等于1")
    if args.regrade:
        result = regrade_saved_run(args.regrade.resolve(), distillation=distillation)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0

    all_cases = build_rag_agent_cases() + load_agent_cases()
    selected = all_cases
    if args.case_ids:
        requested = set(args.case_ids)
        selected = [
            case
            for case in all_cases
            if case.id in requested or case.id.removeprefix("agent_") in requested
        ]
    if args.preflight:
        report = preflight(
            PROJECT_ROOT,
            selected,
            check_rag=args.suite in {"all", "rag"},
            check_agent=args.suite in {"all", "agent"},
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0 if report["passed"] else 1

    result = asyncio.run(
        run_evaluation(
            root=PROJECT_ROOT,
            suite=args.suite,
            case_ids=set(args.case_ids or []),
            repeat=args.repeat,
            concurrency=args.concurrency,
            judge_enabled=not args.no_judge,
            output_root=args.output_root,
            distillation=distillation,
        )
    )
    print(json.dumps({"run_id": result["run_id"], "output_dir": result["output_dir"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
