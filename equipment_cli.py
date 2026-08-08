from __future__ import annotations

import argparse
import json

from smart_equipment import diagnose_user_day, diagnose_user_trend, export_agent_inputs, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="罗茨流量计三轴振动健康诊断")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="训练五阶段健康多任务网络")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--rebuild-cache", action="store_true")

    day = subparsers.add_parser("diagnose", help="诊断单个企业某日状态")
    day.add_argument("--user-id", required=True)
    day.add_argument("--date", required=True)
    day.add_argument("--no-persist", action="store_true")

    trend = subparsers.add_parser("trend", help="诊断单个企业连续健康趋势")
    trend.add_argument("--user-id", required=True)
    trend.add_argument("--start", default="2024-12-25")
    trend.add_argument("--end", default="2025-01-12")
    trend.add_argument("--no-persist", action="store_true")

    subparsers.add_parser("export-agent-inputs", help="为全部企业生成Agent健康输入JSON")

    args = parser.parse_args()
    if args.command == "train":
        result = train_model(args.epochs, args.batch_size, args.learning_rate, args.rebuild_cache)
    elif args.command == "diagnose":
        result = diagnose_user_day(args.user_id, args.date, persist=not args.no_persist)
    elif args.command == "trend":
        result = diagnose_user_trend(args.user_id, args.start, args.end, persist=not args.no_persist)
    else:
        result = export_agent_inputs()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
