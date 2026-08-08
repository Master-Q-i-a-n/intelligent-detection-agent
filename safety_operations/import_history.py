from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from .db import connect_database, sync_run_directory


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("配置文件内容必须是 YAML 对象。")
    return config


def resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 YOLO 运行目录幂等导入 SQLite")
    parser.add_argument("--config", default="config.yaml", help="YAML 配置文件路径")
    parser.add_argument("--db", help="覆盖 database.path")
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="要导入的运行目录，可重复传入",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="导入 output.root 下所有 run_* 目录",
    )
    parser.add_argument(
        "--source",
        help="单个 --run-dir 对应的原始视频；批量导入时不自动猜测",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    config_dir = config_path.parent
    database_cfg = config.get("database", {})
    database_path = resolve_path(
        config_dir, args.db or database_cfg.get("path", "data/security.db")
    )

    run_dirs = [resolve_path(Path.cwd(), value) for value in args.run_dir]
    if args.all_runs:
        output_root = resolve_path(config_dir, config.get("output", {}).get("root", "outputs"))
        run_dirs.extend(
            path.resolve()
            for path in sorted(output_root.glob("run_*"))
            if path.is_dir()
        )
    # 同一路径只导入一次，并保持命令行/目录排序。
    run_dirs = list(dict.fromkeys(run_dirs))
    if not run_dirs:
        raise ValueError("请指定 --run-dir，或使用 --all-runs。")
    if args.source and len(run_dirs) != 1:
        raise ValueError("--source 仅能与单个 --run-dir 一起使用。")

    source_path = resolve_path(config_dir, args.source) if args.source else None
    connection = connect_database(
        database_path, int(database_cfg.get("busy_timeout_ms", 5000))
    )
    failed = 0
    try:
        for run_dir in run_dirs:
            if not run_dir.is_dir():
                print(f"跳过不存在的运行目录: {run_dir}", file=sys.stderr)
                failed += 1
                continue
            try:
                result = sync_run_directory(
                    connection, config, run_dir, source_path if len(run_dirs) == 1 else None
                )
                print(
                    f"已同步 {run_dir.name}: "
                    f"事件状态 {result['event_transitions']} 条，复核 {result['reviews']} 条"
                )
            except Exception as exc:
                failed += 1
                print(
                    f"同步 {run_dir} 失败 [{type(exc).__name__}]: {exc}",
                    file=sys.stderr,
                )
    finally:
        connection.close()

    print(f"SQLite: {database_path}")
    return 1 if failed else 0


def main() -> None:
    try:
        exit_code = run(parse_args())
    except Exception as exc:
        print(f"导入任务启动失败: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
