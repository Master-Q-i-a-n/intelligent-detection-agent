from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime_libs"))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


FLOW_ROOT = Path(r"D:\hf_make\找工作\燃气\数据")
VIBRATION_SOURCE_ROOT = Path(r"D:\hf_make\找工作\燃气\燃气健康状态评估交接内容\数据整理")
DB_PATH = ROOT / "database" / "gas_ai_input.duckdb"
SCHEMA_PATH = ROOT / "schema.sql"
PARQUET_ROOT = ROOT / "dataset" / "vibration"
REPORT_PATH = ROOT / "reports" / "vibration_build_summary.json"

START_DATE = date(2024, 12, 25)
END_DATE = date(2025, 1, 12)
SAMPLE_RATE_HZ = 1000
WINDOW_LENGTH = 500
AXES = ("后盖Signal-x", "后盖Signal_y", "后盖Signal_z")

STAGE_META = {
    "H0": (0, "健康稳定", 0, 85.0, 100.0, "振动形态与健康基准一致"),
    "H1": (1, "轻微衰减", 1, 70.0, 85.0, "出现早期退化但尚未形成显著故障"),
    "H2": (2, "中度衰减", 1, 50.0, 70.0, "退化特征持续增强，建议加强观察"),
    "H3": (3, "重度衰减", 1, 25.0, 50.0, "接近故障边界，建议安排检修"),
    "H4": (4, "故障异常", 2, 0.0, 25.0, "异常振动显著，建议立即核查"),
}

TRAJECTORIES = {
    "stable_healthy": ("健康稳定", "H0→H0→H0", "长期保持健康，仅允许小幅正常波动"),
    "slow_decay": ("缓慢衰减", "H0→H1→H2", "健康指数缓慢下降"),
    "persistent_subhealth": ("持续亚健康", "H1→H2→H2", "由轻微衰减进入并停留在中度衰减"),
    "accelerated_decay": ("加速衰减", "H0→H1→H3→H4", "退化速度逐步加快并最终进入故障"),
    "abrupt_fault": ("突发故障", "H0→H0→H4", "前期稳定，后期发生突发异常"),
    "maintenance_recovery": ("维修恢复", "H4→H3→H1→H0", "故障后经维护逐步恢复"),
}


def stable_int(*parts: object) -> int:
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def date_range(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def collect_enterprises(dates: list[date]) -> list[dict]:
    names: dict[str, Counter] = defaultdict(Counter)
    for current in dates:
        folder = FLOW_ROOT / current.isoformat()
        if not folder.exists():
            raise FileNotFoundError(f"缺少流量日期目录: {folder}")
        for path in folder.iterdir():
            if not path.is_file() or "-" not in path.stem:
                continue
            user_id, raw_name = path.stem.split("-", 1)
            if not user_id.isdigit():
                continue
            company_name = raw_name[len("工业用户") :] if raw_name.startswith("工业用户") else raw_name
            company_name = company_name.strip()
            names[user_id][company_name] += 1
    enterprises = []
    for user_id in sorted(names, key=lambda x: (len(x), x)):
        company_name = names[user_id].most_common(1)[0][0]
        enterprises.append({"user_id": user_id, "company_name": company_name})
    return enterprises


def source_class(path: Path) -> int | None:
    name = path.stem
    if "极端异常" in name:
        return 2
    if "坏表" in name or "疑似坏表" in name:
        return 1
    if "正常" in name or "全新" in name:
        return 0
    return None


def load_signal_bank() -> dict[tuple[int, int], list[dict]]:
    bank: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for condition_dir in sorted(VIBRATION_SOURCE_ROOT.iterdir()):
        if not condition_dir.is_dir() or not condition_dir.name.isdigit():
            continue
        condition = int(condition_dir.name)
        for path in sorted(condition_dir.glob("*.csv")):
            label = source_class(path)
            if label is None:
                continue
            frame = pd.read_csv(path, usecols=list(AXES), encoding="utf-8-sig")
            values = frame.loc[:, AXES].apply(pd.to_numeric, errors="coerce").interpolate().fillna(0.0)
            array = values.to_numpy(dtype=np.float32)
            usable = len(array) // WINDOW_LENGTH
            for index in range(usable):
                start = index * WINDOW_LENGTH
                bank[(condition, label)].append(
                    {
                        "signal": array[start : start + WINDOW_LENGTH].copy(),
                        "source_file": str(path),
                        "source_window": index,
                    }
                )
    for condition in (80, 120, 160):
        for label in (0, 1, 2):
            if not bank[(condition, label)]:
                raise RuntimeError(f"缺少工况{condition}、标签{label}的振动原型")
    return bank


def trajectory_for(user_id: str) -> str:
    value = stable_int("trajectory", user_id) % 100
    if value < 45:
        return "stable_healthy"
    if value < 63:
        return "slow_decay"
    if value < 75:
        return "persistent_subhealth"
    if value < 85:
        return "accelerated_decay"
    if value < 93:
        return "abrupt_fault"
    return "maintenance_recovery"


def stage_for(trajectory: str, day_index: int) -> str:
    if trajectory == "stable_healthy":
        return "H0"
    if trajectory == "slow_decay":
        return "H0" if day_index < 6 else "H1" if day_index < 13 else "H2"
    if trajectory == "persistent_subhealth":
        return "H1" if day_index < 6 else "H2"
    if trajectory == "accelerated_decay":
        return "H0" if day_index < 4 else "H1" if day_index < 8 else "H3" if day_index < 14 else "H4"
    if trajectory == "abrupt_fault":
        return "H0" if day_index < 14 else "H4"
    if trajectory == "maintenance_recovery":
        return "H4" if day_index < 4 else "H3" if day_index < 8 else "H1" if day_index < 13 else "H0"
    raise ValueError(trajectory)


def health_index_for(user_id: str, trajectory: str, day_index: int, stage: str) -> float:
    endpoints = {
        "stable_healthy": (94.0, 92.0),
        "slow_decay": (96.0, 56.0),
        "persistent_subhealth": (82.0, 58.0),
        "accelerated_decay": (96.0, 14.0),
        "abrupt_fault": (94.0, 18.0),
        "maintenance_recovery": (14.0, 93.0),
    }
    start, end = endpoints[trajectory]
    progress = day_index / 18.0
    if trajectory == "abrupt_fault":
        base = 94.0 - 1.5 * min(day_index, 13) / 13.0 if day_index < 14 else 23.0 - 1.25 * (day_index - 14)
    elif trajectory == "stable_healthy":
        base = start + 1.2 * math.sin(day_index * math.pi / 4.0)
    else:
        base = start + (end - start) * progress
    jitter = ((stable_int("health", user_id, day_index) % 1001) / 1000.0 - 0.5) * 1.2
    lower, upper = STAGE_META[stage][3], STAGE_META[stage][4]
    return round(float(np.clip(base + jitter, lower + 0.2, upper - 0.2)), 2)


def stage_weights(stage: str, health_index: float) -> tuple[float, float, float]:
    if stage == "H0":
        return 1.0, 0.0, 0.0
    if stage == "H1":
        alpha = np.clip((85.0 - health_index) / 15.0, 0.15, 0.45)
        return 1.0 - alpha, alpha, 0.0
    if stage == "H2":
        alpha = np.clip((70.0 - health_index) / 20.0, 0.45, 0.80)
        return 1.0 - alpha, alpha, 0.0
    if stage == "H3":
        beta = np.clip((50.0 - health_index) / 25.0, 0.35, 0.80)
        return 0.0, 1.0 - beta, beta
    return 0.0, 0.0, 1.0


def choose(bank: dict, condition: int, label: int, user_id: str, day_index: int) -> dict:
    candidates = bank[(condition, label)]
    return candidates[stable_int("prototype", condition, label, user_id, day_index) % len(candidates)]


def synthesize_signal(bank: dict, condition: int, user_id: str, day_index: int, stage: str, health_index: float):
    prototypes = [choose(bank, condition, label, user_id, day_index) for label in (0, 1, 2)]
    weights = np.asarray(stage_weights(stage, health_index), dtype=np.float32)
    centered = []
    means = []
    rms_values = []
    for item in prototypes:
        signal = item["signal"].astype(np.float32, copy=False)
        mean = signal.mean(axis=0, keepdims=True)
        centered_signal = signal - mean
        centered.append(centered_signal)
        means.append(mean)
        rms_values.append(np.sqrt(np.mean(centered_signal**2, axis=0, keepdims=True) + 1e-12))
    mixed = sum(float(weights[i]) * centered[i] for i in range(3))
    target_rms = sum(float(weights[i]) * rms_values[i] for i in range(3))
    mixed_rms = np.sqrt(np.mean(mixed**2, axis=0, keepdims=True) + 1e-12)
    mixed = mixed * target_rms / mixed_rms
    mean = sum(float(weights[i]) * means[i] for i in range(3))
    rng = np.random.default_rng(stable_int("noise", user_id, day_index) & 0xFFFFFFFF)
    noise_scale = np.maximum(target_rms * 0.01, 1e-6)
    mixed = mixed + mean + rng.normal(0.0, noise_scale, size=mixed.shape).astype(np.float32)
    source_files = json.dumps([Path(p["source_file"]).name for p in prototypes], ensure_ascii=False)
    source_windows = json.dumps([int(p["source_window"]) for p in prototypes])
    return mixed.astype(np.float32), weights, source_files, source_windows


def trend_for(trajectory: str) -> str:
    return {
        "stable_healthy": "stable",
        "slow_decay": "slow_decay",
        "persistent_subhealth": "slow_decay",
        "accelerated_decay": "fast_decay",
        "abrupt_fault": "abrupt_fault",
        "maintenance_recovery": "recovery",
    }[trajectory]


def write_reference_tables(con, enterprises: list[dict], build_id: str) -> None:
    sensor_df = pd.DataFrame(
        [
            {
                "sensor_id": f"VIB-{row['user_id']}-REAR",
                "user_id": row["user_id"],
                "company_name": row["company_name"],
                "meter_id": f"METER-{row['user_id']}-01",
                "sensor_position": "后盖",
                "axis_count": 3,
                "sampling_rate_hz": SAMPLE_RATE_HZ,
                "window_length": WINDOW_LENGTH,
                "source_type": "synthetic_mapping",
            }
            for row in enterprises
        ]
    )
    con.execute("DELETE FROM equipment.vibration_sensor")
    con.register("_sensor_df", sensor_df)
    con.execute(
        """
        INSERT INTO equipment.vibration_sensor
        (sensor_id,user_id,company_name,meter_id,sensor_position,axis_count,sampling_rate_hz,window_length,source_type)
        SELECT sensor_id,user_id,company_name,meter_id,sensor_position,axis_count,sampling_rate_hz,window_length,source_type
        FROM _sensor_df
        """
    )
    con.unregister("_sensor_df")

    con.execute("DELETE FROM vibration.health_label_dictionary")
    con.executemany(
        "INSERT INTO vibration.health_label_dictionary VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(key, *value) for key, value in STAGE_META.items()],
    )
    con.execute("DELETE FROM vibration.trajectory_dictionary")
    con.executemany(
        "INSERT INTO vibration.trajectory_dictionary VALUES (?, ?, ?, ?)",
        [(key, *value) for key, value in TRAJECTORIES.items()],
    )
    con.execute("DELETE FROM vibration.build_manifest")
    con.execute(
        """
        INSERT INTO vibration.build_manifest
        (build_id,start_date,end_date,enterprise_count,date_count,window_count,sample_rate_hz,window_length,axis_count,synthetic_flag,source_root)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, TRUE, ?)
        """,
        [build_id, START_DATE, END_DATE, len(enterprises), 19, len(enterprises) * 19,
         SAMPLE_RATE_HZ, WINDOW_LENGTH, 3, str(VIBRATION_SOURCE_ROOT)],
    )


def build(overwrite: bool) -> dict:
    dates = date_range(START_DATE, END_DATE)
    enterprises = collect_enterprises(dates)
    if len(enterprises) != 715:
        raise RuntimeError(f"企业数量应为715，实际为{len(enterprises)}")
    bank = load_signal_bank()
    if overwrite and PARQUET_ROOT.exists():
        shutil.rmtree(PARQUET_ROOT)
    PARQUET_ROOT.mkdir(parents=True, exist_ok=True)
    trajectory_counts = Counter()
    stage_counts = Counter()
    total_rows = 0
    for day_index, current in enumerate(dates):
        rows = []
        for enterprise in enterprises:
            user_id = enterprise["user_id"]
            trajectory = trajectory_for(user_id)
            stage = stage_for(trajectory, day_index)
            health_index = health_index_for(user_id, trajectory, day_index, stage)
            condition = (80, 120, 160)[stable_int("condition", user_id) % 3]
            signal, weights, source_files, source_windows = synthesize_signal(
                bank, condition, user_id, day_index, stage, health_index
            )
            rows.append(
                {
                    "window_id": hashlib.sha1(f"{user_id}|{current}".encode()).hexdigest()[:24],
                    "user_id": user_id,
                    "company_name": enterprise["company_name"],
                    "meter_id": f"METER-{user_id}-01",
                    "sensor_id": f"VIB-{user_id}-REAR",
                    "observed_at": datetime.combine(current, datetime.min.time()) + timedelta(hours=12),
                    "sampling_rate_hz": SAMPLE_RATE_HZ,
                    "window_length": WINDOW_LENGTH,
                    "operating_condition": condition,
                    "accel_x": signal[:, 0].tolist(),
                    "accel_y": signal[:, 1].tolist(),
                    "accel_z": signal[:, 2].tolist(),
                    "coarse_label": STAGE_META[stage][2],
                    "stage_label": stage,
                    "stage_name": STAGE_META[stage][1],
                    "health_index": health_index,
                    "trend_label": trend_for(trajectory),
                    "trajectory_type": trajectory,
                    "trajectory_name": TRAJECTORIES[trajectory][0],
                    "healthy_weight": round(float(weights[0]), 6),
                    "subhealth_weight": round(float(weights[1]), 6),
                    "fault_weight": round(float(weights[2]), 6),
                    "label_source": "synthetic_from_engineer_labeled_prototypes",
                    "is_synthetic": True,
                    "source_files": source_files,
                    "source_windows": source_windows,
                }
            )
            if day_index == 0:
                trajectory_counts[trajectory] += 1
            stage_counts[stage] += 1
        frame = pd.DataFrame(rows)
        partition = PARQUET_ROOT / f"observation_date={current.isoformat()}"
        partition.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_table(table, partition / "part-00000.parquet", compression="zstd")
        total_rows += len(frame)

    build_id = f"vibration_{datetime.now():%Y%m%d%H%M%S}"
    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        write_reference_tables(con, enterprises, build_id)
        parquet_glob = (PARQUET_ROOT / "observation_date=*" / "*.parquet").as_posix()
        con.execute("DROP VIEW IF EXISTS vibration.daily_health")
        con.execute("DROP VIEW IF EXISTS vibration.acceleration_window")
        con.execute(
            f"""
            CREATE VIEW vibration.acceleration_window AS
            SELECT *, CAST(observation_date AS DATE) AS data_date
            FROM read_parquet('{parquet_glob}', hive_partitioning=1)
            """
        )
        con.execute(
            """
            CREATE VIEW vibration.daily_health AS
            SELECT window_id,user_id,company_name,meter_id,sensor_id,data_date,
                   operating_condition,coarse_label,stage_label,stage_name,
                   health_index,trend_label,trajectory_type,trajectory_name,
                   label_source,is_synthetic
            FROM vibration.acceleration_window
            """
        )
    finally:
        con.close()

    summary = {
        "database": str(DB_PATH),
        "parquet_root": str(PARQUET_ROOT),
        "start_date": START_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
        "date_count": len(dates),
        "enterprise_count": len(enterprises),
        "window_count": total_rows,
        "expected_window_count": 715 * 19,
        "trajectory_counts": dict(trajectory_counts),
        "stage_counts": dict(stage_counts),
        "sampling_rate_hz": SAMPLE_RATE_HZ,
        "window_length": WINDOW_LENGTH,
        "axis_count": 3,
        "synthetic": True,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="构建企业级三轴振动健康轨迹数据库")
    parser.add_argument("--overwrite", action="store_true", help="覆盖既有振动Parquet数据")
    args = parser.parse_args()
    print(json.dumps(build(args.overwrite), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
