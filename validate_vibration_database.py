from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime_libs"))

import duckdb  # noqa: E402


DB_PATH = ROOT / "database" / "gas_ai_input.duckdb"
REPORT_PATH = ROOT / "reports" / "vibration_validation.json"

EXPECTED_SEQUENCES = {
    "stable_healthy": ["H0"],
    "slow_decay": ["H0", "H1", "H2"],
    "persistent_subhealth": ["H1", "H2"],
    "accelerated_decay": ["H0", "H1", "H3", "H4"],
    "abrupt_fault": ["H0", "H4"],
    "maintenance_recovery": ["H4", "H3", "H1", "H0"],
}


def compress(values):
    result = []
    for value in values:
        if not result or result[-1] != value:
            result.append(value)
    return result


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        summary_row = con.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT user_id), COUNT(DISTINCT data_date),
                   MIN(data_date), MAX(data_date),
                   COUNT(*) - COUNT(accel_x), COUNT(*) - COUNT(accel_y), COUNT(*) - COUNT(accel_z),
                   MIN(len(accel_x)), MAX(len(accel_x)),
                   MIN(len(accel_y)), MAX(len(accel_y)),
                   MIN(len(accel_z)), MAX(len(accel_z))
            FROM vibration.acceleration_window
            """
        ).fetchone()
        coverage_violations = con.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT user_id, COUNT(DISTINCT data_date) AS n
                FROM vibration.acceleration_window GROUP BY user_id HAVING n <> 19
            )
            """
        ).fetchone()[0]
        duplicate_pairs = con.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT user_id,data_date,COUNT(*) n FROM vibration.acceleration_window
                GROUP BY user_id,data_date HAVING n <> 1
            )
            """
        ).fetchone()[0]
        bounds_violations = con.execute(
            """
            SELECT COUNT(*)
            FROM vibration.acceleration_window a
            JOIN vibration.health_label_dictionary d USING(stage_label)
            WHERE a.health_index < d.health_index_min OR a.health_index > d.health_index_max
            """
        ).fetchone()[0]
        trajectories = con.execute(
            """
            SELECT user_id,trajectory_type,list(stage_label ORDER BY data_date)
            FROM vibration.acceleration_window
            GROUP BY user_id,trajectory_type ORDER BY user_id
            """
        ).fetchall()
        bad_trajectories = []
        for user_id, trajectory_type, stages in trajectories:
            actual = compress(stages)
            expected = EXPECTED_SEQUENCES[trajectory_type]
            if actual != expected:
                bad_trajectories.append({"user_id": user_id, "actual": actual, "expected": expected})
        report = {
            "window_count": summary_row[0],
            "enterprise_count": summary_row[1],
            "date_count": summary_row[2],
            "min_date": str(summary_row[3]),
            "max_date": str(summary_row[4]),
            "null_axis_counts": list(summary_row[5:8]),
            "axis_length_ranges": {
                "x": [summary_row[8], summary_row[9]],
                "y": [summary_row[10], summary_row[11]],
                "z": [summary_row[12], summary_row[13]],
            },
            "coverage_violations": coverage_violations,
            "duplicate_enterprise_date_pairs": duplicate_pairs,
            "health_index_bound_violations": bounds_violations,
            "trajectory_violations": len(bad_trajectories),
            "trajectory_violation_examples": bad_trajectories[:10],
            "passed": False,
        }
        report["passed"] = all(
            [
                report["window_count"] == 13_585,
                report["enterprise_count"] == 715,
                report["date_count"] == 19,
                report["min_date"] == "2024-12-25",
                report["max_date"] == "2025-01-12",
                report["null_axis_counts"] == [0, 0, 0],
                report["axis_length_ranges"] == {"x": [500, 500], "y": [500, 500], "z": [500, 500]},
                coverage_violations == 0,
                duplicate_pairs == 0,
                bounds_violations == 0,
                len(bad_trajectories) == 0,
            ]
        )
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["passed"]:
            raise SystemExit(1)
    finally:
        con.close()


if __name__ == "__main__":
    main()
