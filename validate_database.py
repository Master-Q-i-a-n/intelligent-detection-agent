from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime_libs"))

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402


DB_PATH = ROOT / "database" / "gas_ai_input.duckdb"
REPORT_DIR = ROOT / "reports"


def scalar(con, sql):
    return con.execute(sql).fetchone()[0]


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        date_row = con.execute(
            """
            SELECT MIN(data_date), MAX(data_date), COUNT(DISTINCT data_date),
                   COUNT(DISTINCT user_id), COUNT(DISTINCT entity_key), COUNT(*)
            FROM telemetry.scada_observation
            """
        ).fetchone()
        summary = {
            "database": str(DB_PATH),
            "input_libraries": 3,
            "asset_user_meter_rows": scalar(con, "SELECT COUNT(*) FROM asset.user_meter"),
            "inspection_record_rows": scalar(con, "SELECT COUNT(*) FROM inspection.meter_check_record"),
            "inspection_point_rows": scalar(con, "SELECT COUNT(*) FROM inspection.meter_check_point"),
            "repair_record_rows": scalar(con, "SELECT COUNT(*) FROM inspection.meter_repair"),
            "scada_min_date": str(date_row[0]),
            "scada_max_date": str(date_row[1]),
            "scada_date_count": date_row[2],
            "scada_distinct_user_ids": date_row[3],
            "scada_distinct_entities": date_row[4],
            "scada_observation_rows": date_row[5],
            "source_files_success": scalar(
                con, "SELECT COUNT(*) FROM telemetry.import_file_log WHERE status='success'"
            ),
            "source_files_failed": scalar(
                con, "SELECT COUNT(*) FROM telemetry.import_file_log WHERE status='failed'"
            ),
            "linked_users_all_three": scalar(
                con,
                """
                SELECT COUNT(DISTINCT o.user_id)
                FROM telemetry.scada_observation o
                JOIN asset.user_meter u ON u.user_id=o.user_id
                JOIN inspection.meter_check_record c ON c.user_id=o.user_id
                WHERE o.user_id IS NOT NULL
                """,
            ),
            "asset_users_with_valid_range": scalar(
                con,
                "SELECT COUNT(*) FROM asset.user_meter WHERE quantity_min IS NOT NULL AND quantity_max IS NOT NULL",
            ),
            "asset_negative_range_rows": scalar(
                con,
                "SELECT COUNT(*) FROM asset.user_meter WHERE quantity_min < 0 OR quantity_max < 0",
            ),
        }
        failed = con.execute(
            """
            SELECT source_file, observation_date, entity_key, user_id, error_message
            FROM telemetry.import_file_log
            WHERE status='failed'
            ORDER BY observation_date, source_file
            """
        ).df()
        failed.to_csv(REPORT_DIR / "failed_scada_files.csv", index=False, encoding="utf-8-sig")
        (REPORT_DIR / "build_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        assert summary["asset_negative_range_rows"] == 0, "量程解析出现负数"
        assert summary["source_files_success"] + summary["source_files_failed"] == 14649
        assert summary["scada_date_count"] == 19
        assert summary["scada_observation_rows"] > 20_000_000
        sample = con.execute(
            """
            SELECT o.user_id, u.meter_model, u.range_text,
                   MIN(o.data_date) AS first_date, MAX(o.data_date) AS last_date,
                   COUNT(*) AS observation_rows
            FROM telemetry.scada_observation o
            JOIN asset.user_meter u ON u.user_id=o.user_id
            JOIN inspection.meter_check_record c ON c.user_id=o.user_id
            GROUP BY o.user_id, u.meter_model, u.range_text
            ORDER BY observation_rows DESC
            LIMIT 3
            """
        ).df()
        print("\nLinked sample queries:")
        print(sample.to_string(index=False))
        if not failed.empty:
            print("\nFailure reasons:")
            print(failed.groupby("error_message").size().sort_values(ascending=False).head(10).to_string())
    finally:
        con.close()


if __name__ == "__main__":
    main()
