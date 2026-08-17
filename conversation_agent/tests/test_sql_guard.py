from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from conversation_agent.sql_guard import ReadOnlyQueryExecutor, ReadOnlySQLRejected, validate_readonly_sql


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM telemetry.scada_observation",
        "SELECT user_id FROM telemetry.scada_observation; DROP TABLE asset.user_meter",
        "SELECT * FROM read_csv_auto('C:/secret.csv')",
        "ATTACH 'other.db' AS other",
        "SELECT user_id FROM unknown.table_name",
        "SELECT source_file FROM telemetry.scada_observation",
    ],
)
def test_business_sql_rejects_writes_external_access_and_unknown_tables(sql: str) -> None:
    with pytest.raises(ReadOnlySQLRejected):
        validate_readonly_sql("business", sql)


def test_security_requires_explicit_safe_columns_but_allows_count_star() -> None:
    with pytest.raises(ReadOnlySQLRejected):
        validate_readonly_sql("security", "SELECT * FROM security_events")
    with pytest.raises(ReadOnlySQLRejected):
        validate_readonly_sql("security", "SELECT payload_json FROM alert_records")
    assert validate_readonly_sql("security", "SELECT COUNT(*) AS total FROM security_events")


def test_executor_returns_capped_traceable_query_artifact() -> None:
    executor = ReadOnlyQueryExecutor(ROOT)
    result = executor.execute(
        "business",
        "SELECT MIN(data_date) AS start_date, MAX(data_date) AS end_date FROM telemetry.scada_observation",
    )
    assert result.query_id.startswith("qry_")
    assert result.source == "business"
    assert result.rows and result.rows[0]["start_date"] <= result.rows[0]["end_date"]
    assert result.artifact()["type"] == "query_result"


def test_latest_diagnosis_query_uses_aggregate_join(tmp_path: Path) -> None:
    """同一用户同一天存在多次诊断时，标准 JOIN 模板只返回最新记录。"""

    database_dir = tmp_path / "database"
    database_dir.mkdir()
    result_db = database_dir / "gas_ai_results.duckdb"
    with duckdb.connect(str(result_db)) as connection:
        connection.execute("CREATE SCHEMA metering")
        connection.execute(
            """
            CREATE TABLE metering.diagnosis_run (
                run_id VARCHAR,
                user_id VARCHAR,
                diagnosis_date DATE,
                user_name VARCHAR,
                quality_status INTEGER,
                risk_level VARCHAR,
                meter_spec_result VARCHAR,
                summary VARCHAR,
                created_at TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            INSERT INTO metering.diagnosis_run VALUES
              ('old', 'u1', DATE '2025-01-12', '测试企业', 0, '低', '待复核', '旧结论', TIMESTAMP '2025-01-12 08:00:00'),
              ('new', 'u1', DATE '2025-01-12', '测试企业', 1, '高', '适配', '最新结论', TIMESTAMP '2025-01-12 09:00:00'),
              ('other', 'u2', DATE '2025-01-12', '另一企业', 1, '中', '偏大', '唯一结论', TIMESTAMP '2025-01-12 08:30:00')
            """
        )

    result = ReadOnlyQueryExecutor(tmp_path).execute(
        "diagnosis",
        """
        WITH latest AS (
          SELECT user_id, diagnosis_date, MAX(created_at) AS max_created_at
          FROM metering.diagnosis_run
          GROUP BY user_id, diagnosis_date
        )
        SELECT d.run_id, d.user_id, d.diagnosis_date, d.summary, d.created_at
        FROM metering.diagnosis_run AS d
        JOIN latest AS l
          ON l.user_id = d.user_id
         AND l.diagnosis_date = d.diagnosis_date
         AND l.max_created_at = d.created_at
        ORDER BY d.user_id
        """,
    )

    assert [row["run_id"] for row in result.rows] == ["new", "other"]
    assert result.rows[0]["summary"] == "最新结论"


def test_five_minute_sql_matches_metering_algorithm_for_one_user_day() -> None:
    from smart_metering import SmartMeteringService

    service = SmartMeteringService(use_deep_model=False)
    with service.repo.connect() as connection:
        user_id, diagnosis_date = connection.execute(
            "SELECT user_id,data_date FROM telemetry.scada_observation ORDER BY data_date DESC,user_id LIMIT 1"
        ).fetchone()
    day_long = service.repo.get_day_long(str(user_id), diagnosis_date)
    expected = float(service._resample_total_flow(day_long).fillna(0).sum() * 5.0 / 60.0)
    executor = ReadOnlyQueryExecutor(ROOT)
    result = executor.execute(
        "business",
        f"""
        WITH site_timestamp AS (
          SELECT entity_name,observed_at,
                 SUM(CASE WHEN standard_instant >= 0 THEN standard_instant END) AS site_flow
          FROM telemetry.scada_observation
          WHERE user_id='{user_id}' AND data_date=DATE '{diagnosis_date}'
          GROUP BY entity_name,observed_at
        ), five_minute AS (
          SELECT entity_name,time_bucket(INTERVAL '5 minutes', observed_at) AS bucket,
                 AVG(site_flow) AS flow_5m
          FROM site_timestamp
          GROUP BY entity_name,bucket
        )
        SELECT SUM(COALESCE(GREATEST(flow_5m,0),0)*5.0/60.0) AS volume_m3
        FROM five_minute
        """,
    )
    actual = float(result.rows[0]["volume_m3"])
    assert actual == pytest.approx(expected, rel=1e-9, abs=1e-6)
