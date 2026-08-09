from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import duckdb
from sqlglot import exp, parse


LOGGER = logging.getLogger("conversation_agent.sql")
DatabaseSource = Literal["business", "diagnosis", "security"]

MAX_ROWS = 500
MAX_RESULT_BYTES = 1_000_000
QUERY_TIMEOUT_SECONDS = 8.0

ALLOWED_TABLES: dict[DatabaseSource, set[str]] = {
    "business": {
        "asset.user_meter",
        "equipment.vibration_sensor",
        "inspection.meter_check_point",
        "inspection.meter_check_record",
        "inspection.meter_repair",
        "telemetry.import_file_log",
        "telemetry.scada_observation",
        "vibration.acceleration_window",
        "vibration.build_manifest",
        "vibration.daily_health",
        "vibration.health_label_dictionary",
        "vibration.trajectory_dictionary",
    },
    "diagnosis": {
        "equipment.health_diagnosis",
        "equipment.health_trend",
        "metering.anomaly_interval",
        "metering.diagnosis_run",
        "metering.work_order",
        "operations.work_order",
        "operations.work_order_audit",
    },
    "security": {
        "alert_records",
        "cameras",
        "event_handling_actions",
        "event_people",
        "event_transitions",
        "llm_reviews",
        "security_events",
        "zones",
    },
}

# 文件路径、完整提示词与原始模型响应不应通过自然语言查询泄露。
DENIED_COLUMNS: dict[DatabaseSource, set[str]] = {
    "business": {"source_file"},
    "diagnosis": set(),
    "security": {
        "source",
        "source_path",
        "output_dir",
        "file_path",
        "config_json",
        "prompt_text",
        "input_json",
        "response_json",
        "payload_json",
    },
}

FORBIDDEN_NODE_NAMES = {
    "Alter",
    "Attach",
    "Command",
    "Commit",
    "Copy",
    "Create",
    "Delete",
    "Detach",
    "Drop",
    "Execute",
    "Grant",
    "Insert",
    "LoadData",
    "Merge",
    "Pragma",
    "Rollback",
    "Set",
    "Transaction",
    "TruncateTable",
    "Update",
    "Use",
}

FORBIDDEN_FUNCTIONS = {
    "read_csv",
    "read_csv_auto",
    "read_json",
    "read_json_auto",
    "read_ndjson",
    "read_parquet",
    "sqlite_scan",
    "postgres_scan",
    "mysql_scan",
    "httpfs",
    "glob",
}


class ReadOnlySQLRejected(ValueError):
    """表示 SQL 未通过只读安全策略。"""


@dataclass(frozen=True)
class QueryResult:
    query_id: str
    source: DatabaseSource
    sql: str
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool
    elapsed_ms: int

    def artifact(self) -> dict[str, Any]:
        return {
            "type": "query_result",
            "query_id": self.query_id,
            "source": self.source,
            "sql": self.sql,
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "elapsed_ms": self.elapsed_ms,
        }


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{len(value)} bytes>"
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_value(item) for key, item in value.items()}
    return str(value)


def validate_readonly_sql(source: DatabaseSource, sql: str) -> str:
    """使用 SQL AST 限制为单条、白名单表上的只读查询。"""

    text = sql.strip()
    if not text:
        raise ReadOnlySQLRejected("SQL 不能为空。")
    dialect = "sqlite" if source == "security" else "duckdb"
    try:
        statements = parse(text, read=dialect)
    except Exception as exc:  # sqlglot 会提供具体语法位置，接口层只返回简洁错误。
        raise ReadOnlySQLRejected(f"SQL 语法无法解析：{exc}") from exc
    if len(statements) != 1:
        raise ReadOnlySQLRejected("只允许执行一条 SQL 语句。")
    statement = statements[0]
    if not isinstance(statement, exp.Query):
        raise ReadOnlySQLRejected("只允许 SELECT 或 WITH ... SELECT 查询。")

    cte_names = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}
    for node in statement.walk():
        if node.__class__.__name__ in FORBIDDEN_NODE_NAMES:
            raise ReadOnlySQLRejected(f"禁止使用 {node.__class__.__name__} 语句。")

    for table in statement.find_all(exp.Table):
        name = table.name.lower()
        if name in cte_names:
            continue
        database = table.db.lower() if table.db else ""
        qualified = f"{database}.{name}" if database else name
        if source != "security" and not database:
            raise ReadOnlySQLRejected(f"DuckDB 表必须带 schema：{name}。")
        if qualified not in ALLOWED_TABLES[source]:
            raise ReadOnlySQLRejected(f"数据源 {source} 不允许访问表 {qualified}。")

    for function in statement.find_all(exp.Func):
        function_name = function.sql_name().lower()
        if function_name in FORBIDDEN_FUNCTIONS:
            raise ReadOnlySQLRejected(f"禁止调用外部访问函数 {function_name}。")

    denied_columns = DENIED_COLUMNS[source]
    for column in statement.find_all(exp.Column):
        if column.name.lower() in denied_columns:
            raise ReadOnlySQLRejected(f"字段 {column.name} 不允许通过对话查询。")
    if source == "security":
        for star in statement.find_all(exp.Star):
            # COUNT(*) 是聚合语义，不会泄露字段；裸星号和 table.* 均拒绝。
            if isinstance(star.parent, (exp.Select, exp.Column)):
                raise ReadOnlySQLRejected("安防查询必须明确列名，不能使用 SELECT *。")

    return statement.sql(dialect=dialect)


class ReadOnlyQueryExecutor:
    def __init__(self, root: Path):
        self.paths: dict[DatabaseSource, Path] = {
            "business": root / "database" / "gas_ai_input.duckdb",
            "diagnosis": root / "database" / "gas_ai_results.duckdb",
            "security": root / "safety_operations" / "data" / "security.db",
        }

    def describe(self, source: DatabaseSource, table: str | None = None) -> dict[str, Any]:
        path = self.paths[source]
        if not path.exists():
            raise FileNotFoundError(f"数据库不存在：{path.name}")
        if source == "security":
            return self._describe_sqlite(path, table)
        return self._describe_duckdb(path, source, table)

    def execute(self, source: DatabaseSource, sql: str) -> QueryResult:
        normalized = validate_readonly_sql(source, sql)
        path = self.paths[source]
        if not path.exists():
            raise FileNotFoundError(f"数据库不存在：{path.name}")
        started = time.perf_counter()
        if source == "security":
            columns, raw_rows = self._query_sqlite(path, normalized)
        else:
            columns, raw_rows = self._query_duckdb(path, normalized)
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        rows: list[dict[str, Any]] = []
        truncated = len(raw_rows) > MAX_ROWS
        for raw in raw_rows[:MAX_ROWS]:
            row = {columns[index]: _safe_value(value) for index, value in enumerate(raw)}
            candidate = [*rows, row]
            if len(json.dumps(candidate, ensure_ascii=False, default=str).encode("utf-8")) > MAX_RESULT_BYTES:
                truncated = True
                break
            rows.append(row)
        result = QueryResult(
            query_id=f"qry_{uuid.uuid4().hex[:16]}",
            source=source,
            sql=normalized,
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            elapsed_ms=elapsed_ms,
        )
        LOGGER.info(
            "readonly_sql source=%s query_id=%s elapsed_ms=%s rows=%s truncated=%s sql=%s",
            source,
            result.query_id,
            elapsed_ms,
            len(rows),
            truncated,
            normalized,
        )
        return result

    @staticmethod
    def _query_duckdb(path: Path, sql: str) -> tuple[list[str], list[tuple[Any, ...]]]:
        with duckdb.connect(str(path), read_only=True) as connection:
            timer = threading.Timer(QUERY_TIMEOUT_SECONDS, connection.interrupt)
            timer.daemon = True
            timer.start()
            try:
                cursor = connection.execute(f"SELECT * FROM ({sql}) AS agent_query LIMIT {MAX_ROWS + 1}")
                columns = [item[0] for item in cursor.description]
                return columns, cursor.fetchall()
            finally:
                timer.cancel()

    @staticmethod
    def _query_sqlite(path: Path, sql: str) -> tuple[list[str], list[tuple[Any, ...]]]:
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=QUERY_TIMEOUT_SECONDS)
        deadline = time.monotonic() + QUERY_TIMEOUT_SECONDS
        connection.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        try:
            connection.execute("PRAGMA query_only=ON")
            cursor = connection.execute(f"SELECT * FROM ({sql}) AS agent_query LIMIT {MAX_ROWS + 1}")
            columns = [item[0] for item in cursor.description or []]
            return columns, cursor.fetchall()
        finally:
            connection.close()

    @staticmethod
    def _describe_duckdb(path: Path, source: DatabaseSource, table: str | None) -> dict[str, Any]:
        allowed = ALLOWED_TABLES[source]
        selected = sorted(allowed if not table else {table.lower()})
        if any(item not in allowed for item in selected):
            raise ReadOnlySQLRejected("只能查看当前数据源白名单内的表。")
        with duckdb.connect(str(path), read_only=True) as connection:
            output = []
            for qualified in selected:
                schema_name, table_name = qualified.split(".", 1)
                columns = connection.execute(
                    """
                    SELECT column_name,data_type
                    FROM information_schema.columns
                    WHERE table_schema=? AND table_name=?
                    ORDER BY ordinal_position
                    """,
                    [schema_name, table_name],
                ).fetchall()
                output.append({"table": qualified, "columns": [{"name": c, "type": t} for c, t in columns]})
        return {"source": source, "tables": output}

    @staticmethod
    def _describe_sqlite(path: Path, table: str | None) -> dict[str, Any]:
        allowed = ALLOWED_TABLES["security"]
        selected = sorted(allowed if not table else {table.lower()})
        if any(item not in allowed for item in selected):
            raise ReadOnlySQLRejected("只能查看安防白名单内的表。")
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        try:
            output = []
            for table_name in selected:
                columns = connection.execute(f'PRAGMA table_info("{table_name}")').fetchall()
                safe_columns = [
                    {"name": row[1], "type": row[2]}
                    for row in columns
                    if str(row[1]).lower() not in DENIED_COLUMNS["security"]
                ]
                output.append({"table": table_name, "columns": safe_columns})
        finally:
            connection.close()
        return {"source": "security", "tables": output}
