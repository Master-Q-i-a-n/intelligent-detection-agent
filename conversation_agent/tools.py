from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import duckdb
from langchain.agents.middleware import ToolCallRequest, ToolErrorMiddleware
from langchain_core.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import interrupt

from .sql_guard import DatabaseSource, ReadOnlyQueryExecutor, ReadOnlySQLRejected


WORK_ORDER_LOCK = threading.RLock()
QUERY_TOOL_NAMES = ["query_business_data", "query_diagnosis_data"]
SQL_QUERY_ERROR_CODE = "SQL_QUERY_ERROR"


def _current_turn_query_error_count(request: ToolCallRequest) -> int:
    """统计当前用户轮次已回传的 SQL 错误，用于只允许一次改写。"""

    state = request.state if isinstance(request.state, dict) else {}
    count = 0
    for message in reversed(state.get("messages", [])):
        if getattr(message, "type", None) == "human":
            break
        if not isinstance(message, ToolMessage) or getattr(message, "status", None) != "error":
            continue
        if SQL_QUERY_ERROR_CODE in str(message.content):
            count += 1
    return count


def _safe_query_error(exc: Exception, request: ToolCallRequest) -> str | None:
    """将可修正的 SQL 异常转换为不包含 SQL、路径和堆栈的模型可见消息。"""

    if isinstance(exc, ReadOnlySQLRejected):
        category = "readonly_policy"
        guidance = "SQL 未通过只读或白名单校验。请仅使用单条 SELECT/WITH SELECT、完整 schema.table 和允许字段。"
    elif isinstance(exc, duckdb.BinderException):
        category = "binder"
        guidance = "DuckDB 无法绑定字段或表达式。请检查字段、别名、聚合和 JOIN；禁止使用多列元组 IN 子查询。"
    elif isinstance(exc, duckdb.ParserException):
        category = "syntax"
        guidance = "SQL 不符合当前 DuckDB 语法。请使用数据库 Skill 中已验证的 DuckDB 模板重写。"
    elif isinstance(exc, duckdb.CatalogException):
        category = "catalog"
        guidance = "表、字段或函数无法解析。请先核对已读取的 reference 或调用 describe_data_source。"
    elif isinstance(exc, duckdb.ConversionException):
        category = "conversion"
        guidance = "数据类型转换失败。请显式使用与字段类型相符的 CAST、DATE 或 TIMESTAMP 表达式。"
    elif isinstance(exc, duckdb.Error):
        category = "duckdb"
        guidance = "DuckDB 无法执行该查询。请依据字段结构和 SQL 示例改写，避免复用原 SQL。"
    else:
        # 文件缺失、编程错误等内部异常仍向上抛出，避免被伪装成可修正 SQL。
        return None

    retry_allowed = _current_turn_query_error_count(request) == 0
    instruction = (
        "允许修正一次：请重写整条 SQL 后重新调用同一查询工具，不要原样重试。"
        if retry_allowed
        else "本轮 SQL 修正机会已用完：不要再次调用查询工具，请说明查询未完成并给出可操作建议。"
    )
    return json.dumps(
        {
            "error_code": SQL_QUERY_ERROR_CODE,
            "category": category,
            "retry_allowed": retry_allowed,
            "message": guidance,
            "instruction": instruction,
        },
        ensure_ascii=False,
    )


def build_query_error_middleware() -> ToolErrorMiddleware:
    """仅处理只读 DuckDB 查询错误，让模型获得一次安全的 SQL 修正机会。"""

    return ToolErrorMiddleware(on_error=_safe_query_error, tools=QUERY_TOOL_NAMES)


def initialize_work_order_schema(result_database: Path) -> None:
    """在现有结果库中建立跨模块工单和审计表，不创建新的数据库文件。"""

    with WORK_ORDER_LOCK, duckdb.connect(str(result_database)) as connection:
        connection.execute("CREATE SCHEMA IF NOT EXISTS operations")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS operations.work_order (
                work_order_id VARCHAR PRIMARY KEY,
                idempotency_key VARCHAR UNIQUE NOT NULL,
                source_module VARCHAR NOT NULL,
                user_id VARCHAR,
                source_reference_json VARCHAR NOT NULL,
                priority VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                title VARCHAR NOT NULL,
                description VARCHAR NOT NULL,
                checklist_json VARCHAR NOT NULL,
                created_by VARCHAR NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS operations.work_order_audit (
                audit_id VARCHAR PRIMARY KEY,
                work_order_id VARCHAR NOT NULL,
                action VARCHAR NOT NULL,
                operator VARCHAR NOT NULL,
                details_json VARCHAR NOT NULL,
                acted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def _query_tool_result(executor: ReadOnlyQueryExecutor, source: DatabaseSource, sql: str) -> tuple[str, dict[str, Any]]:
    result = executor.execute(source, sql)
    preview = result.rows[:20]
    content = json.dumps(
        {
            "query_id": result.query_id,
            "source": source,
            "columns": result.columns,
            "returned_rows": result.row_count,
            "truncated": result.truncated,
            "preview": preview,
        },
        ensure_ascii=False,
        default=str,
    )
    return content, result.artifact()


def _query_artifacts(runtime: ToolRuntime) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    state = runtime.state if isinstance(runtime.state, dict) else {}
    for message in state.get("messages", []):
        artifact = getattr(message, "artifact", None)
        if isinstance(artifact, dict) and artifact.get("type") == "query_result" and artifact.get("query_id"):
            artifacts[str(artifact["query_id"])] = artifact
    return artifacts


def build_agent_tools(root: Path) -> list[Any]:
    executor = ReadOnlyQueryExecutor(root)
    result_database = root / "database" / "gas_ai_results.duckdb"
    initialize_work_order_schema(result_database)

    @tool
    def get_current_time(timezone: str = "Asia/Shanghai") -> dict[str, str]:
        """查询真实当前时间。遇到今天、昨天、上周、最近几天等相对日期时必须先调用。"""

        try:
            zone = ZoneInfo(timezone)
        except Exception as exc:
            raise ValueError("不支持该时区，请使用 Asia/Shanghai 或标准 IANA 时区。") from exc
        now = datetime.now(zone)
        return {
            "timezone": timezone,
            "datetime": now.isoformat(timespec="seconds"),
            "date": now.date().isoformat(),
            "weekday": now.strftime("%A"),
        }

    @tool
    def describe_data_source(
        source: Literal["business", "diagnosis", "security"],
        table: str | None = None,
    ) -> dict[str, Any]:
        """查看白名单数据源的实时表结构。字段不确定时先调用；table 可传 schema.table。"""

        return executor.describe(source, table)

    @tool(response_format="content_and_artifact")
    def query_business_data(sql: str) -> tuple[str, dict[str, Any]]:
        """只读查询业务输入 DuckDB，适合用气、表具、检定、维修和振动原始/日级数据。"""

        return _query_tool_result(executor, "business", sql)

    @tool(response_format="content_and_artifact")
    def query_diagnosis_data(sql: str) -> tuple[str, dict[str, Any]]:
        """只读查询诊断结果 DuckDB，适合计量结论、设备健康、异常区间和工单信息。"""

        return _query_tool_result(executor, "diagnosis", sql)

    @tool(response_format="content_and_artifact")
    def query_security_data(sql: str) -> tuple[str, dict[str, Any]]:
        """只读查询安防 SQLite。只能查询事件、复核、人员、通知和处置审计，不能改变事件状态。"""

        return _query_tool_result(executor, "security", sql)

    @tool
    def ask_user(
        question: str,
        missing_information: list[str],
        suggestions: list[str] | None = None,
    ) -> str:
        """仅当缺少的业务条件会实质改变查询、报告或工单结果时，暂停并向用户补充询问。"""

        answer = interrupt(
            {
                "kind": "clarification",
                "question": question,
                "missing_information": missing_information,
                "suggestions": suggestions or [],
            }
        )
        if isinstance(answer, dict):
            return str(answer.get("message") or answer.get("answer") or answer)
        return str(answer)

    @tool(response_format="content_and_artifact")
    def create_work_order(
        source_module: Literal["metering", "equipment", "safety"],
        priority: Literal["P1", "P2", "P3", "P4"],
        title: str,
        description: str,
        checklist: list[str],
        source_reference: dict[str, Any],
        user_id: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """创建跨模块业务工单。该工具执行前必须由用户审批；不会修改安防事件处置状态。"""

        title = title.strip()
        description = description.strip()
        checklist = [item.strip() for item in checklist if item.strip()]
        if not title or not description or not checklist:
            raise ValueError("工单标题、说明和至少一项检查清单不能为空。")
        normalized = json.dumps(
            {
                "source_module": source_module,
                "user_id": user_id,
                "source_reference": source_reference,
                "priority": priority,
                "title": title,
                "description": description,
                "checklist": checklist,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        idempotency_key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        work_order_id = f"CWO-{datetime.now():%Y%m%d}-{idempotency_key[:10]}"
        with WORK_ORDER_LOCK, duckdb.connect(str(result_database)) as connection:
            existing = connection.execute(
                "SELECT work_order_id,status,created_at FROM operations.work_order WHERE idempotency_key=?",
                [idempotency_key],
            ).fetchone()
            if existing:
                payload = {
                    "work_order_id": existing[0],
                    "status": existing[1],
                    "created_at": str(existing[2]),
                    "duplicate": True,
                }
                return json.dumps(payload, ensure_ascii=False), {"type": "work_order", **payload}
            connection.execute(
                """
                INSERT INTO operations.work_order
                (work_order_id,idempotency_key,source_module,user_id,source_reference_json,
                 priority,status,title,description,checklist_json,created_by)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    work_order_id,
                    idempotency_key,
                    source_module,
                    user_id,
                    json.dumps(source_reference, ensure_ascii=False, default=str),
                    priority,
                    "OPEN",
                    title,
                    description,
                    json.dumps(checklist, ensure_ascii=False),
                    "conversation-agent",
                ],
            )
            connection.execute(
                """
                INSERT INTO operations.work_order_audit
                (audit_id,work_order_id,action,operator,details_json)
                VALUES (?,?,?,?,?)
                """,
                [uuid.uuid4().hex, work_order_id, "CREATE", "conversation-agent", normalized],
            )
        payload = {
            "work_order_id": work_order_id,
            "source_module": source_module,
            "user_id": user_id,
            "priority": priority,
            "status": "OPEN",
            "title": title,
            "description": description,
            "checklist": checklist,
            "source_reference": source_reference,
            "duplicate": False,
        }
        return json.dumps(payload, ensure_ascii=False), {"type": "work_order", **payload}

    @tool(response_format="content_and_artifact")
    def build_report_artifact(
        title: str,
        summary: str,
        report_markdown: str,
        source_query_ids: list[str],
        charts: list[dict[str, Any]],
        recommendations: list[str],
        runtime: ToolRuntime,
    ) -> tuple[str, dict[str, Any]]:
        """用已执行 SQL 的查询编号生成带表格和图表的报告；不写服务器文件。"""

        available = _query_artifacts(runtime)
        requested = list(dict.fromkeys(source_query_ids))
        missing = [query_id for query_id in requested if query_id not in available]
        if missing:
            raise ValueError(f"报告引用了不存在的查询编号：{', '.join(missing)}")
        datasets = [available[query_id] for query_id in requested]
        normalized_charts = []
        for index, chart in enumerate(charts):
            query_id = str(chart.get("source_query_id", ""))
            if query_id not in available:
                raise ValueError(f"图表 {index + 1} 的 source_query_id 不存在。")
            chart_type = str(chart.get("type", ""))
            if chart_type not in {"line", "bar", "pie", "scatter"}:
                raise ValueError(f"图表 {index + 1} 类型不支持：{chart_type}")
            columns = set(available[query_id].get("columns", []))
            x_field = str(chart.get("x_field", ""))
            y_fields = [str(item) for item in chart.get("y_fields", [])]
            if not x_field or x_field not in columns or not y_fields or any(field not in columns for field in y_fields):
                raise ValueError(f"图表 {index + 1} 引用了查询结果中不存在的字段。")
            normalized_charts.append(
                {
                    "id": str(chart.get("id") or f"chart_{index + 1}"),
                    "type": chart_type,
                    "title": str(chart.get("title") or f"图表 {index + 1}"),
                    "unit": str(chart.get("unit") or ""),
                    "source_query_id": query_id,
                    "x_field": x_field,
                    "y_fields": y_fields,
                    "series_names": [str(item) for item in chart.get("series_names", [])],
                }
            )
        report_id = f"rpt_{uuid.uuid4().hex[:16]}"
        artifact = {
            "type": "report",
            "report_id": report_id,
            "title": title.strip(),
            "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
            "summary": summary.strip(),
            "report_markdown": report_markdown.strip(),
            "datasets": datasets,
            "charts": normalized_charts,
            "recommendations": [item.strip() for item in recommendations if item.strip()],
        }
        content = json.dumps(
            {
                "report_id": report_id,
                "title": artifact["title"],
                "dataset_count": len(datasets),
                "chart_count": len(normalized_charts),
                "message": "报告产物已生成，前端可直接渲染和下载。",
            },
            ensure_ascii=False,
        )
        return content, artifact

    return [
        get_current_time,
        describe_data_source,
        query_business_data,
        query_diagnosis_data,
        query_security_data,
        ask_user,
        create_work_order,
        build_report_artifact,
    ]
