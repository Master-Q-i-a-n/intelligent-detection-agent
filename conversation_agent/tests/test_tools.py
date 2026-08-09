from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

from conversation_agent.tools import SQL_QUERY_ERROR_CODE, build_agent_tools, build_query_error_middleware


class _ScriptedToolModel(FakeMessagesListChatModel):
    """按预设消息驱动真实 Agent 工具循环，避免测试依赖外部 LLM。"""

    def bind_tools(self, _tools: Any, **_kwargs: Any):
        return self


def _create_diagnosis_database(root: Path) -> None:
    (root / "database").mkdir()
    result_db = root / "database" / "gas_ai_results.duckdb"
    with duckdb.connect(str(result_db)) as connection:
        connection.execute("CREATE SCHEMA metering")
        connection.execute(
            """
            CREATE TABLE metering.diagnosis_run (
                run_id VARCHAR,
                user_id VARCHAR,
                diagnosis_date DATE,
                summary VARCHAR,
                created_at TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            INSERT INTO metering.diagnosis_run VALUES
              ('old', 'u1', DATE '2025-01-12', '旧结论', TIMESTAMP '2025-01-12 08:00:00'),
              ('new', 'u1', DATE '2025-01-12', '最新结论', TIMESTAMP '2025-01-12 09:00:00')
            """
        )


def test_work_order_tool_is_idempotent(tmp_path) -> None:
    (tmp_path / "database").mkdir()
    result_db = tmp_path / "database" / "gas_ai_results.duckdb"
    duckdb.connect(str(result_db)).close()
    tools = {item.name: item for item in build_agent_tools(tmp_path)}
    arguments = {
        "source_module": "equipment",
        "priority": "P2",
        "title": "复测设备振动",
        "description": "健康指数持续下降，需要现场复测。",
        "checklist": ["检查传感器安装", "复测三轴振动"],
        "source_reference": {"diagnosis_date": "2025-01-12", "diagnosis_id": "d1"},
        "user_id": "u1",
    }
    first = tools["create_work_order"].invoke(arguments)
    second = tools["create_work_order"].invoke(arguments)
    # 直接 invoke 时 LangChain 只返回模型可见 content；在 Agent 工具调用中才包装 artifact。
    first_payload = json.loads(first)
    second_payload = json.loads(second)
    assert first_payload["duplicate"] is False
    assert second_payload["duplicate"] is True
    assert first_payload["work_order_id"] == second_payload["work_order_id"]
    with duckdb.connect(str(result_db), read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM operations.work_order").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM operations.work_order_audit").fetchone()[0] == 1


def test_agent_rewrites_failed_duckdb_sql_once_and_completes(tmp_path: Path) -> None:
    _create_diagnosis_database(tmp_path)
    tools = {item.name: item for item in build_agent_tools(tmp_path)}
    invalid_sql = """
        SELECT run_id, user_id, diagnosis_date, summary
        FROM metering.diagnosis_run
        WHERE (user_id, diagnosis_date, created_at) IN (
          SELECT user_id, diagnosis_date, MAX(created_at)
          FROM metering.diagnosis_run
          GROUP BY user_id, diagnosis_date
        )
    """
    corrected_sql = """
        WITH latest AS (
          SELECT user_id, diagnosis_date, MAX(created_at) AS max_created_at
          FROM metering.diagnosis_run
          GROUP BY user_id, diagnosis_date
        )
        SELECT d.run_id, d.user_id, d.diagnosis_date, d.summary
        FROM metering.diagnosis_run AS d
        JOIN latest AS l
          ON l.user_id = d.user_id
         AND l.diagnosis_date = d.diagnosis_date
         AND l.max_created_at = d.created_at
    """
    model = _ScriptedToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "invalid-query",
                        "name": "query_diagnosis_data",
                        "args": {"sql": invalid_sql},
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "corrected-query",
                        "name": "query_diagnosis_data",
                        "args": {"sql": corrected_sql},
                    }
                ],
            ),
            AIMessage(content="已使用 JOIN 改写并取得最新诊断。"),
        ]
    )
    agent = create_agent(
        model=model,
        tools=[tools["query_diagnosis_data"]],
        middleware=[build_query_error_middleware()],
    )

    output = agent.invoke({"messages": [{"role": "user", "content": "查询最新诊断"}]})
    tool_messages = [message for message in output["messages"] if isinstance(message, ToolMessage)]

    assert len(tool_messages) == 2
    first_error = json.loads(str(tool_messages[0].content))
    assert tool_messages[0].status == "error"
    assert first_error["error_code"] == SQL_QUERY_ERROR_CODE
    assert first_error["category"] == "binder"
    assert first_error["retry_allowed"] is True
    assert invalid_sql.strip() not in str(tool_messages[0].content)
    assert tool_messages[1].status == "success"
    assert tool_messages[1].artifact["rows"][0]["run_id"] == "new"
    assert output["messages"][-1].content == "已使用 JOIN 改写并取得最新诊断。"


def test_second_duckdb_error_disables_further_sql_rewrite(tmp_path: Path) -> None:
    _create_diagnosis_database(tmp_path)
    tools = {item.name: item for item in build_agent_tools(tmp_path)}
    invalid_sql = """
        SELECT run_id FROM metering.diagnosis_run
        WHERE (user_id, diagnosis_date, created_at) IN (
          SELECT user_id, diagnosis_date, MAX(created_at)
          FROM metering.diagnosis_run
          GROUP BY user_id, diagnosis_date
        )
    """
    model = _ScriptedToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"id": "first-error", "name": "query_diagnosis_data", "args": {"sql": invalid_sql}}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"id": "second-error", "name": "query_diagnosis_data", "args": {"sql": invalid_sql}}],
            ),
            AIMessage(content="SQL 修正仍失败，本轮未生成查询结果。"),
        ]
    )
    agent = create_agent(
        model=model,
        tools=[tools["query_diagnosis_data"]],
        middleware=[build_query_error_middleware()],
    )

    output = agent.invoke({"messages": [{"role": "user", "content": "查询最新诊断"}]})
    errors = [
        json.loads(str(message.content))
        for message in output["messages"]
        if isinstance(message, ToolMessage) and message.status == "error"
    ]

    assert [item["retry_allowed"] for item in errors] == [True, False]
    assert "未生成查询结果" in str(output["messages"][-1].content)
