from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import duckdb
import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

from intelligent_detection_agent.conversation_agent import tools as tools_module
from intelligent_detection_agent.conversation_agent.tools import (
    SQL_QUERY_ERROR_CODE,
    build_agent_tools,
    build_query_error_middleware,
)


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


def test_work_order_uses_user_id_from_source_reference(tmp_path) -> None:
    (tmp_path / "database").mkdir()
    result_db = tmp_path / "database" / "gas_ai_results.duckdb"
    duckdb.connect(str(result_db)).close()
    tools = {item.name: item for item in build_agent_tools(tmp_path)}

    payload = json.loads(tools["create_work_order"].invoke({
        "source_module": "metering",
        "priority": "P3",
        "title": "流量异常核查",
        "description": "依据遥测证据开展现场核查。",
        "checklist": ["核对现场表计"],
        "source_reference": {"user_id": "1072548130", "diagnosis_date": "2025-01-12"},
    }))

    assert payload["user_id"] == "1072548130"
    with duckdb.connect(str(result_db), read_only=True) as connection:
        assert connection.execute("SELECT user_id FROM operations.work_order").fetchone()[0] == "1072548130"


def test_rag_tool_returns_passages_and_safe_image_artifact(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "database").mkdir()
    duckdb.connect(str(tmp_path / "database" / "gas_ai_results.duckdb")).close()
    image = tmp_path / "dataset" / "doc" / "压力传感器" / "images" / "fig_001" / "fig_001.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")

    class _FakeRagPipeline:
        def search(self, query: str, *, hybrid_limit: int, top_n: int):
            assert query == "压力传感器如何安装？"
            assert (hybrid_limit, top_n) == (20, 5)
            return {
                "original_query": query,
                "collection": "technical_docs",
                "results": [{
                    "point_id": "point-1",
                    "rrf_score": 0.8,
                    "rerank_score": 0.95,
                    "dense": {"rank": 1, "score": 0.9},
                    "bm25": {"rank": 2, "score": 0.7},
                    "payload": {
                        "chunk_id": "chunk-1",
                        "source": "压力传感器.pdf",
                        "text": "安装时应避免脉动和过热。",
                        "page_numbers": [12],
                        "headings": ["安装要求"],
                        "images": [{
                            "image_path": "images/fig_001/fig_001.png",
                            "page_no": 12,
                            "image_type": "diagram",
                            "description": "压力变送器安装示意图",
                        }],
                    },
                }],
            }

    monkeypatch.setattr(tools_module, "RagPipeline", lambda _config, **_kwargs: _FakeRagPipeline())
    tool = {item.name: item for item in build_agent_tools(tmp_path)}["search_technical_documents"]
    message = tool.invoke({
        "name": "search_technical_documents",
        "args": {"query": "压力传感器如何安装？"},
        "id": "rag-call-1",
        "type": "tool_call",
    })

    content = json.loads(str(message.content))
    assert content["passages"][0]["reference"] == "资料1"
    assert content["passages"][0]["text"] == "安装时应避免脉动和过热。"
    assert content["passages"][0]["images"][0]["markdown"].startswith("![压力变送器安装示意图](/chat/rag-assets/")
    assert message.artifact["type"] == "rag_retrieval"
    stored_image = message.artifact["results"][0]["images"][0]
    assert stored_image["image_path"] == "images/fig_001/fig_001.png"
    assert stored_image["url"].endswith("/images/fig_001/fig_001.png")
    assert str(tmp_path) not in str(message.content)


def test_rag_tool_hides_internal_failure_details(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "database").mkdir()
    duckdb.connect(str(tmp_path / "database" / "gas_ai_results.duckdb")).close()

    class _UnavailableRagPipeline:
        def search(self, *_args, **_kwargs):
            raise RuntimeError("secret-key and D:/private/qdrant")

    monkeypatch.setattr(tools_module, "RagPipeline", lambda _config, **_kwargs: _UnavailableRagPipeline())
    tool = {item.name: item for item in build_agent_tools(tmp_path)}["search_technical_documents"]
    message = tool.invoke({
        "name": "search_technical_documents",
        "args": {"query": "测试"},
        "id": "rag-call-error",
        "type": "tool_call",
    })

    assert json.loads(str(message.content))["status"] == "unavailable"
    assert "secret-key" not in str(message.content)
    assert message.artifact["status"] == "unavailable"


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


@pytest.mark.parametrize(("invalid_sql", "category"), [
    ("SELECT * FROM security_events", "readonly_policy"),
    ("SELECT organization FROM security_events", "sqlite_schema"),
    ("SELECT event_id FROM event_people", "sqlite_schema"),
])
def test_agent_rewrites_rejected_security_select_star_once(tmp_path: Path, invalid_sql: str, category: str) -> None:
    (tmp_path / "database").mkdir()
    duckdb.connect(str(tmp_path / "database" / "gas_ai_results.duckdb")).close()
    security_path = tmp_path / "safety_operations" / "data" / "security.db"
    security_path.parent.mkdir(parents=True)
    with sqlite3.connect(security_path) as connection:
        connection.execute(
            "CREATE TABLE security_events (event_id TEXT, final_decision TEXT, handling_status TEXT)"
        )
        connection.execute("INSERT INTO security_events VALUES ('event-1', 'CONFIRMED', 'NEW')")

    tools = {item.name: item for item in build_agent_tools(tmp_path)}
    model = _ScriptedToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{
                    "id": "unsafe-security-query",
                    "name": "query_security_data",
                    "args": {"sql": invalid_sql},
                }],
            ),
            AIMessage(
                content="",
                tool_calls=[{
                    "id": "safe-security-query",
                    "name": "query_security_data",
                    "args": {
                        "sql": "SELECT event_id, final_decision, handling_status FROM security_events"
                    },
                }],
            ),
            AIMessage(content="已按明确字段完成安防查询。"),
        ]
    )
    agent = create_agent(
        model=model,
        tools=[tools["query_security_data"]],
        middleware=[build_query_error_middleware()],
    )

    output = agent.invoke({"messages": [{"role": "user", "content": "查询安防事件"}]})
    tool_messages = [message for message in output["messages"] if isinstance(message, ToolMessage)]

    assert len(tool_messages) == 2
    assert json.loads(str(tool_messages[0].content))["category"] == category
    assert tool_messages[0].status == "error"
    assert tool_messages[1].status == "success"
    assert tool_messages[1].artifact["rows"][0]["event_id"] == "event-1"


@pytest.mark.parametrize("error", [
    "database is locked", "unable to open database file", "disk I/O error", "unknown internal failure",
])
def test_sqlite_infrastructure_errors_are_not_rewrite_feedback(error: str) -> None:
    from types import SimpleNamespace

    assert tools_module._safe_query_error(
        sqlite3.OperationalError(error), SimpleNamespace(state={}),
    ) is None


def test_sqlite_syntax_feedback_hides_details_and_limits_rewrite() -> None:
    from types import SimpleNamespace

    # 用实际 SQLite 异常验证分类；原始报错文本不能进入模型反馈。
    with sqlite3.connect(":memory:") as connection:
        with pytest.raises(sqlite3.OperationalError) as caught:
            connection.execute("SELECT FROM private_table")
    first = tools_module._safe_query_error(caught.value, SimpleNamespace(state={}))
    assert json.loads(first)["category"] == "sqlite_syntax"
    assert json.loads(first)["retry_allowed"] is True
    assert "private_table" not in first
    previous = ToolMessage(content=first, tool_call_id="failed-sql", status="error")
    second = tools_module._safe_query_error(caught.value, SimpleNamespace(state={"messages": [previous]}))
    assert json.loads(second)["retry_allowed"] is False


def test_denied_field_feedback_identifies_correction_without_leaking_sql() -> None:
    from types import SimpleNamespace
    from intelligent_detection_agent.conversation_agent.sql_guard import ReadOnlySQLRejected, validate_readonly_sql

    sql = "SELECT user_id, Source_File FROM asset.user_meter WHERE user_id='private-user'"
    with pytest.raises(ReadOnlySQLRejected) as caught:
        validate_readonly_sql("business", sql)
    feedback = tools_module._safe_query_error(caught.value, SimpleNamespace(state={}))
    first = json.loads(feedback)
    assert first["category"] == "readonly_policy"
    assert first["retry_allowed"] is True
    assert "source_file" in first["message"]
    assert "移除" in first["message"]
    assert "describe_data_source" in first["message"]
    assert "private-user" not in feedback
    assert sql not in feedback
    previous = ToolMessage(content=feedback, tool_call_id="denied-field", status="error")
    second = json.loads(tools_module._safe_query_error(caught.value, SimpleNamespace(state={"messages": [previous]})))
    assert second["retry_allowed"] is False
    assert "不要再次调用" in second["instruction"]

    # 普通拒绝仍不把可能包含路径、SQL 片段的原始异常直接暴露给模型。
    generic = tools_module._safe_query_error(ReadOnlySQLRejected("private-secret-path"), SimpleNamespace(state={}))
    assert "private-secret-path" not in generic


def test_security_refusal_allows_readonly_checks_but_rejects_write_tools() -> None:
    from intelligent_detection_agent.evaluation.runner import load_agent_cases
    from intelligent_detection_agent.evaluation.grading import grade_tool_policy

    case = next(case for case in load_agent_cases() if case.id == "boundary_02_security_write")
    readonly = [{"name": "describe_data_source"}, {"name": "query_security_data"}]
    assert grade_tool_policy(case, readonly)["passed"]
    assert not grade_tool_policy(case, readonly + [{"name": "create_work_order"}])["passed"]


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
