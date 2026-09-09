from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

from intelligent_detection_agent.conversation_agent.agent import ThreadedSqliteSaver
from intelligent_detection_agent.user_store import UserStore


class _CheckpointState(TypedDict):
    value: int


def test_demo_user_is_idempotent_and_password_is_not_plaintext(tmp_path):
    database = tmp_path / "database" / "user_data.db"
    store = UserStore(database)
    UserStore(database)

    assert store.authenticate("admin", "123456") == store.authenticate("ADMIN", "123456")
    assert store.authenticate("admin", "wrong-password") is None
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT username,password_salt,password_hash FROM users WHERE username='admin'"
        ).fetchone()
        count = connection.execute("SELECT COUNT(*) FROM users WHERE username='admin'").fetchone()[0]
    assert count == 1
    assert row[0] == "admin"
    assert row[1] != "123456"
    assert row[2] != "123456"


def test_registration_session_rotation_expiry_and_logout(tmp_path):
    store = UserStore(tmp_path / "user_data.db")
    user = store.register("测试用户", "secure-pass")
    with pytest.raises(ValueError, match="用户名已存在"):
        store.register("测试用户", "another-pass")

    first = store.create_session(user["user_id"])
    second = store.create_session(user["user_id"])
    assert store.session_user(first) is None
    assert store.session_user(second)["username"] == "测试用户"

    with store.connect() as connection:
        connection.execute(
            "UPDATE auth_sessions SET expires_at=?",
            ((datetime.now(UTC) - timedelta(minutes=1)).isoformat(timespec="seconds"),),
        )
    assert store.session_user(second) is None

    third = store.create_session(user["user_id"])
    store.logout(third)
    assert store.session_user(third) is None


def test_chat_history_is_user_isolated_and_restores_artifacts(tmp_path):
    store = UserStore(tmp_path / "user_data.db")
    first = store.register("user_one", "123456")
    second = store.register("user_two", "123456")
    thread_id = "chat_public_1"

    store.start_turn(first["user_id"], thread_id, "查询昨天用气量")
    store.record_response(first["user_id"], thread_id, {
        "status": "completed",
        "message": "查询完成。",
        "generator": "test",
        "todos": [{"content": "查询用气", "status": "completed"}],
        "artifacts": [{
            "type": "query_result",
            "id": "qry_1",
            "payload": {"sql": "SELECT 1", "rows": [{"value": 1}]},
        }],
    })

    assert len(store.list_threads(first["user_id"])) == 1
    assert store.list_threads(second["user_id"]) == []
    assert store.thread_detail(second["user_id"], thread_id) is None
    assert not store.owns_thread(second["user_id"], thread_id)
    with pytest.raises(PermissionError):
        store.start_turn(second["user_id"], thread_id, "越权续写")

    detail = store.thread_detail(first["user_id"], thread_id)
    assert detail["messages"][-1]["artifact_ids"] == ["qry_1"]
    assert detail["artifacts"][0]["payload"]["rows"] == [{"value": 1}]
    assert detail["todos"][0]["status"] == "completed"

    assert store.delete_thread_records(first["user_id"], thread_id)
    assert store.thread_detail(first["user_id"], thread_id) is None


def test_failed_turn_keeps_question_but_allows_next_message(tmp_path):
    store = UserStore(tmp_path / "user_data.db")
    user = store.register("retry_user", "123456")
    store.start_turn(user["user_id"], "chat_retry", "第一次查询")
    store.record_error(user["user_id"], "chat_retry", "查询服务暂时不可用")

    detail = store.thread_detail(user["user_id"], "chat_retry")
    assert detail["messages"][0]["content"] == "第一次查询"
    assert detail["last_error"] == "查询服务暂时不可用"
    # 失败不是 HITL 中断，用户恢复历史后可以直接继续提问。
    store.start_turn(user["user_id"], "chat_retry", "第二次查询")


def test_chat_images_are_bound_to_message_restored_and_deleted_with_thread(tmp_path):
    store = UserStore(tmp_path / "user_data.db")
    user = store.register("image_user", "123456")
    attachment = store.create_attachment(
        user["user_id"],
        "chat_images",
        original_name="现场照片.png",
        mime_type="image/png",
        width=320,
        height=180,
        data=b"test-image-bytes",
    )
    image_path = store.attachment_root / attachment["id"]
    assert image_path.is_file()

    store.start_turn(user["user_id"], "chat_images", "", [attachment["id"]])
    assert store.attachment_model_inputs(user["user_id"], "chat_images", [attachment["id"]]) == [
        {"id": attachment["id"], "mime_type": "image/png"}
    ]
    detail = store.thread_detail(user["user_id"], "chat_images")
    assert detail["thread"]["title"] == "图片分析"
    assert detail["messages"][0]["content"] == ""
    assert detail["messages"][0]["attachments"][0]["name"] == "现场照片.png"
    assert not store.delete_pending_attachment(user["user_id"], attachment["id"])

    assert store.delete_thread_records(user["user_id"], "chat_images")
    assert not image_path.exists()


def test_chat_image_cannot_be_bound_by_another_user(tmp_path):
    store = UserStore(tmp_path / "user_data.db")
    owner = store.register("image_owner", "123456")
    other = store.register("image_other", "123456")
    attachment = store.create_attachment(
        owner["user_id"],
        "chat_shared",
        original_name="meter.webp",
        mime_type="image/webp",
        width=100,
        height=100,
        data=b"webp-placeholder",
    )

    with pytest.raises(PermissionError):
        store.start_turn(other["user_id"], "chat_shared", "分析图片", [attachment["id"]])
    assert store.delete_pending_attachment(owner["user_id"], attachment["id"])


def test_sqlite_checkpoint_survives_connection_restart_and_can_be_deleted(tmp_path):
    database = tmp_path / "user_data.db"
    UserStore(database)
    builder = StateGraph(_CheckpointState)
    builder.add_node("increment", lambda state: {"value": state["value"] + 1})
    builder.add_edge(START, "increment")
    config = {"configurable": {"thread_id": "usr_one:chat_public"}}

    with sqlite3.connect(database, check_same_thread=False) as connection:
        graph = builder.compile(checkpointer=ThreadedSqliteSaver(connection))
        # 同一 saver 必须同时支持旧同步接口和 SSE 所需的异步接口。
        assert asyncio.run(graph.ainvoke({"value": 1}, config))["value"] == 2

    with sqlite3.connect(database, check_same_thread=False) as connection:
        saver = ThreadedSqliteSaver(connection)
        graph = builder.compile(checkpointer=saver)
        assert graph.get_state(config).values["value"] == 2
        asyncio.run(saver.adelete_thread("usr_one:chat_public"))
        assert not graph.get_state(config).values
