from pathlib import Path

from intelligent_detection_agent.safety_operations.db import (
    accept_agent_notification,
    apply_review_decision,
    connect_database,
    handle_security_event,
    upsert_analysis_run,
    upsert_event_transition,
    upsert_llm_review,
    upsert_reference_data,
)


def prepare_event(tmp_path: Path):
    connection = connect_database(tmp_path / "security.db")
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    config = {
        "camera": {"id": "camera_01", "name": "测试摄像头", "source": "test.mp4"},
        "zone": {"id": "work_area", "name": "测试区域", "polygon": []},
        "model": {"path": "model.pt", "tracker": "bytetrack.yaml"},
        "database": {"rule_version": "test-v1"},
    }
    with connection:
        upsert_reference_data(connection, config)
        upsert_analysis_run(
            connection,
            config=config,
            run_dir=run_dir,
            source_path=None,
            status="COMPLETED",
            started_at="2026-08-08T00:00:00+08:00",
        )
        upsert_event_transition(connection, run_dir.name, {
            "event_id": "event-1", "event_type": "NO_HELMET", "status": "ACTIVE",
            "camera_id": "camera_01", "zone_id": "work_area", "track_id": 7,
            "frame_index": 100, "video_time_seconds": 5.0, "metrics": {}, "thresholds": {},
        })
    return connection


def test_confirmed_review_creates_idempotent_outbox_and_action_audit(tmp_path: Path) -> None:
    connection = prepare_event(tmp_path)
    review = {
        "attempt_id": "attempt-1", "event_id": "event-1", "status": "COMPLETED",
        "mode": "active", "decision": "CONFIRMED", "helmet_status": "NOT_WORN",
        "evidence_quality": "GOOD", "visual_reason": "目标头部未佩戴安全帽",
        "explanation": "复核确认未佩戴安全帽", "evidence_timestamps": [1.0],
        "input_paths": {}, "reviewed_at": "2026-08-08T00:01:00+08:00",
    }
    with connection:
        upsert_llm_review(connection, review)
        alert_id = apply_review_decision(connection, review)
    assert alert_id == "AGENT:event-1:CONFIRMED_ALERT"
    alert = connection.execute("SELECT payload_json,status FROM alert_records").fetchone()
    import json
    payload = json.loads(alert["payload_json"])
    with connection:
        accept_agent_notification(connection, payload)
        handle_security_event(connection, "event-1", "ACKNOWLEDGE", "tester", "已核查")
        # 重复复核只补充源数据，不能把人工状态退回 NEW。
        apply_review_decision(connection, review)
    event = connection.execute(
        "SELECT final_decision,handling_status FROM security_events WHERE event_id='event-1'"
    ).fetchone()
    assert tuple(event) == ("CONFIRMED", "ACKNOWLEDGED")
    assert connection.execute("SELECT count(*) FROM alert_records").fetchone()[0] == 1
    assert connection.execute("SELECT count(*) FROM event_handling_actions").fetchone()[0] == 1
    connection.close()


def test_rejected_review_does_not_create_agent_alert(tmp_path: Path) -> None:
    connection = prepare_event(tmp_path)
    review = {
        "attempt_id": "attempt-2", "event_id": "event-1", "status": "COMPLETED",
        "decision": "REJECTED", "visual_reason": "目标已佩戴安全帽",
        "explanation": "排除违规", "input_paths": {},
    }
    with connection:
        upsert_llm_review(connection, review)
        assert apply_review_decision(connection, review) is None
    assert connection.execute("SELECT count(*) FROM alert_records").fetchone()[0] == 0
    connection.close()
