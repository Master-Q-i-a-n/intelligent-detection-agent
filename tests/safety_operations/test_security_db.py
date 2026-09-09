import json
from pathlib import Path

from intelligent_detection_agent.safety_operations.db import (
    apply_review_decision,
    claim_source_review,
    connect_database,
    finish_source_review_claim,
    get_security_event,
    sanitize_config,
    sync_run_directory,
    upsert_llm_review,
)


def make_config(tmp_path: Path) -> dict:
    return {
        "camera": {"id": "camera_01", "name": "一号摄像头", "source": "source.mp4"},
        "zone": {
            "id": "work_area",
            "name": "中心作业区",
            "polygon": [[0.25, 0.25], [0.75, 0.25], [0.75, 0.8], [0.25, 0.8]],
        },
        "model": {"path": "last.pt", "tracker": "bytetrack.yaml"},
        "output": {"root": str(tmp_path / "outputs")},
        "database": {"path": str(tmp_path / "security.db")},
        "review": {"api_key": "must-not-be-persisted", "model": "doubao-test"},
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_sanitize_config_redacts_nested_credentials() -> None:
    source = {"review": {"api_key": "secret", "nested": {"token": "token-value"}}}
    sanitized = sanitize_config(source)

    assert sanitized["review"]["api_key"] == "<redacted>"
    assert sanitized["review"]["nested"]["token"] == "<redacted>"
    assert source["review"]["api_key"] == "secret"


def test_sync_run_is_idempotent_and_keeps_optional_evidence_optional(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    run_dir = tmp_path / "outputs" / "run_20260808_002828"
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True)
    overview = evidence_dir / "event-1_overview.jpg"
    review_clip = evidence_dir / "event-1_review.mp4"
    overview.write_bytes(b"jpeg-placeholder")
    review_clip.write_bytes(b"mp4-placeholder")
    # person.jpg 故意不创建，数据库中不应产生占位证据。
    missing_person = evidence_dir / "event-1_person.jpg"

    event_rows = [
        {
            "event_id": "event-1",
            "event_type": "NO_HELMET",
            "status": "PENDING",
            "camera_id": "camera_01",
            "zone_id": None,
            "track_id": 7,
            "frame_index": 10,
            "video_time_seconds": 1.0,
            "metrics": {"helmet_ratio": 0.0},
            "thresholds": {"violation_helmet_ratio": 0.2},
            "evidence_paths": [],
        },
        {
            "event_id": "event-1",
            "event_type": "NO_HELMET",
            "status": "ACTIVE",
            "camera_id": "camera_01",
            "zone_id": None,
            "track_id": 7,
            "frame_index": 20,
            "video_time_seconds": 2.0,
            "metrics": {"helmet_ratio": 0.0},
            "thresholds": {"violation_helmet_ratio": 0.2},
            "evidence_paths": [str(overview), str(missing_person)],
        },
    ]
    review_rows = [
        {
            "attempt_id": "attempt-1",
            "event_id": "event-1",
            "status": "COMPLETED",
            "mode": "shadow",
            "provider": "volcengine_ark",
            "model": "doubao-test",
            "prompt_version": "no_helmet_v2",
            "reviewed_at": "2026-08-08T00:30:00+08:00",
            "decision": "CONFIRMED",
            "target_visible": "CLEAR",
            "helmet_status": "NOT_WORN",
            "evidence_quality": "GOOD",
            "visual_reason": "目标头部清晰，未见安全帽。",
            "evidence_timestamps": [1.2],
            "explanation": "规则与视频复核均支持未佩戴安全帽。",
            "input_paths": {"clip": str(review_clip)},
            "rule_metrics": {"helmet_ratio": 0.0},
            "rule_thresholds": {"violation_helmet_ratio": 0.2},
            "usage": {"total_tokens": 100},
            "latency_ms": 1234,
            "error": None,
        }
    ]
    write_jsonl(run_dir / "events.jsonl", event_rows)
    write_jsonl(run_dir / "reviews.jsonl", review_rows)
    write_jsonl(
        run_dir / "states.jsonl",
        [{"frame_index": 29, "video_time_seconds": 2.9, "tracks": []}],
    )

    connection = connect_database(tmp_path / "security.db")
    try:
        sync_run_directory(connection, config, run_dir, tmp_path / "source.mp4")
        sync_run_directory(connection, config, run_dir, tmp_path / "source.mp4")

        assert connection.execute("SELECT count(*) FROM analysis_runs").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM security_events").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM event_transitions").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM event_people").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM llm_reviews").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM event_evidence").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM alert_records").fetchone()[0] == 0

        event = connection.execute(
            "SELECT lifecycle_status, final_decision FROM security_events WHERE event_id='event-1'"
        ).fetchone()
        assert tuple(event) == ("ACTIVE", None)
        summary = connection.execute(
            "SELECT latest_review_decision FROM v_event_summary WHERE event_id='event-1'"
        ).fetchone()
        assert summary[0] == "CONFIRMED"
        run = connection.execute(
            "SELECT processed_frames, config_json FROM analysis_runs"
        ).fetchone()
        assert run[0] == 30
        assert "must-not-be-persisted" not in run[1]
        # 设计上不保存每秒状态快照，因此没有 states/snapshots 业务表。
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "states" not in tables
        assert "state_snapshots" not in tables

        with connection:
            apply_review_decision(connection, review_rows[0])
            connection.execute(
                "UPDATE alert_records SET status='SENT',sent_at='2026-08-08T00:31:00+08:00' "
                "WHERE event_id='event-1'"
            )
        detail = get_security_event(connection, "event-1")
        assert detail is not None
        assert detail["yolo_rule_metrics"] == {"helmet_ratio": 0.0}

        # 截图证据与复核结论解耦：失败记录仍应保留已经生成的异常帧。
        snapshot = evidence_dir / "PPE_hash_snapshot.jpg"
        snapshot.write_bytes(b"jpeg-snapshot")
        with connection:
            upsert_llm_review(
                connection,
                {
                    "attempt_id": "attempt-failed",
                    "event_id": "event-1",
                    "status": "FAILED",
                    "input_paths": {"snapshot": str(snapshot)},
                    "clip_metadata": {
                        "snapshot_video_time_seconds": 2.0,
                        "snapshot_track_ids": [7],
                    },
                    "error": {"code": "API_ERROR", "message": "测试失败"},
                },
            )
        evidence = connection.execute(
            "SELECT evidence_type,metadata_json FROM event_evidence "
            "WHERE event_id='event-1' AND evidence_type='PPE_ANOMALY_IMAGE'"
        ).fetchone()
        assert evidence is not None
        assert json.loads(evidence["metadata_json"]) == {
            "video_time_seconds": 2.0,
            "track_ids": [7],
        }
    finally:
        connection.close()


def test_schema_accepts_event_without_zone_or_evidence(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    run_dir = tmp_path / "outputs" / "run_no_evidence"
    run_dir.mkdir(parents=True)
    write_jsonl(
        run_dir / "events.jsonl",
        [
            {
                "event_id": "over-count-1",
                "event_type": "OVER_COUNT",
                "status": "ACTIVE",
                "camera_id": "camera_01",
                "zone_id": "work_area",
                "track_id": None,
                "frame_index": 100,
                "video_time_seconds": 10.0,
                "metrics": {"person_count": 4},
                "thresholds": {"max_people": 3},
                "evidence_paths": [],
            }
        ],
    )

    connection = connect_database(tmp_path / "security.db")
    try:
        sync_run_directory(connection, config, run_dir)
        assert connection.execute("SELECT count(*) FROM security_events").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM event_evidence").fetchone()[0] == 0
    finally:
        connection.close()


def test_source_video_review_claim_survives_reconnect(tmp_path: Path) -> None:
    database = tmp_path / "security.db"
    connection = connect_database(database)
    try:
        with connection:
            assert claim_source_review(
                connection,
                source_sha256="a" * 64,
                event_id="PPE_event",
                attempt_id="attempt-1",
                provider="volcengine_ark",
                model="doubao-test",
                prompt_version="ppe_inspection_v1",
            )
            finish_source_review_claim(connection, "a" * 64, "FAILED", {"code": "TEST"})
    finally:
        connection.close()

    connection = connect_database(database)
    try:
        with connection:
            assert not claim_source_review(
                connection,
                source_sha256="a" * 64,
                event_id="PPE_event_2",
                attempt_id="attempt-2",
                provider="volcengine_ark",
                model="doubao-test",
                prompt_version="ppe_inspection_v1",
            )
        row = connection.execute(
            "SELECT outcome,attempt_id FROM llm_review_claims WHERE source_sha256=?",
            ("a" * 64,),
        ).fetchone()
        assert tuple(row) == ("FAILED", "attempt-1")
    finally:
        connection.close()
