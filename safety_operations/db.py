from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2
CLOSED_EVENT_STATUSES = {"RESOLVED", "CANCELLED"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sanitize_config(value: Any) -> Any:
    """复制配置并移除可能的明文凭据。"""

    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in {"api_key", "access_key", "secret_key", "password", "token"}:
                sanitized[key] = "<redacted>" if item else item
            else:
                sanitized[key] = sanitize_config(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_config(item) for item in value]
    return value


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def connect_database(path: Path | str, busy_timeout_ms: int = 5000) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=max(1.0, busy_timeout_ms / 1000.0))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    initialize_schema(connection)
    return connection


def initialize_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cameras (
            camera_id TEXT PRIMARY KEY,
            name TEXT,
            source TEXT,
            enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS zones (
            camera_id TEXT NOT NULL,
            zone_id TEXT NOT NULL,
            name TEXT,
            zone_type TEXT NOT NULL DEFAULT 'WORK_AREA',
            polygon_json TEXT CHECK (polygon_json IS NULL OR json_valid(polygon_json)),
            enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (camera_id, zone_id),
            FOREIGN KEY (camera_id) REFERENCES cameras(camera_id)
        );

        CREATE TABLE IF NOT EXISTS analysis_runs (
            run_id TEXT PRIMARY KEY,
            camera_id TEXT NOT NULL,
            source_path TEXT,
            output_dir TEXT NOT NULL UNIQUE,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL CHECK (status IN ('RUNNING','COMPLETED','STOPPED','FAILED')),
            fps REAL,
            frame_width INTEGER,
            frame_height INTEGER,
            total_frames INTEGER,
            processed_frames INTEGER,
            model_path TEXT,
            model_sha256 TEXT,
            tracker_config TEXT,
            rule_version TEXT,
            config_json TEXT CHECK (config_json IS NULL OR json_valid(config_json)),
            annotated_video_path TEXT,
            events_jsonl_path TEXT,
            states_jsonl_path TEXT,
            reviews_jsonl_path TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (camera_id) REFERENCES cameras(camera_id)
        );

        CREATE TABLE IF NOT EXISTS security_events (
            event_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            zone_id TEXT,
            event_type TEXT NOT NULL,
            primary_track_id INTEGER,
            lifecycle_status TEXT NOT NULL CHECK (
                lifecycle_status IN ('PENDING','ACTIVE','RECOVERING','RESOLVED','CANCELLED')
            ),
            first_detected_frame INTEGER,
            first_detected_video_seconds REAL,
            activated_frame INTEGER,
            activated_video_seconds REAL,
            resolved_frame INTEGER,
            resolved_video_seconds REAL,
            occurred_at TEXT,
            severity TEXT,
            final_decision TEXT CHECK (
                final_decision IS NULL OR final_decision IN ('CONFIRMED','REJECTED','UNCERTAIN')
            ),
            rule_reason TEXT,
            final_reason TEXT,
            recommended_action TEXT,
            handling_status TEXT NOT NULL DEFAULT 'NONE' CHECK (
                handling_status IN ('NONE','NEW','ACKNOWLEDGED','PROCESSING','CLOSED')
            ),
            latest_metrics_json TEXT CHECK (
                latest_metrics_json IS NULL OR json_valid(latest_metrics_json)
            ),
            thresholds_json TEXT CHECK (thresholds_json IS NULL OR json_valid(thresholds_json)),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
            FOREIGN KEY (camera_id, zone_id) REFERENCES zones(camera_id, zone_id)
        );

        CREATE TABLE IF NOT EXISTS event_transitions (
            transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK (
                status IN ('PENDING','ACTIVE','RECOVERING','RESOLVED','CANCELLED')
            ),
            frame_index INTEGER NOT NULL,
            video_time_seconds REAL NOT NULL,
            metrics_json TEXT CHECK (metrics_json IS NULL OR json_valid(metrics_json)),
            thresholds_json TEXT CHECK (thresholds_json IS NULL OR json_valid(thresholds_json)),
            recorded_at TEXT NOT NULL,
            UNIQUE (event_id, status, frame_index),
            FOREIGN KEY (event_id) REFERENCES security_events(event_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS event_people (
            event_person_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            person_id TEXT,
            identity_status TEXT,
            identity_confidence REAL,
            helmet_status TEXT,
            workwear_status TEXT,
            zone_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (event_id, track_id),
            FOREIGN KEY (event_id) REFERENCES security_events(event_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS llm_reviews (
            attempt_id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('COMPLETED','FAILED')),
            mode TEXT,
            provider TEXT,
            model TEXT,
            prompt_version TEXT,
            prompt_text TEXT,
            reviewed_at TEXT,
            decision TEXT CHECK (
                decision IS NULL OR decision IN ('CONFIRMED','REJECTED','UNCERTAIN')
            ),
            target_visible TEXT,
            helmet_status TEXT,
            evidence_quality TEXT,
            visual_reason TEXT,
            explanation TEXT,
            evidence_timestamps_json TEXT CHECK (
                evidence_timestamps_json IS NULL OR json_valid(evidence_timestamps_json)
            ),
            input_json TEXT CHECK (input_json IS NULL OR json_valid(input_json)),
            response_json TEXT CHECK (response_json IS NULL OR json_valid(response_json)),
            response_id TEXT,
            usage_json TEXT CHECK (usage_json IS NULL OR json_valid(usage_json)),
            latency_ms INTEGER,
            error_json TEXT CHECK (error_json IS NULL OR json_valid(error_json)),
            created_at TEXT NOT NULL,
            FOREIGN KEY (event_id) REFERENCES security_events(event_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS event_evidence (
            evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL,
            evidence_type TEXT NOT NULL,
            file_path TEXT NOT NULL,
            mime_type TEXT,
            file_exists INTEGER NOT NULL CHECK (file_exists IN (0, 1)),
            file_size INTEGER,
            sha256 TEXT,
            metadata_json TEXT CHECK (metadata_json IS NULL OR json_valid(metadata_json)),
            created_at TEXT NOT NULL,
            UNIQUE (event_id, evidence_type, file_path),
            FOREIGN KEY (event_id) REFERENCES security_events(event_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS alert_records (
            alert_id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            channel TEXT,
            recipient TEXT,
            status TEXT,
            sent_at TEXT,
            acknowledged_at TEXT,
            closed_at TEXT,
            payload_json TEXT CHECK (payload_json IS NULL OR json_valid(payload_json)),
            error_message TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (event_id) REFERENCES security_events(event_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS event_handling_actions (
            action_id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            action TEXT NOT NULL CHECK (
                action IN ('ACKNOWLEDGE','START_PROCESSING','CLOSE')
            ),
            operator TEXT NOT NULL,
            comment TEXT,
            previous_status TEXT NOT NULL,
            new_status TEXT NOT NULL,
            acted_at TEXT NOT NULL,
            FOREIGN KEY (event_id) REFERENCES security_events(event_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_runs_camera_started
            ON analysis_runs(camera_id, started_at DESC);
        CREATE INDEX IF NOT EXISTS idx_events_query
            ON security_events(camera_id, event_type, lifecycle_status, activated_video_seconds);
        CREATE INDEX IF NOT EXISTS idx_events_run_track
            ON security_events(run_id, primary_track_id);
        CREATE INDEX IF NOT EXISTS idx_events_handling
            ON security_events(handling_status, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_transitions_event_time
            ON event_transitions(event_id, video_time_seconds);
        CREATE INDEX IF NOT EXISTS idx_reviews_event_time
            ON llm_reviews(event_id, reviewed_at DESC);
        CREATE INDEX IF NOT EXISTS idx_reviews_decision
            ON llm_reviews(status, decision);
        CREATE INDEX IF NOT EXISTS idx_evidence_event_type
            ON event_evidence(event_id, evidence_type);
        CREATE INDEX IF NOT EXISTS idx_alerts_event_status
            ON alert_records(event_id, status);
        CREATE INDEX IF NOT EXISTS idx_handling_actions_event_time
            ON event_handling_actions(event_id, acted_at DESC);

        DROP VIEW IF EXISTS v_event_summary;
        CREATE VIEW v_event_summary AS
        SELECT
            event.*,
            review.attempt_id AS latest_review_attempt_id,
            review.decision AS latest_review_decision,
            review.helmet_status AS latest_review_helmet_status,
            review.explanation AS latest_review_explanation,
            review.reviewed_at AS latest_reviewed_at
        FROM security_events AS event
        LEFT JOIN llm_reviews AS review
          ON review.attempt_id = (
              SELECT candidate.attempt_id
              FROM llm_reviews AS candidate
              WHERE candidate.event_id = event.event_id
                AND candidate.status = 'COMPLETED'
              ORDER BY candidate.reviewed_at DESC, candidate.created_at DESC
              LIMIT 1
          );
        """
    )
    _ensure_column(connection, "security_events", "source_system", "TEXT NOT NULL DEFAULT 'yolo_track'")
    _ensure_column(connection, "alert_records", "notification_kind", "TEXT")
    _ensure_column(connection, "alert_records", "attempt_count", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(connection, "alert_records", "last_attempt_at", "TEXT")
    _ensure_column(connection, "alert_records", "next_retry_at", "TEXT")
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_events_source_event "
        "ON security_events(source_system, event_id)"
    )
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_alert_event_kind "
        "ON alert_records(event_id, channel, notification_kind)"
    )
    # 旧版在规则 ACTIVE 时提前标记 NEW；正式告警改为最终决策后才进入待处理。
    connection.execute(
        "UPDATE security_events SET handling_status='NONE' "
        "WHERE final_decision IS NULL AND handling_status='NEW'"
    )
    connection.execute(
        "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
        (SCHEMA_VERSION, utc_now()),
    )
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    connection.commit()


def _ensure_column(
    connection: sqlite3.Connection, table: str, column: str, definition: str
) -> None:
    """以幂等方式升级已有 SQLite，避免直接复制 WAL 后修改结构。"""

    existing = {
        str(row["name"] if isinstance(row, sqlite3.Row) else row[1])
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in existing:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def upsert_reference_data(connection: sqlite3.Connection, config: dict[str, Any]) -> None:
    now = utc_now()
    camera = config["camera"]
    zone = config["zone"]
    camera_id = str(camera["id"])
    zone_id = str(zone["id"])
    connection.execute(
        """
        INSERT INTO cameras(camera_id, name, source, enabled, created_at, updated_at)
        VALUES (?, ?, ?, 1, ?, ?)
        ON CONFLICT(camera_id) DO UPDATE SET
            name=excluded.name, source=excluded.source, enabled=1, updated_at=excluded.updated_at
        """,
        (camera_id, camera.get("name"), str(camera.get("source", "")), now, now),
    )
    connection.execute(
        """
        INSERT INTO zones(
            camera_id, zone_id, name, zone_type, polygon_json, enabled, created_at, updated_at
        ) VALUES (?, ?, ?, 'WORK_AREA', ?, 1, ?, ?)
        ON CONFLICT(camera_id, zone_id) DO UPDATE SET
            name=excluded.name, zone_type=excluded.zone_type,
            polygon_json=excluded.polygon_json, enabled=1, updated_at=excluded.updated_at
        """,
        (camera_id, zone_id, zone.get("name"), json_text(zone.get("polygon")), now, now),
    )


def upsert_analysis_run(
    connection: sqlite3.Connection,
    *,
    config: dict[str, Any],
    run_dir: Path,
    source_path: Path | None,
    status: str,
    started_at: str,
    finished_at: str | None = None,
    fps: float | None = None,
    width: int | None = None,
    height: int | None = None,
    total_frames: int | None = None,
    processed_frames: int | None = None,
    model_hash: str | None = None,
    error_message: str | None = None,
) -> str:
    now = utc_now()
    run_id = run_dir.name
    model_value = config.get("model", {}).get("path")
    model_path = Path(str(model_value)).resolve() if model_value else None
    database_cfg = config.get("database", {})
    connection.execute(
        """
        INSERT INTO analysis_runs(
            run_id, camera_id, source_path, output_dir, started_at, finished_at, status,
            fps, frame_width, frame_height, total_frames, processed_frames,
            model_path, model_sha256, tracker_config, rule_version, config_json,
            annotated_video_path, events_jsonl_path, states_jsonl_path, reviews_jsonl_path,
            error_message, created_at, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(run_id) DO UPDATE SET
            source_path=COALESCE(excluded.source_path, analysis_runs.source_path),
            finished_at=COALESCE(excluded.finished_at, analysis_runs.finished_at),
            status=excluded.status,
            fps=COALESCE(excluded.fps, analysis_runs.fps),
            frame_width=COALESCE(excluded.frame_width, analysis_runs.frame_width),
            frame_height=COALESCE(excluded.frame_height, analysis_runs.frame_height),
            total_frames=COALESCE(excluded.total_frames, analysis_runs.total_frames),
            processed_frames=COALESCE(excluded.processed_frames, analysis_runs.processed_frames),
            model_sha256=COALESCE(excluded.model_sha256, analysis_runs.model_sha256),
            config_json=excluded.config_json,
            reviews_jsonl_path=excluded.reviews_jsonl_path,
            error_message=excluded.error_message,
            updated_at=excluded.updated_at
        """,
        (
            run_id,
            str(config["camera"]["id"]),
            str(source_path.resolve()) if source_path else None,
            str(run_dir.resolve()),
            started_at,
            finished_at,
            status,
            fps,
            width,
            height,
            total_frames,
            processed_frames,
            str(model_path) if model_path else None,
            model_hash,
            config.get("model", {}).get("tracker"),
            database_cfg.get("rule_version", "safety_rules_v1"),
            json_text(sanitize_config(config)),
            str((run_dir / "annotated.mp4").resolve()),
            str((run_dir / "events.jsonl").resolve()),
            str((run_dir / "states.jsonl").resolve()),
            str((run_dir / "reviews.jsonl").resolve()),
            error_message,
            now,
            now,
        ),
    )
    return run_id


def event_rule_reason(event_type: str) -> str:
    return {
        "NO_HELMET": "近期安全帽佩戴检测比例低于阈值或显式未佩戴比例达到阈值",
        "DWELL": "人员在作业区连续停留时间达到阈值",
        "OVER_COUNT": "作业区稳定人员数量超过配置上限",
    }.get(event_type, "规则条件达到配置阈值")


def evidence_type_from_path(path: Path) -> str:
    name = path.name.lower()
    if name.endswith("_overview.jpg"):
        return "OVERVIEW_IMAGE"
    if name.endswith("_person.jpg"):
        return "PERSON_IMAGE"
    if name.endswith("_review.mp4"):
        return "REVIEW_VIDEO"
    return "OTHER"


def insert_evidence(
    connection: sqlite3.Connection,
    event_id: str,
    path_value: str | Path,
    evidence_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path_value).resolve()
    exists = path.exists() and path.is_file()
    # 图片和视频证据都是可选项；文件不存在时不创建占位记录。
    if not exists:
        return
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".mp4": "video/mp4",
    }.get(path.suffix.lower())
    connection.execute(
        """
        INSERT INTO event_evidence(
            event_id, evidence_type, file_path, mime_type, file_exists,
            file_size, sha256, metadata_json, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?)
        ON CONFLICT(event_id, evidence_type, file_path) DO UPDATE SET
            mime_type=excluded.mime_type, file_exists=excluded.file_exists,
            file_size=excluded.file_size, metadata_json=excluded.metadata_json
        """,
        (
            event_id,
            evidence_type or evidence_type_from_path(path),
            str(path),
            mime,
            1,
            path.stat().st_size,
            None,
            json_text(metadata) if metadata is not None else None,
            utc_now(),
        ),
    )


def upsert_event_transition(
    connection: sqlite3.Connection, run_id: str, record: dict[str, Any]
) -> None:
    now = utc_now()
    event_id = str(record["event_id"])
    status = str(record["status"])
    frame_index = int(record["frame_index"])
    video_seconds = float(record["video_time_seconds"])
    event_type = str(record["event_type"])
    handling_status = "NEW" if status == "ACTIVE" else "NONE"
    activated_frame = frame_index if status == "ACTIVE" else None
    activated_seconds = video_seconds if status == "ACTIVE" else None
    resolved_frame = frame_index if status in CLOSED_EVENT_STATUSES else None
    resolved_seconds = video_seconds if status in CLOSED_EVENT_STATUSES else None
    connection.execute(
        """
        INSERT INTO security_events(
            event_id, run_id, camera_id, zone_id, event_type, primary_track_id,
            lifecycle_status, first_detected_frame, first_detected_video_seconds,
            activated_frame, activated_video_seconds, resolved_frame, resolved_video_seconds,
            rule_reason, handling_status, latest_metrics_json, thresholds_json,
            created_at, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(event_id) DO UPDATE SET
            zone_id=COALESCE(excluded.zone_id, security_events.zone_id),
            lifecycle_status=excluded.lifecycle_status,
            activated_frame=COALESCE(security_events.activated_frame, excluded.activated_frame),
            activated_video_seconds=COALESCE(
                security_events.activated_video_seconds, excluded.activated_video_seconds
            ),
            resolved_frame=COALESCE(excluded.resolved_frame, security_events.resolved_frame),
            resolved_video_seconds=COALESCE(
                excluded.resolved_video_seconds, security_events.resolved_video_seconds
            ),
            handling_status=CASE
                WHEN security_events.handling_status='NONE' AND excluded.lifecycle_status='ACTIVE'
                THEN 'NEW' ELSE security_events.handling_status END,
            latest_metrics_json=excluded.latest_metrics_json,
            thresholds_json=excluded.thresholds_json,
            updated_at=excluded.updated_at
        """,
        (
            event_id,
            run_id,
            str(record["camera_id"]),
            record.get("zone_id"),
            event_type,
            record.get("track_id"),
            status,
            frame_index,
            video_seconds,
            activated_frame,
            activated_seconds,
            resolved_frame,
            resolved_seconds,
            event_rule_reason(event_type),
            handling_status,
            json_text(record.get("metrics", {})),
            json_text(record.get("thresholds", {})),
            now,
            now,
        ),
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO event_transitions(
            event_id, status, frame_index, video_time_seconds,
            metrics_json, thresholds_json, recorded_at
        ) VALUES (?,?,?,?,?,?,?)
        """,
        (
            event_id,
            status,
            frame_index,
            video_seconds,
            json_text(record.get("metrics", {})),
            json_text(record.get("thresholds", {})),
            now,
        ),
    )
    if record.get("track_id") is not None:
        connection.execute(
            """
            INSERT INTO event_people(
                event_id, track_id, person_id, identity_status, identity_confidence,
                helmet_status, workwear_status, zone_id, created_at, updated_at
            ) VALUES (?, ?, NULL, NULL, NULL, NULL, NULL, ?, ?, ?)
            ON CONFLICT(event_id, track_id) DO UPDATE SET
                zone_id=COALESCE(excluded.zone_id, event_people.zone_id),
                updated_at=excluded.updated_at
            """,
            (event_id, int(record["track_id"]), record.get("zone_id"), now, now),
        )
    for path in record.get("evidence_paths", []):
        insert_evidence(connection, event_id, str(path))


def upsert_llm_review(connection: sqlite3.Connection, record: dict[str, Any]) -> None:
    now = utc_now()
    response_fields = {
        key: record.get(key)
        for key in (
            "decision",
            "target_visible",
            "helmet_status",
            "evidence_quality",
            "visual_reason",
            "evidence_timestamps",
        )
    }
    input_value = {
        "input_paths": record.get("input_paths", {}),
        "rule_metrics": record.get("rule_metrics", {}),
        "rule_thresholds": record.get("rule_thresholds", {}),
        "clip_metadata": record.get("clip_metadata"),
    }
    connection.execute(
        """
        INSERT INTO llm_reviews(
            attempt_id, event_id, status, mode, provider, model, prompt_version,
            prompt_text, reviewed_at, decision, target_visible, helmet_status,
            evidence_quality, visual_reason, explanation, evidence_timestamps_json,
            input_json, response_json, response_id, usage_json, latency_ms,
            error_json, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(attempt_id) DO UPDATE SET
            status=excluded.status, decision=excluded.decision,
            response_json=excluded.response_json, usage_json=excluded.usage_json,
            latency_ms=excluded.latency_ms, error_json=excluded.error_json
        """,
        (
            str(record["attempt_id"]),
            str(record["event_id"]),
            str(record["status"]),
            record.get("mode"),
            record.get("provider"),
            record.get("model"),
            record.get("prompt_version"),
            record.get("prompt_text"),
            record.get("reviewed_at"),
            record.get("decision"),
            record.get("target_visible"),
            record.get("helmet_status"),
            record.get("evidence_quality"),
            record.get("visual_reason"),
            record.get("explanation"),
            json_text(record.get("evidence_timestamps", [])),
            json_text(input_value),
            json_text(response_fields),
            record.get("response_id"),
            json_text(record.get("usage")) if record.get("usage") is not None else None,
            record.get("latency_ms"),
            json_text(record.get("error")) if record.get("error") is not None else None,
            now,
        ),
    )
    clip = (record.get("input_paths") or {}).get("clip")
    if clip:
        insert_evidence(
            connection,
            str(record["event_id"]),
            str(clip),
            "REVIEW_VIDEO",
            record.get("clip_metadata"),
        )


def apply_review_decision(connection: sqlite3.Connection, record: dict[str, Any]) -> str | None:
    """将成功复核转成最终事件决策，并原子创建 Agent 发件箱记录。"""

    if record.get("status") != "COMPLETED":
        return None
    decision = str(record.get("decision") or "")
    if decision not in {"CONFIRMED", "REJECTED", "UNCERTAIN"}:
        raise ValueError(f"不支持的复核结论: {decision}")
    event_id = str(record["event_id"])
    event = connection.execute(
        "SELECT event_type FROM security_events WHERE event_id=?", (event_id,)
    ).fetchone()
    if event is None:
        raise ValueError(f"复核对应的安防事件不存在: {event_id}")
    event_type = str(event["event_type"])
    severity = {
        "NO_HELMET": "MEDIUM",
        "DWELL": "MEDIUM",
        "OVER_COUNT": "HIGH",
    }.get(event_type, "MEDIUM")
    recommended_action = {
        "NO_HELMET": "通知现场负责人核验并立即纠正安全帽佩戴状态。",
        "DWELL": "核验人员身份和作业任务，确认是否存在非授权滞留。",
        "OVER_COUNT": "核对作业票允许人数并组织现场分流。",
    }.get(event_type, "通知现场负责人核验并记录处置结果。")
    handling_status = "NEW" if decision in {"CONFIRMED", "UNCERTAIN"} else "NONE"
    connection.execute(
        """
        UPDATE security_events
        SET final_decision=?, severity=?, final_reason=?, recommended_action=?,
            handling_status=CASE
                WHEN handling_status IN ('ACKNOWLEDGED','PROCESSING','CLOSED')
                THEN handling_status ELSE ? END,
            updated_at=?
        WHERE event_id=?
        """,
        (
            decision,
            severity,
            record.get("explanation") or record.get("visual_reason"),
            recommended_action,
            handling_status,
            utc_now(),
            event_id,
        ),
    )
    if decision == "REJECTED":
        return None

    notification_kind = "CONFIRMED_ALERT" if decision == "CONFIRMED" else "REVIEW_REQUIRED"
    alert_id = f"AGENT:{event_id}:{notification_kind}"
    payload = build_agent_event_payload(connection, event_id, notification_kind, alert_id)
    now = utc_now()
    connection.execute(
        """
        INSERT INTO alert_records(
            alert_id,event_id,channel,recipient,status,payload_json,
            created_at,updated_at,notification_kind,attempt_count
        ) VALUES (?,?,'AGENT_WEBHOOK','intelligent-detection-agent','PENDING',?,?,?,?,0)
        ON CONFLICT(alert_id) DO UPDATE SET
            payload_json=excluded.payload_json,
            status=CASE WHEN alert_records.status='SENT' THEN 'SENT' ELSE 'PENDING' END,
            error_message=NULL,
            updated_at=excluded.updated_at
        """,
        (alert_id, event_id, json_text(payload), now, now, notification_kind),
    )
    return alert_id


def build_agent_event_payload(
    connection: sqlite3.Connection,
    event_id: str,
    notification_kind: str,
    alert_id: str,
) -> dict[str, Any]:
    row = connection.execute(
        """
        SELECT event_id,source_system,event_type,camera_id,zone_id,primary_track_id,
               lifecycle_status,occurred_at,activated_video_seconds,severity,final_decision,
               final_reason,recommended_action,handling_status,
               latest_review_attempt_id,latest_review_helmet_status,
               latest_review_explanation,latest_reviewed_at
        FROM v_event_summary WHERE event_id=?
        """,
        (event_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"安防事件不存在: {event_id}")
    evidence = [
        {
            "evidence_id": int(item["evidence_id"]),
            "evidence_type": item["evidence_type"],
            "mime_type": item["mime_type"],
        }
        for item in connection.execute(
            "SELECT evidence_id,evidence_type,mime_type FROM event_evidence "
            "WHERE event_id=? AND file_exists=1 ORDER BY evidence_id",
            (event_id,),
        ).fetchall()
    ]
    return {
        "schema_version": 1,
        "alert_id": alert_id,
        "notification_kind": notification_kind,
        **dict(row),
        "evidence": evidence,
    }


def accept_agent_notification(connection: sqlite3.Connection, payload: dict[str, Any]) -> dict[str, Any]:
    """Agent 接收同库事件，只确认投递，不复制业务主数据。"""

    event_id = str(payload.get("event_id") or "")
    alert_id = str(payload.get("alert_id") or "")
    row = connection.execute(
        """
        SELECT ar.alert_id,e.final_decision
        FROM alert_records ar JOIN security_events e ON e.event_id=ar.event_id
        WHERE ar.alert_id=? AND ar.event_id=? AND ar.channel='AGENT_WEBHOOK'
        """,
        (alert_id, event_id),
    ).fetchone()
    if row is None:
        raise ValueError("发件箱记录或对应安防事件不存在")
    if row["final_decision"] not in {"CONFIRMED", "UNCERTAIN"}:
        raise ValueError("只有已确认或待人工复核事件可以进入 Agent")
    if payload.get("final_decision") != row["final_decision"]:
        raise ValueError("通知结论与事件最终结论不一致")
    now = utc_now()
    connection.execute(
        """
        UPDATE alert_records
        SET status='SENT',sent_at=COALESCE(sent_at,?),error_message=NULL,
            next_retry_at=NULL,updated_at=?
        WHERE alert_id=?
        """,
        (now, now, alert_id),
    )
    return {"accepted": True, "event_id": event_id, "alert_id": alert_id}


def accept_local_pending_notifications(connection: sqlite3.Connection) -> int:
    """同库部署时由 Agent 查询入口补收待投递事件，保证服务恢复后不丢告警。"""

    accepted = 0
    rows = connection.execute(
        "SELECT payload_json FROM alert_records WHERE channel='AGENT_WEBHOOK' "
        "AND status IN ('PENDING','FAILED') ORDER BY created_at"
    ).fetchall()
    for row in rows:
        payload = json.loads(str(row["payload_json"]))
        accept_agent_notification(connection, payload)
        accepted += 1
    return accepted


def security_overview(connection: sqlite3.Connection) -> dict[str, Any]:
    base = """
        FROM security_events e
        JOIN alert_records ar ON ar.rowid=(
            SELECT candidate.rowid FROM alert_records candidate
            WHERE candidate.event_id=e.event_id AND candidate.channel='AGENT_WEBHOOK'
              AND candidate.status='SENT'
            ORDER BY candidate.rowid DESC LIMIT 1
        )
    """
    counts = connection.execute(
        f"""
        SELECT COUNT(*) total,
               SUM(CASE WHEN e.final_decision='CONFIRMED' THEN 1 ELSE 0 END) confirmed,
               SUM(CASE WHEN e.final_decision='UNCERTAIN' THEN 1 ELSE 0 END) review_required,
               SUM(CASE WHEN e.handling_status='NEW' THEN 1 ELSE 0 END) new_count,
               SUM(CASE WHEN e.handling_status='PROCESSING' THEN 1 ELSE 0 END) processing,
               SUM(CASE WHEN e.severity='HIGH' AND e.handling_status!='CLOSED' THEN 1 ELSE 0 END) high_risk,
               MAX(ar.rowid) latest_sequence
        {base}
        """
    ).fetchone()
    result = dict(counts) if counts else {}
    return {key: int(value or 0) for key, value in result.items()}


def list_security_events(
    connection: sqlite3.Connection,
    *,
    decision: str | None = None,
    handling_status: str | None = None,
    event_type: str | None = None,
    camera_id: str | None = None,
    event_id: str | None = None,
    after_sequence: int | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    values: list[Any] = []
    for column, value in (
        ("e.final_decision", decision),
        ("e.handling_status", handling_status),
        ("e.event_type", event_type),
        ("e.camera_id", camera_id),
        ("e.event_id", event_id),
    ):
        if value:
            clauses.append(f"{column}=?")
            values.append(value)
    if after_sequence is not None:
        clauses.append("ar.rowid>?")
        values.append(int(after_sequence))
    where = " AND " + " AND ".join(clauses) if clauses else ""
    values.append(max(1, min(int(limit), 500)))
    rows = connection.execute(
        f"""
        SELECT ar.rowid notification_sequence,ar.notification_kind,ar.sent_at,
               e.event_id,e.source_system,e.event_type,e.camera_id,e.zone_id,
               e.primary_track_id,e.lifecycle_status,e.activated_video_seconds,e.occurred_at,
               e.severity,e.final_decision,e.final_reason,e.recommended_action,
               e.handling_status,v.latest_review_helmet_status,v.latest_review_explanation,
               v.latest_reviewed_at,
               (SELECT COUNT(*) FROM event_evidence ev WHERE ev.event_id=e.event_id) evidence_count
        FROM security_events e
        JOIN v_event_summary v ON v.event_id=e.event_id
        JOIN alert_records ar ON ar.rowid=(
            SELECT candidate.rowid FROM alert_records candidate
            WHERE candidate.event_id=e.event_id AND candidate.channel='AGENT_WEBHOOK'
              AND candidate.status='SENT'
            ORDER BY candidate.rowid DESC LIMIT 1
        )
        WHERE 1=1 {where}
        ORDER BY ar.rowid DESC LIMIT ?
        """,
        values,
    ).fetchall()
    return [dict(row) for row in rows]


def get_security_event(connection: sqlite3.Connection, event_id: str) -> dict[str, Any] | None:
    items = list_security_events(connection, event_id=event_id, limit=1)
    if not items:
        return None
    event = items[0]
    event["evidence"] = [
        dict(row)
        for row in connection.execute(
            "SELECT evidence_id,evidence_type,mime_type,file_size,metadata_json "
            "FROM event_evidence WHERE event_id=? AND file_exists=1 ORDER BY evidence_id",
            (event_id,),
        ).fetchall()
    ]
    event["actions"] = [
        dict(row)
        for row in connection.execute(
            "SELECT action_id,action,operator,comment,previous_status,new_status,acted_at "
            "FROM event_handling_actions WHERE event_id=? ORDER BY acted_at",
            (event_id,),
        ).fetchall()
    ]
    return event


def get_security_evidence_path(
    connection: sqlite3.Connection, event_id: str, evidence_id: int
) -> Path | None:
    row = connection.execute(
        "SELECT file_path FROM event_evidence WHERE event_id=? AND evidence_id=? AND file_exists=1",
        (event_id, int(evidence_id)),
    ).fetchone()
    return Path(row["file_path"]) if row else None


def handle_security_event(
    connection: sqlite3.Connection,
    event_id: str,
    action: str,
    operator: str,
    comment: str | None,
) -> dict[str, Any]:
    transitions = {
        "ACKNOWLEDGE": ({"NEW"}, "ACKNOWLEDGED"),
        "START_PROCESSING": ({"ACKNOWLEDGED"}, "PROCESSING"),
        "CLOSE": ({"NEW", "ACKNOWLEDGED", "PROCESSING"}, "CLOSED"),
    }
    if action not in transitions:
        raise ValueError(f"不支持的处置动作: {action}")
    row = connection.execute(
        "SELECT handling_status FROM security_events WHERE event_id=?", (event_id,)
    ).fetchone()
    if row is None:
        raise ValueError("安防事件不存在")
    current = str(row["handling_status"])
    allowed, target = transitions[action]
    if current not in allowed:
        raise ValueError(f"事件当前状态 {current} 不能执行 {action}")
    now = utc_now()
    connection.execute(
        "UPDATE security_events SET handling_status=?,updated_at=? WHERE event_id=?",
        (target, now, event_id),
    )
    connection.execute(
        """
        INSERT INTO event_handling_actions(
            action_id,event_id,action,operator,comment,previous_status,new_status,acted_at
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        (uuid.uuid4().hex, event_id, action, operator, comment, current, target, now),
    )
    if action == "ACKNOWLEDGE":
        connection.execute(
            "UPDATE alert_records SET acknowledged_at=COALESCE(acknowledged_at,?),updated_at=? "
            "WHERE event_id=? AND channel='AGENT_WEBHOOK'",
            (now, now, event_id),
        )
    elif action == "CLOSE":
        connection.execute(
            "UPDATE alert_records SET closed_at=COALESCE(closed_at,?),updated_at=? "
            "WHERE event_id=? AND channel='AGENT_WEBHOOK'",
            (now, now, event_id),
        )
    return {"event_id": event_id, "previous_status": current, "handling_status": target}


def append_sync_error(
    run_dir: Path, operation: str, record_key: str | None, exc: Exception
) -> None:
    path = run_dir / "db_sync_errors.jsonl"
    record = {
        "occurred_at": utc_now(),
        "operation": operation,
        "record_key": record_key,
        "error_type": type(exc).__name__,
        "message": str(exc),
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json_text(record) + "\n")


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON。") from exc
            if isinstance(value, dict):
                records.append(value)
    return records


def inferred_run_started_at(run_dir: Path) -> str:
    try:
        parsed = datetime.strptime(run_dir.name, "run_%Y%m%d_%H%M%S").astimezone()
        return parsed.isoformat(timespec="seconds")
    except ValueError:
        return datetime.fromtimestamp(run_dir.stat().st_mtime).astimezone().isoformat(
            timespec="seconds"
        )


def sync_run_directory(
    connection: sqlite3.Connection,
    config: dict[str, Any],
    run_dir: Path,
    source_path: Path | None = None,
) -> dict[str, int]:
    """幂等导入一个运行目录；旧目录缺失的运行元数据保持为空。"""

    run_dir = run_dir.resolve()
    events = read_jsonl_if_exists(run_dir / "events.jsonl")
    reviews = read_jsonl_if_exists(run_dir / "reviews.jsonl")
    states = read_jsonl_if_exists(run_dir / "states.jsonl")
    processed_frames = None
    if states:
        processed_frames = int(states[-1].get("frame_index", -1)) + 1
    finished_at = datetime.fromtimestamp(run_dir.stat().st_mtime).astimezone().isoformat(
        timespec="seconds"
    )
    with connection:
        upsert_reference_data(connection, config)
        run_id = upsert_analysis_run(
            connection,
            config=config,
            run_dir=run_dir,
            source_path=source_path,
            status="COMPLETED",
            started_at=inferred_run_started_at(run_dir),
            finished_at=finished_at,
            processed_frames=processed_frames,
        )
        for event in events:
            upsert_event_transition(connection, run_id, event)
        for review in reviews:
            # 旧的失败记录也需要保留；只有事件已存在时才插入复核。
            exists = connection.execute(
                "SELECT 1 FROM security_events WHERE event_id=?", (review.get("event_id"),)
            ).fetchone()
            if exists:
                upsert_llm_review(connection, review)
    return {
        "runs": 1,
        "event_transitions": len(events),
        "reviews": len(reviews),
    }
