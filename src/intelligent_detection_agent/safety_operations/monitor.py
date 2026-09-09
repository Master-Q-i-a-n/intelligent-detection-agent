from __future__ import annotations

import argparse
import json
import math
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from .db import (
    CLOSED_EVENT_STATUSES,
    append_sync_error,
    connect_database,
    file_sha256,
    sanitize_config,
    upsert_analysis_run,
    upsert_event_transition,
    upsert_reference_data,
    utc_now,
)
from .video import create_browser_video_writer


HELMET = "HELMET"
NO_HELMET = "NO_HELMET"
ABSENT = "ABSENT"
UNKNOWN = "UNKNOWN"


@dataclass
class Detection:
    """单帧检测结果；只有 Person 的 track_id 会进入持续状态。"""

    class_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    track_id: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2


@dataclass
class TrackState:
    track_id: int
    first_seen_at: float
    last_seen_at: float
    bbox: tuple[float, float, float, float]
    person_confidence: float
    foot_point: tuple[float, float]
    head_roi: tuple[float, float, float, float]
    seen_frames: int = 0
    observed_seconds: float = 0.0
    missing_since: float | None = None
    is_missing: bool = False
    raw_zone: str | None = None
    candidate_zone: str | None = None
    candidate_since: float | None = None
    candidate_active: bool = False
    current_zone: str | None = None
    zone_enter_at: float | None = None
    zone_dwell_seconds: float = 0.0
    helmet_observation: str = UNKNOWN
    helmet_confidence: float | None = None
    helmet_history: deque[tuple[float, str]] = field(default_factory=deque)
    helmet_evaluable_frames: int = 0
    helmet_positive_frames: int = 0
    no_helmet_positive_frames: int = 0
    helmet_absent_frames: int = 0
    helmet_unknown_frames: int = 0
    recent_helmet_ratio: float | None = None
    recent_no_helmet_ratio: float | None = None
    recent_evaluable_frames: int = 0
    ppe_status: str = "OBSERVING"
    active_helmet_event_id: str | None = None
    active_dwell_event_id: str | None = None
    helmet_seen_worn_at: float | None = None
    last_helmet_worn_at: float | None = None
    helmet_violation_at: float | None = None
    gloves_history: deque[tuple[float, float]] = field(default_factory=deque)
    goggles_history: deque[tuple[float, float]] = field(default_factory=deque)
    gloves_status: str = "OBSERVING"
    goggles_status: str = "OBSERVING"
    gloves_confirmed_at: float | None = None
    goggles_confirmed_at: float | None = None
    gloves_positive_frames: int = 0
    goggles_positive_frames: int = 0
    no_gloves_positive_frames: int = 0
    no_goggle_positive_frames: int = 0
    gloves_effective_seconds: float = 0.0
    goggles_effective_seconds: float = 0.0
    gloves_all_hits: list[tuple[float, float]] = field(default_factory=list)
    goggles_all_hits: list[tuple[float, float]] = field(default_factory=list)
    first_no_helmet_at: float | None = None
    first_bbox: tuple[float, float, float, float] | None = None
    best_bbox: tuple[float, float, float, float] | None = None
    bbox_history: list[tuple[float, tuple[float, float, float, float]]] = field(
        default_factory=list
    )
    best_visibility_at: float = 0.0
    best_bbox_area: float = 0.0


@dataclass
class ZoneState:
    zone_id: str
    person_ids: set[int] = field(default_factory=set)
    person_count: int = 0
    count_condition: str = "NORMAL"
    active_count_event_id: str | None = None
    last_updated_at: float = 0.0


@dataclass
class EventState:
    event_id: str
    event_key: str
    event_type: str
    camera_id: str
    zone_id: str | None
    track_id: int | None
    status: str
    pending_since: float
    triggered_at: float | None = None
    recovering_since: float | None = None
    resolved_at: float | None = None
    last_updated_at: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    evidence_paths: list[str] = field(default_factory=list)


def load_config(config_path: Path) -> dict[str, Any]:
    """以 UTF-8 读取 YAML，并检查首版必需的配置段。"""

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("配置文件内容必须是 YAML 对象。")
    required_sections = {"camera", "model", "zone", "tracking", "rules", "display", "output"}
    missing = sorted(required_sections - config.keys())
    if missing:
        raise ValueError(f"配置文件缺少字段: {', '.join(missing)}")
    return config


def resolve_config_path(config_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_dir / path).resolve()


def point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    """边界上的脚点也视为处于区域内。"""

    return cv2.pointPolygonTest(polygon.astype(np.float32), point, False) >= 0


def normalized_polygon_to_pixels(
    polygon: list[list[float]], width: int, height: int
) -> np.ndarray:
    points = np.array([(x * width, y * height) for x, y in polygon], dtype=np.int32)
    if len(points) < 3:
        raise ValueError("区域 polygon 至少需要三个点。")
    return points


def make_head_roi(
    bbox: tuple[float, float, float, float], head_height_ratio: float
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return x1, y1, x2, y1 + (y2 - y1) * head_height_ratio


def _associate_one_class(
    persons: list[Detection], items: list[Detection], head_height_ratio: float
) -> dict[int, Detection]:
    """按头部中心归一化距离做贪心一对一关联。"""

    candidates: list[tuple[float, int, int]] = []
    for person_index, person in enumerate(persons):
        if person.track_id is None:
            continue
        x1, y1, x2, y2 = person.bbox
        head_roi = make_head_roi(person.bbox, head_height_ratio)
        hx1, hy1, hx2, hy2 = head_roi
        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)
        expected_x = (x1 + x2) / 2
        expected_y = y1 + height * head_height_ratio * 0.5
        for item_index, item in enumerate(items):
            center_x, center_y = item.center
            if hx1 <= center_x <= hx2 and hy1 <= center_y <= hy2:
                distance = math.hypot(
                    (center_x - expected_x) / width,
                    (center_y - expected_y) / height,
                )
                candidates.append((distance, person_index, item_index))

    matches: dict[int, Detection] = {}
    used_persons: set[int] = set()
    used_items: set[int] = set()
    for _, person_index, item_index in sorted(candidates):
        if person_index in used_persons or item_index in used_items:
            continue
        person = persons[person_index]
        if person.track_id is None:
            continue
        matches[person.track_id] = items[item_index]
        used_persons.add(person_index)
        used_items.add(item_index)
    return matches


def associate_ppe(
    persons: list[Detection],
    helmets: list[Detection],
    no_helmets: list[Detection],
    head_height_ratio: float,
    gloves: list[Detection] | None = None,
    no_gloves: list[Detection] | None = None,
    goggles: list[Detection] | None = None,
    no_goggles: list[Detection] | None = None,
) -> dict[int, dict[str, Detection | list[Detection] | None]]:
    helmet_matches = _associate_one_class(persons, helmets, head_height_ratio)
    no_helmet_matches = _associate_one_class(persons, no_helmets, head_height_ratio)
    goggles_matches = _associate_one_class(persons, goggles or [], head_height_ratio)
    no_goggle_matches = _associate_one_class(persons, no_goggles or [], head_height_ratio)

    # 一个人可能同时检出两只手套，因此手套按“中心落入人物框 + 最近人物中心”多对一关联。
    glove_matches = _associate_body_items(persons, gloves or [])
    no_glove_matches = _associate_body_items(persons, no_gloves or [])
    associations: dict[int, dict[str, Detection | list[Detection] | None]] = {}
    for person in persons:
        if person.track_id is None:
            continue
        associations[person.track_id] = {
            "helmet": helmet_matches.get(person.track_id),
            "no_helmet": no_helmet_matches.get(person.track_id),
            "gloves": glove_matches.get(person.track_id, []),
            "no_gloves": no_glove_matches.get(person.track_id, []),
            "goggles": goggles_matches.get(person.track_id),
            "no_goggle": no_goggle_matches.get(person.track_id),
        }
    return associations


def _associate_body_items(
    persons: list[Detection], items: list[Detection]
) -> dict[int, list[Detection]]:
    """将人物框内的 PPE 小目标分配给中心距离最近的人物。"""

    matches: dict[int, list[Detection]] = {}
    for item in items:
        center_x, center_y = item.center
        candidates: list[tuple[float, int]] = []
        for person in persons:
            if person.track_id is None:
                continue
            x1, y1, x2, y2 = person.bbox
            if not (x1 <= center_x <= x2 and y1 <= center_y <= y2):
                continue
            width = max(x2 - x1, 1.0)
            height = max(y2 - y1, 1.0)
            person_x = (x1 + x2) / 2
            person_y = (y1 + y2) / 2
            distance = math.hypot(
                (center_x - person_x) / width, (center_y - person_y) / height
            )
            candidates.append((distance, person.track_id))
        if candidates:
            _, track_id = min(candidates)
            matches.setdefault(track_id, []).append(item)
    return matches


def positive_evidence_duration(
    history: deque[tuple[float, float]],
    now: float,
    window_seconds: float = 3.0,
    hold_seconds: float = 0.5,
) -> tuple[int, float]:
    """计算最近窗口内命中的帧数，以及每次命中保留后的区间并集时长。"""

    cutoff = now - window_seconds
    while history and history[0][0] < cutoff:
        history.popleft()
    intervals = sorted((timestamp, timestamp + hold_seconds) for timestamp, _ in history)
    merged: list[list[float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    duration = sum(end - start for start, end in merged)
    return len(history), duration


def bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    """计算两个人物框的 IoU，供视频结束后的碎片轨迹合并使用。"""

    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def accessory_evidence_from_hits(
    hits: list[tuple[float, float]],
    window_seconds: float,
    hold_seconds: float,
    min_frames: int,
    min_duration: float,
) -> tuple[bool, float, float | None]:
    """跨碎片轨迹重新累计正向证据，返回是否确认、最大时长和确认时间。"""

    ordered = sorted(hits)
    maximum = 0.0
    for timestamp, _ in ordered:
        window = deque(
            item for item in ordered if timestamp - window_seconds <= item[0] <= timestamp
        )
        frames, duration = positive_evidence_duration(
            window, timestamp, window_seconds, hold_seconds
        )
        maximum = max(maximum, duration)
        if frames >= min_frames and duration >= min_duration:
            return True, maximum, timestamp
    return False, maximum, None


def update_zone_membership(
    state: TrackState,
    raw_zone: str | None,
    now: float,
    enter_confirm_seconds: float,
    exit_confirm_seconds: float,
) -> None:
    """对进入和离开分别做持续时间确认，抑制边界抖动。"""

    state.raw_zone = raw_zone
    if raw_zone == state.current_zone:
        state.candidate_active = False
        state.candidate_zone = None
        state.candidate_since = None
    elif not state.candidate_active or state.candidate_zone != raw_zone:
        state.candidate_active = True
        state.candidate_zone = raw_zone
        state.candidate_since = now
    else:
        confirm_seconds = enter_confirm_seconds if raw_zone is not None else exit_confirm_seconds
        if state.candidate_since is not None and now - state.candidate_since >= confirm_seconds:
            state.current_zone = raw_zone
            state.zone_enter_at = now if raw_zone is not None else None
            state.candidate_active = False
            state.candidate_zone = None
            state.candidate_since = None

    state.zone_dwell_seconds = (
        max(0.0, now - state.zone_enter_at)
        if state.current_zone is not None and state.zone_enter_at is not None
        else 0.0
    )


def prune_helmet_history(state: TrackState, now: float, window_seconds: float) -> None:
    while state.helmet_history and now - state.helmet_history[0][0] > window_seconds:
        state.helmet_history.popleft()

    evaluable = [value for _, value in state.helmet_history if value != UNKNOWN]
    state.recent_evaluable_frames = len(evaluable)
    if not evaluable:
        state.recent_helmet_ratio = None
        state.recent_no_helmet_ratio = None
        return
    state.recent_helmet_ratio = evaluable.count(HELMET) / len(evaluable)
    state.recent_no_helmet_ratio = evaluable.count(NO_HELMET) / len(evaluable)


def add_helmet_observation(
    state: TrackState, observation: str, now: float, window_seconds: float
) -> None:
    state.helmet_observation = observation
    state.helmet_history.append((now, observation))
    if observation == HELMET:
        state.helmet_evaluable_frames += 1
        state.helmet_positive_frames += 1
    elif observation == NO_HELMET:
        state.helmet_evaluable_frames += 1
        state.no_helmet_positive_frames += 1
    elif observation == ABSENT:
        state.helmet_evaluable_frames += 1
        state.helmet_absent_frames += 1
    else:
        state.helmet_unknown_frames += 1
    prune_helmet_history(state, now, window_seconds)


def advance_event(
    event: EventState | None,
    *,
    condition: bool,
    recovery_condition: bool,
    now: float,
    confirm_seconds: float,
    recovery_seconds: float,
    event_key: str,
    event_type: str,
    camera_id: str,
    zone_id: str | None,
    track_id: int | None,
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
) -> tuple[EventState | None, str | None]:
    """推进事件生命周期，返回本次发生的状态变化。"""

    if event is None or event.status in CLOSED_EVENT_STATUSES:
        if not condition:
            return event, None
        status = "ACTIVE" if confirm_seconds <= 0 else "PENDING"
        event = EventState(
            event_id=uuid.uuid4().hex,
            event_key=event_key,
            event_type=event_type,
            camera_id=camera_id,
            zone_id=zone_id,
            track_id=track_id,
            status=status,
            pending_since=now,
            triggered_at=now if status == "ACTIVE" else None,
            last_updated_at=now,
            metrics=metrics,
            thresholds=thresholds,
        )
        return event, status

    event.last_updated_at = now
    event.metrics = metrics
    event.thresholds = thresholds

    if event.status == "PENDING":
        if not condition:
            # 保留被规则防抖过滤掉的候选事件，便于数据库完整还原生命周期。
            event.status = "CANCELLED"
            event.resolved_at = now
            return event, "CANCELLED"
        if now - event.pending_since >= confirm_seconds:
            event.status = "ACTIVE"
            event.triggered_at = now
            return event, "ACTIVE"
        return event, None

    if event.status == "ACTIVE":
        if recovery_condition:
            event.status = "RECOVERING"
            event.recovering_since = now
            return event, "RECOVERING"
        return event, None

    if event.status == "RECOVERING":
        if condition:
            event.status = "ACTIVE"
            event.recovering_since = None
            return event, "ACTIVE"
        if not recovery_condition:
            event.status = "ACTIVE"
            event.recovering_since = None
            return event, "ACTIVE"
        if event.recovering_since is not None and now - event.recovering_since >= recovery_seconds:
            event.status = "RESOLVED"
            event.resolved_at = now
            return event, "RESOLVED"
    return event, None


def _json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, deque):
        return list(value)
    raise TypeError(f"无法序列化类型: {type(value).__name__}")


class SecurityMonitor:
    """单摄像头离线安防分析主流程。"""

    def __init__(
        self,
        config: dict[str, Any],
        config_path: Path,
        source_override: str | None = None,
        show_override: bool | None = None,
    ) -> None:
        self.config = config
        self.config_path = config_path.resolve()
        self.config_dir = self.config_path.parent
        self.camera_id = str(config["camera"]["id"])
        source_value = source_override or config["camera"]["source"]
        self.source_path = resolve_config_path(self.config_dir, source_value)
        self.model_path = resolve_config_path(self.config_dir, config["model"]["path"])
        self.show = config["display"]["show"] if show_override is None else show_override
        self.track_states: dict[int, TrackState] = {}
        self.completed_track_states: dict[int, TrackState] = {}
        self.zone_state = ZoneState(zone_id=str(config["zone"]["id"]))
        self.events: dict[str, EventState] = {}
        self.next_rule_time = 0.0
        self.rule_interval = float(config["tracking"]["rule_interval_seconds"])
        self.events_file = None
        self.states_file = None
        self.video_writer = None
        self.output_dir: Path | None = None
        self.evidence_dir: Path | None = None
        self.db_connection = None
        self.run_started_at = utc_now()
        self.run_fps: float | None = None
        self.run_width: int | None = None
        self.run_height: int | None = None
        self.run_total_frames: int | None = None
        self.processed_frames = 0
        self.source_sha256: str | None = None
        self.raw_accessory_detections: dict[
            str, list[tuple[float, tuple[float, float, float, float], float]]
        ] = {"gloves": [], "goggles": [], "no_gloves": [], "no_goggle": []}
        self._font_cache: dict[int, ImageFont.FreeTypeFont] = {}

    def _prepare(self) -> tuple[cv2.VideoCapture, YOLO, float, int, int, np.ndarray]:
        if not self.source_path.exists():
            raise FileNotFoundError(f"视频不存在: {self.source_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型不存在: {self.model_path}")

        self.source_sha256 = file_sha256(self.source_path)
        if not self.source_sha256:
            raise RuntimeError(f"无法计算源视频 SHA-256: {self.source_path}")

        cap = cv2.VideoCapture(str(self.source_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {self.source_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError("视频 FPS 或分辨率无效。")

        model = YOLO(str(self.model_path))
        zone_polygon = normalized_polygon_to_pixels(
            self.config["zone"]["polygon"], width, height
        )
        self.run_fps = fps
        self.run_width = width
        self.run_height = height
        self.run_total_frames = total_frames if total_frames > 0 else None
        self._prepare_outputs(fps, width, height)
        self._prepare_database()
        return cap, model, fps, width, height, zone_polygon

    def _prepare_outputs(self, fps: float, width: int, height: int) -> None:
        root = resolve_config_path(self.config_dir, self.config["output"]["root"])
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.output_dir = root / run_name
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.evidence_dir = self.output_dir / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        if self.config["output"].get("save_events", True):
            self.events_file = (self.output_dir / "events.jsonl").open(
                "w", encoding="utf-8"
            )
        if self.config["output"].get("save_states", True):
            self.states_file = (self.output_dir / "states.jsonl").open(
                "w", encoding="utf-8"
            )
        if self.config["output"].get("save_video", True):
            panel_width = int(self.config["display"]["panel_width"])
            output_size = (width + panel_width, height)
            self.video_writer = create_browser_video_writer(
                self.output_dir / "annotated.mp4", fps, output_size
            )

    def _prepare_database(self) -> None:
        """初始化数据库；失败时保留 JSONL 主流程继续运行。"""

        database_cfg = self.config.get("database", {})
        if not database_cfg.get("enabled", True):
            return
        assert self.output_dir is not None
        try:
            database_path = resolve_config_path(
                self.config_dir, database_cfg.get("path", "data/security.db")
            )
            self.db_connection = connect_database(
                database_path, int(database_cfg.get("busy_timeout_ms", 5000))
            )
            with self.db_connection:
                upsert_reference_data(self.db_connection, self.config)
                upsert_analysis_run(
                    self.db_connection,
                    config=self.config,
                    run_dir=self.output_dir,
                    source_path=self.source_path,
                    status="RUNNING",
                    started_at=self.run_started_at,
                    fps=self.run_fps,
                    width=self.run_width,
                    height=self.run_height,
                    total_frames=self.run_total_frames,
                    processed_frames=0,
                    model_hash=file_sha256(self.model_path),
                )
            # 运行清单可帮助离线回填；配置副本会剔除密钥等凭据。
            manifest = {
                "run_id": self.output_dir.name,
                "started_at": self.run_started_at,
                "source_path": str(self.source_path.resolve()),
                "config": sanitize_config(self.config),
            }
            (self.output_dir / "run_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:  # 数据库不可用不能影响视频分析。
            if self.db_connection is not None:
                self.db_connection.close()
                self.db_connection = None
            append_sync_error(self.output_dir, "START_RUN", self.output_dir.name, exc)
            print(f"SQLite 初始化失败，继续仅写 JSONL: {type(exc).__name__}: {exc}")

    def _sync_event_to_database(self, record: dict[str, Any]) -> None:
        if self.db_connection is None or self.output_dir is None:
            return
        try:
            with self.db_connection:
                upsert_event_transition(self.db_connection, self.output_dir.name, record)
        except Exception as exc:  # 单条入库失败只记录补偿日志。
            self.db_connection.rollback()
            append_sync_error(
                self.output_dir, "UPSERT_EVENT", str(record.get("event_id")), exc
            )
            print(f"事件写入 SQLite 失败，已保留 JSONL: {type(exc).__name__}: {exc}")

    def _finish_database(self, status: str, error_message: str | None = None) -> None:
        if self.db_connection is None or self.output_dir is None:
            return
        try:
            with self.db_connection:
                upsert_analysis_run(
                    self.db_connection,
                    config=self.config,
                    run_dir=self.output_dir,
                    source_path=self.source_path,
                    status=status,
                    started_at=self.run_started_at,
                    finished_at=utc_now(),
                    fps=self.run_fps,
                    width=self.run_width,
                    height=self.run_height,
                    total_frames=self.run_total_frames,
                    processed_frames=self.processed_frames,
                    error_message=error_message,
                )
        except Exception as exc:
            self.db_connection.rollback()
            append_sync_error(self.output_dir, "FINISH_RUN", self.output_dir.name, exc)
            print(f"运行结束状态写入 SQLite 失败: {type(exc).__name__}: {exc}")
        finally:
            self.db_connection.close()
            self.db_connection = None

    @staticmethod
    def _resolve_class_ids(model: YOLO) -> dict[str, int]:
        name_to_id = {str(name).lower(): int(class_id) for class_id, name in model.names.items()}
        required = [
            "person",
            "helmet",
            "no_helmet",
            "gloves",
            "no_gloves",
            "goggles",
            "no_goggle",
        ]
        missing = [name for name in required if name not in name_to_id]
        if missing:
            actual = ", ".join(str(name) for name in model.names.values())
            raise ValueError(f"模型缺少类别 {missing}，当前类别为: {actual}")
        return {name: name_to_id[name] for name in required}

    @staticmethod
    def _split_detections(
        result: Any, class_ids: dict[str, int]
    ) -> dict[str, list[Detection]]:
        detections = {name: [] for name in class_ids}
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().tolist()
        classes = boxes.cls.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        track_ids = (
            boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
        )
        for bbox, class_id, confidence, track_id in zip(
            xyxy, classes, confidences, track_ids
        ):
            detection = Detection(
                class_id=class_id,
                bbox=tuple(float(value) for value in bbox),
                confidence=float(confidence),
                track_id=int(track_id) if track_id is not None else None,
            )
            for name, expected_id in class_ids.items():
                if class_id == expected_id:
                    detections[name].append(detection)
                    break
        return detections

    def _update_tracks(
        self,
        persons: list[Detection],
        associations: dict[int, dict[str, Detection | list[Detection] | None]],
        now: float,
        zone_polygon: np.ndarray,
    ) -> None:
        tracking_cfg = self.config["tracking"]
        helmet_cfg = self.config["rules"]["helmet"]
        zone_id = str(self.config["zone"]["id"])
        seen_ids: set[int] = set()

        for person in persons:
            if person.track_id is None:
                # ByteTrack 初始帧可能尚未分配 ID，此时只跳过持续状态更新。
                continue
            track_id = person.track_id
            seen_ids.add(track_id)
            x1, y1, x2, y2 = person.bbox
            foot_point = ((x1 + x2) / 2, y2)
            head_roi = make_head_roi(person.bbox, float(helmet_cfg["head_height_ratio"]))

            state = self.track_states.get(track_id)
            if state is None:
                state = TrackState(
                    track_id=track_id,
                    first_seen_at=now,
                    last_seen_at=now,
                    bbox=person.bbox,
                    person_confidence=person.confidence,
                    foot_point=foot_point,
                    head_roi=head_roi,
                    first_bbox=person.bbox,
                    best_bbox=person.bbox,
                    best_visibility_at=now,
                    best_bbox_area=max(0.0, (x2 - x1) * (y2 - y1)),
                )
                self.track_states[track_id] = state
            else:
                delta = max(0.0, now - state.last_seen_at)
                if state.missing_since is None:
                    state.observed_seconds += delta
                state.last_seen_at = now
                state.bbox = person.bbox
                state.person_confidence = person.confidence
                state.foot_point = foot_point
                state.head_roi = head_roi

            state.seen_frames += 1
            state.bbox_history.append((now, person.bbox))
            state.is_missing = False
            state.missing_since = None
            bbox_area = max(0.0, (x2 - x1) * (y2 - y1))
            if bbox_area > state.best_bbox_area:
                state.best_bbox_area = bbox_area
                state.best_visibility_at = now
                state.best_bbox = person.bbox
            raw_zone = zone_id if point_in_polygon(foot_point, zone_polygon) else None
            update_zone_membership(
                state,
                raw_zone,
                now,
                float(tracking_cfg["enter_confirm_seconds"]),
                float(tracking_cfg["exit_confirm_seconds"]),
            )

            matched = associations.get(track_id, {})
            helmet = matched.get("helmet")
            no_helmet = matched.get("no_helmet")
            person_height = y2 - y1
            head_evaluable = (
                person_height >= float(helmet_cfg["min_person_box_height"])
                and y1 >= float(helmet_cfg["head_border_margin_pixels"])
            )
            state.helmet_confidence = None
            if helmet is not None and no_helmet is not None:
                observation = UNKNOWN
            elif helmet is not None:
                observation = HELMET
                state.helmet_confidence = helmet.confidence
                if state.helmet_seen_worn_at is None:
                    state.helmet_seen_worn_at = now
                state.last_helmet_worn_at = now
            elif no_helmet is not None:
                observation = NO_HELMET
                state.helmet_confidence = no_helmet.confidence
                if state.first_no_helmet_at is None:
                    state.first_no_helmet_at = now
            elif head_evaluable:
                observation = ABSENT
            else:
                observation = UNKNOWN
            add_helmet_observation(
                state, observation, now, float(helmet_cfg["window_seconds"])
            )

            accessory_cfg = self.config["rules"].get("ppe_accessories", {})
            window_seconds = float(accessory_cfg.get("window_seconds", 3.0))
            hold_seconds = float(accessory_cfg.get("hit_hold_seconds", 0.5))
            min_frames = int(accessory_cfg.get("min_positive_frames", 2))
            min_duration = float(accessory_cfg.get("min_effective_seconds", 0.5))
            for key, negative_key in (("gloves", "no_gloves"), ("goggles", "no_goggle")):
                positive = matched.get(key)
                if isinstance(positive, list):
                    positive_items = positive
                elif isinstance(positive, Detection):
                    positive_items = [positive]
                else:
                    positive_items = []
                if positive_items:
                    history = state.gloves_history if key == "gloves" else state.goggles_history
                    history.append((now, max(item.confidence for item in positive_items)))
                    all_hits = (
                        state.gloves_all_hits
                        if key == "gloves"
                        else state.goggles_all_hits
                    )
                    all_hits.append((now, max(item.confidence for item in positive_items)))
                    if key == "gloves":
                        state.gloves_positive_frames += 1
                    else:
                        state.goggles_positive_frames += 1
                    frames, duration = positive_evidence_duration(
                        history, now, window_seconds, hold_seconds
                    )
                    if key == "gloves":
                        state.gloves_effective_seconds = max(
                            state.gloves_effective_seconds, duration
                        )
                        if frames >= min_frames and duration >= min_duration:
                            state.gloves_status = "WORN"
                            state.gloves_confirmed_at = state.gloves_confirmed_at or now
                    else:
                        state.goggles_effective_seconds = max(
                            state.goggles_effective_seconds, duration
                        )
                        if frames >= min_frames and duration >= min_duration:
                            state.goggles_status = "WORN"
                            state.goggles_confirmed_at = state.goggles_confirmed_at or now
                negative = matched.get(negative_key)
                if isinstance(negative, list):
                    negative_hit = bool(negative)
                else:
                    negative_hit = isinstance(negative, Detection)
                if negative_hit:
                    if key == "gloves":
                        state.no_gloves_positive_frames += 1
                    else:
                        state.no_goggle_positive_frames += 1

        lost_grace = float(tracking_cfg["lost_grace_seconds"])
        for track_id, state in self.track_states.items():
            if track_id in seen_ids:
                continue
            if state.missing_since is None:
                state.missing_since = now
            state.is_missing = True
            if now - state.missing_since > lost_grace:
                state.raw_zone = None
                state.current_zone = None
                state.zone_enter_at = None
                state.zone_dwell_seconds = 0.0
                state.candidate_active = False

    def _save_evidence(
        self, event: EventState, frame: np.ndarray, zone_polygon: np.ndarray
    ) -> None:
        if not self.config["output"].get("save_evidence", True):
            return
        assert self.evidence_dir is not None
        overview = frame.copy()
        cv2.polylines(overview, [zone_polygon], True, (0, 0, 255), 2)
        state = self.track_states.get(event.track_id) if event.track_id is not None else None
        if state is not None:
            x1, y1, x2, y2 = (int(value) for value in state.bbox)
            cv2.rectangle(overview, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                overview,
                f"Track {state.track_id}",
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        overview_path = self.evidence_dir / f"{event.event_id}_overview.jpg"
        if cv2.imwrite(str(overview_path), overview):
            event.evidence_paths.append(str(overview_path.resolve()))

        if state is not None:
            height, width = frame.shape[:2]
            x1, y1, x2, y2 = (int(value) for value in state.bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 > x1 and y2 > y1:
                crop_path = self.evidence_dir / f"{event.event_id}_person.jpg"
                if cv2.imwrite(str(crop_path), frame[y1:y2, x1:x2]):
                    event.evidence_paths.append(str(crop_path.resolve()))

    def _write_event(self, event: EventState, frame_index: int, now: float) -> None:
        record = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "status": event.status,
            "camera_id": event.camera_id,
            "zone_id": event.zone_id,
            "track_id": event.track_id,
            "frame_index": frame_index,
            "video_time_seconds": round(now, 3),
            "metrics": event.metrics,
            "thresholds": event.thresholds,
            "evidence_paths": event.evidence_paths,
        }
        if self.events_file is not None:
            self.events_file.write(
                json.dumps(record, ensure_ascii=False, default=_json_default) + "\n"
            )
            self.events_file.flush()
        self._sync_event_to_database(record)

    def _update_rule_event(
        self,
        *,
        event_key: str,
        event_type: str,
        zone_id: str | None,
        track_id: int | None,
        condition: bool,
        recovery_condition: bool,
        confirm_seconds: float,
        recovery_seconds: float,
        metrics: dict[str, Any],
        thresholds: dict[str, Any],
        now: float,
        frame_index: int,
        frame: np.ndarray,
        zone_polygon: np.ndarray,
    ) -> EventState | None:
        previous = self.events.get(event_key)
        # 已解决事件再次触发时会创建新事件，因此需要重新保存证据。
        was_triggered = (
            previous is not None
            and previous.status not in CLOSED_EVENT_STATUSES
            and previous.triggered_at is not None
        )
        event, transition = advance_event(
            previous,
            condition=condition,
            recovery_condition=recovery_condition,
            now=now,
            confirm_seconds=confirm_seconds,
            recovery_seconds=recovery_seconds,
            event_key=event_key,
            event_type=event_type,
            camera_id=self.camera_id,
            zone_id=zone_id,
            track_id=track_id,
            metrics=metrics,
            thresholds=thresholds,
        )
        if event is None:
            self.events.pop(event_key, None)
            return None
        self.events[event_key] = event
        if transition is not None:
            if transition == "ACTIVE" and not was_triggered:
                self._save_evidence(event, frame, zone_polygon)
            self._write_event(event, frame_index, now)
        return event

    def _evaluate_rules(
        self, frame: np.ndarray, frame_index: int, now: float, zone_polygon: np.ndarray
    ) -> None:
        tracking_cfg = self.config["tracking"]
        count_cfg = self.config["rules"]["count"]
        dwell_cfg = self.config["rules"]["dwell"]
        helmet_cfg = self.config["rules"]["helmet"]
        zone_id = str(self.config["zone"]["id"])
        lost_grace = float(tracking_cfg["lost_grace_seconds"])

        active_zone_ids = {
            track_id
            for track_id, state in self.track_states.items()
            if state.current_zone == zone_id
            and (
                state.missing_since is None or now - state.missing_since <= lost_grace
            )
        }
        self.zone_state.person_ids = active_zone_ids
        self.zone_state.person_count = len(active_zone_ids)
        self.zone_state.last_updated_at = now

        if count_cfg.get("enabled", True):
            max_people = int(count_cfg["max_people"])
            condition = self.zone_state.person_count > max_people
            count_event = self._update_rule_event(
                event_key=f"OVER_COUNT:{zone_id}",
                event_type="OVER_COUNT",
                zone_id=zone_id,
                track_id=None,
                condition=condition,
                recovery_condition=not condition,
                confirm_seconds=float(count_cfg["violation_confirm_seconds"]),
                recovery_seconds=float(count_cfg["recovery_confirm_seconds"]),
                metrics={"person_count": self.zone_state.person_count},
                thresholds={"max_people": max_people},
                now=now,
                frame_index=frame_index,
                frame=frame,
                zone_polygon=zone_polygon,
            )
            self.zone_state.active_count_event_id = (
                count_event.event_id
                if count_event is not None and count_event.status not in CLOSED_EVENT_STATUSES
                else None
            )
            self.zone_state.count_condition = (
                count_event.status if count_event is not None else "NORMAL"
            )

        for track_id, state in list(self.track_states.items()):
            if state.current_zone is not None and state.zone_enter_at is not None:
                state.zone_dwell_seconds = max(0.0, now - state.zone_enter_at)

            if dwell_cfg.get("enabled", True):
                max_seconds = float(dwell_cfg["max_seconds"])
                condition = (
                    state.current_zone == zone_id
                    and state.zone_dwell_seconds >= max_seconds
                )
                recovery = state.current_zone != zone_id
                dwell_event = self._update_rule_event(
                    event_key=f"DWELL:{zone_id}:{track_id}",
                    event_type="DWELL",
                    zone_id=zone_id,
                    track_id=track_id,
                    condition=condition,
                    recovery_condition=recovery,
                    confirm_seconds=0.0,
                    recovery_seconds=float(tracking_cfg["exit_confirm_seconds"]),
                    metrics={"dwell_seconds": round(state.zone_dwell_seconds, 3)},
                    thresholds={"max_seconds": max_seconds},
                    now=now,
                    frame_index=frame_index,
                    frame=frame,
                    zone_polygon=zone_polygon,
                )
                state.active_dwell_event_id = (
                    dwell_event.event_id
                    if dwell_event is not None and dwell_event.status not in CLOSED_EVENT_STATUSES
                    else None
                )

            if helmet_cfg.get("enabled", True):
                prune_helmet_history(state, now, float(helmet_cfg["window_seconds"]))
                visible_seconds = max(0.0, state.last_seen_at - state.first_seen_at)
                scope_matches = (
                    str(helmet_cfg.get("scope", "all")) == "all"
                    or state.current_zone == zone_id
                )
                eligible = scope_matches and (
                    visible_seconds >= float(helmet_cfg["min_person_visible_seconds"])
                    and state.recent_evaluable_frames
                    >= int(helmet_cfg["min_evaluable_frames"])
                )
                helmet_ratio = state.recent_helmet_ratio
                no_helmet_ratio = state.recent_no_helmet_ratio
                violation = eligible and (
                    (
                        helmet_ratio is not None
                        and helmet_ratio < float(helmet_cfg["violation_helmet_ratio"])
                    )
                    or (
                        no_helmet_ratio is not None
                        and no_helmet_ratio
                        >= float(helmet_cfg["violation_no_helmet_ratio"])
                    )
                )
                missing_recovery = (
                    state.missing_since is not None
                    and now - state.missing_since > lost_grace
                )
                recovery = (not scope_matches) or missing_recovery or (
                    helmet_ratio is not None
                    and helmet_ratio >= float(helmet_cfg["recovery_helmet_ratio"])
                )
                helmet_event = self._update_rule_event(
                    event_key=f"NO_HELMET:{track_id}",
                    event_type="NO_HELMET",
                    zone_id=state.current_zone,
                    track_id=track_id,
                    condition=violation,
                    recovery_condition=recovery,
                    confirm_seconds=float(helmet_cfg["violation_confirm_seconds"]),
                    recovery_seconds=float(helmet_cfg["recovery_confirm_seconds"]),
                    metrics={
                        "helmet_ratio": round(helmet_ratio, 4)
                        if helmet_ratio is not None
                        else None,
                        "no_helmet_ratio": round(no_helmet_ratio, 4)
                        if no_helmet_ratio is not None
                        else None,
                        "evaluable_frames": state.recent_evaluable_frames,
                        "visible_seconds": round(visible_seconds, 3),
                    },
                    thresholds={
                        "violation_helmet_ratio": helmet_cfg[
                            "violation_helmet_ratio"
                        ],
                        "violation_no_helmet_ratio": helmet_cfg[
                            "violation_no_helmet_ratio"
                        ],
                        "recovery_helmet_ratio": helmet_cfg[
                            "recovery_helmet_ratio"
                        ],
                    },
                    now=now,
                    frame_index=frame_index,
                    frame=frame,
                    zone_polygon=zone_polygon,
                )
                state.active_helmet_event_id = (
                    helmet_event.event_id
                    if helmet_event is not None and helmet_event.status not in CLOSED_EVENT_STATUSES
                    else None
                )
                if (
                    helmet_event is not None
                    and helmet_event.status in {"ACTIVE", "RECOVERING", "RESOLVED"}
                    and helmet_event.triggered_at is not None
                ):
                    if state.helmet_violation_at is None:
                        state.helmet_violation_at = helmet_event.triggered_at
                if helmet_event is not None and helmet_event.status in {
                    "PENDING",
                    "ACTIVE",
                    "RECOVERING",
                }:
                    state.ppe_status = (
                        "VIOLATION"
                        if helmet_event.status in {"ACTIVE", "RECOVERING"}
                        else "PENDING"
                    )
                elif eligible and recovery:
                    state.ppe_status = "COMPLIANT"
                elif eligible:
                    state.ppe_status = "UNCERTAIN"
                else:
                    state.ppe_status = "OBSERVING"

                accessory_cfg = self.config["rules"].get("ppe_accessories", {})
                minimum_visible = float(
                    accessory_cfg.get("min_person_visible_seconds", 3.0)
                )
                if visible_seconds >= minimum_visible:
                    if state.gloves_status != "WORN":
                        state.gloves_status = "NEEDS_REVIEW"
                    if state.goggles_status != "WORN":
                        state.goggles_status = "NEEDS_REVIEW"

        self._write_state_snapshot(frame_index, now)
        expire_seconds = float(tracking_cfg["state_expire_seconds"])
        expired_ids = [
            track_id
            for track_id, state in self.track_states.items()
            if state.missing_since is not None
            and now - state.missing_since > expire_seconds
        ]
        for track_id in expired_ids:
            self.completed_track_states[track_id] = self.track_states[track_id]
            del self.track_states[track_id]

    def _write_state_snapshot(self, frame_index: int, now: float) -> None:
        if self.states_file is None:
            return
        tracks = []
        for state in sorted(self.track_states.values(), key=lambda item: item.track_id):
            tracks.append(
                {
                    "track_id": state.track_id,
                    "last_seen_at": round(state.last_seen_at, 3),
                    "is_missing": state.is_missing,
                    "bbox": [round(value, 2) for value in state.bbox],
                    "current_zone": state.current_zone,
                    "zone_dwell_seconds": round(state.zone_dwell_seconds, 3),
                    "helmet_observation": state.helmet_observation,
                    "helmet_ratio": round(state.recent_helmet_ratio, 4)
                    if state.recent_helmet_ratio is not None
                    else None,
                    "no_helmet_ratio": round(state.recent_no_helmet_ratio, 4)
                    if state.recent_no_helmet_ratio is not None
                    else None,
                    "helmet_evaluable_frames": state.recent_evaluable_frames,
                    "ppe_status": state.ppe_status,
                    "gloves_status": state.gloves_status,
                    "goggles_status": state.goggles_status,
                }
            )
        record = {
            "camera_id": self.camera_id,
            "frame_index": frame_index,
            "video_time_seconds": round(now, 3),
            "zone": {
                "zone_id": self.zone_state.zone_id,
                "person_ids": sorted(self.zone_state.person_ids),
                "person_count": self.zone_state.person_count,
                "count_condition": self.zone_state.count_condition,
            },
            "tracks": tracks,
        }
        self.states_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.states_file.flush()

    def _finalize_ppe_inspection(self, frame_index: int, now: float) -> None:
        """将整段视频的所有人员聚合成唯一 PPE_INSPECTION 事件。"""

        if not self.source_sha256 or frame_index < 0:
            return
        all_states = {**self.completed_track_states, **self.track_states}
        accessory_cfg = self.config["rules"].get("ppe_accessories", {})
        minimum_visible = float(accessory_cfg.get("min_person_visible_seconds", 3.0))
        window_seconds = float(accessory_cfg.get("window_seconds", 3.0))
        hold_seconds = float(accessory_cfg.get("hit_hold_seconds", 0.5))
        min_frames = int(accessory_cfg.get("min_positive_frames", 2))
        min_duration = float(accessory_cfg.get("min_effective_seconds", 0.5))
        if not all_states:
            return

        # ByteTrack 在遮挡和转身时可能换 ID，也可能对同一人给出短暂重复框。
        # 合并只用于视频结束后的联合 PPE 复核，不改变实时人数和原始规则事件。
        parents = {track_id: track_id for track_id in all_states}

        def find(track_id: int) -> int:
            while parents[track_id] != track_id:
                parents[track_id] = parents[parents[track_id]]
                track_id = parents[track_id]
            return track_id

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[max(left_root, right_root)] = min(left_root, right_root)

        state_items = list(all_states.items())
        for index, (left_id, left) in enumerate(state_items):
            for right_id, right in state_items[index + 1 :]:
                merged = False
                earlier, later = (left, right)
                if left.first_seen_at > right.first_seen_at:
                    earlier, later = right, left
                gap = later.first_seen_at - earlier.last_seen_at
                # 相邻碎片的时间和空间位置均连续时，视为同一个人。
                if (
                    0.0 <= gap <= 1.5
                    and earlier.bbox
                    and later.first_bbox
                    and bbox_iou(earlier.bbox, later.first_bbox) >= 0.12
                ):
                    merged = True
                if not merged:
                    left_samples = left.bbox_history[
                        :: max(1, len(left.bbox_history) // 30)
                    ]
                    right_samples = right.bbox_history[
                        :: max(1, len(right.bbox_history) // 30)
                    ]
                    # 同时出现且高度重叠的短轨迹通常是同一人的重复检测。
                    merged = any(
                        abs(left_time - right_time) <= 0.15
                        and bbox_iou(left_bbox, right_bbox) >= 0.5
                        for left_time, left_bbox in left_samples
                        for right_time, right_bbox in right_samples
                    )
                if merged:
                    union(left_id, right_id)

        grouped: dict[int, list[TrackState]] = {}
        for track_id, state in all_states.items():
            grouped.setdefault(find(track_id), []).append(state)

        people: list[dict[str, Any]] = []
        group_anchors: list[tuple[float, float, int]] = []
        for _, states in sorted(grouped.items()):
            first_seen = min(state.first_seen_at for state in states)
            last_seen = max(state.last_seen_at for state in states)
            visible_seconds = max(0.0, last_seen - first_seen)
            all_boxes = [bbox for state in states for _, bbox in state.bbox_history]
            if not all_boxes:
                continue
            centers = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in all_boxes]
            heights = [max(1.0, box[3] - box[1]) for box in all_boxes]
            motion = math.hypot(
                max(point[0] for point in centers) - min(point[0] for point in centers),
                max(point[1] for point in centers) - min(point[1] for point in centers),
            ) / (sum(heights) / len(heights))
            best_state = max(states, key=lambda item: item.best_bbox_area)
            best_bbox = best_state.best_bbox or best_state.bbox
            best_aspect = (best_bbox[2] - best_bbox[0]) / max(
                1.0, best_bbox[3] - best_bbox[1]
            )
            has_positive_ppe = any(
                state.helmet_seen_worn_at is not None
                or state.first_no_helmet_at is not None
                or state.gloves_all_hits
                or state.goggles_all_hits
                for state in states
            )
            # 排除被 Person 类误识别的静态管道大框；真实移动人员仍可进入复核。
            credible_person = best_aspect <= 0.95 and (
                has_positive_ppe or motion >= 0.15
            )
            if not credible_person or visible_seconds < minimum_visible:
                continue

            worn_times = [
                state.helmet_seen_worn_at
                for state in states
                if state.helmet_seen_worn_at is not None
            ]
            last_worn_times = [
                state.last_helmet_worn_at
                for state in states
                if state.last_helmet_worn_at is not None
            ]
            violation_times = [
                state.helmet_violation_at
                for state in states
                if state.helmet_violation_at is not None
            ]
            no_helmet_times = [
                state.first_no_helmet_at
                for state in states
                if state.first_no_helmet_at is not None
            ]
            helmet_seen_at = min(worn_times) if worn_times else None
            helmet_last_worn_at = max(last_worn_times) if last_worn_times else None
            helmet_violation_at = min(violation_times) if violation_times else None
            first_no_helmet_at = min(no_helmet_times) if no_helmet_times else None
            if helmet_violation_at is not None:
                helmet_rule_status = (
                    "REMOVED_DURING_WORK"
                    if helmet_seen_at is not None
                    and helmet_seen_at < helmet_violation_at
                    else "NOT_WORN"
                )
            elif helmet_seen_at is not None:
                helmet_rule_status = "WORN"
            else:
                helmet_rule_status = "UNCERTAIN"

            member_samples = [sample for state in states for sample in state.bbox_history]

            def group_hits(kind: str) -> list[tuple[float, float]]:
                matched_hits: list[tuple[float, float]] = []
                for timestamp, item_bbox, confidence in self.raw_accessory_detections[kind]:
                    center_x = (item_bbox[0] + item_bbox[2]) / 2
                    center_y = (item_bbox[1] + item_bbox[3]) / 2
                    if any(
                        abs(timestamp - person_time) <= 0.15
                        and person_bbox[0] <= center_x <= person_bbox[2]
                        and person_bbox[1] <= center_y <= person_bbox[3]
                        for person_time, person_bbox in member_samples
                    ):
                        matched_hits.append((timestamp, confidence))
                return matched_hits

            glove_hits = group_hits("gloves")
            goggle_hits = group_hits("goggles")
            no_glove_hits = group_hits("no_gloves")
            no_goggle_hits = group_hits("no_goggle")
            gloves_worn, gloves_duration, gloves_confirmed_at = (
                accessory_evidence_from_hits(
                    glove_hits, window_seconds, hold_seconds, min_frames, min_duration
                )
            )
            goggles_worn, goggles_duration, goggles_confirmed_at = (
                accessory_evidence_from_hits(
                    goggle_hits, window_seconds, hold_seconds, min_frames, min_duration
                )
            )
            gloves_status = "WORN" if gloves_worn else "NEEDS_REVIEW"
            goggles_status = "WORN" if goggles_worn else "NEEDS_REVIEW"
            canonical_track_id = min(state.track_id for state in states)
            aliases = sorted(state.track_id for state in states)
            people.append(
                {
                    "track_id": canonical_track_id,
                    "track_ids": aliases,
                    "visible_seconds": round(visible_seconds, 3),
                    "helmet_rule_status": helmet_rule_status,
                    "helmet_seen_worn_at": helmet_seen_at,
                    "helmet_last_worn_at": helmet_last_worn_at,
                    "first_no_helmet_at": first_no_helmet_at,
                    "helmet_violation_at": helmet_violation_at,
                    "gloves_rule_status": gloves_status,
                    "goggles_rule_status": goggles_status,
                    "gloves_positive_frames": len(glove_hits),
                    "goggles_positive_frames": len(goggle_hits),
                    "no_gloves_positive_frames": len(no_glove_hits),
                    "no_goggle_positive_frames": len(no_goggle_hits),
                    "gloves_effective_seconds": round(gloves_duration, 3),
                    "goggles_effective_seconds": round(goggles_duration, 3),
                    "gloves_confirmed_at": gloves_confirmed_at,
                    "goggles_confirmed_at": goggles_confirmed_at,
                    "best_visibility_at": round(best_state.best_visibility_at, 3),
                }
            )
            # 摘帽时优先使用首次明确 no_helmet 画面，确保前置片段包含佩戴状态。
            anchor_time = float(
                first_no_helmet_at
                or helmet_violation_at
                or best_state.best_visibility_at
            )
            group_anchors.append((anchor_time, -best_state.best_bbox_area, canonical_track_id))

        has_candidate = any(
            person["helmet_rule_status"]
            in {"NOT_WORN", "REMOVED_DURING_WORK", "UNCERTAIN"}
            or person["gloves_rule_status"] == "NEEDS_REVIEW"
            or person["goggles_rule_status"] == "NEEDS_REVIEW"
            for person in people
        )
        if not people or not has_candidate:
            return

        anchor_time, _, anchor_track_id = min(group_anchors)

        event_id = f"PPE_{self.source_sha256[:32]}"
        event = EventState(
            event_id=event_id,
            event_key=f"PPE_INSPECTION:{self.source_sha256}",
            event_type="PPE_INSPECTION",
            camera_id=self.camera_id,
            zone_id=str(self.config["zone"]["id"]),
            track_id=anchor_track_id,
            status="ACTIVE",
            pending_since=anchor_time,
            triggered_at=anchor_time,
            last_updated_at=now,
            metrics={
                "source_sha256": self.source_sha256,
                "anchor_track_id": anchor_track_id,
                "anchor_time_seconds": round(anchor_time, 3),
                "people": people,
            },
            thresholds={
                "window_seconds": window_seconds,
                "hit_hold_seconds": hold_seconds,
                "min_positive_frames": min_frames,
                "min_effective_seconds": min_duration,
            },
        )
        self.events[event.event_key] = event
        anchor_frame = int(round(anchor_time * float(self.run_fps or 1.0)))
        self._write_event(event, min(max(anchor_frame, 0), frame_index), anchor_time)

    def _font(self, size: int) -> ImageFont.FreeTypeFont:
        if size not in self._font_cache:
            font_path = resolve_config_path(
                self.config_dir, self.config["display"]["font_path"]
            )
            self._font_cache[size] = ImageFont.truetype(str(font_path), size)
        return self._font_cache[size]

    def _draw_unicode(
        self,
        image: np.ndarray,
        texts: list[tuple[tuple[int, int], str, tuple[int, int, int], int]],
    ) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_image)
        for (x, y), text, bgr_color, size in texts:
            draw.text((x, y), text, font=self._font(size), fill=bgr_color[::-1])
        return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

    def _track_color(self, state: TrackState) -> tuple[int, int, int]:
        active_types = {
            event.event_type
            for event in self.events.values()
            if event.track_id == state.track_id and event.status in {"ACTIVE", "RECOVERING"}
        }
        if active_types:
            return 40, 40, 230
        if state.ppe_status in {"PENDING", "UNCERTAIN"}:
            return 0, 210, 255
        if state.ppe_status == "COMPLIANT":
            return 40, 200, 60
        return 220, 170, 40

    def _render(
        self,
        frame: np.ndarray,
        persons: list[Detection],
        helmets: list[Detection],
        no_helmets: list[Detection],
        associations: dict[int, dict[str, Detection | None]],
        zone_polygon: np.ndarray,
        now: float,
    ) -> np.ndarray:
        height, width = frame.shape[:2]
        panel_width = int(self.config["display"]["panel_width"])
        canvas = np.zeros((height, width + panel_width, 3), dtype=np.uint8)
        canvas[:, :width] = frame
        canvas[:, width:] = (28, 30, 34)

        overlay = canvas[:, :width].copy()
        cv2.fillPoly(overlay, [zone_polygon], (180, 120, 20))
        canvas[:, :width] = cv2.addWeighted(overlay, 0.12, canvas[:, :width], 0.88, 0)
        zone_color = (0, 0, 230) if self.zone_state.count_condition == "ACTIVE" else (220, 180, 30)
        cv2.polylines(canvas, [zone_polygon], True, zone_color, 2)

        debug_mode = self.config["display"].get("mode", "debug") == "debug"
        texts: list[tuple[tuple[int, int], str, tuple[int, int, int], int]] = []

        if debug_mode:
            for item in helmets:
                x1, y1, x2, y2 = (int(value) for value in item.bbox)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 220, 40), 2)
                cv2.circle(canvas, tuple(int(v) for v in item.center), 4, (40, 220, 40), -1)
            for item in no_helmets:
                x1, y1, x2, y2 = (int(value) for value in item.bbox)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (30, 30, 230), 2)
                cv2.circle(canvas, tuple(int(v) for v in item.center), 4, (30, 30, 230), -1)

        ppe_names = {
            "COMPLIANT": "安全帽正常",
            "VIOLATION": "未规范佩戴",
            "PENDING": "违规待确认",
            "UNCERTAIN": "无法确定",
            "OBSERVING": "观察中",
        }
        for person in persons:
            if person.track_id is None or person.track_id not in self.track_states:
                continue
            state = self.track_states[person.track_id]
            color = self._track_color(state)
            x1, y1, x2, y2 = (int(value) for value in state.bbox)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.circle(canvas, tuple(int(v) for v in state.foot_point), 5, color, -1)

            if debug_mode:
                hx1, hy1, hx2, hy2 = (int(value) for value in state.head_roi)
                cv2.rectangle(canvas, (hx1, hy1), (hx2, hy2), (0, 210, 255), 1)
                matched = associations.get(state.track_id, {})
                for key, line_color in (("helmet", (40, 220, 40)), ("no_helmet", (30, 30, 230))):
                    item = matched.get(key)
                    if item is not None:
                        cv2.line(
                            canvas,
                            ((hx1 + hx2) // 2, (hy1 + hy2) // 2),
                            tuple(int(value) for value in item.center),
                            line_color,
                            1,
                        )

            zone_name = "作业区" if state.current_zone == self.zone_state.zone_id else "区域外"
            first_line = f"ID {state.track_id} | {zone_name} | {state.zone_dwell_seconds:.1f}s"
            second_line = ppe_names.get(state.ppe_status, state.ppe_status)
            if debug_mode:
                ratio = (
                    f"{state.recent_helmet_ratio:.0%}"
                    if state.recent_helmet_ratio is not None
                    else "--"
                )
                second_line += f" | 帽率 {ratio} ({state.recent_evaluable_frames})"
            label_y = max(2, y1 - 48)
            texts.append(((x1, label_y), first_line, color, 16))
            texts.append(((x1, label_y + 21), second_line, color, 16))

        zone_name = str(self.config["zone"]["name"])
        max_people = self.config["rules"]["count"]["max_people"]
        panel_x = width + 18
        texts.extend(
            [
                ((panel_x, 16), "单摄像头安防监控", (245, 245, 245), 22),
                ((panel_x, 54), f"摄像头：{self.camera_id}", (220, 220, 220), 17),
                ((panel_x, 80), f"视频时间：{now:06.2f}s", (220, 220, 220), 17),
                ((panel_x, 118), zone_name, (255, 210, 80), 20),
                (
                    (panel_x, 150),
                    f"稳定人数：{self.zone_state.person_count} / {max_people}",
                    (230, 230, 230),
                    18,
                ),
                (
                    (panel_x, 178),
                    f"人数规则：{self.zone_state.count_condition}",
                    (230, 230, 230),
                    17,
                ),
                ((panel_x, 218), "当前人员", (255, 210, 80), 20),
            ]
        )
        y = 250
        current_tracks = [state for state in self.track_states.values() if not state.is_missing]
        current_tracks.sort(
            key=lambda state: (
                state.ppe_status not in {"VIOLATION", "PENDING"},
                -state.zone_dwell_seconds,
            )
        )
        for state in current_tracks[:7]:
            status = ppe_names.get(state.ppe_status, state.ppe_status)
            texts.append(
                ((panel_x, y), f"ID {state.track_id}  {state.zone_dwell_seconds:4.1f}s  {status}", self._track_color(state), 16)
            )
            y += 25

        y = max(y + 18, 445)
        texts.append(((panel_x, y), "当前事件", (255, 210, 80), 20))
        y += 34
        active_events = [
            event
            for event in self.events.values()
            if event.status in {"PENDING", "ACTIVE", "RECOVERING"}
        ]
        if not active_events:
            texts.append(((panel_x, y), "当前无异常事件", (160, 200, 160), 16))
        else:
            event_names = {
                "NO_HELMET": "未规范佩戴安全帽",
                "DWELL": "人员滞留",
                "OVER_COUNT": "作业区超员",
            }
            for event in active_events[:5]:
                subject = f" ID {event.track_id}" if event.track_id is not None else ""
                texts.append(
                    (
                        (panel_x, y),
                        f"[{event.status}] {event_names[event.event_type]}{subject}",
                        (60, 80, 240),
                        15,
                    )
                )
                y += 24

        texts.append(
            (
                (int(zone_polygon[0][0]), max(2, int(zone_polygon[0][1]) - 26)),
                f"{zone_name}  人数 {self.zone_state.person_count}/{max_people}",
                zone_color,
                18,
            )
        )
        return self._draw_unicode(canvas, texts)

    def run(self) -> Path:
        cap, model, fps, _, _, zone_polygon = self._prepare()
        frame_index = -1
        last_video_time = -1.0
        run_status = "COMPLETED"
        run_error: str | None = None
        try:
            class_ids = self._resolve_class_ids(model)
            latest_raw_accessories: dict[str, list[Detection]] = {
                "gloves": [],
                "goggles": [],
                "no_gloves": [],
                "no_goggle": [],
            }

            def capture_raw_accessories(predictor: Any) -> None:
                """在 ByteTrack 过滤短时小目标前保留同一次 YOLO 推理的 PPE 框。"""

                for values in latest_raw_accessories.values():
                    values.clear()
                if not predictor.results:
                    return
                raw_result = predictor.results[0]
                if raw_result.boxes is None:
                    return
                for bbox, class_id, confidence in zip(
                    raw_result.boxes.xyxy.cpu().tolist(),
                    raw_result.boxes.cls.int().cpu().tolist(),
                    raw_result.boxes.conf.cpu().tolist(),
                ):
                    for kind in latest_raw_accessories:
                        if class_id == class_ids[kind]:
                            latest_raw_accessories[kind].append(
                                Detection(
                                    class_id=class_id,
                                    bbox=tuple(float(value) for value in bbox),
                                    confidence=float(confidence),
                                )
                            )
                            break

            # 回调注册早于 Ultralytics 的 tracker 回调，因此取得的是原始后处理框，
            # 没有增加第二次模型推理。
            model.add_callback("on_predict_postprocess_end", capture_raw_accessories)
            selected_classes = [
                class_ids["helmet"],
                class_ids["person"],
                class_ids["no_helmet"],
                class_ids["gloves"],
                class_ids["no_gloves"],
                class_ids["goggles"],
                class_ids["no_goggle"],
            ]
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                frame_index += 1
                self.processed_frames = frame_index + 1
                pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                fallback_time = frame_index / fps
                now = pos_msec / 1000.0 if pos_msec > 0 else fallback_time
                if now <= last_video_time:
                    now = fallback_time
                last_video_time = now

                configured_device = self.config["model"].get("device", "auto")
                if str(configured_device).lower() == "auto":
                    import torch

                    configured_device = 0 if torch.cuda.is_available() else "cpu"
                result = model.track(
                    frame,
                    persist=True,
                    tracker=str(self.config["model"]["tracker"]),
                    classes=selected_classes,
                    conf=float(self.config["model"]["confidence"]),
                    imgsz=int(self.config["model"]["image_size"]),
                    device=configured_device,
                    verbose=False,
                )[0]
                detections = self._split_detections(result, class_ids)
                for kind in ("gloves", "goggles", "no_gloves", "no_goggle"):
                    self.raw_accessory_detections[kind].extend(
                        (now, item.bbox, item.confidence)
                        for item in latest_raw_accessories[kind]
                    )
                person_confidence = float(
                    self.config["model"].get(
                        "person_confidence", self.config["model"]["confidence"]
                    )
                )
                # Person 使用独立高阈值；小目标 PPE 类别继续保留全局低阈值结果。
                persons = [
                    item
                    for item in detections["person"]
                    if item.confidence >= person_confidence
                ]
                helmets = detections["helmet"]
                no_helmets = detections["no_helmet"]
                associations = associate_ppe(
                    persons,
                    helmets,
                    no_helmets,
                    float(self.config["rules"]["helmet"]["head_height_ratio"]),
                    detections["gloves"],
                    detections["no_gloves"],
                    detections["goggles"],
                    detections["no_goggle"],
                )
                self._update_tracks(persons, associations, now, zone_polygon)

                if now + 1e-6 >= self.next_rule_time:
                    self._evaluate_rules(frame, frame_index, now, zone_polygon)
                    while self.next_rule_time <= now:
                        self.next_rule_time += self.rule_interval

                rendered = self._render(
                    frame,
                    persons,
                    helmets,
                    no_helmets,
                    associations,
                    zone_polygon,
                    now,
                )
                if self.video_writer is not None:
                    self.video_writer.write(rendered)
                if self.show:
                    cv2.imshow("YOLO Security Monitor", rendered)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("用户按下 q，提前结束处理。")
                        run_status = "STOPPED"
                        break
            self._finalize_ppe_inspection(frame_index, max(last_video_time, 0.0))
        except Exception as exc:
            run_status = "FAILED"
            run_error = str(exc)
            raise
        finally:
            cap.release()
            if self.video_writer is not None:
                self.video_writer.release()
            if self.events_file is not None:
                self.events_file.close()
            if self.states_file is not None:
                self.states_file.close()
            if self.show:
                cv2.destroyAllWindows()
            self._finish_database(run_status, run_error)

        assert self.output_dir is not None
        return self.output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单摄像头 YOLO 安防规则 MVP")
    parser.add_argument("--config", default="config.yaml", help="YAML 配置文件路径")
    parser.add_argument("--source", help="覆盖配置中的本地视频路径")
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖配置中的窗口显示开关，可使用 --show 或 --no-show",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    monitor = SecurityMonitor(config, config_path, args.source, args.show)
    output_dir = monitor.run()
    print(f"处理完成，输出目录: {output_dir}")
    if config.get("review", {}).get("enabled", False):
        # 检测完成后自动处理本次运行的复核任务；独立复核命令仍可用于历史补录。
        from .review import review_run_directory

        review_run_directory(config_path, output_dir, monitor.source_path)


if __name__ == "__main__":
    main()
