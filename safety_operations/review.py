from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import cv2
import yaml
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .db import (
    apply_review_decision,
    append_sync_error,
    connect_database,
    sync_run_directory,
    upsert_llm_review,
)
from .env import load_project_env
from .video import create_browser_video_writer


PROMPT_VERSION = "no_helmet_v2"
REVIEW_FUNCTION_NAME = "submit_no_helmet_review"


class ReviewResult(BaseModel):
    """豆包复核的结构化视觉结论。"""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["CONFIRMED", "REJECTED", "UNCERTAIN"]
    target_visible: Literal["CLEAR", "PARTIAL", "NOT_VISIBLE"]
    helmet_status: Literal["WORN", "NOT_WORN", "UNCERTAIN"]
    evidence_quality: Literal["GOOD", "LIMITED", "POOR"]
    visual_reason: str = Field(min_length=1, max_length=300)
    evidence_timestamps: list[float] = Field(default_factory=list)

    @field_validator("evidence_timestamps")
    @classmethod
    def timestamps_must_be_non_negative(cls, values: list[float]) -> list[float]:
        if any(value < 0 for value in values):
            raise ValueError("证据时间戳不能为负数")
        return values


class ReviewError(RuntimeError):
    """携带稳定错误码的复核异常，便于写入 JSONL。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def load_yaml(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("配置文件内容必须是 YAML 对象。")
    if "review" not in config:
        raise ValueError("配置文件缺少 review 段。")
    return config


def resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL 文件不存在: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON。") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path} 第 {line_number} 行必须是 JSON 对象。")
            records.append(value)
    return records


def select_active_events(
    records: list[dict[str, Any]], event_types: set[str]
) -> list[dict[str, Any]]:
    """只保留每个事件首次进入 ACTIVE 时的记录。"""

    selected: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    for record in records:
        event_id = str(record.get("event_id", ""))
        if (
            event_id
            and record.get("status") == "ACTIVE"
            and record.get("event_type") in event_types
            and event_id not in seen_event_ids
        ):
            selected.append(record)
            seen_event_ids.add(event_id)
    return selected


def load_review_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str, str]] = set()
    for record in read_jsonl(path):
        event_id = record.get("event_id")
        model = record.get("model")
        prompt_version = record.get("prompt_version")
        if event_id and model and prompt_version:
            keys.add((str(event_id), str(model), str(prompt_version)))
    return keys


def load_track_bbox_timeline(
    states_path: Path, track_id: int
) -> list[tuple[float, tuple[float, float, float, float]]]:
    """从每秒状态快照中读取目标轨迹框，用于生成中性标注视频。"""

    timeline: list[tuple[float, tuple[float, float, float, float]]] = []
    for snapshot in read_jsonl(states_path):
        timestamp = float(snapshot.get("video_time_seconds", 0.0))
        for track in snapshot.get("tracks", []):
            if int(track.get("track_id", -1)) != track_id or track.get("is_missing"):
                continue
            bbox = track.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                timeline.append((timestamp, tuple(float(value) for value in bbox)))
            break
    if not timeline:
        raise ReviewError(
            "MISSING_TARGET_TRACK_HISTORY",
            f"states.jsonl 中没有 Track {track_id} 的有效人物框。",
        )
    return timeline


def interpolate_target_bbox(
    timeline: list[tuple[float, tuple[float, float, float, float]]],
    timestamp: float,
    maximum_distance_seconds: float = 1.1,
) -> tuple[float, float, float, float] | None:
    """在相邻状态快照之间线性插值；距离过远时不绘制，避免框错人。"""

    times = [item[0] for item in timeline]
    index = bisect.bisect_left(times, timestamp)
    if index < len(timeline) and abs(times[index] - timestamp) < 1e-6:
        return timeline[index][1]
    if 0 < index < len(timeline):
        before_time, before_bbox = timeline[index - 1]
        after_time, after_bbox = timeline[index]
        if (
            timestamp - before_time <= maximum_distance_seconds
            and after_time - timestamp <= maximum_distance_seconds
        ):
            ratio = (timestamp - before_time) / (after_time - before_time)
            return tuple(
                before + (after - before) * ratio
                for before, after in zip(before_bbox, after_bbox)
            )
    candidates = []
    if index > 0:
        candidates.append(timeline[index - 1])
    if index < len(timeline):
        candidates.append(timeline[index])
    if not candidates:
        return None
    nearest_time, nearest_bbox = min(candidates, key=lambda item: abs(item[0] - timestamp))
    return (
        nearest_bbox
        if abs(nearest_time - timestamp) <= maximum_distance_seconds
        else None
    )


def create_review_clip(
    source_path: Path,
    output_path: Path,
    trigger_seconds: float,
    pre_seconds: float,
    post_seconds: float,
    output_fps: float,
    max_width: int,
    minimum_seconds: float = 2.0,
    target_bbox_timeline: list[
        tuple[float, tuple[float, float, float, float]]
    ]
    | None = None,
) -> dict[str, Any]:
    """从原视频顺序采样事件片段，避免依赖本机 FFmpeg。"""

    if not source_path.exists():
        raise ReviewError("SOURCE_NOT_FOUND", f"原视频不存在: {source_path}")
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise ReviewError("SOURCE_OPEN_FAILED", f"无法打开原视频: {source_path}")

    writer: cv2.VideoWriter | None = None
    try:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if source_fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise ReviewError("INVALID_VIDEO_METADATA", "原视频 FPS、帧数或分辨率无效。")
        if output_fps <= 0:
            raise ReviewError("INVALID_CLIP_FPS", "review.clip_fps 必须大于 0。")

        duration = frame_count / source_fps
        start_seconds = max(0.0, trigger_seconds - pre_seconds)
        end_seconds = min(duration, trigger_seconds + post_seconds)
        actual_seconds = max(0.0, end_seconds - start_seconds)
        if actual_seconds + 1e-6 < minimum_seconds:
            raise ReviewError(
                "CLIP_TOO_SHORT",
                f"事件可用视频仅 {actual_seconds:.3f} 秒，少于 {minimum_seconds:.3f} 秒。",
            )

        scale = min(1.0, max_width / max(width, height))
        output_width = max(2, int(round(width * scale)))
        output_height = max(2, int(round(height * scale)))
        # 部分编码器要求偶数尺寸。
        output_width -= output_width % 2
        output_height -= output_height % 2
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            writer = create_browser_video_writer(
                output_path, output_fps, (output_width, output_height)
            )
        except RuntimeError as exc:
            raise ReviewError("CLIP_WRITER_FAILED", str(exc)) from exc

        start_frame = max(0, int(start_seconds * source_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        next_sample_seconds = start_seconds
        written_frames = 0
        frame_index = start_frame
        while frame_index < frame_count:
            success, frame = cap.read()
            if not success:
                break
            frame_seconds = frame_index / source_fps
            if frame_seconds >= end_seconds:
                break
            if frame_seconds + 1e-6 >= next_sample_seconds:
                target_bbox = (
                    interpolate_target_bbox(target_bbox_timeline, frame_seconds)
                    if target_bbox_timeline
                    else None
                )
                if (output_width, output_height) != (width, height):
                    frame = cv2.resize(
                        frame, (output_width, output_height), interpolation=cv2.INTER_AREA
                    )
                if target_bbox is not None:
                    x1, y1, x2, y2 = (
                        int(round(value * scale)) for value in target_bbox
                    )
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(output_width - 1, x2), min(output_height - 1, y2)
                    # 仅使用中性目标标记，不向模型泄露安全帽规则结论。
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    label_y = min(output_height - 6, y2 + 22)
                    if label_y <= y2 + 5:
                        label_y = max(18, y2 - 6)
                    cv2.putText(
                        frame,
                        "TARGET",
                        (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                writer.write(frame)
                written_frames += 1
                next_sample_seconds += 1.0 / output_fps
            frame_index += 1

        if written_frames == 0:
            raise ReviewError("CLIP_EMPTY", "复核视频没有写入任何帧。")
        return {
            "start_seconds": round(start_seconds, 3),
            "end_seconds": round(end_seconds, 3),
            "duration_seconds": round(written_frames / output_fps, 3),
            "fps": output_fps,
            "frame_count": written_frames,
            "width": output_width,
            "height": output_height,
        }
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def build_prompt(event: dict[str, Any], clip_metadata: dict[str, Any]) -> str:
    facts = {
        "event_id": event.get("event_id"),
        "event_type": event.get("event_type"),
        "track_id": event.get("track_id"),
        "camera_id": event.get("camera_id"),
        "zone_id": event.get("zone_id"),
        "activation_video_time_seconds": event.get("video_time_seconds"),
        "rule_metrics": event.get("metrics", {}),
        "rule_thresholds": event.get("thresholds", {}),
        "clip_start_seconds": clip_metadata["start_seconds"],
        "clip_duration_seconds": clip_metadata["duration_seconds"],
    }
    return (
        "你是工业现场安全帽事件复核助手。事件短视频中使用洋红色矩形和 TARGET 标记了唯一目标人员，"
        "只判断该目标人员，不判断其他人员。该标记只表示复核对象，不代表安全帽结论。\n"
        "程序提供的轨迹ID、时间、帧数、比例和阈值是确定事实，不得重新估算或修改。\n"
        "只有安全帽正确戴在目标人员头顶才算佩戴；拿在手里、夹在腋下、挂在身体上或位于附近均算未佩戴。\n"
        "如果目标身份无法从标注图对应到视频，或头部过小、被遮挡、出画，必须返回 UNCERTAIN。\n"
        "evidence_timestamps 使用相对短视频开头的秒数。visual_reason 只描述视觉证据，不编造规则数值。\n"
        "事件事实如下：\n"
        + json.dumps(facts, ensure_ascii=False, indent=2)
    )


def review_tool_schema() -> dict[str, Any]:
    schema = ReviewResult.model_json_schema()
    schema.pop("title", None)
    return {
        "type": "function",
        "name": REVIEW_FUNCTION_NAME,
        "description": "提交指定人员的安全帽视频复核结论。",
        "parameters": schema,
    }


def parse_review_response(response: Any) -> ReviewResult:
    for item in getattr(response, "output", []):
        if (
            getattr(item, "type", None) == "function_call"
            and getattr(item, "name", None) == REVIEW_FUNCTION_NAME
        ):
            return ReviewResult.model_validate_json(item.arguments)

    text = getattr(response, "output_text", "") or ""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1])
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:].lstrip()
    if not text:
        raise ValueError("豆包响应中没有 Function Call 或 JSON 文本。")
    return ReviewResult.model_validate_json(text)


def is_retryable_api_error(exc: Exception) -> bool:
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return True
    return isinstance(exc, APIStatusError) and exc.status_code in {429, 500, 502, 503, 504}


def extract_usage(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return dict(usage) if isinstance(usage, dict) else None


def call_ark_review(
    client: OpenAI,
    model: str,
    prompt: str,
    video_path: Path,
    max_output_tokens: int,
    retry_count: int,
    clip_duration_seconds: float,
    delete_remote_files: bool,
    video_preprocess_fps: float = 2.0,
    file_processing_timeout_seconds: float = 120.0,
) -> tuple[ReviewResult, str | None, dict[str, Any] | None]:
    uploaded_ids: list[str] = []
    try:
        for index, path in enumerate((video_path,)):
            with path.open("rb") as file:
                upload_kwargs: dict[str, Any] = {
                    "file": file,
                    "purpose": "user_data",
                }
                if index == 0:
                    # 方舟会先按指定 FPS 预处理视频，完成后才能用于 input_video。
                    upload_kwargs["extra_body"] = {
                        "preprocess_configs": {
                            "video": {"fps": video_preprocess_fps}
                        }
                    }
                uploaded = client.files.create(**upload_kwargs)
            uploaded_ids.append(uploaded.id)

            deadline = time.monotonic() + file_processing_timeout_seconds
            while str(getattr(uploaded, "status", "")).lower() == "processing":
                if time.monotonic() >= deadline:
                    raise ReviewError(
                        "FILE_PROCESSING_TIMEOUT",
                        f"文件 {path.name} 预处理超过 {file_processing_timeout_seconds:.0f} 秒。",
                    )
                time.sleep(2.0)
                uploaded = client.files.retrieve(uploaded.id)
            status = str(getattr(uploaded, "status", "")).lower()
            if status in {"error", "failed", "cancelled"}:
                raise ReviewError(
                    "FILE_PROCESSING_FAILED",
                    f"文件 {path.name} 预处理失败，状态为 {status}。",
                )

        video_id = uploaded_ids[0]
        last_error: Exception | None = None
        attempts = retry_count + 1
        for attempt in range(attempts):
            strict_json_fallback = attempt > 0
            request_prompt = prompt
            kwargs: dict[str, Any] = {}
            if strict_json_fallback:
                request_prompt += (
                    "\n上一次响应未通过结构校验。本次不要调用工具，只返回符合以下 JSON Schema 的 JSON 对象：\n"
                    + json.dumps(ReviewResult.model_json_schema(), ensure_ascii=False)
                )
            else:
                kwargs["tools"] = [review_tool_schema()]
                kwargs["tool_choice"] = {
                    "type": "function",
                    "name": REVIEW_FUNCTION_NAME,
                }

            try:
                response = client.responses.create(
                    model=model,
                    store=False,
                    input=[
                        {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": request_prompt},
                                {"type": "input_video", "file_id": video_id},
                            ],
                        }
                    ],
                    max_output_tokens=max_output_tokens,
                    extra_body={"thinking": {"type": "disabled"}},
                    **kwargs,
                )
                result = parse_review_response(response)
                if any(
                    timestamp > clip_duration_seconds + 0.5
                    for timestamp in result.evidence_timestamps
                ):
                    raise ValueError("证据时间戳超出复核短视频时长。")
                return result, getattr(response, "id", None), extract_usage(response)
            except (ValidationError, ValueError) as exc:
                last_error = exc
            except Exception as exc:  # SDK 的具体网络异常由下方函数判定是否重试。
                last_error = exc
                if not is_retryable_api_error(exc):
                    raise
            if attempt + 1 < attempts:
                time.sleep(1.0)

        assert last_error is not None
        raise ReviewError("INVALID_MODEL_RESPONSE", str(last_error)) from last_error
    finally:
        if delete_remote_files:
            for file_id in uploaded_ids:
                try:
                    client.files.delete(file_id)
                except Exception as exc:
                    print(f"警告：远端临时文件 {file_id} 删除失败: {exc}", file=sys.stderr)


def build_explanation(event: dict[str, Any], result: ReviewResult) -> str:
    metrics = event.get("metrics", {})
    track_id = event.get("track_id")
    facts: list[str] = [f"规则检测到 Track {track_id} 疑似未规范佩戴安全帽"]
    if metrics.get("helmet_ratio") is not None:
        facts.append(f"近期安全帽检出比例为 {float(metrics['helmet_ratio']):.1%}")
    if metrics.get("evaluable_frames") is not None:
        facts.append(f"有效观测 {int(metrics['evaluable_frames'])} 帧")
    if metrics.get("visible_seconds") is not None:
        facts.append(f"人员已出现 {float(metrics['visible_seconds']):.1f} 秒")
    decision_text = {
        "CONFIRMED": "豆包视频复核确认该视觉证据支持未佩戴安全帽",
        "REJECTED": "豆包视频复核未支持该违规，观察到目标已佩戴安全帽",
        "UNCERTAIN": "豆包视频复核认为当前证据不足，无法确认是否佩戴安全帽",
    }[result.decision]
    return "，".join(facts) + f"。{decision_text}：{result.visual_reason}。"


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()


def make_base_record(
    event: dict[str, Any],
    model: str,
    prompt_version: str,
    input_paths: dict[str, str],
    provider: str | None = None,
    prompt_text: str | None = None,
) -> dict[str, Any]:
    return {
        "attempt_id": uuid.uuid4().hex,
        "event_id": event.get("event_id"),
        "event_type": event.get("event_type"),
        "camera_id": event.get("camera_id"),
        "zone_id": event.get("zone_id"),
        "track_id": event.get("track_id"),
        "rule_status": event.get("status"),
        "rule_metrics": event.get("metrics", {}),
        "rule_thresholds": event.get("thresholds", {}),
        "mode": "active",
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
        "prompt_text": prompt_text,
        "reviewed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_paths": input_paths,
    }


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    # uv/Python 不会自动读取 .env；复核入口需要显式加载方舟模型与密钥配置。
    load_project_env()
    config = load_yaml(config_path)
    review_cfg = config["review"]
    if not review_cfg.get("enabled", True):
        print("review.enabled=false，未执行豆包复核。")
        return 0
    if review_cfg.get("mode") not in {"shadow", "active"}:
        raise ValueError("review.mode 仅支持 shadow 或 active。")

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"运行目录不存在: {run_dir}")
    events = read_jsonl(run_dir / "events.jsonl")
    source_value = args.source or config.get("camera", {}).get("source")
    source_path = (
        resolve_path(config_path.parent, str(source_value)) if source_value else None
    )

    # 兼容数据库功能上线前的历史运行目录，启动复核时先幂等补录事件。
    db_connection = None
    database_cfg = config.get("database", {})
    if database_cfg.get("enabled", True) and database_cfg.get("auto_sync", True):
        try:
            database_path = resolve_path(
                config_path.parent, database_cfg.get("path", "data/security.db")
            )
            db_connection = connect_database(
                database_path, int(database_cfg.get("busy_timeout_ms", 5000))
            )
            sync_run_directory(db_connection, config, run_dir, source_path)
        except Exception as exc:
            if db_connection is not None:
                db_connection.close()
                db_connection = None
            append_sync_error(run_dir, "SYNC_RUN_FOR_REVIEW", run_dir.name, exc)
            print(
                f"SQLite 同步失败，复核仍继续写 reviews.jsonl: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    event_types = {str(value) for value in review_cfg.get("event_types", ["NO_HELMET"])}
    active_events = select_active_events(events, event_types)
    if not active_events:
        if db_connection is not None:
            db_connection.close()
        print("没有需要复核的 ACTIVE NO_HELMET 事件。")
        return 0

    model_env = str(review_cfg.get("model_env", "ARK_MODEL_ID"))
    api_key_env = str(review_cfg.get("api_key_env", "ARK_API_KEY"))
    # 本地调试允许直接写在 YAML；未填写时再回退到环境变量。
    model = str(review_cfg.get("model") or "").strip() or os.getenv(
        model_env, ""
    ).strip()
    if not model and not args.prepare_only:
        raise RuntimeError(f"缺少环境变量 {model_env}。")
    idempotency_model = model or f"ENV:{model_env}"
    prompt_version = str(review_cfg.get("prompt_version", PROMPT_VERSION))
    reviews_path = run_dir / "reviews.jsonl"
    reviewed_keys = load_review_keys(reviews_path)
    pending_events = [
        event
        for event in active_events
        if args.force
        or (str(event["event_id"]), idempotency_model, prompt_version) not in reviewed_keys
    ]
    if not pending_events:
        if db_connection is not None:
            db_connection.close()
        print("所有匹配事件均已复核；使用 --force 可重新执行。")
        return 0

    api_key = str(review_cfg.get("api_key") or "").strip() or os.getenv(
        api_key_env, ""
    ).strip()
    if not api_key and not args.prepare_only:
        raise RuntimeError(f"缺少环境变量 {api_key_env}。")

    if source_path is None:
        raise ValueError("未通过 --source 或 camera.source 指定原视频。")
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    client = None
    if not args.prepare_only:
        client = OpenAI(
            base_url=str(review_cfg["base_url"]),
            api_key=api_key,
            timeout=float(review_cfg.get("timeout_seconds", 60)),
        )

    failures = 0
    provider = str(review_cfg.get("provider", "volcengine_ark"))
    for event in pending_events:
        event_id = str(event["event_id"])
        clip_path = (evidence_dir / f"{event_id}_review.mp4").resolve()
        input_paths: dict[str, str] = {
            "source": str(source_path),
            "clip": str(clip_path),
        }
        started = time.perf_counter()
        prompt_text: str | None = None
        try:
            track_id = event.get("track_id")
            if track_id is None:
                raise ReviewError(
                    "MISSING_TARGET_TRACK_ID", "NO_HELMET 事件缺少 track_id。"
                )
            target_timeline = load_track_bbox_timeline(
                run_dir / "states.jsonl", int(track_id)
            )
            clip_metadata = create_review_clip(
                source_path=source_path,
                output_path=clip_path,
                trigger_seconds=float(event["video_time_seconds"]),
                pre_seconds=float(review_cfg.get("pre_seconds", 3.0)),
                post_seconds=float(review_cfg.get("post_seconds", 2.0)),
                output_fps=float(review_cfg.get("clip_fps", 5.0)),
                max_width=int(review_cfg.get("max_clip_width", 1280)),
                target_bbox_timeline=target_timeline,
            )
            if args.prepare_only:
                print(
                    f"已准备事件 {event_id}: {clip_path} "
                    f"({clip_metadata['duration_seconds']}s)"
                )
                continue

            assert client is not None
            prompt_text = build_prompt(event, clip_metadata)
            result, response_id, usage = call_ark_review(
                client=client,
                model=model,
                prompt=prompt_text,
                video_path=clip_path,
                max_output_tokens=int(review_cfg.get("max_output_tokens", 500)),
                retry_count=int(review_cfg.get("retry_count", 1)),
                clip_duration_seconds=float(clip_metadata["duration_seconds"]),
                delete_remote_files=bool(review_cfg.get("delete_remote_files", True)),
                video_preprocess_fps=float(
                    review_cfg.get("video_preprocess_fps", 2.0)
                ),
                file_processing_timeout_seconds=float(
                    review_cfg.get("file_processing_timeout_seconds", 120.0)
                ),
            )
            record = make_base_record(
                event, model, prompt_version, input_paths, provider, prompt_text
            )
            record.update(
                {
                    "status": "COMPLETED",
                    **result.model_dump(),
                    "explanation": build_explanation(event, result),
                    "clip_metadata": clip_metadata,
                    "response_id": response_id,
                    "usage": usage,
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                    "error": None,
                }
            )
            append_jsonl(reviews_path, record)
            if db_connection is not None:
                try:
                    with db_connection:
                        upsert_llm_review(db_connection, record)
                        if review_cfg.get("mode") == "active":
                            apply_review_decision(db_connection, record)
                except Exception as exc:
                    db_connection.rollback()
                    append_sync_error(run_dir, "UPSERT_REVIEW", record["attempt_id"], exc)
                    print(
                        f"复核写入 SQLite 失败，已保留 JSONL: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
            print(f"事件 {event_id} 复核完成: {result.decision}")
        except Exception as exc:
            failures += 1
            code = exc.code if isinstance(exc, ReviewError) else type(exc).__name__
            record = make_base_record(
                event,
                idempotency_model,
                prompt_version,
                input_paths,
                provider,
                prompt_text,
            )
            record.update(
                {
                    "status": "FAILED",
                    "decision": None,
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                    "error": {"code": code, "message": str(exc)},
                }
            )
            if not args.prepare_only:
                append_jsonl(reviews_path, record)
                if db_connection is not None:
                    try:
                        with db_connection:
                            upsert_llm_review(db_connection, record)
                    except Exception as db_exc:
                        db_connection.rollback()
                        append_sync_error(
                            run_dir, "UPSERT_REVIEW", record["attempt_id"], db_exc
                        )
                        print(
                            "失败复核记录写入 SQLite 失败，已保留 JSONL: "
                            f"{type(db_exc).__name__}: {db_exc}",
                            file=sys.stderr,
                        )
            print(f"事件 {event_id} 复核失败 [{code}]: {exc}", file=sys.stderr)

    if db_connection is not None:
        db_connection.close()
    if not args.prepare_only and review_cfg.get("mode") == "active":
        from .notifier import dispatch_pending_alerts

        dispatch_pending_alerts(config_path)
    return 1 if failures else 0


def review_run_directory(config_path: Path, run_dir: Path, source_path: Path) -> int:
    """供监控入口复用的自动复核入口，避免重复运行 YOLO。"""

    args = argparse.Namespace(
        config=str(config_path),
        run_dir=str(run_dir),
        source=str(source_path),
        prepare_only=False,
        force=False,
    )
    return run(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Doubao NO_HELMET 视频影子复核")
    parser.add_argument("--config", default="config.yaml", help="YAML 配置文件路径")
    parser.add_argument("--run-dir", required=True, help="包含 events.jsonl 的 YOLO 运行目录")
    parser.add_argument("--source", help="覆盖 camera.source 的原始视频路径")
    parser.add_argument(
        "--prepare-only", action="store_true", help="只生成复核短视频，不调用方舟接口"
    )
    parser.add_argument(
        "--force", action="store_true", help="忽略已有复核记录，重新执行并产生新 attempt_id"
    )
    return parser.parse_args()


def main() -> None:
    try:
        exit_code = run(parse_args())
    except Exception as exc:
        print(f"复核任务启动失败: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
