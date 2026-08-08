import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from safety_operations.review import (
    ReviewError,
    build_explanation,
    call_ark_review,
    create_review_clip,
    interpolate_target_bbox,
    load_review_keys,
    parse_review_response,
    select_active_events,
)


def make_video(path: Path, seconds: float = 3.0, fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120)
    )
    assert writer.isOpened()
    for frame_index in range(int(seconds * fps)):
        frame = np.full((120, 160, 3), frame_index % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_selects_only_first_active_record_per_event() -> None:
    records = [
        {"event_id": "a", "event_type": "NO_HELMET", "status": "PENDING"},
        {"event_id": "a", "event_type": "NO_HELMET", "status": "ACTIVE"},
        {"event_id": "a", "event_type": "NO_HELMET", "status": "ACTIVE"},
        {"event_id": "b", "event_type": "DWELL", "status": "ACTIVE"},
    ]
    selected = select_active_events(records, {"NO_HELMET"})
    assert [record["event_id"] for record in selected] == ["a"]


def test_review_keys_support_idempotency(tmp_path: Path) -> None:
    path = tmp_path / "reviews.jsonl"
    path.write_text(
        json.dumps(
            {"event_id": "event-1", "model": "mini", "prompt_version": "v1"}
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_review_keys(path) == {("event-1", "mini", "v1")}


def test_create_review_clip_samples_five_fps_and_truncates_end(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "clip.mp4"
    make_video(source, seconds=3.0, fps=10.0)

    metadata = create_review_clip(
        source,
        output,
        trigger_seconds=2.0,
        pre_seconds=1.0,
        post_seconds=2.0,
        output_fps=5.0,
        max_width=100,
    )

    assert output.exists()
    assert metadata["start_seconds"] == 1.0
    assert metadata["end_seconds"] == 3.0
    assert metadata["fps"] == 5.0
    assert metadata["frame_count"] == 10
    assert metadata["width"] == 100
    capture = cv2.VideoCapture(str(output))
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    capture.release()
    codec = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4))
    assert codec.lower() in {"h264", "avc1"}


def test_create_review_clip_rejects_less_than_two_seconds(tmp_path: Path) -> None:
    source = tmp_path / "short.mp4"
    make_video(source, seconds=1.0, fps=10.0)
    with pytest.raises(ReviewError, match="少于") as exc_info:
        create_review_clip(source, tmp_path / "clip.mp4", 0.5, 1, 1, 5, 1280)
    assert exc_info.value.code == "CLIP_TOO_SHORT"


def test_target_bbox_is_interpolated_between_state_snapshots() -> None:
    timeline = [
        (1.0, (10.0, 20.0, 30.0, 40.0)),
        (2.0, (20.0, 30.0, 40.0, 50.0)),
    ]
    assert interpolate_target_bbox(timeline, 1.5) == (15.0, 25.0, 35.0, 45.0)
    assert interpolate_target_bbox(timeline, 4.0) is None


def test_parse_function_call_and_build_explanation() -> None:
    arguments = json.dumps(
        {
            "decision": "CONFIRMED",
            "target_visible": "CLEAR",
            "helmet_status": "NOT_WORN",
            "evidence_quality": "GOOD",
            "visual_reason": "目标头顶清晰可见，未见安全帽。",
            "evidence_timestamps": [1.0, 2.0],
        }
    )
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                name="submit_no_helmet_review",
                arguments=arguments,
            )
        ],
        output_text="",
    )
    result = parse_review_response(response)
    explanation = build_explanation(
        {
            "track_id": 7,
            "metrics": {
                "helmet_ratio": 0.1,
                "evaluable_frames": 20,
                "visible_seconds": 6.2,
            },
        },
        result,
    )
    assert result.decision == "CONFIRMED"
    assert "10.0%" in explanation
    assert "20 帧" in explanation
    assert "6.2 秒" in explanation


class FakeFiles:
    def __init__(self) -> None:
        self.created = []
        self.deleted = []

    def create(self, *, file, purpose, **kwargs):
        self.created.append((Path(file.name).name, purpose, kwargs))
        return SimpleNamespace(id=f"file-{len(self.created)}")

    def delete(self, file_id):
        self.deleted.append(file_id)


class FakeResponses:
    def __init__(self, invalid_first: bool = False) -> None:
        self.invalid_first = invalid_first
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.invalid_first and len(self.calls) == 1:
            return SimpleNamespace(id="bad", output=[], output_text="not-json", usage=None)
        payload = {
            "decision": "CONFIRMED",
            "target_visible": "CLEAR",
            "helmet_status": "NOT_WORN",
            "evidence_quality": "GOOD",
            "visual_reason": "连续画面中目标头顶裸露。",
            "evidence_timestamps": [1.0],
        }
        if len(self.calls) == 1:
            output = [
                SimpleNamespace(
                    type="function_call",
                    name="submit_no_helmet_review",
                    arguments=json.dumps(payload),
                )
            ]
            output_text = ""
        else:
            output = []
            output_text = json.dumps(payload)
        return SimpleNamespace(
            id=f"resp-{len(self.calls)}", output=output, output_text=output_text, usage=None
        )


def test_ark_call_uploads_inputs_retries_invalid_json_and_cleans_files(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("safety_operations.review.time.sleep", lambda _: None)
    paths = []
    for name in ("clip.mp4",):
        path = tmp_path / name
        path.write_bytes(b"test")
        paths.append(path)
    client = SimpleNamespace(files=FakeFiles(), responses=FakeResponses(invalid_first=True))

    result, response_id, usage = call_ark_review(
        client=client,
        model="mini",
        prompt="review",
        video_path=paths[0],
        max_output_tokens=500,
        retry_count=1,
        clip_duration_seconds=2.0,
        delete_remote_files=True,
    )

    assert result.decision == "CONFIRMED"
    assert response_id == "resp-2"
    assert usage is None
    assert len(client.responses.calls) == 2
    assert "tools" in client.responses.calls[0]
    assert "tools" not in client.responses.calls[1]
    assert [item[:2] for item in client.files.created] == [
        ("clip.mp4", "user_data"),
    ]
    assert client.files.created[0][2]["extra_body"] == {
        "preprocess_configs": {"video": {"fps": 2.0}}
    }
    first_content = client.responses.calls[0]["input"][0]["content"]
    assert first_content[1]["type"] == "input_video"
    assert client.files.deleted == ["file-1"]
