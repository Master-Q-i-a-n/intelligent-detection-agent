from __future__ import annotations

import os
from pathlib import Path

import cv2


def create_browser_video_writer(
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:
    """创建 H.264 MP4 写入器，避免生成浏览器无法解码的 mp4v 视频。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates: list[tuple[int, str, str]] = []
    if os.name == "nt":
        # Windows Media Foundation 自带 H.264 编码器，不依赖外部 FFmpeg 程序。
        candidates.append((cv2.CAP_MSMF, "H264", "Windows Media Foundation"))
    candidates.extend(
        [
            (cv2.CAP_FFMPEG, "avc1", "OpenCV FFmpeg"),
            (cv2.CAP_ANY, "avc1", "OpenCV automatic backend"),
        ]
    )

    attempted: list[str] = []
    for backend, codec, label in candidates:
        output_path.unlink(missing_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            backend,
            cv2.VideoWriter_fourcc(*codec),
            fps,
            frame_size,
        )
        if writer.isOpened():
            return writer
        writer.release()
        attempted.append(f"{label}/{codec}")

    raise RuntimeError(
        "无法创建浏览器兼容的 H.264 MP4；已尝试 " + ", ".join(attempted)
    )
