"""
图片与 chunk 关联的共享工具。

供 build.py / ingest.py 复用：
    - 稳定主键解析（#/pictures/{M} -> fig_{M:03d}）
    - 图片文件夹 / metadata.json 的读写
    - caption 启发式的正则、bbox 几何与距离阈值
"""

import json
import os
import re
from pathlib import Path

# 默认离线模式：transformers / huggingface_hub 只走本地缓存，
# 避免联网访问 huggingface.co 超时（可用环境变量覆盖）。
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ============================================================
# 常量
# ============================================================

# metadata.json schema 版本：
# 1.0 首版；2.0 精简字段（去掉 folder/label/bbox/image/ocr_refs/
# enriched_at/image_type_confidence/figure_specific，caption 去掉 distance）
IMAGE_SCHEMA_VERSION = "2.0"

# metadata.json 文件名
METADATA_FILE = "metadata.json"

# caption 正则：覆盖 "图1" / "图 2" / "图3a" / "图4：" / "图5." 等
CAPTION_REGEX = re.compile(r"^图\s*\d+[a-zA-Z]?[\.．、:：]?\s*")

# bbox 质心距离阈值：相对页高的比例 + 绝对上下限（pt）
CAPTION_DIST_FACTOR = 0.2
CAPTION_DIST_MIN = 120.0
CAPTION_DIST_MAX = 300.0

# ocr_text 截断上限（字符）
OCR_TEXT_MAX_CHARS = 2000

# ingest 阶段每个图片描述进 embed_text 的 token 预算
DESC_TOKEN_BUDGET = 200


# ============================================================
# 稳定主键 / 路径
# ============================================================

def self_ref_to_index(self_ref: str) -> int:
    """'#/pictures/3' -> 3。"""
    m = re.match(r"^#/pictures/(\d+)$", self_ref)
    if not m:
        raise ValueError(f"无法解析图片 self_ref: {self_ref}")
    return int(m.group(1))


def fig_folder(m: int) -> str:
    """相对路径（相对 output 根目录），如 'images/fig_003/'。"""
    return f"images/fig_{m:03d}/"


def fig_file(m: int) -> str:
    """相对路径，如 'images/fig_003/fig_003.png'。"""
    return f"{fig_folder(m)}fig_{m:03d}.png"


def image_folder_path(output_dir: Path, m: int) -> Path:
    """磁盘上的图片文件夹绝对/相对 Path。"""
    return Path(output_dir) / "images" / f"fig_{m:03d}"


# ============================================================
# metadata.json 读写
# ============================================================

def load_metadata(image_dir: Path) -> dict:
    """读图片文件夹下的 metadata.json；不存在返回 None。"""
    path = Path(image_dir) / METADATA_FILE
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_metadata(image_dir: Path, data: dict) -> None:
    """写图片文件夹下的 metadata.json（含排序键，便于 diff）。"""
    path = Path(image_dir) / METADATA_FILE
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


# ============================================================
# bbox 几何（BOTTOMLEFT 坐标系）
# ============================================================

def bbox_rect_gap(a: dict, b: dict) -> float:
    """两个 bbox 的矩形间隙（两矩形之间的最小欧氏距离，重叠/相接为 0）。

    比质心距离更符合"caption 紧贴图下方"的几何直觉：
    大图质心离下方 caption 天然远，但矩形间隙很小。
    """
    dx = max(
        0.0,
        a["l"] - b["r"],
        b["l"] - a["r"],
    )
    dy = max(
        0.0,
        a["b"] - b["t"],
        b["b"] - a["t"],
    )
    return (dx * dx + dy * dy) ** 0.5


def fully_inside(a: dict, b: dict, tol: float = 2.0) -> bool:
    """a 的 bbox 是否完全落在 b 的 bbox 内部（含容差）。

    BOTTOMLEFT 坐标：t 是上边距底部的距离，b 是下边距底部的距离，故 t > b。
    """
    return (
        a["l"] >= b["l"] - tol
        and a["r"] <= b["r"] + tol
        and a["b"] >= b["b"] - tol
        and a["t"] <= b["t"] + tol
    )


def page_height(pages: dict, page_no: int) -> float:
    """页高（pt）。pages 是 document.json 的 pages dict，key 为页码字符串。"""
    page = pages.get(str(page_no))
    if page and page.get("size"):
        return page["size"].get("height", 0.0)
    return 0.0


def caption_dist_threshold(pages: dict, page_no: int) -> float:
    """caption 启发式的最大质心距离阈值。

    相对页高（0.2x）为主，夹在 [120, 300] pt 之间。
    """
    h = page_height(pages, page_no)
    if h <= 0:
        return CAPTION_DIST_MAX
    return max(CAPTION_DIST_MIN, min(CAPTION_DIST_MAX, CAPTION_DIST_FACTOR * h))
