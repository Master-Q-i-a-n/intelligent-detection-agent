"""
向量库写入前的数据准备（ingest 阶段，DB agnostic）。

管道：build → describe → ingest → index；本模块负责描述注入与记录组装。

输入：
    <output_dir>/chunks/chunks.jsonl
    <output_dir>/images/fig_*/metadata.json
    <output_dir>/rag_image_descriptions.json（视觉模型生成的图片描述）

输出：
    <output_dir>/ingest/records.json
        JSON 数组，每条 = 一个可入库记录：
            text      原样正文（不被污染）
            embed_text 正文 + 图片描述块（用于 embedding，图片描述可被检索命中）
            metadata.images 图片信息（payload，渲染/引用用）

说明：
    - 图片描述注入：ingest 开头读 rag_image_descriptions.json
      （[{name, description, image_type}]，name 为图文件夹名），
      按 name 匹配并回填 images/fig_*/metadata.json 的
      description / image_type / described_at / described_by；
      described_by 取 model 参数，described_at 自动生成。
      描述文件缺失时告警继续（描述保持原值）。
    - 图片描述块格式：正文后追加 "\n********\n本段落对应图片描述：<描述>"
    - 超出总预算时拆成派生记录，保留完整正文和描述。
    - token 计数：可传 --token-model 用 HuggingFaceTokenizer 精确计数；
      否则用字符数/2 的粗略估算（中文为主场景）
    - 本阶段只产出 records.json，由主项目的 Qdrant 流水线入库。
"""

import json
import os
from datetime import datetime
from pathlib import Path

try:
    from . import img_common
except ImportError:
    import img_common


# ============================================================
# token 计数（可选精确，默认粗略估算）
# ============================================================

class TokenCounter:
    """token 计数。

    传入 tokenizer 时精确计数；否则用 len(text) // 2 估算
    （对中文为主的文本近似）。
    """

    def __init__(self, tokenizer=None):
        self._tokenizer = tokenizer

    def count(self, text: str) -> int:
        if self._tokenizer is not None:
            return self._tokenizer.count_tokens(text)
        return max(1, len(text) // 2)


def make_counter(token_model: str | None) -> TokenCounter:
    """按 --token-model 构造计数器。"""

    if not token_model:
        return TokenCounter()

    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )

    tokenizer = HuggingFaceTokenizer.from_pretrained(
        model_name=token_model,
        max_tokens=4096,
    )
    return TokenCounter(tokenizer=tokenizer)


# ============================================================
# 图片描述注入（原 describe 阶段并入 ingest）
# ============================================================

def _inject_descriptions(
    output_dir: Path,
    descriptions_file: Path,
    model: str,
) -> tuple[int, int]:
    """读 rag_image_descriptions.json，按 name 匹配图并回填 metadata.json。

    JSON 格式（用户约定）：
        [
          {"name": "fig_003", "description": "...", "image_type": "diagram"},
          ...
        ]
    name 是图文件夹名（容忍 "images/fig_003/" 等变体）；
    image_type 可选，缺省时保留原值。

    回填字段（字段单写原则，不碰其他键）：
        description / image_type / described_at / described_by

    返回 (注入条数, 未匹配条数)。
    描述文件缺失 / 解析失败时打印 WARNING 返回 (0, 0)，不阻塞 ingest。
    """

    if not descriptions_file.is_file():
        print(
            f"WARNING: 描述文件不存在 "
            f"{descriptions_file}，跳过描述注入"
        )
        return 0, 0

    try:
        with descriptions_file.open(
            "r",
            encoding="utf-8",
        ) as f:
            items = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: 描述文件解析失败: {e}")
        return 0, 0

    if not isinstance(items, list):
        print("WARNING: 描述文件应为 JSON 数组")
        return 0, 0

    injected = 0
    unmatched = 0

    for item in items:
        name = Path(
            str(item.get("name", ""))
        ).name

        if not name:
            unmatched += 1
            continue

        desc = item.get("description")

        if not desc:
            print(
                f"WARNING: {name} 缺 description，跳过"
            )
            unmatched += 1
            continue

        image_dir = (
            output_dir /
            "images" /
            name
        )

        meta = img_common.load_metadata(
            image_dir
        )

        if meta is None:
            print(
                f"WARNING: {name} 未匹配到"
                "图片文件夹/缺 metadata.json"
            )
            unmatched += 1
            continue

        meta["description"] = desc

        if item.get("image_type"):
            meta["image_type"] = (
                item["image_type"]
            )

        meta["described_at"] = (
            datetime.now()
            .isoformat(timespec="seconds")
        )
        meta["described_by"] = model

        img_common.save_metadata(
            image_dir,
            meta,
        )
        injected += 1

    return injected, unmatched


# ============================================================
# 主流程
# ============================================================

def ingest(
    output_dir: str | Path,
    descriptions: str | Path | None = None,
    model: str = "deepseek-v4-flash-vision-exp",
    token_model: str | None = None,
    max_tokens: int = 512,
) -> dict:
    """注入图片描述（原 describe 职责）+ 构建可入库记录。

    Parameters
    ----------
    output_dir:
        输出目录。

    descriptions:
        图片描述 JSON 文件路径；None 时默认
        <output_dir>/rag_image_descriptions.json。

    model:
        描述来源模型标识，写入 metadata.json 的 described_by。

    Returns
    -------
    dict:
        records 路径、记录数量和描述注入/截断统计。
    """

    output_dir = Path(output_dir)
    chunks_path = output_dir / "chunks" / "chunks.jsonl"
    out_path = output_dir / "ingest" / "records.json"

    if descriptions is None:
        descriptions = (
            output_dir /
            "rag_image_descriptions.json"
        )
    descriptions = Path(descriptions)

    counter = make_counter(token_model)

    # 描述注入（原 describe 阶段）：先回填 metadata.json，
    # 后续组装 records 时 load_metadata 才能读到新描述
    injected, unmatched = _inject_descriptions(
        output_dir,
        descriptions,
        model,
    )

    rows = []

    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    desc_truncated = 0
    text_truncated = 0
    with_image = 0

    for record in rows:

        # -----------------------------------------------
        # 图片信息（payload 内容）
        # -----------------------------------------------

        image_infos = []

        for folder in record.get("images", []):

            # folder 是相对 output 根目录的路径（如 "images/fig_002/"）
            image_dir = output_dir / folder

            meta = img_common.load_metadata(image_dir)

            if meta is None:
                image_infos.append(
                    {
                        "folder": folder,
                        "image_error": (
                            "缺 metadata.json"
                        ),
                    }
                )
                continue

            info = {
                "folder": folder,
                "page_no": meta.get("page_no"),
            }

            caption = meta.get("caption")
            if caption and caption.get("text"):
                info["caption"] = caption["text"]

            if meta.get("image_error"):
                info["image_error"] = (
                    meta["image_error"]
                )
            else:
                # description 由 build 预置空串、describe 回填；
                # 空值时不在 payload 里放噪音键
                if meta.get("image_type"):
                    info["image_type"] = (
                        meta["image_type"]
                    )
                if meta.get("description"):
                    info["description"] = (
                        meta["description"]
                    )

            image_infos.append(info)

        if image_infos:
            with_image += 1

        # -----------------------------------------------
        # embed_text：正文 + 逐图追加描述块
        # -----------------------------------------------

        text = record.get("text", "")

        embed_text = text

        for info in image_infos:

            desc = info.get("description")

            if not desc:
                continue

            embed_text += (
                f"\n********\n"
                f"本段落对应图片描述：{desc}"
            )

        # 后续按预算拆成派生记录；禁止按字符截断后丢弃正文或图片描述。

        num_tokens = counter.count(embed_text)

        # -----------------------------------------------
        # 记录
        # -----------------------------------------------

        records.append(
            {
                "chunk_id": record.get("id"),
                "source": record.get("source"),
                "text": text,
                "embed_text": embed_text,
                "num_tokens": num_tokens,
                "page_numbers": record.get(
                    "page_numbers",
                    [],
                ),
                "metadata": {
                    "headings": record.get(
                        "headings",
                        [],
                    ),
                    "images": image_infos,
                },
            }
        )

    expanded = []
    for record in records:
        remaining = record["embed_text"]
        parts = []
        while remaining:
            # 二分查找满足当前计数器预算的最大前缀，文本按原顺序完整覆盖。
            low, high = 1, len(remaining)
            while low < high:
                middle = (low + high + 1) // 2
                if counter.count(remaining[:middle]) <= max_tokens:
                    low = middle
                else:
                    high = middle - 1
            parts.append(remaining[:low])
            remaining = remaining[low:]
        for index, part in enumerate(parts):
            expanded.append({**record, "chunk_id": record["chunk_id"] if len(parts) == 1
                             else f"{record['chunk_id']}_part_{index + 1:03d}",
                             "embed_text": part, "num_tokens": counter.count(part)})
    records = expanded

    # -----------------------------------------------
    # 写 ingest/records.json（JSON 数组，原子替换）
    # -----------------------------------------------

    fd, tmp_path = __import__("tempfile").mkstemp(
        dir=out_path.parent,
        suffix=".tmp",
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                records,
                f,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    # -----------------------------------------------
    # 摘要
    # -----------------------------------------------

    print("=" * 60)
    print("ingest 完成")
    print("=" * 60)
    print(f"注入描述        : {injected}")
    print(f"未匹配(告警)    : {unmatched}")
    print(f"记录数          : {len(records)}")
    print(f"含图片记录      : {with_image}")
    print(f"描述被截断      : {desc_truncated}")
    print(f"正文被截断(告警) : {text_truncated}")
    print(f"输出            : {out_path}")

    # 与 build_pdf_dataset 保持一致，返回后续流程可能使用的结果摘要。
    return {
        "output_dir": output_dir,
        "records_path": out_path,
        "record_count": len(records),
        "with_image_count": with_image,
        "description_injected": injected,
        "description_unmatched": unmatched,
        "description_truncated": desc_truncated,
        "text_truncated": text_truncated,
    }


# ============================================================
# 调用
# ============================================================

if __name__ == "__main__":

    ingest(
        output_dir="output\压力传感器",

        # None 时读取 <output_dir>/rag_image_descriptions.json
        descriptions=None,

        # 写入图片 metadata 的描述模型标识
        model="gpt5.6sol",

        # None 时使用字符数/2估算 token
        token_model=None,

        max_tokens=512,
    )
