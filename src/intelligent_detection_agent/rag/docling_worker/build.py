import os

# Windows 下避免 torch.compile -> Triton 问题
os.environ["DOCLING_INFERENCE_COMPILE_TORCH_MODELS"] = "false"

import json
import re
import tempfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import img_common

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.base_models import (
    InputFormat,
    DocumentStream,
)
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
)
from docling.chunking import HybridChunker

from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
)
from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    BaseSerializerProvider,
    BaseTableSerializer,
)
from docling_core.transforms.serializer.markdown import (
    MarkdownTableSerializer,
)
from docling_core.types.doc import (
    DoclingDocument,
    PictureItem,
    TableItem,
)


# ============================================================
# 默认配置
# ============================================================

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_MAX_TOKENS = 512


class MarkdownTableChunkingSerializer(ChunkingDocSerializer):
    """保留现有正文序列化规则，仅将表格输出为 Markdown。"""

    table_serializer: BaseTableSerializer = MarkdownTableSerializer()


class MarkdownTableSerializerProvider(BaseSerializerProvider):
    """为 HybridChunker 提供使用 Markdown 表格的序列化器。"""

    def get_serializer(
        self,
        doc: DoclingDocument,
    ) -> BaseDocSerializer:
        return MarkdownTableChunkingSerializer(doc=doc)

# 图号解析正则：从 caption 文本提取图号
#   "图3 气体超声流量计的温度计安装示意图" -> "3"
#   "图 A.1 xxx"                          -> "A.1"（附录图）
#   "图3a xxx"                            -> "3a"（子图）
FIGNO_REGEX = re.compile(
    r"^图\s*((?:[A-Za-z]\.)?\d+[a-zA-Z]?)"
)


# ============================================================
# PDF 处理函数
# ============================================================

def build_pdf_dataset(
    pdf_file: str | Path,
    page_range: tuple[int, int] | None = None,
    force_ocr: bool = False,
    output_dir: str | Path | None = None,
    min_image_area: int = 8_000,
):
    """
    使用 Docling 解析 PDF，并生成：
        parsed/document.json
        images/fig_M/(png + metadata.json)
        tables/*.json
        chunks/chunks.jsonl

    本函数同时完成 chunk ↔ 图片关联（原 enrich 阶段已并入）：
        chunks.jsonl 的 images 字段、metadata.json 的
        caption / heading / linked_chunk_ids 一并生成。
        metadata.json 的 description 默认空串，由 ingest 阶段回填
        （数据源 rag_image_descriptions.json）。

    Parameters
    ----------
    pdf_file:
        PDF 文件路径。

    page_range:
        页码范围，例如：
            (1, 20)
            (6, 20)

        None 表示解析全部页面。

    force_ocr:
        False:
            正常模式。有可用文本层时优先使用文本层，
            必要区域才 OCR。

        True:
            强制整页 OCR。
            适合中文乱码、扫描件、文本层损坏的 PDF。

    output_dir:
        输出目录。

        如果不传：
            output/<PDF文件名>/

    min_image_area:
        图片最小像素面积，宽 × 高小于该值时不导出、也不关联到
        chunk。默认 8000；传 0 可关闭尺寸过滤。
    """

    if min_image_area < 0:
        raise ValueError("图片最小像素面积不能小于 0")

    # ========================================================
    # 1. 输入路径
    # ========================================================

    pdf_path = Path(pdf_file)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF 不存在: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"不是 PDF 文件: {pdf_path}"
        )

    # ========================================================
    # 2. 输出目录
    # ========================================================

    if output_dir is None:
        output_dir = Path("output") / pdf_path.stem
    else:
        output_dir = Path(output_dir)

    parsed_dir = output_dir / "parsed"
    images_dir = output_dir / "images"
    tables_dir = output_dir / "tables"
    chunks_dir = output_dir / "chunks"

    for directory in [
        parsed_dir,
        images_dir,
        tables_dir,
        chunks_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # 3. 使用 DocumentStream
    #
    # 避免 docling-parse 在 Windows 下直接处理中文路径
    # ========================================================

    source = DocumentStream(
        name=pdf_path.name,
        stream=BytesIO(
            pdf_path.read_bytes()
        ),
    )

    # ========================================================
    # 4. PDF Pipeline 配置
    # ========================================================

    pipeline_options = PdfPipelineOptions()

    # 开启 OCR
    pipeline_options.do_ocr = True

    # RapidOCR 使用 PyTorch / GPU
    pipeline_options.ocr_options = RapidOcrOptions(
        backend="torch",
        force_full_page_ocr=force_ocr,
    )

    # 表格结构识别
    pipeline_options.do_table_structure = True

    # 图片导出
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    pipeline_options.images_scale = 1.0

    # ========================================================
    # 5. Converter
    # ========================================================

    converter = DocumentConverter(
        allowed_formats=[
            InputFormat.PDF
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        },
    )

    # ========================================================
    # 6. 解析 PDF
    # ========================================================

    print("=" * 60)
    print(f"PDF      : {pdf_path.name}")
    print(f"页码范围 : {page_range or '全部'}")
    print(f"强制 OCR : {force_ocr}")
    print("=" * 60)

    print("开始解析 PDF...")

    if page_range is None:
        result = converter.convert(
            source
        )
    else:
        result = converter.convert(
            source,
            page_range=page_range,
        )

    doc = result.document

    print("PDF 解析完成")

    # ========================================================
    # 7. 保存完整 document.json
    # ========================================================

    document_json = (
        parsed_dir /
        "document.json"
    )

    document_dict = doc.export_to_dict()

    with document_json.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            document_dict,
            f,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    print(
        f"保存结构化文档: {document_json}"
    )

    # ========================================================
    # 8. 导出图片
    #
    # 每个图片一个文件夹 images/fig_{M:03d}/：
    #     图本体   fig_{M:03d}.png
    #     元数据   metadata.json
    #
    # M 取自 item.self_ref（#/pictures/{M}），是稳定主键，
    # 与导出成败无关，保证 document.json / chunks / 文件夹三方对齐。
    # ========================================================

    pictures_dict = (
        document_dict.get(
            "pictures",
            [],
        )
    )

    texts_list = document_dict.get(
        "texts",
        [],
    )

    picture_total = 0
    picture_exported = 0
    picture_filtered_small = 0
    filtered_picture_ids: set[int] = set()

    for item, _ in doc.iterate_items():

        if not isinstance(
            item,
            PictureItem,
        ):
            continue

        m = img_common.self_ref_to_index(
            item.self_ref
        )

        picture_total += 1

        # -----------------------------------------------
        # 导出图片本体
        # -----------------------------------------------

        image_error = None
        image = None

        try:

            image = item.get_image(doc)

            if image is None:
                image_error = (
                    "get_image() 返回 None"
                )
            else:
                width, height = image.size
                image_area = width * height

                if (
                    min_image_area > 0
                    and image_area < min_image_area
                ):
                    filtered_picture_ids.add(m)
                    picture_filtered_small += 1

                    # 重复生成到同一输出目录时，清理该图片以前由流程
                    # 生成的文件，避免过滤后仍留下历史小图。
                    stale_folder = (
                        images_dir /
                        f"fig_{m:03d}"
                    )
                    for stale_name in (
                        f"fig_{m:03d}.png",
                        img_common.METADATA_FILE,
                    ):
                        stale_file = stale_folder / stale_name
                        stale_file.unlink(missing_ok=True)
                    if stale_folder.is_dir():
                        try:
                            stale_folder.rmdir()
                        except OSError:
                            print(
                                "WARNING: 过滤图片目录仍含其他文件，"
                                f"未删除目录: {stale_folder}"
                            )

                    print(
                        f"过滤尺寸过小图片 fig_{m:03d}: "
                        f"{width}x{height}={image_area}px² "
                        f"< {min_image_area}px²"
                    )
                    continue

                folder = (
                    images_dir /
                    f"fig_{m:03d}"
                )
                folder.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                image.save(
                    folder /
                    f"fig_{m:03d}.png",
                    "PNG",
                )
                picture_exported += 1

        except Exception as e:

            image_error = str(e)
            print(
                f"图片导出失败 "
                f"({item.self_ref}): {e}"
            )

        # 导出失败时仍保留 metadata，供后续流程记录错误原因。
        folder = (
            images_dir /
            f"fig_{m:03d}"
        )
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        # -----------------------------------------------
        # 基础元数据（metadata.json）
        #
        # 字段所有权：本函数写基础字段 + description 空占位；
        # caption/heading/linked_chunk_ids 在本函数 chunking 后的
        # 关联步骤回填；image_type/description 由 ingest 阶段回填
        # （数据源 rag_image_descriptions.json）。
        # -----------------------------------------------

        pic_dict = next(
            (
                p
                for p in pictures_dict
                if p.get("self_ref")
                == item.self_ref
            ),
            {},
        )

        prov = (
            pic_dict.get(
                "prov",
                [{}],
            )[0]
            if pic_dict
            else {}
        )

        # 图内 OCR 文字（children 里 label=="caption"
        # 的子项排除，避免 caption 重复出现）
        ocr_lines = []

        for child in (
            pic_dict.get(
                "children",
                [],
            )
            if pic_dict
            else []
        ):

            ref_m = re.match(
                r"#/texts/(\d+)",
                child.get("$ref", ""),
            )

            if not ref_m:
                continue

            text_item = (
                texts_list[
                    int(ref_m.group(1))
                ]
                if 0 <= int(ref_m.group(1))
                < len(texts_list)
                else {}
            )

            if text_item.get(
                "label",
            ) == "caption":
                continue

            text = (
                text_item
                .get("text", "")
                .strip()
            )

            if text:
                ocr_lines.append(text)

        ocr_text = (
            "\n".join(ocr_lines)
            [: img_common.OCR_TEXT_MAX_CHARS]
        )

        img_common.save_metadata(
            folder,
            {
                "schema_version":
                    img_common
                    .IMAGE_SCHEMA_VERSION,

                "self_ref":
                    item.self_ref,

                "image_file":
                    img_common.fig_file(m),

                "page_no":
                    prov.get("page_no"),

                "ocr_text":
                    ocr_text,

                "image_error":
                    image_error,

                # 图片描述：默认空串，describe 阶段回填
                "description":
                    "",
            },
        )

    # ========================================================
    # 9. 导出表格
    # ========================================================

    table_index = 0

    for item, _ in doc.iterate_items():

        if not isinstance(
            item,
            TableItem,
        ):
            continue

        table_index += 1

        page_no = None

        if item.prov:
            page_no = (
                item.prov[0].page_no
            )

        try:

            df = item.export_to_dataframe(
                doc=doc
            )

            table_data = {
                "table_id":
                    f"table_{table_index:03d}",

                "page":
                    page_no,

                "columns": [
                    str(column)
                    for column
                    in df.columns
                ],

                "rows": [
                    [
                        str(value)
                        for value in row
                    ]
                    for row
                    in df.itertuples(
                        index=False,
                        name=None,
                    )
                ],
            }

        except Exception as e:

            table_data = {
                "table_id":
                    f"table_{table_index:03d}",

                "page":
                    page_no,

                "error":
                    str(e),

                "data":
                    item.data.model_dump(
                        mode="json"
                    ),
            }

        table_path = (
            tables_dir /
            f"table_{table_index:03d}.json"
        )

        with table_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                table_data,
                f,
                ensure_ascii=False,
                indent=2,
            )

    # ========================================================
    # 10. HybridChunker
    # ========================================================

    print("初始化 HybridChunker...")

    tokenizer = (
        HuggingFaceTokenizer
        .from_pretrained(
            model_name=EMBED_MODEL_ID,
            max_tokens=CHUNK_MAX_TOKENS,
        )
    )

    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
        repeat_table_header=True,
        serializer_provider=MarkdownTableSerializerProvider(),
    )

    # ========================================================
    # 11. Chunking
    # ========================================================

    chunks_path = (
        chunks_dir /
        "chunks.jsonl"
    )

    chunk_count = 0
    records = []

    with chunks_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for chunk in chunker.chunk(
            dl_doc=doc
        ):

            # -----------------------------------------------
            # 页码
            # -----------------------------------------------

            page_numbers = sorted(
                {
                    prov.page_no
                    for item
                    in chunk.meta.doc_items
                    for prov
                    in item.prov
                }
            )

            # -----------------------------------------------
            # contextualized text
            # heading + caption + chunk
            # -----------------------------------------------

            contextualized_text = (
                chunker.contextualize(
                    chunk
                )
            )

            num_tokens = (
                chunker
                .tokenizer
                .count_tokens(
                    contextualized_text
                )
            )

            # -----------------------------------------------
            # metadata
            # -----------------------------------------------

            metadata = {}

            if chunk.meta.origin:
                metadata["origin"] = (
                    chunk.meta.origin
                    .model_dump(
                        mode="json"
                    )
                )

            # -----------------------------------------------
            # Chunk ID
            # -----------------------------------------------

            chunk_count += 1

            record = {
                "id":
                    f"chunk_{chunk_count:05d}",

                # 真正原始文件名
                "source":
                    pdf_path.name,

                # 推荐用于 embedding
                "text":
                    contextualized_text,

                # 原始 chunk
                "raw_text":
                    chunk.text,

                "num_tokens":
                    num_tokens,

                "headings":
                    chunk.meta.headings
                    or [],

                "captions":
                    chunk.meta.captions
                    or [],

                "doc_items": [
                    item.self_ref
                    for item
                    in chunk.meta.doc_items
                ],

                # 关联图片文件夹路径（相对 output 根目录），
                # 在本函数 chunking 后的关联步骤回填。
                "images": [],

                "page_numbers":
                    page_numbers,

                "metadata":
                    metadata,
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            records.append(record)

    # ========================================================
    # 12. 图片关联（原 enrich 阶段并入）+ 写回 chunks.jsonl
    # ========================================================

    _associate_images(
        output_dir,
        document_dict,
        records,
        filtered_picture_ids,
    )

    fd, tmp_path = tempfile.mkstemp(
        dir=chunks_path.parent,
        suffix=".tmp",
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as f:
            for record in records:
                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        os.replace(tmp_path, chunks_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    # ========================================================
    # 完成
    # ========================================================

    print()
    print("=" * 60)
    print("处理完成")
    print("=" * 60)

    print(
        f"原文件        : {pdf_path.name}"
    )
    print(
        f"Document JSON : {document_json}"
    )
    print(
        f"图片数量      : {picture_total}"
    )
    print(
        f"导出成功      : {picture_exported}"
    )
    print(
        f"尺寸过小过滤  : {picture_filtered_small}"
    )
    print(
        f"表格数量      : {table_index}"
    )
    print(
        f"Chunk 数量    : {chunk_count}"
    )
    print(
        f"Chunks        : {chunks_path}"
    )

    # 返回一些后续可能会用到的对象
    return {
        "document": doc,
        "document_json": document_json,
        "chunks_path": chunks_path,
        "output_dir": output_dir,
        "picture_total": picture_total,
        "picture_exported": picture_exported,
        "picture_filtered_small": picture_filtered_small,
        "table_count": table_index,
        "chunk_count": chunk_count,
    }


# ============================================================
# chunk ↔ 图片关联（原 enrich.py 的逻辑，现已并入 build）
# ============================================================

def _fig_index(folder: str) -> int:
    """'images/fig_003/' -> 3，用于排序。"""
    m = re.search(r"fig_(\d+)", folder)
    return int(m.group(1)) if m else -1


def _resolve_caption(
    pic: dict,
    texts_by_ref: dict,
    pages: dict,
    all_texts: list,
) -> dict | None:
    """解析图片的 caption 文本项信息。

    返回 {text, ref, source}；找不到返回 None。
    source: "docling" | "heuristic"。
    """

    # ① docling 自带 captions 优先
    for cap_ref in pic.get("captions", []):
        ref = cap_ref.get("$ref", "")
        text_item = texts_by_ref.get(ref)
        if text_item is not None:
            return {
                "text": text_item.get("text", ""),
                "ref": ref,
                "source": "docling",
            }

    # ② 启发式：同页 + 正则 + bbox 矩形间隙最近
    prov = pic.get("prov", [{}])[0]
    page = prov.get("page_no")
    pic_box = prov.get("bbox")

    if page is None or not pic_box:
        return None

    threshold = img_common.caption_dist_threshold(
        pages,
        page,
    )

    best = None
    best_dist = None

    for text_item in all_texts:
        t_prov = (
            text_item.get("prov", [{}])[0]
            if text_item.get("prov")
            else {}
        )
        t_box = t_prov.get("bbox")

        if (
            t_prov.get("page_no") != page
            or not t_box
        ):
            continue

        if not img_common.CAPTION_REGEX.match(
            text_item.get("text", "")
        ):
            continue

        # 排除落在图片 bbox 内部的文本（图内标注）
        if img_common.fully_inside(
            t_box,
            pic_box,
        ):
            continue

        dist = img_common.bbox_rect_gap(
            t_box,
            pic_box,
        )

        if dist > threshold:
            continue

        if best is None or dist < best_dist:
            best = text_item
            best_dist = dist

    if best is None:
        return None

    return {
        "text": best.get("text", ""),
        "ref": best.get("self_ref", ""),
        "source": "heuristic",
    }


def _parse_fig_no(caption: str) -> str | None:
    """从 caption 文本解析图号；解析失败返回 None。

    "图3 气体超声流量计的温度计安装示意图" -> "3"
    "图 A.1 附录示例"                    -> "A.1"
    """
    m = FIGNO_REGEX.match(caption)
    return m.group(1) if m else None


def _figure_ref_regex(fig_no: str) -> re.Pattern:
    """正文引用该图号的搜索正则。

    尾部 (?![0-9a-zA-Z]) 排除相邻图号：
        "3"    不匹配 "图30"、"图3a"
        "A.1"  不匹配 "图 A.10"
    """
    return re.compile(
        rf"图\s*{re.escape(fig_no)}(?![0-9a-zA-Z])"
    )


def _associate_images(
    output_dir: Path,
    document_dict: dict,
    records: list[dict],
    filtered_picture_ids: set[int] | None = None,
) -> None:
    """chunk ↔ 图片关联（caption 优先 + 正文引用 + 页级兜底）。

    规则：
      ① caption 取 docling 自带；为空用启发式
         （同页 + 正则 ^图\\s*\\d+ + 矩形间隙最近，带阈值）
      ② 目标 chunk = 含 caption 文本项 self_ref 的所有 chunk
         ∪ 同页正文里引用该图号（"见图3"、"图3示意说明"）的所有 chunk。
         图注文本项会被 chunker 并进"图后"的 chunk，而解释图的
         正文常在图前的相邻 chunk，只挂 caption chunk 会漏语义。
      ③ 两者都为空时，退回图片所在页的所有 chunk
      ④ 尺寸过滤名单中的图片不参与任何关联

    回填：
      records 的 images 字段（调用方负责写回 chunks.jsonl）
      images/fig_*/metadata.json 的 caption / heading / linked_chunk_ids
    """

    texts_by_ref = {
        t["self_ref"]: t
        for t in document_dict.get("texts", [])
    }
    all_texts = document_dict.get("texts", [])
    pages = document_dict.get("pages", {})
    filtered_picture_ids = filtered_picture_ids or set()

    # doc_item self_ref -> chunk 行号
    item_to_chunks: dict[str, list[int]] = (
        defaultdict(list)
    )

    for idx, record in enumerate(records):
        for ref in record.get("doc_items", []):
            item_to_chunks[ref].append(idx)

    # 逐图关联
    chunk_images: dict[int, set[str]] = (
        defaultdict(set)
    )

    # M -> 回填 metadata.json 的字段
    meta_updates: dict[int, dict] = {}

    for pic in document_dict.get("pictures", []):

        m = img_common.self_ref_to_index(
            pic["self_ref"]
        )

        if m in filtered_picture_ids:
            continue

        folder = img_common.fig_folder(m)

        prov = pic.get("prov", [{}])[0]
        page = prov.get("page_no")

        cap = _resolve_caption(
            pic,
            texts_by_ref,
            pages,
            all_texts,
        )

        # 目标 chunk：
        #   ① 含 caption 文本项 self_ref 的所有 chunk（caption 优先）
        #   ② 再并上同页正文里引用该图号（"见图3"、"图3示意说明"）
        #      的所有 chunk —— 图注文本项被 chunker 按阅读顺序并进
        #      "图后"的 chunk，而解释图的正文常在图前的相邻 chunk，
        #      只挂 caption chunk 会漏掉真正的语义上下文
        #   ③ 两者都为空时，页级兜底
        targets: list[int] = []

        if cap:
            targets = item_to_chunks.get(
                cap["ref"],
                [],
            )

            if page is not None:
                fig_no = _parse_fig_no(
                    cap["text"]
                )

                if fig_no:
                    ref_re = (
                        _figure_ref_regex(
                            fig_no
                        )
                    )

                    for i, record in enumerate(records):
                        if i in targets:
                            continue
                        if page not in record.get(
                            "page_numbers",
                            [],
                        ):
                            continue
                        if ref_re.search(
                            record.get("text", "")
                        ):
                            targets.append(i)

        if not targets and page is not None:
            targets = [
                i
                for i, record in enumerate(records)
                if page in record.get(
                    "page_numbers",
                    [],
                )
            ]

        for i in targets:
            chunk_images[i].add(folder)

        # 图片所属标题：取第一个目标 chunk 的 headings
        heading = None

        if targets:
            first = records[sorted(targets)[0]]
            headings = first.get("headings") or []
            if headings:
                heading = " / ".join(headings)

        meta_updates[m] = {
            "caption": cap,
            "heading": heading,
            "linked_chunk_ids": [
                records[i]["id"]
                for i in sorted(targets)
            ],
        }

    # 回填 records 的 images（排序去重）
    for i, record in enumerate(records):
        record["images"] = sorted(
            chunk_images.get(i, set()),
            key=_fig_index,
        )

    # 回填 metadata.json（只写关联阶段的字段）
    for m, update in meta_updates.items():
        image_dir = (
            img_common.image_folder_path(
                output_dir,
                m,
            )
        )
        meta = img_common.load_metadata(image_dir)

        if meta is None:
            print(
                f"WARNING: 缺少 "
                f"{image_dir}/metadata.json"
            )
            continue

        for key, value in update.items():
            meta[key] = value

        img_common.save_metadata(
            image_dir,
            meta,
        )

    # 摘要
    print("=" * 60)
    print("图片关联完成")
    print("=" * 60)

    for m in sorted(meta_updates):
        update = meta_updates[m]
        cap = update["caption"]
        source = (
            cap["source"]
            if cap
            else "无 caption（页级兜底）"
        )
        print(
            f"fig_{m:03d} | {source}"
            f" | caption: {cap['text'][:30] if cap else '-'}"
            f" | chunks: {len(update['linked_chunk_ids'])}"
        )

    print(
        f"有图 chunk 数    : "
        f"{sum(1 for s in chunk_images.values() if s)}"
    )

    _check_consistency(
        output_dir,
        records,
        meta_updates,
    )


def _check_consistency(
    output_dir: Path,
    records: list[dict],
    meta_updates: dict[int, dict],
) -> None:
    """校验三方对齐：chunk.images / metadata.json / 磁盘。"""

    warnings = []

    # 1. chunk.images 里的文件夹必须真实存在且含 metadata.json
    for record in records:
        for folder in record.get("images", []):
            image_dir = output_dir / folder
            if not image_dir.is_dir():
                warnings.append(
                    f"chunk {record['id']} 引用的文件夹不存在: {folder}"
                )
            elif not (image_dir / img_common.METADATA_FILE).is_file():
                warnings.append(
                    f"chunk {record['id']} 引用 {folder} 缺 metadata.json"
                )

    # 2. 反向：metadata.json 的 linked_chunk_ids 与
    #    含该文件夹的 chunk 双向一致
    for m, update in meta_updates.items():
        folder = img_common.fig_folder(m)

        actual = sorted(
            r["id"]
            for r in records
            if folder in r.get("images", [])
        )

        linked = sorted(
            update["linked_chunk_ids"]
        )

        if actual != linked:
            warnings.append(
                f"fig_{m:03d}: linked_chunk_ids 与 "
                f"chunks.images 不一致 "
                f"{linked} vs {actual}"
            )

    if warnings:
        print()
        print("一致性校验: 发现问题")
        for w in warnings:
            print(f"  - {w}")
    else:
        print()
        print("一致性校验: 通过")


# ============================================================
# 调用
# ============================================================

if __name__ == "__main__":

    build_pdf_dataset(
        pdf_file="data\压力传感器.pdf",

        # 第 6～20 页
        page_range=(9, 192),

        # 中文乱码 PDF 强制整页 OCR
        force_ocr=False,

        # 宽 × 高低于该像素面积的碎片图不导出、不关联
        min_image_area=900,
    )
