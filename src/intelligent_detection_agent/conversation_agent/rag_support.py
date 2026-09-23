from __future__ import annotations

from pathlib import Path, PurePosixPath
from urllib.parse import quote


ALLOWED_RAG_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def rag_asset_url(source: str, image_path: str) -> str:
    """生成同源图片地址，避免把服务器绝对路径暴露给浏览器。"""

    from ..rag.documents import managed_id
    if managed_id(source):
        return f"/rag/documents/{source}/assets/{quote(image_path.replace(chr(92), '/'), safe='/')}"
    return (
        f"/chat/rag-assets/{quote(source, safe='')}"
        f"/{quote(image_path.replace('\\', '/'), safe='/')}"
    )


def resolve_rag_image(root: Path, source: str, image_path: str) -> Path:
    """将 RAG Payload 中的相对路径限制在对应文档目录内。"""

    from ..rag.documents import DocumentStore, managed_id
    if managed_id(source):
        try:
            return DocumentStore(root).asset(source, image_path)
        except KeyError:
            raise FileNotFoundError("来源已删除") from None
    source_path = Path(source)
    if source_path.name != source or source_path.suffix.lower() != ".pdf":
        raise ValueError("非法知识库来源文件名。")
    if (root / "database" / "rag_documents.db").exists():
        with DocumentStore(root).connect() as db:
            row = db.execute("SELECT status FROM documents WHERE legacy=1 AND name=?", (source,)).fetchone()
        if row and row["status"] in {"deleting", "deleted"}:
            raise FileNotFoundError("来源已删除")

    normalized = image_path.replace("\\", "/").strip()
    relative = PurePosixPath(normalized)
    if not normalized or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("非法知识库图片路径。")
    if relative.suffix.lower() not in ALLOWED_RAG_IMAGE_SUFFIXES:
        raise ValueError("不支持的知识库图片格式。")

    dataset_root = (root / "dataset" / "doc").resolve()
    document_root = (dataset_root / source_path.stem).resolve()
    try:
        document_root.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError("知识库文档目录超出允许范围。") from exc

    resolved = document_root.joinpath(*relative.parts).resolve()
    try:
        resolved.relative_to(document_root)
    except ValueError as exc:
        raise ValueError("知识库图片路径超出文档目录。") from exc
    if not resolved.is_file():
        raise FileNotFoundError("知识库图片不存在。")
    return resolved
