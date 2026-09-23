"""持久化目录驱动的单 worker；Docling 子进程与50并发识图共享文档状态。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from qdrant_client import models

from ..paths import PROJECT_ROOT
from ..safety_operations.env import load_project_env
from .documents import DocumentStore
from .pipeline import RagConfig, RagPipeline, stable_point_id
from .vision import Cancelled, atomic_json, describe_images


def parser_python(root: Path) -> Path:
    configured = os.getenv("RAG_DOCLING_PYTHON")
    return Path(configured) if configured else root.parent / "docling" / ".venv" / "Scripts" / "python.exe"


def safe_remove(directory: Path, root: Path):
    """只清理已验证的文档子目录，拒绝符号链接及根目录。"""
    base = (root / "dataset" / "doc").resolve()
    if directory.is_symlink() or directory.resolve() == base or base not in directory.resolve().parents:
        raise ValueError("拒绝清理知识库目录之外的文件")
    if directory.exists():
        shutil.rmtree(directory)


def migrate_legacy(store: DocumentStore, pipeline: RagPipeline):
    """保留现有 point ID；逐份补齐目录，不重跑识图和 Embedding。"""
    with store.connect() as db:
        if db.execute("SELECT 1 FROM settings WHERE key='legacy_migrated'").fetchone():
            return
    pipeline.check_qdrant()
    if pipeline._collection_exists():
        pipeline._validate_collection()
        offset = None
        sources = {}
        while True:
            points, offset = pipeline.qdrant.scroll(collection_name=pipeline.config.collection_name,
                                                      limit=200, offset=offset, with_payload=True, with_vectors=False)
            for point in points:
                payload = point.payload or {}
                if payload.get("document_id"):
                    continue
                source = str(payload.get("source", ""))
                if Path(source).name != source or not source.lower().endswith(".pdf"):
                    raise ValueError("旧知识库存在非法 source，迁移已暂停")
                group = sources.setdefault(source, {"ids": [], "pages": set()})
                group["ids"].append(point.id)
                group["pages"].update(payload.get("page_numbers") or [])
            if offset is None:
                break
        for source, group in sources.items():
            document_id = "doc_" + uuid.uuid5(uuid.NAMESPACE_URL, "legacy:" + source).hex
            directory = (store.root / "dataset" / "doc" / Path(source).stem).resolve()
            base = (store.root / "dataset" / "doc").resolve()
            if base not in directory.parents:
                raise ValueError("旧文档目录不合法")
            pages = sorted(group["pages"])
            with store.connect() as db:
                db.execute("INSERT OR IGNORE INTO documents "
                           "(id,name,directory,status,stage,start_page,end_page,chunk_count,legacy) "
                           "VALUES (?,?,?,'migrating','index',?,?,?,1)",
                           (document_id, source, str(directory), pages[0] if pages else None,
                            pages[-1] if pages else None, len(group["ids"])))
            pipeline.qdrant.set_payload(collection_name=pipeline.config.collection_name,
                                        payload={"document_id": document_id}, points=group["ids"], wait=True)
            store.update(document_id, expected=("migrating",), status="ready", stage="ready")
        # 若在 payload 已写入、状态未发布之间退出，继续完成已登记的旧条目。
        with store.connect() as db:
            db.execute("UPDATE documents SET status='ready',stage='ready' WHERE status='migrating'")
    with store.connect() as db:
        db.execute("INSERT OR REPLACE INTO settings VALUES ('legacy_migrated','1')")


def process_document(store: DocumentStore, document_id: str, pipeline: RagPipeline, stopped=lambda: False):
    directory = store.directory(document_id)

    def check():
        if stopped() or store.get(document_id)["status"] != "running":
            raise Cancelled()

    def update(**values):
        check()
        if not store.update(document_id, expected=("running",), **values):
            raise Cancelled()

    row = store.get(document_id)
    check()
    if row["stage"] == "build":
        executable = parser_python(store.root)
        if not executable.is_file():
            raise ValueError("Docling Python 不存在，请配置 RAG_DOCLING_PYTHON")
        # 阶段重试必须移除不完整解析产物，原PDF和其它文档不受影响。
        for name in ("parsed", "images", "chunks", "tables", "vision_cache", "ingest"):
            safe_remove(directory / name, store.root)
        spec = directory / "build_request.json"
        atomic_json(spec, {"pdf": str(directory / "original.pdf"), "output": str(directory),
                           "page_range": [row["start_page"], row["end_page"]], "force_ocr": bool(row["force_ocr"])})
        script = Path(__file__).parent / "docling_worker" / "run.py"
        env = {**os.environ, "PYTHONUTF8": "1"}
        # 解析环境无需云端密钥；不向子进程透传任何 API 凭据。
        env = {key: value for key, value in env.items() if not any(word in key.upper() for word in ("API_KEY", "TOKEN", "SECRET"))}
        with (directory / "build.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([str(executable), str(script), str(spec)], stdout=log,
                                       stderr=subprocess.STDOUT, env=env,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            try:
                started = time.monotonic()
                while process.poll() is None:
                    check()
                    if time.monotonic() - started > 3600:
                        raise ValueError("PDF 解析超过60分钟，请缩小页码范围")
                    time.sleep(0.25)
                if process.returncode:
                    raise ValueError("Docling 解析失败，请检查独立环境及模型缓存；详见该文档 build.log")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        # source 保留上传文件名，避免所有文档都显示 original.pdf。
        path = directory / "chunks" / "chunks.jsonl"
        chunks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not chunks:
            raise ValueError("所选范围没有可索引文本，请检查页码或尝试强制 OCR")
        extracted = "\n".join(chunk.get("text", "") for chunk in chunks)
        encoded = re.findall(r"(?:G[0-9A-F]{2,}){3,}", extracted)
        if sum(map(len, encoded)) > max(60, len(extracted) * 0.15):
            raise ValueError("检测到 PDF 文本层编码异常；请删除此条目后开启强制 OCR 重新上传")
        for chunk in chunks:
            chunk["source"] = row["name"]
            if any(not row["start_page"] <= page <= row["end_page"] for page in chunk.get("page_numbers", [])):
                raise ValueError("解析结果页码与请求范围不一致")
        path.write_text("\n".join(json.dumps(chunk, ensure_ascii=False) for chunk in chunks) + "\n", encoding="utf-8")
        update(stage="describe")

    if store.get(document_id)["stage"] == "describe":
        describe_images(directory, check_cancelled=check,
                        progress=lambda done, total, errors: update(image_done=done, image_count=total,
                                                                   image_errors=json.dumps(errors, ensure_ascii=False)))
        update(stage="ingest")

    if store.get(document_id)["stage"] == "ingest":
        from .docling_worker.ingest import ingest
        result = ingest(directory, model=os.getenv("RAG_VISION_MODEL", "deepseek-v4-flash-vision-exp"))
        if not result["record_count"]:
            raise ValueError("文档未生成有效分块")
        update(stage="index", chunk_count=result["record_count"], indexed_count=0)

    if store.get(document_id)["stage"] == "index":
        row = store.get(document_id)
        pipeline.index_records(directory / "ingest" / "records.json", document_id=document_id,
                               start_index=row["indexed_count"], check_cancelled=check,
                               progress=lambda count: update(indexed_count=count))
        update(stage="ready", status="ready", error="")


def delete_document(store: DocumentStore, document_id: str, pipeline: RagPipeline):
    pipeline.check_qdrant()
    if pipeline._collection_exists():
        pipeline.qdrant.delete(collection_name=pipeline.config.collection_name, wait=True,
                               points_selector=models.FilterSelector(filter=models.Filter(must=[models.FieldCondition(
                                   key="document_id", match=models.MatchValue(value=document_id))])))
    safe_remove(store.directory(document_id), store.root)
    store.update(document_id, expected=("deleting",), status="deleted", error="", fingerprint=None)


def run(root: Path, stop_file: Path):
    load_project_env(root / ".env")
    store = DocumentStore(root)
    lock_path = store.path.with_suffix(".lock")
    with lock_path.open("a+b") as lock:
        # OS 文件锁覆盖整个 worker 生命周期，保证全局最多50个视觉请求。
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        while not stop_file.exists():
            try:
                lock.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                time.sleep(1)
        else:
            return
        with store.connect() as db:
            db.execute("UPDATE documents SET status='queued' WHERE status='running'")
        pipeline = RagPipeline(RagConfig.from_env(), root=root)
        while not stop_file.exists():
            try:
                migrate_legacy(store, pipeline)
                with store.connect() as db:
                    db.execute("DELETE FROM settings WHERE key='worker_error'")
                    row = db.execute("SELECT * FROM documents WHERE status IN ('queued','deleting') "
                                     "AND retry_after<=? ORDER BY status='deleting' DESC,created_at LIMIT 1", (time.time(),)).fetchone()
                if not row:
                    time.sleep(1)
                    continue
                document_id = row["id"]
                try:
                    if row["status"] == "deleting":
                        delete_document(store, document_id, pipeline)
                    elif store.update(document_id, expected=("queued",), status="running", error=""):
                        process_document(store, document_id, pipeline, stopped=stop_file.exists)
                except Cancelled:
                    store.update(document_id, expected=("running",), status="queued")
                except Exception as exc:
                    if store.get(document_id)["status"] == "deleting":
                        store.update(document_id, expected=("deleting",), error="删除未完成，30秒后重试", retry_after=time.time() + 30)
                    else:
                        stage = store.get(document_id)["stage"]
                        message = str(exc) if isinstance(exc, ValueError) else f"{stage} 阶段失败（{type(exc).__name__}），请重试"
                        store.update(document_id, expected=("running",), status="failed", error=message[:500])
            except Exception as exc:
                with store.connect() as db:
                    db.execute("INSERT OR REPLACE INTO settings VALUES ('worker_error',?)",
                               (f"知识库初始化失败（{type(exc).__name__}），请检查 Qdrant；后台将自动重试",))
                for _ in range(30):
                    if stop_file.exists():
                        break
                    time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--stop-file", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.stop_file)
