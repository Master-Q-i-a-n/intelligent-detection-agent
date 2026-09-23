"""知识库管理接口；上传只接收和校验文件，耗时工作由独立进程执行。"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pypdf import PdfReader
from starlette.concurrency import run_in_threadpool

from .documents import DocumentStore, new_document_id
from .worker import parser_python, safe_remove


def inspect_pdf(path: Path, start: int | None, end: int | None):
    try:
        with path.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise ValueError("文件内容不是 PDF")
            handle.seek(0)
            reader = PdfReader(handle)
            if reader.is_encrypted:
                raise ValueError("请先移除 PDF 密码后再上传")
            total = len(reader.pages)
        if total == 0:
            raise ValueError("PDF 没有页面")
    except ValueError:
        raise
    except Exception:
        raise ValueError("PDF 已损坏或无法读取") from None
    if (start is None) != (end is None):
        raise ValueError("请同时填写起始页与结束页")
    start, end = (1, total) if start is None else (start, end)
    if not 1 <= start <= end <= total:
        raise ValueError(f"页码必须满足 1 ≤ 起始页 ≤ 结束页 ≤ {total}（PDF 实际页序）")
    return start, end, total


def create_rag_router(root: Path, *, start_worker: bool = True):
    store = DocumentStore(root)

    def require_user(request: Request):
        if not getattr(request.state, "user", None):
            raise HTTPException(401, "请先登录")

    worker = None
    stop_file = store.path.parent / f"rag-worker-{uuid.uuid4().hex}.stop"

    def startup():
        nonlocal worker
        if start_worker:
            env = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONUTF8": "1"}
            worker = subprocess.Popen([sys.executable, "-m", "intelligent_detection_agent.rag.worker",
                                       "--root", str(root), "--stop-file", str(stop_file)], cwd=root,
                                      env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                      creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)

    def shutdown():
        if worker is not None:
            # 在途模型请求结束后自行释放锁；不能强杀后留下解析子进程。
            stop_file.touch()
            try:
                worker.wait(timeout=5)
                stop_file.unlink(missing_ok=True)
            except subprocess.TimeoutExpired:
                pass

    @asynccontextmanager
    async def lifespan(_app):
        startup()
        try:
            yield
        finally:
            await run_in_threadpool(shutdown)

    router = APIRouter(prefix="/rag", dependencies=[Depends(require_user)], lifespan=lifespan)

    def get(document_id: str):
        try:
            row = store.get(document_id)
        except KeyError:
            raise HTTPException(404, "文档不存在") from None
        return row

    @router.get("/documents")
    def listing():
        with store.connect() as db:
            error = db.execute("SELECT value FROM settings WHERE key='worker_error'").fetchone()
        worker_error = error[0] if error else ""
        if worker is not None and worker.poll() is not None:
            worker_error = "知识库后台进程已退出，请重启后端后继续任务"
        return {"items": store.list(), "worker_error": worker_error,
                "parser_available": parser_python(root).is_file(),
                "upload_limit_mb": int(os.getenv("RAG_UPLOAD_MAX_MB", "100"))}

    @router.post("/documents", status_code=202)
    async def upload(file: UploadFile = File(), start_page: int | None = Form(None),
                     end_page: int | None = Form(None), force_ocr: bool = Form(False)):
        name = (file.filename or "").replace("\\", "/").split("/")[-1]
        if not name.lower().endswith(".pdf") or len(name) > 240:
            raise HTTPException(400, "请选择文件名不超过240字的 PDF")
        document_id = new_document_id()
        directory = root / "dataset" / "doc" / document_id
        directory.mkdir(parents=True)
        digest, size = hashlib.sha256(), 0
        try:
            with (directory / "original.pdf").open("wb") as handle:
                while block := await file.read(1024 * 1024):
                    size += len(block)
                    if size > int(os.getenv("RAG_UPLOAD_MAX_MB", "100")) * 1024 * 1024:
                        raise HTTPException(413, "PDF 超过上传大小限制")
                    handle.write(block)
                    digest.update(block)
            start, end, total = await run_in_threadpool(inspect_pdf, directory / "original.pdf", start_page, end_page)
            row = store.add(document_id=document_id, name=name, file_hash=digest.hexdigest(),
                            start=start, end=end, total=total, force_ocr=force_ocr, directory=directory)
        except Exception as exc:
            safe_remove(directory, root)
            if isinstance(exc, ValueError):
                raise HTTPException(400, str(exc)) from None
            raise
        finally:
            await file.close()
        duplicate = row["id"] != document_id
        if duplicate:
            safe_remove(directory, root)
        return JSONResponse({**store.public(row), "duplicate": duplicate}, status_code=200 if duplicate else 202)

    @router.get("/documents/{document_id}")
    def detail(document_id: str):
        return store.public(get(document_id))

    @router.post("/documents/{document_id}/retry", status_code=202)
    def retry(document_id: str):
        get(document_id)
        if not store.update(document_id, expected=("failed",), status="queued", error="", retry_after=0):
            raise HTTPException(409, "仅失败任务可以重试")
        return store.public(store.get(document_id))

    @router.delete("/documents/{document_id}", status_code=202)
    def delete(document_id: str):
        row = get(document_id)
        if row["status"] != "deleted":
            store.update(document_id, status="deleting", error="", retry_after=0)
        return store.public(store.get(document_id))

    @router.get("/documents/{document_id}/file")
    def original(document_id: str):
        row = get(document_id)
        try:
            path = store.asset(document_id, row["name"] if row["legacy"] else "original.pdf")
        except (FileNotFoundError, ValueError):
            raise HTTPException(404, "原文件不存在或来源已删除") from None
        return FileResponse(path, media_type="application/pdf", filename=row["name"], content_disposition_type="inline")

    @router.get("/documents/{document_id}/assets/{relative:path}")
    def asset(document_id: str, relative: str):
        get(document_id)
        if Path(relative).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            raise HTTPException(400, "仅支持图片文件")
        try:
            path = store.asset(document_id, relative)
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, "图片不存在或来源已删除") from None
        return FileResponse(path)

    return router
