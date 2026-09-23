"""知识库目录与持久化任务。状态写入使用短事务，不与模型调用共用连接。"""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path


class DocumentStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.path = self.root / "database" / "rag_documents.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, fingerprint TEXT UNIQUE,
                    file_hash TEXT, start_page INTEGER, end_page INTEGER, total_pages INTEGER,
                    force_ocr INTEGER NOT NULL DEFAULT 0, directory TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued', stage TEXT NOT NULL DEFAULT 'build',
                    image_count INTEGER NOT NULL DEFAULT 0, image_done INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0, indexed_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '', image_errors TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
                    retry_after REAL NOT NULL DEFAULT 0, legacy INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def public(row):
        item = dict(row)
        item.pop("directory", None)
        item.pop("fingerprint", None)
        item.pop("file_hash", None)
        item["image_errors"] = json.loads(item["image_errors"])
        return item

    def get(self, document_id: str):
        with self.connect() as db:
            row = db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        if row is None:
            raise KeyError(document_id)
        return dict(row)

    def list(self):
        with self.connect() as db:
            return [self.public(row) for row in db.execute(
                "SELECT * FROM documents WHERE status!='deleted' ORDER BY created_at DESC,id"
            )]

    def add(self, *, document_id, name, file_hash, start, end, total, force_ocr, directory):
        # 相同文件与页码范围不重复处理，与文件名无关。
        fingerprint = f"{file_hash}:{start}:{end}"
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO documents "
                       "(id,name,fingerprint,file_hash,start_page,end_page,total_pages,force_ocr,directory) "
                       "VALUES (?,?,?,?,?,?,?,?,?)",
                       (document_id, name, fingerprint, file_hash, start, end, total, int(force_ocr), str(directory)))
            return dict(db.execute("SELECT * FROM documents WHERE fingerprint=?", (fingerprint,)).fetchone())

    def update(self, document_id: str, *, expected: tuple[str, ...] | None = None, **fields):
        allowed = {"status", "stage", "image_count", "image_done", "chunk_count", "indexed_count",
                   "error", "image_errors", "retry_after", "fingerprint"}
        if not fields or set(fields) - allowed:
            raise ValueError("非法任务字段")
        sql = "UPDATE documents SET " + ",".join(f"{key}=?" for key in fields)
        sql += ",updated_at=strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id=?"
        values = [*fields.values(), document_id]
        if expected:
            sql += " AND status IN (" + ",".join("?" for _ in expected) + ")"
            values.extend(expected)
        with self.connect() as db:
            return db.execute(sql, values).rowcount > 0

    def directory(self, document_id: str) -> Path:
        original = Path(self.get(document_id)["directory"])
        if original.is_symlink():
            raise ValueError("文档目录不能是符号链接")
        directory = original.resolve()
        # 所有文件操作只能落在知识库目录的子目录，不能删除知识库根目录。
        base = (self.root / "dataset" / "doc").resolve()
        if directory == base or base not in directory.parents:
            raise ValueError("文档路径超出知识库目录")
        return directory

    def asset(self, document_id: str, relative: str) -> Path:
        row = self.get(document_id)
        if row["status"] in {"deleting", "deleted"}:
            raise FileNotFoundError("来源已删除")
        directory = self.directory(document_id)
        path = (directory / relative).resolve()
        if directory not in path.parents or not path.is_file():
            raise FileNotFoundError("文档文件不存在")
        return path

    def ready_ids(self):
        with self.connect() as db:
            return [row[0] for row in db.execute("SELECT id FROM documents WHERE status='ready'")]


def managed_id(value: str) -> bool:
    return bool(re.fullmatch(r"doc_[0-9a-f]{32}", value))


def new_document_id() -> str:
    return "doc_" + uuid.uuid4().hex
