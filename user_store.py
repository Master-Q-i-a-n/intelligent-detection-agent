from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


SESSION_DAYS = 7
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_\-\u4e00-\u9fff]{3,32}$")


def _now() -> datetime:
    return datetime.now(UTC)


def _timestamp(value: datetime | None = None) -> str:
    return (value or _now()).isoformat(timespec="seconds")


def _password_digest(password: str, salt: bytes) -> bytes:
    """使用标准库 Scrypt 保存密码，避免数据库中出现可逆凭据。"""

    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)


def _session_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class UserStore:
    """用户、浏览器会话和可展示对话历史的 SQLite 数据层。"""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize(self) -> None:
        with self._lock, self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    password_salt TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    disabled INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS auth_sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_auth_sessions_user ON auth_sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_auth_sessions_expiry ON auth_sessions(expires_at);

                CREATE TABLE IF NOT EXISTS chat_threads (
                    thread_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    pending_turn_id TEXT,
                    todos_json TEXT NOT NULL DEFAULT '[]',
                    interrupt_json TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chat_threads_user_updated
                    ON chat_threads(user_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL REFERENCES chat_threads(thread_id) ON DELETE CASCADE,
                    turn_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    generator TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(thread_id, sequence)
                );
                CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_sequence
                    ON chat_messages(thread_id, sequence);

                CREATE TABLE IF NOT EXISTS chat_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL REFERENCES chat_threads(thread_id) ON DELETE CASCADE,
                    turn_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chat_artifacts_thread
                    ON chat_artifacts(thread_id, created_at);
                """
            )
            self._seed_demo_user(connection)

    def _seed_demo_user(self, connection: sqlite3.Connection) -> None:
        existing = connection.execute(
            "SELECT 1 FROM users WHERE username=? COLLATE NOCASE", ("admin",)
        ).fetchone()
        if existing:
            return
        self._insert_user(connection, "admin", "123456")

    @staticmethod
    def validate_credentials(username: str, password: str) -> tuple[str, str]:
        normalized = username.strip()
        if not USERNAME_PATTERN.fullmatch(normalized):
            raise ValueError("用户名需为 3–32 位中英文、数字、下划线或连字符。")
        if not 6 <= len(password) <= 72:
            raise ValueError("密码长度需为 6–72 位。")
        return normalized, password

    def _insert_user(self, connection: sqlite3.Connection, username: str, password: str) -> dict[str, str]:
        salt = secrets.token_bytes(16)
        digest = _password_digest(password, salt)
        now = _timestamp()
        user_id = f"usr_{uuid.uuid4().hex}"
        connection.execute(
            """
            INSERT INTO users(user_id,username,password_salt,password_hash,created_at,updated_at)
            VALUES(?,?,?,?,?,?)
            """,
            (user_id, username, salt.hex(), digest.hex(), now, now),
        )
        return {"user_id": user_id, "username": username}

    def register(self, username: str, password: str) -> dict[str, str]:
        username, password = self.validate_credentials(username, password)
        with self._lock, self.connect() as connection:
            try:
                return self._insert_user(connection, username, password)
            except sqlite3.IntegrityError as exc:
                raise ValueError("用户名已存在。") from exc

    def authenticate(self, username: str, password: str) -> dict[str, str] | None:
        normalized = username.strip()
        if not normalized or not password:
            return None
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT user_id,username,password_salt,password_hash,disabled
                FROM users WHERE username=? COLLATE NOCASE
                """,
                (normalized,),
            ).fetchone()
        if not row or row["disabled"]:
            return None
        try:
            actual = _password_digest(password, bytes.fromhex(row["password_salt"]))
            expected = bytes.fromhex(row["password_hash"])
        except ValueError:
            return None
        if not hmac.compare_digest(actual, expected):
            return None
        return {"user_id": row["user_id"], "username": row["username"]}

    def create_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        now = _now()
        with self._lock, self.connect() as connection:
            connection.execute("DELETE FROM auth_sessions WHERE expires_at<=?", (_timestamp(now),))
            # 同一账号重新登录即轮换浏览器会话，旧 Cookie 立即失效。
            connection.execute("DELETE FROM auth_sessions WHERE user_id=?", (user_id,))
            connection.execute(
                """
                INSERT INTO auth_sessions(token_hash,user_id,created_at,expires_at,last_seen_at)
                VALUES(?,?,?,?,?)
                """,
                (
                    _session_digest(token),
                    user_id,
                    _timestamp(now),
                    _timestamp(now + timedelta(days=SESSION_DAYS)),
                    _timestamp(now),
                ),
            )
        return token

    def session_user(self, token: str | None) -> dict[str, str] | None:
        if not token:
            return None
        now = _timestamp()
        token_hash = _session_digest(token)
        with self._lock, self.connect() as connection:
            row = connection.execute(
                """
                SELECT u.user_id,u.username,s.expires_at,u.disabled
                FROM auth_sessions s JOIN users u ON u.user_id=s.user_id
                WHERE s.token_hash=?
                """,
                (token_hash,),
            ).fetchone()
            if not row or row["disabled"] or row["expires_at"] <= now:
                connection.execute("DELETE FROM auth_sessions WHERE token_hash=?", (token_hash,))
                return None
            connection.execute(
                "UPDATE auth_sessions SET last_seen_at=? WHERE token_hash=?", (now, token_hash)
            )
        return {"user_id": row["user_id"], "username": row["username"]}

    def logout(self, token: str | None) -> None:
        if not token:
            return
        with self._lock, self.connect() as connection:
            connection.execute("DELETE FROM auth_sessions WHERE token_hash=?", (_session_digest(token),))

    @staticmethod
    def _title(message: str) -> str:
        compact = " ".join(message.split())
        return compact[:28] + ("…" if len(compact) > 28 else "")

    @staticmethod
    def _next_sequence(connection: sqlite3.Connection, thread_id: str) -> int:
        row = connection.execute(
            "SELECT COALESCE(MAX(sequence),0)+1 FROM chat_messages WHERE thread_id=?", (thread_id,)
        ).fetchone()
        return int(row[0])

    def start_turn(self, user_id: str, thread_id: str, message: str) -> str:
        """建立或校验线程，并在 Agent 执行前保存用户问题。"""

        with self._lock, self.connect() as connection:
            row = connection.execute(
                "SELECT user_id,pending_turn_id FROM chat_threads WHERE thread_id=?", (thread_id,)
            ).fetchone()
            now = _timestamp()
            if row and row["user_id"] != user_id:
                raise PermissionError("无权访问该对话。")
            if row and row["pending_turn_id"]:
                raise ValueError("该对话仍有待处理的补充信息或工单确认。")
            turn_id = f"turn_{uuid.uuid4().hex}"
            if not row:
                connection.execute(
                    """
                    INSERT INTO chat_threads(thread_id,user_id,title,pending_turn_id,created_at,updated_at)
                    VALUES(?,?,?,?,?,?)
                    """,
                    (thread_id, user_id, self._title(message) or "新对话", turn_id, now, now),
                )
            else:
                connection.execute(
                    """
                    UPDATE chat_threads
                    SET pending_turn_id=?,interrupt_json=NULL,last_error=NULL,updated_at=?
                    WHERE thread_id=?
                    """,
                    (turn_id, now, thread_id),
                )
            connection.execute(
                """
                INSERT INTO chat_messages(message_id,thread_id,turn_id,sequence,role,content,created_at)
                VALUES(?,?,?,?,?,?,?)
                """,
                (
                    f"msg_{uuid.uuid4().hex}",
                    thread_id,
                    turn_id,
                    self._next_sequence(connection, thread_id),
                    "user",
                    message,
                    now,
                ),
            )
        return turn_id

    def prepare_resume(self, user_id: str, thread_id: str, message: str | None = None) -> str:
        with self._lock, self.connect() as connection:
            row = connection.execute(
                "SELECT user_id,pending_turn_id FROM chat_threads WHERE thread_id=?", (thread_id,)
            ).fetchone()
            if not row or row["user_id"] != user_id:
                raise PermissionError("无权访问该对话。")
            turn_id = row["pending_turn_id"]
            if not turn_id:
                raise ValueError("该对话当前没有待恢复操作。")
            if message and message.strip():
                connection.execute(
                    """
                    INSERT INTO chat_messages(message_id,thread_id,turn_id,sequence,role,content,created_at)
                    VALUES(?,?,?,?,?,?,?)
                    """,
                    (
                        f"msg_{uuid.uuid4().hex}",
                        thread_id,
                        turn_id,
                        self._next_sequence(connection, thread_id),
                        "user",
                        message.strip(),
                        _timestamp(),
                    ),
                )
        return str(turn_id)

    def record_response(self, user_id: str, thread_id: str, response: dict[str, Any]) -> None:
        with self._lock, self.connect() as connection:
            row = connection.execute(
                "SELECT user_id,pending_turn_id FROM chat_threads WHERE thread_id=?", (thread_id,)
            ).fetchone()
            if not row or row["user_id"] != user_id:
                raise PermissionError("无权访问该对话。")
            turn_id = row["pending_turn_id"] or f"turn_{uuid.uuid4().hex}"
            now = _timestamp()
            message = str(response.get("message") or "").strip()
            if message:
                connection.execute(
                    """
                    INSERT INTO chat_messages
                    (message_id,thread_id,turn_id,sequence,role,content,generator,created_at)
                    VALUES(?,?,?,?,?,?,?,?)
                    """,
                    (
                        f"msg_{uuid.uuid4().hex}",
                        thread_id,
                        turn_id,
                        self._next_sequence(connection, thread_id),
                        "assistant",
                        message,
                        str(response.get("generator") or ""),
                        now,
                    ),
                )
            for artifact in response.get("artifacts") or []:
                artifact_id = str(artifact.get("id") or uuid.uuid4().hex)
                connection.execute(
                    """
                    INSERT INTO chat_artifacts
                    (artifact_id,thread_id,turn_id,artifact_type,payload_json,created_at)
                    VALUES(?,?,?,?,?,?)
                    ON CONFLICT(artifact_id) DO UPDATE SET
                      payload_json=excluded.payload_json,
                      artifact_type=excluded.artifact_type
                    """,
                    (
                        artifact_id,
                        thread_id,
                        turn_id,
                        str(artifact.get("type") or "query_result"),
                        json.dumps(artifact.get("payload") or {}, ensure_ascii=False, default=str),
                        now,
                    ),
                )
            interrupted = response.get("status") == "interrupted"
            connection.execute(
                """
                UPDATE chat_threads
                SET pending_turn_id=?,todos_json=?,interrupt_json=?,last_error=NULL,updated_at=?
                WHERE thread_id=?
                """,
                (
                    turn_id if interrupted else None,
                    json.dumps(response.get("todos") or [], ensure_ascii=False, default=str),
                    json.dumps(response.get("interrupt"), ensure_ascii=False, default=str)
                    if interrupted and response.get("interrupt")
                    else None,
                    now,
                    thread_id,
                ),
            )

    def record_error(self, user_id: str, thread_id: str, message: str, *, keep_pending: bool = False) -> None:
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                UPDATE chat_threads
                SET last_error=?,pending_turn_id=CASE WHEN ? THEN pending_turn_id ELSE NULL END,updated_at=?
                WHERE thread_id=? AND user_id=?
                """,
                (message[:500], int(keep_pending), _timestamp(), thread_id, user_id),
            )

    def list_threads(self, user_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT thread_id,title,created_at,updated_at,interrupt_json,last_error
                FROM chat_threads WHERE user_id=? ORDER BY updated_at DESC LIMIT 100
                """,
                (user_id,),
            ).fetchall()
        return [
            {
                "thread_id": row["thread_id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "status": "interrupted" if row["interrupt_json"] else "error" if row["last_error"] else "completed",
            }
            for row in rows
        ]

    def thread_detail(self, user_id: str, thread_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            thread = connection.execute(
                """
                SELECT thread_id,title,created_at,updated_at,todos_json,interrupt_json,last_error
                FROM chat_threads WHERE thread_id=? AND user_id=?
                """,
                (thread_id, user_id),
            ).fetchone()
            if not thread:
                return None
            message_rows = connection.execute(
                """
                SELECT message_id,turn_id,role,content,generator,created_at
                FROM chat_messages WHERE thread_id=? ORDER BY sequence
                """,
                (thread_id,),
            ).fetchall()
            artifact_rows = connection.execute(
                """
                SELECT artifact_id,turn_id,artifact_type,payload_json
                FROM chat_artifacts WHERE thread_id=? ORDER BY created_at
                """,
                (thread_id,),
            ).fetchall()
        artifacts = [
            {
                "id": row["artifact_id"],
                "type": row["artifact_type"],
                "payload": json.loads(row["payload_json"]),
                "turn_id": row["turn_id"],
            }
            for row in artifact_rows
        ]
        artifact_ids_by_turn: dict[str, list[str]] = {}
        for artifact in artifacts:
            artifact_ids_by_turn.setdefault(str(artifact.pop("turn_id")), []).append(str(artifact["id"]))
        messages = [
            {
                "id": row["message_id"],
                "role": row["role"],
                "content": row["content"],
                "generator": row["generator"],
                "artifact_ids": artifact_ids_by_turn.get(row["turn_id"], []) if row["role"] == "assistant" else [],
                "created_at": row["created_at"],
            }
            for row in message_rows
        ]
        return {
            "thread": {
                "thread_id": thread["thread_id"],
                "title": thread["title"],
                "created_at": thread["created_at"],
                "updated_at": thread["updated_at"],
            },
            "messages": messages,
            "artifacts": artifacts,
            "todos": json.loads(thread["todos_json"] or "[]"),
            "interrupt": json.loads(thread["interrupt_json"]) if thread["interrupt_json"] else None,
            "last_error": thread["last_error"],
        }

    def owns_thread(self, user_id: str, thread_id: str) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM chat_threads WHERE thread_id=? AND user_id=?", (thread_id, user_id)
            ).fetchone()
        return bool(row)

    def delete_thread_records(self, user_id: str, thread_id: str) -> bool:
        with self._lock, self.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM chat_threads WHERE thread_id=? AND user_id=?", (thread_id, user_id)
            )
        return cursor.rowcount > 0
