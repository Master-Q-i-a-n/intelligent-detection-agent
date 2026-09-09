from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
import yaml

from .db import connect_database, utc_now
from .env import load_project_env


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def dispatch_pending_alerts(config_path: Path) -> int:
    """投递 Agent 发件箱；失败记录留在 SQLite，供后续显式或自动重试。"""

    load_project_env()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    notify_cfg = config.get("agent_notification", {})
    if not notify_cfg.get("enabled", True):
        return 0
    token = os.getenv(str(notify_cfg.get("token_env", "SAFETY_AGENT_TOKEN")), "").strip()
    if not token:
        print("未设置 SAFETY_AGENT_TOKEN，Agent 通知保留为 PENDING。", file=sys.stderr)
        return 0

    database_cfg = config.get("database", {})
    database_path = _resolve(config_path.parent, database_cfg.get("path", "data/security.db"))
    connection = connect_database(database_path, int(database_cfg.get("busy_timeout_ms", 5000)))
    endpoint = str(notify_cfg.get("endpoint", "http://127.0.0.1:8000/internal/security/events"))
    timeout = float(notify_cfg.get("timeout_seconds", 10))
    max_attempts = int(notify_cfg.get("max_attempts", 5))
    now = utc_now()
    rows = connection.execute(
        """
        SELECT alert_id,payload_json,attempt_count
        FROM alert_records
        WHERE channel='AGENT_WEBHOOK' AND status IN ('PENDING','FAILED')
          AND attempt_count<? AND (next_retry_at IS NULL OR next_retry_at<=?)
        ORDER BY created_at
        """,
        (max_attempts, now),
    ).fetchall()
    failures = 0
    for row in rows:
        alert_id = str(row["alert_id"])
        attempt = int(row["attempt_count"] or 0) + 1
        attempted_at = utc_now()
        try:
            request = urllib.request.Request(
                endpoint,
                data=str(row["payload_json"]).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status < 200 or response.status >= 300:
                    raise RuntimeError(f"Agent 返回 HTTP {response.status}")
            with connection:
                connection.execute(
                    """
                    UPDATE alert_records SET status='SENT',sent_at=COALESCE(sent_at,?),
                        attempt_count=?,last_attempt_at=?,next_retry_at=NULL,
                        error_message=NULL,updated_at=? WHERE alert_id=?
                    """,
                    (attempted_at, attempt, attempted_at, attempted_at, alert_id),
                )
            print(f"Agent 通知已送达: {alert_id}")
        except Exception as exc:
            failures += 1
            delay_seconds = min(300, 2 * (5 ** max(0, attempt - 1)))
            retry_at = (datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)).isoformat(
                timespec="seconds"
            ).replace("+00:00", "Z")
            with connection:
                connection.execute(
                    """
                    UPDATE alert_records SET status='FAILED',attempt_count=?,last_attempt_at=?,
                        next_retry_at=?,error_message=?,updated_at=? WHERE alert_id=?
                    """,
                    (attempt, attempted_at, retry_at, f"{type(exc).__name__}: {exc}", attempted_at, alert_id),
                )
            print(f"Agent 通知失败，将保留重试 [{alert_id}]: {exc}", file=sys.stderr)
    connection.close()
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="重试安全作业 Agent 发件箱")
    parser.add_argument("--config", default="safety_operations/config.yaml")
    args = parser.parse_args()
    raise SystemExit(1 if dispatch_pending_alerts(Path(args.config)) else 0)


if __name__ == "__main__":
    main()
