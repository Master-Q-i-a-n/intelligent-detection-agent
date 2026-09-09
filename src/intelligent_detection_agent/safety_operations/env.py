from __future__ import annotations

import os
from pathlib import Path

from ..paths import PROJECT_ROOT


def load_project_env(path: Path | None = None) -> None:
    """加载项目 .env，并保留当前终端中已经显式设置的环境变量。"""

    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))
