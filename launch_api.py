import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000

stdout = open(LOG_DIR / "api_stdout.log", "ab", buffering=0)
stderr = open(LOG_DIR / "api_stderr.log", "ab", buffering=0)
process = subprocess.Popen(
    [sys.executable, str(ROOT / "run_api.py")],
    cwd=str(ROOT),
    stdin=subprocess.DEVNULL,
    stdout=stdout,
    stderr=stderr,
    close_fds=True,
    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
)
print(process.pid)
