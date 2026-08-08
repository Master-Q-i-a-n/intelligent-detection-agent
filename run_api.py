import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime_libs"))

import uvicorn


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
