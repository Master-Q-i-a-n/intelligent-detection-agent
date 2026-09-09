from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from intelligent_detection_agent.smart_metering import SmartMeteringService


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Run smart gas metering diagnosis")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--date", required=True, type=date.fromisoformat)
    parser.add_argument("--no-model", action="store_true", help="Disable the deep gas-state model")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    service = SmartMeteringService(use_deep_model=not args.no_model)
    result = service.diagnose(args.user_id, args.date, save=not args.no_save)
    text = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output)
    else:
        print(text)


if __name__ == "__main__":
    main()
