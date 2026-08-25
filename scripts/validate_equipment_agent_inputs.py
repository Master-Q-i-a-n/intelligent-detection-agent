from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "agent_inputs" / "equipment_health"
REPORT_PATH = ROOT / "reports" / "equipment_agent_input_validation.json"
REQUIRED_CURRENT = {
    "date", "stage", "stage_name", "health_index", "confidence",
    "probabilities", "ordinal_exceedance", "risk_level",
}


def main() -> None:
    index = json.loads((INPUT_ROOT / "index.json").read_text(encoding="utf-8"))
    files = sorted((INPUT_ROOT / "users").glob("*.json"))
    errors = []
    for path in files:
        document = json.loads(path.read_text(encoding="utf-8"))
        user_id = document.get("entity", {}).get("user_id")
        history = document.get("daily_history", [])
        dates = [row.get("date") for row in history]
        if document.get("schema") != "gas-agent.equipment-health.v1":
            errors.append(f"{path.name}: schema错误")
        if path.stem != user_id:
            errors.append(f"{path.name}: user_id不匹配")
        if len(history) != 19 or len(set(dates)) != 19:
            errors.append(f"{path.name}: 日期历史不完整")
        if dates and (min(dates) != "2024-12-25" or max(dates) != "2025-01-12"):
            errors.append(f"{path.name}: 日期范围错误")
        if not REQUIRED_CURRENT.issubset(document.get("current_assessment", {})):
            errors.append(f"{path.name}: 当前状态字段缺失")
        if any(key.startswith("accel_") for row in history for key in row):
            errors.append(f"{path.name}: 不应包含原始振动数组")
        if document.get("model", {}).get("version") != "morph-resnet-ordinal-v2":
            errors.append(f"{path.name}: 模型版本错误")
    report = {
        "index_enterprise_count": index.get("enterprise_count"),
        "user_json_count": len(files),
        "date_range": index.get("date_range"),
        "model_version": index.get("model_version"),
        "validation_error_count": len(errors),
        "validation_error_examples": errors[:20],
        "passed": (
            index.get("enterprise_count") == 715
            and len(files) == 715
            and index.get("date_range") == ["2024-12-25", "2025-01-12"]
            and index.get("model_version") == "morph-resnet-ordinal-v2"
            and not errors
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
