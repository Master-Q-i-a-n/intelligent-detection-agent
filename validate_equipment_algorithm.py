from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from smart_equipment import (  # noqa: E402
    MODEL_PATH,
    STAGES,
    VibrationDataset,
    apply_temporal_consistency,
    classify_trend,
    load_model,
    load_npz,
    CACHE_PATH,
)


REPORT_PATH = ROOT / "reports" / "equipment_algorithm_validation.json"
EXPECTED_TREND = {
    "stable_healthy": "stable",
    "slow_decay": "slow_decay",
    "persistent_subhealth": "slow_decay",
    "accelerated_decay": "fast_decay",
    "abrupt_fault": "abrupt_fault",
    "maintenance_recovery": "recovery",
}


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("请先训练设备健康模型")
    data = load_npz(CACHE_PATH)
    model, checkpoint = load_model()
    indices = np.where(data["splits"] == "test")[0]
    dataset = VibrationDataset(
        data,
        indices,
        np.asarray(checkpoint["condition_mean"]),
        np.asarray(checkpoint["condition_std"]),
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    predictions = {}
    model.eval()
    with torch.no_grad():
        for signal, condition, _, _, source_indices in loader:
            stage_logits, health, _ = model(signal, condition)
            stage_indices = torch.softmax(stage_logits, dim=1).argmax(1).numpy()
            health_values = health.numpy() * 100.0
            for source_index, stage_index, health_value in zip(
                source_indices.numpy(), stage_indices, health_values
            ):
                predictions[int(source_index)] = (STAGES[int(stage_index)], float(health_value))

    by_user = defaultdict(list)
    for index in indices:
        user_id = str(data["user_ids"][index])
        by_user[user_id].append(index)
    results = []
    raw_labels_all, raw_predictions_all, temporal_predictions_all = [], [], []
    confusion = defaultdict(lambda: defaultdict(int))
    for user_id, user_indices in by_user.items():
        user_indices.sort(key=lambda i: str(data["dates"][i]))
        stages = [predictions[i][0] for i in user_indices]
        health = [predictions[i][1] for i in user_indices]
        daily = [
            {
                "date": str(data["dates"][i]),
                "predicted_stage": predictions[i][0],
                "predicted_health_index": predictions[i][1],
            }
            for i in user_indices
        ]
        consistent_daily, consistent_trend = apply_temporal_consistency(daily)
        ground_truth = [int(data["labels"][i]) for i in user_indices]
        raw_labels_all.extend(ground_truth)
        raw_predictions_all.extend([STAGES.index(stage) for stage in stages])
        temporal_predictions_all.extend([STAGES.index(item["temporal_stage"]) for item in consistent_daily])
        trajectory = str(data["trajectories"][user_indices[0]])
        expected = EXPECTED_TREND[trajectory]
        trend = consistent_trend
        actual = trend["trend_label"]
        confusion[expected][actual] += 1
        results.append(
            {
                "user_id": user_id,
                "trajectory_type": trajectory,
                "expected_trend": expected,
                "predicted_trend": actual,
                **{key: value for key, value in trend.items() if key != "trend_label"},
            }
        )
    correct = sum(item["expected_trend"] == item["predicted_trend"] for item in results)
    representative = {}
    for item in results:
        representative.setdefault(item["trajectory_type"], item)
    report = {
        "test_users": len(results),
        "trend_accuracy": round(correct / len(results), 6),
        "correct_users": correct,
        "raw_stage_metrics": {
            "accuracy": round(float(accuracy_score(raw_labels_all, raw_predictions_all)), 6),
            "macro_f1": round(float(f1_score(raw_labels_all, raw_predictions_all, average="macro")), 6),
            "confusion_matrix": confusion_matrix(
                raw_labels_all, raw_predictions_all, labels=list(range(5))
            ).tolist(),
        },
        "temporal_stage_metrics": {
            "accuracy": round(float(accuracy_score(raw_labels_all, temporal_predictions_all)), 6),
            "macro_f1": round(float(f1_score(raw_labels_all, temporal_predictions_all, average="macro")), 6),
            "confusion_matrix": confusion_matrix(
                raw_labels_all, temporal_predictions_all, labels=list(range(5))
            ).tolist(),
        },
        "confusion": {expected: dict(values) for expected, values in confusion.items()},
        "representative_examples": representative,
        "passed": correct / len(results) >= 0.75,
        "data_notice": "趋势验证基于模拟退化轨迹，仅验证算法闭环与轨迹可分性。",
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
