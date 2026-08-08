from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parent

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402


INPUT_DB = ROOT / "database" / "gas_ai_input.duckdb"
RESULT_DB = ROOT / "database" / "gas_ai_results.duckdb"
RESULT_SCHEMA = ROOT / "result_schema.sql"
MODEL_DIR = ROOT / "models" / "equipment_health"
CACHE_DIR = ROOT / "dataset" / "vibration_ml"
CACHE_PATH = CACHE_DIR / "vibration_multitask_dataset.npz"
MODEL_PATH = MODEL_DIR / "morph_resnet_ordinal_v2.pt"
MANIFEST_PATH = MODEL_DIR / "model_manifest.json"
TRAIN_REPORT_PATH = ROOT / "reports" / "equipment_training_report.json"

STAGES = ["H0", "H1", "H2", "H3", "H4"]
STAGE_TO_INDEX = {name: index for index, name in enumerate(STAGES)}
STAGE_NAMES = {
    "H0": "健康稳定",
    "H1": "轻微衰减",
    "H2": "中度衰减",
    "H3": "重度衰减",
    "H4": "故障异常",
}
CONDITIONS = [80, 120, 160]
CONDITION_TO_INDEX = {value: index for index, value in enumerate(CONDITIONS)}
AGENT_INPUT_ROOT = ROOT / "agent_inputs" / "equipment_health"
MODEL_VERSION = "morph-resnet-ordinal-v2"


def set_seed(seed: int = 20260806) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def stable_bucket(user_id: str) -> int:
    digest = hashlib.sha256(f"equipment-split|{user_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 100


def split_name(user_id: str) -> str:
    value = stable_bucket(user_id)
    return "train" if value < 70 else "validation" if value < 85 else "test"


def build_dataset_cache(overwrite: bool = False) -> Path:
    if CACHE_PATH.exists() and not overwrite:
        return CACHE_PATH
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(INPUT_DB), read_only=True)
    try:
        cursor = con.execute(
            """
            SELECT user_id,company_name,meter_id,data_date,operating_condition,
                   accel_x,accel_y,accel_z,stage_label,health_index,trajectory_type
            FROM vibration.acceleration_window
            ORDER BY user_id,data_date
            """
        )
        rows = []
        while True:
            batch = cursor.fetchmany(512)
            if not batch:
                break
            rows.extend(batch)
    finally:
        con.close()
    count = len(rows)
    if count != 13_585:
        raise RuntimeError(f"振动窗口数量应为13585，实际为{count}")
    signals = np.empty((count, 3, 500), dtype=np.float32)
    labels = np.empty(count, dtype=np.int64)
    health = np.empty(count, dtype=np.float32)
    conditions = np.empty(count, dtype=np.int64)
    user_ids = np.empty(count, dtype="U32")
    company_names = np.empty(count, dtype="U128")
    meter_ids = np.empty(count, dtype="U64")
    dates = np.empty(count, dtype="U10")
    trajectories = np.empty(count, dtype="U32")
    splits = np.empty(count, dtype="U10")
    for index, row in enumerate(rows):
        user_id, company_name, meter_id, data_date, condition, x, y, z, stage, hi, trajectory = row
        signals[index, 0] = np.asarray(x, dtype=np.float32)
        signals[index, 1] = np.asarray(y, dtype=np.float32)
        signals[index, 2] = np.asarray(z, dtype=np.float32)
        labels[index] = STAGE_TO_INDEX[stage]
        health[index] = float(hi)
        conditions[index] = CONDITION_TO_INDEX[int(condition)]
        user_ids[index] = str(user_id)
        company_names[index] = str(company_name)
        meter_ids[index] = str(meter_id)
        dates[index] = str(data_date)
        trajectories[index] = str(trajectory)
        splits[index] = split_name(str(user_id))
    np.savez_compressed(
        CACHE_PATH,
        signals=signals,
        labels=labels,
        health=health,
        conditions=conditions,
        user_ids=user_ids,
        company_names=company_names,
        meter_ids=meter_ids,
        dates=dates,
        trajectories=trajectories,
        splits=splits,
    )
    return CACHE_PATH


class VibrationDataset(Dataset):
    def __init__(self, data: dict, indices: np.ndarray, condition_mean: np.ndarray, condition_std: np.ndarray):
        self.signals = data["signals"]
        self.labels = data["labels"]
        self.health = data["health"]
        self.conditions = data["conditions"]
        self.indices = np.asarray(indices, dtype=np.int64)
        self.condition_mean = condition_mean.astype(np.float32)
        self.condition_std = condition_std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]
        condition = int(self.conditions[index])
        signal = self.signals[index]
        signal = (signal - self.condition_mean[condition, :, None]) / self.condition_std[condition, :, None]
        return (
            torch.from_numpy(signal.astype(np.float32, copy=False)),
            torch.tensor(condition, dtype=torch.long),
            torch.tensor(int(self.labels[index]), dtype=torch.long),
            torch.tensor(float(self.health[index]) / 100.0, dtype=torch.float32),
            torch.tensor(index, dtype=torch.long),
        )


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return F.gelu(x + residual)


class MorphologicalPyramid(nn.Module):
    def __init__(self, axis_count: int = 3, kernels: tuple[int, ...] = (3, 5, 9, 17)):
        super().__init__()
        self.axis_count = axis_count
        self.kernels = kernels
        scale_count = len(kernels)
        hidden = max(4, scale_count * 2)
        self.scale_attention = nn.Sequential(
            nn.Linear(scale_count * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, scale_count),
        )

    @staticmethod
    def pool(x: torch.Tensor, kernel: int) -> torch.Tensor:
        return F.max_pool1d(x, kernel_size=kernel, stride=1, padding=kernel // 2)

    def forward(self, x: torch.Tensor):
        residuals = []
        for kernel in self.kernels:
            dilation = self.pool(x, kernel)
            erosion = -self.pool(-x, kernel)
            opening = self.pool(erosion, kernel)
            closing = -self.pool(-dilation, kernel)
            residuals.append(2.0 * x - opening - closing)
        stacked = torch.stack(residuals, dim=1)  # B,S,3,L
        mean_energy = stacked.abs().mean(dim=(2, 3))
        max_energy = stacked.abs().amax(dim=(2, 3))
        attention = torch.softmax(self.scale_attention(torch.cat([mean_energy, max_energy], dim=1)), dim=1)
        weighted = stacked * attention[:, :, None, None]
        return weighted.flatten(1, 2), attention


class MorphResNetMultiTask(nn.Module):
    def __init__(self, stage_count: int = 5):
        super().__init__()
        self.morphology = MorphologicalPyramid()
        self.stem = nn.Sequential(
            nn.Conv1d(12, 24, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(24),
            nn.GELU(),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.project = nn.Sequential(
            nn.Conv1d(24, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.residual = nn.Sequential(
            ResidualBlock1D(32, dilation=1),
            ResidualBlock1D(32, dilation=2),
        )
        self.axis_gate = nn.Sequential(
            nn.Linear(6, 12), nn.GELU(), nn.Linear(12, 3), nn.Sigmoid()
        )
        self.condition_embedding = nn.Embedding(3, 8)
        self.shared = nn.Sequential(nn.Linear(32 * 2 + 3 + 8, 64), nn.GELU(), nn.Dropout(0.2))
        # CORAL有序分类：一个共享退化得分与四个递增阈值，天然保证
        # P(y>H0) >= P(y>H1) >= P(y>H2) >= P(y>H3)。
        self.ordinal_score = nn.Linear(64, 1)
        self.ordinal_start = nn.Parameter(torch.tensor(-1.5))
        self.ordinal_deltas = nn.Parameter(torch.tensor([0.7, 0.7, 0.7]))
        self.stage_head = nn.Linear(64, stage_count)
        self.health_head = nn.Sequential(nn.Linear(64, 24), nn.GELU(), nn.Linear(24, 1), nn.Sigmoid())

    def forward(self, signal: torch.Tensor, condition: torch.Tensor):
        axis_mean = signal.abs().mean(dim=2)
        axis_max = signal.abs().amax(dim=2)
        axis_weights = self.axis_gate(torch.cat([axis_mean, axis_max], dim=1))
        gated_signal = signal * axis_weights[:, :, None]
        # 500点原始窗口先做4倍平均池化，保留包络变化并显著降低CPU形态学运算量。
        reduced_signal = F.avg_pool1d(gated_signal, kernel_size=4, stride=4)
        morph, scale_attention = self.morphology(reduced_signal)
        features = self.residual(self.project(self.stem(morph)))
        pooled = torch.cat([features.mean(dim=2), features.amax(dim=2)], dim=1)
        shared = self.shared(torch.cat([pooled, axis_weights, self.condition_embedding(condition)], dim=1))
        score = self.ordinal_score(shared)
        positive_deltas = F.softplus(self.ordinal_deltas)
        thresholds = torch.cat(
            [self.ordinal_start.reshape(1), self.ordinal_start + torch.cumsum(positive_deltas, dim=0)]
        )
        ordinal_logits = score - thresholds[None, :]
        stage_logits = self.stage_head(shared)
        health = self.health_head(shared).squeeze(1)
        return stage_logits, health, {
            "scale_attention": scale_attention,
            "axis_weights": axis_weights,
            "ordinal_thresholds": thresholds,
            "ordinal_logits": ordinal_logits,
        }


def ordinal_targets(labels: torch.Tensor) -> torch.Tensor:
    boundaries = torch.arange(4, device=labels.device)[None, :]
    return (labels[:, None] > boundaries).float()


def ordinal_probabilities(logits: torch.Tensor) -> torch.Tensor:
    exceedance = torch.sigmoid(logits)
    probabilities = torch.stack(
        [
            1.0 - exceedance[:, 0],
            exceedance[:, 0] - exceedance[:, 1],
            exceedance[:, 1] - exceedance[:, 2],
            exceedance[:, 2] - exceedance[:, 3],
            exceedance[:, 3],
        ],
        dim=1,
    )
    probabilities = probabilities.clamp_min(1e-7)
    return probabilities / probabilities.sum(dim=1, keepdim=True)


def multitask_loss(
    stage_logits: torch.Tensor,
    ordinal_logits: torch.Tensor,
    predicted_health: torch.Tensor,
    labels: torch.Tensor,
    health: torch.Tensor,
    stage_criterion,
    ordinal_criterion,
    health_criterion,
):
    probabilities = torch.softmax(stage_logits, dim=1)
    stage_loss = stage_criterion(stage_logits, labels)
    ordinal_loss = ordinal_criterion(ordinal_logits, ordinal_targets(labels))
    health_loss = health_criterion(predicted_health, health)
    # 各阶段的代表健康度，用于约束分类分布与连续健康指数方向一致。
    health_centers = torch.tensor([0.925, 0.775, 0.600, 0.375, 0.125], device=health.device)
    stage_implied_health = (probabilities * health_centers[None, :]).sum(dim=1)
    consistency_loss = F.smooth_l1_loss(stage_implied_health, predicted_health, beta=0.05)
    total = stage_loss + 0.35 * ordinal_loss + 2.0 * health_loss + 0.35 * consistency_loss
    return total, probabilities


def compute_normalizer(data: dict, train_indices: np.ndarray):
    means = np.zeros((3, 3), dtype=np.float32)
    stds = np.ones((3, 3), dtype=np.float32)
    for condition in range(3):
        indices = train_indices[data["conditions"][train_indices] == condition]
        values = data["signals"][indices]
        means[condition] = values.mean(axis=(0, 2))
        stds[condition] = values.std(axis=(0, 2))
    return means, np.maximum(stds, 1e-6)


@dataclass
class EpochMetrics:
    loss: float
    stage_accuracy: float
    macro_f1: float
    health_mae: float


def evaluate(model, loader, device, criterion_stage, criterion_ordinal, criterion_health):
    model.eval()
    total_loss = 0.0
    labels_all, predictions_all, health_true, health_pred = [], [], [], []
    with torch.no_grad():
        for signal, condition, label, health, _ in loader:
            signal, condition, label, health = signal.to(device), condition.to(device), label.to(device), health.to(device)
            stage_logits, predicted_health, explain = model(signal, condition)
            loss, probabilities = multitask_loss(
                stage_logits,
                explain["ordinal_logits"],
                predicted_health,
                label,
                health,
                criterion_stage,
                criterion_ordinal,
                criterion_health,
            )
            total_loss += float(loss.item()) * len(label)
            labels_all.extend(label.cpu().numpy().tolist())
            predictions_all.extend(probabilities.argmax(1).cpu().numpy().tolist())
            health_true.extend((health.cpu().numpy() * 100.0).tolist())
            health_pred.extend((predicted_health.cpu().numpy() * 100.0).tolist())
    return EpochMetrics(
        loss=total_loss / max(1, len(loader.dataset)),
        stage_accuracy=float(accuracy_score(labels_all, predictions_all)),
        macro_f1=float(f1_score(labels_all, predictions_all, average="macro")),
        health_mae=float(mean_absolute_error(health_true, health_pred)),
    ), labels_all, predictions_all


def load_npz(path: Path) -> dict:
    loaded = np.load(path, allow_pickle=False)
    return {key: loaded[key] for key in loaded.files}


def train_model(epochs: int = 20, batch_size: int = 128, learning_rate: float = 1e-3, rebuild_cache: bool = False):
    set_seed()
    cache = build_dataset_cache(overwrite=rebuild_cache)
    data = load_npz(cache)
    train_indices = np.where(data["splits"] == "train")[0]
    validation_indices = np.where(data["splits"] == "validation")[0]
    test_indices = np.where(data["splits"] == "test")[0]
    condition_mean, condition_std = compute_normalizer(data, train_indices)
    train_set = VibrationDataset(data, train_indices, condition_mean, condition_std)
    validation_set = VibrationDataset(data, validation_indices, condition_mean, condition_std)
    test_set = VibrationDataset(data, test_indices, condition_mean, condition_std)
    generator = torch.Generator().manual_seed(20260806)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MorphResNetMultiTask().to(device)
    train_labels = data["labels"][train_indices]
    class_counts = np.bincount(train_labels, minlength=5).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts * len(class_counts), 1.0)
    criterion_stage = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    ordinal_numpy = (train_labels[:, None] > np.arange(4)[None, :]).astype(np.float32)
    positive = ordinal_numpy.sum(axis=0)
    negative = len(ordinal_numpy) - positive
    pos_weight = torch.tensor(negative / np.maximum(positive, 1.0), dtype=torch.float32, device=device)
    criterion_ordinal = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_health = nn.SmoothL1Loss(beta=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    best_state = None
    best_score = -math.inf
    patience = 5
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for signal, condition, label, health, _ in train_loader:
            signal, condition, label, health = signal.to(device), condition.to(device), label.to(device), health.to(device)
            stage_logits, predicted_health, explain = model(signal, condition)
            loss, _ = multitask_loss(
                stage_logits,
                explain["ordinal_logits"],
                predicted_health,
                label,
                health,
                criterion_stage,
                criterion_ordinal,
                criterion_health,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += float(loss.item()) * len(label)
        validation_metrics, _, _ = evaluate(
            model, validation_loader, device, criterion_stage, criterion_ordinal, criterion_health
        )
        scheduler.step(validation_metrics.loss)
        score = validation_metrics.macro_f1 - validation_metrics.health_mae / 100.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / len(train_set),
                "validation": asdict(validation_metrics),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(json.dumps(history[-1], ensure_ascii=False))
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("模型训练未产生有效权重")
    model.load_state_dict(best_state)
    test_metrics, test_labels, test_predictions = evaluate(
        model, test_loader, device, criterion_stage, criterion_ordinal, criterion_health
    )
    confusion = confusion_matrix(test_labels, test_predictions, labels=list(range(5))).tolist()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_version": MODEL_VERSION,
        "state_dict": model.state_dict(),
        "condition_mean": condition_mean,
        "condition_std": condition_std,
        "stages": STAGES,
        "conditions": CONDITIONS,
        "window_length": 500,
        "axis_order": ["x", "y", "z"],
    }
    torch.save(checkpoint, MODEL_PATH)
    report = {
        "model_version": MODEL_VERSION,
        "device": str(device),
        "train_samples": len(train_set),
        "validation_samples": len(validation_set),
        "test_samples": len(test_set),
        "train_users": int(len(np.unique(data["user_ids"][train_indices]))),
        "validation_users": int(len(np.unique(data["user_ids"][validation_indices]))),
        "test_users": int(len(np.unique(data["user_ids"][test_indices]))),
        "best_epoch": int(np.argmax([h["validation"]["macro_f1"] - h["validation"]["health_mae"] / 100 for h in history]) + 1),
        "test_metrics": asdict(test_metrics),
        "confusion_matrix": confusion,
        "history": history,
        "data_notice": "训练和测试来自工程标签原型生成的模拟退化轨迹，不代表现场真实寿命试验。",
    }
    TRAIN_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {key: value for key, value in report.items() if key != "history"}
    manifest.update({"model_path": str(MODEL_PATH), "created_at": datetime.now().isoformat(timespec="seconds")})
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model_path": str(MODEL_PATH), "test_metrics": asdict(test_metrics)}, ensure_ascii=False, indent=2))
    return report


def load_model(model_path: Path = MODEL_PATH):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = MorphResNetMultiTask()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def infer_array(model, checkpoint, signal: np.ndarray, condition_value: int):
    condition_index = CONDITION_TO_INDEX[int(condition_value)]
    mean = np.asarray(checkpoint["condition_mean"], dtype=np.float32)[condition_index, :, None]
    std = np.asarray(checkpoint["condition_std"], dtype=np.float32)[condition_index, :, None]
    normalized = (signal.astype(np.float32) - mean) / std
    with torch.no_grad():
        stage_logits, health, explain = model(
            torch.from_numpy(normalized[None]), torch.tensor([condition_index], dtype=torch.long)
        )
        probabilities = torch.softmax(stage_logits, dim=1)[0].numpy()
        exceedance = torch.sigmoid(explain["ordinal_logits"])[0].numpy()
        stage_index = int(probabilities.argmax())
        stage = STAGES[stage_index]
        return {
            "predicted_stage": stage,
            "predicted_stage_name": STAGE_NAMES[stage],
            "predicted_health_index": round(float(health.item() * 100.0), 2),
            "confidence": round(float(probabilities[stage_index]), 6),
            "probabilities": {STAGES[i]: round(float(probabilities[i]), 6) for i in range(5)},
            "ordinal_exceedance": {
                "beyond_H0": round(float(exceedance[0]), 6),
                "beyond_H1": round(float(exceedance[1]), 6),
                "beyond_H2": round(float(exceedance[2]), 6),
                "beyond_H3": round(float(exceedance[3]), 6),
            },
            "axis_weights": [round(float(v), 6) for v in explain["axis_weights"][0].numpy()],
            "scale_weights": [round(float(v), 6) for v in explain["scale_attention"][0].numpy()],
        }


def risk_from_stage(stage: str) -> tuple[str, str]:
    if stage == "H0":
        return "健康", "低风险"
    if stage in ("H1", "H2"):
        return "性能衰减", "中风险"
    return "故障/临故障", "高风险"


def diagnose_user_day(user_id: str, diagnosis_date: str, persist: bool = True):
    model, checkpoint = load_model()
    con = duckdb.connect(str(INPUT_DB), read_only=True)
    try:
        row = con.execute(
            """
            SELECT company_name,meter_id,operating_condition,accel_x,accel_y,accel_z
            FROM vibration.acceleration_window
            WHERE user_id=? AND data_date=CAST(? AS DATE)
            """,
            [str(user_id), diagnosis_date],
        ).fetchone()
    finally:
        con.close()
    if row is None:
        raise ValueError(f"未找到企业{user_id}在{diagnosis_date}的振动数据")
    company_name, meter_id, condition, x, y, z = row
    signal = np.stack([x, y, z], axis=0).astype(np.float32)
    result = infer_array(model, checkpoint, signal, int(condition))
    coarse_status, risk_level = risk_from_stage(result["predicted_stage"])
    result.update(
        {
            "user_id": str(user_id),
            "company_name": str(company_name),
            "meter_id": str(meter_id),
            "diagnosis_date": diagnosis_date,
            "coarse_status": coarse_status,
            "risk_level": risk_level,
            "model_version": checkpoint["model_version"],
        }
    )
    if persist:
        save_day_result(result)
    return result


def compress_stages(stages: list[str]) -> list[str]:
    result = []
    for stage in stages:
        if not result or result[-1] != stage:
            result.append(stage)
    return result


def classify_trend(health_values: list[float], stages: list[str]) -> dict:
    x = np.arange(len(health_values), dtype=np.float64)
    values = np.asarray(health_values, dtype=np.float64)
    slope = float(np.polyfit(x, values, 1)[0]) if len(values) > 1 else 0.0
    drops = values[:-1] - values[1:] if len(values) > 1 else np.asarray([0.0])
    maximum_drop = float(max(0.0, drops.max()))
    maximum_drop_index = int(np.argmax(drops)) if len(values) > 1 else 0
    pre_drop_values = values[: maximum_drop_index + 1]
    pre_drop_slope = (
        float(np.polyfit(np.arange(len(pre_drop_values)), pre_drop_values, 1)[0])
        if len(pre_drop_values) > 2
        else 0.0
    )
    if values[-1] - values[0] >= 20.0 and slope > 1.0:
        trend = "recovery"
    elif (
        maximum_drop >= 25.0
        and stages[-1] == "H4"
        and maximum_drop_index >= int((len(values) - 1) * 0.55)
        and abs(pre_drop_slope) <= 0.8
    ):
        trend = "abrupt_fault"
    elif slope <= -2.0 or (stages[-1] in ("H3", "H4") and slope <= -1.0):
        trend = "fast_decay"
    elif slope <= -0.35:
        trend = "slow_decay"
    else:
        trend = "stable"
    return {
        "trend_label": trend,
        "daily_slope": round(slope, 4),
        "maximum_daily_drop": round(maximum_drop, 4),
        "maximum_drop_day_index": maximum_drop_index,
        "pre_drop_daily_slope": round(pre_drop_slope, 4),
        "health_index_start": round(float(values[0]), 2),
        "health_index_end": round(float(values[-1]), 2),
        "stage_sequence": compress_stages(stages),
    }


def health_to_stage(health_index: float) -> str:
    if health_index >= 85.0:
        return "H0"
    if health_index >= 70.0:
        return "H1"
    if health_index >= 50.0:
        return "H2"
    if health_index >= 25.0:
        return "H3"
    return "H4"


def stabilize_daily_predictions(daily: list[dict]) -> list[dict]:
    """对连续健康指数做保留突变的稳健平滑，再统一阶段与健康指数。"""
    if not daily:
        return daily
    raw = np.asarray([item["predicted_health_index"] for item in daily], dtype=np.float64)
    median = raw.copy()
    for index in range(1, len(raw) - 1):
        median[index] = float(np.median(raw[index - 1 : index + 2]))
    stabilized = raw.copy()
    for index in range(1, len(raw)):
        change = raw[index] - raw[index - 1]
        # 大幅突降或突升代表故障/维修事件，直接保留，不被平滑抹掉。
        if abs(change) >= 25.0:
            stabilized[index] = raw[index]
        else:
            stabilized[index] = 0.60 * median[index] + 0.40 * stabilized[index - 1]
    output = []
    for item, value in zip(daily, stabilized):
        stage = health_to_stage(float(value))
        enriched = dict(item)
        enriched["raw_stage"] = item["predicted_stage"]
        enriched["raw_health_index"] = item["predicted_health_index"]
        enriched["stabilized_stage"] = stage
        enriched["stabilized_stage_name"] = STAGE_NAMES[stage]
        enriched["stabilized_health_index"] = round(float(value), 2)
        output.append(enriched)
    return output


def apply_temporal_consistency(daily: list[dict]) -> tuple[list[dict], dict]:
    """依据序列趋势对阶段进行有序约束，同时保留原始和仅平滑结果。"""
    stabilized = stabilize_daily_predictions(daily)
    preliminary = classify_trend(
        [item["stabilized_health_index"] for item in stabilized],
        [item["stabilized_stage"] for item in stabilized],
    )
    trend = preliminary["trend_label"]
    indices = np.asarray([STAGE_TO_INDEX[item["stabilized_stage"]] for item in stabilized], dtype=int)
    if trend in ("slow_decay", "fast_decay"):
        consistent = np.maximum.accumulate(indices)
    elif trend == "recovery":
        consistent = np.minimum.accumulate(indices)
    elif trend == "abrupt_fault":
        breakpoint = int(preliminary["maximum_drop_day_index"]) + 1
        before = int(np.median(indices[:breakpoint])) if breakpoint > 0 else int(indices[0])
        after = max(4, int(np.median(indices[breakpoint:]))) if breakpoint < len(indices) else 4
        consistent = np.asarray([before] * breakpoint + [after] * (len(indices) - breakpoint), dtype=int)
    else:
        stable_stage = int(np.median(indices))
        consistent = np.full_like(indices, stable_stage)
    output = []
    for item, stage_index in zip(stabilized, consistent):
        enriched = dict(item)
        stage = STAGES[int(stage_index)]
        enriched["temporal_stage"] = stage
        enriched["temporal_stage_name"] = STAGE_NAMES[stage]
        output.append(enriched)
    final_trend = classify_trend(
        [item["stabilized_health_index"] for item in output],
        [item["temporal_stage"] for item in output],
    )
    return output, final_trend


def diagnose_user_trend(user_id: str, start_date: str = "2024-12-25", end_date: str = "2025-01-12", persist: bool = True):
    model, checkpoint = load_model()
    con = duckdb.connect(str(INPUT_DB), read_only=True)
    try:
        rows = con.execute(
            """
            SELECT data_date,operating_condition,accel_x,accel_y,accel_z
            FROM vibration.acceleration_window
            WHERE user_id=? AND data_date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
            ORDER BY data_date
            """,
            [str(user_id), start_date, end_date],
        ).fetchall()
    finally:
        con.close()
    if not rows:
        raise ValueError(f"未找到企业{user_id}的振动序列")
    daily = []
    for data_date, condition, x, y, z in rows:
        prediction = infer_array(model, checkpoint, np.stack([x, y, z], axis=0), int(condition))
        daily.append({"date": str(data_date), **prediction})
    daily, trend = apply_temporal_consistency(daily)
    result = {
        "user_id": str(user_id),
        "start_date": str(rows[0][0]),
        "end_date": str(rows[-1][0]),
        "model_version": checkpoint["model_version"],
        **trend,
        "daily_predictions": daily,
    }
    if persist:
        save_trend_result(result)
    return result


def recommended_action(stage: str, trend: str) -> str:
    if stage == "H4" or trend == "abrupt_fault":
        return "立即生成高优先级核查工单，检查桨叶、轴承、传动及传感器安装状态。"
    if stage == "H3" or trend == "fast_decay":
        return "建议24小时内复测振动并安排计划检修，重点核查高能冲击与异常频带。"
    if stage == "H2" or trend == "slow_decay":
        return "提高采集频次，结合流量工况持续观察，并纳入预防性维护清单。"
    if trend == "recovery":
        return "保持复测，确认维修后健康指数稳定且无再次衰减。"
    if stage == "H1":
        return "维持常规运行，缩短巡检周期并观察健康指数趋势。"
    return "设备状态稳定，保持常规监测。"


def export_agent_inputs(output_root: Path = AGENT_INPUT_ROOT) -> dict:
    """为全部企业生成轻量JSON，不包含三轴原始数组。"""
    data = load_npz(build_dataset_cache())
    model, checkpoint = load_model()
    indices = np.arange(len(data["signals"]), dtype=np.int64)
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
            stage_logits, health, explain = model(signal, condition)
            probabilities = torch.softmax(stage_logits, dim=1).numpy()
            exceedance = torch.sigmoid(explain["ordinal_logits"]).numpy()
            axis_weights = explain["axis_weights"].numpy()
            scale_weights = explain["scale_attention"].numpy()
            for position, source_index in enumerate(source_indices.numpy()):
                stage_index = int(probabilities[position].argmax())
                predictions[int(source_index)] = {
                    "predicted_stage": STAGES[stage_index],
                    "predicted_stage_name": STAGE_NAMES[STAGES[stage_index]],
                    "predicted_health_index": round(float(health[position].item() * 100.0), 2),
                    "confidence": round(float(probabilities[position, stage_index]), 6),
                    "probabilities": {
                        STAGES[i]: round(float(probabilities[position, i]), 6) for i in range(5)
                    },
                    "ordinal_exceedance": {
                        "beyond_H0": round(float(exceedance[position, 0]), 6),
                        "beyond_H1": round(float(exceedance[position, 1]), 6),
                        "beyond_H2": round(float(exceedance[position, 2]), 6),
                        "beyond_H3": round(float(exceedance[position, 3]), 6),
                    },
                    "axis_weights": [round(float(v), 6) for v in axis_weights[position]],
                    "scale_weights": [round(float(v), 6) for v in scale_weights[position]],
                }
    by_user = {}
    for index in indices:
        user_id = str(data["user_ids"][index])
        by_user.setdefault(user_id, []).append(int(index))
    users_root = output_root / "users"
    users_root.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for user_id, user_indices in by_user.items():
        user_indices.sort(key=lambda value: str(data["dates"][value]))
        daily = []
        for index in user_indices:
            daily.append(
                {
                    "date": str(data["dates"][index]),
                    "operating_condition": int(CONDITIONS[int(data["conditions"][index])]),
                    **predictions[index],
                }
            )
        daily, trend = apply_temporal_consistency(daily)
        latest = daily[-1]
        coarse_status, risk_level = risk_from_stage(latest["temporal_stage"])
        company_name = str(data["company_names"][user_indices[0]])
        meter_id = str(data["meter_ids"][user_indices[0]])
        document = {
            "schema": "gas-agent.equipment-health.v1",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "entity": {
                "user_id": user_id,
                "company_name": company_name,
                "meter_id": meter_id,
                "sensor_position": "后盖",
                "axis_order": ["x", "y", "z"],
                "sampling_rate_hz": 1000,
                "window_length": 500,
            },
            "current_assessment": {
                "date": latest["date"],
                "stage": latest["temporal_stage"],
                "stage_name": latest["temporal_stage_name"],
                "health_index": latest["stabilized_health_index"],
                "raw_model_stage": latest["raw_stage"],
                "raw_model_health_index": latest["raw_health_index"],
                "confidence": latest["confidence"],
                "probabilities": latest["probabilities"],
                "ordinal_exceedance": latest["ordinal_exceedance"],
                "coarse_status": coarse_status,
                "risk_level": risk_level,
            },
            "trend_assessment": trend,
            "model_explanation": {
                "axis_weights": latest["axis_weights"],
                "morphological_scale_weights": latest["scale_weights"],
                "scale_kernel_sizes": [3, 5, 9, 17],
            },
            "recommended_action": recommended_action(latest["temporal_stage"], trend["trend_label"]),
            "daily_history": daily,
            "model": {
                "version": checkpoint["model_version"],
                "method": "condition-normalized tri-axis morphological attention ResNet with CORAL ordinal classification and health regression",
            },
            "data_notice": "当前退化轨迹由工程师三分类原型模拟生成，仅用于算法和Agent闭环验证。",
        }
        path = users_root / f"{user_id}.json"
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        index_rows.append(
            {
                "user_id": user_id,
                "company_name": company_name,
                "latest_stage": latest["temporal_stage"],
                "latest_health_index": latest["stabilized_health_index"],
                "trend_label": trend["trend_label"],
                "risk_level": risk_level,
                "path": f"users/{user_id}.json",
            }
        )
    index_document = {
        "schema": "gas-agent.equipment-health-index.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_version": checkpoint["model_version"],
        "enterprise_count": len(index_rows),
        "date_range": [str(sorted(data["dates"])[0]), str(sorted(data["dates"])[-1])],
        "users": sorted(index_rows, key=lambda item: item["user_id"]),
    }
    (output_root / "index.json").write_text(
        json.dumps(index_document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "output_root": str(output_root),
        "index_path": str(output_root / "index.json"),
        "enterprise_count": len(index_rows),
        "user_json_count": len(list(users_root.glob("*.json"))),
        "model_version": checkpoint["model_version"],
    }


def result_connection():
    RESULT_DB.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(RESULT_DB))
    con.execute(RESULT_SCHEMA.read_text(encoding="utf-8"))
    return con


def save_day_result(result: dict) -> None:
    diagnosis_id = hashlib.sha1(
        f"{result['user_id']}|{result['diagnosis_date']}|{result['model_version']}".encode()
    ).hexdigest()[:24]
    details = json.dumps(
        {
            key: result[key]
            for key in ("probabilities", "ordinal_exceedance", "axis_weights", "scale_weights")
        },
        ensure_ascii=False,
    )
    con = result_connection()
    try:
        con.execute("DELETE FROM equipment.health_diagnosis WHERE diagnosis_id=?", [diagnosis_id])
        con.execute(
            """
            INSERT INTO equipment.health_diagnosis
            (diagnosis_id,user_id,company_name,meter_id,diagnosis_date,predicted_stage,
             predicted_stage_name,predicted_health_index,confidence,coarse_status,risk_level,
             trend_label,model_version,details_json)
            VALUES (?,?,?,?,CAST(? AS DATE),?,?,?,?,?,?,NULL,?,?)
            """,
            [diagnosis_id, result["user_id"], result["company_name"], result["meter_id"],
             result["diagnosis_date"], result["predicted_stage"], result["predicted_stage_name"],
             result["predicted_health_index"], result["confidence"], result["coarse_status"],
             result["risk_level"], result["model_version"], details],
        )
    finally:
        con.close()


def save_trend_result(result: dict) -> None:
    trend_id = hashlib.sha1(
        f"{result['user_id']}|{result['start_date']}|{result['end_date']}|{result['model_version']}".encode()
    ).hexdigest()[:24]
    details = json.dumps({"daily_predictions": result["daily_predictions"]}, ensure_ascii=False)
    con = result_connection()
    try:
        con.execute("DELETE FROM equipment.health_trend WHERE trend_id=?", [trend_id])
        con.execute(
            """
            INSERT INTO equipment.health_trend
            (trend_id,user_id,start_date,end_date,trend_label,health_index_start,health_index_end,
             daily_slope,maximum_daily_drop,stage_sequence,model_version,details_json)
            VALUES (?,?,CAST(? AS DATE),CAST(? AS DATE),?,?,?,?,?,?,?,?)
            """,
            [trend_id, result["user_id"], result["start_date"], result["end_date"],
             result["trend_label"], result["health_index_start"], result["health_index_end"],
             result["daily_slope"], result["maximum_daily_drop"],
             "→".join(result["stage_sequence"]), result["model_version"], details],
        )
    finally:
        con.close()
