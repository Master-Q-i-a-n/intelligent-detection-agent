from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class AnomalyDetector:
    """本地可审计规则诊断器；用于缺少旧版私有模块时维持基础诊断能力。"""

    def __init__(self) -> None:
        self.flow_imbalance_params = {"ratio_threshold": 0.2, "minimum_points": 12}
        self.pressure_diag_params = {"rolling_window": 12}

    def check_user(
        self,
        user_name: str,
        gas_state: int,
        quality: dict[str, Any],
        _: Any,
    ) -> list[str]:
        frame = quality.get("处理后数据")
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return ["数据质量不足，不做诊断"]

        alerts: list[str] = []
        active_flows: list[pd.Series] = []
        for pipeline_no in range(1, 5):
            column = f"{pipeline_no}号标况瞬时"
            if column in frame.columns:
                series = pd.to_numeric(frame[column], errors="coerce")
                if int((series.fillna(0) > 0).sum()) >= self.flow_imbalance_params["minimum_points"]:
                    active_flows.append(series)

        # 双管同时供气时，持续性的相对流量偏差作为过滤器/阀门核查信号。
        for left in range(len(active_flows)):
            for right in range(left + 1, len(active_flows)):
                pair = pd.concat([active_flows[left], active_flows[right]], axis=1).dropna()
                pair = pair[(pair.iloc[:, 0] > 0) & (pair.iloc[:, 1] > 0)]
                if len(pair) < self.flow_imbalance_params["minimum_points"]:
                    continue
                high = pair.max(axis=1)
                relative_gap = (high - pair.min(axis=1)) / high.clip(lower=1e-6)
                if float((relative_gap > self.flow_imbalance_params["ratio_threshold"]).mean()) >= 0.3:
                    alerts.append("流量异常：流量计用气期间标况瞬时流量差距过大，过滤器可能阻塞")
                    break
            if alerts:
                break

        pressures: list[pd.Series] = []
        for pipeline_no in range(1, 5):
            column = f"{pipeline_no}号压力"
            if column not in frame.columns:
                continue
            series = self._fix_pressure_unit(pd.to_numeric(frame[column], errors="coerce"))
            if series.notna().sum() >= 40 and float(series.std() or 0) >= 0.1:
                pressures.append(series)

        for left in range(len(pressures)):
            for right in range(left + 1, len(pressures)):
                distance = self._calculate_dtw(pressures[left], pressures[right])
                correlation = self._calculate_hf_correlation(
                    pressures[left], pressures[right], self.pressure_diag_params
                )
                if np.isfinite(distance) and np.isfinite(correlation) and distance > 0.38 and correlation < 0.6:
                    alerts.append("压力异常（主备管道压力波动不相似，压力传感器可能故障）")
                    return alerts
        return alerts

    @staticmethod
    def _fix_pressure_unit(series: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce").astype(float)
        median = float(values.abs().median()) if values.notna().any() else 0.0
        # 超过常见 MPa 数值范围时按 kPa 转换，避免不同源文件单位造成误判。
        return values / 1000.0 if median > 20.0 else values

    @staticmethod
    def _calculate_dtw(left: pd.Series, right: pd.Series) -> float:
        pair = pd.concat([left, right], axis=1).interpolate(limit_direction="both").dropna()
        if len(pair) < 2:
            return float("nan")
        normalized = (pair - pair.mean()) / pair.std().replace(0, 1.0)
        return float(np.mean(np.abs(normalized.iloc[:, 0] - normalized.iloc[:, 1])))

    @staticmethod
    def _calculate_hf_correlation(left: pd.Series, right: pd.Series, params: dict[str, Any]) -> float:
        pair = pd.concat([left, right], axis=1).interpolate(limit_direction="both").dropna()
        if len(pair) < 3:
            return float("nan")
        window = int(params.get("rolling_window", 12))
        high_frequency = pair - pair.rolling(window, min_periods=1, center=True).mean()
        correlation = high_frequency.iloc[:, 0].corr(high_frequency.iloc[:, 1])
        return float(correlation) if pd.notna(correlation) else float("nan")
