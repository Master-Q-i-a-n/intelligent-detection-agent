from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data_analyse import DataQualityAnalyzer  # noqa: E402


INPUT_DB = ROOT / "database" / "gas_ai_input.duckdb"
RESULT_DB = ROOT / "database" / "gas_ai_results.duckdb"
MODEL_PATH = Path(
    os.getenv("METERING_MODEL_PATH", str(ROOT / "models" / "metering" / "best_model_xiugai_new.pth"))
)


@dataclass
class IntervalResult:
    interval_id: str
    run_id: str
    user_id: str
    start_time: datetime
    end_time: datetime
    anomaly_type: str
    pipeline_no: Optional[int]
    observed_value: float
    expected_value: float
    estimated_missing_volume: float
    severity: str
    evidence_json: str


class GasDataRepository:
    def __init__(self, db_path: Path = INPUT_DB):
        self.db_path = db_path

    def connect(self):
        return duckdb.connect(str(self.db_path), read_only=True)

    def available_dates(self, user_id: str) -> List[date]:
        with self.connect() as con:
            return [
                row[0]
                for row in con.execute(
                    "SELECT DISTINCT data_date FROM telemetry.scada_observation WHERE user_id=? ORDER BY 1",
                    [str(user_id)],
                ).fetchall()
            ]

    def get_user(self, user_id: str) -> Dict[str, Any]:
        with self.connect() as con:
            df = con.execute("SELECT * FROM asset.user_meter WHERE user_id=?", [str(user_id)]).df()
        if df.empty:
            return {"user_id": str(user_id), "station_name": None, "quantity_min": None, "quantity_max": None}
        row = df.iloc[0].where(pd.notna(df.iloc[0]), None).to_dict()
        return row

    def get_day_long(self, user_id: str, diagnosis_date: date) -> pd.DataFrame:
        with self.connect() as con:
            return con.execute(
                """
                SELECT observed_at, entity_name, source_file, pipeline_no, pressure, temperature,
                       operational_instant, operational_cumulative,
                       standard_instant, standard_cumulative
                FROM telemetry.scada_observation
                WHERE user_id=? AND data_date=?
                ORDER BY observed_at, pipeline_no
                """,
                [str(user_id), diagnosis_date],
            ).df()

    def get_flow_history(self, user_id: str, end_date: date, lookback_days: int = 30) -> pd.DataFrame:
        start_date = end_date - timedelta(days=lookback_days)
        with self.connect() as con:
            return con.execute(
                """
                SELECT observed_at, data_date, entity_name, source_file,
                       pipeline_no, standard_instant, standard_cumulative
                FROM telemetry.scada_observation
                WHERE user_id=? AND data_date>=? AND data_date<?
                ORDER BY observed_at, pipeline_no
                """,
                [str(user_id), start_date, end_date],
            ).df()

    def get_check_points(self, user_id: str) -> pd.DataFrame:
        with self.connect() as con:
            return con.execute(
                """
                WITH ranked AS (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY user_id ORDER BY check_time DESC NULLS LAST, source_row DESC
                    ) AS rn
                    FROM inspection.meter_check_record
                    WHERE user_id=?
                )
                SELECT r.check_record_id, r.check_time, r.base_meter_no,
                       p.point_no, p.check_flow, p.indication_error, p.repeatability
                FROM ranked r
                JOIN inspection.meter_check_point p USING(check_record_id)
                WHERE r.rn=1
                ORDER BY p.point_no
                """,
                [str(user_id)],
            ).df()

    def get_history_for_spec(self, user_id: str, end_date: date, lookback_days: int = 30) -> pd.DataFrame:
        start_date = end_date - timedelta(days=lookback_days - 1)
        with self.connect() as con:
            return con.execute(
                """
                SELECT data_date, pipeline_no, standard_instant
                FROM telemetry.scada_observation
                WHERE user_id=? AND data_date BETWEEN ? AND ?
                ORDER BY data_date, observed_at, pipeline_no
                """,
                [str(user_id), start_date, end_date],
            ).df()


def long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    pieces = []
    metrics = {
        "pressure": "压力",
        "temperature": "温度",
        "operational_instant": "工况瞬时",
        "operational_cumulative": "工况累计",
        "standard_instant": "标况瞬时",
        "standard_cumulative": "标况累计",
    }
    for pipeline_no, group in df.groupby("pipeline_no"):
        pipe = int(pipeline_no)
        group = group.set_index("observed_at").sort_index()
        group = group[~group.index.duplicated(keep="last")]
        renamed = group[list(metrics)].rename(columns={k: f"{pipe}号{v}" for k, v in metrics.items()})
        pieces.append(renamed)
    wide = pd.concat(pieces, axis=1).sort_index()
    wide.index = pd.DatetimeIndex(wide.index)
    return wide


class OptionalGasStateModel:
    def __init__(self):
        self.predictor = None
        self.load_error = None

    def load(self) -> bool:
        if self.predictor is not None:
            return True
        try:
            from model_inference import GasUsagePredictor

            predictor = GasUsagePredictor(str(MODEL_PATH), device="cpu", use_gas_prob_threshold=0.8)
            if not predictor.load_model():
                self.load_error = "模型权重加载失败"
                return False
            self.predictor = predictor
            return True
        except Exception as exc:
            self.load_error = str(exc)
            return False

    def predict(self, quality: Dict[str, Any]) -> Optional[int]:
        if not self.load():
            return None
        return self.predictor.predict(quality)


class SmartMeteringService:
    def __init__(self, use_deep_model: bool = True):
        self.repo = GasDataRepository()
        self.quality_analyzer = DataQualityAnalyzer()
        self.use_deep_model = use_deep_model
        self.model = OptionalGasStateModel()
        self._prepare_result_db()

    def _prepare_result_db(self):
        RESULT_DB.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(RESULT_DB)) as con:
            con.execute((ROOT / "result_schema.sql").read_text(encoding="utf-8"))

    @staticmethod
    def _observed_state(quality: Dict[str, Any]) -> int:
        volume = sum(float(quality.get(f"管道{i}用气量", 0) or 0) for i in range(1, 5))
        return 1 if volume > 1e-6 else 0

    def _legacy_anomalies(self, quality: Dict[str, Any], model_label: int) -> List[str]:
        try:
            from anomaly_detector import AnomalyDetector

            alerts = AnomalyDetector().check_user(
                str(quality.get("用户", "Unknown")), model_label, quality, None
            )
            # “不做诊断”表示当前字段不足，不应作为真实异常或触发工单。
            return [item for item in alerts if "不做诊断" not in item]
        except Exception as exc:
            return [f"联合诊断不可用：{exc}"]

    @staticmethod
    def _joint_diagnostic_evidence(quality: Dict[str, Any], alerts: List[str]) -> Dict[str, Any]:
        """提取闫赛联合诊断的数值证据，供详情页解释告警原因。"""
        evidence: Dict[str, Any] = {"flow": [], "pressure": [], "temperature": []}
        df = quality.get("处理后数据") if isinstance(quality, dict) else None
        if not isinstance(df, pd.DataFrame) or df.empty:
            return evidence
        try:
            from anomaly_detector import AnomalyDetector
            detector = AnomalyDetector()
        except Exception:
            return evidence

        if any("过滤器可能阻塞" in alert for alert in alerts):
            used = []
            for i in range(1, 5):
                col = f"{i}号标况瞬时"
                if col not in df.columns:
                    continue
                series = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                if (series > 0).any():
                    used.append((i, series.to_numpy(dtype=float)))
            if len(used) == 2:
                (p1, s1), (p2, s2) = used
                mask = (s1 > 0) & (s2 > 0)
                high, low = np.maximum(s1[mask], s2[mask]), np.minimum(s1[mask], s2[mask])
                ratio_threshold = float(detector.flow_imbalance_params.get("ratio_threshold", 0.2))
                relative_gap = (high - low) / np.maximum(high, 1e-6)
                violation_ratio = float((relative_gap > ratio_threshold).mean()) if mask.any() else 0.0
                evidence["flow"].append({
                    "pipelines": [p1, p2],
                    "simultaneous_use_points": int(mask.sum()),
                    "imbalance_ratio": round(violation_ratio, 4),
                    "ratio_threshold": ratio_threshold,
                    "pipeline_daily_volume": [
                        round(float(quality.get(f"管道{p1}用气量", 0) or 0), 4),
                        round(float(quality.get(f"管道{p2}用气量", 0) or 0), 4),
                    ],
                    "pipeline_completeness": [
                        round(float(quality.get(f"管道{p1}完整度", 0) or 0), 4),
                        round(float(quality.get(f"管道{p2}完整度", 0) or 0), 4),
                    ],
                })

        if any("压力传感器可能故障" in alert for alert in alerts):
            pressure_series = []
            for i in range(1, 5):
                col = f"{i}号压力"
                if col in df.columns:
                    series = pd.to_numeric(df[col], errors="coerce")
                    fixed_series = detector._fix_pressure_unit(series)
                    if fixed_series.notna().sum() >= 40 and float(fixed_series.std() or 0) >= 0.1:
                        pressure_series.append((i, fixed_series))
            for left in range(len(pressure_series)):
                for right in range(left + 1, len(pressure_series)):
                    p1, s1 = pressure_series[left]
                    p2, s2 = pressure_series[right]
                    dtw_distance = detector._calculate_dtw(s1.interpolate().bfill().ffill(), s2.interpolate().bfill().ffill())
                    hf_correlation = detector._calculate_hf_correlation(s1, s2, detector.pressure_diag_params)
                    if np.isfinite(dtw_distance) and np.isfinite(hf_correlation):
                        evidence["pressure"].append({
                            "pipelines": [p1, p2],
                            "dtw_distance": round(float(dtw_distance), 4),
                            "dtw_threshold": 0.38,
                            "hf_correlation": round(float(hf_correlation), 4),
                            "hf_correlation_threshold": 0.6,
                            "dissimilar": bool(dtw_distance > 0.38 and hf_correlation < 0.6),
                        })
        return evidence

    @staticmethod
    def _resample_total_flow(day_long: pd.DataFrame) -> pd.Series:
        if day_long.empty:
            return pd.Series(dtype=float)
        df = day_long.copy()
        df["observed_at"] = pd.to_datetime(df["observed_at"], errors="coerce")
        df["standard_instant"] = pd.to_numeric(df["standard_instant"], errors="coerce").clip(lower=0)
        df = df.dropna(subset=["observed_at"])
        if df.empty:
            return pd.Series(dtype=float)
        # 同一用户可能包含多个厂区；每个厂区先独立重采样，再汇总为企业流量，
        # 避免不同采样频率的原始行被直接相加或同名管路互相覆盖。
        if "entity_name" in df.columns:
            site_key = df["entity_name"].fillna("").astype(str)
        elif "source_file" in df.columns:
            site_key = df["source_file"].fillna("").astype(str)
        else:
            site_key = pd.Series("__default__", index=df.index)
        df["_site_key"] = site_key.where(site_key.str.len() > 0, "__default__")
        site_flows = []
        for _, site in df.groupby("_site_key", sort=False):
            total = site.groupby("observed_at")["standard_instant"].sum(min_count=1).sort_index()
            total.index = pd.DatetimeIndex(total.index)
            site_flows.append(total.resample("5min").mean())
        total = pd.concat(site_flows, axis=1).sum(axis=1, min_count=1).sort_index()
        start = pd.DatetimeIndex(total.index).min().normalize()
        full_index = pd.date_range(start, start + timedelta(days=1) - timedelta(minutes=5), freq="5min")
        return total.reindex(full_index)

    @staticmethod
    def _history_baseline(history: pd.DataFrame, target_index: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series, int]:
        if history.empty:
            return pd.Series(np.nan, index=target_index), pd.Series(np.nan, index=target_index), 0
        hist = history.copy()
        hist["standard_instant"] = pd.to_numeric(hist["standard_instant"], errors="coerce").clip(lower=0)
        hist["observed_at"] = pd.to_datetime(hist["observed_at"], errors="coerce")
        hist = hist.dropna(subset=["observed_at"])
        if "entity_name" in hist.columns:
            site_key = hist["entity_name"].fillna("").astype(str)
        elif "source_file" in hist.columns:
            site_key = hist["source_file"].fillna("").astype(str)
        else:
            site_key = pd.Series("__default__", index=hist.index)
        hist["_site_key"] = site_key.where(site_key.str.len() > 0, "__default__")
        site_flows = []
        for _, site in hist.groupby("_site_key", sort=False):
            flow = site.groupby("observed_at")["standard_instant"].sum(min_count=1).sort_index()
            flow.index = pd.DatetimeIndex(flow.index)
            resampled = flow.resample("5min").mean()
            # 短缺口只允许在同一天内部插值，避免跨日连接夜间边界。
            resampled = resampled.groupby(resampled.index.normalize(), group_keys=False).apply(
                lambda values: values.interpolate(limit=2)
            )
            site_flows.append(resampled)
        total = pd.concat(site_flows, axis=1).sum(axis=1, min_count=1).sort_index()
        table = total.to_frame("flow")
        table["slot"] = table.index.hour * 60 + table.index.minute
        table["day"] = table.index.date
        day_count = table["day"].nunique()
        grouped = table.groupby("slot")["flow"]
        median = grouped.median()
        mad = grouped.apply(lambda x: float(np.nanmedian(np.abs(x - np.nanmedian(x)))) if x.notna().any() else np.nan)
        slots = pd.Series(target_index.hour * 60 + target_index.minute, index=target_index)
        pred = slots.map(median).astype(float)
        spread = slots.map(mad).astype(float)
        return pred, spread, int(day_count)

    @staticmethod
    def _contiguous_intervals(mask: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Index]]:
        if mask.empty:
            return []
        mask = mask.fillna(False).astype(bool)
        group_id = mask.ne(mask.shift(fill_value=False)).cumsum()
        intervals = []
        for _, group in mask.groupby(group_id):
            if not bool(group.iloc[0]):
                continue
            idx = group.index
            intervals.append((idx[0], idx[-1] + timedelta(minutes=5), idx))
        return intervals

    def _detect_intervals(
        self,
        run_id: str,
        user_id: str,
        observed: pd.Series,
        predicted: pd.Series,
        mad: pd.Series,
        model_label: int,
    ) -> List[IntervalResult]:
        aligned = pd.concat([observed.rename("observed"), predicted.rename("expected"), mad.rename("mad")], axis=1)
        min_abs = max(1.0, float(np.nanmedian(predicted)) * 0.15) if predicted.notna().any() else 1.0
        threshold = np.maximum(3.0 * aligned["mad"].fillna(0), min_abs)
        missing_mask = (aligned["expected"] - aligned["observed"].fillna(0)) > threshold
        if model_label == 1:
            missing_mask = missing_mask | ((aligned["observed"].fillna(0) <= 1e-6) & (aligned["expected"] > min_abs))
        results = []
        for start, end, idx in self._contiguous_intervals(missing_mask):
            if len(idx) < 2:
                continue
            obs_mean = float(aligned.loc[idx, "observed"].fillna(0).mean())
            exp_mean = float(aligned.loc[idx, "expected"].fillna(0).mean())
            missing_volume = float((aligned.loc[idx, "expected"] - aligned.loc[idx, "observed"].fillna(0)).clip(lower=0).sum() * 5 / 60)
            anomaly_type = "疑似走气未走字" if obs_mean <= 1e-6 and model_label == 1 else "用气量显著低于正常基线"
            severity = "高" if missing_volume >= 100 else "中"
            results.append(
                IntervalResult(
                    interval_id=f"INT-{uuid.uuid4().hex[:16]}",
                    run_id=run_id,
                    user_id=str(user_id),
                    start_time=start.to_pydatetime(),
                    end_time=end.to_pydatetime(),
                    anomaly_type=anomaly_type,
                    pipeline_no=None,
                    observed_value=round(obs_mean, 4),
                    expected_value=round(exp_mean, 4),
                    estimated_missing_volume=round(missing_volume, 4),
                    severity=severity,
                    evidence_json=json.dumps(
                        {"points": len(idx), "threshold": round(float(np.nanmean(threshold.loc[idx])), 4)},
                        ensure_ascii=False,
                    ),
                )
            )
        return results

    def _meter_error_bias(self, user_id: str, observed: pd.Series) -> Tuple[float, Dict[str, Any]]:
        points = self.repo.get_check_points(user_id)
        points = points.dropna(subset=["check_flow", "indication_error"])
        if len(points) < 2 or observed.empty:
            return 0.0, {"available": False, "reason": "有效检定点不足"}
        degree = 2 if len(points) >= 3 else 1
        coeff = np.polyfit(points["check_flow"].astype(float), points["indication_error"].astype(float), degree)
        model = np.poly1d(coeff)
        flow = observed.fillna(0).clip(lower=0).to_numpy(dtype=float)
        error_pct = np.clip(model(flow), -10.0, 10.0)
        indicated_volume = flow * 5 / 60
        actual_volume = indicated_volume / np.maximum(1.0 + error_pct / 100.0, 0.1)
        bias = float(np.sum(actual_volume - indicated_volume))
        return bias, {
            "available": True,
            "point_count": int(len(points)),
            "degree": degree,
            "coefficients": [float(x) for x in coeff],
            "base_meter_no": None if points.empty else str(points.iloc[0].get("base_meter_no")),
        }

    def _meter_spec(self, user: Dict[str, Any], history: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
        qmax = user.get("quantity_max")
        if qmax is None or not math.isfinite(float(qmax)):
            return "无表具量程信息", {}
        qmax = float(qmax)
        active = pd.to_numeric(history.get("standard_instant"), errors="coerce").dropna()
        active = active[active > 0]
        if active.empty:
            return "无有效流量数据", {}
        small = float((active < 0.2 * qmax).mean() * 100)
        normal = float(((active >= 0.2 * qmax) & (active <= qmax)).mean() * 100)
        over = float((active > qmax).mean() * 100)
        if small - normal > 10:
            result = "表具选型偏大"
        elif over - normal > 10:
            result = "表具选型偏小"
        else:
            result = "表具选型正常"
        return result, {
            "qmax": qmax,
            "small_flow_percentage": round(small, 2),
            "normal_flow_percentage": round(normal, 2),
            "over_flow_percentage": round(over, 2),
            "history_days": int(history["data_date"].nunique()) if "data_date" in history else 0,
        }

    @staticmethod
    def primary_risk_score(alerts: List[str]) -> Tuple[List[str], float]:
        """统一计算总览和详情共用的联合诊断主告警分数。"""
        primary_alerts = [a for a in alerts if "不做诊断" not in a and "不可用" not in a]
        score = 0.0
        if any("走气未走字" in a for a in primary_alerts):
            score += 75
        if any("计数时长" in a for a in primary_alerts):
            score += 60
        if any("过滤器可能阻塞" in a for a in primary_alerts):
            score += 55
        if any("负压损" in a or "高压损" in a for a in primary_alerts):
            score += 55
        if any("压力传感器可能故障" in a for a in primary_alerts):
            score += 40
        if any("温度异常" in a for a in primary_alerts):
            score += 30
        return primary_alerts, score

    @staticmethod
    def _risk(alerts: List[str], intervals: List[IntervalResult], spec_result: str, makeup: float, quality_status: int):
        # 精准稽查策略：闫赛联合诊断结果是主证据；预测基线、量程和补量仅作辅助，
        # 避免企业停产/减产被误判并形成大量低价值工单。
        primary_alerts, score = SmartMeteringService.primary_risk_score(alerts)
        if primary_alerts:
            if intervals:
                score += min(10, len(intervals) * 2)
            if makeup >= 100:
                score += 10
            if spec_result in ("表具选型偏大", "表具选型偏小"):
                score += 5
        elif quality_status == 2:
            # 数据不可诊断只进入数据治理，不直接形成计量异常工单。
            score = 15
        score = min(100.0, score)
        level = "严重" if score >= 80 else "高" if score >= 60 else "中" if score >= 35 else "低"
        return score, level

    @staticmethod
    def _work_order(run_id: str, user_id: str, d: date, level: str, alerts: List[str], spec: str, makeup: float):
        if level == "低" and not alerts:
            return None
        priority = {"严重": "P1", "高": "P2", "中": "P3", "低": "P4"}[level]
        checklist = ["核对远传累计量与机械字轮示值", "检查流量、压力和温度传感器供电及通信", "核验表具型号、量程与实际负荷"]
        if any("走气未走字" in a for a in alerts):
            checklist.insert(0, "现场确认阀门状态并开展计量旁路排查")
        return {
            "work_order_id": f"WO-{d:%Y%m%d}-{user_id}-{uuid.uuid4().hex[:6]}",
            "run_id": run_id,
            "user_id": str(user_id),
            "diagnosis_date": d,
            "order_type": "计量异常核查",
            "priority": priority,
            "status": "待派发",
            "title": f"{user_id} 智能计量核查",
            "description": f"风险等级：{level}；表具适配：{spec}；估算补气量：{makeup:.2f} m³；告警：{'；'.join(alerts) if alerts else '基线异常'}",
            "checklist_json": json.dumps(checklist, ensure_ascii=False),
        }

    def diagnose(
        self,
        user_id: str,
        diagnosis_date: date,
        save: bool = True,
        create_work_order: bool = True,
    ) -> Dict[str, Any]:
        user_id = str(user_id)
        user = self.repo.get_user(user_id)
        day_long = self.repo.get_day_long(user_id, diagnosis_date)
        if day_long.empty:
            raise ValueError(f"用户 {user_id} 在 {diagnosis_date} 无SCADA数据")
        user_name = user.get("station_name") or user_id
        grouped = day_long.copy()
        if "entity_name" in grouped.columns:
            site_key = grouped["entity_name"].fillna("").astype(str)
        elif "source_file" in grouped.columns:
            site_key = grouped["source_file"].fillna("").astype(str)
        else:
            site_key = pd.Series(str(user_name), index=grouped.index)
        grouped["_site_key"] = site_key.where(site_key.str.len() > 0, str(user_name))

        site_results: List[Dict[str, Any]] = []
        site_qualities: List[Dict[str, Any]] = []
        site_flow_series: List[pd.Series] = []
        for raw_site_name, site_data in grouped.groupby("_site_key", sort=False):
            site_name = str(raw_site_name)
            # 修复历史导入时形成的中文乱码，厂区名称仅用于证据展示和告警定位。
            for _ in range(2):
                try:
                    site_name = site_name.encode("gbk").decode("utf-8")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    break
            wide = long_to_wide(site_data)
            quality = self.quality_analyzer.process_user_data(wide, site_name)
            if quality is None:
                continue
            observed_state = self._observed_state(quality)
            model_state = self.model.predict(quality) if self.use_deep_model else None
            effective_state = observed_state if model_state is None else int(model_state)
            site_alerts = self._legacy_anomalies(quality, effective_state)
            # 数据库侧以标况瞬时流量积分判断计量状态，避免累计量跳变或卡死掩盖“走气未走字”。
            if model_state == 1 and observed_state == 0:
                cross_alert = "流量异常：模型识别为用气，但远传瞬时流量未计量，疑似走气未走字"
                if cross_alert not in site_alerts:
                    site_alerts.insert(0, cross_alert)
            site_volume = sum(float(quality.get(f"管道{i}用气量", 0) or 0) for i in range(1, 5))
            processed = quality.get("处理后数据")
            flow_columns = [
                f"{i}号标况瞬时" for i in range(1, 5)
                if isinstance(processed, pd.DataFrame) and f"{i}号标况瞬时" in processed.columns
            ]
            if flow_columns:
                site_flow_series.append(processed[flow_columns].sum(axis=1, min_count=1))
            site_evidence = self._joint_diagnostic_evidence(quality, site_alerts)
            for evidence_items in site_evidence.values():
                for item in evidence_items:
                    item["site_name"] = site_name
            site_score, site_level = self._risk(
                site_alerts, [], "无表具量程信息", 0.0, int(quality.get("是否有效", 2))
            )
            site_results.append({
                "site_name": site_name,
                "observed_volume": round(site_volume, 4),
                "quality_status": int(quality.get("是否有效", 2)),
                "pipeline_completeness": {str(i): quality.get(f"管道{i}完整度") for i in range(1, 5)},
                "pipeline_daily_volume": {str(i): quality.get(f"管道{i}用气量") for i in range(1, 5)},
                "model_state": model_state,
                "observed_state": observed_state,
                "effective_state": effective_state,
                "alerts": site_alerts,
                "risk_score": round(site_score, 2),
                "risk_level": site_level,
                "diagnostic_evidence": site_evidence,
            })
            site_qualities.append(quality)
        if not site_results:
            raise RuntimeError("数据质量分析失败")

        multi_site = len(site_results) > 1
        alerts = []
        for site in site_results:
            for alert in site["alerts"]:
                labeled = f"{site['site_name']}：{alert}" if multi_site else alert
                if labeled not in alerts:
                    alerts.append(labeled)
        observed_state = 1 if any(site["observed_state"] == 1 for site in site_results) else 0
        model_values = [site["model_state"] for site in site_results if site["model_state"] is not None]
        model_state = (1 if any(value == 1 for value in model_values) else 0) if model_values else None
        effective_state = observed_state if model_state is None else int(model_state)

        # 企业总流量直接汇总各厂区完成质量处理后的五分钟序列，确保总量与厂区明细严格一致。
        observed_flow = pd.concat(site_flow_series, axis=1).sum(axis=1, min_count=1).sort_index()
        history = self.repo.get_flow_history(user_id, diagnosis_date, 30)
        predicted_flow, mad, baseline_days = self._history_baseline(history, observed_flow.index)
        observed_volume = float(observed_flow.fillna(0).sum() * 5 / 60)
        predicted_volume = float(predicted_flow.fillna(0).sum() * 5 / 60) if baseline_days else 0.0

        run_id = f"RUN-{diagnosis_date:%Y%m%d}-{user_id}-{uuid.uuid4().hex[:8]}"
        intervals = self._detect_intervals(run_id, user_id, observed_flow, predicted_flow, mad, effective_state) if baseline_days >= 3 else []
        baseline_missing = sum(x.estimated_missing_volume for x in intervals)
        if multi_site:
            meter_bias, error_model = 0.0, {"available": False, "reason": "多厂区合并数据无法关联到单一检定表具"}
        else:
            meter_bias, error_model = self._meter_error_bias(user_id, observed_flow)
        makeup = max(0.0, baseline_missing) + max(0.0, meter_bias)

        if multi_site:
            spec_result, spec_metrics = "多厂区合并，表具量程不适用", {}
        else:
            spec_history = self.repo.get_history_for_spec(user_id, diagnosis_date, 30)
            spec_result, spec_metrics = self._meter_spec(user, spec_history)
        quality_status = 0 if any(site["quality_status"] == 0 for site in site_results) else 2
        if multi_site:
            # 企业风险取各厂区最高值，不把不同厂区的风险分数累加。
            risk_score = max(float(site["risk_score"]) for site in site_results)
            if risk_score > 0 and intervals:
                risk_score += min(10, len(intervals) * 2)
            if risk_score > 0 and makeup >= 100:
                risk_score += 10
            risk_score = min(100.0, risk_score)
            risk_level = "严重" if risk_score >= 80 else "高" if risk_score >= 60 else "中" if risk_score >= 35 else "低"
        else:
            risk_score, risk_level = self._risk(alerts, intervals, spec_result, makeup, quality_status)
        work_order = self._work_order(run_id, user_id, diagnosis_date, risk_level, alerts, spec_result, makeup)

        diagnostic_evidence: Dict[str, List[Dict[str, Any]]] = {"flow": [], "pressure": [], "temperature": []}
        for site in site_results:
            for evidence_type, evidence_items in site["diagnostic_evidence"].items():
                diagnostic_evidence.setdefault(evidence_type, []).extend(evidence_items)
        primary_quality = site_qualities[0]

        details = {
            "user": user,
            "data_quality": {
                "status": quality_status,
                # 单厂区保留旧字段；多厂区必须查看 sites，避免同号管路再次被误解为同一设备。
                "pipeline_completeness": (
                    {str(i): primary_quality.get(f"管道{i}完整度") for i in range(1, 5)} if not multi_site else {}
                ),
                "pipeline_daily_volume": (
                    {str(i): primary_quality.get(f"管道{i}用气量") for i in range(1, 5)} if not multi_site else {}
                ),
                "sites": site_results,
            },
            "gas_state": {
                "model_state": model_state,
                "observed_state": observed_state,
                "effective_state": effective_state,
                "model_available": self.model.predictor is not None,
                "model_error": self.model.load_error,
            },
            "alerts": alerts,
            "diagnostic_evidence": diagnostic_evidence,
            "site_results": site_results,
            "baseline": {"history_days": baseline_days, "predicted_normal_volume": predicted_volume},
            "meter_error_model": error_model,
            "meter_spec": spec_metrics,
        }
        summary = self._summary(alerts, intervals, spec_result, risk_level)
        result = {
            "run_id": run_id,
            "user_id": user_id,
            "user_name": user_name,
            "diagnosis_date": str(diagnosis_date),
            "status": "completed",
            "quality_status": quality_status,
            "model_gas_state": model_state,
            "observed_gas_state": observed_state,
            "observed_volume": round(observed_volume, 4),
            "predicted_normal_volume": round(predicted_volume, 4),
            "baseline_missing_volume": round(baseline_missing, 4),
            "meter_bias_volume": round(meter_bias, 4),
            "makeup_volume": round(makeup, 4),
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "meter_spec_result": spec_result,
            "summary": summary,
            "alerts": alerts,
            "anomaly_intervals": [asdict(x) for x in intervals],
            "work_order": work_order,
            "details": details,
        }
        if save:
            # 诊断证据落库与工单创建分离；网页查看详情只保存诊断，不绕过人工审批创建工单。
            self._save(result, intervals, work_order if create_work_order else None)
        return result

    @staticmethod
    def _summary(alerts, intervals, spec, level):
        parts = [f"风险等级{level}", f"表具适配：{spec}"]
        if alerts:
            parts.append("；".join(alerts))
        if intervals and alerts:
            parts.append(f"定位到{len(intervals)}个异常区间")
        if not alerts:
            parts.append("流量、压力、温度联合诊断未发现可推送异常")
            if intervals:
                parts.append(f"历史基线存在{len(intervals)}个偏离区间，仅作辅助观察，不触发企业复核")
        return "；".join(parts)

    def _save(self, result: Dict[str, Any], intervals: List[IntervalResult], work_order):
        with duckdb.connect(str(RESULT_DB)) as con:
            old_run_ids = [
                row[0]
                for row in con.execute(
                    "SELECT run_id FROM metering.diagnosis_run WHERE user_id=? AND diagnosis_date=?",
                    [result["user_id"], result["diagnosis_date"]],
                ).fetchall()
            ]
            for old_run_id in old_run_ids:
                con.execute("DELETE FROM metering.anomaly_interval WHERE run_id=?", [old_run_id])
                if work_order:
                    con.execute("DELETE FROM metering.work_order WHERE run_id=?", [old_run_id])
                else:
                    # 仅刷新诊断时保留历史人工处置记录，并关联到新的诊断运行。
                    con.execute("UPDATE metering.work_order SET run_id=? WHERE run_id=?", [result["run_id"], old_run_id])
            con.execute("DELETE FROM metering.diagnosis_run WHERE user_id=? AND diagnosis_date=?", [result["user_id"], result["diagnosis_date"]])
            con.execute(
                """
                INSERT INTO metering.diagnosis_run
                (run_id,user_id,diagnosis_date,user_name,status,quality_status,model_gas_state,observed_gas_state,
                 observed_volume,predicted_normal_volume,baseline_missing_volume,meter_bias_volume,makeup_volume,
                 risk_score,risk_level,meter_spec_result,summary,details_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    result["run_id"], result["user_id"], result["diagnosis_date"], result["user_name"], result["status"],
                    result["quality_status"], result["model_gas_state"], result["observed_gas_state"], result["observed_volume"],
                    result["predicted_normal_volume"], result["baseline_missing_volume"], result["meter_bias_volume"],
                    result["makeup_volume"], result["risk_score"], result["risk_level"], result["meter_spec_result"],
                    result["summary"], json.dumps(result["details"], ensure_ascii=False, default=str),
                ],
            )
            if intervals:
                ids = [x.interval_id for x in intervals]
                con.executemany("DELETE FROM metering.anomaly_interval WHERE interval_id=?", [[x] for x in ids])
                con.executemany(
                    """
                    INSERT INTO metering.anomaly_interval VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    [
                        [x.interval_id,x.run_id,x.user_id,x.start_time,x.end_time,x.anomaly_type,x.pipeline_no,x.observed_value,
                         x.expected_value,x.estimated_missing_volume,x.severity,x.evidence_json]
                        for x in intervals
                    ],
                )
            if work_order:
                con.execute(
                    """
                    INSERT INTO metering.work_order
                    (work_order_id,run_id,user_id,diagnosis_date,order_type,priority,status,title,description,checklist_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    [
                        work_order["work_order_id"], work_order["run_id"], work_order["user_id"],
                        work_order["diagnosis_date"], work_order["order_type"], work_order["priority"],
                        work_order["status"], work_order["title"], work_order["description"],
                        work_order["checklist_json"],
                    ],
                )

    def get_saved_result(self, user_id: str, diagnosis_date: date) -> Optional[Dict[str, Any]]:
        with duckdb.connect(str(RESULT_DB), read_only=True) as con:
            df = con.execute(
                "SELECT * FROM metering.diagnosis_run WHERE user_id=? AND diagnosis_date=?",
                [str(user_id), diagnosis_date],
            ).df()
        if df.empty:
            return None
        row = {
            key: self._native_value(value)
            for key, value in df.iloc[0].where(pd.notna(df.iloc[0]), None).to_dict().items()
        }
        if row.get("details_json"):
            row["details"] = json.loads(row.pop("details_json"))
        return row

    def list_work_orders(self, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM metering.work_order"
        params: List[Any] = []
        if status:
            sql += " WHERE status=?"
            params.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        with duckdb.connect(str(RESULT_DB), read_only=True) as con:
            df = con.execute(sql, params).df()
        return [
            {key: self._native_value(value) for key, value in row.items()}
            for row in df.where(pd.notna(df), None).to_dict(orient="records")
        ]

    @staticmethod
    def _native_value(value):
        if value is None:
            return None
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (pd.Timestamp, datetime, date)):
            return value.isoformat()
        return value
