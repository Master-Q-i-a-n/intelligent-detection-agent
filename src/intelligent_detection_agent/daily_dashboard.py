from __future__ import annotations

import json
import threading
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .anomaly_detector import AnomalyDetector
from .smart_metering import DataQualityAnalyzer, SmartMeteringService, long_to_wide

DAILY_OVERVIEW_ALGORITHM_VERSION = "metering-site-aware-v5-issue-display"


class DailyDiagnosisDashboard:
    """全量日筛查：一次查询完成所有企业计量统计，并合并设备模型日结果。"""

    def __init__(self, root: Path):
        self.root = root
        self.input_db = root / "database" / "gas_ai_input.duckdb"
        self.equipment_index = root / "agent_inputs" / "equipment_health" / "index.json"
        self.equipment_users = root / "agent_inputs" / "equipment_health" / "users"
        self.cache_root = root / "reports" / "daily_precision_overview_v4"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict[str, Any]] = {}
        # 同一日期只允许一个线程执行首次全量诊断，其他请求等待并复用其缓存结果。
        self._overview_locks: dict[str, threading.Lock] = {}
        self._overview_locks_guard = threading.Lock()

    @staticmethod
    def _apply_daily_review_budget(result: dict[str, Any], max_enterprises: int = 5) -> dict[str, Any]:
        """计量和设备分别选取高风险企业，并保留完整候选统计。"""
        candidates = list(result.get("issues", []))
        candidates.sort(key=lambda x: (-float(x.get("risk_score", 0)), x.get("user_id", ""), x.get("module", "")))
        selected: list[dict[str, Any]] = []
        selected_enterprise_ids: set[str] = set()
        for module in ("metering", "equipment"):
            module_candidates = [item for item in candidates if item.get("module") == module]
            module_selected_ids: list[str] = []
            for item in module_candidates:
                user_id = str(item.get("user_id"))
                if user_id not in module_selected_ids:
                    module_selected_ids.append(user_id)
                if len(module_selected_ids) >= max_enterprises:
                    break
            selected.extend(item for item in module_candidates if str(item.get("user_id")) in module_selected_ids)
            selected_enterprise_ids.update(module_selected_ids)

        candidate_enterprise_ids = {str(item.get("user_id")) for item in candidates}
        risk_distribution: dict[str, int] = {"严重": 0, "高": 0, "中": 0, "较低": 0, "低": 0}
        issue_type_counts: dict[str, int] = {}
        for item in candidates:
            risk_level = str(item.get("risk_level", "未知"))
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
            issue_type = str(item.get("issue_type", "未分类"))
            issue_type_counts[issue_type] = issue_type_counts.get(issue_type, 0) + 1

        metering_count = sum(item.get("module") == "metering" for item in candidates)
        equipment_count = sum(item.get("module") == "equipment" for item in candidates)
        high_risk_count = sum(item.get("risk_level") in ("严重", "高") for item in candidates)
        result = dict(result)
        result.update({
            "issues": selected,
            "abnormal_enterprises": len(candidate_enterprise_ids),
            "metering_issue_count": metering_count,
            "equipment_issue_count": equipment_count,
            "high_risk_count": high_risk_count,
            "severe_count": high_risk_count,
            "risk_distribution": risk_distribution,
            "issue_type_distribution": [
                {"name": name, "value": value}
                for name, value in sorted(issue_type_counts.items(), key=lambda item: (-item[1], item[0]))
            ],
            "candidate_enterprises": len(candidate_enterprise_ids),
            "observation_enterprises": max(0, len(candidate_enterprise_ids) - len(selected_enterprise_ids)),
            "suppressed_issue_count": max(0, len(candidates) - len(selected)),
            "normal_enterprises": max(0, int(result.get("diagnosed_enterprises", 715)) - len(candidate_enterprise_ids)),
            "review_budget_per_module": max_enterprises,
        })
        return result

    @staticmethod
    def _repair(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        output = value
        for _ in range(2):
            try:
                candidate = output.encode("gbk").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                break
            output = candidate
        return output.rstrip("?")

    def _equipment_rows(self, target: str) -> dict[str, dict[str, Any]]:
        index = json.loads(self.equipment_index.read_text(encoding="utf-8"))
        rows = {}
        for user in index.get("users", []):
            path = self.equipment_users / f"{user['user_id']}.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            day = next((x for x in payload.get("daily_history", []) if x.get("date") == target), None)
            if day is None:
                continue
            stage = day.get("temporal_stage") or day.get("stabilized_stage") or day.get("predicted_stage")
            health = float(day.get("stabilized_health_index", day.get("predicted_health_index", 0)))
            recent = [x for x in payload.get("daily_history", []) if x.get("date", "") <= target][-7:]
            values = [float(x.get("stabilized_health_index", x.get("predicted_health_index", 0))) for x in recent]
            n = len(values)
            mean_x = (n - 1) / 2 if n else 0
            mean_y = sum(values) / n if n else health
            denom = sum((i - mean_x) ** 2 for i in range(n))
            slope = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values)) / denom if denom else 0.0
            maximum_drop = max([values[i-1] - values[i] for i in range(1, n)] or [0.0])
            trend = "fast_decay" if slope <= -1.0 or maximum_drop >= 12 else "slow_decay" if slope <= -0.35 else "stable"
            # 经济性优先：H1不推送；H2只有出现明确恶化趋势才推送；H3/H4直接推送。
            confidence = float(day.get("confidence", 0) or 0)
            actionable = stage in ("H3", "H4") or (
                stage == "H2" and health < 55.0 and confidence >= 0.85
                and slope <= -1.5 and maximum_drop >= 12.0
            )
            if not actionable:
                continue
            issue = {"H1": "轻度性能衰减", "H2": "持续性能衰减", "H3": "严重性能衰减", "H4": "疑似设备故障"}.get(stage, "健康趋势异常")
            score = max(0.0, min(100.0, 100.0 - health))
            rows[str(user["user_id"])] = {
                "user_id": str(user["user_id"]),
                "company_name": self._repair(payload.get("entity", {}).get("company_name") or user.get("company_name")),
                "module": "equipment",
                "issue_type": issue,
                "issue_tags": [issue, {"slow_decay": "缓慢衰减", "fast_decay": "加速衰减", "abrupt_fault": "突发故障", "recovery": "维修恢复"}.get(trend, "状态波动")],
                "risk_score": round(score, 2),
                "risk_level": "严重" if stage == "H4" else "高" if stage == "H3" else "中" if stage == "H2" else "较低",
                "primary_metric": round(health, 2),
                "primary_metric_name": "健康指数",
                "stage": stage,
                "evidence_summary": f"阶段 {stage}，健康指数 {health:.2f}，近7日斜率 {slope:.2f}点/日，置信度 {float(day.get('confidence', 0))*100:.1f}%",
            }
        return rows

    def _metering_rows(self, target: date) -> dict[str, dict[str, Any]]:
        """以闫赛流量、压力、温度联合诊断为主的高精度日筛查。"""
        with duckdb.connect(str(self.input_db), read_only=True) as con:
            day_long = con.execute(
                """
                SELECT s.*,u.station_name
                FROM telemetry.scada_observation s
                LEFT JOIN asset.user_meter u ON u.user_id=s.user_id
                WHERE s.data_date=?
                ORDER BY s.user_id,s.observed_at,s.pipeline_no
                """,
                [target],
            ).df()
        analyzer = DataQualityAnalyzer()
        detector = AnomalyDetector()
        equipment_index = json.loads(self.equipment_index.read_text(encoding="utf-8"))
        equipment_names = {
            str(item.get("user_id")): self._repair(item.get("company_name"))
            for item in equipment_index.get("users", [])
            if item.get("user_id") is not None and item.get("company_name")
        }
        output = {}
        for user_id, group in day_long.groupby("user_id", sort=False):
            station_names = group["station_name"].dropna()
            scada_names = group["entity_name"].dropna() if "entity_name" in group else pd.Series(dtype=str)
            # 部分新编号尚未进入企业表具档案，但设备档案和 SCADA 文件仍有可用名称。
            company = (
                station_names.iloc[0]
                if not station_names.empty
                else equipment_names.get(str(user_id))
                or (self._repair(str(scada_names.iloc[0])) if not scada_names.empty else str(user_id))
            )
            site_key = group["entity_name"].fillna("").astype(str) if "entity_name" in group else group["source_file"].fillna("").astype(str)
            grouped = group.copy()
            grouped["_site_key"] = site_key.where(site_key.str.len() > 0, str(company))
            site_results = []
            for raw_site_name, site_data in grouped.groupby("_site_key", sort=False):
                try:
                    site_name = self._repair(str(raw_site_name))
                    wide = long_to_wide(site_data)
                    quality = analyzer.process_user_data(wide, site_name)
                    if not quality or int(quality.get("是否有效", 2)) != 0:
                        continue
                    observed_volume = sum(
                        float(quality.get(f"管道{i}用气量", 0) or 0) for i in range(1, 5)
                    )
                    observed_state = 1 if observed_volume > 1e-6 else 0
                    raw_alerts = detector.check_user(site_name, observed_state, quality, None)
                except Exception:
                    continue
                alerts = [a for a in raw_alerts if "不做诊断" not in a]
                # 单一温度特征易受环境与工况影响，只在与流量/压力异常共现时推送。
                hard_alerts = [a for a in alerts if "流量异常" in a or "压力异常" in a]
                site_score, site_level = SmartMeteringService._risk(
                    hard_alerts, [], "无表具量程信息", 0.0, int(quality.get("是否有效", 2))
                )
                site_results.append({
                    "site_name": site_name,
                    "observed_volume": round(observed_volume, 2),
                    "alerts": hard_alerts,
                    "risk_score": round(site_score, 2),
                    "risk_level": site_level,
                    "pipeline_daily_volume": {
                        str(i): round(float(quality.get(f"管道{i}用气量", 0) or 0), 2) for i in range(1, 5)
                    },
                })
            if not site_results or not any(site["alerts"] for site in site_results):
                continue
            multi_site = len(site_results) > 1
            alerts = []
            for site in site_results:
                for alert in site["alerts"]:
                    alerts.append(f"{site['site_name']}：{alert}" if multi_site else alert)
            # 企业风险取厂区最高值；企业气量则是各厂区重采样积分之和。
            score = max(float(site["risk_score"]) for site in site_results)
            risk_level = "严重" if score >= 80 else "高" if score >= 60 else "中" if score >= 35 else "低"
            observed_volume = sum(float(site["observed_volume"]) for site in site_results)
            primary_site = next(site for site in site_results if site["alerts"])
            # 企业名称已经在独立列展示，主要问题只保留异常内容；厂区来源放入证据摘要。
            primary = primary_site["alerts"][0]
            evidence_summary = (
                "；".join(f"问题厂区：{site['site_name']}；{alert}" for site in site_results for alert in site["alerts"])
                if multi_site else "；".join(alerts[:3])
            )
            output[str(user_id)] = {
                "user_id": str(user_id), "company_name": self._repair(str(company)), "module": "metering",
                "issue_type": primary, "issue_tags": alerts, "risk_score": score,
                "risk_level": risk_level,
                "primary_metric": round(observed_volume, 2), "primary_metric_name": "当日计量气量",
                "observed_volume": round(observed_volume, 2),
                "evidence_summary": evidence_summary,
                "affected_sites": [site["site_name"] for site in site_results if site["alerts"]],
                "site_results": site_results,
            }
        return output

    def overview(self, target: date) -> dict[str, Any]:
        cache_key = str(target)
        if cache_key in self._cache:
            return self._cache[cache_key]
        with self._overview_locks_guard:
            date_lock = self._overview_locks.setdefault(cache_key, threading.Lock())

        with date_lock:
            # 等待锁期间，前一个请求可能已经生成了内存或磁盘缓存，因此必须再次检查。
            if cache_key in self._cache:
                return self._cache[cache_key]
            cache_path = self.cache_root / f"{cache_key}.json"
            if cache_path.exists():
                result = json.loads(cache_path.read_text(encoding="utf-8"))
                if result.get("algorithm_version") == DAILY_OVERVIEW_ALGORITHM_VERSION:
                    self._cache[cache_key] = result
                    return result
            metering = self._metering_rows(target)
            equipment = self._equipment_rows(str(target))
            issues = list(metering.values()) + list(equipment.values())
            issues.sort(key=lambda x: (-x["risk_score"], x["user_id"], x["module"]))
            unique = len({x["user_id"] for x in issues})
            result = {
                "algorithm_version": DAILY_OVERVIEW_ALGORITHM_VERSION,
                "diagnosis_date": str(target), "status": "completed", "diagnosed_enterprises": 715,
                "abnormal_enterprises": unique, "normal_enterprises": max(0, 715-unique),
                "metering_issue_count": len(metering), "equipment_issue_count": len(equipment),
                "severe_count": sum(x["risk_level"] in ("严重", "高") for x in issues),
                "issues": issues,
            }
            result = self._apply_daily_review_budget(result)
            self._cache[cache_key] = result
            cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return result

    def metering_history(self, user_id: str, target: date, days: int = 7) -> list[dict[str, Any]]:
        start = target - timedelta(days=days-1)
        with duckdb.connect(str(self.input_db), read_only=True) as con:
            history = con.execute(
                """
                SELECT data_date,observed_at,entity_name,source_file,pipeline_no,
                       pressure,temperature,operational_instant,operational_cumulative,
                       standard_instant,standard_cumulative
                FROM telemetry.scada_observation
                WHERE user_id=? AND data_date BETWEEN ? AND ?
                ORDER BY data_date,entity_name,observed_at,pipeline_no
                """, [str(user_id), start, target]
            ).df()
        analyzer = DataQualityAnalyzer()
        rows = []
        for data_date, day in history.groupby("data_date", sort=True):
            site_flows = []
            for site_name, site_data in day.groupby("entity_name", dropna=False, sort=False):
                quality = analyzer.process_user_data(long_to_wide(site_data), self._repair(str(site_name)))
                processed = quality.get("处理后数据") if quality else None
                if not isinstance(processed, pd.DataFrame) or processed.empty:
                    continue
                flow_columns = [f"{i}号标况瞬时" for i in range(1, 5) if f"{i}号标况瞬时" in processed.columns]
                if flow_columns:
                    site_flows.append(processed[flow_columns].sum(axis=1, min_count=1))
            if not site_flows:
                continue
            total = pd.concat(site_flows, axis=1).sum(axis=1, min_count=1)
            rows.append({
                "date": str(pd.Timestamp(data_date).date()),
                "volume": round(float(total.fillna(0).sum() * 5.0 / 60.0), 2),
                "avg_flow": round(float(total.mean() or 0), 2),
                "max_flow": round(float(total.max() or 0), 2),
                "completeness": round(float(total.notna().mean()) * 100, 2),
                "active_ratio": round(float((total.fillna(0) > 0).mean()) * 100, 2),
            })
        return rows
