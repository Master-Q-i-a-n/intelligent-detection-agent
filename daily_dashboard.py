from __future__ import annotations

import json
import threading
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb

from smart_metering import DataQualityAnalyzer, long_to_wide
from anomaly_detector import AnomalyDetector


class DailyDiagnosisDashboard:
    """全量日筛查：一次查询完成所有企业计量统计，并合并设备模型日结果。"""

    def __init__(self, root: Path):
        self.root = root
        self.input_db = root / "database" / "gas_ai_input.duckdb"
        self.equipment_index = root / "agent_inputs" / "equipment_health" / "index.json"
        self.equipment_users = root / "agent_inputs" / "equipment_health" / "users"
        self.cache_root = root / "reports" / "daily_precision_overview_v3"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict[str, Any]] = {}
        # 同一日期只允许一个线程执行首次全量诊断，其他请求等待并复用其缓存结果。
        self._overview_locks: dict[str, threading.Lock] = {}
        self._overview_locks_guard = threading.Lock()

    @staticmethod
    def _apply_daily_review_budget(result: dict[str, Any], max_enterprises: int = 5) -> dict[str, Any]:
        """按风险/经济优先级只推送有限企业，其余保留在后台观察池。"""
        candidates = list(result.get("issues", []))
        candidates.sort(key=lambda x: (-float(x.get("risk_score", 0)), x.get("user_id", ""), x.get("module", "")))
        selected_ids = []
        for item in candidates:
            user_id = str(item.get("user_id"))
            if user_id not in selected_ids:
                selected_ids.append(user_id)
            if len(selected_ids) >= max_enterprises:
                break
        selected = [item for item in candidates if str(item.get("user_id")) in selected_ids]
        candidate_enterprises = len({str(item.get("user_id")) for item in candidates})
        result = dict(result)
        result.update({
            "issues": selected,
            "abnormal_enterprises": len(selected_ids),
            "metering_issue_count": sum(item.get("module") == "metering" for item in selected),
            "equipment_issue_count": sum(item.get("module") == "equipment" for item in selected),
            "severe_count": sum(item.get("risk_level") in ("严重", "高") for item in selected),
            "candidate_enterprises": candidate_enterprises,
            "observation_enterprises": max(0, candidate_enterprises - len(selected_ids)),
            "suppressed_issue_count": max(0, len(candidates) - len(selected)),
            "normal_enterprises": max(0, int(result.get("diagnosed_enterprises", 715)) - candidate_enterprises),
            "review_budget": max_enterprises,
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
        output = {}
        for user_id, group in day_long.groupby("user_id", sort=False):
            try:
                wide = long_to_wide(group)
                company = group["station_name"].dropna().iloc[0] if group["station_name"].notna().any() else str(user_id)
                quality = analyzer.process_user_data(wide, company)
                if not quality or int(quality.get("是否有效", 2)) != 0:
                    continue
                observed_state = 1 if sum(float(quality.get(f"管道{i}用气量", 0) or 0) for i in range(1, 5)) > 1e-6 else 0
                raw_alerts = detector.check_user(str(company), observed_state, quality, None)
            except Exception:
                continue
            alerts = [a for a in raw_alerts if "不做诊断" not in a]
            # 单一温度特征易受环境与工况影响，只在与流量/压力异常共现时推送。
            has_hard = any("流量异常" in a or "压力异常" in a for a in alerts)
            if not has_hard:
                continue
            score = 0.0
            for alert in alerts:
                if "走气未走字" in alert: score += 80
                elif "计数时长" in alert: score += 65
                elif "过滤器可能阻塞" in alert: score += 60
                elif "负压损" in alert or "高压损" in alert: score += 60
                elif "压力传感器可能故障" in alert: score += 45
                elif "温度异常" in alert: score += 15
            score = round(min(100.0, score), 2)
            volumes = [float(quality.get(f"管道{i}用气量", 0) or 0) for i in range(1, 5)]
            primary = alerts[0]
            output[str(user_id)] = {
                "user_id": str(user_id), "company_name": self._repair(str(company)), "module": "metering",
                "issue_type": primary, "issue_tags": alerts, "risk_score": score,
                "risk_level": "严重" if score>=80 else "高" if score>=60 else "中" if score>=35 else "较低",
                "primary_metric": round(sum(volumes), 2), "primary_metric_name": "当日计量气量",
                "observed_volume": round(sum(volumes), 2),
                "evidence_summary": "；".join(alerts[:3]),
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
                self._cache[cache_key] = result
                return result
            previous_cache = self.root / "reports" / "daily_precision_overview_v2" / f"{cache_key}.json"
            if previous_cache.exists():
                result = self._apply_daily_review_budget(json.loads(previous_cache.read_text(encoding="utf-8")))
                cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                self._cache[cache_key] = result
                return result
            metering = self._metering_rows(target)
            equipment = self._equipment_rows(str(target))
            issues = list(metering.values()) + list(equipment.values())
            issues.sort(key=lambda x: (-x["risk_score"], x["user_id"], x["module"]))
            unique = len({x["user_id"] for x in issues})
            result = {
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
            rows = con.execute(
                """
                WITH slot_flow AS (
                  SELECT data_date,observed_at,SUM(GREATEST(COALESCE(standard_instant,0),0)) total_flow
                  FROM telemetry.scada_observation WHERE user_id=? AND data_date BETWEEN ? AND ?
                  GROUP BY data_date,observed_at
                )
                SELECT data_date,SUM(total_flow)*5.0/60.0 volume,AVG(total_flow) avg_flow,
                       MAX(total_flow) max_flow,COUNT(*) slots,
                       AVG(CASE WHEN total_flow>0 THEN 1.0 ELSE 0.0 END) active_ratio
                FROM slot_flow GROUP BY data_date ORDER BY data_date
                """, [str(user_id), start, target]
            ).fetchall()
        return [{"date":str(r[0]),"volume":round(float(r[1] or 0),2),"avg_flow":round(float(r[2] or 0),2),
                 "max_flow":round(float(r[3] or 0),2),"completeness":round(min(1,float(r[4] or 0)/288)*100,2),
                 "active_ratio":round(float(r[5] or 0)*100,2)} for r in rows]
