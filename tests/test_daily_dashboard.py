from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from intelligent_detection_agent.daily_dashboard import (
    DAILY_OVERVIEW_ALGORITHM_VERSION,
    DailyDiagnosisDashboard,
)


def test_same_date_concurrent_overview_only_computes_once(tmp_path, monkeypatch):
    service = DailyDiagnosisDashboard(tmp_path)
    target = date(2025, 1, 9)
    start_together = threading.Barrier(2)
    call_count = 0
    count_lock = threading.Lock()

    def fake_metering_rows(_target):
        nonlocal call_count
        with count_lock:
            call_count += 1
        # 留出重叠窗口，确保第二个请求会遇到同日期正在计算。
        time.sleep(0.1)
        return {}

    monkeypatch.setattr(service, "_metering_rows", fake_metering_rows)
    monkeypatch.setattr(service, "_equipment_rows", lambda _target: {})

    def request_overview():
        start_together.wait()
        return service.overview(target)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: request_overview(), range(2)))

    assert call_count == 1
    assert results[0] == results[1]
    assert (service.cache_root / "2025-01-09.json").exists()


def test_overview_recomputes_cache_from_old_algorithm_version(tmp_path, monkeypatch):
    service = DailyDiagnosisDashboard(tmp_path)
    target = date(2025, 1, 9)
    cache_path = service.cache_root / "2025-01-09.json"
    cache_path.write_text('{"algorithm_version":"old","issues":[{"user_id":"stale"}]}', encoding="utf-8")
    calls = 0

    def fake_metering_rows(_target):
        nonlocal calls
        calls += 1
        return {}

    monkeypatch.setattr(service, "_metering_rows", fake_metering_rows)
    monkeypatch.setattr(service, "_equipment_rows", lambda _target: {})

    result = service.overview(target)

    assert calls == 1
    assert result["algorithm_version"] == DAILY_OVERVIEW_ALGORITHM_VERSION
    assert result["issues"] == []


def test_review_budget_selects_top_five_per_module_and_keeps_full_statistics(tmp_path):
    service = DailyDiagnosisDashboard(tmp_path)

    def issue(module, index, score, risk_level, issue_type):
        return {
            "module": module,
            "user_id": f"{module}-{index}",
            "risk_score": score,
            "risk_level": risk_level,
            "issue_type": issue_type,
        }

    candidates = [
        issue("metering", index, 70 - index, "高" if index < 2 else "中", "计量异常")
        for index in range(7)
    ] + [
        issue("equipment", index, 100 - index, "严重", "设备异常")
        for index in range(8)
    ]

    result = service._apply_daily_review_budget({"issues": candidates, "diagnosed_enterprises": 20})
    selected_metering = [item for item in result["issues"] if item["module"] == "metering"]
    selected_equipment = [item for item in result["issues"] if item["module"] == "equipment"]

    assert [item["user_id"] for item in selected_metering] == [f"metering-{index}" for index in range(5)]
    assert [item["user_id"] for item in selected_equipment] == [f"equipment-{index}" for index in range(5)]
    assert result["metering_issue_count"] == 7
    assert result["equipment_issue_count"] == 8
    assert result["abnormal_enterprises"] == 15
    assert result["normal_enterprises"] == 5
    assert result["high_risk_count"] == 10
    assert result["risk_distribution"] == {"严重": 8, "高": 2, "中": 5, "较低": 0, "低": 0}
    assert result["issue_type_distribution"] == [
        {"name": "设备异常", "value": 8},
        {"name": "计量异常", "value": 7},
    ]
    assert result["suppressed_issue_count"] == 5
    assert result["review_budget_per_module"] == 5
