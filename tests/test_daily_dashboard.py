from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from daily_dashboard import DailyDiagnosisDashboard


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
