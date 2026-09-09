from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import duckdb

from intelligent_detection_agent import smart_metering
from intelligent_detection_agent.data_analyse import DataQualityAnalyzer
from intelligent_detection_agent.smart_metering import SmartMeteringService


def _site_rows(site_name: str, pipeline_no: int, flow: float) -> pd.DataFrame:
    times = pd.date_range("2025-01-12", periods=288, freq="5min")
    return pd.DataFrame({
        "observed_at": times,
        "entity_name": site_name,
        "source_file": f"1063636-{site_name}.xls",
        "pipeline_no": pipeline_no,
        "pressure": 0.3,
        "temperature": 20.0,
        "operational_instant": flow,
        "operational_cumulative": np.arange(len(times), dtype=float),
        "standard_instant": flow,
        "standard_cumulative": np.arange(len(times), dtype=float),
    })


def test_multi_site_diagnosis_keeps_enterprise_total_without_cross_site_pipeline_alert():
    # 两个厂区各自只有一条供气管路；如果按用户直接拼接，会被误认为两条并行管路并触发流量失衡。
    day_long = pd.concat([
        _site_rows("健鼎团结厂", 1, 300.0),
        _site_rows("健鼎芙蓉厂", 2, 100.0),
    ], ignore_index=True)

    class Repository:
        def get_user(self, _user_id):
            return {"station_name": "健鼎（团结厂）电子有限公司", "quantity_max": 1000.0}

        def get_day_long(self, _user_id, _diagnosis_date):
            return day_long

        def get_flow_history(self, _user_id, _diagnosis_date, _days):
            return pd.DataFrame()

    service = SmartMeteringService.__new__(SmartMeteringService)
    service.repo = Repository()
    service.quality_analyzer = DataQualityAnalyzer()
    service.use_deep_model = False
    service.model = SimpleNamespace(predictor=None, load_error=None)

    result = service.diagnose("1063636", date(2025, 1, 12), save=False)

    assert result["alerts"] == []
    assert result["risk_score"] == 0
    assert result["risk_level"] == "低"
    assert result["observed_volume"] == 9600.0
    assert result["meter_spec_result"] == "多厂区合并，表具量程不适用"
    assert [site["site_name"] for site in result["details"]["site_results"]] == ["健鼎团结厂", "健鼎芙蓉厂"]
    assert [site["observed_volume"] for site in result["details"]["site_results"]] == [7200.0, 2400.0]


def test_detail_diagnosis_persists_result_without_legacy_work_order(tmp_path, monkeypatch):
    day_long = pd.concat([
        _site_rows("健鼎团结厂", 1, 300.0),
        _site_rows("健鼎芙蓉厂", 2, 100.0),
    ], ignore_index=True)

    class Repository:
        def get_user(self, _user_id):
            return {"station_name": "健鼎电子", "quantity_max": 1000.0}

        def get_day_long(self, _user_id, _diagnosis_date):
            return day_long

        def get_flow_history(self, _user_id, _diagnosis_date, _days):
            return pd.DataFrame()

    result_db = tmp_path / "gas_ai_results.duckdb"
    monkeypatch.setattr(smart_metering, "RESULT_DB", result_db)
    service = SmartMeteringService.__new__(SmartMeteringService)
    service.repo = Repository()
    service.quality_analyzer = DataQualityAnalyzer()
    service.use_deep_model = False
    service.model = SimpleNamespace(predictor=None, load_error=None)
    service._prepare_result_db()

    result = service.diagnose(
        "1063636", date(2025, 1, 12), save=True, create_work_order=False
    )

    assert result["work_order"] is None
    with duckdb.connect(str(result_db), read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM metering.diagnosis_run").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM metering.work_order").fetchone()[0] == 0
