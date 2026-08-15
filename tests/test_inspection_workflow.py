from __future__ import annotations

import json
from types import SimpleNamespace

from inspection_agent import InspectionAgent


class FakeJsonModel:
    def __init__(self, payload: dict | str):
        self.payload = payload
        self.calls = 0
        self.bound_options: dict = {}

    def bind(self, **kwargs):
        self.bound_options = kwargs
        return self

    def invoke(self, messages):
        self.calls += 1
        content = self.payload if isinstance(self.payload, str) else json.dumps(self.payload, ensure_ascii=False)
        return SimpleNamespace(content=content)


def valid_interpretation() -> dict:
    return {
        "inspection_conclusion": "规则告警与状态交叉验证共同支持优先现场核查。",
        "analysis_confidence": 0.86,
        "possible_causes": [
            {
                "rank": 1,
                "cause": "远传计量链路可能未正确记录实际用气",
                "confidence": 0.82,
                "supporting_evidence_ids": ["M-STATE", "M-ALERT-1", "FAKE-ID"],
                "counter_evidence_ids": [],
            }
        ],
        "counter_evidence": ["仍需排除生产停运造成的基线差异。"],
        "missing_evidence": ["缺少现场机械字轮读数。"],
        "checklist": ["核对机械字轮与远传累计量"],
        "work_order_advice": "建议生成优先核查工单。",
    }


def metering_context(risk_score: float = 75, run_id: str = "RUN-random") -> dict:
    return {
        "run_id": run_id,
        "user_id": "u1",
        "diagnosis_date": "2025-01-12",
        "quality_status": 0,
        "model_gas_state": 1,
        "observed_gas_state": 0,
        "observed_volume": 0,
        "predicted_normal_volume": 120,
        "baseline_missing_volume": 30,
        "meter_bias_volume": 2,
        "makeup_volume": 32,
        "risk_score": risk_score,
        "risk_level": "高",
        "meter_spec_result": "表具选型正常",
        "summary": "疑似走气未走字",
        "alerts": ["流量异常：疑似走气未走字"],
        "anomaly_intervals": [],
        "details": {
            "data_quality": {"pipeline_completeness": {"1": 1.0}},
            "meter_error_model": {"available": True},
            "meter_spec": {"qmax": 200},
        },
    }


def equipment_context() -> dict:
    return {
        "current_assessment": {
            "stage": "H2",
            "stage_name": "性能衰减",
            "health_index": 68.4,
            "confidence": 0.84,
            "risk_level": "中风险",
            "probabilities": {"H0": 0.02, "H1": 0.12, "H2": 0.84, "H3": 0.02, "H4": 0},
            "operating_condition": 600,
        },
        "trend_assessment": {
            "trend_label": "slow_decay",
            "trend_name": "缓慢衰减",
            "daily_slope": -0.6,
            "maximum_daily_drop": 2.3,
            "window_days": 7,
        },
        "model_explanation": {
            "axis_weights": [0.7, 0.2, 0.1],
            "scale_kernel_sizes": [3, 5, 9, 17],
            "morphological_scale_weights": [0.1, 0.2, 0.5, 0.2],
        },
        "recommended_action": "提高采集频次并安排预防性维护。",
    }


def test_metering_workflow_routes_and_reuses_successful_fingerprint(tmp_path):
    current = metering_context()
    model = FakeJsonModel(valid_interpretation())
    agent = InspectionAgent(
        tmp_path,
        metering_loader=lambda _user, _date: current,
        equipment_loader=lambda _user, _date: equipment_context(),
        result_db=tmp_path / "results.duckdb",
        model_client=model,
    )

    first = agent.generate("metering", "u1", "2025-01-12", "阀门已核对")
    # 随机运行编号不属于算法内容，不应破坏相同诊断的指纹复用。
    current = metering_context(run_id="RUN-another-random")
    second = agent.generate("metering", "u1", "2025-01-12", "阀门已核对")

    assert first["workflow_route"] == "high_risk"
    assert first["risk_score"] == 75
    assert first["generator"].startswith("workflow-llm:")
    assert first["possible_causes"][0]["supporting_evidence_ids"] == ["M-STATE", "M-ALERT-1"]
    assert second["cache_hit"] is True
    assert model.calls == 1
    assert model.bound_options == {"response_format": {"type": "json_object"}}


def test_frontend_context_is_ignored_and_algorithm_values_are_immutable(tmp_path):
    model = FakeJsonModel(valid_interpretation())
    agent = InspectionAgent(
        tmp_path,
        metering_loader=lambda _user, _date: metering_context(),
        equipment_loader=lambda _user, _date: equipment_context(),
        result_db=tmp_path / "results.duckdb",
        model_client=model,
    )

    result = agent.generate(
        "metering",
        "u1",
        "2025-01-12",
        context={"risk_score": 0, "risk_level": "低", "makeup_volume": 999999},
    )

    assert result["risk_score"] == 75
    assert result["risk_level"] == "高"
    assert result["evidence_items"][5]["value"] == 32


def test_equipment_uses_independent_degradation_path(tmp_path):
    equipment_payload = valid_interpretation()
    equipment_payload["possible_causes"][0]["supporting_evidence_ids"] = ["E-STATE", "E-TREND"]
    model = FakeJsonModel(equipment_payload)
    agent = InspectionAgent(
        tmp_path,
        metering_loader=lambda _user, _date: metering_context(),
        equipment_loader=lambda _user, _date: equipment_context(),
        result_db=tmp_path / "results.duckdb",
        model_client=model,
    )

    result = agent.generate("equipment", "u2", "2025-01-12")

    assert agent.metering_graph is not agent.equipment_graph
    assert result["workflow_route"] == "degradation"
    assert result["risk_level"] == "中风险"
    assert result["risk_score"] == 31.6
    assert {item["evidence_id"] for item in result["evidence_items"]} >= {"E-STATE", "E-TREND", "E-AXIS"}


def test_invalid_model_json_returns_deterministic_fallback_without_cache(tmp_path):
    model = FakeJsonModel("not-json")
    agent = InspectionAgent(
        tmp_path,
        metering_loader=lambda _user, _date: metering_context(),
        equipment_loader=lambda _user, _date: equipment_context(),
        result_db=tmp_path / "results.duckdb",
        model_client=model,
    )

    result = agent.generate("metering", "u1", "2025-01-12")

    assert result["generator"] == "workflow-local-fallback"
    assert result["cache_hit"] is False
    assert "ValidationError" in result["llm_notice"]
    assert result["possible_causes"]
