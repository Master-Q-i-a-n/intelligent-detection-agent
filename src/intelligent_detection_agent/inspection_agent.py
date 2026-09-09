from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict

import duckdb
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .paths import PROJECT_ROOT

WORKFLOW_VERSION = "inspection-workflow-v1"
RESULT_SCHEMA = PROJECT_ROOT / "result_schema.sql"
RESULT_DB = PROJECT_ROOT / "database" / "gas_ai_results.duckdb"


class EvidenceItem(BaseModel):
    evidence_id: str
    category: str
    statement: str
    source: str
    value: Any | None = None
    unit: str | None = None


class CauseAssessment(BaseModel):
    rank: int = Field(ge=1)
    cause: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    counter_evidence_ids: list[str] = Field(default_factory=list)


class LlmInterpretation(BaseModel):
    """只允许 LLM 填写解释字段，算法原值不进入此模型。"""

    inspection_conclusion: str = Field(min_length=1)
    analysis_confidence: float = Field(ge=0, le=1)
    possible_causes: list[CauseAssessment] = Field(default_factory=list)
    counter_evidence: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    checklist: list[str] = Field(default_factory=list)
    work_order_advice: str = Field(min_length=1)


class InspectionState(TypedDict, total=False):
    module: Literal["metering", "equipment"]
    user_id: str
    diagnosis_date: str
    field_text: str
    context: dict[str, Any]
    workflow_route: str
    evidence_items: list[dict[str, Any]]
    fallback_report: dict[str, Any]
    final_report: dict[str, Any]


ContextLoader = Callable[[str, date], dict[str, Any]]


class InspectionAgent:
    """两个独立 StateGraph 工作流的统一入口。

    工作流只读取后端加载的可信诊断结果。前端即使仍传入旧版 context，
    也不会参与分析或指纹计算。
    """

    def __init__(
        self,
        root: Path,
        *,
        metering_loader: ContextLoader,
        equipment_loader: ContextLoader,
        result_db: Path | None = None,
        model_client: Any | None = None,
    ):
        self.root = root
        self._load_env(root / ".env")
        self.loaders = {"metering": metering_loader, "equipment": equipment_loader}
        self.result_db = result_db or RESULT_DB
        self.api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")
        self.model_name = os.getenv("LLM_MODEL", "deepseek-chat")
        self.provider = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
        self.model_client = model_client if model_client is not None else self._build_model()
        # 同一进程内串行执行“查缓存 -> 调模型 -> 写缓存”，防止相同指纹并发重复调用。
        self._generation_lock = threading.RLock()
        self._prepare_result_db()
        self.metering_graph = self._build_metering_graph()
        self.equipment_graph = self._build_equipment_graph()

    @staticmethod
    def _load_env(path: Path) -> None:
        if not path.exists():
            return
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not raw.strip() or raw.lstrip().startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

    def _build_model(self) -> Any | None:
        if not self.api_key:
            return None
        common = {
            "model": self.model_name,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "temperature": 0.1,
            "timeout": 60,
            # 一次工作流只做一次模型请求，失败直接使用确定性回退。
            "max_retries": 0,
            "streaming": False,
        }
        if self.provider == "deepseek":
            return ChatDeepSeek(**common)
        return ChatOpenAI(**common)

    def _prepare_result_db(self) -> None:
        self.result_db.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(self.result_db)) as con:
            con.execute(RESULT_SCHEMA.read_text(encoding="utf-8"))

    def _build_metering_graph(self):
        graph = StateGraph(InspectionState)
        graph.add_node("load_metering", self._load_context)
        graph.add_node("route_metering", self._route_metering)
        for route in ("data_quality", "normal", "single_anomaly", "high_risk"):
            graph.add_node(route, self._evidence_node(route))
            graph.add_edge(route, "interpret_metering")
        graph.add_node("interpret_metering", self._interpret)
        graph.add_edge(START, "load_metering")
        graph.add_edge("load_metering", "route_metering")
        graph.add_conditional_edges(
            "route_metering",
            lambda state: state["workflow_route"],
            {route: route for route in ("data_quality", "normal", "single_anomaly", "high_risk")},
        )
        graph.add_edge("interpret_metering", END)
        return graph.compile()

    def _build_equipment_graph(self):
        graph = StateGraph(InspectionState)
        graph.add_node("load_equipment", self._load_context)
        graph.add_node("route_equipment", self._route_equipment)
        for route in ("uncertainty", "recovery", "critical", "degradation", "stable"):
            graph.add_node(route, self._evidence_node(route))
            graph.add_edge(route, "interpret_equipment")
        graph.add_node("interpret_equipment", self._interpret)
        graph.add_edge(START, "load_equipment")
        graph.add_edge("load_equipment", "route_equipment")
        graph.add_conditional_edges(
            "route_equipment",
            lambda state: state["workflow_route"],
            {route: route for route in ("uncertainty", "recovery", "critical", "degradation", "stable")},
        )
        graph.add_edge("interpret_equipment", END)
        return graph.compile()

    def _load_context(self, state: InspectionState) -> dict[str, Any]:
        module = state["module"]
        loader = self.loaders[module]
        context = loader(state["user_id"], date.fromisoformat(state["diagnosis_date"]))
        if not isinstance(context, dict):
            raise ValueError(f"{module} 诊断数据格式无效")
        return {"context": context}

    @staticmethod
    def _route_metering(state: InspectionState) -> dict[str, str]:
        context = state["context"]
        raw_quality_status = context.get("quality_status")
        quality_status = int(raw_quality_status) if raw_quality_status is not None else 2
        alerts = [str(item) for item in context.get("alerts") or [] if "不做诊断" not in str(item)]
        risk_level = str(context.get("risk_level") or "")
        if quality_status == 2:
            route = "data_quality"
        elif len(alerts) >= 2 or risk_level in {"高", "严重", "高风险", "严重风险"}:
            route = "high_risk"
        elif alerts:
            route = "single_anomaly"
        else:
            route = "normal"
        return {"workflow_route": route}

    @staticmethod
    def _route_equipment(state: InspectionState) -> dict[str, str]:
        current = state["context"].get("current_assessment") or {}
        trend = state["context"].get("trend_assessment") or {}
        confidence = float(current.get("confidence") or 0)
        stage = str(current.get("stage") or "")
        trend_label = str(trend.get("trend_label") or "")
        if confidence < 0.65 or not stage:
            route = "uncertainty"
        elif trend_label == "recovery":
            route = "recovery"
        elif stage in {"H3", "H4"} or trend_label in {"fast_decay", "abrupt_fault"}:
            route = "critical"
        elif stage in {"H1", "H2"} or trend_label == "slow_decay":
            route = "degradation"
        else:
            route = "stable"
        return {"workflow_route": route}

    def _evidence_node(self, route: str):
        def collect(state: InspectionState) -> dict[str, Any]:
            items = self._metering_evidence(state) if state["module"] == "metering" else self._equipment_evidence(state)
            return {"workflow_route": route, "evidence_items": [item.model_dump(mode="json") for item in items]}

        return collect

    @staticmethod
    def _item(
        evidence_id: str,
        category: str,
        statement: str,
        source: str,
        value: Any | None = None,
        unit: str | None = None,
    ) -> EvidenceItem:
        return EvidenceItem(evidence_id=evidence_id, category=category, statement=statement, source=source, value=value, unit=unit)

    def _metering_evidence(self, state: InspectionState) -> list[EvidenceItem]:
        context = state["context"]
        details = context.get("details") or {}
        quality = details.get("data_quality") or {}
        items = [
            self._item("M-QUALITY", "数据质量", f"诊断数据质量状态为 {context.get('quality_status', '未知')}，各管路完整度为 {quality.get('pipeline_completeness') or '暂无'}。", "计量诊断/数据质量", quality.get("pipeline_completeness")),
            self._item("M-VOLUME", "流量基线", f"当天观测用气量 {context.get('observed_volume', 0)} m³，历史基线正常量 {context.get('predicted_normal_volume', 0)} m³。", "计量诊断/历史基线", {"observed": context.get("observed_volume"), "baseline": context.get("predicted_normal_volume")}, "m³"),
            self._item("M-STATE", "状态交叉验证", f"模型用气状态为 {context.get('model_gas_state')}，远传观测状态为 {context.get('observed_gas_state')}。", "计量诊断/用气状态"),
            self._item("M-RISK", "综合风险", f"算法风险等级为 {context.get('risk_level')}，评分 {context.get('risk_score')}。", "计量诊断/风险规则", context.get("risk_score"), "分"),
            self._item("M-SPEC", "表具量程", f"表具量程判定为：{context.get('meter_spec_result', '未判定')}。", "计量诊断/量程适配", details.get("meter_spec")),
            self._item("M-MAKEUP", "补量估算", f"基线缺口 {context.get('baseline_missing_volume', 0)} m³，检定误差修正 {context.get('meter_bias_volume', 0)} m³，估算补量 {context.get('makeup_volume', 0)} m³。", "计量诊断/估算补量", context.get("makeup_volume"), "m³"),
        ]
        for index, alert in enumerate(context.get("alerts") or [], start=1):
            items.append(self._item(f"M-ALERT-{index}", "规则告警", str(alert), "计量诊断/联合规则"))
        intervals = context.get("anomaly_intervals") or []
        if intervals:
            estimated = sum(float(item.get("estimated_missing_volume") or 0) for item in intervals if isinstance(item, dict))
            items.append(self._item("M-INTERVAL", "异常区间", f"算法定位到 {len(intervals)} 个连续异常区间，区间估算缺口合计 {estimated:.2f} m³。", "计量诊断/异常区间", len(intervals), "个"))
        if state.get("field_text", "").strip():
            items.append(self._item("M-FIELD", "现场信息", state["field_text"].strip(), "人工补充"))
        return items

    def _equipment_evidence(self, state: InspectionState) -> list[EvidenceItem]:
        context = state["context"]
        current = context.get("current_assessment") or {}
        trend = context.get("trend_assessment") or {}
        explanation = context.get("model_explanation") or {}
        probabilities = current.get("probabilities") or {}
        top_probability = max(probabilities.items(), key=lambda item: float(item[1])) if probabilities else ("未知", 0)
        items = [
            self._item("E-STATE", "健康状态", f"设备处于 {current.get('stage', '未知')}（{current.get('stage_name', '未命名')}），健康指数 {current.get('health_index')}，风险等级 {current.get('risk_level')}。", "设备诊断/当前评估", current.get("health_index"), "HI"),
            self._item("E-CONFIDENCE", "模型置信", f"五阶段分类最大置信度为 {float(current.get('confidence') or 0) * 100:.1f}%，最高后验阶段为 {top_probability[0]}。", "设备诊断/阶段分类", current.get("confidence")),
            self._item("E-TREND", "时序趋势", f"近 {trend.get('window_days', 0)} 天趋势为 {trend.get('trend_name', trend.get('trend_label', '未知'))}，日斜率 {trend.get('daily_slope')} 点/日，最大单日下降 {trend.get('maximum_daily_drop')} 点。", "设备诊断/时序约束", trend),
            self._item("E-AXIS", "振动方向", f"三轴自适应权重为 {explanation.get('axis_weights') or '暂无'}。", "设备模型/轴注意力", explanation.get("axis_weights")),
            self._item("E-SCALE", "形态学尺度", f"尺度核 {explanation.get('scale_kernel_sizes') or '暂无'} 的权重为 {explanation.get('morphological_scale_weights') or '暂无'}。", "设备模型/多尺度特征", explanation.get("morphological_scale_weights")),
            self._item("E-CONDITION", "运行工况", f"当前采样窗口运行工况为 {current.get('operating_condition', '暂无')} m³/h。", "设备诊断/采样窗口", current.get("operating_condition"), "m³/h"),
        ]
        if state.get("field_text", "").strip():
            items.append(self._item("E-FIELD", "现场信息", state["field_text"].strip(), "人工补充"))
        return items

    @staticmethod
    def _level_rank(level: str) -> int:
        if any(word in str(level) for word in ("严重", "极高")):
            return 4
        if "高" in str(level):
            return 3
        if "中" in str(level):
            return 2
        return 1

    def _fallback(self, state: InspectionState) -> dict[str, Any]:
        return self._metering_fallback(state) if state["module"] == "metering" else self._equipment_fallback(state)

    def _metering_fallback(self, state: InspectionState) -> dict[str, Any]:
        context = state["context"]
        alerts = [str(item) for item in context.get("alerts") or []]
        route = state["workflow_route"]
        evidence_ids = [item["evidence_id"] for item in state["evidence_items"]]
        cause_map = [
            ("走气未走字", "计量脉冲、表体或远传链路未正确记录实际用气"),
            ("计数时长", "远传计数持续时间与压力响应所反映的实际用气时长不一致"),
            ("过滤器可能阻塞", "过滤器阻塞、阀门开度或管路阻力差异导致双管流量失衡"),
            ("负压损", "调压或管路阻力异常造成压力损失"),
            ("高压损", "调压或管路阻力异常造成压力损失"),
            ("压力传感器可能故障", "压力传感器漂移、安装或采集链路异常"),
            ("温度异常", "温度传感器或温度补偿链路异常"),
        ]
        causes = []
        for keyword, cause in cause_map:
            if any(keyword in alert for alert in alerts):
                causes.append({"rank": len(causes) + 1, "cause": cause, "confidence": 0.78 if not causes else 0.62, "supporting_evidence_ids": [item for item in evidence_ids if item.startswith("M-ALERT") or item in {"M-STATE", "M-RISK"}], "counter_evidence_ids": []})
        if not causes:
            causes.append({"rank": 1, "cause": "当前结构化证据未支持明确计量故障", "confidence": 0.82 if route == "normal" else 0.5, "supporting_evidence_ids": ["M-QUALITY", "M-RISK"], "counter_evidence_ids": ["M-VOLUME"] if context.get("baseline_missing_volume") else []})
        counter = []
        if not alerts:
            counter.append("联合规则未产生可推送告警；历史基线偏差本身不能证明计量故障。")
        if context.get("observed_gas_state") == 1:
            counter.append("远传瞬时流量仍形成有效计量，不支持完全不走字。")
        missing = []
        if not state.get("field_text", "").strip():
            missing.append("缺少停产、阀门状态、现场字轮及维护情况等现场记录。")
        if not (context.get("details") or {}).get("meter_error_model", {}).get("available"):
            missing.append("有效检定点不足，无法形成可靠的检定误差修正模型。")
        raw_quality_status = context.get("quality_status")
        if (int(raw_quality_status) if raw_quality_status is not None else 2) == 2:
            missing.append("当天数据完整度不足，部分异常类型无法可靠判断。")
        checks = ["核对远传累计量与机械字轮示值", "核查阀门状态、生产负荷与停产记录", "检查流量、压力、温度传感器供电和通信"]
        if alerts:
            checks.insert(0, "按算法告警对应时段回放曲线，并在相同工况下现场复测")
        return {
            "inspection_conclusion": context.get("summary") or "计量诊断已完成。",
            "analysis_confidence": 0.45 if route == "data_quality" else 0.72 if alerts else 0.82,
            "possible_causes": causes,
            "counter_evidence": counter,
            "missing_evidence": missing,
            "checklist": checks,
            "work_order_advice": "高风险告警建议形成核查工单；无规则告警时继续监测，不应仅凭历史基线差异直接追补。",
        }

    def _equipment_fallback(self, state: InspectionState) -> dict[str, Any]:
        context = state["context"]
        current = context.get("current_assessment") or {}
        trend = context.get("trend_assessment") or {}
        route = state["workflow_route"]
        stage = str(current.get("stage") or "未知")
        trend_name = str(trend.get("trend_name") or trend.get("trend_label") or "未知")
        cause_text = {
            "critical": "振动特征与近期健康轨迹共同指向明显性能退化或故障状态",
            "degradation": "振动特征持续偏离健康分布，设备可能处于早期或渐进式退化",
            "recovery": "健康指标回升更符合检修后恢复或工况恢复",
            "uncertainty": "阶段置信度不足，当前样本可能受工况变化或采集质量影响",
            "stable": "当前振动和健康轨迹未支持明确设备故障",
        }[route]
        counter = []
        if route in {"critical", "degradation"} and float(trend.get("daily_slope") or 0) >= 0:
            counter.append("近期健康指数没有继续下降，弱化了持续恶化假设。")
        if route == "stable":
            counter.append("当前阶段和近七天趋势均稳定，不支持立即停机处置。")
        missing = []
        if not state.get("field_text", "").strip():
            missing.append("缺少现场异响、温升、润滑、负荷变化及近期维修记录。")
        if float(current.get("confidence") or 0) < 0.65:
            missing.append("模型置信度偏低，需要复测振动并核对传感器安装状态。")
        confidence = max(0.35, min(0.92, float(current.get("confidence") or 0.5)))
        return {
            "inspection_conclusion": f"设备当前为 {stage}（{current.get('stage_name', '待判定')}），健康指数 {current.get('health_index')}，近期趋势为{trend_name}。",
            "analysis_confidence": confidence,
            "possible_causes": [{"rank": 1, "cause": cause_text, "confidence": confidence, "supporting_evidence_ids": ["E-STATE", "E-CONFIDENCE", "E-TREND", "E-AXIS", "E-SCALE"], "counter_evidence_ids": []}],
            "counter_evidence": counter,
            "missing_evidence": missing,
            "checklist": ["复测三轴振动并确认传感器安装牢固", "核对运行负荷，排除工况变化造成的特征漂移", "检查轴承、转子、齿轮、联轴器和润滑状态"],
            "work_order_advice": context.get("recommended_action") or "结合风险等级安排复测或检修。",
        }

    @staticmethod
    def _message_text(message: Any) -> str:
        content = getattr(message, "content", message)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
        return str(content or "")

    def _invoke_llm(self, state: InspectionState, fallback: dict[str, Any]) -> LlmInterpretation:
        if self.model_client is None:
            raise RuntimeError("未配置检查解读模型")
        prompt = {
            "task": "基于可信算法证据完成燃气工业诊断解释。对原因排序，同时列出支持证据、反证、缺失证据和现场核查顺序。不得修改、重算或虚构任何风险分数、健康指数、用气量、补量、阶段、日期和算法告警。不要把相关性写成已证实因果。",
            "module": state["module"],
            "workflow_route": state["workflow_route"],
            "algorithm_summary": state["context"].get("summary") if state["module"] == "metering" else {"current_assessment": state["context"].get("current_assessment"), "trend_assessment": state["context"].get("trend_assessment")},
            "evidence_items": state["evidence_items"],
            "field_information": state.get("field_text") or "未提供",
            "fallback_reference": fallback,
            "output_schema": LlmInterpretation.model_json_schema(),
        }
        messages = [
            ("system", "你是燃气计量与旋转设备诊断解释专家。只输出一个符合给定 JSON Schema 的 JSON 对象，不输出 Markdown。"),
            ("human", json.dumps(prompt, ensure_ascii=False, default=str)),
        ]
        # JSON mode只保证语法，随后必须由 Pydantic 校验字段和范围。
        model = self.model_client.bind(response_format={"type": "json_object"}) if hasattr(self.model_client, "bind") else self.model_client
        response = model.invoke(messages)
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", self._message_text(response).strip(), flags=re.IGNORECASE)
        return LlmInterpretation.model_validate_json(content)

    @staticmethod
    def _stable_value(value: Any) -> Any:
        volatile_keys = {"run_id", "report_id", "work_order_id", "interval_id", "created_at", "updated_at", "created_time"}
        if isinstance(value, dict):
            return {str(key): InspectionAgent._stable_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0])) if str(key) not in volatile_keys}
        if isinstance(value, (list, tuple)):
            return [InspectionAgent._stable_value(item) for item in value]
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if hasattr(value, "item"):
            try:
                return InspectionAgent._stable_value(value.item())
            except (TypeError, ValueError):
                pass
        return value

    def _fingerprint(self, state: InspectionState) -> str:
        payload = {
            "workflow_version": WORKFLOW_VERSION,
            "model": self.model_name,
            "module": state["module"],
            "user_id": state["user_id"],
            "diagnosis_date": state["diagnosis_date"],
            "field_text": state.get("field_text", "").strip(),
            "route": state["workflow_route"],
            "context": self._stable_value(state["context"]),
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _get_cached(self, fingerprint: str) -> dict[str, Any] | None:
        with duckdb.connect(str(self.result_db), read_only=True) as con:
            row = con.execute("SELECT report_json FROM inspection.workflow_report WHERE input_fingerprint=?", [fingerprint]).fetchone()
        if row is None:
            return None
        report = json.loads(row[0])
        report["cache_hit"] = True
        return report

    def _save_cached(self, fingerprint: str, state: InspectionState, report: dict[str, Any]) -> None:
        with duckdb.connect(str(self.result_db)) as con:
            con.execute(
                """
                INSERT INTO inspection.workflow_report
                (report_id,input_fingerprint,module,user_id,diagnosis_date,workflow_route,workflow_version,model_name,report_json)
                VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT (input_fingerprint) DO NOTHING
                """,
                [report["report_id"], fingerprint, state["module"], state["user_id"], state["diagnosis_date"], state["workflow_route"], WORKFLOW_VERSION, self.model_name, json.dumps(report, ensure_ascii=False, default=str)],
            )

    def _merge_report(self, state: InspectionState, interpretation: dict[str, Any], *, fingerprint: str, generator: str) -> dict[str, Any]:
        context = state["context"]
        level = str(context.get("risk_level") or (context.get("current_assessment") or {}).get("risk_level") or "待判定")
        score = context.get("risk_score")
        if state["module"] == "equipment":
            score = round(100 - float((context.get("current_assessment") or {}).get("health_index") or 0), 2)
        rank = self._level_rank(level)
        evidence = state["evidence_items"]
        valid_ids = {item["evidence_id"] for item in evidence}
        causes = []
        for index, cause in enumerate(interpretation.get("possible_causes") or [], start=1):
            item = dict(cause)
            item["rank"] = index
            item["supporting_evidence_ids"] = [value for value in item.get("supporting_evidence_ids", []) if value in valid_ids]
            item["counter_evidence_ids"] = [value for value in item.get("counter_evidence_ids", []) if value in valid_ids]
            causes.append(item)
        return {
            "report_id": f"IR-{fingerprint[:16]}", "module": state["module"], "user_id": state["user_id"], "diagnosis_date": state["diagnosis_date"],
            "workflow_route": state["workflow_route"], "workflow_version": WORKFLOW_VERSION,
            "inspection_conclusion": interpretation["inspection_conclusion"], "risk_level": level, "risk_score": score,
            "decision": "立即处置" if rank >= 4 else "优先核查" if rank == 3 else "计划核查" if rank == 2 else "持续监测",
            "analysis_confidence": interpretation["analysis_confidence"], "possible_causes": causes,
            "evidence_items": evidence, "evidence_chain": [item["statement"] for item in evidence],
            "counter_evidence": interpretation.get("counter_evidence") or [], "missing_evidence": interpretation.get("missing_evidence") or [],
            "field_information": state.get("field_text", "").strip() or "未录入现场补充信息",
            "checklist": interpretation.get("checklist") or [], "work_order_advice": interpretation["work_order_advice"],
            "data_boundary": "算法数值、状态和风险等级保持后端诊断原值；LLM只负责原因排序、反证分析和核查建议，不执行代码或重新计算。",
            "generator": generator, "cache_hit": False,
        }

    def _interpret(self, state: InspectionState) -> dict[str, Any]:
        fallback = self._fallback(state)
        fingerprint = self._fingerprint(state)
        with self._generation_lock:
            cached = self._get_cached(fingerprint)
            if cached is not None:
                return {"fallback_report": fallback, "final_report": cached}
            try:
                parsed = self._invoke_llm(state, fallback)
                report = self._merge_report(state, parsed.model_dump(mode="json"), fingerprint=fingerprint, generator=f"workflow-llm:{self.model_name}")
                self._save_cached(fingerprint, state, report)
            except Exception as exc:
                # 网络、供应商兼容或结构校验失败均不得让算法页面失去解释结果。
                report = self._merge_report(state, fallback, fingerprint=fingerprint, generator="workflow-local-fallback")
                report["llm_notice"] = f"模型解读不可用，已采用可审计工作流回退：{type(exc).__name__}"
            return {"fallback_report": fallback, "final_report": report}

    def generate(self, module: str, user_id: str, diagnosis_date: str, field_text: str = "", context: dict[str, Any] | None = None) -> dict[str, Any]:
        """执行指定模块工作流；context 参数仅为旧客户端兼容占位。"""

        if module not in {"metering", "equipment"}:
            raise ValueError("module 仅支持 metering 或 equipment")
        try:
            date.fromisoformat(str(diagnosis_date))
        except ValueError as exc:
            raise ValueError("diagnosis_date 必须是 YYYY-MM-DD") from exc
        initial: InspectionState = {"module": module, "user_id": str(user_id), "diagnosis_date": str(diagnosis_date), "field_text": str(field_text or "")[:4000]}  # type: ignore[typeddict-item]
        graph = self.metering_graph if module == "metering" else self.equipment_graph
        result = graph.invoke(initial)
        return result["final_report"]
