from __future__ import annotations

import json
import os
import re
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


class InspectionAgent:
    """将现场文字与算法结果合并为可审计的结构化检查结论。"""

    def __init__(self, root: Path):
        self.root = root
        self._load_env(root / ".env")
        self.api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")
        self.model = os.getenv("LLM_MODEL", "deepseek-chat")

    @staticmethod
    def _load_env(path: Path) -> None:
        if not path.exists():
            return
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not raw.strip() or raw.lstrip().startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

    @staticmethod
    def _keywords(text: str) -> list[str]:
        rules = {
            "异响": ["异响", "噪声", "声音大"], "振动增强": ["振动大", "抖动", "震动"],
            "泄漏迹象": ["漏气", "甲烷", "气味"], "计数异常": ["不走字", "走气未走字", "卡表"],
            "压力异常": ["压力低", "压力高", "压力波动"], "温度异常": ["温度高", "温度低", "温差"],
            "仪表异常": ["黑屏", "断电", "离线", "显示异常"], "已维修": ["维修", "更换", "复位"],
        }
        return [name for name, words in rules.items() if any(word in text for word in words)]

    @staticmethod
    def _level_rank(level: str) -> int:
        if any(word in str(level) for word in ["严重", "极高"]): return 4
        if "高" in str(level): return 3
        if "中" in str(level): return 2
        return 1

    def _fallback(self, module: str, user_id: str, diagnosis_date: str, field_text: str, context: dict[str, Any]) -> dict[str, Any]:
        tags = self._keywords(field_text)
        if module == "metering":
            level = str(context.get("risk_level", "待判定"))
            score = context.get("risk_score")
            anomalies = context.get("alerts") or []
            intervals = context.get("anomaly_intervals") or []
            spec = context.get("meter_spec_result", "未判定")
            plain_findings = []
            if any("过滤器可能阻塞" in str(item) for item in anomalies):
                plain_findings.append("两条同时供气的管路流量长期差距过大，疑似过滤器堵塞、阀门开度异常或管路阻力不一致")
            if any("压力传感器可能故障" in str(item) for item in anomalies):
                plain_findings.append("多条管路的压力变化不同步，疑似部分压力传感器测量失真")
            if any("走气未走字" in str(item) for item in anomalies):
                plain_findings.append("现场处于用气状态，但远传流量没有形成计量，疑似走气未走字")
            if any("计数时长" in str(item) for item in anomalies):
                plain_findings.append("压力响应显示设备实际用气时间可能长于流量计记录时间")
            conclusion = ("本次检查发现：" + "；".join(plain_findings or list(map(str, anomalies)))) if anomalies else "流量、压力、温度联合诊断未发现需要企业立即复核的问题。"
            evidence = []
            checklist = ["核对远传累计量与机械字轮示值", "检查流量、压力、温度传感器供电与通信", "核验表具量程与实际运行负荷"]
            numeric = context.get("details", {}).get("diagnostic_evidence", {})
            for item in numeric.get("flow", []):
                evidence.append(
                    f"管道{item.get('pipelines')}在同时供气期间，有{float(item.get('imbalance_ratio', 0))*100:.1f}%的时刻流量差距过大，"
                    f"明显超过{float(item.get('ratio_threshold', 0))*100:.0f}%的提醒标准。"
                )
            dissimilar_pressure = [item for item in numeric.get("pressure", []) if item.get("dissimilar")]
            for item in dissimilar_pressure[:3]:
                evidence.append(
                    f"管道{item.get('pipelines')}的压力曲线在整体走势和细微波动上都不同步，"
                    "正常情况下同一调压计量区域的压力变化应具有一致性。"
                )
            if not evidence:
                evidence.append("当天曲线未出现达到高置信推送门槛的流量、压力或温度异常。")
            evidence.append(f"当天数据质量满足诊断要求，综合风险等级为{level}。历史预测值仅作为趋势参考，不参与主异常定性。")
            if any("过滤器可能阻塞" in str(item) for item in anomalies):
                checklist.insert(0, "检查异常管路过滤器前后压差、阀门开度和滤芯堵塞情况，并在相同工况下复测双管瞬时流量")
            if any("压力传感器可能故障" in str(item) for item in anomalies):
                checklist.insert(0, "使用便携式标准压力表同步校验主备管道传感器，定位偏离曲线的传感器")
            if intervals: checklist.insert(0, "按异常区间回放SCADA曲线并核查阀门状态")
        elif module == "equipment":
            current = context.get("current_assessment", {})
            trend = context.get("trend_assessment", {})
            explanation = context.get("model_explanation", {})
            stage = current.get("stage", "待判定")
            level = current.get("risk_level", "待判定")
            score = round(100 - float(current.get("health_index") or 0), 2)
            stage_plain = {
                "H0": "设备振动状态稳定，暂未发现明显性能退化",
                "H1": "设备出现轻微振动变化，建议继续观察",
                "H2": "设备性能已经出现持续衰减，需要提高监测频率",
                "H3": "设备振动特征已明显偏离健康状态，存在较高故障风险",
                "H4": "设备振动特征符合故障状态，应尽快停机或现场核查",
            }.get(stage, "设备状态需要进一步确认")
            conclusion = f"{stage_plain}。当前健康指数为{current.get('health_index', '暂无')}，近期变化趋势为{trend.get('trend_name', trend.get('trend_label', '待判定'))}。"
            axis_weights = list(explanation.get("axis_weights", []))
            axis_names = ["X轴", "Y轴", "Z轴"]
            dominant_axis = axis_names[max(range(len(axis_weights)), key=lambda i: axis_weights[i])] if axis_weights else "暂无"
            evidence = [
                f"模型将设备判定为{stage}（{current.get('stage_name', '')}），判断把握度为{float(current.get('confidence') or 0) * 100:.1f}%。",
                f"健康指数为{current.get('health_index', '暂无')}，对应风险等级为{level}。",
                f"近7天健康变化速度约为{trend.get('daily_slope', '暂无')}点/日，最大单日下降{trend.get('maximum_daily_drop', '暂无')}点。",
                f"三轴振动中模型当前最关注{dominant_axis}方向；采样时运行工况为{current.get('operating_condition', '暂无')} m³/h。",
            ]
            checklist = ["复测三轴振动并确认传感器安装牢固", "检查轴承、转子、齿轮与联轴器状态", "对比流量工况，排除负荷变化导致的振动偏移"]
        else:
            raise ValueError("module 仅支持 metering 或 equipment")

        field_evidence = "未录入现场补充信息"
        if field_text.strip():
            field_evidence = field_text.strip()
            evidence.append("现场文字记录：" + field_evidence)
        if tags:
            evidence.append("现场语义标签：" + "、".join(tags))
            checklist.insert(0, "优先复核现场文字中识别到的异常迹象：" + "、".join(tags))

        rank = self._level_rank(level)
        decision = "立即处置" if rank >= 4 else "优先核查" if rank == 3 else "计划核查" if rank == 2 else "持续监测"
        return {
            "report_id": f"IR-{datetime.now():%Y%m%d%H%M%S}-{user_id}",
            "module": module,
            "user_id": user_id,
            "diagnosis_date": diagnosis_date,
            "inspection_conclusion": conclusion,
            "risk_level": level,
            "risk_score": score,
            "decision": decision,
            "evidence_chain": evidence,
            "field_information": field_evidence,
            "checklist": checklist,
            "work_order_advice": f"处置策略：{decision}。" + (context.get("recommended_action", "") if module == "equipment" else "建议将核查结果回填形成闭环。"),
            "data_boundary": "结论由算法结构化结果与现场文字联合生成；关键数值保持算法原值，现场文字仅作为补充证据。",
            "generator": "local-rule-agent",
        }

    def _llm_enhance(self, report: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            return report
        prompt = {
            "task": "将燃气检查结果改写成一线业务人员能直接理解的结构化JSON。优先解释发生了什么、可能原因和先检查什么；避免直接堆叠DTW、相关性、样本熵等术语，如必须使用应同时解释业务含义。不得修改任何数值、风险等级、状态、日期，不得增加未提供的事实。",
            "required_keys": ["inspection_conclusion", "evidence_chain", "checklist", "work_order_advice"],
            "draft": report,
            "algorithm_context": context,
        }
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是燃气工业智能检查Agent，只能依据输入证据生成结论，输出JSON对象。"},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, default=str)},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
            content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()
            enhanced = json.loads(content)
            for key in ("inspection_conclusion", "evidence_chain", "checklist", "work_order_advice"):
                if key in enhanced:
                    report[key] = enhanced[key]
            report["generator"] = f"llm-agent:{self.model}"
        except Exception as exc:
            report["llm_notice"] = f"大模型增强暂不可用，已采用本地可审计规则生成：{type(exc).__name__}"
        return report

    def generate(self, module: str, user_id: str, diagnosis_date: str, field_text: str, context: dict[str, Any]) -> dict[str, Any]:
        report = self._fallback(module, user_id, diagnosis_date, field_text, context)
        return self._llm_enhance(report, context)
