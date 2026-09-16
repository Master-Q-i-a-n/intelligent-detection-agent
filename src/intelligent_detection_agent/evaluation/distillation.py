"""评测专用轨迹采集、SFT 序列化和确定性蒸馏评分。"""

from __future__ import annotations

import copy
import json
from collections import Counter
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from .models import AgentEvalCase, DistillationConfig, JointJudgeResult


SCORING_VERSION = "distillation-v1"


def sft_message(message: Any) -> dict[str, Any]:
    """仅序列化训练字段，避免把 reasoning、usage 或附件引用混入文本数据。"""
    roles = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}
    role = roles.get(getattr(message, "type", ""))
    if role is None:
        raise ValueError("不支持的消息角色")
    if getattr(message, "additional_kwargs", {}).get("image_attachments"):
        raise ValueError("第一版不支持多模态附件")
    content = message.content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in {"reasoning", "thinking"}:
                continue
            if isinstance(block, dict) and block.get("type") == "tool_call" and role == "assistant":
                # v3 将工具调用同时投影到 content 和 tool_calls；验证一致后只导出一次。
                mirrored = next((call for call in message.tool_calls if call.get("id") == block.get("id")), None)
                if mirrored is None or any(mirrored.get(key) != block.get(key) for key in ("name", "args")):
                    raise ValueError("内容块与工具调用字段不一致")
                continue
            if not isinstance(block, dict) or block.get("type") not in {"text", "output_text"}:
                kind = block.get("type", "unknown") if isinstance(block, dict) else type(block).__name__
                raise ValueError(f"第一版不支持非文本内容块：{kind}")
            parts.append(block.get("text", ""))
        content = "".join(parts)
    if not isinstance(content, str):
        raise ValueError("消息正文不是文本")
    if "<think>" in content or "<thinking>" in content:
        raise ValueError("正文包含推理标签，无法安全导出")
    result: dict[str, Any] = {"role": role, "content": content}
    if role == "assistant":
        if getattr(message, "invalid_tool_calls", None):
            raise ValueError("存在无法解析的工具调用")
        calls = getattr(message, "tool_calls", [])
        if calls:
            result["tool_calls"] = []
            for call in calls:
                if not call.get("id") or not isinstance(call.get("args"), dict):
                    raise ValueError("工具调用缺少ID或对象参数")
                result["tool_calls"].append({
                    "id": call["id"], "type": "function",
                    "function": {"name": call["name"], "arguments": copy.deepcopy(call["args"])},
                })
    elif role == "tool":
        result["tool_call_id"] = message.tool_call_id
    return result


class TrajectoryCollector(BaseCallbackHandler):
    """只随被测运行注入；完成消息由 v3 模型回调提供，不拼接 token。"""

    run_inline = True

    def __init__(self) -> None:
        self.calls: dict[str, dict[str, Any]] = {}
        self.errors: list[str] = []
        self.tool_messages: dict[str, dict[str, Any]] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        if (metadata or {}).get("langgraph_node") != "model":
            return
        try:
            if len(messages) != 1:
                raise ValueError("不支持批量模型消息")
            # invocation_params 可能包含凭证；只读取工具 Schema 和模型名称。
            params = kwargs.get("invocation_params") or {}
            tools = []
            for tool in params.get("tools") or []:
                if not isinstance(tool, dict) or tool.get("type") != "function":
                    raise ValueError("不支持非函数工具")
                function = tool["function"]
                tools.append({"type": "function", "function": {
                    "name": function["name"], "description": function.get("description", ""),
                    "parameters": copy.deepcopy(function["parameters"]),
                }})
            self.calls[str(run_id)] = {
                "messages": [sft_message(m) for m in messages[0]], "tools": tools,
                "model": params.get("model") or params.get("model_name"),
            }
        except (ValueError, KeyError, TypeError) as exc:
            self.errors.append(f"模型输入无法序列化：{exc}")

    def on_llm_end(self, response, *, run_id, **kwargs):
        call = self.calls.get(str(run_id))
        if call is not None:
            try:
                call["output"] = sft_message(response.generations[0][0].message)
            except (ValueError, IndexError, AttributeError, TypeError) as exc:
                self.errors.append(f"模型输出无法序列化：{exc}")

    def on_llm_error(self, error, *, run_id, **kwargs):
        if str(run_id) in self.calls:
            self.errors.append("模型调用未完整结束")

    def observe_state(self, messages: list[Any]) -> None:
        for message in messages:
            if getattr(message, "type", None) == "tool":
                try:
                    normalized = sft_message(message)
                    call_id = message.tool_call_id
                    prior = self.tool_messages.get(call_id)
                    if prior is not None and prior != normalized:
                        raise ValueError("同一工具返回内容发生改写")
                    self.tool_messages[call_id] = normalized
                except (ValueError, TypeError) as exc:
                    self.errors.append(str(exc))

    def sample(self, *, expected_interrupt: bool = False) -> dict[str, Any]:
        if self.errors:
            raise ValueError("；".join(dict.fromkeys(self.errors)))
        if not self.calls:
            raise ValueError("未采集到模型调用")
        messages: list[dict[str, Any]] = []
        tools = next(iter(self.calls.values()))["tools"]
        for call_index, call in enumerate(self.calls.values(), start=1):
            if "output" not in call:
                raise ValueError("模型完成输出缺失")
            if call["tools"] != tools:
                raise ValueError("工具定义变化，无法表示为固定 tools 样本")
            current = call["messages"]
            comparable = list(current)
            for position, message in enumerate(messages):
                calls = message.get("tool_calls", []) if message.get("role") == "assistant" else []
                if len(calls) < 2 or position >= len(current) or current[position] != message:
                    continue
                start, end = position + 1, position + 1 + len(calls)
                old_batch, new_batch = messages[start:end], current[start:end]
                expected_ids = {item["id"] for item in calls}
                # 只比较紧邻同一 assistant 调用的完整并行返回块，拒绝缺失、重复和跨批移动。
                if len(expected_ids) != len(calls) or any(
                    len(batch) != len(calls)
                    or any(item.get("role") != "tool" for item in batch)
                    or {item.get("tool_call_id") for item in batch} != expected_ids
                    for batch in (old_batch, new_batch)
                ):
                    continue
                if {item["tool_call_id"]: item for item in old_batch} == {
                    item["tool_call_id"]: item for item in new_batch
                }:
                    comparable[start:end] = old_batch
            if comparable[:len(messages)] != messages:
                # 记录首个差异的位置和字段，不归档原文，避免保存候选轨迹或推理内容。
                index = next((i for i, pair in enumerate(zip(messages, current)) if pair[0] != pair[1]), min(len(messages), len(current)))
                old = messages[index] if index < len(messages) else {}
                new = current[index] if index < len(current) else {}
                fields = sorted(key for key in old.keys() | new.keys() if old.get(key) != new.get(key))
                raise ValueError(
                    "模型上下文被压缩或改写，无法合并整段轨迹："
                    f"模型调用序号={call_index}，消息序号={index + 1}，"
                    f"字段={','.join(fields)}，角色={old.get('role')}->{new.get('role')}，"
                    f"历史长度={len(messages)}，当前长度={len(current)}，"
                    f"正文长度={len(old.get('content', ''))}->{len(new.get('content', ''))}"
                )
            # 保留各批首次实际观察到的顺序；只追加新消息，不改写已验证的历史或模型输入。
            messages = copy.deepcopy(messages) + copy.deepcopy(current[len(messages):]) + [copy.deepcopy(call["output"])]
        # 终点可能是工具完成或人工审批中断，不能杜撰尚未发生的返回。
        for call in messages[-1].get("tool_calls", []):
            result = self.tool_messages.get(call["id"])
            if result is not None:
                messages.append(copy.deepcopy(result))
        validate_messages(messages, tools, expected_interrupt=expected_interrupt)
        sample = {"messages": messages, "tools": copy.deepcopy(tools)}
        # 在评分前确认可写入严格 JSON，避免入选后才发现 NaN 或非 JSON 参数。
        try:
            json.dumps(sample, ensure_ascii=False, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise ValueError("轨迹包含非JSON值") from exc
        return sample


def validate_messages(messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, expected_interrupt: bool) -> None:
    names = {tool["function"]["name"] for tool in tools}
    pending: set[str] = set()
    seen: set[str] = set()
    for message in messages:
        if message["role"] == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                raise ValueError("工具返回没有对应调用或重复返回")
            pending.remove(call_id)
            continue
        if pending:
            raise ValueError("后续消息前存在缺失的工具返回")
        for call in message.get("tool_calls", []):
            if call["id"] in seen or call["function"]["name"] not in names:
                raise ValueError("工具调用ID重复或工具定义缺失")
            seen.add(call["id"])
            pending.add(call["id"])
    if pending and not expected_interrupt:
        raise ValueError("轨迹终点缺少工具返回")
    if not expected_interrupt and messages[-1]["role"] != "assistant":
        raise ValueError("轨迹缺少最终助手输出")
    if not expected_interrupt and not messages[-1]["content"].strip():
        raise ValueError("轨迹最终助手输出为空")


def score_distillation(
    case: AgentEvalCase, calls: list[dict[str, Any]], hard_passed: bool,
    joint: JointJudgeResult | None, config: DistillationConfig,
    *, trace_error: str | None = None, judge_error: str | None = None,
) -> dict[str, Any]:
    reasons = []
    if not hard_passed:
        reasons.append("硬规则未通过")
    if trace_error:
        reasons.append(trace_error)
    if judge_error:
        reasons.append(judge_error)
    if joint is None:
        reasons.append("缺少有效联合评分")
    elif not joint.passed or joint.score < 4:
        reasons.append("答案评审未通过")
    if joint and joint.unrecovered_failure:
        reasons.append("存在未恢复的执行失败")
    penalties: dict[str, int] = {}
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    budgets = {item.name: item.max_calls for item in case.required_tools}
    redundant = set(joint.redundant_call_ids) if joint else set()
    for call in calls:
        call_id = call["tool_call_id"]
        if call_id in seen:
            continue
        seen.add(call_id)
        name = call["name"]
        counts[name] += 1
        budget = budgets.get(name)
        penalty = 10 if budget is not None and counts[name] > budget else 0
        if call_id in redundant:
            penalty = max(penalty, 15)
        if penalty:
            penalties[call_id] = penalty
    answer = joint.score * 20 if joint else None
    process = (joint.parameter_score * 0.6 + joint.dependency_score * 0.25 + joint.recovery_score * 0.15) if joint else None
    efficiency = max(0, 100 - sum(penalties.values())) if joint else None
    total = sum(value * weight for value, weight in zip((answer, process, efficiency), config.weights)) if joint else None
    eligible = not reasons
    if eligible and total < config.threshold:
        reasons.append("总分低于阈值")
    return {
        "scoring_version": SCORING_VERSION, "answer_score": answer, "process_score": process,
        "efficiency_score": efficiency, "total_score": total,
        "threshold": config.threshold, "weights": list(config.weights),
        "eligible": eligible, "selected": eligible and total >= config.threshold,
        "issues": reasons, "penalties": penalties,
        "joint_judge": joint.model_dump(mode="json") if joint else None,
        "trace_error": trace_error, "judge_error": judge_error,
        "exported": False, "sft_line": None,
    }
