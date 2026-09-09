"""开放式任务的独立 LLM 评审。"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI

from ..safety_operations.env import load_project_env
from .models import AgentEvalCase, JudgeResult
from .telemetry import normalize_usage


def _message_text(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    parts = []
    for item in message.content if isinstance(message.content, list) else []:
        if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
            parts.append(str(item.get("text") or ""))
    return "\n".join(parts)


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("评审结果必须是 JSON 对象")
    return value


class EvaluationJudge:
    """使用独立、关闭思考的模型调用评估答案质量。"""

    def __init__(self, root: Path) -> None:
        load_project_env(root / ".env")
        self.model_name = os.getenv("EVAL_JUDGE_MODEL") or os.getenv("CHAT_LLM_MODEL") or "deepseek-chat"
        self.base_url = (
            os.getenv("EVAL_JUDGE_BASE_URL")
            or os.getenv("CHAT_LLM_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or "https://api.deepseek.com"
        ).rstrip("/")
        self.api_key = (
            os.getenv("EVAL_JUDGE_API_KEY")
            or os.getenv("CHAT_LLM_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.provider = (
            os.getenv("EVAL_JUDGE_PROVIDER")
            or os.getenv("CHAT_LLM_PROVIDER")
            or ("deepseek" if "deepseek" in self.base_url else "openai-compatible")
        ).lower()
        if not self.api_key:
            raise RuntimeError("LLM 评审未配置 API Key，请设置 EVAL_JUDGE_API_KEY 或聊天模型 Key。")

    def _build_model(self):
        common = {
            "model": self.model_name,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "temperature": 0,
            "timeout": 120,
            "max_retries": 1,
            "streaming": False,
        }
        if self.provider == "deepseek":
            common.pop("temperature", None)
            common["extra_body"] = {"thinking": {"type": "disabled"}}
            return ChatDeepSeek(**common)
        return ChatOpenAI(**common)

    async def grade(
        self,
        case: AgentEvalCase,
        answer: str,
        oracle_results: list[dict[str, Any]],
        execution_context: dict[str, Any],
    ) -> JudgeResult:
        criteria = case.judge_criteria or ["事实准确", "回答完整", "结论相关", "说明数据边界"]
        prompt = f"""
你是燃气业务 Agent 测评员。只评估最终答案，不评价文风偏好，也不要输出推理过程。

任务：{case.description}
用户输入：{json.dumps([turn.message or turn.resume for turn in case.turns], ensure_ascii=False)}
参考事实：{json.dumps(oracle_results, ensure_ascii=False, default=str)[:20000]}
执行记录：{json.dumps(execution_context, ensure_ascii=False, default=str)[:12000]}
评分要求：{json.dumps(criteria, ensure_ascii=False)}
被测答案：{answer[:30000]}

参考事实只用于核验，不能视为被测答案已经陈述的内容。执行记录中的状态、中断和产物是可核查事实；
若执行报错、应有最终回答但答案为空，或要求的产物缺失，必须判为不通过。

返回且只返回 JSON：
{{"score": 1到5的整数, "passed": true或false, "reason": "不超过120字的可核查理由"}}
4分及以上且没有关键事实错误时 passed=true。
""".strip()
        model = self._build_model()
        started = time.monotonic()
        messages: list[AIMessage] = []
        parse_error: Exception | None = None
        for attempt in range(2):
            request = prompt if attempt == 0 else prompt + "\n上一次格式无效，请严格只返回一个 JSON 对象。"
            message = await model.ainvoke(request)
            messages.append(message)
            try:
                parsed = _parse_json(_message_text(message))
                usage_rows = [normalize_usage(item) for item in messages]
                usage = {
                    key: sum(int(row[key] or 0) for row in usage_rows)
                    for key in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens", "total_tokens")
                }
                return JudgeResult(
                    score=parsed.get("score"),
                    passed=parsed.get("passed"),
                    reason=str(parsed.get("reason") or ""),
                    usage=usage,
                    latency_ms=round((time.monotonic() - started) * 1000, 2),
                )
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                parse_error = exc
        raise RuntimeError(f"LLM 评审返回格式无效：{parse_error}")
