from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from deepagents import (
    FilesystemPermission,
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from safety_operations.env import load_project_env

from .schemas import ChatArtifact, ChatResumeRequest, ChatTurnResponse
from .tools import build_agent_tools, build_query_error_middleware


SYSTEM_PROMPT = """
你是曜衡智控燃气业务智能助手。你在一个单智能体工具调用循环中工作，不得委派任务。

工作边界：
1. 只回答本项目中的用气、智能计量、智能设备、安全作业、工单和报告问题。
2. 数据事实必须来自只读查询工具；不得凭常识补造用户、时间、数值、事件或诊断结果。
3. 遇到“今天、昨天、上周、最近几天、本月”等相对日期，必须先调用 get_current_time。
4. 查询前按需读取 /skills/ 下的数据库查询 Skill；不确定真实字段时调用 describe_data_source。
5. 用气量必须先按用户、厂区、管路计算五分钟平均标况瞬时流量，再汇总企业并积分；不能跨厂区合并同号管路，也不能将原始分钟行直接乘 5/60。
6. 查询没有数据时说明实际日期和数据覆盖范围，绝不把自然日期偷偷替换成数据库最新日期。
7. 只有缺失条件会实质改变结果时才调用 ask_user。零行、无数据或安全限制不是信息不足。
8. 创建工单前先收集证据；create_work_order 会自动进入人工批准，不得声称未批准的工单已经创建。
9. 安防事件仅可查询。不得确认、开始处理或关闭安防事件，也不得通过 SQL 修改任何处置状态。
10. 用户要求报告时先调用 write_todos 建立计划，再读取报告生成 Skill、执行必要查询并调用 build_report_artifact。图表只能引用真实 query_id 和字段。
11. 清晰区分算法实测值、推导值、估算值和 LLM 解释；回答中标注单位和数据边界。
12. Skills 文件读取请显式使用 limit=1000，避免默认行数截断。
13. 不得向用户输出隐藏推理或 reasoning_content；执行进度只通过 Todo 和工具状态表达。
14. 查询工具返回 SQL_QUERY_ERROR 且 retry_allowed=true 时，根据错误类别重写整条 SQL，最多修正一次；
    retry_allowed=false 时不得继续调用查询工具，应说明查询未完成，不得捏造结果。
""".strip()


class ThreadedSqliteSaver(SqliteSaver):
    """让同步 SqliteSaver 同时满足 LangGraph 的异步流式 checkpoint 协议。

    官方 SqliteSaver 的异步方法会直接抛 NotImplementedError；这里把同步数据库操作
    放入工作线程执行。底层 saver 自带线程锁，且连接启用了 check_same_thread=False，
    因而同步接口、SSE 异步接口和删除操作可以共享同一份用户 checkpoint。
    """

    async def aget_tuple(self, config):
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(self, config, *, filter=None, before=None, limit=None):
        items = await asyncio.to_thread(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit))
        )
        for item in items:
            yield item

    async def aput(self, config, checkpoint, metadata, new_versions):
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id):
        await asyncio.to_thread(self.delete_thread, thread_id)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _safe_todos(value: Any) -> list[dict[str, str]]:
    """只向前端暴露 Todo 文本和三种受控状态。"""

    if not isinstance(value, list):
        return []
    items: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        status = str(item.get("status") or "pending")
        if content and status in {"pending", "in_progress", "completed"}:
            items.append({"content": content[:500], "status": status})
    return items


def _current_turn_messages(messages: Any) -> list[Any]:
    """只保留最后一条用户消息开始的内容，让每个回答只绑定本轮产物。"""

    if not isinstance(messages, list):
        return []
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[index:]
    return messages


def _artifact_from_tool_message(message: ToolMessage) -> ChatArtifact | None:
    artifact = message.artifact
    if not isinstance(artifact, dict) or artifact.get("type") not in {
        "query_result",
        "report",
        "work_order",
    }:
        return None
    artifact_type = str(artifact["type"])
    artifact_id = str(
        artifact.get("query_id")
        or artifact.get("report_id")
        or artifact.get("work_order_id")
        or uuid_from_payload(artifact)
    )
    return ChatArtifact(type=artifact_type, id=artifact_id, payload=artifact)


def _artifact_progress(artifact: ChatArtifact | None) -> dict[str, Any] | None:
    """生成执行轨迹摘要，完整数据只进入右侧可追溯产物区。"""

    if artifact is None:
        return None
    payload = artifact.payload
    if artifact.type == "query_result":
        return {
            "id": artifact.id,
            "source": payload.get("source"),
            "row_count": payload.get("row_count"),
            "truncated": payload.get("truncated"),
            "elapsed_ms": payload.get("elapsed_ms"),
        }
    if artifact.type == "report":
        return {
            "id": artifact.id,
            "dataset_count": len(payload.get("datasets") or []),
            "chart_count": len(payload.get("charts") or []),
        }
    return {"id": artifact.id, "status": payload.get("status")}


class ConversationAgentService:
    """持有按登录用户隔离、可跨进程恢复的对话与 HITL 检查点。"""

    def __init__(self, root: Path):
        self.root = root
        load_project_env(root / ".env")
        self.model_name = os.getenv("CHAT_LLM_MODEL") or os.getenv("LLM_MODEL", "deepseek-chat")
        self.base_url = (os.getenv("CHAT_LLM_BASE_URL") or os.getenv("LLM_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.api_key = (
            os.getenv("CHAT_LLM_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("LLM_API_KEY")
        )
        self.provider = (os.getenv("CHAT_LLM_PROVIDER") or ("deepseek" if "deepseek" in self.base_url else "openai-compatible")).lower()
        os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
        checkpoint_path = root / "database" / "user_data.db"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_connection = sqlite3.connect(
            str(checkpoint_path), check_same_thread=False, timeout=10
        )
        self._checkpoint_connection.execute("PRAGMA journal_mode=WAL")
        self._checkpoint_connection.execute("PRAGMA busy_timeout=10000")
        self._checkpointer = ThreadedSqliteSaver(self._checkpoint_connection)
        # 启动阶段显式建表，历史接口无需等到第一次模型调用后才具备 checkpoint 结构。
        self._checkpointer.setup()
        self._agent: Any | None = None
        self._agent_lock = threading.RLock()
        self._thread_locks: dict[str, threading.Lock] = {}
        self._thread_locks_guard = threading.Lock()

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model_name and self.base_url)

    @property
    def tracing_enabled(self) -> bool:
        flag = os.getenv("LANGSMITH_TRACING", "").strip().lower()
        return flag in {"1", "true", "yes", "on"} and bool(os.getenv("LANGSMITH_API_KEY"))

    def _thread_lock(self, thread_id: str) -> threading.Lock:
        with self._thread_locks_guard:
            return self._thread_locks.setdefault(thread_id, threading.Lock())

    @staticmethod
    def _scoped_thread_id(user_id: str, thread_id: str) -> str:
        """内部 checkpoint 键加入用户编号，阻断跨用户 thread_id 碰撞。"""

        return f"{user_id}:{thread_id}"

    def _build_model(self):
        if not self.configured:
            raise RuntimeError("对话 Agent 未配置 API Key，请设置 CHAT_LLM_API_KEY 或 DEEPSEEK_API_KEY。")
        common = {
            "model": self.model_name,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "temperature": 0.1,
            "timeout": 120,
            "max_retries": 2,
            "streaming": True,
        }
        if self.provider == "deepseek":
            return ChatDeepSeek(**common)
        # 非推理型 OpenAI 兼容模型可以走该分支；推理模型需另做兼容性验证。
        return ChatOpenAI(**common)

    def _get_agent(self):
        if self._agent is not None:
            return self._agent
        with self._agent_lock:
            if self._agent is not None:
                return self._agent
            model = self._build_model()
            profile = HarnessProfile(
                excluded_tools=frozenset({"write_file", "edit_file"}),
                general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
            )
            # 精确模型和 provider 两级注册，兼容预构建模型的 profile 查找逻辑。
            harness_provider = "deepseek" if self.provider == "deepseek" else "openai"
            register_harness_profile(harness_provider, profile)
            register_harness_profile(f"{harness_provider}:{self.model_name}", profile)

            skills_root = self.root / "conversation_agent" / "skills"
            backend = CompositeBackend(
                default=StateBackend(),
                routes={
                    "/skills/": FilesystemBackend(root_dir=skills_root, virtual_mode=True),
                },
            )
            permissions = [
                FilesystemPermission(operations=["read"], paths=["/skills/**"], mode="allow"),
                FilesystemPermission(operations=["read"], paths=["/**"], mode="deny"),
                FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
            ]
            self._agent = create_deep_agent(
                model=model,
                tools=build_agent_tools(self.root),
                system_prompt=SYSTEM_PROMPT,
                skills=["/skills/"],
                permissions=permissions,
                backend=backend,
                middleware=[build_query_error_middleware(), TodoListMiddleware()],
                interrupt_on={
                    "create_work_order": {
                        "allowed_decisions": ["approve", "edit", "reject"],
                        "description": "创建工单前请确认标题、优先级、说明和检查清单。",
                    }
                },
                checkpointer=self._checkpointer,
                subagents=[],
                name="gas-business-conversation-agent",
            )
        return self._agent

    def turn(self, user_id: str, thread_id: str, message: str) -> ChatTurnResponse:
        agent = self._get_agent()
        scoped_thread_id = self._scoped_thread_id(user_id, thread_id)
        config = {"configurable": {"thread_id": scoped_thread_id}}
        with self._thread_lock(scoped_thread_id):
            output = agent.invoke(
                {"messages": [{"role": "user", "content": message.strip()}]},
                config=config,
                version="v2",
            )
        return self._response(output)

    def resume(self, user_id: str, request: ChatResumeRequest) -> ChatTurnResponse:
        agent = self._get_agent()
        scoped_thread_id = self._scoped_thread_id(user_id, request.thread_id)
        config = {"configurable": {"thread_id": scoped_thread_id}}
        command = self._resume_command(request)
        with self._thread_lock(scoped_thread_id):
            output = agent.invoke(command, config=config, version="v2")
        return self._response(output)

    @staticmethod
    def _resume_command(request: ChatResumeRequest) -> Command:
        if request.kind == "clarification":
            if request.decision != "answer" or not request.message.strip():
                raise ValueError("信息补充必须提供 answer 和非空 message。")
            return Command(resume={"message": request.message.strip()})
        if request.decision == "approve":
            decision: dict[str, Any] = {"type": "approve"}
        elif request.decision == "reject":
            decision = {"type": "reject", "message": request.message.strip() or "用户取消创建工单。"}
        elif request.decision == "edit":
            if not request.edited_action:
                raise ValueError("修改工单时必须提供 edited_action。")
            edited_action = dict(request.edited_action)
            if "name" not in edited_action:
                edited_action = {"name": "create_work_order", "args": edited_action}
            decision = {"type": "edit", "edited_action": edited_action}
        else:
            raise ValueError("工单审批仅支持 approve、edit 或 reject。")
        return Command(resume={"decisions": [decision]})

    async def stream_turn(self, user_id: str, thread_id: str, message: str) -> AsyncIterator[dict[str, Any]]:
        """流式执行一轮新消息；事件只暴露安全化进度，不包含隐藏推理。"""

        async for event in self._stream_agent(
            user_id,
            thread_id,
            {"messages": [{"role": "user", "content": message.strip()}]},
        ):
            yield event

    async def stream_resume(self, user_id: str, request: ChatResumeRequest) -> AsyncIterator[dict[str, Any]]:
        """流式恢复补充信息或工单审批中断。"""

        async for event in self._stream_agent(user_id, request.thread_id, self._resume_command(request)):
            yield event

    async def _stream_agent(self, user_id: str, thread_id: str, agent_input: Any) -> AsyncIterator[dict[str, Any]]:
        agent = self._get_agent()
        scoped_thread_id = self._scoped_thread_id(user_id, thread_id)
        run_uuid = uuid.uuid4()
        config = {
            "configurable": {"thread_id": scoped_thread_id},
            "run_id": run_uuid,
            "run_name": "gas-business-conversation",
            "tags": ["gas-business-chat"],
        }
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        thread_lock = self._thread_lock(scoped_thread_id)
        run: Any | None = None
        runner: asyncio.Task[None] | None = None
        acquired = False

        yield {"event": "meta", "data": {"run_id": str(run_uuid), "thread_id": thread_id}}
        yield {"event": "status", "data": {"stage": "starting", "label": "正在启动分析"}}
        acquire_task = asyncio.create_task(asyncio.to_thread(thread_lock.acquire))
        try:
            # shield 防止客户端取消时后台线程稍后拿到锁却无人释放。
            try:
                await asyncio.shield(acquire_task)
            except asyncio.CancelledError:
                await acquire_task
                thread_lock.release()
                raise
            acquired = True
            run = await agent.astream_events(agent_input, config=config, version="v3")

            async def consume_messages() -> None:
                async for message_stream in run.messages:
                    # 根图中只有 model 节点的文本才属于助手输出；reasoning 投影不会被消费。
                    if message_stream.node != "model":
                        await message_stream.output
                        continue
                    message_id = message_stream.message_id or uuid.uuid4().hex
                    emitted = False
                    async for delta in message_stream.text:
                        if not delta:
                            continue
                        emitted = True
                        await queue.put(
                            {
                                "event": "answer_delta",
                                "data": {"message_id": message_id, "delta": str(delta)},
                            }
                        )
                    completed_message = await message_stream.output
                    if isinstance(completed_message, AIMessage) and completed_message.tool_calls:
                        if emitted:
                            await queue.put(
                                {"event": "answer_reset", "data": {"message_id": message_id}}
                            )
                        await queue.put(
                            {
                                "event": "status",
                                "data": {"stage": "planning", "label": "已规划下一步数据操作"},
                            }
                        )

            async def consume_values() -> None:
                seen_tool_calls: set[str] = set()
                seen_tool_messages: set[str] = set()
                tool_started_at: dict[str, float] = {}
                tool_names: dict[str, str] = {}
                previous_todos = ""
                async for state in run.values:
                    todos = _safe_todos(state.get("todos", [])) if isinstance(state, dict) else []
                    serialized_todos = json.dumps(todos, ensure_ascii=False, sort_keys=True)
                    if serialized_todos != previous_todos:
                        previous_todos = serialized_todos
                        await queue.put({"event": "todo", "data": {"items": todos}})

                    messages = _current_turn_messages(state.get("messages", [])) if isinstance(state, dict) else []
                    for message in messages:
                        if isinstance(message, AIMessage):
                            for tool_call in message.tool_calls:
                                call_id = str(tool_call.get("id") or uuid.uuid4().hex)
                                if call_id in seen_tool_calls:
                                    continue
                                seen_tool_calls.add(call_id)
                                name = str(tool_call.get("name") or "tool")
                                tool_names[call_id] = name
                                tool_started_at[call_id] = time.monotonic()
                                await queue.put(
                                    {
                                        "event": "tool_start",
                                        "data": {"tool_call_id": call_id, "name": name},
                                    }
                                )
                        elif isinstance(message, ToolMessage):
                            call_id = str(message.tool_call_id or message.id or uuid.uuid4().hex)
                            if call_id in seen_tool_messages:
                                continue
                            seen_tool_messages.add(call_id)
                            artifact = _artifact_from_tool_message(message)
                            elapsed_ms = None
                            if call_id in tool_started_at:
                                elapsed_ms = round((time.monotonic() - tool_started_at[call_id]) * 1000)
                            await queue.put(
                                {
                                    "event": "tool_end",
                                    "data": {
                                        "tool_call_id": call_id,
                                        "name": message.name or tool_names.get(call_id, "tool"),
                                        "status": getattr(message, "status", "success") or "success",
                                        "elapsed_ms": elapsed_ms,
                                        "result": _artifact_progress(artifact),
                                    },
                                }
                            )
                            if artifact is not None:
                                await queue.put(
                                    {
                                        "event": "artifact",
                                        "data": artifact.model_dump(mode="json"),
                                    }
                                )

            async def finish_run() -> None:
                try:
                    await asyncio.gather(consume_messages(), consume_values())
                    state = await run.output() or {}
                    interrupts = await run.interrupts()
                    response = self._response_from_state(state, list(interrupts))
                    if response.interrupt:
                        await queue.put({"event": "interrupt", "data": response.interrupt})
                    await queue.put({"event": "done", "data": response.model_dump(mode="json")})
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await queue.put(
                        {
                            "event": "error",
                            "data": {
                                "message": f"Agent 执行失败（{type(exc).__name__}），请稍后重试或检查服务端日志。"
                            },
                        }
                    )
                finally:
                    await queue.put(None)

            runner = asyncio.create_task(finish_run())
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
            await runner
        finally:
            if runner is not None and not runner.done():
                runner.cancel()
                await asyncio.gather(runner, return_exceptions=True)
            try:
                if run is not None:
                    await run.abort()
            finally:
                if acquired:
                    thread_lock.release()

    def delete_thread(self, user_id: str, thread_id: str) -> None:
        scoped_thread_id = self._scoped_thread_id(user_id, thread_id)
        with self._thread_lock(scoped_thread_id):
            self._checkpointer.delete_thread(scoped_thread_id)

    def close(self) -> None:
        """应用退出时关闭持久 checkpoint 连接。"""

        self._checkpoint_connection.close()

    def _response(self, output: Any) -> ChatTurnResponse:
        value = getattr(output, "value", output)
        state = value if isinstance(value, dict) else {}
        raw_interrupts = list(getattr(output, "interrupts", []) or state.get("__interrupt__", []) or [])
        return self._response_from_state(state, raw_interrupts)

    def _response_from_state(self, state: dict[str, Any], raw_interrupts: list[Any]) -> ChatTurnResponse:
        messages = state.get("messages", [])
        current_turn_messages = _current_turn_messages(messages)
        artifacts: list[ChatArtifact] = []
        seen: set[str] = set()
        for message in current_turn_messages:
            if not isinstance(message, ToolMessage):
                continue
            artifact = _artifact_from_tool_message(message)
            if artifact is None:
                continue
            if artifact.id in seen:
                continue
            seen.add(artifact.id)
            artifacts.append(artifact)

        todos = _safe_todos(state.get("todos", []))

        if raw_interrupts:
            return ChatTurnResponse(
                status="interrupted",
                message="",
                generator=f"deepagents:{self.provider}:{self.model_name}",
                artifacts=artifacts,
                interrupt=self._normalize_interrupt(raw_interrupts[0]),
                todos=todos,
            )
        answer = ""
        for message in reversed(current_turn_messages):
            if isinstance(message, AIMessage) and not message.tool_calls:
                answer = _content_text(message.content).strip()
                if answer:
                    break
        return ChatTurnResponse(
            status="completed",
            message=answer or "本轮处理已完成。",
            generator=f"deepagents:{self.provider}:{self.model_name}",
            artifacts=artifacts,
            todos=todos,
        )

    @staticmethod
    def _normalize_interrupt(raw: Any) -> dict[str, Any]:
        value = getattr(raw, "value", raw)
        if isinstance(value, dict) and value.get("kind") == "clarification":
            return value
        if isinstance(value, dict) and value.get("action_requests"):
            action = value["action_requests"][0]
            review = (value.get("review_configs") or [{}])[0]
            return {
                "kind": "work_order_approval",
                "action": action,
                "allowed_decisions": review.get("allowed_decisions", ["approve", "edit", "reject"]),
            }
        return {"kind": "clarification", "question": json.dumps(value, ensure_ascii=False, default=str)}


def uuid_from_payload(payload: dict[str, Any]) -> str:
    import hashlib

    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
