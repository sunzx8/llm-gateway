"""Atomic_Code_T2 - Ingest Task。

agent loop 风格：模型自主决定先查（ls/read_file/grep/vec_search/graph_search）
再写（write_file/edit_file/vec_write/graph_write），最后调 finish 退出。

对齐 T3 的关键行为由 SessionContext 统一驱动：
- FS 写入自动加 [session_time-turn | event_time] 前缀
- Vec 写入自动提取 occurred_at + 写入 ingest_time / ingest_turn
- Graph 写入自动写入 ingest_time / ingest_turn 到 properties + 计算 node/edge embedding
- Graph edge 具有 upsert 语义（同 source+target 旧边自动替换）
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import logger.logger as logger

from context_task.base_context_task import BaseContextTask
from storage.file_system_store import FileSystemStore
from storage.stores_base import GraphStoreBase, VectorStoreBase
from utils.memory_llm_interface import LLMInterface

from .agent_loop import run_agent_loop
from .prompt import INGEST_SYSTEM_PROMPT, INGEST_USER_TEMPLATE
from .tool import build_default_registry


def _format_conversation(messages: list[dict[str, Any]], start_index: int = 1) -> str:
    """把 messages 拼成可读的对话文本（仅 user / assistant）。

    Args:
        messages: 消息列表。
        start_index: 编号起始值。当 batch 切片时，传入全局偏移量（如第 2 个 batch
                     的 start_index = batch_size + 1），保证 LLM 看到的轮次编号反映
                     该片段在完整对话中的真实位置。
    """
    lines: list[str] = []
    for i, m in enumerate(messages, start=start_index):
        role = m.get("role", "?")
        if role not in ("user", "assistant"):
            continue
        content = m.get("content", "")
        if isinstance(content, list):
            # OpenAI 多模态格式：取 text 段拼起来
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            content = "\n".join(text_parts)
        lines.append(f"[{i}] {role}: {content}")
    return "\n".join(lines)


class IngestContextAtomicCodeT2Task(BaseContextTask):
    """Atomic_Code_T2 模式下的摄入任务（agent loop 形式）。"""

    task_name = "ingest_context_atomic_code_t2"

    def __init__(
        self,
        llm: LLMInterface,
        fs_store: FileSystemStore,
        vec_store: VectorStoreBase,
        graph_store: GraphStoreBase,
        *,
        token_budget: int | None = None,
        max_messages: int | None = None,
        max_turns: int = 15,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm)
        self.fs = fs_store
        self.vec = vec_store
        self.graph = graph_store
        # token_budget / max_messages 当前版本未使用（保留兼容工厂签名）
        self.token_budget = token_budget
        self.max_messages = max_messages
        self.max_turns = max_turns
        self.extra_kwargs = kwargs
        # 业务结果（agent loop 摘要 / 工具轨迹），由 run() 写入
        self.result_extra: dict[str, Any] = {}

    async def run(
        self,
        user_id: str = "default_user",
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """从 messages 中抽取信息，按 agent loop 写入记忆库。"""
        messages = messages or []
        if not messages:
            logger.warning(
                "[%s] run: messages 为空，跳过 (user_id=%s, session_id=%s)",
                self.task_name, user_id, session_id,
            )
            return

        # 1. 时间锚：优先 kwargs["session_time"]，否则系统时间
        session_time = kwargs.get("session_time") or datetime.now().strftime("%Y-%m-%d %H:%M:%S, %a")

        # 2. 拼对话文本（message_offset 让编号反映在完整对话中的真实位置）
        message_offset: int = kwargs.get("message_offset", 1)
        conversation = _format_conversation(messages, start_index=message_offset)

        # 3. 组装初始 user prompt
        user_prompt = INGEST_USER_TEMPLATE.format(
            current_time=session_time,
            memory_root=self.fs.base_path,
            conversation=conversation,
        )

        # 4. 装配工具集（传入 session_time + embedder，让 SessionContext 驱动自动注入）
        #    embedder 优先取 graph 的（T3 用它给 node/edge 算向量），fallback 到 vec 的
        embedder = getattr(self.graph, "embedder", None) or getattr(self.vec, "embedder", None)
        registry = build_default_registry(
            self.fs, self.vec, self.graph,
            session_time=session_time,
            embedder=embedder,
        )

        # 5. 跑 agent loop
        result = await run_agent_loop(
            llm=self.llm,
            system_prompt=INGEST_SYSTEM_PROMPT,
            initial_messages=[{"role": "user", "content": user_prompt}],
            tools=registry,
            max_turns=self.max_turns,
            label=f"{self.task_name}[{session_id}]",
        )

        # 6. 把 loop 结果挂到 self.result_extra 里
        self.result_extra["ingest_finish_summary"] = result.finish_summary
        self.result_extra["ingest_finish_reason"] = result.finish_reason
        self.result_extra["ingest_turns_used"] = result.turns_used
        self.result_extra["ingest_tool_calls"] = [
            {
                "turn": t.turn,
                "name": t.name,
                "is_error": t.is_error,
                "result_preview": t.result_content[:200],
            }
            for t in result.tool_traces
        ]

        logger.info(
            "[%s] ingest done (turns=%d, reason=%s, tool_calls=%d, summary=%.200s)",
            self.task_name, result.turns_used, result.finish_reason,
            len(result.tool_traces), result.finish_summary,
        )
