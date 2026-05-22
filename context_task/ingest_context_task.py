"""
IngestContextTask：完备封装 T2 摄入流程的 ContextTask。

整合了以下逻辑：
1. **streaming ingest**：按 token 阈值将 session messages 分批
2. **source session 归档**：每批 ingest 前将原始消息写入 source_sessions/
3. **agent loop**：多轮 LLM tool-calling 循环，提取信息并写入三后端
4. **去重**：基于 content hash 跳过重复内容
5. **git commit 兜底**：每批 ingest 后自动 commit
6. **统计**：所有 LLM 调用和工具操作自动记录到 TaskStats

生命周期：
    execute(session_id, messages, user_id) →
        pre_run: 初始化 session 状态
        run: streaming ingest（分批 → agent loop）
        post_run: 清理状态
"""

import asyncio
import hashlib
import json
import re
import time
import tiktoken
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from context_task.base_context_task import BaseContextTask, TaskStats
from context_task.tool import INGEST_TOOLS
from context_task.prompt.chinese_prompt import INGEST_SYSTEM_PROMPT, INGEST_USER_TEMPLATE
import logger.logger as logger
from storage.stores_base import GraphStoreBase,VectorStoreBase
from storage.file_system_store import FileSystemStore, INDEX_FILE_PATH, INDEX_FILE_PATH_FALLBACK
from utils.memory_llm_interface import LLMInterface


@dataclass
class IngestBatchStats:
    """单批次摄入的业务统计数据。

    对应 _run_agent_loop 返回的单批次统计，
    每次 _flush_buffer 产生一个实例。
    """

    # ── 任务结果状态 ──
    status: str = ""
    """批次执行状态，取值: 'success' | 'skipped'"""

    reason: str = ""
    """跳过原因（仅 status='skipped' 时有值）"""

    # ── 写入统计 ──
    facts_added: int = 0
    """本批次写入的事实数（fs_write/fs_append/vec_add 各计 1 次或按条目数计）"""

    graph_nodes_added: int = 0
    """本批次添加的图节点数"""

    tools_called: int = 0
    """本批次 agent loop 中工具调用总次数"""

    # ── 结果摘要 ──
    finish_result: str = ""
    """finish 工具返回的完整结果文本"""

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "status": self.status,
            "reason": self.reason,
            "facts_added": self.facts_added,
            "graph_nodes_added": self.graph_nodes_added,
            "tools_called": self.tools_called,
            "finish_result": self.finish_result,
        }



@dataclass
class IngestContextTaskStats(TaskStats):
    """IngestContextTask 的任务级业务统计数据。

    继承 TaskStats 获得 LLM/工具通用统计字段，
    同时扩展 ingest 任务特有的业务指标。
    采用 dataclass 形式，字段即指标，便于查看、维护和序列化。
    """

    # ── 任务结果状态 ──
    status: str = ""
    """任务执行状态，取值: 'success' | 'noop'"""

    reason: str = ""
    """noop 原因（仅 status='noop' 时有值）"""

    # ── 上下文标识 ──
    user_id: str = ""
    """用户 ID"""

    session_id: str = ""
    """当前会话 ID"""

    # ── 聚合统计 ──
    messages_count: int = 0
    """本次摄入处理的 message 数量"""

    total_facts: int = 0
    """所有批次写入的事实总数"""

    total_tools_called: int = 0
    """所有批次工具调用总次数（业务层面，区别于基类 tool_calls 的底层统计）"""

    # ── 批次详情 ──
    batch_results: list[IngestBatchStats] = field(default_factory=list)
    """各批次的业务统计数据列表"""

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典，合并基类统计和业务统计。"""
        base = super().to_dict()
        base.update({
            "status": self.status,
            "reason": self.reason,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "messages_count": self.messages_count,
            "total_facts": self.total_facts,
            "total_tools_called": self.total_tools_called,
            "batch_results": [b.to_dict() for b in self.batch_results],
        })
        return base


# ---------------------------------------------------------------------------
# IngestContextTask
# ---------------------------------------------------------------------------


class IngestContextTask(BaseContextTask):
    """完备封装 T2 摄入流程的 ContextTask。

    整合了以下逻辑：
    1. **streaming ingest**：按 token 阈值将 session messages 分批
    2. **source session 归档**：每批 ingest 前将原始消息写入 source_sessions/
    3. **agent loop**：多轮 LLM tool-calling 循环，提取信息并写入三后端
    4. **去重**：基于 content hash 跳过重复内容
    5. **git commit 兜底**：每批 ingest 后自动 commit
    6. **统计**：所有 LLM 调用和工具操作自动记录到 TaskStats

    生命周期：
        execute(session_id, messages, user_id) →
            pre_run: 初始化 session 状态
            run: streaming ingest（分批 → agent loop）
            post_run: 清理状态
    """

    task_name = "ingest_context"

    # ── 流式摄入配置（从全局 YAML 配置读取，构造参数可覆盖） ──
    _CFG_SECTION = "ingest_context_task"

    TOKEN_BUDGET_PER_INGEST: int = 2000
    """每次 ingest 的 token 阈值"""

    MAX_MESSAGES_PER_INGEST: int = 10
    """消息条数的硬性上限（防御极端情况）"""

    MAX_TURNS: int = 15
    """agent loop 的最大轮次"""

    def __init__(
        self,
        llm: "LLMInterface",
        fs_store: "FileSystemStore",
        vec_store: "VectorStoreBase",
        graph_store: "GraphStoreBase",
        *,
        token_budget: int,
        max_messages: int,
        max_turns: int,
    ):
        """初始化 IngestContextTask。

        配置优先级：构造参数 > 全局 YAML 配置 > 类属性默认值。

        Args:
            llm: LLM 接口实例。
            fs_store: 文件系统存储后端。
            vec_store: 向量数据库存储后端。
            graph_store: 图数据库存储后端。
            token_budget: 覆盖 TOKEN_BUDGET_PER_INGEST。
            max_messages: 覆盖 MAX_MESSAGES_PER_INGEST。
            max_turns: 覆盖 MAX_TURNS。
        """
        super().__init__(llm)
        self.fs = fs_store
        self.vec = vec_store
        self.graph = graph_store

        # 配置优先级：构造参数 > 全局 YAML 配置 > 类属性默认值
        self.TOKEN_BUDGET_PER_INGEST = token_budget
        self.MAX_MESSAGES_PER_INGEST = max_messages
        self.MAX_TURNS = max_turns

        # Session 级去重状态
        self._seen_hashes: set[str] = set()
        # Session 级已提取内容（用于 prompt 中的去重提示）
        self._session_extractions: list[str] = []
        # tiktoken 编码器缓存
        self._tiktoken_enc: tiktoken.Encoding | None = None

    # ------------------------------------------------------------------
    # 生命周期实现
    # ------------------------------------------------------------------

    async def pre_run(self, session_id: str = "", messages: list[dict[str, str]] = None, **kwargs: Any) -> None:
        """前置处理：校验参数，初始化业务统计。"""
        # 保留基类 execute() 中已设置的 start_time
        _start_time = getattr(self._stats, "start_time", 0.0) if hasattr(self, "_stats") else 0.0

        # 使用 IngestContextTaskStats 替代基类的 TaskStats，
        # 这样 llm_generate_with_stat / tool_with_stat 的统计会自动记录到同一实例
        self._stats = IngestContextTaskStats(task_name=self.task_name)
        self._stats.start_time = _start_time

        if not session_id:
            logger.warning("IngestContextTask: empty session_id")
        if not messages:
            logger.warning("IngestContextTask: empty messages list")

    async def run(self, session_id: str = "", messages: list[dict[str, str]] = None, user_id: str = "default_user", **kwargs: Any) -> None:
        """核心逻辑：streaming ingest。

        按 token 阈值将 messages 分批，每批执行：
        1. source session 归档
        2. agent loop（多轮 LLM tool-calling）
        3. git commit 兆底

        将业务结果直接写入 self._stats（IngestContextTaskStats 实例）。

        Args:
            session_id: 会话 ID。
            messages: 消息列表 [{"role": ..., "content": ...}]。
            user_id: 用户 ID（默认 "default_user"）。
        """
        messages = messages or []

        logger.info(
            "IngestContextTask: run start "
            "(session_id=%s, user_id=%s, messages=%d, "
            "token_budget=%d, max_messages=%d)",
            session_id, user_id, len(messages),
            self.TOKEN_BUDGET_PER_INGEST, self.MAX_MESSAGES_PER_INGEST,
        )

        # 将业务字段写入 self._stats（已在 pre_run 中初始化为 IngestContextTaskStats）
        biz_stats: IngestContextTaskStats = self._stats  # type: ignore[assignment]
        biz_stats.user_id = user_id
        biz_stats.session_id = session_id
        biz_stats.messages_count = len(messages) / 2

        if not messages:
            biz_stats.status = "noop"
            biz_stats.reason = "empty messages"
            return

        biz_stats.status = "success"

        # Streaming ingest：按 token 阈值分批
        buffer: list[dict[str, str]] = []
        buffer_tokens = 0
        batch_index = 0

        for i, msg in enumerate(messages):
            buffer.append(msg)
            buffer_tokens += self._count_tokens(msg)

            at_turn_boundary = self._is_turn_boundary(msg, messages, i)
            budget_reached = buffer_tokens >= self.TOKEN_BUDGET_PER_INGEST
            should_flush = at_turn_boundary and budget_reached

            # 硬性上限兜底
            if not should_flush and len(buffer) >= self.MAX_MESSAGES_PER_INGEST:
                should_flush = True

            if should_flush:
                batch_index += 1
                logger.info(
                    "IngestContextTask: batch #%d at msg %d/%d "
                    "(%d msgs, %d tokens)",
                    batch_index, i + 1, len(messages),
                    len(buffer), buffer_tokens,
                )
                batch_stats = await self._flush_buffer(
                    buffer, session_id, user_id, batch_index,
                )
                biz_stats.batch_results.append(batch_stats)
                biz_stats.total_facts += batch_stats.facts_added
                biz_stats.total_tools_called += batch_stats.tools_called
                buffer = []
                buffer_tokens = 0

        # 强制 flush 剩余 buffer
        if buffer:
            batch_index += 1
            logger.info(
                "IngestContextTask: force-flush remaining %d msgs (%d tokens) "
                "as batch #%d",
                len(buffer), buffer_tokens, batch_index,
            )
            batch_stats = await self._flush_buffer(
                buffer, session_id, user_id, batch_index,
            )
            biz_stats.batch_results.append(batch_stats)
            biz_stats.total_facts += batch_stats.facts_added
            biz_stats.total_tools_called += batch_stats.tools_called

        logger.info(
            "IngestContextTask: run completed "
            "(session_id=%s, batches=%d, total_facts=%d, "
            "total_tools_called=%d)",
            session_id, batch_index,
            biz_stats.total_facts, biz_stats.total_tools_called,
        )

    async def post_run(self, *, session_id: str = "", **kwargs: Any) -> None:
        """后置处理：记录完成日志。"""
        biz_stats: IngestContextTaskStats = self._stats  # type: ignore[assignment]
        logger.info(
            "IngestContextTask: session=%s completed, "
            "facts=%d, tools=%d",
            session_id,
            biz_stats.total_facts,
            biz_stats.total_tools_called,
        )

    # ------------------------------------------------------------------
    # 核心内部方法：flush buffer（对应 EventDispatcher._flush_ingest_buffer）
    # ------------------------------------------------------------------

    async def _flush_buffer(
        self,
        buffer: list[dict[str, str]],
        session_id: str,
        user_id: str,
        batch_index: int,
    ) -> IngestBatchStats:
        """Flush 一批消息：归档 → agent loop → git commit。

        对应 EventDispatcher._flush_ingest_buffer 的完整逻辑。

        Args:
            buffer: 待摄入的消息列表。
            session_id: 当前 session ID。
            user_id: 当前用户 ID。
            batch_index: 当前批次序号（仅用于日志）。

        Returns:
            本批次的业务统计数据。
        """
        # Phase 0: source session 归档（append-only）
        await self._archive_source_session(buffer, session_id, batch_index)

        # Phase 1: 格式化 buffer 内容
        content = self._format_buffer_as_content(buffer)

        # Phase 2: 去重检查
        content_hash = hashlib.md5(content.encode()).hexdigest()
        if content_hash in self._seen_hashes:
            logger.info(
                "IngestContextTask: skipping duplicate content (batch #%d)",
                batch_index,
            )
            return IngestBatchStats(status="skipped", reason="duplicate_content")
        self._seen_hashes.add(content_hash)

        # Phase 3: 构建 user prompt
        user_prompt = self._build_user_prompt(session_id, content)

        # Phase 4: 执行 agent loop
        batch_stats = await self._run_agent_loop(
            session_id=session_id,
            user_prompt=user_prompt,
            batch_index=batch_index,
        )

        # Phase 5: git commit 兜底
        await self._fallback_commit(session_id, batch_index)

        return batch_stats

    # ------------------------------------------------------------------
    # Source session 归档
    # ------------------------------------------------------------------

    async def _archive_source_session(
        self,
        buffer: list[dict[str, str]],
        session_id: str,
        batch_index: int,
    ) -> None:
        """将原始消息归档到 source_sessions/（append-only）。"""
        if not buffer:
            return
        try:
            result = await self.tool_with_stat(
                "fs_archive_source_session",
                self.fs.append_source_session_messages,
                session_id=session_id,
                messages=buffer,
                arguments_summary={
                    "session_id": session_id,
                    "message_count": len(buffer),
                },
            )
            logger.info(
                "IngestContextTask: source session archived (batch #%d): %s",
                batch_index, result,
            )
        except Exception as e:
            # 归档失败不阻塞 ingest
            logger.warning(
                "IngestContextTask: source session archive failed (batch #%d): %s",
                batch_index, e,
            )

    # ------------------------------------------------------------------
    # Agent Loop（对应 T2BaseTask.execute 中的多轮循环）
    # ------------------------------------------------------------------

    async def _run_agent_loop(
        self,
        session_id: str,
        user_prompt: str,
        batch_index: int,
    ) -> IngestBatchStats:
        """执行多轮 LLM tool-calling agent loop。

        Args:
            session_id: 当前 session ID。
            user_prompt: 构建好的 user prompt。
            batch_index: 当前批次序号（仅用于日志）。

        Returns:
            本批次的业务统计数据。
        """
        system_prompt = INGEST_SYSTEM_PROMPT
        tools = INGEST_TOOLS
        messages_history: list[dict[str, Any]] = [
            {"role": "user", "content": user_prompt}
        ]

        batch_stats = IngestBatchStats(status="success")
        finish_result: str = ""

        for turn in range(self.MAX_TURNS):
            logger.info(
                "IngestContextTask: batch #%d agent loop turn %d/%d start",
                batch_index, turn + 1, self.MAX_TURNS,
            )
            # 调用 LLM（自动统计）
            response = await self.llm_generate_with_stat(
                system_prompt,
                messages_history,
                tools=tools,
                label=f"ingest_loop_turn_{turn + 1}",
            )

            if not response.tool_calls:
                # 模型返回纯文本 → 视为最终回答
                if response.content:
                    finish_result = response.content
                    self._session_extractions.append(
                        f"[{session_id}] {response.content[:200]}"
                    )
                logger.info(
                    "IngestContextTask: batch #%d agent loop turn %d/%d "
                    "ended with text response (len=%d)",
                    batch_index, turn + 1, self.MAX_TURNS,
                    len(response.content or ""),
                )
                break

            # 将模型回复加入历史
            messages_history.append(response.to_message())
            should_break = False

            logger.info(
                "IngestContextTask: batch #%d agent loop turn %d/%d "
                "got %d tool_calls",
                batch_index, turn + 1, self.MAX_TURNS,
                len(response.tool_calls),
            )

            for tc in response.tool_calls:
                if tc.name == "finish":
                    finish_result = tc.arguments.get("result", "")
                    self._session_extractions.append(
                        f"[{session_id}] {finish_result[:200]}"
                    )
                    messages_history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "(task finished)",
                    })
                    should_break = True
                    continue

                # 执行工具调用（自动统计）
                result_str = await self._execute_tool_with_stat(
                    tc.name, tc.arguments, batch_stats, session_id,
                )
                messages_history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

            if should_break:
                logger.info(
                    "IngestContextTask: batch #%d agent loop finished at turn %d/%d "
                    "(finish tool called, result_len=%d)",
                    batch_index, turn + 1, self.MAX_TURNS,
                    len(finish_result),
                )
                break
        else:
            logger.warning(
                "IngestContextTask: reached max_turns=%d without finish (batch #%d)",
                self.MAX_TURNS, batch_index,
            )

        logger.info(
            "IngestContextTask: batch #%d agent loop summary "
            "(total_turns=%d, tools_called=%d, facts_added=%d, "
            "finish_result_len=%d)",
            batch_index,
            self.MAX_TURNS if self.MAX_TURNS > 0 else 0,
            batch_stats.tools_called,
            batch_stats.facts_added,
            len(finish_result),
        )
        batch_stats.finish_result = finish_result
        return batch_stats

    # ------------------------------------------------------------------
    # 工具执行（对应 IngestT2Task.execute_tool）
    # ------------------------------------------------------------------

    async def _execute_tool_with_stat(
        self,
        tool_name: str,
        args: dict[str, Any],
        batch_stats: IngestBatchStats,
        session_id: str,
    ) -> str:
        """执行单个工具调用，自动记录统计。

        通过 BaseContextTask.dispatch_tool 统一分派，
        并在调用后更新业务统计。

        Args:
            tool_name: 工具名称。
            args: 工具参数。
            batch_stats: 当前批次的业务统计数据实例。
            session_id: 当前 session ID.

        Returns:
            工具执行结果字符串。
        """
        try:
            result = await self.dispatch_tool(
                tool_name, args,
                session_id=session_id,
                allow_write=True,
            )
            batch_stats.tools_called += 1
            # 业务统计：写入类工具计数
            if tool_name in ("fs_write", "fs_append"):
                batch_stats.facts_added += 1
            elif tool_name == "vec_add":
                batch_stats.facts_added += len(args.get("texts", []))
            elif tool_name == "graph_add_node":
                batch_stats.graph_nodes_added += 1
            return result
        except Exception as e:
            logger.error(
                "IngestContextTask: tool %s failed: %s", tool_name, e,
            )
            return f"ERROR: {e}"

    # ------------------------------------------------------------------
    # User Prompt 构建（对应 IngestT2Task.build_user_prompt）
    # ------------------------------------------------------------------

    def _build_user_prompt(self, session_id: str, conversation_content: str) -> str:
        """构建 ingest 的 user prompt。

        读取当前记忆库状态（index.md、fs_tree、vec_collections、graph_stats），
        结合已提取内容和待处理对话，生成完整的 user prompt。
        """
        # 解析 buffer content 为消息列表并格式化
        messages = self._parse_buffer_content(conversation_content) or [
            {"role": "user", "content": conversation_content}
        ]
        conversation = self._format_conversation(messages)

        # 读取 index.md
        _, index_content = self.fs.read_index()

        return INGEST_USER_TEMPLATE.format(
            index_content=index_content,
            fs_tree=self.fs.tree(max_depth=3),
            vec_collections=str(self.vec.list_collections()),
            graph_stats=str(self.graph.get_stats()),
            previous_extractions=(
                "\n".join(self._session_extractions[-20:])
                if self._session_extractions else "(none)"
            ),
            session_id=session_id,
            conversation=conversation,
        )

    # ------------------------------------------------------------------
    # Git Commit 兜底
    # ------------------------------------------------------------------

    async def _fallback_commit(self, session_id: str, batch_index: int) -> None:
        """每批 ingest 后自动 git commit（兜底）。

        对应 EventDispatcher._flush_ingest_buffer 中的 Phase 3 逻辑。
        """
        if not getattr(self.fs, "enable_git", False):
            return

        try:
            info = await self.tool_with_stat(
                "git_commit",
                self.fs.commit_all,
                f"auto-commit: batch #{batch_index} (session={session_id or '-'})",
                arguments_summary={
                    "message": f"batch #{batch_index}",
                    "session_id": session_id,
                },
            )
            if info.get("committed"):
                logger.info(
                    "IngestContextTask: fallback commit #%d -> %s",
                    batch_index, info.get("hash", ""),
                )
        except Exception as e:
            logger.debug(
                "IngestContextTask: fallback commit failed (non-critical): %s", e,
            )

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _get_tiktoken_enc(self) -> tiktoken.Encoding:
        """获取 tiktoken 编码器（实例级缓存）。"""
        if self._tiktoken_enc is None:
            self._tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        return self._tiktoken_enc

    def _count_tokens(self, msg: dict[str, str]) -> int:
        """使用 tiktoken 精确计算消息的 token 数。"""
        enc = self._get_tiktoken_enc()
        role = msg.get("role", "")
        content = msg.get("content", "")
        return len(enc.encode(role)) + len(enc.encode(content)) + 4

    @staticmethod
    def _is_turn_boundary(
        msg: dict[str, str],
        messages: list[dict[str, str]],
        idx: int,
    ) -> bool:
        """判断当前消息是否在完整 turn 边界。

        边界定义：当前消息是 assistant/ai，下一条是 user/human。
        """
        role = msg.get("role", "").lower()

        # 最后一条消息总是边界
        if idx >= len(messages) - 1:
            return True

        next_role = messages[idx + 1].get("role", "").lower()

        # assistant 后面跟 user → 完整 turn 结束
        if role in ("assistant", "ai", "system") and next_role in ("user", "human"):
            return True

        # user→user 序列（没有 assistant 回复）→ 也视为边界
        if role in ("user", "human") and next_role in ("user", "human"):
            return True

        return False

    @staticmethod
    def _format_buffer_as_content(buffer: list[dict[str, str]]) -> str:
        """将消息 buffer 格式化为可读的对话文本。

        格式：[1] role: content
        """
        lines = []
        for i, msg in enumerate(buffer):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            tool_name = msg.get("tool_name", "")

            if role == "tool" and tool_name:
                lines.append(f"[{i+1}] tool({tool_name}): {content}")
            elif role == "assistant" and tool_calls:
                parts = []
                if content:
                    parts.append(content)
                for tc in tool_calls:
                    tc_name = tc.get("name", "unknown_tool")
                    tc_args = tc.get("arguments", "")
                    if isinstance(tc_args, str) and len(tc_args) > 500:
                        tc_args = tc_args[:500] + "..."
                    parts.append(f"[tool_call: {tc_name}({tc_args})]")
                lines.append(f"[{i+1}] {role}: {' '.join(parts) if parts else '(empty)'}")
            else:
                lines.append(f"[{i+1}] {role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _format_conversation(messages: list[dict[str, str]]) -> str:
        """将消息列表格式化为 [role]: content 形式。"""
        return "\n".join(
            f"[{m.get('role', 'user')}]: {m.get('content', '')}" for m in messages
        )

    @staticmethod
    def _parse_buffer_content(content: str) -> list[dict[str, str]]:
        """Reverse of EventDispatcher._format_buffer_as_content.

        Input format (produced by the dispatcher)::

            [1] user: ...
            [2] assistant: ...
        """
        if not content:
            return []
        pattern = re.compile(r"^\[\d+\]\s+([^:]+):\s?", re.MULTILINE)
        matches = list(pattern.finditer(content))
        if not matches:
            return []
        messages: list[dict[str, str]] = []
        for i, m in enumerate(matches):
            role = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            body = content[start:end].rstrip("\n")
            messages.append({"role": role, "content": body})
        return messages
