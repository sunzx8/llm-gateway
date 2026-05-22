"""
Consolidate Context Task — 完备封装 T2 演进流程。

将 ConsolidateT2Task 的核心 agent loop 逻辑和 ConsolidateScheduler 的上层调度逻辑
统一封装为可独立调用的 ContextTask。

使用方式（直接调用）：
    task = ConsolidateContextTask(llm, fs_store, vec_store, graph_store)
    result = await task.execute(
        user_id="user_001",
        session_id="session_001",
    )

使用方式（通过调度器）：
    scheduler = ConsolidateContextScheduler(task)
    # ingest 完成后通知
    result = await scheduler.notify_ingest(user_id, session_id, ingest_count, ingest_stats)
    # session 结束后通知
    result = await scheduler.notify_session_end(user_id, session_id)

设计原则：
- 继承 BaseContextTask，复用 llm_generate_with_stat / tool_with_stat / save_checkpoint
- 内置完整的 consolidate agent loop（多轮 LLM tool-calling）
- 内置 pre_execute 检查（记忆库过小时跳过）
- 内置 ConsolidateContextScheduler（阈值触发、cooldown、inflight 保护）
- 所有 LLM 调用和工具操作自动统计
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from context_task.base_context_task import BaseContextTask, TaskStats
from context_task.prompt.chinese_prompt import (
    EVOLVE_SYSTEM_PROMPT,
    EVOLVE_USER_TEMPLATE,
    EVOLVE_STRATEGY_SYSTEM_PROMPT,
    EVOLVE_STRATEGY_USER_TEMPLATE,
    PG_BACKEND_NOTICE,
)
from context_task.tool import (
    CONSOLIDATE_TOOLS,
    READ_REFERENCE_STRATEGY_TOOL,
    SUBMIT_STRATEGY_TOOL,
    _PG_CONSOLIDATE_WRITE_TOOL,
)
from datetime import datetime

if TYPE_CHECKING:
    from utils.memory_llm_interface import LLMInterface
    from storage.file_system_store import FileSystemStore
    from storage.stores_base import GraphStoreBase
    from storage.stores_base import VectorStoreBase
import logger.logger as logger

from storage.file_system_store import (
    CODEGEN_DIR,
    EVOLVE_STRATEGY_FILENAME,
    EVOLVE_STRATEGY_FILE_PATH,
    INDEX_FILE_PATH,
    INDEX_FILE_PATH_FALLBACK,
)


_CONTEXT_TASK_DIR = Path(__file__).resolve().parent

evolve_reference_file_path_list = [
    str(_CONTEXT_TASK_DIR / "prompt" / "reference" / "evolve" / "mem0_evolve_strategy.md"),
]


# ---------------------------------------------------------------------------
# 业务统计数据结构
# ---------------------------------------------------------------------------


@dataclass
class ConsolidateContextTaskStats(TaskStats):
    """ConsolidateContextTask 的任务级业务统计数据。

    继承 TaskStats 获得 LLM/工具通用统计字段，
    同时扩展 consolidate 任务特有的业务指标。
    采用 dataclass 形式，字段即指标，便于查看、维护和序列化。
    """

    # ── 任务结果状态 ──
    status: str = ""
    """任务执行状态，取值: 'success' | 'skipped'"""

    reason: str = ""
    """跳过原因（仅 status='skipped' 时有值）"""

    # ── 上下文标识 ──
    user_id: str = ""
    """用户 ID"""

    session_id: str = ""
    """当前会话 ID"""

    scope: str = ""
    """演进范围，如 'incremental'、'periodic'"""

    # ── 执行统计 ──
    operations_performed: int = 0
    """实际执行的演进操作数（每次非 finish 的工具调用计 1 次）"""

    tools_called: int = 0
    """agent loop 中工具调用总次数"""

    # ── 结果摘要 ──
    changes_summary: str = ""
    """本轮演进的变更摘要（来自 finish 工具或模型最终回复，截断至 500 字符）"""

    finish_result: str = ""
    """finish 工具返回的完整结果文本"""

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典，合并基类统计和业务统计。"""
        base = super().to_dict()
        base.update({
            "status": self.status,
            "reason": self.reason,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "scope": self.scope,
            "operations_performed": self.operations_performed,
            "tools_called": self.tools_called,
            "changes_summary": self.changes_summary,
            "finish_result": self.finish_result,
        })
        return base


# ---------------------------------------------------------------------------
# Prompts（从 prompt/chinese_prompt.py 导入）
# ---------------------------------------------------------------------------
# EVOLVE_SYSTEM_PROMPT 和 EVOLVE_USER_TEMPLATE 已移至
# agent_memory.context_task.prompt.chinese_prompt


# ---------------------------------------------------------------------------
# 以下为原 EVOLVE_SYSTEM_PROMPT 内容的占位注释（已删除，从 chinese_prompt 导入）
# ---------------------------------------------------------------------------


# 以下为原 EVOLVE_USER_TEMPLATE 内容的占位注释（已删除，从 chinese_prompt 导入）


# ---------------------------------------------------------------------------
# Tools（复用统一的工具定义）
# ---------------------------------------------------------------------------

# 基础完整工具集（从统一的工具模块导入）
CONSOLIDATE_TOOLS = CONSOLIDATE_TOOLS


# PG 后端专属的 consolidate 写工具
_PG_CONSOLIDATE_WRITE_TOOL = _PG_CONSOLIDATE_WRITE_TOOL


# ---------------------------------------------------------------------------
# ConsolidateContextTask
# ---------------------------------------------------------------------------


class ConsolidateContextTask(BaseContextTask):
    """完备封装 T2 演进流程的 ContextTask。

    整合了以下逻辑：
    1. **pre_execute 检查**：记忆库过小时跳过演进
    2. **状态采集**：读取 index.md、fs_tree、vec_stats、graph_stats 构建 user prompt
    3. **agent loop**：多轮 LLM tool-calling 循环，执行演进操作
    4. **git commit 兜底**：演进完成后自动 commit
    5. **统计**：所有 LLM 调用和工具操作自动记录到 TaskStats

    生命周期：
        execute(user_id, session_id) →
            pre_run: 检查记忆库大小
            run: 构建 prompt → agent loop → git commit
            post_run: 记录完成日志
    """

    task_name = "consolidate_context"

    # ── 配置（从全局 YAML 配置读取，构造参数可覆盖） ──
    _CFG_SECTION = "consolidate_context_task"

    MIN_ITEMS_FOR_EVOLUTION: int = 5
    """记忆库条目数低于此阈值时跳过演进"""

    MAX_TURNS: int = 20
    """agent loop 的最大轮次"""

    STRATEGY_DIR: str = CODEGEN_DIR
    """策略文件存储目录"""

    STRATEGY_FILENAME: str = EVOLVE_STRATEGY_FILENAME
    """策略文件名"""

    def __init__(
        self,
        llm: "LLMInterface",
        fs_store: "FileSystemStore",
        vec_store: "VectorStoreBase",
        graph_store: "GraphStoreBase",
        *,
        min_items: int,
        max_turns: int,
    ):
        """初始化 ConsolidateContextTask。

        配置优先级：构造参数 > 全局 YAML 配置 > 类属性默认值。

        Args:
            llm: LLM 接口实例。
            fs_store: 文件系统存储后端。
            vec_store: 向量数据库存储后端。
            graph_store: 图数据库存储后端。
            min_items: 覆盖 MIN_ITEMS_FOR_EVOLUTION。
            max_turns: 覆盖 MAX_TURNS。
        """
        super().__init__(llm)
        self.fs = fs_store
        self.vec = vec_store
        self.graph = graph_store

        # 配置优先级：构造参数 > 全局 YAML 配置 > 类属性默认值
        self.MIN_ITEMS_FOR_EVOLUTION = min_items
        self.MAX_TURNS = max_turns

    # ------------------------------------------------------------------
    # 生命周期实现
    # ------------------------------------------------------------------

    async def pre_run(self, **kwargs: Any) -> None:
        """前置处理：初始化业务统计。"""
        # 保留基类 execute() 中已设置的 start_time
        _start_time = getattr(self._stats, "start_time", 0.0) if hasattr(self, "_stats") else 0.0

        # 使用 ConsolidateContextTaskStats 替代基类的 TaskStats
        self._stats = ConsolidateContextTaskStats(task_name=self.task_name)
        self._stats.start_time = _start_time

    @property
    def _strategy_file_path(self) -> str:
        return f"{self.STRATEGY_DIR}/{self.STRATEGY_FILENAME}"

    async def run(self, user_id: str = "default_user", session_id: str = "", scope: str = "periodic", **kwargs: Any) -> None:
        """核心逻辑：策略生成 → 检查记忆库大小 → 构建 prompt → agent loop。

        将业务结果直接写入 self._stats（ConsolidateContextTaskStats 实例）。

        Args:
            user_id: 用户 ID（默认 "default_user"）。
            session_id: 当前 session ID（用于日志和 git commit）。
            scope: consolidation scope 标记（如 "incremental"、"periodic"）。
        """
        biz_stats: ConsolidateContextTaskStats = self._stats  # type: ignore[assignment]

        # 存储当前上下文信息（供 session_view 工具使用）
        self._current_user_id = user_id
        self._current_session_id = session_id

        logger.info(
            "ConsolidateContextTask: run start "
            "(user_id=%s, session_id=%s, scope=%s)",
            user_id, session_id, scope,
        )

        # Phase 0: 检查记忆库是否足够大
        if self._check_and_skip_if_small(biz_stats):
            return

        # Phase 0.5: 演进策略生成
        logger.info("ConsolidateContextTask: Phase 0.5 - 演进策略生成开始")
        strategy_content = await self._generate_evolve_strategy(user_id)
        logger.info(
            "ConsolidateContextTask: Phase 0.5 完成 (strategy_len=%d)",
            len(strategy_content),
        )

        # Phase 1: 构建 user prompt（注入策略）
        logger.info("ConsolidateContextTask: building user prompt...")
        user_prompt = self._build_user_prompt(evolve_strategy=strategy_content)
        logger.info(
            "ConsolidateContextTask: user prompt built (len=%d)",
            len(user_prompt),
        )

        # Phase 2: 执行 agent loop
        logger.info("ConsolidateContextTask: starting agent loop (max_turns=%d)", self.MAX_TURNS)
        loop_stats = await self._run_agent_loop(
            user_prompt=user_prompt,
            session_id=session_id,
        )

        # Phase 3: git commit 兆底
        logger.info("ConsolidateContextTask: fallback git commit...")
        await self._fallback_commit(session_id, scope)

        biz_stats.status = "success"
        biz_stats.user_id = user_id
        biz_stats.session_id = session_id
        biz_stats.scope = scope
        biz_stats.tools_called = loop_stats.tools_called
        biz_stats.operations_performed = loop_stats.operations_performed
        biz_stats.finish_result = loop_stats.finish_result
        biz_stats.changes_summary = loop_stats.changes_summary

        logger.info(
            "ConsolidateContextTask: run completed "
            "(ops=%d, tools_called=%d, finish_result_len=%d)",
            biz_stats.operations_performed,
            biz_stats.tools_called,
            len(biz_stats.finish_result),
        )

    async def post_run(self, *, session_id: str = "", **kwargs: Any) -> None:
        """后置处理：记录完成日志。"""
        biz_stats: ConsolidateContextTaskStats = self._stats  # type: ignore[assignment]
        logger.info(
            "ConsolidateContextTask: completed, "
            "ops=%d, tools=%d, summary=%s",
            biz_stats.operations_performed,
            biz_stats.tools_called,
            biz_stats.changes_summary[:200],
        )

    # ------------------------------------------------------------------
    # 记忆库大小检查
    # ------------------------------------------------------------------

    def _check_and_skip_if_small(self, biz_stats: ConsolidateContextTaskStats) -> bool:
        """检查记忆库是否足够大，过小则设置跳过状态。

        Returns:
            若跳过则返回 True，否则返回 False。
        """
        fs_files = self.fs.list_files() if self.fs else []
        vec_stats = self.vec.get_stats() if self.vec else {}
        graph_stats = self.graph.get_stats() if self.graph else {}

        total_items = (
            len(fs_files)
            + vec_stats.get("total_entries", 0)
            + graph_stats.get("total_nodes", 0)
        )

        if total_items < self.MIN_ITEMS_FOR_EVOLUTION:
            logger.info(
                "ConsolidateContextTask: memory too small (%d items, "
                "threshold=%d), skipping evolution",
                total_items, self.MIN_ITEMS_FOR_EVOLUTION,
            )
            biz_stats.status = "skipped"
            biz_stats.reason = (
                f"memory too small ({total_items} items, "
                f"threshold={self.MIN_ITEMS_FOR_EVOLUTION})"
            )
            return True
        return False

    # ------------------------------------------------------------------
    # 演进策略生成
    # ------------------------------------------------------------------

    async def _generate_evolve_strategy(self, user_id: str) -> str:
        """通过多轮 agent loop 生成/更新记忆演进策略。

        策略是记忆重组更新的思路，用于后续指导记忆的演进操作。
        使用 consolidate 的工具集（只读部分）进行记忆库状态探索。

        Args:
            user_id: 用户 ID。

        Returns:
            生成的策略文档内容，失败时返回空字符串。
        """
        try:
            # 1. 构建策略生成的 user prompt
            user_prompt = self._build_strategy_prompt(user_id)

            # 2. 获取工具列表（只读工具 + finish）
            tools = self._get_strategy_tools()

            # 3. 执行多轮 agent loop
            logger.info("ConsolidateContextTask: 策略生成 agent loop 开始 (max_turns=%d)", self.MAX_TURNS)
            messages_history: list[dict[str, Any]] = [
                {"role": "user", "content": user_prompt}
            ]

            strategy_result = ""
            submitted_strategy = ""  # 优先级1: submit_strategy 工具提交
            finish_result = ""       # 优先级2: finish 的 result 参数
            content_fallback = ""    # 优先级3: 模型直接输出的 content

            for turn in range(self.MAX_TURNS):
                logger.info(
                    "ConsolidateContextTask: 策略 agent loop turn %d/%d",
                    turn + 1, self.MAX_TURNS,
                )

                response = await self.llm_generate_with_stat(
                    EVOLVE_STRATEGY_SYSTEM_PROMPT,
                    messages_history,
                    tools=tools,
                    label=f"strategy_loop_turn_{turn + 1}",
                )

                if not response.tool_calls:
                    # 模型返回纯文本 → 视为最终策略（回退级别3）
                    if response.content:
                        content_fallback = response.content
                    elif response.reasoning_content:
                        # 推理模型可能将内容放在 reasoning_content 中
                        content_fallback = response.reasoning_content
                    break

                # 将模型回复加入历史
                messages_history.append(response.to_message())
                should_break = False

                for tc in response.tool_calls:
                    if tc.name == "submit_strategy":
                        # 优先级1: 通过 submit_strategy 工具提交策略
                        new_content = tc.arguments.get("content", "")
                        is_append = tc.arguments.get("append", False)
                        if new_content:
                            if is_append and submitted_strategy:
                                submitted_strategy += "\n" + new_content
                            else:
                                submitted_strategy = new_content
                            messages_history.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": f"✅ 策略文档已成功提交（当前共 {len(submitted_strategy)} 字符）。请调用 `finish` 工具结束任务。",
                            })
                            logger.info(
                                "ConsolidateContextTask: submit_strategy 收到内容 (chars=%d, append=%s)",
                                len(new_content), is_append,
                            )
                        else:
                            messages_history.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": "❌ 提交失败：content 参数为空。请在 content 参数中填写完整的策略文档内容后重新提交。",
                            })
                        continue

                    if tc.name == "finish":
                        finish_result = tc.arguments.get("result", "")
                        messages_history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "(strategy generation finished)",
                        })
                        should_break = True
                        continue

                    # 执行只读工具调用
                    result_str = await self._dispatch_strategy_tool(tc.name, tc.arguments)
                    messages_history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    })

                if should_break:
                    break
            else:
                logger.warning(
                    "ConsolidateContextTask: 策略 agent loop 达到最大轮次 %d",
                    self.MAX_TURNS,
                )

            # 三级回退：选择最终策略内容
            if submitted_strategy:
                strategy_result = submitted_strategy
                logger.info("ConsolidateContextTask: 使用 submit_strategy 提交的内容 (chars=%d)", len(strategy_result))
            elif finish_result:
                strategy_result = finish_result
                logger.info("ConsolidateContextTask: 回退到 finish.result (chars=%d)", len(strategy_result))
            elif content_fallback:
                strategy_result = content_fallback
                logger.info("ConsolidateContextTask: 回退到 content (chars=%d)", len(strategy_result))
            else:
                logger.warning("ConsolidateContextTask: 所有回退均为空，策略生成失败")

            # 4. 备份旧策略文件（带日期）并写入新策略
            if strategy_result:
                self._backup_old_strategy()
                self.fs.write_file(self._strategy_file_path, strategy_result)
                logger.info(
                    "ConsolidateContextTask: 策略生成成功 (path=%s, length=%d)",
                    self._strategy_file_path, len(strategy_result),
                )
            else:
                logger.warning("ConsolidateContextTask: 策略生成未产出有效内容")

            return strategy_result

        except Exception as e:
            logger.error("ConsolidateContextTask: 策略生成异常: %s", e)
            return ""

    def _backup_old_strategy(self) -> str:
        """备份旧策略文件为带日期的文件。

        Returns:
            备份文件路径，如果旧文件不存在则返回空字符串。
        """
        old_content = self.fs.read_file(self._strategy_file_path)
        if old_content.startswith("ERROR"):
            return ""

        # 生成带日期的备份文件名
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"evolve_strategy_{date_str}.md"
        backup_path = f"{self.STRATEGY_DIR}/{backup_filename}"

        self.fs.write_file(backup_path, old_content)
        logger.info(
            "ConsolidateContextTask: 旧策略已备份至 %s", backup_path,
        )
        return backup_path

    def _build_strategy_prompt(self, user_id: str) -> str:
        """构建策略生成的 user prompt。

        注入信息：当前历史记忆的入口、旧策略的存储文件、当前 consolidate 的工具列表。

        Args:
            user_id: 用户 ID。

        Returns:
            完整的 user prompt。
        """
        # 读取 index.md（记忆入口）
        index_path, index_content = self.fs.read_index()

        # 读取旧策略文件
        old_strategy_content = self.fs.read_file(self._strategy_file_path)
        if old_strategy_content.startswith("ERROR"):
            old_strategy_content = "(尚无旧策略文件 — 这是首次生成)"

        # 收集记忆库概况
        fs_files = self.fs.list_files()
        vec_stats = self.vec.get_stats()
        graph_stats = self.graph.get_stats()

        return EVOLVE_STRATEGY_USER_TEMPLATE.format(
            index_path=index_path,
            index_content=index_content,
            strategy_file_path=self._strategy_file_path,
            old_strategy_content=old_strategy_content,
            reference_strategy_file_path_list=json.dumps(evolve_reference_file_path_list),
            fs_file_count=len(fs_files),
            fs_tree=self.fs.tree(max_depth=3),
            vec_total=vec_stats.get("total_entries", 0),
            vec_collections_count=len(vec_stats.get("collections", {})),
            vec_collections=str(vec_stats.get("collections", {})),
            graph_nodes=graph_stats.get("total_nodes", 0),
            graph_edges=graph_stats.get("total_edges", 0),
            graph_stats=str(graph_stats),
        )

    def _get_strategy_tools(self) -> list[dict]:
        """获取策略生成阶段可用的工具列表。

        复用 consolidate 的完整工具集，但只保留只读工具 + finish。

        Returns:
            只读工具列表（含 finish）。
        """
        all_tools = self._get_tools()

        # 只保留只读工具（排除写入类工具）
        write_tool_names = {
            "fs_write", "fs_append", "fs_delete",
            "vec_add", "vec_delete", "vec_delete_collection", "vec_create_collection",
            "graph_add_node", "graph_add_edge", "graph_delete_node", "graph_delete_edge",
            "graph_cypher_write",
            "fs_execute_bash",
        }

        read_only_tools = [
            tool for tool in all_tools
            if tool.get("function", {}).get("name", "") not in write_tool_names
        ]

        # 替换 finish 工具为策略生成专用版本（描述和 required 更明确）
        strategy_finish_tool = {
            "type": "function",
            "function": {
                "name": "finish",
                "description": (
                    "输出最终的策略文档并结束策略生成。\n\n"
                    "**必须**在 `result` 参数中填写完整的策略文档（Markdown 格式）。\n"
                    "策略文档应包含：记忆库健康度评估、本轮演进重点、存储路由策略、"
                    "记忆分层规则、代码生成指导等内容。\n\n"
                    "示例调用：finish(result=\"## 记忆库健康度评估\\n...\")"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "result": {
                            "type": "string",
                            "description": "完整的策略文档内容（Markdown 格式），不可为空",
                        },
                    },
                    "required": ["result"],
                },
            },
        }

        # 用策略专用 finish 替换原始 finish
        read_only_tools = [
            strategy_finish_tool if tool.get("function", {}).get("name") == "finish" else tool
            for tool in read_only_tools
        ]

        # 加入 submit_strategy 工具（推荐模型使用此工具提交策略内容）
        read_only_tools.append(SUBMIT_STRATEGY_TOOL)

        # 加入参考策略读取工具（允许模型查阅已有的优秀策略方案）
        read_only_tools.append(READ_REFERENCE_STRATEGY_TOOL)

        return read_only_tools

    async def _dispatch_strategy_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> str:
        """分派策略生成阶段的工具调用（只读操作）。

        Args:
            tool_name: 工具名称。
            args: 工具参数。

        Returns:
            工具执行结果字符串。
        """
        try:
            result = await self.dispatch_tool(
                tool_name, args,
                user_id=getattr(self, "_current_user_id", ""),
                session_id=getattr(self, "_current_session_id", ""),
                allow_write=True,
            )
            return result
        except Exception as e:
            logger.error(
                "ConsolidateContextTask: 策略工具 %s 执行失败: %s",
                tool_name, e,
            )
            return f"ERROR: {e}"

    # ------------------------------------------------------------------
    # User Prompt 构建
    # ------------------------------------------------------------------

    def _get_tools(self) -> list[dict]:
        """动态构建工具列表，PG 后端时追加专属工具。"""
        extra_tools = []
        if hasattr(self.vec, "sql_query_read"):
            # 从 retrieve_context_task 导入 PG 后端工具
            from .retrieve_context_task import _PG_RETRIEVE_TOOLS
            extra_tools += _PG_RETRIEVE_TOOLS
            extra_tools.append(_PG_CONSOLIDATE_WRITE_TOOL)
        return CONSOLIDATE_TOOLS + extra_tools

    def _build_user_prompt(self, evolve_strategy: str = "") -> str:
        """构建 consolidate 的 user prompt。

        读取当前记忆库状态（index.md、fs_tree、fs 文件采样、vec_stats、graph_stats），
        注入演进策略，生成完整的 user prompt。

        Args:
            evolve_strategy: 演进策略文档内容（由 Phase 0.5 生成）。
        """
        fs_files = self.fs.list_files()
        vec_stats = self.vec.get_stats()
        graph_stats = self.graph.get_stats()

        # 采样前 5 个文件内容作为上下文
        sample_contents: list[str] = []
        for f in fs_files[:5]:
            content = self.fs.read_file(f)
            if not content.startswith("ERROR"):
                sample_contents.append(f"### {f}\n{content[:500]}")

        # 读取 index.md
        _, index_content = self.fs.read_index()

        # 策略内容（如果为空则提供默认提示）
        strategy_text = evolve_strategy if evolve_strategy else "(本轮未生成演进策略，请根据记忆库状态自行判断)"

        msg = EVOLVE_USER_TEMPLATE.format(
            index_content=index_content,
            fs_file_count=len(fs_files),
            fs_tree=self.fs.tree(max_depth=3),
            fs_sample_contents=(
                "\n\n".join(sample_contents) if sample_contents else "(no files)"
            ),
            vec_total=vec_stats.get("total_entries", 0),
            vec_collections_count=len(vec_stats.get("collections", {})),
            vec_collections=str(vec_stats.get("collections", {})),
            graph_nodes=graph_stats.get("total_nodes", 0),
            graph_edges=graph_stats.get("total_edges", 0),
            graph_stats=str(graph_stats),
            evolve_strategy=strategy_text,
        )
        # PG 后端：附加 SQL / Cypher 能力说明 + 写能力提示
        if hasattr(self.vec, "sql_query_read"):
            msg += "\n\n" + PG_BACKEND_NOTICE + (
                "\n\n**Consolidate-only**: you also have `graph_cypher_write` "
                "for batch graph rewrites (補全 stance edges, dedup nodes, etc.) — "
                "use it for C7/C8 bulk fixes when fs_append+vec_add isn't enough.\n"
            )
        return msg

    # ------------------------------------------------------------------
    # Agent Loop
    # ------------------------------------------------------------------

    async def _run_agent_loop(
        self,
        user_prompt: str,
        session_id: str,
    ) -> ConsolidateContextTaskStats:
        """执行多轮 LLM tool-calling agent loop。

        Args:
            user_prompt: 构建好的 user prompt。
            session_id: 当前 session ID（仅用于日志）。

        Returns:
            本次演进的业务统计数据。
        """
        system_prompt = EVOLVE_SYSTEM_PROMPT
        tools = self._get_tools()
        messages_history: list[dict[str, Any]] = [
            {"role": "user", "content": user_prompt}
        ]

        biz_stats = ConsolidateContextTaskStats(status="success")
        finish_result: str = ""

        for turn in range(self.MAX_TURNS):
            logger.info(
                "ConsolidateContextTask: agent loop turn %d/%d start",
                turn + 1, self.MAX_TURNS,
            )
            # 调用 LLM（自动统计）
            response = await self.llm_generate_with_stat(
                system_prompt,
                messages_history,
                tools=tools,
                label=f"consolidate_loop_turn_{turn + 1}",
            )

            if not response.tool_calls:
                # 模型返回纯文本 → 视为最终回答
                if response.content:
                    finish_result = response.content
                    biz_stats.changes_summary = response.content[:500]
                logger.info(
                    "ConsolidateContextTask: agent loop turn %d/%d "
                    "ended with text response (len=%d)",
                    turn + 1, self.MAX_TURNS,
                    len(response.content or ""),
                )
                break

            # 将模型回复加入历史
            messages_history.append(response.to_message())
            should_break = False

            logger.info(
                "ConsolidateContextTask: agent loop turn %d/%d "
                "got %d tool_calls",
                turn + 1, self.MAX_TURNS, len(response.tool_calls),
            )

            for tc in response.tool_calls:
                if tc.name == "finish":
                    finish_result = tc.arguments.get("result", "")
                    biz_stats.changes_summary = finish_result
                    messages_history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "(task finished)",
                    })
                    should_break = True
                    continue

                # 执行工具调用（自动统计）
                result_str = await self._execute_tool_with_stat(
                    tc.name, tc.arguments, biz_stats, session_id,
                )
                messages_history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

            if should_break:
                logger.info(
                    "ConsolidateContextTask: agent loop finished at turn %d/%d "
                    "(finish tool called)",
                    turn + 1, self.MAX_TURNS,
                )
                break
        else:
            logger.warning(
                "ConsolidateContextTask: reached max_turns=%d without finish",
                self.MAX_TURNS,
            )

        logger.info(
            "ConsolidateContextTask: agent loop summary "
            "(total_turns=%d, ops=%d, tools_called=%d, finish_result_len=%d)",
            self.MAX_TURNS if self.MAX_TURNS > 0 else 0,
            biz_stats.operations_performed,
            biz_stats.tools_called,
            len(finish_result),
        )
        biz_stats.finish_result = finish_result
        return biz_stats

    # ------------------------------------------------------------------
    # 工具执行
    # ------------------------------------------------------------------

    async def _execute_tool_with_stat(
        self,
        tool_name: str,
        args: dict[str, Any],
        biz_stats: ConsolidateContextTaskStats,
        session_id: str,
    ) -> str:
        """执行单个工具调用，自动记录统计。

        通过 BaseContextTask.dispatch_tool 统一分派。

        Args:
            tool_name: 工具名称。
            args: 工具参数。
            biz_stats: 业务统计数据实例。
            session_id: 当前 session ID。

        Returns:
            工具执行结果字符串。
        """
        try:
            result = await self.dispatch_tool(
                tool_name, args,
                user_id=getattr(self, "_current_user_id", ""),
                session_id=session_id,
                allow_write=True,
            )
            biz_stats.tools_called += 1
            biz_stats.operations_performed += 1
            return result
        except Exception as e:
            logger.error(
                "ConsolidateContextTask: tool %s failed: %s", tool_name, e,
            )
            return f"ERROR: {e}"

    # ------------------------------------------------------------------
    # Git Commit 兜底
    # ------------------------------------------------------------------

    async def _fallback_commit(self, session_id: str, scope: str) -> None:
        """演进完成后自动 git commit（兜底）。"""
        if not getattr(self.fs, "enable_git", False):
            return

        try:
            info = await self.tool_with_stat(
                "git_commit",
                self.fs.commit_all,
                f"auto-commit: consolidate (session={session_id or '-'}, scope={scope})",
                arguments_summary={
                    "message": f"consolidate (scope={scope})",
                    "session_id": session_id,
                },
            )
            if info.get("committed"):
                logger.info(
                    "ConsolidateContextTask: fallback commit -> %s",
                    info.get("hash", ""),
                )
        except Exception as e:
            logger.debug(
                "ConsolidateContextTask: fallback commit failed (non-critical): %s", e,
            )
