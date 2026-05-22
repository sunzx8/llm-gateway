"""
Generate Code Context Code T2 Task — 纯代码生成任务（不含演进）。

生成摄入代码 + 消费代码：
1. Phase 1: 摄入策略生成（多轮 agent loop）
2. Phase 2: 消费策略生成（多轮 agent loop，依赖摄入策略）
3. Phase 3: 摄入代码生成（LLM 生成 + 文件写入 + 快照备份）
4. Phase 4: 消费代码生成（LLM 生成 + 文件写入 + 快照备份）

不包含记忆演进逻辑，完全依赖上层调度决定何时触发。

生成的代码：
- 摄入代码写入固定目录：<user_memory_base>/.codegen/ingest_memory.py
- 消费代码写入固定目录：<user_memory_base>/.codegen/retrieve_memory.py
- 旧代码文件备份为快照：<user_memory_base>/.codegen/snapshots/<filename>_<timestamp>.py
- 调用方通过 importlib 动态加载子类，在进程内实例化并调用

使用方式：
    task = GenerateCodeContextCodeT2Task(llm, fs_store, vec_store, graph_store)
    result = await task.execute(
        user_id="user_001",
        session_id="session_001",
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from context_task.base_context_task import BaseContextTask, TaskStats
from context_task.utils.text_utils import truncate_with_hint

from context_task.codegen.loader import load_ingestor_class, load_consumer_class
from context_task.codegen.staging import CodegenStagingArea
from context_task.tool import (
    EVAL_CODE_TOOL,
    FINISH_TOOL,
    PYTHON_SYNTAX_CHECK_TOOL,
    READ_ONLY_TOOLS,
    READ_REFERENCE_STRATEGY_TOOL,
    SOURCE_CODE_TOOLS,
    SUBMIT_INGEST_CODE_TOOL,
    SUBMIT_RETRIEVE_CODE_TOOL,
    SUBMIT_STRATEGY_TOOL,
)
from context_task.prompt.chinese_prompt import (
    CODEGEN_INGEST_SYSTEM_PROMPT,
    CODEGEN_INGEST_USER_TEMPLATE,
    CODEGEN_RETRIEVE_SYSTEM_PROMPT,
    CODEGEN_RETRIEVE_USER_TEMPLATE,
    CODEGEN_INGEST_STRATEGY_SYSTEM_PROMPT,
    CODEGEN_INGEST_STRATEGY_USER_TEMPLATE,
    CODEGEN_RETRIEVE_STRATEGY_SYSTEM_PROMPT,
    CODEGEN_RETRIEVE_STRATEGY_USER_TEMPLATE,
)

if TYPE_CHECKING:
    from utils.memory_llm_interface import LLMInterface
    from storage.file_system_store import FileSystemStore
    from storage.stores_base import GraphStoreBase
    from storage.stores_base import VectorStoreBase
import logger.logger as logger

from storage.file_system_store import (
    CODEGEN_DIR,
    EVOLVE_STRATEGY_FILE_PATH as STRATEGY_FILE_PATH,
    INGEST_CODEGEN_FILENAME,
    RETRIEVE_CODEGEN_FILENAME,
    CODEGEN_INGEST_STRATEGY_FILENAME,
    CODEGEN_INGEST_STRATEGY_FILE_PATH,
    CODEGEN_RETRIEVE_STRATEGY_FILENAME,
    CODEGEN_RETRIEVE_STRATEGY_FILE_PATH,
)


_CONTEXT_TASK_DIR = Path(__file__).resolve().parent

ingest_reference_file_path_list = [
    str(_CONTEXT_TASK_DIR / "prompt" / "reference" / "ingest" / "mem0_ingest_strategy.md"),
]

retrieve_reference_file_path_list = [
    str(_CONTEXT_TASK_DIR / "prompt" / "reference" / "retrieve" / "mem0_retrieve_strategy.md"),
    str(_CONTEXT_TASK_DIR / "prompt" / "reference" / "retrieve" / "t3_retrieve_strategy.md"),
]

# ---------------------------------------------------------------------------
# 业务统计数据结构
# ---------------------------------------------------------------------------


@dataclass
class GenerateCodeContextCodeT2TaskStats(TaskStats):
    """GenerateCodeContextCodeT2Task 的任务级业务统计数据。

    继承 TaskStats 获得 LLM/工具通用统计字段，
    同时扩展代码生成任务特有的业务指标。
    """

    # ── 任务结果状态 ──
    status: str = ""
    """任务执行状态，取值: 'success' | 'skipped' | 'partial'"""

    reason: str = ""
    """跳过或部分完成的原因"""

    # ── 上下文标识 ──
    user_id: str = ""
    """用户 ID"""

    session_id: str = ""
    """当前会话 ID"""

    scope: str = ""
    """演进范围"""

    # ── Phase 1: 演进统计 ──
    evolve_status: str = ""
    """演进阶段状态"""

    evolve_operations: int = 0
    """演进阶段执行的操作数"""

    evolve_summary: str = ""
    """演进阶段的变更摘要"""

    # ── Phase 2a: 摄入策略生成统计 ──
    ingest_strategy_status: str = ""
    """摄入策略生成阶段状态"""

    ingest_strategy_summary: str = ""
    """摄入策略摘要（截断至 500 字符）"""

    # ── Phase 2b: 消费策略生成统计 ──
    retrieve_strategy_status: str = ""
    """消费策略生成阶段状态"""

    retrieve_strategy_summary: str = ""
    """消费策略摘要（截断至 500 字符）"""

    # ── Phase 3: 摄入代码生成统计 ──
    ingest_codegen_status: str = ""
    """摄入代码生成阶段状态"""

    ingest_codegen_path: str = ""
    """生成的摄入代码文件路径"""

    ingest_codegen_snapshot_path: str = ""
    """旧摄入代码的快照文件路径"""

    ingest_codegen_length: int = 0
    """生成的摄入代码长度"""

    # ── Phase 4: 消费代码生成统计 ──
    retrieve_codegen_status: str = ""
    """消费代码生成阶段状态"""

    retrieve_codegen_path: str = ""
    """生成的消费代码文件路径"""

    retrieve_codegen_snapshot_path: str = ""
    """旧消费代码的快照文件路径"""

    retrieve_codegen_length: int = 0
    """生成的消费代码长度"""

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典，合并基类统计和业务统计。"""
        base = super().to_dict()
        base.update({
            "status": self.status,
            "reason": self.reason,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "scope": self.scope,
            "evolve_status": self.evolve_status,
            "evolve_operations": self.evolve_operations,
            "evolve_summary": self.evolve_summary,
            "ingest_strategy_status": self.ingest_strategy_status,
            "ingest_strategy_summary": self.ingest_strategy_summary,
            "retrieve_strategy_status": self.retrieve_strategy_status,
            "retrieve_strategy_summary": self.retrieve_strategy_summary,
            "ingest_codegen_status": self.ingest_codegen_status,
            "ingest_codegen_path": self.ingest_codegen_path,
            "ingest_codegen_snapshot_path": self.ingest_codegen_snapshot_path,
            "ingest_codegen_length": self.ingest_codegen_length,
            "retrieve_codegen_status": self.retrieve_codegen_status,
            "retrieve_codegen_path": self.retrieve_codegen_path,
            "retrieve_codegen_snapshot_path": self.retrieve_codegen_snapshot_path,
            "retrieve_codegen_length": self.retrieve_codegen_length,
        })
        return base


# ---------------------------------------------------------------------------
# GenerateCodeContextCodeT2Task
# ---------------------------------------------------------------------------


class GenerateCodeContextCodeT2Task(BaseContextTask):
    """纯代码生成任务（不含演进）。

    整合四个阶段：
    1. **Phase 1 - 摄入策略生成**：通过多轮 agent loop 生成摄入代码生成策略
    2. **Phase 2 - 消费策略生成**：通过多轮 agent loop 生成消费代码生成策略（依赖摄入策略）
    3. **Phase 3 - 摄入代码生成**：调用 LLM 根据记忆库状态生成 Python 摄入脚本
    4. **Phase 4 - 消费代码生成**：调用 LLM 根据记忆库状态和摄入代码生成 Python 消费脚本

    生命周期：
        execute(user_id, session_id) →
            pre_run: 初始化统计
            run:
                Phase 1: 摄入策略生成（多轮 agent loop）
                Phase 2: 消费策略生成（多轮 agent loop，依赖摄入策略）
                Phase 3: 摄入代码生成（LLM 生成 + 文件写入 + 快照备份）
                Phase 4: 消费代码生成（LLM 生成 + 文件写入 + 快照备份）
            post_run: 记录完成日志
    """

    task_name = "generate_code_context_code_t2"

    # ── 配置 ──
    MIN_ITEMS_FOR_EVOLUTION: int = 5
    """记忆库条目数低于此阈值时跳过演进"""

    MAX_TURNS: int = 20
    """agent loop 的最大轮次"""

    def __init__(
        self,
        llm: "LLMInterface",
        fs_store: "FileSystemStore",
        vec_store: "VectorStoreBase",
        graph_store: "GraphStoreBase",
        *,
        min_items: int = 5,
        max_turns: int = 20,
    ):
        """初始化 GenerateCodeContextCodeT2Task。

        Args:
            llm: LLM 接口实例。
            fs_store: 文件系统存储后端。
            vec_store: 向量数据库存储后端。
            graph_store: 图数据库存储后端。
            min_items: 预留参数（兼容工厂方法调用）。
            max_turns: 覆盖 MAX_TURNS。
        """
        super().__init__(llm)
        self.fs = fs_store
        self.vec = vec_store
        self.graph = graph_store

        self.MIN_ITEMS_FOR_EVOLUTION = min_items
        self.MAX_TURNS = max_turns

    # ------------------------------------------------------------------
    # 生命周期实现
    # ------------------------------------------------------------------

    async def pre_run(self, **kwargs: Any) -> None:
        """前置处理：初始化业务统计。"""
        _start_time = getattr(self._stats, "start_time", 0.0) if hasattr(self, "_stats") else 0.0
        self._stats = GenerateCodeContextCodeT2TaskStats(task_name=self.task_name)
        self._stats.start_time = _start_time

    async def run(
        self,
        user_id: str = "default_user",
        session_id: str = "",
        scope: str = "periodic",
        **kwargs: Any,
    ) -> None:
        """核心逻辑：4 个 Phase 顺序执行（纯代码生成，不含演进）。

        使用暂存区保证原子性：所有代码文件先写入暂存区，全部完成后一次性提交。

        Args:
            user_id: 用户 ID。
            session_id: 当前 session ID。
            scope: consolidation scope 标记。
        """
        biz_stats: GenerateCodeContextCodeT2TaskStats = self._stats  # type: ignore[assignment]
        biz_stats.user_id = user_id
        biz_stats.session_id = session_id
        biz_stats.scope = scope

        logger.info(
            "GenerateCodeContextCodeT2Task: run start "
            "(user_id=%s, session_id=%s, scope=%s)",
            user_id, session_id, scope,
        )

        # 创建暂存区，保证代码修改的原子性
        self._staging = CodegenStagingArea(self.fs)

        # ── Phase 1: 摄入策略生成 ──
        logger.info("GenerateCodeContextCodeT2Task: Phase 1 - 摄入策略生成开始")
        ingest_strategy = await self._generate_ingest_strategy(user_id, session_id, biz_stats)
        logger.info(
            "GenerateCodeContextCodeT2Task: Phase 1 完成 (status=%s)",
            biz_stats.ingest_strategy_status,
        )

        # ── Phase 2: 消费策略生成（依赖摄入策略） ──
        logger.info("GenerateCodeContextCodeT2Task: Phase 2 - 消费策略生成开始")
        retrieve_strategy = await self._generate_retrieve_strategy(user_id, session_id, biz_stats, ingest_strategy)
        logger.info(
            "GenerateCodeContextCodeT2Task: Phase 2 完成 (status=%s)",
            biz_stats.retrieve_strategy_status,
        )

        # 合并策略供后续代码生成使用
        codegen_strategy = ""
        if ingest_strategy:
            codegen_strategy += f"## 摄入策略\n\n{ingest_strategy}\n\n"
        if retrieve_strategy:
            codegen_strategy += f"## 消费策略\n\n{retrieve_strategy}\n\n"

        # ── Phase 3: 摄入代码生成 ──
        logger.info("GenerateCodeContextCodeT2Task: Phase 3 - 摄入代码生成开始")
        ingest_code = await self._generate_ingest_code(user_id, session_id, biz_stats, codegen_strategy)
        logger.info(
            "GenerateCodeContextCodeT2Task: Phase 3 完成 (status=%s)",
            biz_stats.ingest_codegen_status,
        )

        # ── Phase 4: 消费代码生成 ──
        logger.info("GenerateCodeContextCodeT2Task: Phase 4 - 消费代码生成开始")
        await self._generate_retrieve_code(user_id, session_id, biz_stats, ingest_code, codegen_strategy)
        logger.info(
            "GenerateCodeContextCodeT2Task: Phase 4 完成 (status=%s)",
            biz_stats.retrieve_codegen_status,
        )

        # 最终状态
        all_success = (
            biz_stats.ingest_strategy_status in ("success", "skipped")
            and biz_stats.retrieve_strategy_status in ("success", "skipped")
            and biz_stats.ingest_codegen_status == "success"
            and biz_stats.retrieve_codegen_status == "success"
        )
        if all_success:
            biz_stats.status = "success"
        else:
            biz_stats.status = "partial"
            failed_phases = []
            if biz_stats.ingest_strategy_status not in ("success", "skipped"):
                failed_phases.append(f"ingest_strategy: {biz_stats.ingest_strategy_status}")
            if biz_stats.retrieve_strategy_status not in ("success", "skipped"):
                failed_phases.append(f"retrieve_strategy: {biz_stats.retrieve_strategy_status}")
            if biz_stats.ingest_codegen_status != "success":
                failed_phases.append(f"ingest_codegen: {biz_stats.ingest_codegen_status}")
            if biz_stats.retrieve_codegen_status != "success":
                failed_phases.append(f"retrieve_codegen: {biz_stats.retrieve_codegen_status}")
            biz_stats.reason = ", ".join(failed_phases)

        # ── 原子性提交或丢弃暂存区 ──
        if biz_stats.status == "success":
            commit_result = self._staging.commit()
            if not commit_result["committed"]:
                logger.warning(
                    "GenerateCodeContextCodeT2Task: 暂存区提交失败: %s",
                    commit_result["error"],
                )
                biz_stats.status = "partial"
                biz_stats.reason = f"暂存区提交失败: {commit_result['error']}"
            else:
                logger.info(
                    "GenerateCodeContextCodeT2Task: 代码已原子性提交 "
                    "(backed_up=%d, deployed=%d)",
                    len(commit_result["backed_up"]),
                    len(commit_result["deployed"]),
                )
        else:
            self._staging.discard()
            logger.info(
                "GenerateCodeContextCodeT2Task: 任务未完全成功，暂存区已丢弃"
            )

        logger.info(
            "GenerateCodeContextCodeT2Task: run 完成 (status=%s)",
            biz_stats.status,
        )

    async def post_run(self, *, session_id: str = "", **kwargs: Any) -> None:
        """后置处理：记录完成日志。"""
        biz_stats: GenerateCodeContextCodeT2TaskStats = self._stats  # type: ignore[assignment]
        logger.info(
            "GenerateCodeContextCodeT2Task: completed, "
            "status=%s, ingest_codegen_len=%d, retrieve_codegen_len=%d",
            biz_stats.status,
            biz_stats.ingest_codegen_length,
            biz_stats.retrieve_codegen_length,
        )

    # ------------------------------------------------------------------
    # Phase 2a: 摄入策略生成
    # ------------------------------------------------------------------

    async def _generate_ingest_strategy(
        self,
        user_id: str,
        session_id: str,
        biz_stats: GenerateCodeContextCodeT2TaskStats,
    ) -> str:
        """通过多轮 agent loop 生成/更新记忆摄入的代码生成策略。

        策略用于指导后续的摄入代码生成。

        Args:
            user_id: 用户 ID。
            session_id: 当前 session ID。
            biz_stats: 业务统计实例。

        Returns:
            生成的摄入策略文档内容，失败时返回空字符串。
        """
        try:
            # 1. 构建策略生成的 user prompt
            user_prompt = self._build_ingest_strategy_prompt(user_id)

            # 2. 获取工具列表（只读工具 + session 工具 + finish）
            tools = self._get_strategy_tools("ingest")

            # 3. 执行多轮 agent loop
            logger.info(
                "GenerateCodeContextCodeT2Task: 摄入策略 agent loop 开始 (max_turns=%d)",
                self.MAX_TURNS,
            )
            messages_history: list[dict[str, Any]] = [
                {"role": "user", "content": user_prompt}
            ]

            strategy_result = ""
            submitted_strategy = ""  # 优先级1: submit_strategy 工具提交
            finish_result = ""       # 优先级2: finish 的 result 参数
            content_fallback = ""    # 优先级3: 模型直接输出的 content

            for turn in range(self.MAX_TURNS):
                logger.info(
                    "GenerateCodeContextCodeT2Task: 摄入策略 agent loop turn %d/%d",
                    turn + 1, self.MAX_TURNS,
                )

                response = await self.llm_generate_with_stat(
                    CODEGEN_INGEST_STRATEGY_SYSTEM_PROMPT,
                    messages_history,
                    tools=tools,
                    label=f"ingest_strategy_loop_turn_{turn + 1}",
                )

                if not response.tool_calls:
                    if response.content:
                        content_fallback = response.content
                    elif response.reasoning_content:
                        content_fallback = response.reasoning_content
                    break

                messages_history.append(response.to_message())
                should_break = False

                for tc in response.tool_calls:
                    if tc.name == "submit_strategy":
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
                            "content": "(ingest strategy generation finished)",
                        })
                        should_break = True
                        continue

                    result_str = await self.dispatch_tool(
                        tc.name, tc.arguments,
                        user_id=user_id,
                        session_id=session_id,
                        allow_write=True,
                    )
                    messages_history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    })

                if should_break:
                    break
            else:
                logger.warning(
                    "GenerateCodeContextCodeT2Task: 摄入策略 agent loop 达到最大轮次 %d",
                    self.MAX_TURNS,
                )

            # 三级回退：选择最终策略内容
            if submitted_strategy:
                strategy_result = submitted_strategy
                logger.info("GenerateCodeContextCodeT2Task: 摄入策略使用 submit_strategy 提交的内容 (chars=%d)", len(strategy_result))
            elif finish_result:
                strategy_result = finish_result
                logger.info("GenerateCodeContextCodeT2Task: 摄入策略回退到 finish.result (chars=%d)", len(strategy_result))
            elif content_fallback:
                strategy_result = content_fallback
                logger.info("GenerateCodeContextCodeT2Task: 摄入策略回退到 content (chars=%d)", len(strategy_result))
            else:
                logger.warning("GenerateCodeContextCodeT2Task: 摄入策略所有回退均为空，策略生成失败")

            # 4. 备份旧策略文件并写入新策略
            if strategy_result:
                self._staging.write_file(CODEGEN_INGEST_STRATEGY_FILENAME, strategy_result)
                biz_stats.ingest_strategy_status = "success"
                biz_stats.ingest_strategy_summary = strategy_result[:500]
                logger.info(
                    "GenerateCodeContextCodeT2Task: 摄入策略生成成功 (path=%s, length=%d)",
                    CODEGEN_INGEST_STRATEGY_FILE_PATH, len(strategy_result),
                )
            else:
                biz_stats.ingest_strategy_status = "skipped"
                biz_stats.ingest_strategy_summary = "未生成有效摄入策略"
                logger.warning("GenerateCodeContextCodeT2Task: 摄入策略生成未产出有效内容")

            return strategy_result

        except Exception as e:
            biz_stats.ingest_strategy_status = "error"
            biz_stats.ingest_strategy_summary = f"摄入策略生成异常: {e}"
            logger.error("GenerateCodeContextCodeT2Task: 摄入策略生成异常: %s", e)
            return ""

    # ------------------------------------------------------------------
    # Phase 2b: 消费策略生成
    # ------------------------------------------------------------------

    async def _generate_retrieve_strategy(
        self,
        user_id: str,
        session_id: str,
        biz_stats: GenerateCodeContextCodeT2TaskStats,
        ingest_strategy: str = "",
    ) -> str:
        """通过多轮 agent loop 生成/更新记忆消费的代码生成策略。

        消费策略依赖摄入策略，需要了解数据写入的格式和路由规则。

        Args:
            user_id: 用户 ID。
            session_id: 当前 session ID。
            biz_stats: 业务统计实例。
            ingest_strategy: 摄入策略文档内容（由 Phase 2a 生成）。

        Returns:
            生成的消费策略文档内容，失败时返回空字符串。
        """
        try:
            # 1. 构建策略生成的 user prompt
            user_prompt = self._build_retrieve_strategy_prompt(user_id, ingest_strategy)

            # 2. 获取工具列表（只读工具 + session 工具 + finish）
            tools = self._get_strategy_tools("retrieve")

            # 3. 执行多轮 agent loop
            logger.info(
                "GenerateCodeContextCodeT2Task: 消费策略 agent loop 开始 (max_turns=%d)",
                self.MAX_TURNS,
            )
            messages_history: list[dict[str, Any]] = [
                {"role": "user", "content": user_prompt}
            ]

            strategy_result = ""
            submitted_strategy = ""  # 优先级1: submit_strategy 工具提交
            finish_result = ""       # 优先级2: finish 的 result 参数
            content_fallback = ""    # 优先级3: 模型直接输出的 content

            for turn in range(self.MAX_TURNS):
                logger.info(
                    "GenerateCodeContextCodeT2Task: 消费策略 agent loop turn %d/%d",
                    turn + 1, self.MAX_TURNS,
                )

                response = await self.llm_generate_with_stat(
                    CODEGEN_RETRIEVE_STRATEGY_SYSTEM_PROMPT,
                    messages_history,
                    tools=tools,
                    label=f"retrieve_strategy_loop_turn_{turn + 1}",
                )

                if not response.tool_calls:
                    if response.content:
                        content_fallback = response.content
                    elif response.reasoning_content:
                        content_fallback = response.reasoning_content
                    break

                messages_history.append(response.to_message())
                should_break = False

                for tc in response.tool_calls:
                    if tc.name == "submit_strategy":
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
                            "content": "(retrieve strategy generation finished)",
                        })
                        should_break = True
                        continue

                    result_str = await self.dispatch_tool(
                        tc.name, tc.arguments,
                        user_id=user_id,
                        session_id=session_id,
                        allow_write=True,
                    )
                    messages_history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    })

                if should_break:
                    break
            else:
                logger.warning(
                    "GenerateCodeContextCodeT2Task: 消费策略 agent loop 达到最大轮次 %d",
                    self.MAX_TURNS,
                )

            # 三级回退：选择最终策略内容
            if submitted_strategy:
                strategy_result = submitted_strategy
                logger.info("GenerateCodeContextCodeT2Task: 消费策略使用 submit_strategy 提交的内容 (chars=%d)", len(strategy_result))
            elif finish_result:
                strategy_result = finish_result
                logger.info("GenerateCodeContextCodeT2Task: 消费策略回退到 finish.result (chars=%d)", len(strategy_result))
            elif content_fallback:
                strategy_result = content_fallback
                logger.info("GenerateCodeContextCodeT2Task: 消费策略回退到 content (chars=%d)", len(strategy_result))
            else:
                logger.warning("GenerateCodeContextCodeT2Task: 消费策略所有回退均为空，策略生成失败")

            # 4. 备份旧策略文件并写入新策略
            if strategy_result:
                self._staging.write_file(CODEGEN_RETRIEVE_STRATEGY_FILENAME, strategy_result)
                biz_stats.retrieve_strategy_status = "success"
                biz_stats.retrieve_strategy_summary = strategy_result[:500]
                logger.info(
                    "GenerateCodeContextCodeT2Task: 消费策略生成成功 (path=%s, length=%d)",
                    CODEGEN_RETRIEVE_STRATEGY_FILE_PATH, len(strategy_result),
                )
            else:
                biz_stats.retrieve_strategy_status = "skipped"
                biz_stats.retrieve_strategy_summary = "未生成有效消费策略"
                logger.warning("GenerateCodeContextCodeT2Task: 消费策略生成未产出有效内容")

            return strategy_result

        except Exception as e:
            biz_stats.retrieve_strategy_status = "error"
            biz_stats.retrieve_strategy_summary = f"消费策略生成异常: {e}"
            logger.error("GenerateCodeContextCodeT2Task: 消费策略生成异常: %s", e)
            return ""

    def _build_ingest_strategy_prompt(self, user_id: str) -> str:
        """构建摄入策略生成的 user prompt。

        Args:
            user_id: 用户 ID。

        Returns:
            完整的 user prompt。
        """
        memory_state = self._collect_memory_state(user_id)

        # 读取演进策略文件
        evolve_strategy_content = self.fs.read_file(STRATEGY_FILE_PATH)
        if evolve_strategy_content.startswith("ERROR"):
            evolve_strategy_content = "(演进策略文件不存在)"

        # 读取旧的摄入策略文件
        old_ingest_strategy_content = self.fs.read_file(CODEGEN_INGEST_STRATEGY_FILE_PATH)
        if old_ingest_strategy_content.startswith("ERROR"):
            old_ingest_strategy_content = "(尚无旧摄入策略文件 — 这是首次生成)"

        # 构建工具描述列表
        return CODEGEN_INGEST_STRATEGY_USER_TEMPLATE.format(
            index_path=memory_state["index_path"],
            index_content=memory_state["index_content"],
            evolve_strategy_file_path=STRATEGY_FILE_PATH,
            evolve_strategy_content=truncate_with_hint(evolve_strategy_content, 3000, STRATEGY_FILE_PATH),
            codegen_ingest_strategy_file_path=CODEGEN_INGEST_STRATEGY_FILE_PATH,
            old_codegen_ingest_strategy_content=truncate_with_hint(old_ingest_strategy_content, 3000, CODEGEN_INGEST_STRATEGY_FILE_PATH),
            reference_strategy_file_path_list=json.dumps(ingest_reference_file_path_list),
            fs_file_count=memory_state["fs_file_count"],
            fs_tree=memory_state["fs_tree"],
            fs_sample_contents=memory_state["fs_sample_contents"],
            vec_total=memory_state["vec_total"],
            vec_collections_count=memory_state["vec_collections_count"],
            vec_collections=memory_state["vec_collections"],
            graph_nodes=memory_state["graph_nodes"],
            graph_edges=memory_state["graph_edges"],
            graph_stats=memory_state["graph_stats"],
        )

    def _build_retrieve_strategy_prompt(self, user_id: str, ingest_strategy: str = "") -> str:
        """构建消费策略生成的 user prompt。

        Args:
            user_id: 用户 ID。
            ingest_strategy: 摄入策略文档内容。

        Returns:
            完整的 user prompt。
        """
        memory_state = self._collect_memory_state(user_id)

        # 读取演进策略文件
        evolve_strategy_content = self.fs.read_file(STRATEGY_FILE_PATH)
        if evolve_strategy_content.startswith("ERROR"):
            evolve_strategy_content = "(演进策略文件不存在)"

        # 摄入策略内容
        ingest_strategy_content = ingest_strategy if ingest_strategy else "(摄入策略不可用)"

        # 读取旧的消费策略文件
        old_retrieve_strategy_content = self.fs.read_file(CODEGEN_RETRIEVE_STRATEGY_FILE_PATH)
        if old_retrieve_strategy_content.startswith("ERROR"):
            old_retrieve_strategy_content = "(尚无旧消费策略文件 — 这是首次生成)"

        return CODEGEN_RETRIEVE_STRATEGY_USER_TEMPLATE.format(
            index_path=memory_state["index_path"],
            index_content=memory_state["index_content"],
            evolve_strategy_file_path=STRATEGY_FILE_PATH,
            evolve_strategy_content=truncate_with_hint(evolve_strategy_content, 3000, STRATEGY_FILE_PATH),
            ingest_strategy_content=truncate_with_hint(ingest_strategy_content, 3000, CODEGEN_INGEST_STRATEGY_FILE_PATH),
            codegen_retrieve_strategy_file_path=CODEGEN_RETRIEVE_STRATEGY_FILE_PATH,
            old_codegen_retrieve_strategy_content=truncate_with_hint(old_retrieve_strategy_content, 3000, CODEGEN_RETRIEVE_STRATEGY_FILE_PATH),
            reference_strategy_file_path_list=json.dumps(retrieve_reference_file_path_list),
            fs_file_count=memory_state["fs_file_count"],
            fs_tree=memory_state["fs_tree"],
            fs_sample_contents=memory_state["fs_sample_contents"],
            vec_total=memory_state["vec_total"],
            vec_collections_count=memory_state["vec_collections_count"],
            vec_collections=memory_state["vec_collections"],
            graph_nodes=memory_state["graph_nodes"],
            graph_edges=memory_state["graph_edges"],
            graph_stats=memory_state["graph_stats"],
        )

    def _get_strategy_tools(self, phase: str) -> list[dict]:
        """获取策略生成阶段可用的工具列表。

        包含只读工具 + session 工具 + finish。

        Args:
            phase: 阶段标识，"ingest" 或 "retrieve"。

        Returns:
            工具列表。
        """
        # 直接使用只读工具列表
        read_only_tools = list(READ_ONLY_TOOLS)

        # 替换 finish 工具为策略专用版本
        if phase == "ingest":
            finish_description = (
                "输出最终的摄入策略文档并结束策略生成。\n\n"
                "**必须**在 `result` 参数中填写完整的摄入策略文档（Markdown 格式）。\n"
                "策略文档应包含：信息提取规则、存储路由决策、查重策略、"
                "Source 锚点规范、index.md 维护规则、数据格式规范等内容。\n\n"
                "示例调用：finish(result=\"## 1. 信息提取规则\\n...\")"
            )
        else:
            finish_description = (
                "输出最终的消费策略文档并结束策略生成。\n\n"
                "**必须**在 `result` 参数中填写完整的消费策略文档（Markdown 格式）。\n"
                "策略文档应包含：检索路由决策、结果组织方式、性能优化、"
                "数据格式兼容性等内容。\n\n"
                "示例调用：finish(result=\"## 1. 检索路由决策\\n...\")"
            )

        strategy_finish_tool = {
            "type": "function",
            "function": {
                "name": "finish",
                "description": finish_description,
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

        # 加入源代码读取工具（允许模型查阅存储后端的接口实现）
        read_only_tools.extend(SOURCE_CODE_TOOLS)

        return read_only_tools

    # ------------------------------------------------------------------
    # Phase 3: 摄入代码生成
    # ------------------------------------------------------------------

    async def _generate_ingest_code(
        self,
        user_id: str,
        session_id: str,
        biz_stats: GenerateCodeContextCodeT2TaskStats,
        codegen_strategy: str = "",
    ) -> str:
        """通过多轮 agent loop 生成摄入代码，写入固定目录，备份旧文件为快照。

        LLM 可以使用只读工具探索记忆库结构，然后通过 submit_ingest_code 工具提交代码。
        代码提交后会自动验证（importlib 加载），验证失败时将错误反馈给 LLM 修复。

        Args:
            user_id: 用户 ID。
            session_id: 当前 session ID。
            biz_stats: 业务统计实例。
            codegen_strategy: 摄入消费策略文档内容。

        Returns:
            生成的摄入代码内容（供 Phase 4 使用），失败时返回空字符串。
        """
        try:
            # 1. 构建代码生成的 user prompt
            user_prompt = self._build_ingest_codegen_prompt(user_id, codegen_strategy)

            # 2. 获取工具列表（只读工具 + 源代码工具 + eval_code + submit_ingest_code + finish）
            tools = READ_ONLY_TOOLS + SOURCE_CODE_TOOLS + [PYTHON_SYNTAX_CHECK_TOOL, EVAL_CODE_TOOL, SUBMIT_INGEST_CODE_TOOL, FINISH_TOOL]

            # 3. 执行多轮 agent loop
            logger.info(
                "GenerateCodeContextCodeT2Task: 摄入代码 agent loop 开始 (max_turns=%d)",
                self.MAX_TURNS,
            )
            messages_history: list[dict[str, Any]] = [
                {"role": "user", "content": user_prompt}
            ]

            code_content = ""
            last_submitted_code = ""

            for turn in range(self.MAX_TURNS):
                logger.info(
                    "GenerateCodeContextCodeT2Task: 摄入代码 agent loop turn %d/%d",
                    turn + 1, self.MAX_TURNS,
                )

                response = await self.llm_generate_with_stat(
                    CODEGEN_INGEST_SYSTEM_PROMPT,
                    messages_history,
                    tools=tools,
                    label=f"codegen_ingest_loop_turn_{turn + 1}",
                )

                if not response.tool_calls:
                    # 模型返回纯文本且无工具调用，记录并继续
                    if response.content:
                        messages_history.append({
                            "role": "assistant",
                            "content": response.content,
                        })
                    break

                # 将模型回复加入历史
                messages_history.append(response.to_message())
                should_break = False

                for tc in response.tool_calls:
                    if tc.name == "finish":
                        # finish 工具标识代码生成结束
                        messages_history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "(task finished)",
                        })
                        should_break = True
                        continue

                    if tc.name == "submit_ingest_code":
                        submitted_code = tc.arguments.get("code", "")
                        last_submitted_code = submitted_code
                        code_content = submitted_code
                        messages_history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "✅ 摄入代码已成功提交。",
                        })
                        continue

                    # 执行其他工具调用
                    result_str = await self.dispatch_tool(
                        tc.name, tc.arguments,
                        user_id=user_id,
                        session_id=session_id,
                        allow_write=True,
                    )
                    messages_history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    })

                if should_break:
                    break
            else:
                logger.warning(
                    "GenerateCodeContextCodeT2Task: 摄入代码 agent loop 达到最大轮次 %d",
                    self.MAX_TURNS,
                )
                # 降级：尝试使用最后一次提交的代码
                if last_submitted_code and not code_content:
                    code_content = last_submitted_code

            # 4. 处理结果
            if not code_content:
                biz_stats.ingest_codegen_status = "failed"
                biz_stats.reason = "LLM 未生成有效的摄入 Python 代码"
                logger.warning(
                    "GenerateCodeContextCodeT2Task: 摄入代码生成失败 - 无有效代码"
                )
                return ""

            # 5. 写入暂存区
            self._staging.write_file(INGEST_CODEGEN_FILENAME, code_content)

            # 6. 验证代码（从暂存区路径加载）
            code_abs_path = self._staging.get_staged_abs_path(INGEST_CODEGEN_FILENAME)
            try:
                load_ingestor_class(code_abs_path)
                biz_stats.ingest_codegen_status = "success"
                logger.info(
                    "GenerateCodeContextCodeT2Task: 摄入代码验证通过"
                )
            except Exception as verify_err:
                # 验证失败
                biz_stats.ingest_codegen_status = "failed"
                biz_stats.reason = f"生成的摄入代码验证失败: {verify_err}"
                logger.warning(
                    "GenerateCodeContextCodeT2Task: 摄入代码验证失败: %s",
                    verify_err,
                )
                return ""

            codegen_rel_path = f"{CODEGEN_DIR}/{INGEST_CODEGEN_FILENAME}"
            biz_stats.ingest_codegen_path = codegen_rel_path
            biz_stats.ingest_codegen_length = len(code_content)

            logger.info(
                "GenerateCodeContextCodeT2Task: 摄入代码生成成功 "
                "(path=%s, length=%d)",
                codegen_rel_path, len(code_content),
            )
            return code_content

        except Exception as e:
            biz_stats.ingest_codegen_status = "error"
            biz_stats.reason = f"摄入代码生成异常: {e}"
            logger.error(
                "GenerateCodeContextCodeT2Task: 摄入代码生成异常: %s", e,
            )
            return ""

    # ------------------------------------------------------------------
    # Phase 4: 消费代码生成
    # ------------------------------------------------------------------

    async def _generate_retrieve_code(
        self,
        user_id: str,
        session_id: str,
        biz_stats: GenerateCodeContextCodeT2TaskStats,
        ingest_code: str,
        codegen_strategy: str = "",
    ) -> None:
        """通过多轮 agent loop 生成消费代码（BaseMemoryConsumer 子类），写入固定目录，备份旧文件为快照。

        LLM 可以使用只读工具探索记忆库结构，然后通过 submit_retrieve_code 工具提交代码。
        代码提交后会自动验证（importlib 加载），验证失败时将错误反馈给 LLM 修复。

        Args:
            user_id: 用户 ID。
            session_id: 当前 session ID。
            biz_stats: 业务统计实例。
            ingest_code: Phase 3 生成的摄入代码内容（用于提供给 LLM 参考）。
            codegen_strategy: 摄入消费策略文档内容。
        """
        try:
            # 1. 构建代码生成的 user prompt
            user_prompt = self._build_retrieve_codegen_prompt(user_id, ingest_code, codegen_strategy)

            # 2. 获取工具列表（只读工具 + 源代码工具 + eval_code + submit_retrieve_code + finish）
            tools = READ_ONLY_TOOLS + SOURCE_CODE_TOOLS + [PYTHON_SYNTAX_CHECK_TOOL, EVAL_CODE_TOOL, SUBMIT_RETRIEVE_CODE_TOOL, FINISH_TOOL]

            # 3. 执行多轮 agent loop
            logger.info(
                "GenerateCodeContextCodeT2Task: 消费代码 agent loop 开始 (max_turns=%d)",
                self.MAX_TURNS,
            )
            messages_history: list[dict[str, Any]] = [
                {"role": "user", "content": user_prompt}
            ]

            code_content = ""
            last_submitted_code = ""

            for turn in range(self.MAX_TURNS):
                logger.info(
                    "GenerateCodeContextCodeT2Task: 消费代码 agent loop turn %d/%d",
                    turn + 1, self.MAX_TURNS,
                )

                response = await self.llm_generate_with_stat(
                    CODEGEN_RETRIEVE_SYSTEM_PROMPT,
                    messages_history,
                    tools=tools,
                    label=f"codegen_retrieve_loop_turn_{turn + 1}",
                )

                if not response.tool_calls:
                    # 模型返回纯文本且无工具调用，记录并继续
                    if response.content:
                        messages_history.append({
                            "role": "assistant",
                            "content": response.content,
                        })
                    break

                # 将模型回复加入历史
                messages_history.append(response.to_message())
                should_break = False

                for tc in response.tool_calls:
                    if tc.name == "finish":
                        # finish 工具标识代码生成结束
                        messages_history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "(task finished)",
                        })
                        should_break = True
                        continue

                    if tc.name == "submit_retrieve_code":
                        submitted_code = tc.arguments.get("code", "")
                        last_submitted_code = submitted_code
                        code_content = submitted_code
                        messages_history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "✅ 消费代码已成功提交。",
                        })
                        continue

                    # 执行其他工具调用
                    result_str = await self.dispatch_tool(
                        tc.name, tc.arguments,
                        user_id=user_id,
                        session_id=session_id,
                        allow_write=True,
                    )
                    messages_history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    })

                if should_break:
                    break
            else:
                logger.warning(
                    "GenerateCodeContextCodeT2Task: 消费代码 agent loop 达到最大轮次 %d",
                    self.MAX_TURNS,
                )
                # 降级：尝试使用最后一次提交的代码
                if last_submitted_code and not code_content:
                    code_content = last_submitted_code

            # 4. 处理结果
            if not code_content:
                biz_stats.retrieve_codegen_status = "failed"
                biz_stats.reason = "LLM 未生成有效的消费 Python 代码"
                logger.warning(
                    "GenerateCodeContextCodeT2Task: 消费代码生成失败 - 无有效代码"
                )
                return

            # 5. 写入暂存区
            self._staging.write_file(RETRIEVE_CODEGEN_FILENAME, code_content)

            # 6. 验证代码（从暂存区路径加载）
            code_abs_path = self._staging.get_staged_abs_path(RETRIEVE_CODEGEN_FILENAME)
            try:
                load_consumer_class(code_abs_path)
                biz_stats.retrieve_codegen_status = "success"
                logger.info(
                    "GenerateCodeContextCodeT2Task: 消费代码验证通过"
                )
            except Exception as verify_err:
                # 验证失败
                biz_stats.retrieve_codegen_status = "failed"
                biz_stats.reason = f"生成的消费代码验证失败: {verify_err}"
                logger.warning(
                    "GenerateCodeContextCodeT2Task: 消费代码验证失败: %s",
                    verify_err,
                )
                return

            codegen_rel_path = f"{CODEGEN_DIR}/{RETRIEVE_CODEGEN_FILENAME}"
            biz_stats.retrieve_codegen_path = codegen_rel_path
            biz_stats.retrieve_codegen_length = len(code_content)

            logger.info(
                "GenerateCodeContextCodeT2Task: 消费代码生成成功 "
                "(path=%s, length=%d)",
                codegen_rel_path, len(code_content),
            )

        except Exception as e:
            biz_stats.retrieve_codegen_status = "error"
            biz_stats.reason = f"消费代码生成异常: {e}"
            logger.error(
                "GenerateCodeContextCodeT2Task: 消费代码生成异常: %s", e,
            )

    # ------------------------------------------------------------------
    # Prompt 构建
    # ------------------------------------------------------------------

    def _build_ingest_codegen_prompt(self, user_id: str, codegen_strategy: str = "") -> str:
        """构建摄入代码生成的 user prompt。

        读取当前记忆库状态，注入到提示词模板中。
        自动注入当前时间戳供 LLM 用于类名命名。

        Args:
            user_id: 用户 ID。
            codegen_strategy: 摄入消费策略文档内容。

        Returns:
            完整的 user prompt。
        """
        memory_state = self._collect_memory_state(user_id)
        # 注入时间戳，用于类名命名（MemoryIngestor_YYYYMMDD_HHMMSS）
        memory_state["codegen_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 策略内容（如果为空则提供默认提示）
        strategy_text = codegen_strategy if codegen_strategy else "(本轮未生成代码生成策略，请根据记忆库状态自行判断)"

        return CODEGEN_INGEST_USER_TEMPLATE.format(
            **memory_state,
            codegen_strategy=strategy_text,
        )



    def _build_retrieve_codegen_prompt(self, user_id: str, ingest_code: str, codegen_strategy: str = "") -> str:
        """构建消费代码生成的 user prompt。

        读取当前记忆库状态和摄入代码，注入到提示词模板中。
        自动注入当前时间戳供 LLM 用于类名命名。

        Args:
            user_id: 用户 ID。
            ingest_code: 摄入代码内容。
            codegen_strategy: 摄入消费策略文档内容。

        Returns:
            完整的 user prompt。
        """
        memory_state = self._collect_memory_state(user_id)
        # 注入时间戳，用于类名命名（MemoryConsumer_YYYYMMDD_HHMMSS）
        memory_state["codegen_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 摄入代码文件路径（供提示词参考）
        ingest_codegen_path = f"{self.fs.base_path}/{CODEGEN_DIR}/{INGEST_CODEGEN_FILENAME}"

        # 如果摄入代码为空（Phase 3 失败），尝试读取已有的摄入代码文件
        if not ingest_code:
            ingest_rel_path = f"{CODEGEN_DIR}/{INGEST_CODEGEN_FILENAME}"
            existing_code = self.fs.read_file(ingest_rel_path)
            if not existing_code.startswith("ERROR"):
                ingest_code = existing_code
            else:
                ingest_code = "(摄入代码不可用 — 请根据记忆库结构自行推断数据格式)"

        # 策略内容（如果为空则提供默认提示）
        strategy_text = codegen_strategy if codegen_strategy else "(本轮未生成代码生成策略，请根据记忆库状态自行判断)"

        return CODEGEN_RETRIEVE_USER_TEMPLATE.format(
            **memory_state,
            ingest_strategy_path=ingest_codegen_path,
            ingest_code_path=truncate_with_hint(ingest_code, 5000, f"{CODEGEN_DIR}/{INGEST_CODEGEN_FILENAME}"),
            codegen_strategy=strategy_text,
        )



    def _collect_memory_state(self, user_id: str) -> dict[str, Any]:
        """收集当前记忆库状态信息，供提示词模板使用。

        Args:
            user_id: 用户 ID。

        Returns:
            包含记忆库状态的字典。
        """
        fs_files = self.fs.list_files()
        vec_stats = self.vec.get_stats()
        graph_stats = self.graph.get_stats()

        # 读取 index.md
        index_path, index_content = self.fs.read_index()

        # 采样前 5 个文件内容
        sample_contents: list[str] = []
        for f in fs_files[:5]:
            content = self.fs.read_file(f)
            if not content.startswith("ERROR"):
                sample_contents.append(f"### {f}\n{content[:500]}")

        return {
            "index_path": index_path,
            "index_content": index_content,
            "fs_file_count": len(fs_files),
            "fs_tree": self.fs.tree(max_depth=3),
            "fs_sample_contents": (
                "\n\n".join(sample_contents) if sample_contents else "(无文件)"
            ),
            "vec_total": vec_stats.get("total_entries", 0),
            "vec_collections_count": len(vec_stats.get("collections", {})),
            "vec_collections": str(vec_stats.get("collections", {})),
            "graph_nodes": graph_stats.get("total_nodes", 0),
            "graph_edges": graph_stats.get("total_edges", 0),
            "graph_stats": str(graph_stats),
        }

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------








