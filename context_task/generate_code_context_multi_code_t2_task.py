"""
Generate Code Context Multi Code T2 Task — 纯多策略代码生成任务（不含演进）。

多策略代码生成流程：
1. Phase 1: 生成记忆摄入代码的多个策略设计（agent loop，≤5 个方案）
2. Phase 2: 摄入原子函数封装方案设计（代码工具）
3. Phase 3: 生成摄入增强基类（含原子方法实现）
4. Phase 4: 生成多个摄入子类实现（每个策略方案一个子类）
5. Phase 5: 生成记忆消费代码的多个策略设计（agent loop，≤5 个方案）
6. Phase 6: 消费原子函数封装方案设计（代码工具）
7. Phase 7: 生成消费增强基类（含原子方法实现）
8. Phase 8: 生成多个消费子类实现（每个策略方案一个子类）

生成的代码：
- 多摄入策略文档：.codegen/multi_ingest_strategies.md
- 摄入原子函数设计：.codegen/ingest_atomic_design.md
- 摄入增强基类：.codegen/ingest_base_memory.py
- 摄入子类：.codegen/ingest_memory_v1.py ~ v5.py
- 多消费策略文档：.codegen/multi_retrieve_strategies.md
- 消费原子函数设计：.codegen/retrieve_atomic_design.md
- 消费增强基类：.codegen/retrieve_base_memory.py
- 消费子类：.codegen/retrieve_memory_v1.py ~ v5.py

使用方式：
    task = GenerateCodeContextMultiCodeT2Task(llm, fs_store, vec_store, graph_store)
    result = await task.execute(
        user_id="user_001",
        session_id="session_001",
    )
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from context_task.base_context_task import BaseContextTask, TaskStats
from context_task.generate_code_context_code_t2_task import ingest_reference_file_path_list, \
    retrieve_reference_file_path_list

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
    SUBMIT_BASE_CLASS_CODE_TOOL,
    SUBMIT_SUBCLASS_CODE_TOOL,
    SUBMIT_STRATEGY_TOOL,
    SUBMIT_INGEST_CODE_TOOL,
    SUBMIT_RETRIEVE_CODE_TOOL,
)
from context_task.prompt.chinese_prompt import (
    MULTI_INGEST_STRATEGY_SYSTEM_PROMPT,
    MULTI_INGEST_STRATEGY_USER_TEMPLATE,
    MULTI_RETRIEVE_STRATEGY_SYSTEM_PROMPT,
    MULTI_RETRIEVE_STRATEGY_USER_TEMPLATE,
    ATOMIC_DESIGN_SYSTEM_PROMPT,
    INGEST_ATOMIC_DESIGN_USER_TEMPLATE,
    RETRIEVE_ATOMIC_DESIGN_USER_TEMPLATE,
    BASE_CLASS_GEN_SYSTEM_PROMPT,
    INGEST_BASE_CLASS_USER_TEMPLATE,
    RETRIEVE_BASE_CLASS_USER_TEMPLATE,
    MULTI_SUBCLASS_GEN_SYSTEM_PROMPT,
    MULTI_INGEST_SUBCLASS_USER_TEMPLATE,
    MULTI_RETRIEVE_SUBCLASS_USER_TEMPLATE,
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
    MULTI_INGEST_STRATEGIES_FILENAME,
    MULTI_INGEST_STRATEGIES_FILE_PATH,
    MULTI_RETRIEVE_STRATEGIES_FILENAME,
    MULTI_RETRIEVE_STRATEGIES_FILE_PATH,
    INGEST_ATOMIC_DESIGN_FILENAME,
    INGEST_ATOMIC_DESIGN_FILE_PATH,
    RETRIEVE_ATOMIC_DESIGN_FILENAME,
    RETRIEVE_ATOMIC_DESIGN_FILE_PATH,
    INGEST_BASE_CLASS_FILENAME,
    INGEST_BASE_CLASS_FILE_PATH,
    RETRIEVE_BASE_CLASS_FILENAME,
    RETRIEVE_BASE_CLASS_FILE_PATH,
    CODEGEN_INGEST_STRATEGY_FILE_PATH,
    CODEGEN_RETRIEVE_STRATEGY_FILE_PATH,
    INDEX_FILE_PATH,
    INDEX_FILE_PATH_FALLBACK,
)


# ---------------------------------------------------------------------------
# 业务统计数据结构
# ---------------------------------------------------------------------------


@dataclass
class GenerateCodeContextMultiCodeT2TaskStats(TaskStats):
    """GenerateCodeContextMultiCodeT2Task 的任务级业务统计数据。"""

    # ── 任务结果状态 ──
    status: str = ""
    reason: str = ""

    # ── 上下文标识 ──
    user_id: str = ""
    session_id: str = ""
    scope: str = ""

    # ── Phase 1&2: 演进 ──
    evolve_status: str = ""
    evolve_operations: int = 0
    evolve_summary: str = ""

    # ── Phase 3: 多摄入策略 ──
    multi_ingest_strategy_status: str = ""
    multi_ingest_strategy_count: int = 0

    # ── Phase 4: 摄入原子函数方案 ──
    ingest_atomic_design_status: str = ""
    ingest_atomic_function_count: int = 0

    # ── Phase 5: 摄入增强基类 ──
    ingest_base_class_status: str = ""
    ingest_base_class_path: str = ""

    # ── Phase 6: 摄入子类 ──
    ingest_subclass_status: str = ""
    ingest_subclass_count: int = 0
    ingest_subclass_paths: list = field(default_factory=list)

    # ── Phase 7: 多消费策略 ──
    multi_retrieve_strategy_status: str = ""
    multi_retrieve_strategy_count: int = 0

    # ── Phase 8: 消费原子函数方案 ──
    retrieve_atomic_design_status: str = ""
    retrieve_atomic_function_count: int = 0

    # ── Phase 9: 消费增强基类 ──
    retrieve_base_class_status: str = ""
    retrieve_base_class_path: str = ""

    # ── Phase 10: 消费子类 ──
    retrieve_subclass_status: str = ""
    retrieve_subclass_count: int = 0
    retrieve_subclass_paths: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
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
            "multi_ingest_strategy_status": self.multi_ingest_strategy_status,
            "multi_ingest_strategy_count": self.multi_ingest_strategy_count,
            "ingest_atomic_design_status": self.ingest_atomic_design_status,
            "ingest_atomic_function_count": self.ingest_atomic_function_count,
            "ingest_base_class_status": self.ingest_base_class_status,
            "ingest_base_class_path": self.ingest_base_class_path,
            "ingest_subclass_status": self.ingest_subclass_status,
            "ingest_subclass_count": self.ingest_subclass_count,
            "ingest_subclass_paths": self.ingest_subclass_paths,
            "multi_retrieve_strategy_status": self.multi_retrieve_strategy_status,
            "multi_retrieve_strategy_count": self.multi_retrieve_strategy_count,
            "retrieve_atomic_design_status": self.retrieve_atomic_design_status,
            "retrieve_atomic_function_count": self.retrieve_atomic_function_count,
            "retrieve_base_class_status": self.retrieve_base_class_status,
            "retrieve_base_class_path": self.retrieve_base_class_path,
            "retrieve_subclass_status": self.retrieve_subclass_status,
            "retrieve_subclass_count": self.retrieve_subclass_count,
            "retrieve_subclass_paths": self.retrieve_subclass_paths,
        })
        return base


# ---------------------------------------------------------------------------
# GenerateCodeContextMultiCodeT2Task
# ---------------------------------------------------------------------------


class GenerateCodeContextMultiCodeT2Task(BaseContextTask):
    """多策略代码生成任务（不含演进）。

    整合八个阶段：
    1. Phase 1: 多摄入策略设计（agent loop，≤5 个方案）
    2. Phase 2: 摄入原子函数封装方案
    3. Phase 3: 生成摄入增强基类
    4. Phase 4: 生成多个摄入子类
    5. Phase 5: 多消费策略设计（agent loop，≤5 个方案）
    6. Phase 6: 消费原子函数封装方案
    7. Phase 7: 生成消费增强基类
    8. Phase 8: 生成多个消费子类
    """

    task_name = "generate_code_context_multi_code_t2"

    # ── 配置 ──
    MIN_ITEMS_FOR_EVOLUTION: int = 5
    MAX_TURNS: int = 20

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
        self._stats = GenerateCodeContextMultiCodeT2TaskStats(task_name=self.task_name)
        self._stats.start_time = _start_time

    async def run(
        self,
        user_id: str = "default_user",
        session_id: str = "",
        scope: str = "periodic",
        **kwargs: Any,
    ) -> None:
        """核心逻辑：8 个 Phase 顺序执行（纯代码生成，不含演进）。

        使用暂存区保证原子性：所有代码文件先写入暂存区，全部完成后一次性提交。
        """
        biz_stats: GenerateCodeContextMultiCodeT2TaskStats = self._stats  # type: ignore[assignment]
        biz_stats.user_id = user_id
        biz_stats.session_id = session_id
        biz_stats.scope = scope

        logger.info(
            "GenerateCodeContextMultiCodeT2Task: run start "
            "(user_id=%s, session_id=%s, scope=%s)",
            user_id, session_id, scope,
        )

        # 创建暂存区，保证代码修改的原子性
        self._staging = CodegenStagingArea(self.fs)

        # ── Phase 1: 多摄入策略设计 ──
        logger.info("GenerateCodeContextMultiCodeT2Task: Phase 1 - 多摄入策略设计开始")
        multi_ingest_strategies = await self._generate_multi_ingest_strategies(
            user_id, session_id, biz_stats,
        )
        logger.info(
            "GenerateCodeContextMultiCodeT2Task: Phase 1 完成 (status=%s, count=%d)",
            biz_stats.multi_ingest_strategy_status, biz_stats.multi_ingest_strategy_count,
        )

        # ── Phase 2: 摄入原子函数封装方案 ──
        logger.info("GenerateCodeContextMultiCodeT2Task: Phase 2 - 摄入原子函数封装方案开始")
        ingest_atomic_design = await self._generate_atomic_design(
            user_id, session_id, biz_stats,
            multi_strategies=multi_ingest_strategies,
            phase_type="ingest",
        )
        logger.info(
            "GenerateCodeContextMultiCodeT2Task: Phase 2 完成 (status=%s)",
            biz_stats.ingest_atomic_design_status,
        )

        # ── Phase 3: 生成摄入增强基类 ──
        logger.info("GenerateCodeContextMultiCodeT2Task: Phase 3 - 生成摄入增强基类开始")
        ingest_base_class_code = await self._generate_base_class(
            user_id, session_id, biz_stats,
            atomic_design=ingest_atomic_design,
            phase_type="ingest",
        )
        logger.info(
            "GenerateCodeContextMultiCodeT2Task: Phase 3 完成 (status=%s)",
            biz_stats.ingest_base_class_status,
        )

        # ── Phase 4: 生成多个摄入子类 ──
        logger.info("GenerateCodeContextMultiCodeT2Task: Phase 4 - 生成多个摄入子类开始")
        await self._generate_multi_subclasses(
            user_id, session_id, biz_stats,
            multi_strategies=multi_ingest_strategies,
            base_class_code=ingest_base_class_code,
            phase_type="ingest",
        )
        logger.info(
            "GenerateCodeContextMultiCodeT2Task: Phase 4 完成 (status=%s, count=%d)",
            biz_stats.ingest_subclass_status, biz_stats.ingest_subclass_count,
        )

        # ── Phase 5: 多消费策略设计 ──
        logger.info("GenerateCodeContextMultiCodeT2Task: Phase 5 - 多消费策略设计开始")
        multi_retrieve_strategies = await self._generate_multi_retrieve_strategies(
            user_id, session_id, biz_stats,
            ingest_strategies=multi_ingest_strategies,
        )
        logger.info(
            "GenerateCodeContextMultiCodeT2Task: Phase 5 完成 (status=%s, count=%d)",
            biz_stats.multi_retrieve_strategy_status, biz_stats.multi_retrieve_strategy_count,
        )

        # ── Phase 6: 消费原子函数封装方案 ──
        logger.info("GenerateCodeContextMultiCodeT2Task: Phase 6 - 消费原子函数封装方案开始")
        retrieve_atomic_design = await self._generate_atomic_design(
            user_id, session_id, biz_stats,
            multi_strategies=multi_retrieve_strategies,
            phase_type="retrieve",
        )
        logger.info(
            "GenerateCodeContextMultiCodeT2Task: Phase 6 完成 (status=%s)",
            biz_stats.retrieve_atomic_design_status,
        )

        # ── Phase 7: 生成消费增强基类 ──
        logger.info("GenerateCodeContextMultiCodeT2Task: Phase 7 - 生成消费增强基类开始")
        retrieve_base_class_code = await self._generate_base_class(
            user_id, session_id, biz_stats,
            atomic_design=retrieve_atomic_design,
            phase_type="retrieve",
        )
        logger.info(
            "GenerateCodeContextMultiCodeT2Task: Phase 7 完成 (status=%s)",
            biz_stats.retrieve_base_class_status,
        )

        # ── Phase 8: 生成多个消费子类 ──
        logger.info("GenerateCodeContextMultiCodeT2Task: Phase 8 - 生成多个消费子类开始")
        await self._generate_multi_subclasses(
            user_id, session_id, biz_stats,
            multi_strategies=multi_retrieve_strategies,
            base_class_code=retrieve_base_class_code,
            phase_type="retrieve",
        )
        logger.info(
            "GenerateCodeContextMultiCodeT2Task: Phase 8 完成 (status=%s, count=%d)",
            biz_stats.retrieve_subclass_status, biz_stats.retrieve_subclass_count,
        )

        # ── 最终状态判定 ──
        self._determine_final_status(biz_stats)

        # ── 原子性提交或丢弃暂存区 ──
        if biz_stats.status == "success":
            commit_result = self._staging.commit()
            if not commit_result["committed"]:
                logger.warning(
                    "GenerateCodeContextMultiCodeT2Task: 暂存区提交失败: %s",
                    commit_result["error"],
                )
                biz_stats.status = "partial"
                biz_stats.reason = f"暂存区提交失败: {commit_result['error']}"
            else:
                logger.info(
                    "GenerateCodeContextMultiCodeT2Task: 代码已原子性提交 "
                    "(backed_up=%d, deployed=%d)",
                    len(commit_result["backed_up"]),
                    len(commit_result["deployed"]),
                )
        else:
            self._staging.discard()
            logger.info(
                "GenerateCodeContextMultiCodeT2Task: 任务未完全成功，暂存区已丢弃"
            )

        logger.info(
            "GenerateCodeContextMultiCodeT2Task: run 完成 (status=%s)",
            biz_stats.status,
        )

    async def post_run(self, *, session_id: str = "", **kwargs: Any) -> None:
        """后置处理：记录完成日志。"""
        biz_stats: GenerateCodeContextMultiCodeT2TaskStats = self._stats  # type: ignore[assignment]
        logger.info(
            "GenerateCodeContextMultiCodeT2Task: completed, "
            "status=%s, ingest_subclasses=%d, retrieve_subclasses=%d",
            biz_stats.status,
            biz_stats.ingest_subclass_count,
            biz_stats.retrieve_subclass_count,
        )

    # ------------------------------------------------------------------
    # Phase 3: 多摄入策略设计
    # ------------------------------------------------------------------

    async def _generate_multi_ingest_strategies(
        self,
        user_id: str,
        session_id: str,
        biz_stats: GenerateCodeContextMultiCodeT2TaskStats,
    ) -> str:
        """通过多轮 agent loop 生成多个摄入策略方案。

        Returns:
            多策略文档内容，失败时返回空字符串。
        """
        try:
            user_prompt = self._build_multi_ingest_strategy_prompt(user_id)
            tools = self._get_strategy_tools("ingest")

            strategy_result = await self._run_strategy_agent_loop(
                system_prompt=MULTI_INGEST_STRATEGY_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                tools=tools,
                user_id=user_id,
                session_id=session_id,
                label_prefix="multi_ingest_strategy",
            )

            if strategy_result:
                self._staging.write_file(MULTI_INGEST_STRATEGIES_FILENAME, strategy_result)
                biz_stats.multi_ingest_strategy_status = "success"
                biz_stats.multi_ingest_strategy_count = self._count_strategies(strategy_result)
                logger.info(
                    "GenerateCodeContextMultiCodeT2Task: 多摄入策略生成成功 (count=%d)",
                    biz_stats.multi_ingest_strategy_count,
                )
            else:
                biz_stats.multi_ingest_strategy_status = "failed"
                logger.warning("GenerateCodeContextMultiCodeT2Task: 多摄入策略生成未产出有效内容")

            return strategy_result

        except Exception as e:
            biz_stats.multi_ingest_strategy_status = "error"
            logger.error("GenerateCodeContextMultiCodeT2Task: 多摄入策略生成异常: %s", e)
            return ""

    # ------------------------------------------------------------------
    # Phase 4/8: 原子函数封装方案
    # ------------------------------------------------------------------

    async def _generate_atomic_design(
        self,
        user_id: str,
        session_id: str,
        biz_stats: GenerateCodeContextMultiCodeT2TaskStats,
        multi_strategies: str,
        phase_type: str,
    ) -> str:
        """通过多轮 agent loop 生成原子函数封装方案。

        Args:
            multi_strategies: 多策略文档内容。
            phase_type: "ingest" 或 "retrieve"。

        Returns:
            原子函数设计文档内容，失败时返回空字符串。
        """
        try:
            if not multi_strategies:
                if phase_type == "ingest":
                    biz_stats.ingest_atomic_design_status = "skipped"
                else:
                    biz_stats.retrieve_atomic_design_status = "skipped"
                return ""

            user_prompt = self._build_atomic_design_prompt(multi_strategies, phase_type)
            tools = READ_ONLY_TOOLS + SOURCE_CODE_TOOLS + [PYTHON_SYNTAX_CHECK_TOOL, SUBMIT_STRATEGY_TOOL, FINISH_TOOL]

            design_result = await self._run_strategy_agent_loop(
                system_prompt=ATOMIC_DESIGN_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                tools=tools,
                user_id=user_id,
                session_id=session_id,
                label_prefix=f"{phase_type}_atomic_design",
            )

            if design_result:
                if phase_type == "ingest":
                    self._staging.write_file(INGEST_ATOMIC_DESIGN_FILENAME, design_result)
                    biz_stats.ingest_atomic_design_status = "success"
                    biz_stats.ingest_atomic_function_count = self._count_atomic_functions(design_result)
                else:
                    self._staging.write_file(RETRIEVE_ATOMIC_DESIGN_FILENAME, design_result)
                    biz_stats.retrieve_atomic_design_status = "success"
                    biz_stats.retrieve_atomic_function_count = self._count_atomic_functions(design_result)
            else:
                if phase_type == "ingest":
                    biz_stats.ingest_atomic_design_status = "failed"
                else:
                    biz_stats.retrieve_atomic_design_status = "failed"

            return design_result

        except Exception as e:
            if phase_type == "ingest":
                biz_stats.ingest_atomic_design_status = "error"
            else:
                biz_stats.retrieve_atomic_design_status = "error"
            logger.error(
                "GenerateCodeContextMultiCodeT2Task: %s 原子函数设计异常: %s",
                phase_type, e,
            )
            return ""

    # ------------------------------------------------------------------
    # Phase 5/9: 增强基类生成
    # ------------------------------------------------------------------

    async def _generate_base_class(
        self,
        user_id: str,
        session_id: str,
        biz_stats: GenerateCodeContextMultiCodeT2TaskStats,
        atomic_design: str,
        phase_type: str,
    ) -> str:
        """通过多轮 agent loop 生成增强基类代码。

        Args:
            atomic_design: 原子函数设计文档内容。
            phase_type: "ingest" 或 "retrieve"。

        Returns:
            增强基类代码内容，失败时返回空字符串。
        """
        try:
            if not atomic_design:
                if phase_type == "ingest":
                    biz_stats.ingest_base_class_status = "skipped"
                else:
                    biz_stats.retrieve_base_class_status = "skipped"
                return ""

            user_prompt = self._build_base_class_prompt(atomic_design, phase_type)
            tools = READ_ONLY_TOOLS + SOURCE_CODE_TOOLS + [PYTHON_SYNTAX_CHECK_TOOL, EVAL_CODE_TOOL, SUBMIT_BASE_CLASS_CODE_TOOL, FINISH_TOOL]

            messages_history: list[dict[str, Any]] = [
                {"role": "user", "content": user_prompt}
            ]

            code_content = ""

            for turn in range(self.MAX_TURNS):
                logger.info(
                    "GenerateCodeContextMultiCodeT2Task: %s 基类生成 turn %d/%d",
                    phase_type, turn + 1, self.MAX_TURNS,
                )

                response = await self.llm_generate_with_stat(
                    BASE_CLASS_GEN_SYSTEM_PROMPT,
                    messages_history,
                    tools=tools,
                    label=f"{phase_type}_base_class_turn_{turn + 1}",
                )

                if not response.tool_calls:
                    if response.content:
                        messages_history.append({"role": "assistant", "content": response.content})
                    break

                messages_history.append(response.to_message())
                should_break = False

                for tc in response.tool_calls:
                    if tc.name == "finish":
                        messages_history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "(task finished)",
                        })
                        should_break = True
                        continue

                    if tc.name == "submit_base_class_code":
                        submitted_code = tc.arguments.get("code", "")
                        code_content = submitted_code
                        messages_history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "✅ 增强基类代码已成功提交。",
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

            # 处理结果
            if code_content:
                if phase_type == "ingest":
                    file_path = INGEST_BASE_CLASS_FILE_PATH
                    self._staging.write_file(INGEST_BASE_CLASS_FILENAME, code_content)
                else:
                    file_path = RETRIEVE_BASE_CLASS_FILE_PATH
                    self._staging.write_file(RETRIEVE_BASE_CLASS_FILENAME, code_content)

                if phase_type == "ingest":
                    biz_stats.ingest_base_class_status = "success"
                    biz_stats.ingest_base_class_path = file_path
                else:
                    biz_stats.retrieve_base_class_status = "success"
                    biz_stats.retrieve_base_class_path = file_path

                logger.info(
                    "GenerateCodeContextMultiCodeT2Task: %s 增强基类生成成功 (path=%s)",
                    phase_type, file_path,
                )
            else:
                if phase_type == "ingest":
                    biz_stats.ingest_base_class_status = "failed"
                else:
                    biz_stats.retrieve_base_class_status = "failed"

            return code_content

        except Exception as e:
            if phase_type == "ingest":
                biz_stats.ingest_base_class_status = "error"
            else:
                biz_stats.retrieve_base_class_status = "error"
            logger.error(
                "GenerateCodeContextMultiCodeT2Task: %s 增强基类生成异常: %s",
                phase_type, e,
            )
            return ""

    # ------------------------------------------------------------------
    # Phase 6/10: 多子类生成
    # ------------------------------------------------------------------

    async def _generate_multi_subclasses(
        self,
        user_id: str,
        session_id: str,
        biz_stats: GenerateCodeContextMultiCodeT2TaskStats,
        multi_strategies: str,
        base_class_code: str,
        phase_type: str,
    ) -> None:
        """为每个策略方案生成一个子类实现。

        Args:
            multi_strategies: 多策略文档内容。
            base_class_code: 增强基类代码内容。
            phase_type: "ingest" 或 "retrieve"。
        """
        try:
            if not multi_strategies or not base_class_code:
                if phase_type == "ingest":
                    biz_stats.ingest_subclass_status = "skipped"
                else:
                    biz_stats.retrieve_subclass_status = "skipped"
                return

            # 拆分多策略文档为各个方案
            strategies = self._split_strategies(multi_strategies)
            if not strategies:
                if phase_type == "ingest":
                    biz_stats.ingest_subclass_status = "failed"
                else:
                    biz_stats.retrieve_subclass_status = "failed"
                return

            success_count = 0
            subclass_paths: list[str] = []

            for idx, strategy_content in enumerate(strategies, start=1):
                logger.info(
                    "GenerateCodeContextMultiCodeT2Task: %s 子类 v%d 生成开始",
                    phase_type, idx,
                )

                code = await self._generate_single_subclass(
                    user_id=user_id,
                    session_id=session_id,
                    strategy_content=strategy_content,
                    base_class_code=base_class_code,
                    phase_type=phase_type,
                    version=idx,
                )

                if code:
                    # 写入暂存区
                    if phase_type == "ingest":
                        filename = f"ingest_memory_v{idx}.py"
                    else:
                        filename = f"retrieve_memory_v{idx}.py"

                    self._staging.write_file(filename, code)

                    # 验证代码（从暂存区路径加载）
                    abs_path = self._staging.get_staged_abs_path(filename)
                    try:
                        if phase_type == "ingest":
                            load_ingestor_class(abs_path)
                        else:
                            load_consumer_class(abs_path)
                        success_count += 1
                        subclass_paths.append(f"{CODEGEN_DIR}/{filename}")
                        logger.info(
                            "GenerateCodeContextMultiCodeT2Task: %s 子类 v%d 验证通过",
                            phase_type, idx,
                        )
                    except Exception as verify_err:
                        logger.warning(
                            "GenerateCodeContextMultiCodeT2Task: %s 子类 v%d 验证失败: %s",
                            phase_type, idx, verify_err,
                        )
                else:
                    logger.warning(
                        "GenerateCodeContextMultiCodeT2Task: %s 子类 v%d 生成失败",
                        phase_type, idx,
                    )

            # 更新统计
            if phase_type == "ingest":
                biz_stats.ingest_subclass_count = success_count
                biz_stats.ingest_subclass_paths = subclass_paths
                biz_stats.ingest_subclass_status = "success" if success_count > 0 else "failed"
            else:
                biz_stats.retrieve_subclass_count = success_count
                biz_stats.retrieve_subclass_paths = subclass_paths
                biz_stats.retrieve_subclass_status = "success" if success_count > 0 else "failed"

        except Exception as e:
            if phase_type == "ingest":
                biz_stats.ingest_subclass_status = "error"
            else:
                biz_stats.retrieve_subclass_status = "error"
            logger.error(
                "GenerateCodeContextMultiCodeT2Task: %s 子类生成异常: %s",
                phase_type, e,
            )

    async def _generate_single_subclass(
        self,
        user_id: str,
        session_id: str,
        strategy_content: str,
        base_class_code: str,
        phase_type: str,
        version: int,
    ) -> str:
        """生成单个子类实现。

        Returns:
            子类代码内容，失败时返回空字符串。
        """
        try:
            user_prompt = self._build_subclass_prompt(
                strategy_content, base_class_code, phase_type, version,
            )

            if phase_type == "ingest":
                tools = READ_ONLY_TOOLS + SOURCE_CODE_TOOLS + [PYTHON_SYNTAX_CHECK_TOOL, EVAL_CODE_TOOL, SUBMIT_SUBCLASS_CODE_TOOL, FINISH_TOOL]
            else:
                tools = READ_ONLY_TOOLS + SOURCE_CODE_TOOLS + [PYTHON_SYNTAX_CHECK_TOOL, EVAL_CODE_TOOL, SUBMIT_SUBCLASS_CODE_TOOL, FINISH_TOOL]

            messages_history: list[dict[str, Any]] = [
                {"role": "user", "content": user_prompt}
            ]

            code_content = ""

            for turn in range(self.MAX_TURNS):
                response = await self.llm_generate_with_stat(
                    MULTI_SUBCLASS_GEN_SYSTEM_PROMPT,
                    messages_history,
                    tools=tools,
                    label=f"{phase_type}_subclass_v{version}_turn_{turn + 1}",
                )

                if not response.tool_calls:
                    if response.content:
                        messages_history.append({"role": "assistant", "content": response.content})
                    break

                messages_history.append(response.to_message())
                should_break = False

                for tc in response.tool_calls:
                    if tc.name == "finish":
                        messages_history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "(task finished)",
                        })
                        should_break = True
                        continue

                    if tc.name == "submit_subclass_code":
                        submitted_code = tc.arguments.get("code", "")
                        if not submitted_code:
                            messages_history.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": "❌ 提交失败：代码为空，请重新提交。",
                            })
                            continue

                        # 即时语法验证：写入暂存区，尝试 load 验证
                        if phase_type == "ingest":
                            tmp_filename = f"ingest_memory_v{version}.py"
                        else:
                            tmp_filename = f"retrieve_memory_v{version}.py"
                        self._staging.write_file(tmp_filename, submitted_code)
                        abs_tmp_path = self._staging.get_staged_abs_path(tmp_filename)

                        try:
                            if phase_type == "ingest":
                                load_ingestor_class(abs_tmp_path)
                            else:
                                load_consumer_class(abs_tmp_path)
                            # 验证通过
                            code_content = submitted_code
                            messages_history.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": f"✅ 子类 v{version} 代码已成功提交并验证通过。",
                            })
                        except Exception as verify_err:
                            # 验证失败，返回错误信息让 agent 修复
                            logger.warning(
                                "GenerateCodeContextMultiCodeT2Task: %s 子类 v%d 提交时验证失败: %s",
                                phase_type, version, verify_err,
                            )
                            messages_history.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": (
                                    f"❌ 子类 v{version} 验证失败，请修复后重新提交。\n"
                                    f"错误信息：{verify_err}\n"
                                    f"请检查代码中的语法错误（如未终止的字符串、缩进错误、import 错误等），"
                                    f"修复后再次调用 submit_subclass_code 提交。"
                                ),
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

            return code_content

        except Exception as e:
            logger.error(
                "GenerateCodeContextMultiCodeT2Task: %s 子类 v%d 生成异常: %s",
                phase_type, version, e,
            )
            return ""

    # ------------------------------------------------------------------
    # Phase 7: 多消费策略设计
    # ------------------------------------------------------------------

    async def _generate_multi_retrieve_strategies(
        self,
        user_id: str,
        session_id: str,
        biz_stats: GenerateCodeContextMultiCodeT2TaskStats,
        ingest_strategies: str = "",
    ) -> str:
        """通过多轮 agent loop 生成多个消费策略方案。

        Args:
            ingest_strategies: 摄入策略文档内容（供消费策略参考）。

        Returns:
            多策略文档内容，失败时返回空字符串。
        """
        try:
            user_prompt = self._build_multi_retrieve_strategy_prompt(user_id, ingest_strategies)
            tools = self._get_strategy_tools("retrieve")

            strategy_result = await self._run_strategy_agent_loop(
                system_prompt=MULTI_RETRIEVE_STRATEGY_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                tools=tools,
                user_id=user_id,
                session_id=session_id,
                label_prefix="multi_retrieve_strategy",
            )

            if strategy_result:
                self._staging.write_file(MULTI_RETRIEVE_STRATEGIES_FILENAME, strategy_result)
                biz_stats.multi_retrieve_strategy_status = "success"
                biz_stats.multi_retrieve_strategy_count = self._count_strategies(strategy_result)
                logger.info(
                    "GenerateCodeContextMultiCodeT2Task: 多消费策略生成成功 (count=%d)",
                    biz_stats.multi_retrieve_strategy_count,
                )
            else:
                biz_stats.multi_retrieve_strategy_status = "failed"
                logger.warning("GenerateCodeContextMultiCodeT2Task: 多消费策略生成未产出有效内容")

            return strategy_result

        except Exception as e:
            biz_stats.multi_retrieve_strategy_status = "error"
            logger.error("GenerateCodeContextMultiCodeT2Task: 多消费策略生成异常: %s", e)
            return ""

    # ------------------------------------------------------------------
    # 通用 Agent Loop
    # ------------------------------------------------------------------

    async def _run_strategy_agent_loop(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict],
        user_id: str,
        session_id: str,
        label_prefix: str,
    ) -> str:
        """通用的策略生成 agent loop（方案C：混合回退机制）。

        策略内容获取优先级：
        1. submit_strategy 工具提交的内容（推荐，避免 JSON 转义和截断问题）
        2. finish 工具的 result 参数（兼容旧行为）
        3. 模型的 content / reasoning_content（最终回退）

        Returns:
            策略文档内容，失败时返回空字符串。
        """
        messages_history: list[dict[str, Any]] = [
            {"role": "user", "content": user_prompt}
        ]

        # 三级回退变量
        submitted_strategy = ""  # 优先级1: submit_strategy 工具提交
        finish_result = ""       # 优先级2: finish 的 result 参数
        content_fallback = ""    # 优先级3: 模型直接输出的 content

        for turn in range(self.MAX_TURNS):
            logger.info(
                "GenerateCodeContextMultiCodeT2Task: %s agent loop turn %d/%d",
                label_prefix, turn + 1, self.MAX_TURNS,
            )

            response = await self.llm_generate_with_stat(
                system_prompt,
                messages_history,
                tools=tools,
                label=f"{label_prefix}_turn_{turn + 1}",
            )

            if not response.tool_calls:
                # 模型没有调用工具，直接输出了内容
                if response.content:
                    content_fallback = response.content
                elif response.reasoning_content:
                    content_fallback = response.reasoning_content
                break

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

                        char_count = len(submitted_strategy)
                        messages_history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": (
                                f"✅ 策略文档已成功提交（当前共 {char_count} 字符）。\n"
                                f"请调用 `finish` 工具结束任务。"
                            ),
                        })
                        logger.info(
                            "GenerateCodeContextMultiCodeT2Task: %s submit_strategy 收到内容 "
                            "(chars=%d, append=%s)",
                            label_prefix, len(new_content), is_append,
                        )
                    else:
                        messages_history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": (
                                "❌ 提交失败：content 参数为空。\n"
                                "请在 content 参数中填写完整的策略文档内容后重新提交。"
                            ),
                        })
                    continue

                if tc.name == "finish":
                    # 优先级2: finish 的 result 参数
                    finish_result = tc.arguments.get("result", "")
                    messages_history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "(strategy generation finished)",
                    })
                    should_break = True
                    continue

                # 执行其他工具调用（只读工具）
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
                "GenerateCodeContextMultiCodeT2Task: %s agent loop 达到最大轮次 %d",
                label_prefix, self.MAX_TURNS,
            )

        # ── 三级回退：选择最终策略内容 ──
        strategy_result = ""

        if submitted_strategy:
            strategy_result = submitted_strategy
            logger.info(
                "GenerateCodeContextMultiCodeT2Task: %s 使用 submit_strategy 提交的内容 (chars=%d)",
                label_prefix, len(strategy_result),
            )
        elif finish_result:
            strategy_result = finish_result
            logger.info(
                "GenerateCodeContextMultiCodeT2Task: %s 回退到 finish.result (chars=%d)",
                label_prefix, len(strategy_result),
            )
        elif content_fallback:
            strategy_result = content_fallback
            logger.info(
                "GenerateCodeContextMultiCodeT2Task: %s 回退到 content (chars=%d)",
                label_prefix, len(strategy_result),
            )
        else:
            logger.warning(
                "GenerateCodeContextMultiCodeT2Task: %s 所有回退均为空，策略生成失败",
                label_prefix,
            )

        return strategy_result

    # ------------------------------------------------------------------
    # Prompt 构建
    # ------------------------------------------------------------------

    def _build_multi_ingest_strategy_prompt(self, user_id: str) -> str:
        """构建多摄入策略生成的 user prompt。"""
        memory_state = self._collect_memory_state(user_id)

        # 读取演进策略文件
        evolve_strategy_content = self.fs.read_file(STRATEGY_FILE_PATH)
        if evolve_strategy_content.startswith("ERROR"):
            evolve_strategy_content = "(演进策略文件不存在)"

        # 读取旧的摄入策略文件
        old_ingest_strategy_content = self.fs.read_file(CODEGEN_INGEST_STRATEGY_FILE_PATH)
        if old_ingest_strategy_content.startswith("ERROR"):
            old_ingest_strategy_content = "(尚无旧摄入策略文件 — 这是首次生成)"

        return MULTI_INGEST_STRATEGY_USER_TEMPLATE.format(
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

    def _build_multi_retrieve_strategy_prompt(self, user_id: str, ingest_strategies: str = "") -> str:
        """构建多消费策略生成的 user prompt。"""
        memory_state = self._collect_memory_state(user_id)

        # 读取演进策略文件
        evolve_strategy_content = self.fs.read_file(STRATEGY_FILE_PATH)
        if evolve_strategy_content.startswith("ERROR"):
            evolve_strategy_content = "(演进策略文件不存在)"

        # 摄入策略内容
        ingest_strategies_content = ingest_strategies if ingest_strategies else "(摄入策略不可用)"

        # 读取旧的消费策略文件
        old_retrieve_strategy_content = self.fs.read_file(CODEGEN_RETRIEVE_STRATEGY_FILE_PATH)
        if old_retrieve_strategy_content.startswith("ERROR"):
            old_retrieve_strategy_content = "(尚无旧消费策略文件 — 这是首次生成)"

        return MULTI_RETRIEVE_STRATEGY_USER_TEMPLATE.format(
            index_path=memory_state["index_path"],
            index_content=memory_state["index_content"],
            evolve_strategy_file_path=STRATEGY_FILE_PATH,
            evolve_strategy_content=truncate_with_hint(evolve_strategy_content, 3000, STRATEGY_FILE_PATH),
            ingest_strategies_content=truncate_with_hint(ingest_strategies_content, 3000, MULTI_INGEST_STRATEGIES_FILE_PATH),
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

    def _build_atomic_design_prompt(self, multi_strategies: str, phase_type: str) -> str:
        """构建原子函数设计的 user prompt。"""
        memory_state = self._collect_memory_state("")

        if phase_type == "ingest":
            return INGEST_ATOMIC_DESIGN_USER_TEMPLATE.format(
                multi_strategies_content=truncate_with_hint(multi_strategies, 8000, MULTI_INGEST_STRATEGIES_FILE_PATH),
                multi_strategies_file_path=MULTI_INGEST_STRATEGIES_FILE_PATH,
                fs_tree=memory_state["fs_tree"],
                vec_total=memory_state["vec_total"],
                vec_collections_count=memory_state["vec_collections_count"],
                vec_collections=memory_state["vec_collections"],
                graph_nodes=memory_state["graph_nodes"],
                graph_edges=memory_state["graph_edges"],
                graph_stats=memory_state["graph_stats"],
            )
        else:
            return RETRIEVE_ATOMIC_DESIGN_USER_TEMPLATE.format(
                multi_strategies_content=truncate_with_hint(multi_strategies, 8000, MULTI_RETRIEVE_STRATEGIES_FILE_PATH),
                multi_strategies_file_path=MULTI_RETRIEVE_STRATEGIES_FILE_PATH,
                fs_tree=memory_state["fs_tree"],
                vec_total=memory_state["vec_total"],
                vec_collections_count=memory_state["vec_collections_count"],
                vec_collections=memory_state["vec_collections"],
                graph_nodes=memory_state["graph_nodes"],
                graph_edges=memory_state["graph_edges"],
                graph_stats=memory_state["graph_stats"],
            )

    def _build_base_class_prompt(self, atomic_design: str, phase_type: str) -> str:
        """构建增强基类生成的 user prompt。"""
        if phase_type == "ingest":
            return INGEST_BASE_CLASS_USER_TEMPLATE.format(
                atomic_design_content=truncate_with_hint(atomic_design, 8000, INGEST_ATOMIC_DESIGN_FILE_PATH),
                atomic_design_file_path=INGEST_ATOMIC_DESIGN_FILE_PATH,
            )
        else:
            return RETRIEVE_BASE_CLASS_USER_TEMPLATE.format(
                atomic_design_content=truncate_with_hint(atomic_design, 8000, RETRIEVE_ATOMIC_DESIGN_FILE_PATH),
                atomic_design_file_path=RETRIEVE_ATOMIC_DESIGN_FILE_PATH,
            )

    def _build_subclass_prompt(
        self,
        strategy_content: str,
        base_class_code: str,
        phase_type: str,
        version: int,
    ) -> str:
        """构建子类生成的 user prompt。"""
        memory_state = self._collect_memory_state("")
        codegen_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if phase_type == "ingest":
            return MULTI_INGEST_SUBCLASS_USER_TEMPLATE.format(
                strategy_content=truncate_with_hint(strategy_content, 5000, MULTI_INGEST_STRATEGIES_FILE_PATH),
                multi_strategies_file_path=MULTI_INGEST_STRATEGIES_FILE_PATH,
                base_class_code=truncate_with_hint(base_class_code, 8000, INGEST_BASE_CLASS_FILE_PATH),
                base_class_file_path=INGEST_BASE_CLASS_FILE_PATH,
                index_path=memory_state["index_path"],
                index_content=memory_state["index_content"],
                fs_file_count=memory_state["fs_file_count"],
                fs_tree=memory_state["fs_tree"],
                vec_total=memory_state["vec_total"],
                vec_collections_count=memory_state["vec_collections_count"],
                vec_collections=memory_state["vec_collections"],
                graph_nodes=memory_state["graph_nodes"],
                graph_edges=memory_state["graph_edges"],
                graph_stats=memory_state["graph_stats"],
                version_tag=f"v{version}_{codegen_timestamp}",
                codegen_timestamp=f"v{version}_{codegen_timestamp}",
            )
        else:
            from pathlib import PurePosixPath
            base_class_file_stem = PurePosixPath(RETRIEVE_BASE_CLASS_FILE_PATH).stem

            return MULTI_RETRIEVE_SUBCLASS_USER_TEMPLATE.format(
                strategy_content=truncate_with_hint(strategy_content, 5000, MULTI_RETRIEVE_STRATEGIES_FILE_PATH),
                multi_strategies_file_path=MULTI_RETRIEVE_STRATEGIES_FILE_PATH,
                base_class_code=truncate_with_hint(base_class_code, 8000, RETRIEVE_BASE_CLASS_FILE_PATH),
                base_class_file_path=RETRIEVE_BASE_CLASS_FILE_PATH,
                base_class_name="EnhancedMemoryConsumer",
                base_class_file_stem=base_class_file_stem,
                base_class_section_title="增强基类骨架",
                base_class_description="增强基类的代码（包含所有可用的原子方法）",
                index_path=memory_state["index_path"],
                index_content=memory_state["index_content"],
                fs_file_count=memory_state["fs_file_count"],
                fs_tree=memory_state["fs_tree"],
                vec_total=memory_state["vec_total"],
                vec_collections_count=memory_state["vec_collections_count"],
                vec_collections=memory_state["vec_collections"],
                graph_nodes=memory_state["graph_nodes"],
                graph_edges=memory_state["graph_edges"],
                graph_stats=memory_state["graph_stats"],
                version_tag=f"v{version}_{codegen_timestamp}",
                codegen_timestamp=f"v{version}_{codegen_timestamp}",
            )

    # ------------------------------------------------------------------
    # 工具获取
    # ------------------------------------------------------------------

    def _get_strategy_tools(self, phase: str) -> list[dict]:
        """获取策略生成阶段可用的工具列表。

        包含只读工具 + finish（策略专用版本）。

        Args:
            phase: 阶段标识，"ingest" 或 "retrieve"。

        Returns:
            工具列表。
        """
        # 直接使用只读工具列表
        read_only_tools = list(READ_ONLY_TOOLS)

        # 替换 finish 工具为多策略专用版本
        if phase == "ingest":
            finish_description = (
                "输出最终的多摄入策略文档并结束策略生成。\n\n"
                "**必须**在 `result` 参数中填写完整的多策略文档（Markdown 格式）。\n"
                "文档应包含 2~5 个不同的摄入策略方案，每个方案含决策表和伪代码。\n\n"
                "示例调用：finish(result=\"## 方案 1: ...\\n...\")"
            )
        else:
            finish_description = (
                "输出最终的多消费策略文档并结束策略生成。\n\n"
                "**必须**在 `result` 参数中填写完整的多策略文档（Markdown 格式）。\n"
                "文档应包含 2~5 个不同的消费策略方案，每个方案含检索路由决策表和伪代码。\n\n"
                "示例调用：finish(result=\"## 方案 1: ...\\n...\")"
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
                            "description": "完整的多策略文档内容（Markdown 格式），不可为空",
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
    # 辅助方法
    # ------------------------------------------------------------------

    def _collect_memory_state(self, user_id: str) -> dict[str, Any]:
        """收集当前记忆库状态信息。"""
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

    def _count_strategies(self, content: str) -> int:
        """统计策略文档中的方案数量。"""
        # 匹配 "## 方案 N" 或 "### 方案 N" 模式
        matches = re.findall(r"#{2,3}\s*方案\s*\d+", content)
        return len(matches) if matches else 1

    def _count_atomic_functions(self, content: str) -> int:
        """统计原子函数设计文档中的函数数量。"""
        # 匹配 "async def" 或 "def" 函数定义
        matches = re.findall(r"(?:async\s+)?def\s+\w+\s*\(", content)
        return len(matches)

    def _split_strategies(self, content: str) -> list[str]:
        """将多策略文档拆分为各个方案。

        Returns:
            各方案内容列表。
        """
        # 按 "## 方案" 或 "### 方案" 分割
        parts = re.split(r"(?=#{2,3}\s*方案\s*\d+)", content)
        strategies = [p.strip() for p in parts if p.strip() and re.match(r"#{2,3}\s*方案\s*\d+", p.strip())]

        if not strategies:
            # 降级：尝试按 "# 方案" 分割
            parts = re.split(r"(?=#+\s*方案\s*\d+)", content)
            strategies = [p.strip() for p in parts if p.strip() and re.match(r"#+\s*方案", p.strip())]

        if not strategies:
            # 最终降级：将整个文档作为一个方案
            if content.strip():
                strategies = [content.strip()]

        # 限制最多 5 个
        return strategies[:5]

    def _determine_final_status(self, biz_stats: GenerateCodeContextMultiCodeT2TaskStats) -> None:
        """判定最终任务状态。"""
        all_success = (
            biz_stats.multi_ingest_strategy_status == "success"
            and biz_stats.ingest_atomic_design_status == "success"
            and biz_stats.ingest_base_class_status == "success"
            and biz_stats.ingest_subclass_status == "success"
            and biz_stats.multi_retrieve_strategy_status == "success"
            and biz_stats.retrieve_atomic_design_status == "success"
            and biz_stats.retrieve_base_class_status == "success"
            and biz_stats.retrieve_subclass_status == "success"
        )

        if all_success:
            biz_stats.status = "success"
        else:
            biz_stats.status = "partial"
            failed_phases = []
            if biz_stats.multi_ingest_strategy_status != "success":
                failed_phases.append(f"multi_ingest_strategy: {biz_stats.multi_ingest_strategy_status}")
            if biz_stats.ingest_atomic_design_status != "success":
                failed_phases.append(f"ingest_atomic_design: {biz_stats.ingest_atomic_design_status}")
            if biz_stats.ingest_base_class_status != "success":
                failed_phases.append(f"ingest_base_class: {biz_stats.ingest_base_class_status}")
            if biz_stats.ingest_subclass_status != "success":
                failed_phases.append(f"ingest_subclass: {biz_stats.ingest_subclass_status}")
            if biz_stats.multi_retrieve_strategy_status != "success":
                failed_phases.append(f"multi_retrieve_strategy: {biz_stats.multi_retrieve_strategy_status}")
            if biz_stats.retrieve_atomic_design_status != "success":
                failed_phases.append(f"retrieve_atomic_design: {biz_stats.retrieve_atomic_design_status}")
            if biz_stats.retrieve_base_class_status != "success":
                failed_phases.append(f"retrieve_base_class: {biz_stats.retrieve_base_class_status}")
            if biz_stats.retrieve_subclass_status != "success":
                failed_phases.append(f"retrieve_subclass: {biz_stats.retrieve_subclass_status}")
            biz_stats.reason = ", ".join(failed_phases)
