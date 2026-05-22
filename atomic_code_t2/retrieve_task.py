"""Atomic_Code_T2 - Retrieve Task。

设计：**整目录复制 + 包动态加载**，每个用户在记忆库下拥有完全独立、可演进的副本。

执行流程：
1. 启动时确保 `<fs.base_path>/.codegen/retrieve/` 目录存在且内容齐全。
   不存在则把项目内置模板 `atomic_code_t2/retrieve/templates/` 整目录复制过去。
2. 通过 `loader.load_plan_class(retrieve_dir, plan_module)` 把该目录当作包动态加载，
   找到 BasePlan 子类。
3. 实例化后调 `plan.run(query, ...)`，返回 memory context。

文件布局（项目侧）：
    atomic_code_t2/retrieve/
    ├── base_plan.py             # 稳定接口（留在项目，不复制）
    ├── loader.py                # 加载器
    └── templates/               # 模板目录，整目录复制
        ├── __init__.py
        ├── atomic.py
        ├── prompt.py
        └── plans/
            ├── __init__.py
            └── default_plan.py

文件布局（用户副本）：
    <fs.base_path>/.codegen/retrieve/
    ├── __init__.py
    ├── atomic.py                # ← 用户级可改
    ├── prompt.py                # ← 用户级可改
    └── plans/
        ├── __init__.py
        └── default_plan.py      # ← 用户级可改 / LLM 可在 plans/ 下加新方案

隔离策略：副本一旦生成就完全独立，项目模板的后续更新**不会**同步到用户副本。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import logger.logger as logger

from context_task.base_context_task import BaseContextTask, TaskStats
from atomic_code_t2.prompt.retrieve_guidance import ATOMIC_T2_QUERY_GUIDANCE_TEMPLATE
from storage.file_system_store import CODEGEN_DIR, FileSystemStore
from storage.stores_base import GraphStoreBase, VectorStoreBase
from utils.memory_llm_interface import LLMInterface

from .retrieve import load_plan_class


# ---------------------------------------------------------------------------
# 业务统计数据结构
# ---------------------------------------------------------------------------


@dataclass
class RetrieveAtomicCodeT2Stats(TaskStats):
    """RetrieveContextAtomicCodeT2Task 的任务级业务统计。

    继承 TaskStats 获得 LLM / 工具 / 存储访问的通用统计字段，
    再扩展业务字段（最终检索到的 context、所用 plan、状态等）。

    必须覆写 to_dict() 合并业务字段，否则 BaseContextTask.execute() 返回的
    结果只有通用统计、没有 retrieved_context，调用方拿不到记忆。
    """

    # ── 任务结果状态 ──
    status: str = ""
    """'success' | 'empty' | 'error'"""

    reason: str = ""
    """失败或空结果的原因"""

    # ── 上下文标识 ──
    user_id: str = ""
    session_id: str = ""
    query: str = ""

    # ── Plan 标识 ──
    plan_name: str = ""
    """实际执行的 plan 名称"""

    plan_path: str = ""
    """plan 加载目录（用户副本路径）"""

    # ── 检索结果 ──
    retrieved_context: str = ""
    """最终检索到的 memory context 字符串"""

    retrieved_context_length: int = 0
    """检索 context 的字符数"""

    query_memory: str = ""
    """完整的记忆上下文（带 prompt 解释，由 llm_hook_handler 注入到 system message）。

    格式与 RetrieveContextTask / RetrieveContextMultiCodeTask 保持一致：
        # Memory Context
        <retrieved_context>
        ---
        <T2 schema notes>
    """

    def to_dict(self) -> dict[str, Any]:
        """合并基类通用统计 + 业务字段。"""
        base = super().to_dict()
        base.update({
            "status": self.status,
            "reason": self.reason,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "query": self.query,
            "plan_name": self.plan_name,
            "plan_path": self.plan_path,
            "retrieved_context": self.retrieved_context,
            "retrieved_context_length": self.retrieved_context_length,
            "query_memory": self.query_memory,
        })
        return base


# ---------------------------------------------------------------------------
# 项目内置模板路径
# ---------------------------------------------------------------------------
_PROJECT_RETRIEVE_DIR = Path(__file__).resolve().parent / "retrieve"
_TEMPLATE_DIR = _PROJECT_RETRIEVE_DIR / "templates"

# `<fs.base_path>/.codegen/<_USER_PLANS_SUBDIR>/` 是用户副本根。
# 走子目录（`retrieve/`）而不是与 Multi_Code 的 `retrieve_memory_v*.py` 平铺，
# 是为了避免命名空间冲突。
_USER_PLANS_SUBDIR = "retrieve"

# 当前阶段固定使用的方案模块名（点号分隔，相对用户副本根）
_DEFAULT_PLAN_MODULE = "plans.default_plan"


class RetrieveContextAtomicCodeT2Task(BaseContextTask):
    """Atomic_Code_T2 模式下的检索任务。

    走目录动态加载：从 `<fs.base_path>/.codegen/retrieve/plans/<plan>.py` 加载
    BasePlan 子类（首次运行时从项目内置模板整目录复制初始化），
    实例化后执行 `plan.run(query, ...)`。
    """

    task_name = "retrieve_context_atomic_code_t2"

    def __init__(
        self,
        llm: LLMInterface,
        fs_store: FileSystemStore,
        vec_store: VectorStoreBase,
        graph_store: GraphStoreBase,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm)
        self.fs = fs_store
        self.vec = vec_store
        self.graph = graph_store
        self.extra_kwargs = kwargs

    async def pre_run(
        self,
        query: str = "",
        session_id: str = "",
        user_id: str = "default_user",
        **kwargs: Any,
    ) -> None:
        """初始化业务统计：用 RetrieveAtomicCodeT2Stats 替代基类的 TaskStats。

        保留基类 execute() 已写好的 start_time，避免 latency 失真。
        """
        _start_time = getattr(self._stats, "start_time", 0.0) if hasattr(self, "_stats") else 0.0
        self._stats = RetrieveAtomicCodeT2Stats(task_name=self.task_name)
        self._stats.start_time = _start_time
        self._stats.user_id = user_id
        self._stats.session_id = session_id
        self._stats.query = query

    async def run(
        self,
        query: str = "",
        session_id: str = "",
        user_id: str = "default_user",
        **kwargs: Any,
    ) -> None:
        """执行检索：确保用户副本存在 → 动态加载 → 实例化 → 调 plan.run。

        所有业务结果写入 `self._stats`（RetrieveAtomicCodeT2Stats），由基类
        execute() 末尾的 `self._stats.to_dict()` 自动合并到返回值里，
        调用方因此能从 `result["retrieved_context"]` 拿到记忆。
        """
        stats: RetrieveAtomicCodeT2Stats = self._stats  # type: ignore[assignment]

        if not query or not query.strip():
            logger.warning(
                "[%s] run: 空 query，跳过 (user_id=%s, session_id=%s)",
                self.task_name, user_id, session_id,
            )
            stats.status = "empty"
            stats.reason = "empty query"
            return

        session_time = kwargs.get("session_time") or datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S, %a"
        )

        # 1. 确保用户副本目录存在
        retrieve_dir = self._ensure_user_retrieve_dir()
        stats.plan_path = str(retrieve_dir)

        # 2. 动态加载 plan
        try:
            plan_cls = load_plan_class(retrieve_dir, plan_module=_DEFAULT_PLAN_MODULE)
        except Exception as e:
            logger.error(
                "[%s] 加载 plan 失败: %s (retrieve_dir=%s, module=%s)",
                self.task_name, e, retrieve_dir, _DEFAULT_PLAN_MODULE,
            )
            stats.status = "error"
            stats.reason = f"load plan failed: {e}"
            return

        stats.plan_name = getattr(plan_cls, "plan_name", plan_cls.__name__)

        # 3. 实例化 + 执行
        plan = plan_cls(self.fs, self.vec, self.graph, self.llm)
        try:
            context = await plan.run(query=query, session_time=session_time)
        except Exception as e:
            logger.error(
                "[%s] plan.run 执行失败: %s (plan=%s)",
                self.task_name, e, stats.plan_name,
            )
            stats.status = "error"
            stats.reason = f"plan.run failed: {e}"
            return

        stats.retrieved_context = context or ""
        stats.retrieved_context_length = len(stats.retrieved_context)
        stats.status = "success" if context else "empty"
        stats.reason = "" if context else "plan returned empty context"

        # 拼装 query_memory（带 schema 注释的完整记忆上下文）
        # llm_hook_handler 会从 result["query_memory"] 取出来注入到 system message。
        if stats.retrieved_context:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            memory_body = stats.retrieved_context.strip() or "(No relevant memories found)"
            t2_schema_notes = ATOMIC_T2_QUERY_GUIDANCE_TEMPLATE.format(now=now)
            stats.query_memory = (
                f"# Memory Context\n\n"
                f"{memory_body}\n\n"
                f"---\n\n"
                f"{t2_schema_notes}\n"
            )

        logger.info(
            "[%s] retrieve done (plan=%s, context_len=%d, query_memory_len=%d, retrieve_dir=%s)",
            self.task_name,
            stats.plan_name,
            stats.retrieved_context_length,
            len(stats.query_memory),
            retrieve_dir,
        )

    # ------------------------------------------------------------------
    # 内部：用户副本目录管理
    # ------------------------------------------------------------------

    def _ensure_user_retrieve_dir(self) -> Path:
        """确保 `<fs.base_path>/{CODEGEN_DIR}/{_USER_PLANS_SUBDIR}/` 存在且齐全；
        首次运行时把项目内置模板 `templates/` 整目录复制过去。

        策略（按你和我商量过的"用户隔离"）：**已存在则不覆盖**。
        模板若有更新，由用户/调用方主动删除该目录后再触发本任务来重建。

        Returns:
            用户副本目录的绝对路径。

        Raises:
            FileNotFoundError: 项目模板目录都不存在（说明项目代码出问题了）。
        """
        target_dir = Path(self.fs.base_path) / CODEGEN_DIR / _USER_PLANS_SUBDIR

        if target_dir.is_dir() and (target_dir / "__init__.py").is_file():
            # 已存在并看起来是有效包 → 直接用
            return target_dir

        if not _TEMPLATE_DIR.is_dir():
            raise FileNotFoundError(
                f"项目内置模板目录不存在: {_TEMPLATE_DIR}（项目代码异常）"
            )

        # 整目录复制（拒绝目标已存在的"半残"情况——先清掉再复制）
        if target_dir.exists():
            logger.warning(
                "[%s] 用户副本目录存在但缺少 __init__.py，重建: %s",
                self.task_name, target_dir,
            )
            shutil.rmtree(target_dir)

        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(_TEMPLATE_DIR, target_dir)
        logger.info(
            "[%s] 已从内置模板初始化用户 retrieve 副本: %s -> %s",
            self.task_name, _TEMPLATE_DIR, target_dir,
        )
        return target_dir

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    async def query_memory(
        self,
        query: str,
        session_id: str = "",
        user_id: str = "default_user",
        **kwargs: Any,
    ) -> str:
        """检索并返回 memory context 字符串（无结果返回空串）。

        execute() 返回的是 `self._stats.to_dict()`，里面带 retrieved_context。
        """
        result = await self.execute(
            query=query,
            session_id=session_id,
            user_id=user_id,
            **kwargs,
        )
        return result.get("retrieved_context", "") if isinstance(result, dict) else ""
