"""Retrieve Context Code Task — 通过加载生成的消费代码类来检索记忆。

与 RetrieveContextTask 的区别：
- RetrieveContextTask 使用 agent loop（多轮 LLM tool-calling）进行检索
- RetrieveContextCodeTask 加载 GenerateCodeContextCodeT2Task
  生成的 Python 消费代码类，通过 importlib 动态加载并在进程内执行获取记忆信息

执行逻辑：
1. Phase 0: 前置检查消费代码文件是否存在
2. Phase 1: 加载并执行生成的消费代码类，获取 memory 信息

生成的消费代码路径（按优先级查找）：
- .codegen/retrieve_memory.py（GenerateCodeContextCodeT2Task 生成）

使用方式：
    task = RetrieveContextCodeTask(llm, fs_store, vec_store, graph_store)
    result = await task.execute(
        query="用户的编程语言偏好是什么？",
        session_id="session_001",
        user_id="user_001",
    )
    # result["retrieved_context"] 即为检索到的记忆上下文
"""
from __future__ import annotations

import inspect
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from context_task.base_context_task import BaseContextTask, TaskStats
from context_task.codegen.loader import load_consumer_class
from context_task.prompt.chinese_prompt import (
    _T2_QUERY_GUIDANCE_TEMPLATE,
)

if TYPE_CHECKING:
    from utils.memory_llm_interface import LLMInterface
    from storage.file_system_store import FileSystemStore
    from storage.stores_base import GraphStoreBase
    from storage.stores_base import VectorStoreBase
import logger.logger as logger


# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

# 消费代码的查找路径
RETRIEVE_CODE_PATHS = [
    ".codegen/retrieve_memory.py",   # GenerateCodeContextCodeT2Task 生成
]

# 脚本执行超时时间（秒）



# ---------------------------------------------------------------------------
# 业务统计数据结构
# ---------------------------------------------------------------------------


@dataclass
class RetrieveContextCodeTaskStats(TaskStats):
    """RetrieveContextCodeTask 的任务级业务统计数据。

    继承 TaskStats 获得 LLM/工具通用统计字段，
    同时扩展 retrieve code 任务特有的业务指标。
    """

    # ── 任务结果状态 ──
    status: str = ""
    """任务执行状态，取值: 'success' | 'skipped' | 'empty' | 'error'"""

    reason: str = ""
    """失败或空结果的原因"""

    # ── 上下文标识 ──
    user_id: str = ""
    """用户 ID"""

    session_id: str = ""
    """当前会话 ID"""

    query: str = ""
    """检索查询文本"""

    # ── 代码执行统计 ──
    code_path: str = ""
    """实际执行的消费代码路径"""

    execute_latency_ms: float = 0.0
    """代码执行耗时（毫秒）"""

    execute_returncode: int = -1
    """代码执行的返回码"""

    execute_stderr: str = ""
    """代码执行的 stderr 输出（截断）"""

    # ── 检索结果 ──
    retrieved_context: str = ""
    """最终检索到的上下文内容（stdout 输出）"""

    retrieved_context_length: int = 0
    """检索上下文内容的长度"""

    query_memory: str = ""
    """完整的记忆上下文（带 prompt 解释）"""

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典，合并基类统计和业务统计。"""
        base = super().to_dict()
        base.update({
            "status": self.status,
            "reason": self.reason,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "query": self.query,
            "code_path": self.code_path,
            "execute_latency_ms": round(self.execute_latency_ms, 1),
            "execute_returncode": self.execute_returncode,
            "execute_stderr": self.execute_stderr,
            "retrieved_context": self.retrieved_context,
            "retrieved_context_length": self.retrieved_context_length,
            "query_memory": self.query_memory,
        })
        return base


# ---------------------------------------------------------------------------
# RetrieveContextCodeTask
# ---------------------------------------------------------------------------


class RetrieveContextCodeTask(BaseContextTask):
    """通过执行生成的消费代码脚本来检索记忆的 ContextTask。

    整合以下逻辑：
    1. **Phase 0 - 前置检查**：检查消费代码文件是否存在
    2. **Phase 1 - 执行消费代码**：加载并执行生成的 Python 消费代码类，
       获取记忆检索结果

    生命周期：
        execute(query, session_id, user_id) →
            pre_run: 初始化统计
            run:
                Phase 0: 前置检查（代码文件存在性）
                Phase 1: 执行消费代码脚本
            post_run: 记录完成日志
    """

    task_name = "retrieve_context_code"

    def __init__(
        self,
        llm: "LLMInterface",
        fs_store: "FileSystemStore",
        vec_store: "VectorStoreBase",
        graph_store: "GraphStoreBase",
        **kwargs: Any,
    ):
        """初始化 RetrieveContextCodeTask。

        Args:
            llm: LLM 接口实例。
            fs_store: 文件系统存储后端。
            vec_store: 向量数据库存储后端。
            graph_store: 图数据库存储后端。
            **kwargs: 兼容工厂方法传入的额外参数（如 max_turns），本类不使用。
        """
        super().__init__(llm)
        self.fs = fs_store
        self.vec = vec_store
        self.graph = graph_store

    # ------------------------------------------------------------------
    # 生命周期实现
    # ------------------------------------------------------------------

    async def pre_run(self, query: str = "", session_id: str = "", **kwargs: Any) -> None:
        """前置处理：初始化业务统计。"""
        _start_time = getattr(self._stats, "start_time", 0.0) if hasattr(self, "_stats") else 0.0
        self._stats = RetrieveContextCodeTaskStats(task_name=self.task_name)
        self._stats.start_time = _start_time

    async def run(
        self,
        query: str = "",
        session_id: str = "",
        user_id: str = "default_user",
        messages: list[dict[str, Any]] | None = None,
        latest_memory: str = "",
        **kwargs: Any,
    ) -> None:
        """核心逻辑：前置检查 → 执行消费代码脚本。

        Args:
            query: 检索查询文本（必需）。
            session_id: 当前 session ID。
            user_id: 用户 ID。
            messages: 当前对话的完整消息列表（可选，传递给消费代码脚本）。
            latest_memory: 上一轮召回的最新记忆信息（可选，保留接口兼容性）。
        """
        biz_stats: RetrieveContextCodeTaskStats = self._stats  # type: ignore[assignment]
        biz_stats.user_id = user_id
        biz_stats.session_id = session_id
        biz_stats.query = query

        if not query:
            biz_stats.status = "empty"
            biz_stats.reason = "empty query"
            return

        logger.info(
            "RetrieveContextCodeTask: run start "
            "(query_len=%d, user_id=%s, session_id=%s)",
            len(query), user_id, session_id,
        )

        # ── Phase 0: 前置检查消费代码文件是否存在 ──
        code_rel_path = self._find_retrieve_code()
        if not code_rel_path:
            biz_stats.status = "skipped"
            biz_stats.reason = (
                "消费代码脚本不存在，请先执行 GenerateCodeContextCodeT2Task 生成代码"
            )
            logger.info(
                "RetrieveContextCodeTask: Phase 0 - skipped (消费代码文件不存在)",
            )
            return

        logger.info(
            "RetrieveContextCodeTask: Phase 0 - 消费代码文件存在: %s",
            code_rel_path,
        )

        # ── Phase 1: 执行消费代码脚本 ──
        logger.info("RetrieveContextCodeTask: Phase 1 - 执行消费代码脚本")
        await self._execute_retrieve_code(
            query=query,
            session_id=session_id,
            user_id=user_id,
            messages=messages,
            biz_stats=biz_stats,
        )

        # 拼接 query_memory（参考 RetrieveContextTask 的拼接方式）
        context = biz_stats.retrieved_context
        if context:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            memory_body = context.strip() or "(No relevant memories found)"
            t2_schema_notes = _T2_QUERY_GUIDANCE_TEMPLATE.format(now=now)
            biz_stats.query_memory = (
                f"# Memory Context\n\n"
                f"{memory_body}\n\n"
                f"---\n\n"
                f"{t2_schema_notes}\n"
            )

        logger.info(
            "RetrieveContextCodeTask: run completed "
            "(status=%s, context_len=%d)",
            biz_stats.status, biz_stats.retrieved_context_length,
        )

    async def post_run(self, *, session_id: str = "", **kwargs: Any) -> None:
        """后置处理：记录完成日志。"""
        biz_stats: RetrieveContextCodeTaskStats = self._stats  # type: ignore[assignment]
        logger.info(
            "RetrieveContextCodeTask: completed, "
            "status=%s, code_path=%s, context_len=%d",
            biz_stats.status,
            biz_stats.code_path,
            biz_stats.retrieved_context_length,
        )

    # ------------------------------------------------------------------
    # Phase 2: 执行消费代码（importlib 动态加载）
    # ------------------------------------------------------------------

    async def _execute_retrieve_code(
        self,
        query: str,
        session_id: str,
        user_id: str,
        messages: list[dict[str, Any]] | None,
        biz_stats: RetrieveContextCodeTaskStats,
    ) -> None:
        """加载并执行消费代码，通过 importlib 动态加载 BaseMemoryConsumer 子类。

        按优先级查找消费代码文件：
        1. .codegen/retrieve_memory.py（T2 生成）
        2. .codegen/consume_memory.py（Code 生成）

        Args:
            query: 用户查询文本。
            session_id: 当前 session ID。
            user_id: 用户 ID。
            messages: 当前对话消息列表。
            biz_stats: 业务统计实例。
        """
        # 1. 查找可用的消费代码文件
        code_rel_path = self._find_retrieve_code()
        if not code_rel_path:
            biz_stats.status = "error"
            biz_stats.reason = (
                "未找到消费代码脚本，请先执行 GenerateCodeContextCodeT2Task 生成代码"
            )
            logger.warning(
                "RetrieveContextCodeTask: Phase 2 - 未找到消费代码脚本"
            )
            return

        biz_stats.code_path = code_rel_path
        code_abs_path = f"{self.fs.base_path}/{code_rel_path}"

        logger.info(
            "RetrieveContextCodeTask: Phase 2 - 加载消费代码: %s",
            code_rel_path,
        )

        start = time.monotonic()
        try:
            # 2. 动态加载子类
            ConsumerClass = load_consumer_class(code_abs_path)

            # 3. 实例化（直接传入已有的 Store 引用，无需序列化）
            consumer = ConsumerClass(
                fs=self.fs,
                vec=self.vec,
                graph=self.graph,
                llm=self.llm,
            )

            # 4. 执行（防御性参数过滤：兼容签名不完整的子类）
            call_kwargs = dict(
                query=query,
                messages=messages or [],
                user_id=user_id,
                session_id=session_id,
            )
            try:
                sig = inspect.signature(consumer.retrieve_memory)
                accepted = set(sig.parameters.keys()) - {"self"}
                call_kwargs = {k: v for k, v in call_kwargs.items() if k in accepted}
            except (ValueError, TypeError):
                pass
            result = await consumer.retrieve_memory(**call_kwargs)

            biz_stats.execute_latency_ms = (time.monotonic() - start) * 1000
            biz_stats.execute_returncode = 0

            # 捕获结果
            if result:
                biz_stats.retrieved_context = result
                biz_stats.retrieved_context_length = len(result)
                biz_stats.status = "success"
                logger.info(
                    "RetrieveContextCodeTask: Phase 2 - 检索成功 "
                    "(context_len=%d, latency=%.1fms)",
                    len(result), biz_stats.execute_latency_ms,
                )
            else:
                biz_stats.status = "empty"
                biz_stats.reason = "消费代码执行成功但无输出"
                logger.info(
                    "RetrieveContextCodeTask: Phase 2 - 执行成功但无输出"
                )

        except FileNotFoundError as e:
            biz_stats.execute_latency_ms = (time.monotonic() - start) * 1000
            biz_stats.status = "error"
            biz_stats.reason = f"消费代码文件不存在: {e}"
            logger.error(
                "RetrieveContextCodeTask: Phase 2 - 消费代码文件不存在: %s", e,
            )

        except Exception as e:
            biz_stats.execute_latency_ms = (time.monotonic() - start) * 1000
            biz_stats.status = "error"
            biz_stats.reason = f"消费代码执行异常: {e}"
            biz_stats.execute_stderr = traceback.format_exc()[:2000]
            logger.error(
                "RetrieveContextCodeTask: Phase 2 - 执行失败: %s (code_path=%s)",
                e, code_abs_path,
            )


    def _find_retrieve_code(self) -> str:
        """按优先级查找可用的消费代码文件。

        查找路径：
        - .codegen/retrieve_memory.py（GenerateCodeContextCodeT2Task 生成）

        Returns:
            找到的代码文件相对路径，未找到返回空字符串。
        """
        for rel_path in RETRIEVE_CODE_PATHS:
            content = self.fs.read_file(rel_path)
            if not content.startswith("ERROR"):
                return rel_path
        return ""

    # ------------------------------------------------------------------
    # 便捷方法：query_memory
    # ------------------------------------------------------------------

    async def query_memory(
        self,
        query: str,
        session_id: str = "",
        user_id: str = "default_user",
        messages: list[dict[str, Any]] | None = None,
    ) -> str:
        """查询记忆，返回检索到的上下文字符串。

        对齐 RetrieveContextTask.query_memory 的接口语义，
        便于上层系统直接替换调用。

        Args:
            query: 查询文本。
            session_id: 当前 session ID。
            user_id: 用户 ID。
            messages: 当前对话消息列表（可选）。

        Returns:
            检索到的记忆上下文字符串，无结果时返回空字符串。
        """
        logger.info("RetrieveContextCodeTask.query_memory: query=%s...", query[:80])
        result = await self.execute(
            query=query,
            session_id=session_id,
            user_id=user_id,
            messages=messages,
        )
        context = result.get("retrieved_context", "")
        if context:
            logger.info(
                "RetrieveContextCodeTask.query_memory: retrieved %d chars",
                len(context),
            )
        else:
            logger.info(
                "RetrieveContextCodeTask.query_memory: no relevant context found"
            )
        return context
