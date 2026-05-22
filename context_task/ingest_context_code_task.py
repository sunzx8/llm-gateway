"""
Ingest Context Code Task — 通过加载生成的摄入代码类来摄入记忆。

与 IngestContextTask 的区别：
- IngestContextTask 使用 agent loop（多轮 LLM tool-calling）进行摄入
- IngestContextCodeTask 加载 GenerateCodeContextCodeT2Task 生成的 Python 摄入代码类，
  通过 importlib 动态加载并在进程内执行完成记忆摄入

执行逻辑：
1. 加载 GenerateCodeContextCodeT2Task 生成的摄入代码类（.codegen/ingest_memory.py）
2. 实例化并调用 ingest_memory 方法

生成的摄入代码路径：
- .codegen/ingest_memory.py（GenerateCodeContextCodeT2Task 生成）

使用方式：
    task = IngestContextCodeTask(llm, fs_store, vec_store, graph_store)
    result = await task.execute(
        session_id="session_001",
        user_id="user_001",
        messages=[{"role": "user", "content": "..."}, ...],
    )
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from context_task.base_context_task import BaseContextTask, TaskStats
from context_task.codegen.loader import load_ingestor_class

if TYPE_CHECKING:
    from utils.memory_llm_interface import LLMInterface
    from storage.file_system_store import FileSystemStore
    from storage.stores_base import GraphStoreBase
    from storage.stores_base import VectorStoreBase
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

# 摄入代码的固定路径
INGEST_CODE_PATH = ".codegen/ingest_memory.py"

# 脚本执行超时时间（秒）



# ---------------------------------------------------------------------------
# 业务统计数据结构
# ---------------------------------------------------------------------------


@dataclass
class IngestContextCodeTaskStats(TaskStats):
    """IngestContextCodeTask 的任务级业务统计数据。

    继承 TaskStats 获得 LLM/工具通用统计字段，
    同时扩展 ingest code 任务特有的业务指标。
    """

    # ── 任务结果状态 ──
    status: str = ""
    """任务执行状态，取值: 'success' | 'noop' | 'error'"""

    reason: str = ""
    """失败或跳过的原因"""

    # ── 上下文标识 ──
    user_id: str = ""
    """用户 ID"""

    session_id: str = ""
    """当前会话 ID"""

    # ── 输入统计 ──
    messages_count: int = 0
    """本次摄入处理的 message 数量"""

    # ── 代码执行统计 ──
    code_path: str = ""
    """实际执行的摄入代码路径"""

    execute_latency_ms: float = 0.0
    """代码执行耗时（毫秒）"""

    execute_returncode: int = -1
    """代码执行的返回码"""

    execute_stdout: str = ""
    """代码执行的 stdout 输出（截断）"""

    execute_stderr: str = ""
    """代码执行的 stderr 输出（截断）"""

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典，合并基类统计和业务统计。"""
        base = super().to_dict()
        base.update({
            "status": self.status,
            "reason": self.reason,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "messages_count": self.messages_count,
            "code_path": self.code_path,
            "execute_latency_ms": round(self.execute_latency_ms, 1),
            "execute_returncode": self.execute_returncode,
            "execute_stdout": self.execute_stdout,
            "execute_stderr": self.execute_stderr,
        })
        return base


# ---------------------------------------------------------------------------
# IngestContextCodeTask
# ---------------------------------------------------------------------------


class IngestContextCodeTask(BaseContextTask):
    """通过加载生成的摄入代码类来摄入记忆的 ContextTask。

    执行逻辑：
    1. 检查摄入代码文件是否存在
    2. 通过 importlib 动态加载 BaseMemoryIngestor 子类
    3. 实例化并调用 ingest_memory 方法

    生命周期：
        execute(session_id, messages, user_id) →
            pre_run: 初始化统计
            run: 加载并执行摄入代码
            post_run: 记录完成日志
    """

    task_name = "ingest_context_code"

    def __init__(
        self,
        llm: "LLMInterface",
        fs_store: "FileSystemStore",
        vec_store: "VectorStoreBase",
        graph_store: "GraphStoreBase",
        **kwargs: Any,
    ):
        """初始化 IngestContextCodeTask。

        Args:
            llm: LLM 接口实例。
            fs_store: 文件系统存储后端。
            vec_store: 向量数据库存储后端。
            graph_store: 图数据库存储后端。
            **kwargs: 兼容工厂方法传入的额外参数（如 max_turns），本类不使用。
        """
        super().__init__(llm)
        self.fs: FileSystemStore = fs_store
        self.vec: VectorStoreBase = vec_store
        self.graph: GraphStoreBase = graph_store

    # ------------------------------------------------------------------
    # 生命周期实现
    # ------------------------------------------------------------------

    async def pre_run(self, session_id: str = "", messages: list[dict[str, str]] = None, **kwargs: Any) -> None:
        """前置处理：初始化业务统计。"""
        _start_time = getattr(self._stats, "start_time", 0.0) if hasattr(self, "_stats") else 0.0
        self._stats = IngestContextCodeTaskStats(task_name=self.task_name)
        self._stats.start_time = _start_time

    async def run(
        self,
        session_id: str = "",
        messages: list[dict[str, str]] = None,
        user_id: str = "default_user",
        **kwargs: Any,
    ) -> None:
        """核心逻辑：加载并执行摄入代码。

        Args:
            session_id: 会话 ID。
            messages: 消息列表 [{"role": ..., "content": ...}]。
            user_id: 用户 ID（默认 "default_user"）。
        """
        messages = messages or []

        biz_stats: IngestContextCodeTaskStats = self._stats  # type: ignore[assignment]
        biz_stats.user_id = user_id
        biz_stats.session_id = session_id
        biz_stats.messages_count = len(messages)

        if not messages:
            biz_stats.status = "noop"
            biz_stats.reason = "empty messages"
            logger.info("IngestContextCodeTask: noop - empty messages")
            return

        logger.info(
            "IngestContextCodeTask: run start "
            "(session_id=%s, user_id=%s, messages=%d)",
            session_id, user_id, len(messages),
        )

        # 检查摄入代码文件是否存在
        code_content = self.fs.read_file(INGEST_CODE_PATH)
        if code_content.startswith("ERROR"):
            biz_stats.status = "error"
            biz_stats.reason = (
                "未找到摄入代码脚本，请先执行 GenerateCodeContextCodeT2Task 生成代码"
            )
            logger.warning(
                "IngestContextCodeTask: 未找到摄入代码脚本: %s",
                INGEST_CODE_PATH,
            )
            return

        biz_stats.code_path = INGEST_CODE_PATH

        # 执行摄入代码
        await self._execute_ingest_code(
            session_id=session_id,
            user_id=user_id,
            messages=messages,
            biz_stats=biz_stats,
        )

        logger.info(
            "IngestContextCodeTask: run completed "
            "(status=%s, latency=%.1fms)",
            biz_stats.status, biz_stats.execute_latency_ms,
        )

    async def post_run(self, *, session_id: str = "", **kwargs: Any) -> None:
        """后置处理：记录完成日志。"""
        biz_stats: IngestContextCodeTaskStats = self._stats  # type: ignore[assignment]
        logger.info(
            "IngestContextCodeTask: completed, "
            "status=%s, code_path=%s, messages_count=%d, latency=%.1fms",
            biz_stats.status,
            biz_stats.code_path,
            biz_stats.messages_count,
            biz_stats.execute_latency_ms,
        )

    # ------------------------------------------------------------------
    # 执行摄入代码（importlib 动态加载）
    # ------------------------------------------------------------------

    async def _execute_ingest_code(
        self,
        session_id: str,
        user_id: str,
        messages: list[dict[str, str]],
        biz_stats: IngestContextCodeTaskStats,
    ) -> None:
        """加载并执行摄入代码，通过 importlib 动态加载 BaseMemoryIngestor 子类。

        Args:
            session_id: 当前 session ID。
            user_id: 用户 ID。
            messages: 当前对话消息列表。
            biz_stats: 业务统计实例。
        """
        code_abs_path = f"{self.fs.base_path}/{INGEST_CODE_PATH}"

        logger.info(
            "IngestContextCodeTask: 加载摄入代码: %s",
            INGEST_CODE_PATH,
        )

        start = time.monotonic()
        try:
            # 1. 动态加载子类
            IngestorClass = load_ingestor_class(code_abs_path)

            # 2. 实例化（直接传入已有的 Store 引用，无需序列化）
            ingestor = IngestorClass(
                fs=self.fs,
                vec=self.vec,
                graph=self.graph,
                llm=self.llm,
                memory_base=self.fs.base_path,
            )

            # 3. 执行
            result = await ingestor.ingest_memory(
                messages=messages,
                user_id=user_id,
                session_id=session_id,
            )

            biz_stats.execute_latency_ms = (time.monotonic() - start) * 1000
            biz_stats.execute_returncode = 0
            biz_stats.execute_stdout = (result or "")[:2000]
            biz_stats.status = "success"

            logger.info(
                "IngestContextCodeTask: 摄入成功 "
                "(latency=%.1fms, result_len=%d)",
                biz_stats.execute_latency_ms, len(result or ""),
            )

        except FileNotFoundError as e:
            biz_stats.execute_latency_ms = (time.monotonic() - start) * 1000
            biz_stats.status = "error"
            biz_stats.reason = f"摄入代码文件不存在: {e}"
            logger.error(
                "IngestContextCodeTask: 摄入代码文件不存在: %s", e,
            )

        except Exception as e:
            biz_stats.execute_latency_ms = (time.monotonic() - start) * 1000
            biz_stats.status = "error"
            biz_stats.reason = f"摄入代码执行异常: {e}"
            biz_stats.execute_stderr = traceback.format_exc()[:2000]
            logger.error(
                "IngestContextCodeTask: 摄入代码执行异常: %s", e,
            )
