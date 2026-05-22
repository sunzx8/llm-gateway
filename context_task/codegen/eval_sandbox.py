"""代码评测沙箱 — 在隔离的记忆库副本上执行代码。

核心设计：
- 复制当前用户的记忆库到隔离环境（不污染真实数据）
- Memory 后端：deepcopy 内存数据
- PG 后端：创建临时 schema 复制数据
- FS 后端：copytree 到临时目录
- 执行完毕后自动清理隔离环境

使用方式：
    sandbox = EvalSandbox(fs, vec, graph, llm, user_id)
    await sandbox.setup()
    try:
        result = await sandbox.run_ingest(code, messages, user_id, session_id)
        # 或
        result = await sandbox.run_retrieve(code, query, messages, user_id, session_id)
    finally:
        await sandbox.teardown()
"""

from __future__ import annotations

import asyncio
import copy
import io
import os
import shutil
import sys
import tempfile
import time
import traceback
import uuid
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

import logger.logger as logger

if TYPE_CHECKING:
    from storage.file_system_store import FileSystemStore
    from storage.stores_base import VectorStoreBase, GraphStoreBase
    from utils.memory_llm_interface import LLMInterface


# ---------------------------------------------------------------------------
# 评测结果数据结构
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """代码评测执行结果。"""

    success: bool = False
    """代码是否执行成功（无异常）"""

    return_value: str = ""
    """代码执行的返回值（字符串化）"""

    stdout: str = ""
    """标准输出内容"""

    stderr: str = ""
    """标准错误内容"""

    error: str = ""
    """异常信息（如果执行失败）"""

    execution_time_ms: int = 0
    """执行耗时（毫秒）"""

    sandbox_changes: dict[str, Any] = field(default_factory=dict)
    """沙箱中的变更统计"""

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "success": self.success,
            "return_value": self.return_value,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "sandbox_changes": self.sandbox_changes,
        }

    def to_tool_response(self) -> str:
        """格式化为工具调用的返回字符串。"""
        parts = []
        if self.success:
            parts.append("✅ 代码执行成功")
        else:
            parts.append("❌ 代码执行失败")

        if self.return_value:
            parts.append(f"\n【返回值】\n{self.return_value}")

        if self.stdout:
            parts.append(f"\n【标准输出】\n{self.stdout[:2000]}")

        if self.stderr:
            parts.append(f"\n【标准错误】\n{self.stderr[:2000]}")

        if self.error:
            parts.append(f"\n【异常信息】\n{self.error[:2000]}")

        parts.append(f"\n【执行耗时】{self.execution_time_ms}ms")

        if self.sandbox_changes:
            changes_parts = []
            if self.sandbox_changes.get("fs_files_written"):
                changes_parts.append(
                    f"  文件写入: {self.sandbox_changes['fs_files_written']}"
                )
            if self.sandbox_changes.get("vec_entries_added", 0) > 0:
                changes_parts.append(
                    f"  向量条目新增: {self.sandbox_changes['vec_entries_added']}"
                )
            if self.sandbox_changes.get("graph_nodes_added", 0) > 0:
                changes_parts.append(
                    f"  图节点新增: {self.sandbox_changes['graph_nodes_added']}"
                )
            if self.sandbox_changes.get("graph_edges_added", 0) > 0:
                changes_parts.append(
                    f"  图边新增: {self.sandbox_changes['graph_edges_added']}"
                )
            if changes_parts:
                parts.append("\n【沙箱变更】\n" + "\n".join(changes_parts))

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# 评测沙箱
# ---------------------------------------------------------------------------

# 单次 eval_code 执行超时（秒）
EVAL_TIMEOUT_S = 30


class EvalSandbox:
    """代码评测沙箱 — 在隔离的记忆库副本上执行代码。

    生命周期：
        1. setup() — 复制记忆库到隔离环境
        2. run_ingest() / run_retrieve() — 在沙箱中执行代码
        3. teardown() — 清理隔离环境
    """

    def __init__(
        self,
        fs: "FileSystemStore",
        vec: "VectorStoreBase",
        graph: "GraphStoreBase",
        llm: "LLMInterface",
        user_id: str,
    ):
        """初始化评测沙箱。

        Args:
            fs: 原始 FileSystemStore 实例。
            vec: 原始 VectorStoreBase 实例。
            graph: 原始 GraphStoreBase 实例。
            llm: LLM 接口（共享，不复制）。
            user_id: 用户 ID。
        """
        self._fs = fs
        self._vec = vec
        self._graph = graph
        self._llm = llm
        self._user_id = user_id

        # 沙箱实例（setup 后填充）
        self._sandbox_fs: "FileSystemStore | None" = None
        self._sandbox_vec: "VectorStoreBase | None" = None
        self._sandbox_graph: "GraphStoreBase | None" = None

        # 临时资源（teardown 时清理）
        self._temp_dir: str | None = None
        self._temp_schema: str | None = None
        self._sandbox_id: str = uuid.uuid4().hex[:12]

        # 执行前的快照统计（用于计算变更）
        self._pre_vec_stats: dict[str, Any] = {}
        self._pre_graph_stats: dict[str, Any] = {}
        self._pre_fs_files: set[str] = set()

    async def setup(self) -> None:
        """创建隔离环境（复制记忆库）。

        根据存储后端类型选择不同的复制策略：
        - Memory 后端：deepcopy 内存数据
        - PG 后端：创建临时 schema 并复制数据
        - FS：copytree 到临时目录
        """
        logger.info(
            "EvalSandbox[%s]: 开始创建隔离环境 (user_id=%s)",
            self._sandbox_id, self._user_id,
        )

        # ── 1. 复制 FileSystem ──
        self._temp_dir = tempfile.mkdtemp(prefix=f"eval_{self._sandbox_id}_")
        sandbox_fs_path = os.path.join(self._temp_dir, "memory_repo")

        # 复制原始 FS 目录内容
        src_path = self._fs.base_path
        if os.path.exists(src_path):
            shutil.copytree(src_path, sandbox_fs_path)
        else:
            os.makedirs(sandbox_fs_path, exist_ok=True)

        from storage.file_system_store import FileSystemStore
        self._sandbox_fs = FileSystemStore(
            base_path=sandbox_fs_path, enable_git=False
        )

        # ── 2. 复制 VectorStore ──
        self._sandbox_vec = await self._clone_vec_store()

        # ── 3. 复制 GraphStore ──
        self._sandbox_graph = await self._clone_graph_store()

        # ── 4. 记录执行前快照 ──
        self._pre_vec_stats = self._sandbox_vec.get_stats() if self._sandbox_vec else {}
        self._pre_graph_stats = self._sandbox_graph.get_stats() if self._sandbox_graph else {}
        self._pre_fs_files = self._list_fs_files(sandbox_fs_path)

        logger.info(
            "EvalSandbox[%s]: 隔离环境创建完成 (temp_dir=%s)",
            self._sandbox_id, self._temp_dir,
        )

    async def _clone_vec_store(self) -> "VectorStoreBase":
        """复制向量存储。"""
        from storage.vector_stores import VectorStore, PgVectorStore

        if isinstance(self._vec, PgVectorStore):
            # PG 后端：创建临时 schema 并复制数据
            return await self._clone_pg_vec_store(self._vec)
        else:
            # Memory 后端：通过 serialize/from_config 深拷贝
            config = self._vec.serialize()
            config_copy = copy.deepcopy(config)
            return VectorStore.from_config(config_copy)

    async def _clone_pg_vec_store(self, pg_vec: Any) -> Any:
        """复制 PG 向量存储到临时 schema。"""
        from storage.vector_stores import PgVectorStore
        from storage.pg_store_backend import PgBackend

        source_schema = pg_vec.schema
        target_schema = f"eval_{self._sandbox_id}"
        self._temp_schema = target_schema

        backend = pg_vec.backend

        # 创建临时 schema 并复制数据
        async with await backend.conn() as conn:
            async with conn.cursor() as cur:
                # 创建临时 schema
                await cur.execute(
                    f"CREATE SCHEMA IF NOT EXISTS {target_schema}"
                )
                # 创建向量表（与源 schema 结构相同）
                await cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {target_schema}.t2_vec_entries (
                        LIKE {source_schema}.t2_vec_entries INCLUDING ALL
                    )
                """)
                # 复制数据
                await cur.execute(f"""
                    INSERT INTO {target_schema}.t2_vec_entries
                    SELECT * FROM {source_schema}.t2_vec_entries
                """)

        # 创建指向临时 schema 的 PgVectorStore 实例
        sandbox_vec = PgVectorStore(
            backend=backend,
            schema=target_schema,
            embedder=pg_vec.embedder,
        )
        return sandbox_vec

    async def _clone_graph_store(self) -> "GraphStoreBase":
        """复制图存储。"""
        from storage.graph_stores import GraphStore, AgePgGraphStore

        if isinstance(self._graph, AgePgGraphStore):
            # PG 后端：复制 AGE graph 数据到临时 schema 的内存 GraphStore
            # （AGE graph 复制成本高，改用内存版 GraphStore 加载数据）
            return await self._clone_pg_graph_to_memory(self._graph)
        else:
            # Memory 后端：通过 serialize/from_config 深拷贝
            config = self._graph.serialize()
            config_copy = copy.deepcopy(config)
            return GraphStore.from_config(config_copy)

    async def _clone_pg_graph_to_memory(self, pg_graph: Any) -> "GraphStoreBase":
        """将 PG AGE graph 数据加载到内存 GraphStore 中（避免复杂的 AGE graph 复制）。"""
        from storage.graph_stores import GraphStore

        # 获取所有节点和边
        try:
            stats = pg_graph.get_stats()
            nodes_data = pg_graph.search_nodes()
            edges_data = pg_graph.search_edges()
        except Exception as e:
            logger.warning(
                "EvalSandbox[%s]: 复制 PG graph 数据失败: %s，使用空 GraphStore",
                self._sandbox_id, e,
            )
            return GraphStore()

        # 构建内存版 GraphStore
        mem_graph = GraphStore()
        for node in nodes_data:
            mem_graph.add_node(
                node_id=node.get("id", ""),
                label=node.get("label", ""),
                properties=node.get("properties", {}),
                timestamp=node.get("created_at", ""),
            )
        for edge in edges_data:
            mem_graph.add_edge(
                source=edge.get("source", ""),
                target=edge.get("target", ""),
                relation=edge.get("relation", ""),
                properties=edge.get("properties", {}),
                timestamp=edge.get("timestamp", ""),
            )

        return mem_graph

    async def run_ingest(
        self,
        code: str,
        messages: list[dict[str, Any]],
        user_id: str,
        session_id: str,
    ) -> EvalResult:
        """在沙箱中执行摄入代码。

        Args:
            code: 完整的摄入代码文件内容。
            messages: 测试消息列表。
            user_id: 用户 ID。
            session_id: 会话 ID。

        Returns:
            EvalResult 评测结果。
        """
        if not self._sandbox_fs:
            return EvalResult(
                success=False,
                error="沙箱未初始化，请先调用 setup()",
            )

        return await self._execute_code(
            code=code,
            code_type="ingest",
            messages=messages,
            user_id=user_id,
            session_id=session_id,
        )

    async def run_retrieve(
        self,
        code: str,
        query: str,
        messages: list[dict[str, Any]],
        user_id: str,
        session_id: str,
    ) -> EvalResult:
        """在沙箱中执行消费代码。

        Args:
            code: 完整的消费代码文件内容。
            query: 测试查询。
            messages: 测试消息列表。
            user_id: 用户 ID。
            session_id: 会话 ID。

        Returns:
            EvalResult 评测结果。
        """
        if not self._sandbox_fs:
            return EvalResult(
                success=False,
                error="沙箱未初始化，请先调用 setup()",
            )

        return await self._execute_code(
            code=code,
            code_type="retrieve",
            query=query,
            messages=messages,
            user_id=user_id,
            session_id=session_id,
        )

    async def _execute_code(
        self,
        code: str,
        code_type: str,
        messages: list[dict[str, Any]],
        user_id: str,
        session_id: str,
        query: str = "",
    ) -> EvalResult:
        """在沙箱中执行代码的核心逻辑。

        步骤：
        1. 将代码写入临时文件
        2. 通过 importlib 加载类
        3. 实例化并调用对应方法
        4. 捕获输出和异常
        5. 计算沙箱变更
        """
        import importlib.util

        result = EvalResult()
        tmp_path = ""
        start_time = time.monotonic()

        try:
            # 写入临时文件
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                prefix=f"eval_{code_type}_",
                dir=self._temp_dir,
            ) as f:
                f.write(code)
                tmp_path = f.name

            # 加载类
            if code_type == "ingest":
                from context_task.codegen.loader import load_ingestor_class
                cls = load_ingestor_class(tmp_path)
            else:
                from context_task.codegen.loader import load_consumer_class
                cls = load_consumer_class(tmp_path)

            # 实例化
            if code_type == "ingest":
                instance = cls(
                    fs=self._sandbox_fs,
                    vec=self._sandbox_vec,
                    graph=self._sandbox_graph,
                    llm=self._llm,
                    memory_base=self._sandbox_fs.base_path,
                )
            else:
                instance = cls(
                    fs=self._sandbox_fs,
                    vec=self._sandbox_vec,
                    graph=self._sandbox_graph,
                    llm=self._llm,
                )

            # 捕获 stdout/stderr
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()

            # 执行代码（带超时）
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                if code_type == "ingest":
                    ret = await asyncio.wait_for(
                        instance.ingest_memory(
                            messages=messages,
                            user_id=user_id,
                            session_id=session_id,
                        ),
                        timeout=EVAL_TIMEOUT_S,
                    )
                else:
                    ret = await asyncio.wait_for(
                        instance.retrieve_memory(
                            query=query,
                            messages=messages,
                            user_id=user_id,
                            session_id=session_id,
                        ),
                        timeout=EVAL_TIMEOUT_S,
                    )

            result.success = True
            result.return_value = str(ret) if ret else ""
            result.stdout = stdout_buf.getvalue()
            result.stderr = stderr_buf.getvalue()

        except asyncio.TimeoutError:
            result.success = False
            result.error = f"执行超时（超过 {EVAL_TIMEOUT_S} 秒）"

        except Exception as e:
            result.success = False
            result.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        finally:
            # 清理临时代码文件
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

            # 计算执行耗时
            result.execution_time_ms = int(
                (time.monotonic() - start_time) * 1000
            )

        # 计算沙箱变更
        result.sandbox_changes = self._compute_changes()

        return result

    def _compute_changes(self) -> dict[str, Any]:
        """计算沙箱执行前后的变更。"""
        changes: dict[str, Any] = {}

        try:
            # FS 变更
            if self._sandbox_fs and self._temp_dir:
                sandbox_fs_path = os.path.join(self._temp_dir, "memory_repo")
                current_files = self._list_fs_files(sandbox_fs_path)
                new_files = current_files - self._pre_fs_files
                if new_files:
                    changes["fs_files_written"] = sorted(new_files)

            # Vec 变更
            if self._sandbox_vec:
                post_stats = self._sandbox_vec.get_stats()
                pre_total = self._pre_vec_stats.get("total_entries", 0)
                post_total = post_stats.get("total_entries", 0)
                diff = post_total - pre_total
                if diff > 0:
                    changes["vec_entries_added"] = diff

            # Graph 变更
            if self._sandbox_graph:
                post_stats = self._sandbox_graph.get_stats()
                pre_nodes = self._pre_graph_stats.get("total_nodes", 0)
                post_nodes = post_stats.get("total_nodes", 0)
                pre_edges = self._pre_graph_stats.get("total_edges", 0)
                post_edges = post_stats.get("total_edges", 0)
                if post_nodes - pre_nodes > 0:
                    changes["graph_nodes_added"] = post_nodes - pre_nodes
                if post_edges - pre_edges > 0:
                    changes["graph_edges_added"] = post_edges - pre_edges

        except Exception as e:
            logger.warning(
                "EvalSandbox[%s]: 计算变更失败: %s",
                self._sandbox_id, e,
            )

        return changes

    def _list_fs_files(self, base_path: str) -> set[str]:
        """列出目录下所有文件的相对路径。"""
        files = set()
        if not os.path.exists(base_path):
            return files
        for root, _, filenames in os.walk(base_path):
            for fname in filenames:
                rel = os.path.relpath(os.path.join(root, fname), base_path)
                files.add(rel)
        return files

    async def teardown(self) -> None:
        """清理隔离环境。"""
        logger.info(
            "EvalSandbox[%s]: 开始清理隔离环境",
            self._sandbox_id,
        )

        # 清理临时目录
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
            except Exception as e:
                logger.warning(
                    "EvalSandbox[%s]: 清理临时目录失败: %s",
                    self._sandbox_id, e,
                )

        # 清理 PG 临时 schema
        if self._temp_schema:
            await self._drop_temp_schema()

        self._sandbox_fs = None
        self._sandbox_vec = None
        self._sandbox_graph = None

        logger.info(
            "EvalSandbox[%s]: 隔离环境清理完成",
            self._sandbox_id,
        )

    async def _drop_temp_schema(self) -> None:
        """删除 PG 临时 schema。"""
        if not self._temp_schema:
            return

        try:
            from storage.vector_stores import PgVectorStore
            if isinstance(self._vec, PgVectorStore):
                backend = self._vec.backend
                async with await backend.conn() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            f"DROP SCHEMA IF EXISTS {self._temp_schema} CASCADE"
                        )
                logger.info(
                    "EvalSandbox[%s]: 已删除临时 schema %s",
                    self._sandbox_id, self._temp_schema,
                )
        except Exception as e:
            logger.warning(
                "EvalSandbox[%s]: 删除临时 schema 失败: %s",
                self._sandbox_id, e,
            )
