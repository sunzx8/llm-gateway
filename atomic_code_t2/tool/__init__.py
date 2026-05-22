"""Atomic_Code_T2 方案专用工具集。

每个工具一个文件：
- 文件系统：ls / read_file / write_file / edit_file / grep / bm25_search / fs_delete
- 向量库：vec_search / vec_write / vec_delete
- 图库：graph_search / graph_write / graph_delete
- 控制：finish

SessionContext 贯穿所有写工具，统一注入时间前缀、metadata、embedding。

两个 registry 构建函数：
- build_default_registry: ingest 用（11 个工具，无删除）
- build_consolidate_registry: consolidate 用（15 个工具，含删除 + fs_execute_bash）
"""

from __future__ import annotations

from typing import Any

from storage.file_system_store import FileSystemStore
from storage.stores_base import GraphStoreBase, VectorStoreBase

from .base import BaseTool, ToolDefinition, ToolResult
from .registry import ToolRegistry
from .session_context import SessionContext

from .ls import LsTool
from .read_file import ReadFileTool
from .write_file import WriteFileTool
from .edit_file import EditFileTool
from .grep import GrepTool
from .bm25_search import BM25SearchTool
from .vec_search import VecSearchTool
from .vec_write import VecWriteTool
from .vec_delete import VecDeleteTool
from .graph_search import GraphSearchTool
from .graph_write import GraphWriteTool
from .graph_delete import GraphDeleteTool
from .fs_delete import FsDeleteTool
from .fs_execute_bash import FsExecuteBashTool
from .finish import FinishTool


def build_default_registry(
    fs: FileSystemStore,
    vec: VectorStoreBase,
    graph: GraphStoreBase,
    *,
    session_time: str = "",
    embedder: Any = None,
) -> ToolRegistry:
    """构造 ingest 用的 ToolRegistry（11 个工具，无删除）。

    Args:
        fs: 文件系统存储后端。
        vec: 向量存储后端。
        graph: 图存储后端。
        session_time: 本次摄入的对话时间（由 ingest_task 传入）。
        embedder: embedding 接口（由 ingest_task 传入，用于 graph node/edge 自动嵌入）。
    """
    # 构建 session context（所有写工具共享）
    ctx = SessionContext(session_time=session_time, embedder=embedder)

    reg = ToolRegistry()
    reg.register_many([
        LsTool(fs),
        ReadFileTool(fs),
        WriteFileTool(fs, session_ctx=ctx),
        EditFileTool(fs, session_ctx=ctx),
        GrepTool(fs),
        BM25SearchTool(fs),
        VecSearchTool(vec),
        VecWriteTool(vec, session_ctx=ctx),
        GraphSearchTool(graph),
        GraphWriteTool(graph, session_ctx=ctx),
        FinishTool(),
    ])
    return reg


def build_consolidate_registry(
    fs: FileSystemStore,
    vec: VectorStoreBase,
    graph: GraphStoreBase,
    *,
    session_time: str = "",
    embedder: Any = None,
) -> ToolRegistry:
    """构造 consolidate 用的 ToolRegistry（15 个工具，含删除 + fs_execute_bash）。

    相比 ingest 多出：fs_delete / vec_delete / graph_delete / fs_execute_bash。
    consolidate 是重型演进任务，需要删除能力来做去重、合并、重组，
    并通过 fs_execute_bash 做批量审计 / 精确定位 / 仓库自检。
    fs_execute_bash 仅注册到 consolidate，不会暴露给 ingest 或 retrieve。

    Args:
        fs: 文件系统存储后端。
        vec: 向量存储后端。
        graph: 图存储后端。
        session_time: 演进时间戳。
        embedder: embedding 接口（用于 graph node/edge 自动嵌入）。
    """
    ctx = SessionContext(session_time=session_time, embedder=embedder)

    reg = ToolRegistry()
    reg.register_many([
        # 读
        LsTool(fs),
        ReadFileTool(fs),
        GrepTool(fs),
        BM25SearchTool(fs),
        VecSearchTool(vec),
        GraphSearchTool(graph),
        # 写
        WriteFileTool(fs, session_ctx=ctx),
        EditFileTool(fs, session_ctx=ctx),
        VecWriteTool(vec, session_ctx=ctx),
        GraphWriteTool(graph, session_ctx=ctx),
        # 删除（consolidate 专属）
        FsDeleteTool(fs),
        VecDeleteTool(vec),
        GraphDeleteTool(graph),
        # Shell（consolidate 专属：批量审计 / 精确定位 / 仓库自检）
        FsExecuteBashTool(fs, session_ctx=ctx),
        # 控制
        FinishTool(),
    ])
    return reg


__all__ = [
    "BaseTool",
    "ToolDefinition",
    "ToolResult",
    "ToolRegistry",
    "SessionContext",
    "build_default_registry",
    "build_consolidate_registry",
    "LsTool",
    "ReadFileTool",
    "WriteFileTool",
    "EditFileTool",
    "GrepTool",
    "BM25SearchTool",
    "VecSearchTool",
    "VecWriteTool",
    "VecDeleteTool",
    "GraphSearchTool",
    "GraphWriteTool",
    "GraphDeleteTool",
    "FsDeleteTool",
    "FsExecuteBashTool",
    "FinishTool",
]
