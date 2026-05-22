"""
Context Task 工具模块

统一管理所有 context task 的工具定义。
通过 Tool 类实现自包含的定义 + 执行逻辑，通过 ToolRegistry 统一管理。
"""

from .base_tool import BaseTool
from .registry import ToolRegistry, default_registry

# ── 导入所有工具类 ──
from .fs_tools import (
    FsWriteTool,
    FsAppendTool,
    FsReadTool,
    FsTreeTool,
    FsDeleteTool,
    FsSearchTool,
    FsGrepTool,
    FsReadLinesTool,
    FsExecuteBashTool,
    FsListTool,
    FsGetStructureSummaryTool,
    FsReadIndexTool,
)
from .vec_tools import (
    VecAddTool,
    VecSearchTool,
    VecSearchAllTool,
    VecListCollectionsTool,
    VecScrollTool,
    VecDeleteCollectionTool,
    VecCreateCollectionTool,
    VecDeleteTool,
    VecUpdateTool,
    VecSearchAllWithEmbeddingTool,
)
from .graph_tools import (
    GraphAddNodeTool,
    GraphAddEdgeTool,
    GraphGetNodeTool,
    GraphGetNeighborsTool,
    GraphSearchNodesTool,
    GraphSearchEdgesTool,
    GraphSearchNodesByEmbeddingTool,
    GraphSearchByTimeTool,
    GraphExecuteQueryTool,
    GraphStatsTool,
    GraphDeleteNodeTool,
    GraphDeleteEdgeTool,
    GraphGetSubgraphTool,
    GraphSetNodeEmbeddingTool,
    GraphGetNodeEmbeddingTool,
    GraphSetEdgeEmbeddingTool,
    GraphGetEdgeEmbeddingTool,
    GraphCypherReadTool,
    GraphCypherWriteTool,
)
from .sql_tools import SqlQueryReadTool
from .session_tools import SessionViewTool
from .special_tools import FinishTool, ReadReferenceStrategyTool
from .source_code_tools import (
    ReadFileSystemStoreSourceTool,
    ReadVectorStoreBaseSourceTool,
    ReadGraphStoreBaseSourceTool,
)
from .code_tools import (
    EvalCodeTool,
    SubmitIngestCodeTool,
    SubmitRetrieveCodeTool,
    SubmitRetrieveCodeFuncTool,
    SubmitStrategyTool,
    SubmitBaseClassCodeTool,
    SubmitSubclassCodeTool,
    PythonSyntaxCheckTool,
)

# ── 注册所有工具到全局 registry ──
default_registry.register_many([
    # 文件系统工具
    FsWriteTool(),
    FsAppendTool(),
    FsReadTool(),
    FsTreeTool(),
    FsDeleteTool(),
    FsSearchTool(),
    FsGrepTool(),
    FsReadLinesTool(),
    FsExecuteBashTool(),
    FsListTool(),
    FsGetStructureSummaryTool(),
    FsReadIndexTool(),
    # 向量数据库工具
    VecAddTool(),
    VecSearchTool(),
    VecSearchAllTool(),
    VecListCollectionsTool(),
    VecScrollTool(),
    VecDeleteCollectionTool(),
    VecCreateCollectionTool(),
    VecDeleteTool(),
    VecUpdateTool(),
    VecSearchAllWithEmbeddingTool(),
    # 图数据库工具
    GraphAddNodeTool(),
    GraphAddEdgeTool(),
    GraphGetNodeTool(),
    GraphGetNeighborsTool(),
    GraphSearchNodesTool(),
    GraphSearchEdgesTool(),
    GraphSearchNodesByEmbeddingTool(),
    GraphSearchByTimeTool(),
    GraphExecuteQueryTool(),
    GraphStatsTool(),
    GraphDeleteNodeTool(),
    GraphDeleteEdgeTool(),
    GraphGetSubgraphTool(),
    GraphSetNodeEmbeddingTool(),
    GraphGetNodeEmbeddingTool(),
    GraphSetEdgeEmbeddingTool(),
    GraphGetEdgeEmbeddingTool(),
    GraphCypherReadTool(),
    GraphCypherWriteTool(),
    # SQL 工具
    SqlQueryReadTool(),
    # 会话工具
    SessionViewTool(),
    # 特殊工具
    FinishTool(),
    ReadReferenceStrategyTool(),
    # 源代码读取工具
    ReadFileSystemStoreSourceTool(),
    ReadVectorStoreBaseSourceTool(),
    ReadGraphStoreBaseSourceTool(),
    # 代码工具
    EvalCodeTool(),
    SubmitIngestCodeTool(),
    SubmitRetrieveCodeTool(),
    SubmitRetrieveCodeFuncTool(),
    SubmitStrategyTool(),
    SubmitBaseClassCodeTool(),
    SubmitSubclassCodeTool(),
    PythonSyntaxCheckTool(),
])


# ── 向后兼容：生成旧的 dict 格式工具集 ──
# 这些变量保持与旧 base_tools.py / task_tools.py 相同的接口，
# 供尚未迁移到 registry 的代码使用。

# 基础工具分类（schema 列表）
FS_TOOLS = default_registry.get_schemas(categories=["fs"])
VECTOR_TOOLS = default_registry.get_schemas(
    categories=["vec"],
    exclude_names=["vec_scroll"],
)
GRAPH_TOOLS = default_registry.get_schemas(
    categories=["graph"],
    exclude_names=[
        "graph_cypher_read", "graph_cypher_write",
    ],
)
SQL_TOOLS = default_registry.get_schemas(categories=["sql"], backend="pg")
SESSION_TOOLS = default_registry.get_schemas(categories=["session"])
SPECIAL_TOOLS = default_registry.get_schemas(categories=["special"], names=["finish"]) + [
    default_registry.get("vec_delete_collection").to_llm_schema(),
]

# 组合工具集
BASE_TOOLS = FS_TOOLS + VECTOR_TOOLS + GRAPH_TOOLS + SESSION_TOOLS
READ_ONLY_TOOLS = default_registry.get_schemas(
    readonly_only=True,
    categories=["fs", "vec", "graph", "session"],
) + SQL_TOOLS
WRITE_TOOLS = BASE_TOOLS + SPECIAL_TOOLS
PG_TOOLS = default_registry.get_schemas(
    backend="pg",
    names=["vec_scroll", "sql_query_read", "graph_cypher_read", "graph_cypher_write"],
)

# 代码工具（schema dict）
EVAL_CODE_TOOL = default_registry.get("eval_code").to_llm_schema()
SUBMIT_INGEST_CODE_TOOL = default_registry.get("submit_ingest_code").to_llm_schema()
SUBMIT_RETRIEVE_CODE_TOOL = default_registry.get("submit_retrieve_code").to_llm_schema()
SUBMIT_RETRIEVE_CODE_FUNC_TOOL = default_registry.get("submit_retrieve_code_func").to_llm_schema()
SUBMIT_STRATEGY_TOOL = default_registry.get("submit_strategy").to_llm_schema()
SUBMIT_BASE_CLASS_CODE_TOOL = default_registry.get("submit_base_class_code").to_llm_schema()
SUBMIT_SUBCLASS_CODE_TOOL = default_registry.get("submit_subclass_code").to_llm_schema()
PYTHON_SYNTAX_CHECK_TOOL = default_registry.get("python_syntax_check").to_llm_schema()
FINISH_TOOL = default_registry.get("finish").to_llm_schema()
READ_REFERENCE_STRATEGY_TOOL = default_registry.get("read_reference_strategy").to_llm_schema()
READ_FS_STORE_SOURCE_TOOL = default_registry.get("read_fs_store_source").to_llm_schema()
READ_VEC_STORE_SOURCE_TOOL = default_registry.get("read_vec_store_source").to_llm_schema()
READ_GRAPH_STORE_SOURCE_TOOL = default_registry.get("read_graph_store_source").to_llm_schema()
SOURCE_CODE_TOOLS = [READ_FS_STORE_SOURCE_TOOL, READ_VEC_STORE_SOURCE_TOOL, READ_GRAPH_STORE_SOURCE_TOOL]

# Task-specific 工具集
INGEST_TOOLS = WRITE_TOOLS
CONSOLIDATE_TOOLS = WRITE_TOOLS
RETRIEVE_TOOLS = READ_ONLY_TOOLS

# PG 后端专用
_PG_RETRIEVE_TOOLS = PG_TOOLS
_PG_CONSOLIDATE_WRITE_TOOL = default_registry.get("graph_cypher_write").to_llm_schema()



# ── 向后兼容：工具获取方法 ──

def get_ingest_tools() -> list[dict]:
    """获取 ingest context task 的工具集"""
    return INGEST_TOOLS


def get_consolidate_tools() -> list[dict]:
    """获取 consolidate context task 的工具集"""
    return CONSOLIDATE_TOOLS


def get_retrieve_tools() -> list[dict]:
    """获取 retrieve context task 的工具集"""
    return RETRIEVE_TOOLS


def get_pg_retrieve_tools() -> list[dict]:
    """获取 PostgreSQL 后端专用的检索工具集"""
    return _PG_RETRIEVE_TOOLS


def get_pg_consolidate_write_tool() -> dict:
    """获取 PostgreSQL 后端专用的 consolidate 写入工具"""
    return _PG_CONSOLIDATE_WRITE_TOOL




# ── 导出 ──
__all__ = [
    # 核心类
    "BaseTool",
    "ToolRegistry",
    "default_registry",

    # 基础工具分类（向后兼容）
    "FS_TOOLS",
    "VECTOR_TOOLS",
    "GRAPH_TOOLS",
    "SQL_TOOLS",
    "SESSION_TOOLS",
    "SPECIAL_TOOLS",

    # 组合工具集（向后兼容）
    "BASE_TOOLS",
    "READ_ONLY_TOOLS",
    "WRITE_TOOLS",
    "PG_TOOLS",

    # 代码工具（向后兼容）
    "EVAL_CODE_TOOL",
    "SUBMIT_INGEST_CODE_TOOL",
    "SUBMIT_RETRIEVE_CODE_TOOL",
    "SUBMIT_RETRIEVE_CODE_FUNC_TOOL",
    "SUBMIT_STRATEGY_TOOL",
    "SUBMIT_BASE_CLASS_CODE_TOOL",
    "SUBMIT_SUBCLASS_CODE_TOOL",
    "PYTHON_SYNTAX_CHECK_TOOL",
    "FINISH_TOOL",
    "READ_REFERENCE_STRATEGY_TOOL",
    "READ_FS_STORE_SOURCE_TOOL",
    "READ_VEC_STORE_SOURCE_TOOL",
    "READ_GRAPH_STORE_SOURCE_TOOL",
    "SOURCE_CODE_TOOLS",

    # Task-specific 工具集（向后兼容）
    "INGEST_TOOLS",
    "CONSOLIDATE_TOOLS",
    "RETRIEVE_TOOLS",
    "_PG_RETRIEVE_TOOLS",
    "_PG_CONSOLIDATE_WRITE_TOOL",

    # 工具获取方法（向后兼容）
    "get_ingest_tools",
    "get_consolidate_tools",
    "get_retrieve_tools",
    "get_pg_retrieve_tools",
    "get_pg_consolidate_write_tool",
]
