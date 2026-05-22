# Context Task 工具模块

本模块通过 Tool 类实现自包含的工具定义 + 执行逻辑，通过 ToolRegistry 统一管理。

## 目录结构

```
context_task/tool/
├── __init__.py              # 模块导出、工具注册、向后兼容接口
├── base_tool.py             # BaseTool 抽象基类
├── registry.py              # ToolRegistry 注册表
├── fs_tools.py              # 文件系统工具类
├── vec_tools.py             # 向量数据库工具类
├── graph_tools.py           # 图数据库工具类
├── sql_tools.py             # SQL 工具类
├── session_tools.py         # 会话工具类
├── special_tools.py         # 特殊工具类（FinishTool）
├── code_tools.py            # 代码工具类（eval/submit）
└── README.md                # 本文档
```

## 架构设计

### BaseTool 基类

每个工具继承 `BaseTool`，实现：
- **元信息**：`name`、`description`、`parameters`（JSON Schema）
- **标记**：`is_readonly`、`requires_backend`、`category`
- **方法**：`to_llm_schema()` 生成 LLM function calling 格式；`execute(args, **deps)` 执行逻辑

### ToolRegistry 注册表

- `register(tool)` / `register_many(tools)`：注册工具
- `get(name)`：按名称获取工具实例
- `get_schemas(...)`：按条件过滤生成 LLM schema 列表
- `execute(name, args, **deps)`：统一执行入口

### 依赖传递

工具执行时通过 `**deps` 接收依赖（`fs`、`vec`、`graph`、`session_id`、`allow_write` 等），
由调用方（task）负责传入，不需要额外的 Context 对象。

## 工具分类

| 分类 | 文件 | 工具 |
|------|------|------|
| fs | fs_tools.py | FsWriteTool, FsAppendTool, FsReadTool, FsTreeTool, FsDeleteTool, FsSearchTool, FsGrepTool, FsReadLinesTool, FsExecuteBashTool, FsListTool |
| vec | vec_tools.py | VecAddTool, VecSearchTool, VecSearchAllTool, VecListCollectionsTool, VecScrollTool, VecDeleteCollectionTool, VecCreateCollectionTool, VecDeleteTool |
| graph | graph_tools.py | GraphAddNodeTool, GraphAddEdgeTool, GraphGetNeighborsTool, GraphSearchNodesTool, GraphStatsTool, GraphDeleteNodeTool, GraphDeleteEdgeTool, GraphGetSubgraphTool, GraphCypherReadTool, GraphCypherWriteTool |
| sql | sql_tools.py | SqlQueryReadTool |
| session | session_tools.py | SessionViewTool |
| special | special_tools.py | FinishTool |
| code | code_tools.py | EvalCodeTool, SubmitIngestCodeTool, SubmitRetrieveCodeTool, SubmitRetrieveCodeFuncTool |

## 使用方法

### 新方式（推荐）：使用 registry

```python
from context_task.tool import default_registry

# 获取工具 schema 列表
schemas = default_registry.get_schemas(
    readonly_only=True,
    categories=["fs", "vec", "graph"],
)

# 执行工具
result = await default_registry.execute(
    "fs_read", {"path": "facts.md"},
    fs=self.fs, vec=self.vec, graph=self.graph,
)
```

### 向后兼容方式：使用 dict 变量

```python
from context_task.tool import (
    INGEST_TOOLS, CONSOLIDATE_TOOLS, RETRIEVE_TOOLS,
    EVAL_CODE_TOOL, SUBMIT_INGEST_CODE_TOOL,
    _PG_RETRIEVE_TOOLS,
)
```

## 新增工具步骤

1. 在对应的 `*_tools.py` 文件中创建新的 Tool 类
2. 在 `__init__.py` 中导入并注册到 `default_registry`
3. 如需向后兼容，更新相应的 dict 变量


