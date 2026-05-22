"""graph_search 工具 — 在图库中按关键词查找节点和边。"""

from __future__ import annotations

from storage.stores_base import GraphStoreBase

from .base import BaseTool, ToolDefinition, ToolResult


_TOOL_DESCRIPTION = """\
在图库中按关键词查找节点和边。

返回：
- [Nodes] 列表：`- <node_id> (label=<label>)`
- [Edges] 列表：`- <source> --[<relation>]--> <target>`

适用场景：
- 判断某个实体（人/话题/活动）是否已在图中
- 查询某实体的关系网，再决定 graph_write 怎么补充
"""


class GraphSearchTool(BaseTool):
    """关键词搜节点和边。"""

    def __init__(self, graph: GraphStoreBase) -> None:
        self.graph = graph

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="graph_search",
            description=_TOOL_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "关键词，会用作节点 keyword 与边 relation 的子串匹配",
                    },
                },
                "required": ["query"],
            },
        )

    async def execute(self, query: str) -> ToolResult:
        if not query:
            return ToolResult(content="query 不能为空", is_error=True)

        try:
            nodes = self.graph.search_nodes(keyword=query)
        except Exception as e:
            return ToolResult(content=f"节点检索失败: {e}", is_error=True)

        try:
            edges = self.graph.search_edges(relation=query)
        except Exception:
            edges = []

        parts: list[str] = []
        if nodes:
            parts.append("[Nodes]")
            for n in nodes[:30]:
                nid = n.get("node_id") or n.get("id", "?")
                label = n.get("label", "")
                parts.append(f"- {nid} (label={label})")
        if edges:
            parts.append("[Edges]")
            for e in edges[:30]:
                s = e.get("source", "?")
                t = e.get("target", "?")
                rel = e.get("relation", "?")
                parts.append(f"- {s} --[{rel}]--> {t}")

        if not parts:
            return ToolResult(content="(无相关节点或边)")
        return ToolResult(content="\n".join(parts))
