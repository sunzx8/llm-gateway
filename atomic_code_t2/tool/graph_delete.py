"""graph_delete 工具 — 删除图库中的节点或边。

用于 consolidate 阶段的图结构优化：合并重复节点、清理过时边。
"""

from __future__ import annotations

from storage.stores_base import GraphStoreBase

from .base import BaseTool, ToolDefinition, ToolResult


_TOOL_DESCRIPTION = """\
删除图库中的节点或边。

- kind=node: 删除指定节点及其所有关联边
- kind=edge: 删除指定边（不影响节点）

适用场景：
- 节点合并：将 "italian_food" 和 "italian_cuisine" 合并后删除旧节点
- 清理过时边：立场已变化，旧的 stance 边不再准确

参数：
- kind: "node" 或 "edge"
- node_id: kind=node 时必填，要删除的节点 ID
- edge_id: kind=edge 时必填，要删除的边 ID
"""


class GraphDeleteTool(BaseTool):
    """删除节点或边。"""

    def __init__(self, graph: GraphStoreBase) -> None:
        self.graph = graph

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="graph_delete",
            description=_TOOL_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["node", "edge"],
                        "description": "'node' 或 'edge'",
                    },
                    "node_id": {
                        "type": "string",
                        "description": "kind=node 时必填，要删除的节点 ID",
                    },
                    "edge_id": {
                        "type": "string",
                        "description": "kind=edge 时必填，要删除的边 ID",
                    },
                },
                "required": ["kind"],
            },
        )

    async def execute(
        self,
        kind: str,
        node_id: str | None = None,
        edge_id: str | None = None,
    ) -> ToolResult:
        if kind == "node":
            if not node_id:
                return ToolResult(content="kind=node 时 node_id 必填", is_error=True)
            try:
                result = self.graph.delete_node(node_id)
                return ToolResult(content=f"已删除节点 {node_id}: {result}")
            except Exception as e:
                return ToolResult(content=f"删除节点失败: {e}", is_error=True)

        if kind == "edge":
            if not edge_id:
                return ToolResult(content="kind=edge 时 edge_id 必填", is_error=True)
            try:
                result = self.graph.delete_edge(edge_id)
                return ToolResult(content=f"已删除边 {edge_id}: {result}")
            except Exception as e:
                return ToolResult(content=f"删除边失败: {e}", is_error=True)

        return ToolResult(content=f"未知 kind: {kind}（应为 'node' 或 'edge'）", is_error=True)
