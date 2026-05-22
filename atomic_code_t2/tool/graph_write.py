"""graph_write 工具 — 写节点或边（upsert 语义）。

对齐 T3 ingest 的关键行为：
- add_node / add_edge 时自动注入 ingest_time / ingest_turn 到 properties
- 写完后自动计算 node/edge embedding（如果 session_ctx.embedder 可用）
- add_edge 具有 upsert 语义：同 source+target 的旧边会被自动替换
"""

from __future__ import annotations

import re

from storage.stores_base import GraphStoreBase

from .base import BaseTool, ToolDefinition, ToolResult
from .session_context import SessionContext


_TOOL_DESCRIPTION = """\
向图库写入节点或边。

- 同 node_id 写入时自动覆盖旧节点
- 同 source→target 写入新边时自动替换旧边（无需手动删除）

参数：
- kind: "node" 或 "edge"
- kind=node 时必填：node_id（小写下划线 slug，如 italian_food）、label（Person/Topic/Activity/Genre/Preference）
- kind=edge 时必填：source、target、relation（如 prefers/avoids/tried/enjoys/dislikes/interested_in/participated_in）
"""


def _build_node_embed_text(node_id: str, label: str) -> str:
    """构建节点的文本表示用于 embedding（对齐 T3 ingest_t3.py:2384-2397）。"""
    readable_id = node_id.replace("_", " ").replace("-", " ")
    parts = [readable_id]
    if label:
        parts.append(f"({label})")
    return " ".join(parts)


def _build_edge_embed_text(source: str, relation: str, target: str) -> str:
    """构建边的文本表示用于 embedding（对齐 T3 ingest_t3.py:2400-2411）。"""
    src = source.replace("_", " ").replace("-", " ")
    rel = relation.replace("_", " ").replace("-", " ")
    tgt = target.replace("_", " ").replace("-", " ")
    return f"{src} {rel} {tgt}"


def _extract_edge_id(result_msg: str) -> str | None:
    """从 add_edge 返回消息中提取 edge_id（对齐 T3 ingest_t3.py:2413-2421）。"""
    m = re.search(r'\(id=(edge_\d+)\)', result_msg)
    return m.group(1) if m else None


class GraphWriteTool(BaseTool):
    """写节点或边。"""

    def __init__(
        self,
        graph: GraphStoreBase,
        *,
        session_ctx: SessionContext | None = None,
    ) -> None:
        self.graph = graph
        self.session_ctx = session_ctx

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="graph_write",
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
                        "description": "kind=node 时必填，小写 slug 形式",
                    },
                    "label": {
                        "type": "string",
                        "description": "kind=node 时必填，节点类型（Person/Topic/Activity/Genre/Preference 等）",
                    },
                    "source": {
                        "type": "string",
                        "description": "kind=edge 时必填，源节点 id",
                    },
                    "target": {
                        "type": "string",
                        "description": "kind=edge 时必填，目标节点 id",
                    },
                    "relation": {
                        "type": "string",
                        "description": "kind=edge 时必填，关系类型",
                    },
                },
                "required": ["kind"],
            },
        )

    async def execute(
        self,
        kind: str,
        node_id: str | None = None,
        label: str | None = None,
        source: str | None = None,
        target: str | None = None,
        relation: str | None = None,
    ) -> ToolResult:
        # 递增 turn counter
        if self.session_ctx:
            self.session_ctx.next_turn()

        # 构建 properties（ingest_time + ingest_turn）
        properties = self.session_ctx.build_graph_properties() if self.session_ctx else {}

        if kind == "node":
            if not node_id:
                return ToolResult(content="kind=node 时 node_id 必填", is_error=True)
            try:
                # 计算 node embedding
                embedding = None
                if self.session_ctx and self.session_ctx.embedder:
                    embed_text = _build_node_embed_text(node_id, label or "")
                    try:
                        embs = await self.session_ctx.embedder.embed([embed_text])
                        embedding = embs[0] if embs else None
                    except Exception:
                        pass  # embedding 失败不阻断写入

                msg = self.graph.add_node(
                    node_id=node_id,
                    label=label or "",
                    properties=properties,
                    embedding=embedding,
                )
                return ToolResult(content=f"已写入节点 {node_id}: {msg}")
            except Exception as e:
                return ToolResult(content=f"写入节点失败: {e}", is_error=True)

        if kind == "edge":
            if not source or not target or not relation:
                return ToolResult(
                    content="kind=edge 时 source/target/relation 必填", is_error=True
                )
            try:
                # Upsert 语义：先查找同 source+target 的旧边并删除（对齐 T3 ingest_t3.py:2329-2337）
                try:
                    neighbors = self.graph.get_neighbors(source)
                    for nb in neighbors:
                        nb_node = nb.get("node", {}) if isinstance(nb, dict) else {}
                        nb_edge = nb.get("edge", {}) if isinstance(nb, dict) else {}
                        if nb_node.get("id") == target:
                            old_edge_id = nb_edge.get("id")
                            if old_edge_id:
                                self.graph.delete_edge(old_edge_id)
                except Exception:
                    pass  # 查找失败不影响新增

                msg = self.graph.add_edge(
                    source=source,
                    target=target,
                    relation=relation,
                    properties=properties,
                )

                # 计算 edge embedding（对齐 T3 ingest_t3.py:2354-2358, 2364-2381）
                if self.session_ctx and self.session_ctx.embedder:
                    edge_id = _extract_edge_id(msg)
                    if edge_id:
                        embed_text = _build_edge_embed_text(source, relation, target)
                        try:
                            embs = await self.session_ctx.embedder.embed([embed_text])
                            if embs and embs[0]:
                                self.graph.set_edge_embedding(edge_id, embs[0])
                        except Exception:
                            pass  # embedding 失败不阻断

                return ToolResult(
                    content=f"已写入边 {source}--[{relation}]-->{target}: {msg}"
                )
            except Exception as e:
                return ToolResult(content=f"写入边失败: {e}", is_error=True)

        return ToolResult(content=f"未知 kind: {kind}（应为 'node' 或 'edge'）", is_error=True)
