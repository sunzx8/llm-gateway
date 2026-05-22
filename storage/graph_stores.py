"""图存储后端 — 统一入口。

包含：
- :class:`GraphStoreBase`   — 抽象接口（从 stores_base 再导出）
- :class:`GraphStore`       — 基于内存的实现（开发 / 测试 / 轻量部署）
- :class:`AgePgGraphStore`  — 基于 Apache AGE 的生产级实现

调用方只需从本模块导入，无需关心具体实现来自哪个子模块::

    from agent_memory.core.graph_stores import GraphStore, AgePgGraphStore
"""

from __future__ import annotations

import json
import re
from typing import Any

from .stores_base import GraphStoreBase
import logger.logger as logger

__all__ = [
    "GraphStoreBase",
    "GraphStore",
    "AgePgGraphStore",
]


# ---------------------------------------------------------------------------
# 内存版实现
# ---------------------------------------------------------------------------

class GraphStore(GraphStoreBase):
    """时序图数据库存储后端（内存版）。

    基于内存中的图结构实现，支持：
    - 节点 CRUD（带时间戳和属性）
    - 边 CRUD（带关系类型、时间戳和属性）
    - 图查询（邻居、子图、模式匹配）
    - 节点/边嵌入向量（语义检索）
    - 时序查询（按时间范围过滤）
    """

    def __init__(self, embedding_interface=None):
        # nodes: {node_id: {id, label, properties, created_at, updated_at}}
        self._nodes: dict[str, dict[str, Any]] = {}
        # edges: [{id, source, target, relation, properties, created_at}]
        self._edges: list[dict[str, Any]] = []
        self._edge_counter = 0
        # 节点嵌入向量: {node_id: [float, ...]}
        self._node_embeddings: dict[str, list[float]] = {}
        # 边嵌入向量: {edge_id: [float, ...]}（用于 1 跳边的精确粗排）
        self._edge_embeddings: dict[str, list[float]] = {}
        # Embedding 接口（用于自动计算节点向量）
        self.embedder = embedding_interface

    # ------------------------------------------------------------------
    # 序列化 / 反序列化
    # ------------------------------------------------------------------

    def serialize(self) -> dict[str, Any]:
        """将 GraphStore 序列化为可 JSON 化的配置字典。

        内存版需要序列化所有内部数据（nodes、edges、edge_counter、embeddings），
        以便在消费脚本中完整重建实例。

        Returns:
            包含重建实例所需全部参数和数据的字典。
        """
        data: dict[str, Any] = {
            "backend": "memory",
            "edge_counter": self._edge_counter,
            "nodes": self._nodes,
            "edges": self._edges,
            "node_embeddings": self._node_embeddings,
            "edge_embeddings": self._edge_embeddings,
        }
        if self.embedder is not None and hasattr(self.embedder, "serialize"):
            data["embedding_config"] = self.embedder.serialize()
        return data

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GraphStore":
        """从配置字典反序列化创建 GraphStore 实例。

        内存版会恢复所有内部数据（nodes、edges、edge_counter、embeddings）。

        Args:
            config: 序列化配置字典，包含 'nodes'、'edges'、'edge_counter' 和可选的 embedding 字段。

        Returns:
            GraphStore 实例。
        """
        embedder = None
        embedding_config = config.get("embedding_config")
        if embedding_config:
            from utils.memory_llm_interface import EmbeddingInterface
            embedder = EmbeddingInterface(embedding_config)
        instance = cls(embedding_interface=embedder)
        instance._edge_counter = config.get("edge_counter", 0)
        instance._nodes = config.get("nodes", {})
        instance._edges = config.get("edges", [])
        instance._node_embeddings = config.get("node_embeddings", {})
        instance._edge_embeddings = config.get("edge_embeddings", {})
        return instance

    def add_node(
        self,
        node_id: str,
        label: str = "",
        properties: dict[str, Any] | None = None,
        timestamp: str = "",
        record_id: int | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        import time as _time
        if record_id is None:
            record_id = int(_time.time() * 1000)
        if node_id in self._nodes:
            node = self._nodes[node_id]
            if label:
                node["label"] = label
            if properties:
                node["properties"].update(properties)
            node["updated_at"] = timestamp
            node["record_id"] = record_id
            if embedding is not None:
                self._node_embeddings[node_id] = embedding
            return f"Node '{node_id}' updated."
        self._nodes[node_id] = {
            "id": node_id,
            "label": label,
            "properties": properties or {},
            "created_at": timestamp,
            "updated_at": timestamp,
            "record_id": record_id,
        }
        if embedding is not None:
            self._node_embeddings[node_id] = embedding
        return f"Node '{node_id}' created."

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        return self._nodes.get(node_id)

    def delete_node(self, node_id: str) -> str:
        if node_id not in self._nodes:
            return f"Node '{node_id}' not found."
        del self._nodes[node_id]
        # 删除嵌入向量
        self._node_embeddings.pop(node_id, None)
        self._edges = [
            e for e in self._edges
            if e["source"] != node_id and e["target"] != node_id
        ]
        return f"Node '{node_id}' and its edges deleted."

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        properties: dict[str, Any] | None = None,
        timestamp: str = "",
        record_id: int | None = None,
    ) -> str:
        import time as _time
        if record_id is None:
            record_id = int(_time.time() * 1000)
        self._edge_counter += 1
        edge_id = f"edge_{self._edge_counter}"
        self._edges.append({
            "id": edge_id,
            "source": source,
            "target": target,
            "relation": relation,
            "properties": properties or {},
            "created_at": timestamp,
            "record_id": record_id,
        })
        return f"Edge '{source}' --[{relation}]--> '{target}' created (id={edge_id})."

    def delete_edge(self, edge_id: str) -> str:
        before = len(self._edges)
        self._edges = [e for e in self._edges if e["id"] != edge_id]
        if len(self._edges) < before:
            return f"Edge '{edge_id}' deleted."
        return f"Edge '{edge_id}' not found."

    def get_neighbors(
        self,
        node_id: str,
        relation: str | None = None,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        results = []
        for edge in self._edges:
            if relation and edge["relation"] != relation:
                continue
            if direction in ("out", "both") and edge["source"] == node_id:
                target_node = self._nodes.get(edge["target"])
                if target_node:
                    results.append({"node": target_node, "edge": edge, "direction": "out"})
            if direction in ("in", "both") and edge["target"] == node_id:
                source_node = self._nodes.get(edge["source"])
                if source_node:
                    results.append({"node": source_node, "edge": edge, "direction": "in"})
        return results

    def search_nodes(
        self,
        label: str | None = None,
        properties_filter: dict[str, Any] | None = None,
        keyword: str | None = None,
    ) -> list[dict[str, Any]]:
        results = []
        for node in self._nodes.values():
            if label and node["label"] != label:
                continue
            if properties_filter:
                match = all(
                    node["properties"].get(k) == v
                    for k, v in properties_filter.items()
                )
                if not match:
                    continue
            if keyword:
                searchable = f"{node['id']} {node['label']} {str(node['properties'])}".lower()
                if keyword.lower() not in searchable:
                    continue
            results.append(node)
        return results

    def search_edges(
        self,
        relation: str | None = None,
        source: str | None = None,
        target: str | None = None,
        keyword: str | None = None,
    ) -> list[dict[str, Any]]:
        results = []
        for edge in self._edges:
            if relation and edge["relation"] != relation:
                continue
            if source and edge["source"] != source:
                continue
            if target and edge["target"] != target:
                continue
            if keyword:
                keyword_lower = keyword.lower()
                searchable = f"{edge['source']} {edge['target']} {edge['relation']} {str(edge['properties'])}".lower()
                if keyword_lower not in searchable:
                    continue
            results.append(edge)
        return results

    def get_subgraph(self, node_id: str, depth: int = 2) -> dict[str, Any]:
        visited_nodes: set[str] = set()
        visited_edges: set[str] = set()
        queue = [(node_id, 0)]
        while queue:
            current, d = queue.pop(0)
            if current in visited_nodes or d > depth:
                continue
            visited_nodes.add(current)
            for edge in self._edges:
                if edge["source"] == current:
                    visited_edges.add(edge["id"])
                    if d + 1 <= depth:
                        queue.append((edge["target"], d + 1))
                elif edge["target"] == current:
                    visited_edges.add(edge["id"])
                    if d + 1 <= depth:
                        queue.append((edge["source"], d + 1))
        return {
            "nodes": [self._nodes[nid] for nid in visited_nodes if nid in self._nodes],
            "edges": [e for e in self._edges if e["id"] in visited_edges],
        }

    def get_stats(self) -> dict[str, Any]:
        relation_counts: dict[str, int] = {}
        for edge in self._edges:
            r = edge["relation"]
            relation_counts[r] = relation_counts.get(r, 0) + 1
        label_counts: dict[str, int] = {}
        for node in self._nodes.values():
            lbl = node.get("label") or "(unlabeled)"
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "node_labels": label_counts,
            "relation_types": relation_counts,
        }

    # ------------------------------------------------------------------
    # Embedding 相关方法
    # ------------------------------------------------------------------

    def set_node_embedding(self, node_id: str, embedding: list[float]) -> str:
        """为已有节点设置嵌入向量。"""
        if node_id not in self._nodes:
            return f"Node '{node_id}' not found."
        self._node_embeddings[node_id] = embedding
        return f"Embedding set for node '{node_id}'."

    def get_node_embedding(self, node_id: str) -> list[float] | None:
        """获取节点的嵌入向量。"""
        return self._node_embeddings.get(node_id)

    def set_edge_embedding(self, edge_id: str, embedding: list[float]) -> str:
        """为边设置嵌入向量（基于边的全信息文本：source relation target）。"""
        self._edge_embeddings[edge_id] = embedding
        return f"Embedding set for edge '{edge_id}'."

    def get_edge_embedding(self, edge_id: str) -> list[float] | None:
        """获取边的嵌入向量。"""
        return self._edge_embeddings.get(edge_id)

    def search_nodes_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """基于向量相似度搜索节点。

        Args:
            query_embedding: 查询向量。
            top_k: 返回最相似的前 k 个节点。
            threshold: 最低相似度阈值。

        Returns:
            按相似度降序排列的节点列表，每个节点附带 similarity 字段。
        """
        import math

        if not query_embedding:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for node_id, emb in self._node_embeddings.items():
            if not emb or len(emb) != len(query_embedding):
                continue
            # 余弦相似度
            dot = sum(a * b for a, b in zip(query_embedding, emb))
            norm_q = math.sqrt(sum(a * a for a in query_embedding)) or 1e-10
            norm_e = math.sqrt(sum(b * b for b in emb)) or 1e-10
            sim = dot / (norm_q * norm_e)
            if sim >= threshold:
                node = self._nodes.get(node_id)
                if node:
                    scored.append((sim, {**node, "similarity": sim}))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

    # ------------------------------------------------------------------
    # 时序查询
    # ------------------------------------------------------------------

    def search_by_time(
        self,
        time_query: str,
        node_id: str | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        """基于时间范围搜索图中的节点和边。

        支持粗粒度时间匹配：
        - "2017" → 匹配 occurred_at 以 "2017" 开头的所有记录
        - "2019-03" → 匹配 occurred_at 以 "2019-03" 开头的记录
        - "2018~2020" → 匹配 2018 到 2020 年的记录（范围查询）

        Args:
            time_query: 时间查询字符串（支持单值和范围）。
            node_id: 可选，限定与某个节点相关的记录。
            label: 可选，限定节点标签。

        Returns:
            {"nodes": [...], "edges": [...]} 匹配的节点和边。
        """
        matched_nodes: list[dict[str, Any]] = []
        matched_edges: list[dict[str, Any]] = []

        # 解析时间范围
        time_start, time_end = self._parse_time_range(time_query)

        # 搜索节点
        for node in self._nodes.values():
            if label and node.get("label") != label:
                continue
            if node_id and node["id"] != node_id:
                continue
            occurred_at = node.get("properties", {}).get("occurred_at", "")
            if occurred_at and self._time_in_range(occurred_at, time_start, time_end):
                matched_nodes.append(node)

        # 搜索边
        for edge in self._edges:
            if node_id and edge["source"] != node_id and edge["target"] != node_id:
                continue
            occurred_at = edge.get("properties", {}).get("occurred_at", "")
            if occurred_at and self._time_in_range(occurred_at, time_start, time_end):
                matched_edges.append(edge)

        return {"nodes": matched_nodes, "edges": matched_edges}

    @staticmethod
    def _parse_time_range(time_query: str) -> tuple[str, str]:
        """解析时间查询字符串为 (start, end) 范围。

        支持格式：
        - "2017" → ("2017", "2017")
        - "2018~2020" → ("2018", "2020")
        - "2019-03~2019-06" → ("2019-03", "2019-06")
        """
        time_query = time_query.strip()
        if "~" in time_query:
            parts = time_query.split("~", 1)
            return parts[0].strip(), parts[1].strip()
        return time_query, time_query

    @staticmethod
    def _time_in_range(occurred_at: str, time_start: str, time_end: str) -> bool:
        """判断 occurred_at 是否在 [time_start, time_end] 范围内。

        使用字符串前缀匹配 + 字典序比较实现粗粒度时间过滤：
        - "2017" 匹配 "2017", "2017-01", "2017-06-15" 等
        - 范围 "2018"~"2020" 匹配 "2018", "2019-03", "2020-12-31" 等
        """
        if not occurred_at:
            return False

        # 前缀匹配：如果 start == end，则做前缀匹配
        if time_start == time_end:
            return occurred_at.startswith(time_start)

        # 范围匹配：将 occurred_at 截取到与 start/end 相同的精度进行比较
        min_len = min(len(time_start), len(time_end))
        occurred_prefix = occurred_at[:min_len]
        return time_start <= occurred_prefix <= time_end

    # ------------------------------------------------------------------
    # 灵活查询
    # ------------------------------------------------------------------

    def execute_query(self, query_str: str) -> str:
        """执行类 Cypher 查询语句（简化版，供模型灵活使用）。

        支持的查询模式：
        - MATCH (n) WHERE n.label = 'X' RETURN n
        - MATCH (n)-[r:RELATION]->(m) RETURN n, r, m
        - 更复杂的查询通过 execute_python 实现
        """
        import json
        query_lower = query_str.lower().strip()

        if "all nodes" in query_lower or query_lower == "show nodes":
            return json.dumps(list(self._nodes.values())[:50], ensure_ascii=False, indent=2)
        elif "all edges" in query_lower or query_lower == "show edges":
            return json.dumps(self._edges[:50], ensure_ascii=False, indent=2)
        elif "stats" in query_lower:
            return json.dumps(self.get_stats(), ensure_ascii=False, indent=2)
        else:
            return (
                f"Query not recognized. Use search_nodes/search_edges/get_neighbors "
                f"for structured queries, or execute_python for complex operations. "
                f"Available nodes: {len(self._nodes)}, edges: {len(self._edges)}"
            )


# ---------------------------------------------------------------------------
# PG 版实现（Apache AGE）
# ---------------------------------------------------------------------------

class AgePgGraphStore(GraphStoreBase):
    """基于 Apache AGE 的图存储（生产级，per-schema 独立 graph）。

    额外实现 :class:`stores_base.CypherCapableMixin`，由 T2 task 层按
    ``isinstance`` 检测后选择是否暴露 ``cypher_read`` / ``cypher_write``
    工具给 LLM。

    每个用户 schema 对应一个 AGE graph（``t2g_<sha12>``）。

    Args:
        backend: :class:`stores_pg.PgBackend` 共享实例。
        schema: 用户独立的 PostgreSQL schema 名称（如 'u_user_42'）。
    """

    def __init__(self, backend: Any, schema: str):
        from .pg_store_backend import _check_pg_available, _safe_graph_name, _check_safe_ident
        _check_pg_available()
        _check_safe_ident(schema, "schema")
        from .stores_base import CypherCapableMixin
        self.__class__ = type(
            "AgePgGraphStore",
            (AgePgGraphStore, CypherCapableMixin),
            {},
        )
        self.backend = backend
        self.schema = schema
        self.graph_name = _safe_graph_name(schema)
        self._graph_ready = False
        self._edge_counter = 0

    # ------------------------------------------------------------------
    # 序列化 / 反序列化
    # ------------------------------------------------------------------

    def serialize(self) -> dict[str, Any]:
        """将 AgePgGraphStore 序列化为可 JSON 化的配置字典。

        PG 版只需序列化连接参数和 schema，数据存储在 PG 中无需序列化。

        Returns:
            包含重建实例所需全部参数的字典。
        """
        return {
            "backend": "pg",
            "dsn": self.backend.dsn,
            "schema": self.schema,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AgePgGraphStore":
        """从配置字典反序列化创建 AgePgGraphStore 实例。

        Args:
            config: 序列化配置字典，包含 'dsn' 和 'schema' 字段。

        Returns:
            AgePgGraphStore 实例。
        """
        from .pg_store_backend import PgBackend
        dsn = config.get("dsn", "")
        schema = config.get("schema", "")
        backend = PgBackend(dsn)
        return cls(backend=backend, schema=schema)

    # ------------------------------------------------------------------
    # 图初始化（幂等）
    # ------------------------------------------------------------------

    async def _ensure_graph(self) -> None:
        if self._graph_ready:
            return
        async with await self.backend.conn() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT 1 FROM ag_catalog.ag_graph WHERE name = %s", (self.graph_name,)
                )
                if not await cur.fetchone():
                    await cur.execute(f"SELECT create_graph('{self.graph_name}')")
                await cur.execute(
                    f"INSERT INTO {self.schema}.t2_graph_registry (graph_name) VALUES (%s) "
                    "ON CONFLICT (graph_name) DO NOTHING",
                    (self.graph_name,),
                )
        self._graph_ready = True

    def _ensure_graph_sync(self) -> None:
        if self._graph_ready:
            return
        with self.backend.conn_sync() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM ag_catalog.ag_graph WHERE name = %s", (self.graph_name,)
                )
                if not cur.fetchone():
                    cur.execute(f"SELECT create_graph('{self.graph_name}')")
                cur.execute(
                    f"INSERT INTO {self.schema}.t2_graph_registry (graph_name) VALUES (%s) "
                    "ON CONFLICT (graph_name) DO NOTHING",
                    (self.graph_name,),
                )
        self._graph_ready = True

    # ------------------------------------------------------------------
    # GraphStoreBase 接口（同步，走 conn_sync）
    # ------------------------------------------------------------------

    def add_node(
        self,
        node_id: str,
        label: str = "",
        properties: dict[str, Any] | None = None,
        timestamp: str = "",
        record_id: int | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        self._ensure_graph_sync()
        import time as _time
        from .pg_store_backend import _NAME_PAT
        if record_id is None:
            record_id = int(_time.time() * 1000)
        label = (label or "Node").strip()
        if not _NAME_PAT.match(label):
            label = "Node"
        props = dict(properties or {})
        props["__id"] = node_id
        if timestamp:
            props["__ts"] = timestamp
        props["__record_id"] = record_id
        # 将 embedding 存储在节点属性中（JSON 数组格式）
        if embedding is not None:
            props["__embedding"] = embedding
        cy = (
            f"MERGE (n:{label} {{__id: {_cypher_str_lit(node_id)}}}) "
            f"SET n += {_to_cypher_map(props)} "
            f"RETURN n.__id AS id"
        )
        self._cypher_exec_sync(cy, write=True)
        return f"Node '{node_id}' (label={label}) added/updated."

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        properties: dict[str, Any] | None = None,
        timestamp: str = "",
        record_id: int | None = None,
    ) -> str:
        self._ensure_graph_sync()
        import time as _time
        from .pg_store_backend import _NAME_PAT
        if record_id is None:
            record_id = int(_time.time() * 1000)
        rel = (relation or "RELATED").strip()
        if not _NAME_PAT.match(rel):
            rel = "RELATED"
        self._edge_counter += 1
        edge_id = f"e_{self._edge_counter}"
        props = dict(properties or {})
        props["__id"] = edge_id
        if timestamp:
            props["__ts"] = timestamp
        props["__record_id"] = record_id
        cy = (
            f"MERGE (s {{__id: {_cypher_str_lit(source)}}}) "
            f"MERGE (t {{__id: {_cypher_str_lit(target)}}}) "
            f"CREATE (s)-[r:{rel}]->(t) "
            f"SET r += {_to_cypher_map(props)} "
            f"RETURN r.__id AS id"
        )
        self._cypher_exec_sync(cy, write=True)
        return f"Edge {edge_id}: {source} --[{rel}]--> {target}"

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        self._ensure_graph_sync()
        cy = (
            f"MATCH (n {{__id: {_cypher_str_lit(node_id)}}}) "
            f"RETURN labels(n)[0] AS label, properties(n) AS props"
        )
        rows = self._cypher_exec_sync(cy)
        if not rows:
            return None
        r = rows[0]
        return _parse_node_row(node_id, {
            "label": _unwrap_agtype(r["label"]),
            "props": _unwrap_agtype(r["props"]) or {},
        })

    def delete_node(self, node_id: str) -> str:
        self._ensure_graph_sync()
        self._cypher_exec_sync(
            f"MATCH (n {{__id: {_cypher_str_lit(node_id)}}}) DETACH DELETE n",
            write=True,
        )
        return f"Node '{node_id}' deleted."

    def delete_edge(self, edge_id: str) -> str:
        self._ensure_graph_sync()
        self._cypher_exec_sync(
            f"MATCH ()-[r {{__id: {_cypher_str_lit(edge_id)}}}]->() DELETE r",
            write=True,
        )
        return f"Edge '{edge_id}' deleted."

    def get_neighbors(
        self,
        node_id: str,
        relation: str | None = None,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        self._ensure_graph_sync()
        from .pg_store_backend import _NAME_PAT
        rel_part = f":{relation}" if relation and _NAME_PAT.match(relation) else ""
        node_match = f"(n {{__id: {_cypher_str_lit(node_id)}}})"
        if direction == "out":
            pattern = f"{node_match}-[r{rel_part}]->(m)"
        elif direction == "in":
            pattern = f"{node_match}<-[r{rel_part}]-(m)"
        else:
            pattern = f"{node_match}-[r{rel_part}]-(m)"
        cy = (
            f"MATCH {pattern} "
            f"RETURN type(r) AS rel, properties(r) AS edge_props, "
            f"       labels(m)[0] AS m_label, properties(m) AS m_props"
        )
        rows = self._cypher_exec_sync(cy)
        out: list[dict[str, Any]] = []
        for r in rows:
            rel_name = _unwrap_agtype(r["rel"])
            edge_props = _unwrap_agtype(r["edge_props"]) or {}
            m_label = _unwrap_agtype(r["m_label"]) or ""
            m_props = _unwrap_agtype(r["m_props"]) or {}
            out.append({
                "edge": {
                    "id": edge_props.get("__id", ""),
                    "relation": rel_name,
                    "properties": {k: v for k, v in edge_props.items() if not k.startswith("__")},
                },
                "node": {
                    "id": m_props.get("__id", ""),
                    "label": m_label,
                    "properties": {k: v for k, v in m_props.items() if not k.startswith("__")},
                },
            })
        return out

    def search_nodes(
        self,
        label: str | None = None,
        properties_filter: dict[str, Any] | None = None,
        keyword: str | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_graph_sync()
        from .pg_store_backend import _NAME_PAT
        label_part = f":{label}" if label and _NAME_PAT.match(label) else ""
        cy = (
            f"MATCH (n{label_part}) "
            f"RETURN labels(n)[0] AS label, properties(n) AS props LIMIT 100"
        )
        rows = self._cypher_exec_sync(cy)
        all_nodes = []
        for r in rows:
            props = _unwrap_agtype(r["props"]) or {}
            lab = _unwrap_agtype(r["label"]) or ""
            all_nodes.append({"label": lab, "props": props})
        # 属性过滤
        if properties_filter:
            all_nodes = [
                n for n in all_nodes
                if all(
                    {k: v for k, v in n["props"].items() if not k.startswith("__")}.get(k) == v
                    for k, v in properties_filter.items()
                )
            ]
        if keyword:
            kw = keyword.lower()
            all_nodes = [
                n for n in all_nodes
                if kw in (json.dumps(n["props"], ensure_ascii=False) + " " + n["label"]).lower()
            ]
        return [_parse_node_row(n["props"].get("__id", ""), n) for n in all_nodes]

    def search_edges(
        self,
        relation: str | None = None,
        source: str | None = None,
        target: str | None = None,
        keyword: str | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_graph_sync()
        from .pg_store_backend import _NAME_PAT
        rel_part = f":{relation}" if relation and _NAME_PAT.match(relation) else ""
        s_part = f"{{__id: {_cypher_str_lit(source)}}}" if source else ""
        t_part = f"{{__id: {_cypher_str_lit(target)}}}" if target else ""
        cy = (
            f"MATCH (s {s_part})-[r{rel_part}]->(t {t_part}) "
            f"RETURN type(r) AS rel, properties(r) AS rprops, "
            f"       properties(s) AS sprops, properties(t) AS tprops "
            f"LIMIT 200"
        )
        rows = self._cypher_exec_sync(cy)
        out = []
        for r in rows:
            rprops = _unwrap_agtype(r["rprops"]) or {}
            sprops = _unwrap_agtype(r["sprops"]) or {}
            tprops = _unwrap_agtype(r["tprops"]) or {}
            edge_item = {
                "id": rprops.get("__id", ""),
                "source": sprops.get("__id", ""),
                "target": tprops.get("__id", ""),
                "relation": _unwrap_agtype(r["rel"]) or "",
                "properties": {k: v for k, v in rprops.items() if not k.startswith("__")},
            }
            # 关键词过滤
            if keyword:
                keyword_lower = keyword.lower()
                searchable = f"{edge_item['source']} {edge_item['target']} {edge_item['relation']} {str(edge_item['properties'])}".lower()
                if keyword_lower not in searchable:
                    continue
            out.append(edge_item)
        return out

    def get_subgraph(self, node_id: str, depth: int = 2) -> dict[str, Any]:
        self._ensure_graph_sync()
        depth = max(1, min(int(depth), 4))
        cy = (
            f"MATCH p = (n {{__id: {_cypher_str_lit(node_id)}}})-[*1..{depth}]-(m) "
            f"RETURN nodes(p) AS ns, relationships(p) AS es LIMIT 50"
        )
        rows = self._cypher_exec_sync(cy)
        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[str, dict[str, Any]] = {}
        for r in rows:
            for n in _unwrap_agtype(r["ns"]) or []:
                p = (n.get("properties") or {}) if isinstance(n, dict) else {}
                nid = p.get("__id", "")
                if nid:
                    nodes[nid] = {
                        "id": nid,
                        "label": (n.get("label") if isinstance(n, dict) else "") or "",
                        "properties": {k: v for k, v in p.items() if not k.startswith("__")},
                    }
            for e in _unwrap_agtype(r["es"]) or []:
                p = (e.get("properties") or {}) if isinstance(e, dict) else {}
                eid = p.get("__id", "")
                if eid:
                    edges[eid] = {
                        "id": eid,
                        "relation": (e.get("label") if isinstance(e, dict) else "") or "",
                        "properties": {k: v for k, v in p.items() if not k.startswith("__")},
                    }
        return {"center": node_id, "nodes": list(nodes.values()), "edges": list(edges.values())}

    def get_stats(self) -> dict[str, Any]:
        try:
            self._ensure_graph_sync()
        except Exception:
            return {"total_nodes": 0, "total_edges": 0, "node_labels": {}, "relation_types": {}}
        rows_n = self._cypher_exec_sync("MATCH (n) RETURN labels(n)[0] AS lab, count(n) AS c")
        rows_e = self._cypher_exec_sync("MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS c")
        node_labels = {
            _unwrap_agtype(r["lab"]) or "?": int(_unwrap_agtype(r["c"]) or 0)
            for r in rows_n
        }
        relation_types = {
            _unwrap_agtype(r["rel"]) or "?": int(_unwrap_agtype(r["c"]) or 0)
            for r in rows_e
        }
        return {
            "total_nodes": sum(node_labels.values()),
            "total_edges": sum(relation_types.values()),
            "node_labels": node_labels,
            "relation_types": relation_types,
        }

    # ------------------------------------------------------------------
    # Embedding 相关方法（PG 版：embedding 存储在节点属性 __embedding 中）
    # ------------------------------------------------------------------

    def set_node_embedding(self, node_id: str, embedding: list[float]) -> str:
        """为已有节点设置嵌入向量（存储在节点属性 __embedding 中）。"""
        self._ensure_graph_sync()
        cy = (
            f"MATCH (n {{__id: {_cypher_str_lit(node_id)}}}) "
            f"SET n.__embedding = {_to_cypher_value(embedding)} "
            f"RETURN n.__id AS id"
        )
        rows = self._cypher_exec_sync(cy, write=True)
        if not rows:
            return f"Node '{node_id}' not found."
        return f"Embedding set for node '{node_id}'."

    def get_node_embedding(self, node_id: str) -> list[float] | None:
        """获取节点的嵌入向量。"""
        self._ensure_graph_sync()
        cy = (
            f"MATCH (n {{__id: {_cypher_str_lit(node_id)}}}) "
            f"RETURN n.__embedding AS emb"
        )
        rows = self._cypher_exec_sync(cy)
        if not rows:
            return None
        emb = _unwrap_agtype(rows[0]["emb"])
        return emb if isinstance(emb, list) else None

    def set_edge_embedding(self, edge_id: str, embedding: list[float]) -> str:
        """为边设置嵌入向量（存储在边属性 __embedding 中）。"""
        self._ensure_graph_sync()
        cy = (
            f"MATCH ()-[r {{__id: {_cypher_str_lit(edge_id)}}}]->() "
            f"SET r.__embedding = {_to_cypher_value(embedding)} "
            f"RETURN r.__id AS id"
        )
        rows = self._cypher_exec_sync(cy, write=True)
        if not rows:
            return f"Edge '{edge_id}' not found."
        return f"Embedding set for edge '{edge_id}'."

    def get_edge_embedding(self, edge_id: str) -> list[float] | None:
        """获取边的嵌入向量。"""
        self._ensure_graph_sync()
        cy = (
            f"MATCH ()-[r {{__id: {_cypher_str_lit(edge_id)}}}]->() "
            f"RETURN r.__embedding AS emb"
        )
        rows = self._cypher_exec_sync(cy)
        if not rows:
            return None
        emb = _unwrap_agtype(rows[0]["emb"])
        return emb if isinstance(emb, list) else None

    def search_nodes_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """基于向量相似度搜索节点（PG 版：从节点属性中读取 __embedding 计算余弦相似度）。

        注意：AGE 不原生支持向量运算，此处拉取所有含 embedding 的节点到内存中计算。
        大规模场景建议使用 VectorStore 做语义检索。
        """
        import math

        if not query_embedding:
            return []

        self._ensure_graph_sync()
        cy = (
            "MATCH (n) WHERE n.__embedding IS NOT NULL "
            "RETURN n.__id AS id, labels(n)[0] AS label, "
            "properties(n) AS props, n.__embedding AS emb LIMIT 500"
        )
        rows = self._cypher_exec_sync(cy)

        scored: list[tuple[float, dict[str, Any]]] = []
        for r in rows:
            emb = _unwrap_agtype(r["emb"])
            if not isinstance(emb, list) or len(emb) != len(query_embedding):
                continue
            # 余弦相似度
            dot = sum(a * b for a, b in zip(query_embedding, emb))
            norm_q = math.sqrt(sum(a * a for a in query_embedding)) or 1e-10
            norm_e = math.sqrt(sum(b * b for b in emb)) or 1e-10
            sim = dot / (norm_q * norm_e)
            if sim >= threshold:
                props = _unwrap_agtype(r["props"]) or {}
                node_id = _unwrap_agtype(r["id"]) or props.get("__id", "")
                lab = _unwrap_agtype(r["label"]) or ""
                scored.append((sim, {
                    "id": node_id,
                    "label": lab,
                    "properties": {k: v for k, v in props.items() if not k.startswith("__")},
                    "similarity": sim,
                }))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

    # ------------------------------------------------------------------
    # 时序查询
    # ------------------------------------------------------------------

    def search_by_time(
        self,
        time_query: str,
        node_id: str | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        """基于时间范围搜索图中的节点和边（PG 版）。

        使用 Cypher 查询 + 内存过滤实现。
        """
        self._ensure_graph_sync()
        from .pg_store_backend import _NAME_PAT

        time_start, time_end = GraphStore._parse_time_range(time_query)

        # 搜索节点
        label_part = f":{label}" if label and _NAME_PAT.match(label) else ""
        if node_id:
            cy_nodes = (
                f"MATCH (n{label_part} {{__id: {_cypher_str_lit(node_id)}}}) "
                f"RETURN labels(n)[0] AS label, properties(n) AS props LIMIT 200"
            )
        else:
            cy_nodes = (
                f"MATCH (n{label_part}) "
                f"RETURN labels(n)[0] AS label, properties(n) AS props LIMIT 200"
            )
        rows_n = self._cypher_exec_sync(cy_nodes)
        matched_nodes = []
        for r in rows_n:
            props = _unwrap_agtype(r["props"]) or {}
            occurred_at = props.get("occurred_at", "")
            if occurred_at and GraphStore._time_in_range(occurred_at, time_start, time_end):
                matched_nodes.append(_parse_node_row(props.get("__id", ""), {
                    "label": _unwrap_agtype(r["label"]) or "",
                    "props": props,
                }))

        # 搜索边
        if node_id:
            cy_edges = (
                f"MATCH (s)-[r]-(t) WHERE s.__id = {_cypher_str_lit(node_id)} OR t.__id = {_cypher_str_lit(node_id)} "
                f"RETURN type(r) AS rel, properties(r) AS rprops, "
                f"properties(s) AS sprops, properties(t) AS tprops LIMIT 200"
            )
        else:
            cy_edges = (
                "MATCH (s)-[r]->(t) "
                "RETURN type(r) AS rel, properties(r) AS rprops, "
                "properties(s) AS sprops, properties(t) AS tprops LIMIT 200"
            )
        rows_e = self._cypher_exec_sync(cy_edges)
        matched_edges = []
        for r in rows_e:
            rprops = _unwrap_agtype(r["rprops"]) or {}
            occurred_at = rprops.get("occurred_at", "")
            if occurred_at and GraphStore._time_in_range(occurred_at, time_start, time_end):
                sprops = _unwrap_agtype(r["sprops"]) or {}
                tprops = _unwrap_agtype(r["tprops"]) or {}
                matched_edges.append({
                    "id": rprops.get("__id", ""),
                    "source": sprops.get("__id", ""),
                    "target": tprops.get("__id", ""),
                    "relation": _unwrap_agtype(r["rel"]) or "",
                    "properties": {k: v for k, v in rprops.items() if not k.startswith("__")},
                })

        return {"nodes": matched_nodes, "edges": matched_edges}

    # ------------------------------------------------------------------
    # 灵活查询
    # ------------------------------------------------------------------

    def execute_query(self, query_str: str) -> str:
        """执行类 Cypher 查询语句（简化版，供模型灵活使用）。

        PG 版直接委托给 cypher_read（同步版本），支持更丰富的 Cypher 语法。
        """
        query_lower = query_str.lower().strip()

        if "all nodes" in query_lower or query_lower == "show nodes":
            rows = self._cypher_exec_sync(
                "MATCH (n) RETURN labels(n)[0] AS label, properties(n) AS props LIMIT 50"
            )
            nodes = [_parse_node_row(
                (_unwrap_agtype(r["props"]) or {}).get("__id", ""),
                {"label": _unwrap_agtype(r["label"]) or "", "props": _unwrap_agtype(r["props"]) or {}}
            ) for r in rows]
            return json.dumps(nodes, ensure_ascii=False, indent=2)
        elif "all edges" in query_lower or query_lower == "show edges":
            rows = self._cypher_exec_sync(
                "MATCH (s)-[r]->(t) RETURN type(r) AS rel, properties(r) AS rprops, "
                "properties(s) AS sprops, properties(t) AS tprops LIMIT 50"
            )
            edges = []
            for r in rows:
                rprops = _unwrap_agtype(r["rprops"]) or {}
                sprops = _unwrap_agtype(r["sprops"]) or {}
                tprops = _unwrap_agtype(r["tprops"]) or {}
                edges.append({
                    "id": rprops.get("__id", ""),
                    "source": sprops.get("__id", ""),
                    "target": tprops.get("__id", ""),
                    "relation": _unwrap_agtype(r["rel"]) or "",
                    "properties": {k: v for k, v in rprops.items() if not k.startswith("__")},
                })
            return json.dumps(edges, ensure_ascii=False, indent=2)
        elif "stats" in query_lower:
            return json.dumps(self.get_stats(), ensure_ascii=False, indent=2)
        else:
            # 尝试作为 Cypher 查询执行
            try:
                rows = self._cypher_exec_sync(query_str)
                result = [{k: _unwrap_agtype(v) for k, v in r.items()} for r in rows]
                return json.dumps(result, ensure_ascii=False, indent=2)
            except Exception as e:
                stats = self.get_stats()
                return (
                    f"Query execution failed: {e}. "
                    f"Use search_nodes/search_edges/get_neighbors for structured queries. "
                    f"Available nodes: {stats['total_nodes']}, edges: {stats['total_edges']}"
                )

    # ------------------------------------------------------------------
    # 兼容字段（内存版有 _nodes / _edges；这里给快照式 lazy property）
    # ------------------------------------------------------------------

    @property
    def _nodes(self) -> dict[str, dict[str, Any]]:
        try: self._ensure_graph_sync()
        except Exception: return {}
        rows = self._cypher_exec_sync(
            "MATCH (n) RETURN labels(n)[0] AS label, properties(n) AS props LIMIT 500"
        )
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            props = _unwrap_agtype(r["props"]) or {}
            nid = props.get("__id", "")
            if nid:
                out[nid] = {
                    "id": nid,
                    "label": _unwrap_agtype(r["label"]) or "",
                    "properties": {k: v for k, v in props.items() if not k.startswith("__")},
                    "created_at": props.get("__ts", ""),
                    "updated_at": props.get("__ts", ""),
                }
        return out

    @property
    def _edges(self) -> list[dict[str, Any]]:
        try: self._ensure_graph_sync()
        except Exception: return []
        rows = self._cypher_exec_sync(
            "MATCH (s)-[r]->(t) RETURN type(r) AS rel, properties(r) AS rprops, "
            "properties(s) AS sprops, properties(t) AS tprops LIMIT 500"
        )
        out = []
        for r in rows:
            rprops = _unwrap_agtype(r["rprops"]) or {}
            sprops = _unwrap_agtype(r["sprops"]) or {}
            tprops = _unwrap_agtype(r["tprops"]) or {}
            out.append({
                "id": rprops.get("__id", ""),
                "source": sprops.get("__id", ""),
                "target": tprops.get("__id", ""),
                "relation": _unwrap_agtype(r["rel"]) or "",
                "properties": {k: v for k, v in rprops.items() if not k.startswith("__")},
                "created_at": rprops.get("__ts", ""),
            })
        return out

    # ------------------------------------------------------------------
    # 扩展能力（CypherCapableMixin）
    # ------------------------------------------------------------------

    async def cypher_read(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        limit: int = 100,
        timeout_s: float = 5.0,
    ) -> list[dict[str, Any]]:
        """只读 Cypher（拒绝写动词，自动注入 LIMIT）。"""
        bad = re.search(
            r"\b(create|merge|set|delete|remove|drop|detach)\b",
            query, re.IGNORECASE,
        )
        if bad:
            raise PermissionError(
                f"cypher_read: write keyword '{bad.group(0)}' not allowed; use cypher_write."
            )
        q = query.strip().rstrip(";")
        q = _rewrite_order_by_aliases(q)
        if not re.search(r"\blimit\s+\d+\b", q, re.IGNORECASE):
            q = f"{q} LIMIT {int(limit)}"
        rows = await self._cypher_exec(q, params, write=False, timeout_s=timeout_s)
        return [{k: _unwrap_agtype(v) for k, v in r.items()} for r in rows]

    async def cypher_write(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        """可写 Cypher（锁定在本 namespace 的 graph 内）。"""
        rows = await self._cypher_exec(query, params, write=True, timeout_s=timeout_s)
        return {"rows_returned": len(rows), "graph": self.graph_name}

    # ------------------------------------------------------------------
    # 底层 Cypher 执行（async + sync）
    # ------------------------------------------------------------------

    def _cypher_exec_sync(
        self,
        cypher: str,
        write: bool = False,
        timeout_s: float = 5.0,
    ) -> list[dict[str, Any]]:
        ret = _extract_return_aliases(cypher)
        col_defs = ", ".join(f'"{a}" agtype' for a in ret) if ret else "result agtype"
        sql_q = (
            f"SELECT * FROM ag_catalog.cypher('{self.graph_name}', "
            f"$cypherq${cypher}$cypherq$, %s::agtype) "
            f"AS ({col_defs})"
        )
        with self.backend.conn_sync() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SET LOCAL statement_timeout = {int(timeout_s * 1000)}")
                if not write:
                    cur.execute("SET LOCAL transaction_read_only = on")
                cur.execute(sql_q, ("{}",))
                if cur.description is None:
                    return []
                return cur.fetchall()

    async def _cypher_exec(
        self,
        cypher: str,
        params: dict[str, Any] | None,
        *,
        write: bool,
        timeout_s: float = 5.0,
    ) -> list[dict[str, Any]]:
        ret = _extract_return_aliases(cypher)
        col_defs = ", ".join(f'"{a}" agtype' for a in ret) if ret else "result agtype"
        params_json = json.dumps(params or {}, ensure_ascii=False)
        sql_q = (
            f"SELECT * FROM ag_catalog.cypher('{self.graph_name}', "
            f"$cypherq${cypher}$cypherq$, %s::agtype) "
            f"AS ({col_defs})"
        )
        async with await self.backend.conn() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"SET LOCAL statement_timeout = {int(timeout_s * 1000)}")
                if not write:
                    await cur.execute("SET LOCAL transaction_read_only = on")
                await cur.execute(sql_q, (params_json,))
                if cur.description is None:
                    return []
                return await cur.fetchall()


# ---------------------------------------------------------------------------
# 私有工具函数（两个实现类共用，或供 stores_pg 复用）
# ---------------------------------------------------------------------------

def _cypher_str_lit(s: str) -> str:
    """把 Python 字符串转成 Cypher 字符串字面量（含转义）。"""
    if s is None:
        return "null"
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _to_cypher_value(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return _cypher_str_lit(v)
    if isinstance(v, dict):
        return _to_cypher_map(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_to_cypher_value(x) for x in v) + "]"
    return _cypher_str_lit(str(v))


def _to_cypher_map(d: dict[str, Any]) -> str:
    """把 Python dict 转成 Cypher map 字面量（递归处理嵌套）。"""
    if not isinstance(d, dict):
        return "{}"
    parts: list[str] = []
    for k, v in d.items():
        key_lit = k if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k) else f"`{k}`"
        parts.append(f"{key_lit}: {_to_cypher_value(v)}")
    return "{" + ", ".join(parts) + "}"


def _unwrap_agtype(v: Any) -> Any:
    """把 AGE 返回的 agtype 字符串解析成 Python 原生值。

    剥掉所有非 string-literal 内的 ``::<ident>`` 类型标注后 json.loads。
    失败时降级返回原字符串。
    """
    if v is None:
        return None
    if not isinstance(v, str):
        return v
    s = v.strip()
    if not s:
        return s

    out: list[str] = []
    i = 0
    n = len(s)
    in_str = False
    while i < n:
        ch = s[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(s[i + 1])
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == ":" and i + 1 < n and s[i + 1] == ":":
            j = i + 2
            while j < n and (s[j].isalnum() or s[j] == "_"):
                j += 1
            if j > i + 2:
                i = j
                continue
        out.append(ch)
        i += 1

    cleaned = "".join(out).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
            try:
                return json.loads(cleaned)
            except Exception:
                return cleaned[1:-1]
        return cleaned


def _parse_node_row(node_id: str, row: dict[str, Any]) -> dict[str, Any]:
    props = row.get("props") or {}
    if not isinstance(props, dict):
        props = {}
    return {
        "id": node_id or props.get("__id", ""),
        "label": row.get("label") or "",
        "properties": {k: v for k, v in props.items() if not k.startswith("__")},
        "created_at": props.get("__ts", ""),
        "updated_at": props.get("__ts", ""),
    }


def _extract_return_aliases(cypher: str) -> list[str]:
    """从 Cypher 末段 RETURN ... 提取列别名（string-literal + bracket aware）。

    例：``MATCH (n) RETURN n.id AS id, count(*) AS c`` → ``["id", "c"]``
    """
    positions: list[int] = []
    in_str_kind: str | None = None
    i = 0
    s_lower = cypher.lower()
    while i < len(cypher):
        ch = cypher[i]
        if in_str_kind:
            if ch == "\\" and i + 1 < len(cypher):
                i += 2
                continue
            if ch == in_str_kind:
                in_str_kind = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_str_kind = ch
            i += 1
            continue
        if (s_lower[i:i+6] == "return"
            and (i == 0 or not (cypher[i-1].isalnum() or cypher[i-1] == "_"))
            and (i + 6 == len(cypher) or not (cypher[i+6].isalnum() or cypher[i+6] == "_"))):
            positions.append(i)
            i += 6
            continue
        i += 1
    if not positions:
        return []
    ret_start = positions[-1] + len("return")

    stop_keywords = ("order by", "skip", "limit", "union")
    end = len(cypher)
    in_str_kind = None
    depth = 0
    i = ret_start
    while i < len(cypher):
        ch = cypher[i]
        if in_str_kind:
            if ch == "\\" and i + 1 < len(cypher):
                i += 2
                continue
            if ch == in_str_kind:
                in_str_kind = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_str_kind = ch
            i += 1
            continue
        if ch in "([{":
            depth += 1
            i += 1
            continue
        if ch in ")]}":
            depth -= 1
            i += 1
            continue
        if depth == 0:
            low = s_lower[i:]
            for kw in stop_keywords:
                if low.startswith(kw):
                    before_ok = (i == 0 or not (cypher[i-1].isalnum() or cypher[i-1] == "_"))
                    after_idx = i + len(kw)
                    after_ok = (after_idx == len(cypher)
                                or not (cypher[after_idx].isalnum() or cypher[after_idx] == "_"))
                    if before_ok and after_ok:
                        end = i
                        break
            if end < len(cypher):
                break
        i += 1
    body = cypher[ret_start:end].strip().rstrip(";").rstrip()

    # 顶层逗号切
    parts: list[str] = []
    cur_chars: list[str] = []
    depth = 0
    in_str_kind = None
    i = 0
    while i < len(body):
        ch = body[i]
        if in_str_kind:
            cur_chars.append(ch)
            if ch == "\\" and i + 1 < len(body):
                cur_chars.append(body[i + 1])
                i += 2
                continue
            if ch == in_str_kind:
                in_str_kind = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_str_kind = ch
            cur_chars.append(ch)
            i += 1
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur_chars).strip())
            cur_chars = []
        else:
            cur_chars.append(ch)
        i += 1
    if cur_chars:
        parts.append("".join(cur_chars).strip())

    aliases: list[str] = []
    for idx, p in enumerate(parts):
        if not p:
            aliases.append(f"col_{idx}")
            continue
        m = re.search(r"\bas\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", p, re.IGNORECASE)
        if m:
            aliases.append(m.group(1))
            continue
        stripped = p.strip()
        simple = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)", stripped)
        if simple:
            aliases.append(simple.group(1))
            continue
        prop = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.([A-Za-z_][A-Za-z0-9_]*)", stripped)
        if prop:
            aliases.append(prop.group(1))
            continue
        func = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", stripped)
        if func:
            aliases.append(func.group(1))
            continue
        tok = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", stripped)
        aliases.append(tok[-1] if tok else f"col_{idx}")

    seen: set[str] = set()
    out: list[str] = []
    for a in aliases:
        base = a
        j = 1
        while a in seen:
            j += 1
            a = f"{base}_{j}"
        seen.add(a)
        out.append(a)
    return out


def _rewrite_order_by_aliases(cypher: str) -> str:
    """规避 AGE 的 ORDER BY alias 限制：把别名替换回原表达式。"""
    s_lower = cypher.lower()

    def _find_top_kw(start: int, end_limit: int, kw: str) -> int:
        kw_l = kw.lower()
        depth = 0
        in_kind: str | None = None
        j = start
        while j < end_limit:
            c = cypher[j]
            if in_kind:
                if c == "\\" and j + 1 < end_limit:
                    j += 2
                    continue
                if c == in_kind:
                    in_kind = None
                j += 1
                continue
            if c in ('"', "'"):
                in_kind = c
                j += 1
                continue
            if c in "([{":
                depth += 1
                j += 1
                continue
            if c in ")]}":
                depth -= 1
                j += 1
                continue
            if depth == 0 and s_lower[j:j + len(kw_l)] == kw_l:
                before_ok = (j == 0 or not (cypher[j - 1].isalnum() or cypher[j - 1] == "_"))
                after_idx = j + len(kw_l)
                after_ok = (after_idx == end_limit
                            or not (cypher[after_idx].isalnum() or cypher[after_idx] == "_"))
                if before_ok and after_ok:
                    return j
            j += 1
        return -1

    ret_positions: list[int] = []
    in_str_kind: str | None = None
    i = 0
    while i < len(cypher):
        ch = cypher[i]
        if in_str_kind:
            if ch == "\\" and i + 1 < len(cypher):
                i += 2
                continue
            if ch == in_str_kind:
                in_str_kind = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_str_kind = ch
            i += 1
            continue
        if (s_lower[i:i + 6] == "return"
            and (i == 0 or not (cypher[i - 1].isalnum() or cypher[i - 1] == "_"))
            and (i + 6 == len(cypher) or not (cypher[i + 6].isalnum() or cypher[i + 6] == "_"))):
            ret_positions.append(i)
            i += 6
            continue
        i += 1
    if not ret_positions:
        return cypher
    ret_start = ret_positions[-1] + len("return")

    ob_start = _find_top_kw(ret_start, len(cypher), "order by")
    if ob_start < 0:
        return cypher
    ob_clause_start = ob_start + len("order by")

    end_candidates = [
        pos for kw in ("skip", "limit", "union")
        if (pos := _find_top_kw(ob_clause_start, len(cypher), kw)) >= 0
    ]
    ob_clause_end = min(end_candidates) if end_candidates else len(cypher)
    ob_body = cypher[ob_clause_start:ob_clause_end]
    if not re.search(r"[A-Za-z_]", ob_body):
        return cypher

    # 解析 RETURN body → alias -> expr
    ret_body = cypher[ret_start:ob_start].strip().rstrip(";").rstrip()
    parts: list[str] = []
    cur_chars: list[str] = []
    depth = 0
    in_kind: str | None = None
    i = 0
    while i < len(ret_body):
        ch = ret_body[i]
        if in_kind:
            cur_chars.append(ch)
            if ch == "\\" and i + 1 < len(ret_body):
                cur_chars.append(ret_body[i + 1])
                i += 2
                continue
            if ch == in_kind:
                in_kind = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_kind = ch
            cur_chars.append(ch)
            i += 1
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur_chars).strip())
            cur_chars = []
        else:
            cur_chars.append(ch)
        i += 1
    if cur_chars:
        parts.append("".join(cur_chars).strip())

    alias_to_expr: dict[str, str] = {}
    for p in parts:
        m = re.search(r"^(.*?)\s+as\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", p, re.IGNORECASE | re.DOTALL)
        if m:
            expr = m.group(1).strip()
            alias = m.group(2)
            if expr:
                alias_to_expr[alias] = expr
    if not alias_to_expr:
        return cypher

    # 重写 ORDER BY 子句
    ob_parts: list[str] = []
    cur_chars = []
    depth = 0
    in_kind = None
    i = 0
    while i < len(ob_body):
        ch = ob_body[i]
        if in_kind:
            cur_chars.append(ch)
            if ch == "\\" and i + 1 < len(ob_body):
                cur_chars.append(ob_body[i + 1])
                i += 2
                continue
            if ch == in_kind:
                in_kind = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_kind = ch
            cur_chars.append(ch)
            i += 1
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            ob_parts.append("".join(cur_chars))
            cur_chars = []
        else:
            cur_chars.append(ch)
        i += 1
    if cur_chars:
        ob_parts.append("".join(cur_chars))

    new_ob_parts: list[str] = []
    changed = False
    for op in ob_parts:
        stripped = op.strip()
        m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(\s+(?:ASC|DESC))?", stripped, re.IGNORECASE)
        if m and m.group(1) in alias_to_expr:
            expr = alias_to_expr[m.group(1)]
            direction = m.group(2) or ""
            lead = op[:len(op) - len(op.lstrip())] or " "
            trail = op[len(op.rstrip()):]
            new_ob_parts.append(f"{lead}{expr}{direction}{trail}")
            changed = True
        else:
            new_ob_parts.append(op)

    if not changed:
        return cypher
    return cypher[:ob_clause_start] + ",".join(new_ob_parts) + cypher[ob_clause_end:]
