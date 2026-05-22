"""存储后端抽象接口 — 让 VectorStore / GraphStore 有可替换实现。

历史上 :mod:`agent_memory.core.stores` 里的 ``VectorStore`` / ``GraphStore``
是直接用 dict 实现的"内存版"。本模块定义抽象基类，让 PG / Qdrant / Neo4j 等
真实后端可以无侵入地替换进来——T2 system 通过 :mod:`stores_factory` 选择具体实现。

设计原则：
1. **方法签名严格对齐**现有内存版（参见 ``stores.VectorStore`` /
   ``stores.GraphStore``）。任何后端实现必须满足这套接口，retrieve/ingest/
   consolidate task 不需要任何分支判断。
2. 接口故意保留**最小公共子集**，不暴露后端特有能力（例如 PG 的
   `sql_query_read` / Cypher）。后端特有能力另开独立 mixin 接口（见
   :class:`SqlCapableMixin` / :class:`CypherCapableMixin`），由 T2 task 在
   工具分发时按 ``isinstance`` 检测后选择是否暴露给 LLM。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

from debug import LoggedMethodsMixin


# ---------------------------------------------------------------------------
# Vector store base
# ---------------------------------------------------------------------------


class VectorStoreBase(ABC, LoggedMethodsMixin):
    """所有 VectorStore 实现必须满足的接口。

    与现存内存版 ``stores.VectorStore`` 行为对齐——method 名 / 参数 / 返回值
    格式一一对应，T2 task 调度层无需关心具体后端。
    """

    _logged_methods: set[str] = {
        'add', 'search', 'search_all', 'search_all_with_embedding',
        'delete', 'update',
        'list_collections', 'create_collection', 'delete_collection', 'get_stats',
    }

    # 由实现类在 add() 之后写入，task 层可读取做 warning 透传
    _last_add_warnings: list[str]

    @abstractmethod
    def list_collections(self) -> list[str]:
        ...

    @abstractmethod
    def create_collection(self, name: str) -> str:
        ...

    @abstractmethod
    def delete_collection(self, name: str) -> str:
        ...

    @abstractmethod
    async def add(
        self,
        collection: str,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        record_id: int | None = None,
    ) -> list[str]:
        ...

    @abstractmethod
    async def search(
        self,
        collection: str,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """返回 ``[{id, text, score, metadata}]`` ——score 越大越相关。"""
        ...

    @abstractmethod
    async def search_all(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """跨所有 collection 检索，结果含 ``collection`` 字段。"""
        ...

    @abstractmethod
    def delete(self, collection: str, ids: list[str]) -> str:
        ...

    @abstractmethod
    def update(
        self,
        collection: str,
        entry_id: str,
        new_text: str | None = None,
        new_metadata: dict[str, Any] | None = None,
    ) -> str:
        ...

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """返回 ``{total_entries, collections: {name: count, ...}}``。"""
        ...

    @abstractmethod
    async def search_all_with_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """使用预计算的 embedding 跨所有 collection 检索（避免重复 embed 调用）。

        返回 ``[{id, text, score, metadata, collection}]``。
        """
        ...

    @abstractmethod
    def serialize(self) -> dict[str, Any]:
        """序列化为 JSON 可存储的格式。"""
        ...


# ---------------------------------------------------------------------------
# Graph store base
# ---------------------------------------------------------------------------


class GraphStoreBase(ABC, LoggedMethodsMixin):
    """所有 GraphStore 实现必须满足的接口。

    保持与内存版 ``stores.GraphStore`` 一致的 method 集合。
    """

    _logged_methods: set[str] = {
        'add_node', 'add_edge', 'get_node', 'delete_node', 'delete_edge',
        'get_neighbors', 'search_nodes', 'search_edges', 'get_subgraph', 'get_stats',
        'set_node_embedding', 'get_node_embedding',
        'set_edge_embedding', 'get_edge_embedding',
        'search_nodes_by_embedding', 'search_by_time', 'execute_query',
    }

    @abstractmethod
    def add_node(
        self,
        node_id: str,
        label: str = "",
        properties: dict[str, Any] | None = None,
        timestamp: str = "",
        record_id: int | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        ...

    @abstractmethod
    def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        properties: dict[str, Any] | None = None,
        timestamp: str = "",
        record_id: int | None = None,
    ) -> str:
        ...

    @abstractmethod
    def get_node(self, node_id: str) -> dict[str, Any] | None:
        ...

    @abstractmethod
    def delete_node(self, node_id: str) -> str:
        ...

    @abstractmethod
    def delete_edge(self, edge_id: str) -> str:
        ...

    @abstractmethod
    def get_neighbors(
        self,
        node_id: str,
        relation: str | None = None,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def search_nodes(
        self,
        label: str | None = None,
        properties_filter: dict[str, Any] | None = None,
        keyword: str | None = None,
    ) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def search_edges(
        self,
        relation: str | None = None,
        source: str | None = None,
        target: str | None = None,
        keyword: str | None = None,
    ) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def get_subgraph(self, node_id: str, depth: int = 2) -> dict[str, Any]:
        ...

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def serialize(self) -> dict[str, Any]:
        ...

    # ------------------------------------------------------------------
    # Embedding 相关方法
    # ------------------------------------------------------------------

    @abstractmethod
    def set_node_embedding(self, node_id: str, embedding: list[float]) -> str:
        """为已有节点设置嵌入向量。"""
        ...

    @abstractmethod
    def get_node_embedding(self, node_id: str) -> list[float] | None:
        """获取节点的嵌入向量。"""
        ...

    @abstractmethod
    def set_edge_embedding(self, edge_id: str, embedding: list[float]) -> str:
        """为边设置嵌入向量。"""
        ...

    @abstractmethod
    def get_edge_embedding(self, edge_id: str) -> list[float] | None:
        """获取边的嵌入向量。"""
        ...

    @abstractmethod
    def search_nodes_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """基于向量相似度搜索节点。

        返回按相似度降序排列的节点列表，每个节点附带 similarity 字段。
        """
        ...

    # ------------------------------------------------------------------
    # 时序查询
    # ------------------------------------------------------------------

    @abstractmethod
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

        返回 ``{"nodes": [...], "edges": [...]}``。
        """
        ...

    # ------------------------------------------------------------------
    # 灵活查询
    # ------------------------------------------------------------------

    @abstractmethod
    def execute_query(self, query_str: str) -> str:
        """执行类 Cypher 查询语句（简化版，供模型灵活使用）。"""
        ...

# ---------------------------------------------------------------------------
# Capability mixins — 后端特有能力（PG 才有；in-memory 不实现）
# ---------------------------------------------------------------------------


class SqlCapableMixin(Protocol):
    """实现该 Protocol 的 store 表示后端支持任意 SQL 查询（仅 PG）。

    T2 task 在 ``execute_tool`` 里用 ``isinstance(self.vec, SqlCapableMixin)``
    或 ``hasattr(store, 'sql_query_read')`` 检测后才暴露 ``sql_query_read``
    工具给 LLM。
    """

    async def sql_query_read(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        limit: int = 200,
        timeout_s: float = 5.0,
    ) -> list[dict[str, Any]]:
        ...


class CypherCapableMixin(Protocol):
    """实现该 Protocol 的 graph store 表示支持完整 OpenCypher（PG+AGE / Neo4j）。"""

    async def cypher_read(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        limit: int = 100,
        timeout_s: float = 5.0,
    ) -> list[dict[str, Any]]:
        ...

    async def cypher_write(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...


class ScrollCapableMixin(Protocol):
    """支持按 filter 翻页流式遍历（PG / Qdrant 等真后端）。"""

    async def scroll(
        self,
        collection: str | None,
        metadata_filter: dict[str, Any] | None = None,
        cursor: str | None = None,
        page_size: int = 100,
    ) -> dict[str, Any]:
        """返回 ``{items: [...], next_cursor: '...' | None}``。"""
        ...
