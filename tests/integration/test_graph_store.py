"""AgePgGraphStore 集成测试。

测试目标：验证 AgePgGraphStore 在真实 PostgreSQL（含 Apache AGE 扩展）上的
图操作功能，包括节点/边的 CRUD、邻居查询、子图提取、Cypher 读写等。

运行方式：
    # 确保本地 PG 容器已启动（含 AGE 扩展）
    cd /data/home/limaoqiu/llm_gateway
    docker compose -f docker/pg/docker-compose.yaml up -d

    # 运行测试
    pytest tests/integration/test_graph_store.py -v
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from storage.graph_stores import AgePgGraphStore


# ============================================================
# 节点操作测试
# ============================================================


class TestGraphStoreNode:
    """测试 AgePgGraphStore 的节点 CRUD 操作。"""

    def test_add_node_basic(self, age_graph_store: AgePgGraphStore) -> None:
        """基本写入：创建一个带 label 和 properties 的节点。"""
        result = age_graph_store.add_node(
            node_id="alice",
            label="Person",
            properties={"name": "Alice", "age": 30},
            timestamp="2024-01-01T00:00:00Z",
        )

        assert "alice" in result.lower() or "added" in result.lower() or "created" in result.lower()

    def test_add_node_update_existing(self, age_graph_store: AgePgGraphStore) -> None:
        """对已存在的节点再次 add_node 应执行 MERGE（更新而非重复创建）。"""
        age_graph_store.add_node(
            node_id="bob",
            label="Person",
            properties={"name": "Bob", "age": 25},
        )

        # 再次写入，更新属性
        age_graph_store.add_node(
            node_id="bob",
            label="Person",
            properties={"name": "Bob", "age": 26, "city": "Beijing"},
        )

        node = age_graph_store.get_node("bob")
        assert node is not None
        assert node["properties"]["age"] == 26
        assert node["properties"]["city"] == "Beijing"

    def test_get_node_existing(self, age_graph_store: AgePgGraphStore) -> None:
        """获取已存在的节点，应返回完整信息。"""
        age_graph_store.add_node(
            node_id="charlie",
            label="Developer",
            properties={"lang": "Python", "level": "senior"},
        )

        node = age_graph_store.get_node("charlie")

        assert node is not None
        assert node["id"] == "charlie"
        assert node["label"] == "Developer"
        assert node["properties"]["lang"] == "Python"
        assert node["properties"]["level"] == "senior"

    def test_get_node_nonexistent(self, age_graph_store: AgePgGraphStore) -> None:
        """获取不存在的节点应返回 None。"""
        node = age_graph_store.get_node("nonexistent_node_xyz")
        assert node is None

    def test_delete_node(self, age_graph_store: AgePgGraphStore) -> None:
        """删除节点后应无法再获取到。"""
        age_graph_store.add_node(node_id="to_delete", label="Temp")

        result = age_graph_store.delete_node("to_delete")
        assert "deleted" in result.lower()

        node = age_graph_store.get_node("to_delete")
        assert node is None

    def test_delete_node_cascades_edges(self, age_graph_store: AgePgGraphStore) -> None:
        """删除节点时应同时删除关联的边（DETACH DELETE）。"""
        age_graph_store.add_node(node_id="n1", label="Node")
        age_graph_store.add_node(node_id="n2", label="Node")
        age_graph_store.add_edge(source="n1", target="n2", relation="CONNECTS")

        # 删除 n1，关联的边也应被删除
        age_graph_store.delete_node("n1")

        # n2 应该没有邻居了
        neighbors = age_graph_store.get_neighbors("n2")
        assert len(neighbors) == 0


# ============================================================
# 边操作测试
# ============================================================


class TestGraphStoreEdge:
    """测试 AgePgGraphStore 的边 CRUD 操作。"""

    def test_add_edge_basic(self, age_graph_store: AgePgGraphStore) -> None:
        """基本写入：创建两个节点之间的边。"""
        age_graph_store.add_node(node_id="src", label="Person")
        age_graph_store.add_node(node_id="dst", label="Company")

        result = age_graph_store.add_edge(
            source="src",
            target="dst",
            relation="WORKS_AT",
            properties={"since": "2020"},
            timestamp="2024-01-01T00:00:00Z",
        )

        assert "src" in result or "WORKS_AT" in result or "Edge" in result

    def test_add_edge_auto_creates_nodes(self, age_graph_store: AgePgGraphStore) -> None:
        """如果源/目标节点不存在，MERGE 应自动创建。"""
        result = age_graph_store.add_edge(
            source="auto_src",
            target="auto_dst",
            relation="KNOWS",
        )

        assert "Edge" in result or "auto_src" in result

        # 验证节点被自动创建
        neighbors = age_graph_store.get_neighbors("auto_src", direction="out")
        assert len(neighbors) >= 1

    def test_add_edge_with_properties(self, age_graph_store: AgePgGraphStore) -> None:
        """边应能携带自定义属性。"""
        age_graph_store.add_node(node_id="a", label="Person")
        age_graph_store.add_node(node_id="b", label="Person")

        age_graph_store.add_edge(
            source="a",
            target="b",
            relation="FRIEND",
            properties={"weight": 0.9, "context": "同事"},
        )

        edges = age_graph_store.search_edges(relation="FRIEND", source="a")
        assert len(edges) >= 1
        edge = edges[0]
        assert edge["relation"] == "FRIEND"
        assert edge["properties"]["weight"] == 0.9
        assert edge["properties"]["context"] == "同事"

    def test_delete_edge(self, age_graph_store: AgePgGraphStore) -> None:
        """按 edge_id 删除边。"""
        age_graph_store.add_node(node_id="x", label="Node")
        age_graph_store.add_node(node_id="y", label="Node")

        result = age_graph_store.add_edge(
            source="x", target="y", relation="LINK",
        )

        # 从 search_edges 获取 edge_id
        edges = age_graph_store.search_edges(source="x", target="y")
        assert len(edges) >= 1
        edge_id = edges[0]["id"]

        # 删除
        del_result = age_graph_store.delete_edge(edge_id)
        assert "deleted" in del_result.lower()

        # 验证已删除
        edges_after = age_graph_store.search_edges(source="x", target="y")
        assert len(edges_after) == 0


# ============================================================
# 邻居查询测试
# ============================================================


class TestGraphStoreNeighbors:
    """测试 AgePgGraphStore.get_neighbors() 方法。"""

    def _setup_triangle(self, store: AgePgGraphStore) -> None:
        """创建一个三角形图：A→B, A→C, B→C"""
        store.add_node(node_id="A", label="Person", properties={"name": "Alice"})
        store.add_node(node_id="B", label="Person", properties={"name": "Bob"})
        store.add_node(node_id="C", label="Person", properties={"name": "Charlie"})
        store.add_edge(source="A", target="B", relation="KNOWS")
        store.add_edge(source="A", target="C", relation="LIKES")
        store.add_edge(source="B", target="C", relation="KNOWS")

    def test_neighbors_out(self, age_graph_store: AgePgGraphStore) -> None:
        """direction='out' 应只返回出边指向的节点。"""
        self._setup_triangle(age_graph_store)

        neighbors = age_graph_store.get_neighbors("A", direction="out")

        neighbor_ids = {n["node"]["id"] for n in neighbors}
        assert "B" in neighbor_ids
        assert "C" in neighbor_ids

    def test_neighbors_in(self, age_graph_store: AgePgGraphStore) -> None:
        """direction='in' 应只返回入边来源的节点。"""
        self._setup_triangle(age_graph_store)

        neighbors = age_graph_store.get_neighbors("C", direction="in")

        neighbor_ids = {n["node"]["id"] for n in neighbors}
        assert "A" in neighbor_ids
        assert "B" in neighbor_ids

    def test_neighbors_both(self, age_graph_store: AgePgGraphStore) -> None:
        """direction='both' 应返回所有方向的邻居。"""
        self._setup_triangle(age_graph_store)

        neighbors = age_graph_store.get_neighbors("B", direction="both")

        neighbor_ids = {n["node"]["id"] for n in neighbors}
        # B 有入边来自 A，出边到 C
        assert "A" in neighbor_ids
        assert "C" in neighbor_ids

    def test_neighbors_filter_by_relation(self, age_graph_store: AgePgGraphStore) -> None:
        """指定 relation 应只返回该类型边的邻居。"""
        self._setup_triangle(age_graph_store)

        neighbors = age_graph_store.get_neighbors("A", relation="KNOWS", direction="out")

        neighbor_ids = {n["node"]["id"] for n in neighbors}
        assert "B" in neighbor_ids
        assert "C" not in neighbor_ids  # C 是 LIKES 关系

    def test_neighbors_empty(self, age_graph_store: AgePgGraphStore) -> None:
        """孤立节点应返回空列表。"""
        age_graph_store.add_node(node_id="lonely", label="Isolated")

        neighbors = age_graph_store.get_neighbors("lonely")
        assert neighbors == []


# ============================================================
# 搜索测试
# ============================================================


class TestGraphStoreSearch:
    """测试 AgePgGraphStore 的搜索功能。"""

    def _setup_data(self, store: AgePgGraphStore) -> None:
        """创建测试数据。"""
        store.add_node(node_id="py", label="Language", properties={"name": "Python", "type": "dynamic"})
        store.add_node(node_id="rs", label="Language", properties={"name": "Rust", "type": "static"})
        store.add_node(node_id="go", label="Language", properties={"name": "Go", "type": "static"})
        store.add_node(node_id="alice", label="Developer", properties={"name": "Alice"})
        store.add_edge(source="alice", target="py", relation="USES")
        store.add_edge(source="alice", target="rs", relation="LEARNING")

    def test_search_nodes_by_label(self, age_graph_store: AgePgGraphStore) -> None:
        """按 label 搜索节点。"""
        self._setup_data(age_graph_store)

        results = age_graph_store.search_nodes(label="Language")

        assert len(results) == 3
        ids = {r["id"] for r in results}
        assert ids == {"py", "rs", "go"}

    def test_search_nodes_by_keyword(self, age_graph_store: AgePgGraphStore) -> None:
        """按关键字搜索节点（模糊匹配 properties）。"""
        self._setup_data(age_graph_store)

        results = age_graph_store.search_nodes(keyword="Python")

        assert len(results) >= 1
        assert any(r["id"] == "py" for r in results)

    def test_search_nodes_by_label_and_keyword(self, age_graph_store: AgePgGraphStore) -> None:
        """同时指定 label + keyword 应取交集。"""
        self._setup_data(age_graph_store)

        results = age_graph_store.search_nodes(label="Language", keyword="static")

        ids = {r["id"] for r in results}
        assert "rs" in ids
        assert "go" in ids
        assert "py" not in ids  # Python 是 dynamic

    def test_search_edges_by_relation(self, age_graph_store: AgePgGraphStore) -> None:
        """按 relation 搜索边。"""
        self._setup_data(age_graph_store)

        results = age_graph_store.search_edges(relation="USES")

        assert len(results) >= 1
        assert results[0]["source"] == "alice"
        assert results[0]["target"] == "py"

    def test_search_edges_by_source(self, age_graph_store: AgePgGraphStore) -> None:
        """按 source 搜索边。"""
        self._setup_data(age_graph_store)

        results = age_graph_store.search_edges(source="alice")

        assert len(results) == 2
        relations = {r["relation"] for r in results}
        assert "USES" in relations
        assert "LEARNING" in relations

    def test_search_edges_by_source_and_target(self, age_graph_store: AgePgGraphStore) -> None:
        """同时指定 source + target 精确查找。"""
        self._setup_data(age_graph_store)

        results = age_graph_store.search_edges(source="alice", target="py")

        assert len(results) == 1
        assert results[0]["relation"] == "USES"


# ============================================================
# 子图提取测试
# ============================================================


class TestGraphStoreSubgraph:
    """测试 AgePgGraphStore.get_subgraph() 方法。"""

    def _setup_chain(self, store: AgePgGraphStore) -> None:
        """创建链式图：A→B→C→D"""
        store.add_node(node_id="A", label="Node")
        store.add_node(node_id="B", label="Node")
        store.add_node(node_id="C", label="Node")
        store.add_node(node_id="D", label="Node")
        store.add_edge(source="A", target="B", relation="NEXT")
        store.add_edge(source="B", target="C", relation="NEXT")
        store.add_edge(source="C", target="D", relation="NEXT")

    def test_subgraph_depth_1(self, age_graph_store: AgePgGraphStore) -> None:
        """depth=1 应只返回直接邻居。"""
        self._setup_chain(age_graph_store)

        subgraph = age_graph_store.get_subgraph("B", depth=1)

        node_ids = {n["id"] for n in subgraph["nodes"]}
        # B 的直接邻居是 A 和 C
        assert "A" in node_ids
        assert "C" in node_ids
        assert "D" not in node_ids  # depth=1 到不了 D

    def test_subgraph_depth_2(self, age_graph_store: AgePgGraphStore) -> None:
        """depth=2 应返回两跳内的所有节点。"""
        self._setup_chain(age_graph_store)

        subgraph = age_graph_store.get_subgraph("B", depth=2)

        node_ids = {n["id"] for n in subgraph["nodes"]}
        # B → A (1跳), B → C (1跳), C → D (2跳)
        assert "A" in node_ids
        assert "C" in node_ids
        assert "D" in node_ids

    def test_subgraph_isolated_node(self, age_graph_store: AgePgGraphStore) -> None:
        """孤立节点的子图应为空（或只有自身）。"""
        age_graph_store.add_node(node_id="isolated", label="Alone")

        subgraph = age_graph_store.get_subgraph("isolated", depth=2)

        # 孤立节点没有路径可走，nodes 和 edges 应为空
        assert len(subgraph["edges"]) == 0


# ============================================================
# 统计信息测试
# ============================================================


class TestGraphStoreStats:
    """测试 AgePgGraphStore.get_stats() 方法。"""

    def test_stats_empty(self, age_graph_store: AgePgGraphStore) -> None:
        """空图的统计信息。"""
        stats = age_graph_store.get_stats()

        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0

    def test_stats_after_operations(self, age_graph_store: AgePgGraphStore) -> None:
        """写入数据后统计信息应正确反映。"""
        age_graph_store.add_node(node_id="n1", label="TypeA")
        age_graph_store.add_node(node_id="n2", label="TypeA")
        age_graph_store.add_node(node_id="n3", label="TypeB")
        age_graph_store.add_edge(source="n1", target="n2", relation="REL_X")
        age_graph_store.add_edge(source="n2", target="n3", relation="REL_Y")

        stats = age_graph_store.get_stats()

        assert stats["total_nodes"] == 3
        assert stats["total_edges"] == 2
        assert stats["node_labels"]["TypeA"] == 2
        assert stats["node_labels"]["TypeB"] == 1
        assert stats["relation_types"]["REL_X"] == 1
        assert stats["relation_types"]["REL_Y"] == 1


# ============================================================
# Cypher 读写测试（CypherCapableMixin）
# ============================================================


class TestGraphStoreCypher:
    """测试 AgePgGraphStore 的 Cypher 扩展能力。"""

    @pytest.mark.asyncio
    async def test_cypher_read_basic(self, age_graph_store: AgePgGraphStore) -> None:
        """cypher_read 应能执行只读查询。"""
        age_graph_store.add_node(node_id="cr1", label="Test", properties={"val": 42})

        results = await age_graph_store.cypher_read(
            "MATCH (n {__id: \"cr1\"}) RETURN n.__id AS id, n.val AS val"
        )

        assert len(results) == 1
        assert results[0]["id"] == "cr1"
        assert results[0]["val"] == 42

    @pytest.mark.asyncio
    async def test_cypher_read_rejects_write(self, age_graph_store: AgePgGraphStore) -> None:
        """cypher_read 应拒绝包含写操作关键字的查询。"""
        with pytest.raises(PermissionError, match="write keyword"):
            await age_graph_store.cypher_read(
                "CREATE (n:Bad {name: 'hacker'})"
            )

    @pytest.mark.asyncio
    async def test_cypher_read_auto_limit(self, age_graph_store: AgePgGraphStore) -> None:
        """cypher_read 应自动注入 LIMIT。"""
        # 写入一些数据
        for i in range(5):
            age_graph_store.add_node(node_id=f"lim_{i}", label="Batch")

        results = await age_graph_store.cypher_read(
            "MATCH (n:Batch) RETURN n.__id AS id",
            limit=3,
        )

        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_cypher_write_basic(self, age_graph_store: AgePgGraphStore) -> None:
        """cypher_write 应能执行写操作。"""
        result = await age_graph_store.cypher_write(
            "CREATE (n:CypherCreated {__id: \"cw1\", name: \"from_cypher\"}) RETURN n.__id AS id"
        )

        assert result["rows_returned"] == 1
        assert result["graph"] == age_graph_store.graph_name

        # 验证节点确实被创建
        node = age_graph_store.get_node("cw1")
        assert node is not None
        assert node["properties"]["name"] == "from_cypher"
