"""PgVectorStore 集成测试。

测试目标：验证 PgVectorStore 在真实 PostgreSQL（含 pgvector 扩展）上的
基本 CRUD 和语义检索功能。

运行方式：
    # 确保本地 PG 容器已启动
    cd /data/home/limaoqiu/llm_gateway
    docker compose -f docker/pg/docker-compose.yaml up -d

    # 运行测试
    pytest tests/integration/test_pg_vector_store.py -v
"""

from __future__ import annotations
from storage.pg_store_backend import PgVectorStore
import pytest


# ============================================================
# 测试用例
# ============================================================


class TestPgVectorStoreAdd:
    """测试 PgVectorStore.add() 方法。"""

    @pytest.mark.asyncio
    async def test_add_basic(self, pg_vector_store:PgVectorStore):
        """基本写入：传入 texts + metadatas，应返回正确数量的 id。"""
        ids = await pg_vector_store.add(
            collection="facts_test",
            texts=["Python 是一门编程语言", "PostgreSQL 是关系型数据库"],
            metadatas=[
                {"type": "fact", "subject": "python"},
                {"type": "fact", "subject": "postgres"},
            ],
        )

        assert len(ids) == 2
        assert all(id_.startswith("vec_") for id_ in ids)

    @pytest.mark.asyncio
    async def test_add_without_metadatas(self, pg_vector_store:PgVectorStore):
        """metadatas=None 时应自动填充并产生 warning。"""
        ids = await pg_vector_store.add(
            collection="facts_test",
            texts=["hello world", "foo bar"],
            metadatas=None,
        )

        assert len(ids) == 2
        # 应该有 warning 记录
        assert len(pg_vector_store._last_add_warnings) > 0
        assert "missing" in pg_vector_store._last_add_warnings[0].lower()

    @pytest.mark.asyncio
    async def test_add_with_custom_ids(self, pg_vector_store:PgVectorStore):
        """指定自定义 id 时应使用传入的 id。"""
        custom_ids = ["my_id_1", "my_id_2"]
        ids = await pg_vector_store.add(
            collection="facts_test",
            texts=["text1", "text2"],
            metadatas=[{"type": "test"}, {"type": "test"}],
            ids=custom_ids,
        )

        assert ids == custom_ids

    @pytest.mark.asyncio
    async def test_add_upsert_on_conflict(self, pg_vector_store:PgVectorStore):
        """相同 collection + entry_id 应执行 upsert（更新而非报错）。"""
        await pg_vector_store.add(
            collection="facts_test",
            texts=["原始文本"],
            metadatas=[{"type": "fact", "subject": "test"}],
            ids=["upsert_id"],
        )

        # 再次写入相同 id，文本不同
        await pg_vector_store.add(
            collection="facts_test",
            texts=["更新后的文本"],
            metadatas=[{"type": "fact", "subject": "test_updated"}],
            ids=["upsert_id"],
        )

        # 验证只有 1 条记录（upsert 而非 insert 两条）
        stats = pg_vector_store.get_stats()
        assert stats["total_entries"] == 1
        assert stats["collections"]["facts_test"] == 1


class TestPgVectorStoreSearch:
    """测试 PgVectorStore.search() 方法。"""

    @pytest.mark.asyncio
    async def test_search_returns_results(self, pg_vector_store:PgVectorStore):
        """写入数据后，search 应能返回结果。"""
        await pg_vector_store.add(
            collection="facts_test",
            texts=[
                "Python 是一门动态类型的编程语言",
                "PostgreSQL 支持向量检索",
                "Docker 是容器化技术",
            ],
            metadatas=[
                {"type": "fact", "subject": "python"},
                {"type": "fact", "subject": "postgres"},
                {"type": "fact", "subject": "docker"},
            ],
        )

        results = await pg_vector_store.search(
            collection="facts_test",
            query="数据库向量搜索",
            top_k=2,
        )

        assert len(results) > 0
        assert len(results) <= 2
        # 验证返回结构
        for r in results:
            assert "id" in r
            assert "text" in r
            assert "score" in r
            assert "metadata" in r
            assert isinstance(r["score"], float)

    @pytest.mark.asyncio
    async def test_search_with_metadata_filter(self, pg_vector_store:PgVectorStore):
        """metadata_filter 应正确过滤结果。"""
        await pg_vector_store.add(
            collection="facts_test",
            texts=["Python 很好用", "Java 也不错"],
            metadatas=[
                {"type": "fact", "subject": "python", "lang": "python"},
                {"type": "fact", "subject": "java", "lang": "java"},
            ],
        )

        results = await pg_vector_store.search(
            collection="facts_test",
            query="编程语言",
            top_k=10,
            metadata_filter={"lang": "python"},
        )

        # 只应返回 python 相关的
        assert len(results) == 1
        assert results[0]["metadata"]["lang"] == "python"

    @pytest.mark.asyncio
    async def test_search_empty_collection(self, pg_vector_store:PgVectorStore):
        """查询不存在的 collection 应返回空列表。"""
        results = await pg_vector_store.search(
            collection="nonexistent_collection",
            query="anything",
            top_k=5,
        )

        assert results == []


class TestPgVectorStoreDelete:
    """测试 PgVectorStore.delete() 方法。"""

    @pytest.mark.asyncio
    async def test_delete_by_ids(self, pg_vector_store:PgVectorStore):
        """按 id 删除应正确移除指定条目。"""
        ids = await pg_vector_store.add(
            collection="facts_test",
            texts=["要删除的", "要保留的"],
            metadatas=[{"type": "temp"}, {"type": "keep"}],
        )

        result = pg_vector_store.delete("facts_test", [ids[0]])
        assert "1" in result  # 删除了 1 条

        # 验证剩余数据
        stats = pg_vector_store.get_stats()
        assert stats["total_entries"] == 1


class TestPgVectorStoreUpdate:
    """测试 PgVectorStore.update() 方法。"""

    @pytest.mark.asyncio
    async def test_update_text(self, pg_vector_store:PgVectorStore):
        """更新 text 字段。"""
        ids = await pg_vector_store.add(
            collection="facts_test",
            texts=["原始文本"],
            metadatas=[{"type": "fact", "subject": "test"}],
        )

        result = pg_vector_store.update(
            "facts_test", ids[0], new_text="更新后的文本"
        )
        assert "Updated" in result
        assert "1" in result

    @pytest.mark.asyncio
    async def test_update_metadata(self, pg_vector_store:PgVectorStore):
        """更新 metadata 字段（合并而非覆盖）。"""
        ids = await pg_vector_store.add(
            collection="facts_test",
            texts=["测试文本"],
            metadatas=[{"type": "fact", "subject": "test"}],
        )

        pg_vector_store.update(
            "facts_test", ids[0], new_metadata={"priority": "high"}
        )

        # 通过 search 验证 metadata 已合并
        results = await pg_vector_store.search(
            collection="facts_test",
            query="测试",
            metadata_filter={"priority": "high"},
        )
        assert len(results) == 1


class TestPgVectorStoreStats:
    """测试 PgVectorStore.get_stats() 方法。"""

    @pytest.mark.asyncio
    async def test_stats_empty(self, pg_vector_store:PgVectorStore):
        """空 store 的统计信息。"""
        stats = pg_vector_store.get_stats()
        assert stats["total_entries"] == 0
        assert stats["collections"] == {}

    @pytest.mark.asyncio
    async def test_stats_after_add(self, pg_vector_store:PgVectorStore):
        """写入后统计信息应正确反映。"""
        await pg_vector_store.add(
            collection="facts_a",
            texts=["text1", "text2"],
            metadatas=[{"type": "a"}, {"type": "a"}],
        )
        await pg_vector_store.add(
            collection="facts_b",
            texts=["text3"],
            metadatas=[{"type": "b"}],
        )

        stats = pg_vector_store.get_stats()
        assert stats["total_entries"] == 3
        assert stats["collections"]["facts_a"] == 2
        assert stats["collections"]["facts_b"] == 1


class TestPgVectorStoreSearchAll:
    """测试 PgVectorStore.search_all() 跨 collection 检索。"""

    @pytest.mark.asyncio
    async def test_search_all_across_collections(self, pg_vector_store:PgVectorStore):
        """search_all 应跨所有 collection 返回结果。"""
        await pg_vector_store.add(
            collection="facts_python",
            texts=["Python 支持异步编程"],
            metadatas=[{"type": "fact", "subject": "python"}],
        )
        await pg_vector_store.add(
            collection="facts_rust",
            texts=["Rust 是系统编程语言"],
            metadatas=[{"type": "fact", "subject": "rust"}],
        )

        results = await pg_vector_store.search_all(
            query="编程语言",
            top_k=10,
        )

        assert len(results) == 2
        # 结果应包含 collection 字段
        collections = {r["collection"] for r in results}
        assert "facts_python" in collections
        assert "facts_rust" in collections
