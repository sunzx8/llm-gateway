"""测试共享 Fixtures。

PG 连接策略：
- 如果设置了环境变量 TEST_PG_DSN，则直连该 DSN（适合本地已有 PG 的场景）
- 否则使用 testcontainers 自动拉起临时 PG 容器

本地 PG 容器连接信息（来自 docker/pg/docker-compose.yaml）：
    用户: limaoqiu / 密码: limaoqiu / 数据库: t2 / 端口: 5432
    TEST_PG_DSN=postgresql://limaoqiu:limaoqiu@127.0.0.1:5432/t2
"""

from __future__ import annotations

import os
import sys
import hashlib
from pathlib import Path

import pytest
import pytest_asyncio

from typing import TYPE_CHECKING,Generator

if TYPE_CHECKING:
    from storage.pg_store_backend import PgBackend  # 仅类型检查时导入

# 将项目根目录加入 sys.path，使 storage 等模块可直接导入
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 默认 DSN：直连本地 docker/pg 容器
# ---------------------------------------------------------------------------
DEFAULT_TEST_PG_DSN = "postgresql://limaoqiu:limaoqiu@127.0.0.1:5432/t2"


def get_test_dsn() -> str:
    """获取测试用 PG DSN。优先使用环境变量，否则使用本地容器默认值。"""
    return os.environ.get("TEST_PG_DSN", DEFAULT_TEST_PG_DSN)


# ---------------------------------------------------------------------------
# Fake Embedder（测试用，不依赖真实 embedding 服务）
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """确定性的假 embedding 接口。

    基于文本内容的 MD5 生成固定 128 维向量，保证：
    - 相同文本 → 相同向量
    - 不同文本 → 不同向量（大概率）
    - 无需网络调用
    """

    DIM = 128

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._text_to_vec(t) for t in texts]

    async def embed_single(self, text: str) -> list[float]:
        return self._text_to_vec(text)

    def _text_to_vec(self, text: str) -> list[float]:
        """将文本确定性地映射为归一化的 128 维向量。"""
        # 用 SHA-256 获取足够的字节
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # 扩展到 128 维：重复 hash
        raw = []
        seed = text.encode("utf-8")
        for i in range(self.DIM):
            b = hashlib.md5(seed + i.to_bytes(2, "big")).digest()[0]
            raw.append(b / 255.0)
        # 归一化
        norm = sum(x * x for x in raw) ** 0.5 or 1e-10
        return [x / norm for x in raw]


# ---------------------------------------------------------------------------
# PgBackend Fixture（module 级别，整个测试模块共享一个连接池）
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pg_backend()->Generator["PgBackend"]:
    """创建 PgBackend 实例，连接到测试 PG。

    scope=module：同一测试文件内的所有用例共享连接池，避免频繁创建/销毁。
    """
    from storage.pg_store_backend import PgBackend

    dsn = get_test_dsn()
    backend = PgBackend(dsn, min_size=1, max_size=4)

    # 安装全局扩展（幂等）
    backend.ensure_schema_sync()

    yield backend

    # 清理：关闭连接池
    backend.close_sync()
    # 清除单例缓存，避免影响其他测试模块
    PgBackend._instances.pop(dsn, None)


# ---------------------------------------------------------------------------
# 测试专用 Schema Fixture
# ---------------------------------------------------------------------------

TEST_SCHEMA = "u_test_integration"


@pytest.fixture(scope="module")
def test_schema(pg_backend)->Generator[str]:
    """为集成测试创建专用 schema（module 级别）。

    测试结束后清空数据（但保留 schema 结构，避免反复 DDL）。
    """
    pg_backend.ensure_user_schema_sync(TEST_SCHEMA)
    yield TEST_SCHEMA

    # module 结束时清空测试数据
    with pg_backend.conn_sync() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {TEST_SCHEMA}.t2_vec_entries")


# ---------------------------------------------------------------------------
# PgVectorStore Fixture（function 级别，每个用例独立）
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def pg_vector_store(pg_backend:PgBackend, test_schema:str):
    """每个测试用例获得一个干净的 PgVectorStore 实例。

    用例开始前清空数据，确保测试之间互不干扰。
    """
    from storage.vector_stores import PgVectorStore

    # 清空上一个用例的残留数据
    with pg_backend.conn_sync() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {test_schema}.t2_vec_entries")

    store = PgVectorStore(pg_backend, schema=test_schema, embedder=FakeEmbedder())
    yield store


# ---------------------------------------------------------------------------
# AgePgGraphStore Fixture（function 级别，每个用例独立）
# ---------------------------------------------------------------------------

@pytest.fixture
def age_graph_store(pg_backend, test_schema):
    """每个测试用例获得一个干净的 AgePgGraphStore 实例。

    用例开始前清空图数据，确保测试之间互不干扰。
    """
    from storage.graph_stores import AgePgGraphStore

    store = AgePgGraphStore(backend=pg_backend, schema=test_schema)
    # 确保图已创建
    store._ensure_graph_sync()

    # 清空上一个用例的残留图数据（删除所有节点和边）
    try:
        store._cypher_exec_sync("MATCH (n) DETACH DELETE n", write=True)
    except Exception:
        pass  # 图为空时可能报错，忽略

    yield store


# ---------------------------------------------------------------------------
# FakeEmbedder Fixture（供单元测试使用）
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_embedder():
    """返回一个 FakeEmbedder 实例。"""
    return FakeEmbedder()
