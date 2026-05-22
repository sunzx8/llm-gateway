"""存储后端工厂 — 根据 StorageConfig 显式配置选择 in-memory 或 PostgreSQL 后端。

调用方（典型 :class:`AgentMemoryT2System`）只需要：

    from storage.stores_factory import make_stores

    fs, vec, graph, backend_info = make_stores(
        storage_config=memory_config.storage,
        user_id="user_42",
        embedder=embedder,
    )

返回的 ``vec`` / ``graph`` 实例符合 :mod:`stores_base` 的抽象接口；具体后端
由 ``StorageConfig.backend`` 字段控制（"memory" 或 "pg"）。

PG backend 通过 StorageConfig.pg 显式传入连接配置
（详见 :class:`pg_store_backend.PgBackend.from_config`）

隔离策略：
    每个用户拥有独立的 PostgreSQL schema（如 u_<user_id>），
    schema 内表结构相同但数据完全隔离。
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

from utils.memory_llm_interface import EmbeddingInterface
from .file_system_store import FileSystemStore
from .vector_stores import VectorStore,VectorStoreBase
from .graph_stores import GraphStore,GraphStoreBase
import logger.logger as logger
from config.models import StorageConfig

# ============================================================
# 内存后端全局 store 缓存
# 确保同一 user_id 的并发请求共享同一组 stores，避免数据互不可见
# ============================================================
_memory_store_cache: dict[str, tuple[FileSystemStore, Any, Any, dict[str, Any]]] = {}
_memory_store_lock = threading.Lock()


def _user_id_to_schema(user_id: str) -> str:
    """将 user_id 转换为合法的 PostgreSQL schema 名称。

    规则：
    - 前缀 'u_' 避免与系统 schema 冲突
    - 只保留字母、数字、下划线
    - 长度限制 63 字符（PG 标识符上限）
    """
    safe = re.sub(r"[^A-Za-z0-9_]", "_", user_id)
    schema = f"u_{safe}"
    return schema[:63]


def make_stores(
        storage_config:StorageConfig,
        user_id: str,
        embedder: EmbeddingInterface | None = None,
) -> tuple[FileSystemStore, VectorStoreBase, GraphStoreBase]:
    """构造一组三后端 stores（FS 始终是本地，Vec/Graph 按 backend 选择）。

    对于 memory 后端，同一 user_id 会复用缓存中的 stores 实例，
    确保并发请求之间数据可见性一致。
    """
    # 创建存储目录
    base_dir = str(
        Path(storage_config.memory_fs.root_dir) / user_id
    )
    os.makedirs(base_dir, exist_ok=True)
    fs_path = os.path.join(base_dir, "memory_repo")

    schema = _user_id_to_schema(user_id)

    info: dict[str, Any] = {"backend": storage_config.backend, "schema": schema}

    if storage_config.backend == "memory":
        with _memory_store_lock:
            if user_id in _memory_store_cache:
                logger.debug("make_stores: 复用缓存的 memory stores, user_id=%s", user_id)
                return _memory_store_cache[user_id]
            fs = FileSystemStore(fs_path, enable_git=storage_config.memory_fs.enable_git)
            vec = VectorStore(embedder)
            graph = GraphStore()
            cached = (fs, vec, graph)
            _memory_store_cache[user_id] = cached
            logger.debug("make_stores: 创建并缓存 memory stores, user_id=%s", user_id)
            return cached

    # memory 以外的后端不走缓存
    fs = FileSystemStore(fs_path, enable_git=storage_config.memory_fs.enable_git)

    if storage_config.backend == "pg":
        from .pg_store_backend import PgBackend
        from .vector_stores import PgVectorStore
        from .graph_stores import AgePgGraphStore
        pg = PgBackend.from_config(storage_config.pg)
        info["dsn"] = pg.dsn
        # 安装全局扩展
        pg.ensure_schema_sync()
        # 为用户创建独立 schema 及其内部表
        pg.ensure_user_schema_sync(schema)

        vec = PgVectorStore(pg, schema=schema, embedder=embedder)
        graph = AgePgGraphStore(pg, schema=schema)
        info["graph_name"] = graph.graph_name
        return fs, vec, graph

    raise ValueError(f"unknown MEM_BACKEND={storage_config.backend!r}; expected 'memory' or 'pg'")


def clear_memory_stores() -> int:
    """清空所有内存后端的 store 缓存。

    适用于测试清理、服务重启等场景。

    Returns:
        被清除的缓存条目数。
    """
    with _memory_store_lock:
        count = len(_memory_store_cache)
        _memory_store_cache.clear()
        if count > 0:
            logger.info("clear_memory_stores: 已清除 %d 个用户的 store 缓存", count)
        return count


def evict_memory_stores(user_id: str) -> bool:
    """从缓存中移除指定用户的 memory stores。

    适用于用户数据清理、评测轮次切换等场景。

    Args:
        user_id: 要移除的用户 ID。

    Returns:
        True 表示成功移除，False 表示该用户不在缓存中。
    """
    with _memory_store_lock:
        if user_id in _memory_store_cache:
            del _memory_store_cache[user_id]
            logger.debug("evict_memory_stores: 已移除 user_id=%s 的 store 缓存", user_id)
            return True
        return False


async def cleanup_user(
    user_id: str,
    *,
    storage_config: StorageConfig,
    drop_graph: bool = False,
) -> None:
    """清理指定用户的 PG 数据（in-memory 模式无操作）。

    - 删除用户 schema 下 ``t2_vec_entries`` 中的所有行
    - 可选 drop AGE graph（drop_graph=True）
    - 可选 drop 整个 schema（通过 drop_schema=True）

    适合 eval cleanup / 切换 experiment 时调用，避免 PG 数据无限堆。
    """
    if storage_config.backend != "pg":
        return
    from .pg_store_backend import PgBackend, _safe_graph_name
    schema = _user_id_to_schema(user_id)
    pg = PgBackend.from_config(storage_config.pg)
    async with await pg.conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(f"DELETE FROM {schema}.t2_vec_entries")
            if drop_graph:
                gname = _safe_graph_name(schema)
                # AGE 在 graph 不存在时 drop_graph 会报错；先查
                await cur.execute(
                    "SELECT 1 FROM ag_catalog.ag_graph WHERE name = %s", (gname,),
                )
                if await cur.fetchone():
                    await cur.execute(f"SELECT drop_graph('{gname}', true)")
                await cur.execute(
                    f"DELETE FROM {schema}.t2_graph_registry WHERE graph_name = %s",
                    (gname,),
                )


async def drop_user_schema(user_id: str, *, storage_config: StorageConfig) -> None:
    """彻底删除用户 schema（含所有表和数据）。

    **危险操作**，仅用于用户注销或测试清理。
    """
    if storage_config.backend != "pg":
        return
    from .pg_store_backend import PgBackend, _safe_graph_name, _check_safe_ident
    schema = _user_id_to_schema(user_id)
    _check_safe_ident(schema, "schema")
    pg = PgBackend.from_config(storage_config.pg)
    async with await pg.conn() as conn:
        async with conn.cursor() as cur:
            # 先 drop graph
            gname = _safe_graph_name(schema)
            await cur.execute(
                "SELECT 1 FROM ag_catalog.ag_graph WHERE name = %s", (gname,),
            )
            if await cur.fetchone():
                await cur.execute(f"SELECT drop_graph('{gname}', true)")
            # drop schema cascade
            await cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


# ---------------------------------------------------------------------------
# 反序列化工厂方法 — 从序列化配置重建 Store 实例
# ---------------------------------------------------------------------------


def deserialize_vec_store(config: dict[str, Any]) -> Any:
    """从序列化配置字典反序列化创建 VectorStore 实例。

    根据 config['backend'] 字段选择具体实现：
    - 'memory': 使用 VectorStore.from_config() 恢复内存版实例（含全部数据）
    - 'pg': 使用 PgVectorStore.from_config() 创建 PG 版实例

    Args:
        config: 由 VectorStore.serialize() 或 PgVectorStore.serialize() 生成的配置字典。

    Returns:
        VectorStoreBase 实例。
    """
    backend = config.get("backend", "memory")
    if backend == "pg":
        from .vector_stores import PgVectorStore
        return PgVectorStore.from_config(config)
    else:
        return VectorStore.from_config(config)


def deserialize_graph_store(config: dict[str, Any]) -> Any:
    """从序列化配置字典反序列化创建 GraphStore 实例。

    根据 config['backend'] 字段选择具体实现：
    - 'memory': 使用 GraphStore.from_config() 恢复内存版实例（含全部数据）
    - 'pg': 使用 AgePgGraphStore.from_config() 创建 PG 版实例

    Args:
        config: 由 GraphStore.serialize() 或 AgePgGraphStore.serialize() 生成的配置字典。

    Returns:
        GraphStoreBase 实例。
    """
    backend = config.get("backend", "memory")
    if backend == "pg":
        from .graph_stores import AgePgGraphStore
        return AgePgGraphStore.from_config(config)
    else:
        return GraphStore.from_config(config)


def deserialize_fs_store(config: dict[str, Any]) -> FileSystemStore:
    """从序列化配置字典反序列化创建 FileSystemStore 实例。

    Args:
        config: 由 FileSystemStore.serialize() 生成的配置字典。

    Returns:
        FileSystemStore 实例。
    """
    return FileSystemStore.from_config(config)
