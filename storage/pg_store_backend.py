"""PostgreSQL 后端基础设施：连接池单例 + Schema 管理。

本模块职责已收窄为：
- :class:`PgBackend`：连接池单例、Schema DDL、HNSW 索引懒建。
- 工具函数：``_safe_graph_name`` / ``_NAME_PAT`` / ``_check_safe_ident`` 等。

具体 Store 实现已迁移至专属模块：
- :mod:`agent_memory.core.vector_stores` — ``VectorStore`` / ``PgVectorStore``
- :mod:`agent_memory.core.graph_stores`  — ``GraphStore``  / ``AgePgGraphStore``

本模块保留 ``PgVectorStore`` / ``AgePgGraphStore`` 的向后兼容再导出，
现有调用方无需修改。
"""

from __future__ import annotations

import hashlib
import os
import re
import pathlib
from typing import TYPE_CHECKING
import logger.logger as logger

from config.models import PgConfig

# 延迟导入 psycopg/pgvector，让 in-memory 用户在没装这两个包时仍能 import 本文件

import psycopg
from psycopg import sql as pgsql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool

_PG_AVAILABLE = True

try:
    from pgvector.psycopg import register_vector_async
    _PGVECTOR_AVAILABLE = True
except ImportError:  # pragma: no cover
    register_vector_async = None  # type: ignore
    _PGVECTOR_AVAILABLE = False


# ---------------------------------------------------------------------------
# 全局后端 / 连接池单例
# ---------------------------------------------------------------------------


class PgBackend:
    """PG 连接池单例（per process）。

    用法（典型，由 :mod:`stores_factory` 调用）::

        from config.models import PgConfig
        backend = PgBackend.from_config(pg_config)
        await backend.ensure_schema()         # 一次性安装全局扩展
        backend.ensure_user_schema_sync(schema)  # 为用户创建独立 schema
        vec_store = PgVectorStore(backend, schema="u_user_42", embedder=...)
        graph_store = AgePgGraphStore(backend, schema="u_user_42")
    """

    _instances: dict[str, "PgBackend"] = {}

    def __init__(
        self,
        dsn: str,
        *,
        min_size: int = 2,
        max_size: int = 16,
    ):
        # if not _PG_AVAILABLE:
        #     raise RuntimeError(
        #         "psycopg[binary,pool] is required for PgBackend. "
        #         "Install with `pip install 'psycopg[binary,pool]>=3.2' pgvector`."
        #     )
        self.dsn = dsn
        self._pool: "AsyncConnectionPool | None" = None
        # 同步 pool：用于 sync method（list_collections / get_stats / 各种
        # 兼容接口）。async 与 sync 必须分别走不同 pool —— async pool 是
        # event-loop bound，主 loop 之外的 sync wrapper 没法借用。
        self._sync_pool: "ConnectionPool | None" = None
        self._schema_ready: bool = False
        self._min_size = min_size
        self._max_size = max_size

    @classmethod
    def from_config(cls, pg_config: "PgConfig") -> "PgBackend":
        """从 PgConfig 显式构造 PgBackend 实例（带单例缓存）。

        Args:
            pg_config: PostgreSQL 连接配置，来自 StorageConfig.pg。
        """
        dsn = pg_config.dsn
        if not dsn:
            dsn = (
                f"postgresql://{pg_config.user}:{pg_config.password}"
                f"@{pg_config.host}:{pg_config.port}/{pg_config.database}"
            )

        if dsn not in cls._instances:
            cls._instances[dsn] = cls(
                dsn,
                min_size=pg_config.pool_min_size,
                max_size=pg_config.pool_max_size,
            )
        return cls._instances[dsn]

    async def _get_pool(self) -> "AsyncConnectionPool":
        if self._pool is None:
            # configure: register pgvector + AGE search_path on every conn
            async def _conf(conn):
                if _PGVECTOR_AVAILABLE:
                    try:
                        await register_vector_async(conn)
                    except Exception as e:  # pragma: no cover
                        logger.debug("register_vector_async failed: %s", e)
                # AGE 必须 LOAD 'age' 才能调 cypher()；search_path 必须含 ag_catalog
                async with conn.cursor() as cur:
                    await cur.execute("LOAD 'age';")
                    await cur.execute("SET search_path = ag_catalog, public;")

            self._pool = AsyncConnectionPool(
                conninfo=self.dsn,
                min_size=self._min_size,
                max_size=self._max_size,
                open=False,
                configure=_conf,
                kwargs={"row_factory": dict_row, "autocommit": True},
            )
            await self._pool.open(wait=True, timeout=30)
        return self._pool

    async def conn(self):
        """Async context manager — `async with backend.conn() as cur: ...`"""
        pool = await self._get_pool()
        return pool.connection()

    def _get_sync_pool(self) -> "ConnectionPool":
        """同步 pool（按需 lazy 建）。用于 list_collections / get_stats 等 sync method。"""
        if self._sync_pool is None:
            from pgvector.psycopg import register_vector
            def _conf(conn):
                try:
                    register_vector(conn)
                except Exception as e:  # pragma: no cover
                    logger.debug("register_vector failed: %s", e)
                with conn.cursor() as cur:
                    cur.execute("LOAD 'age';")
                    cur.execute("SET search_path = ag_catalog, public;")

            self._sync_pool = ConnectionPool(
                conninfo=self.dsn,
                min_size=1,
                max_size=4,
                open=False,
                configure=_conf,
                kwargs={"row_factory": dict_row, "autocommit": True},
            )
            self._sync_pool.open(wait=True, timeout=30)
        return self._sync_pool

    def conn_sync(self):
        """同步 connection context manager — `with backend.conn_sync() as conn: ...`"""
        return self._get_sync_pool().connection()

    def _read_sql_file(self, filename: str) -> str:
        """读取SQL文件内容。"""
        # 获取项目根目录
        project_root = pathlib.Path(__file__).parent
        sql_file_path = project_root / "pg_tables" / filename
        
        try:
            return sql_file_path.read_text(encoding='utf-8')
        except FileNotFoundError:
            logger.error(f"SQL file {sql_file_path} not found, using fallback DDL")

    def ensure_schema_sync(self) -> None:
        """同步版本的 ensure_schema：安装全局扩展（factory 用）。"""
        if self._schema_ready:
            return
        ddl = self._read_sql_file("tables.sql")
        with self.conn_sync() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
        self._schema_ready = True

    def ensure_user_schema_sync(self, schema: str) -> None:
        """为指定用户创建独立 schema 及其内部表（幂等，同步版）。

        Args:
            schema: 用户 schema 名称（如 'u_user_42'），必须是合法 SQL 标识符。
        """
        _check_safe_ident(schema, "schema")
        ddl_template = self._read_sql_file("user_schema.sql")
        ddl = ddl_template.format(schema=schema)
        with self.conn_sync() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)

    async def ensure_user_schema(self, schema: str) -> None:
        """为指定用户创建独立 schema 及其内部表（幂等，异步版）。

        Args:
            schema: 用户 schema 名称（如 'u_user_42'），必须是合法 SQL 标识符。
        """
        _check_safe_ident(schema, "schema")
        ddl_template = self._read_sql_file("user_schema.sql")
        ddl = ddl_template.format(schema=schema)
        async with await self.conn() as conn:
            async with conn.cursor() as cur:
                await cur.execute(ddl)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        if self._sync_pool is not None:
            self._sync_pool.close()
            self._sync_pool = None

    def close_sync(self) -> None:
        if self._sync_pool is not None:
            self._sync_pool.close()
            self._sync_pool = None

    async def ensure_schema(self) -> None:
        """一次性安装全局扩展（幂等）。用户表结构由 ensure_user_schema() 按需创建。"""
        if self._schema_ready:
            return
        ddl = self._read_sql_file("tables.sql")
        async with await self.conn() as conn:
            async with conn.cursor() as cur:
                await cur.execute(ddl)
        self._schema_ready = True

    async def lazy_create_hnsw(self, schema: str, dim: int) -> None:
        """首次见到 embedding 维度后在指定 schema 下建 HNSW 索引（幂等）。

        Args:
            schema: 用户 schema 名称。
            dim: embedding 维度。
        """
        _check_safe_ident(schema, "schema")
        idx_name = f"t2_vec_hnsw_d{dim}_idx"
        # pgvector HNSW 需要明确维度，所以索引名带维度做隔离
        async with await self.conn() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_class c
                            JOIN pg_namespace n ON n.oid = c.relnamespace
                            WHERE c.relname = '{idx_name}' AND n.nspname = '{schema}'
                        ) THEN
                            BEGIN
                                EXECUTE 'CREATE INDEX {idx_name}
                                    ON {schema}.t2_vec_entries USING hnsw ((embedding::vector({dim})) vector_cosine_ops)
                                    WHERE vector_dims(embedding) = {dim}';
                            EXCEPTION WHEN OTHERS THEN
                                -- 旧版 pgvector 可能不支持 partial HNSW，忽略；查询会回退到顺序扫描
                                RAISE NOTICE 'HNSW index creation skipped: %', SQLERRM;
                            END;
                        END IF;
                    END $$;
                    """
                )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _safe_graph_name(ns: str) -> str:
    """把任意 namespace 字符串映射为合法的 AGE graph 名（小写 + 下划线 + 长度有限）。

    AGE 的 graph_name 实际是 SQL 标识符，必须以字母开头、只含字母数字下划线、
    长度 ≤ 63。我们用 ``t2g_`` 前缀 + ns 的 sha1 前 12 位（保证唯一且短）。
    """
    h = hashlib.sha1(ns.encode("utf-8")).hexdigest()[:12]
    return f"t2g_{h}"


_NAME_PAT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _check_safe_ident(name: str, kind: str) -> None:
    if not _NAME_PAT.match(name) or len(name) > 63:
        raise ValueError(f"invalid {kind}: {name!r}")


def _check_pg_available() -> None:
    """若 psycopg 未安装则抛 RuntimeError（供 vector_stores / graph_stores 调用）。"""
    if not _PG_AVAILABLE:
        raise RuntimeError(
            "psycopg[binary,pool] is required for PG backend. "
            "Install with `pip install 'psycopg[binary,pool]>=3.2' pgvector`."
        )


def _check_pgvector_available() -> None:
    """若 pgvector 未安装则抛 RuntimeError（供 vector_stores 调用）。"""
    if not _PGVECTOR_AVAILABLE:
        raise RuntimeError(
            "pgvector is required for PgVectorStore. "
            "Install with `pip install pgvector`."
        )


# ---------------------------------------------------------------------------
# PgVectorStore / AgePgGraphStore — 向后兼容再导出
# 实现已迁移至 vector_stores.py / graph_stores.py
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# （以下为历史实现，已由上方导入替代，保留注释供 git blame 追溯）
# ---------------------------------------------------------------------------

