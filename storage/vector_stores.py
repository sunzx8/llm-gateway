"""向量存储后端 — 统一入口。

包含：
- :class:`VectorStoreBase`  — 抽象接口（从 stores_base 再导出）
- :class:`VectorStore`      — 基于内存的实现（开发 / 测试 / 轻量部署）
- :class:`PgVectorStore`    — 基于 pgvector 的生产级实现

调用方只需从本模块导入，无需关心具体实现来自哪个子模块::

    from agent_memory.core.vector_stores import VectorStore, PgVectorStore
"""

from __future__ import annotations

import re
from typing import Any

from .stores_base import VectorStoreBase
import logger.logger as logger
from .pg_store_backend import PgBackend
from utils.memory_llm_interface import EmbeddingInterface
from .file_system_store import SOURCE_SESSIONS_DIR

__all__ = [
    "VectorStoreBase",
    "VectorStore",
    "PgVectorStore",
]


# ---------------------------------------------------------------------------
# 内存版实现
# ---------------------------------------------------------------------------

class VectorStore(VectorStoreBase):
    """向量数据库存储后端（内存版）。

    基于内存中的向量索引实现，支持：
    - 多 collection（模型自行决定如何分类存储）
    - 语义检索（cosine similarity）
    - 元数据过滤
    """

    def __init__(self, embedding_interface=None):
        self.embedder = embedding_interface
        # collection_name -> list of {id, text, embedding, metadata}
        self._collections: dict[str, list[dict[str, Any]]] = {}
        self._id_counter = 0
        self._last_add_warnings: list[str] = []

    # ------------------------------------------------------------------
    # 序列化 / 反序列化
    # ------------------------------------------------------------------

    def serialize(self) -> dict[str, Any]:
        """将 VectorStore 序列化为可 JSON 化的配置字典。

        内存版需要序列化所有内部数据（collections、id_counter、embedding 配置），
        以便在消费脚本中完整重建实例。

        Returns:
            包含重建实例所需全部参数和数据的字典。
        """
        data: dict[str, Any] = {
            "backend": "memory",
            "id_counter": self._id_counter,
            "collections": self._collections,
        }
        if self.embedder is not None and hasattr(self.embedder, "serialize"):
            data["embedding_config"] = self.embedder.serialize()
        return data

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "VectorStore":
        """从配置字典反序列化创建 VectorStore 实例。

        内存版会恢复所有内部数据（collections、id_counter），
        并根据 embedding_config 重建 embedder 实例。

        Args:
            config: 序列化配置字典，包含 'collections'、'id_counter' 和可选的 'embedding_config' 字段。

        Returns:
            VectorStore 实例。
        """
        embedder = None
        embedding_config = config.get("embedding_config")
        if embedding_config:
            from utils.memory_llm_interface import EmbeddingInterface
            embedder = EmbeddingInterface(embedding_config)
        instance = cls(embedding_interface=embedder)
        instance._id_counter = config.get("id_counter", 0)
        instance._collections = config.get("collections", {})
        return instance

    def list_collections(self) -> list[str]:
        return list(self._collections.keys())

    def create_collection(self, name: str) -> str:
        if name not in self._collections:
            self._collections[name] = []
            return f"Collection '{name}' created."
        return f"Collection '{name}' already exists."

    def delete_collection(self, name: str) -> str:
        if name in self._collections:
            del self._collections[name]
            return f"Collection '{name}' deleted."
        return f"Collection '{name}' not found."

    async def add(
        self,
        collection: str,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        record_id: int | None = None,
    ) -> list[str]:
        """向 collection 添加文本（自动计算 embedding）。

        弱契约 + 自动兜底：缺失 / 空 metadata 自动填占位，不报错。
        仅当 metadatas 长度不匹配 texts，或 source 字段格式明显错误时抛 ValueError。

        Args:
            record_id: 操作记录ID（毫秒时间戳），未指定时自动生成。
        """
        logger.info(f"VectorStore.add of  {collection} for {texts}")
        n = len(texts)
        warnings: list[str] = []
        if metadatas is None:
            warnings.append(f"metadatas was missing — auto-filled {n} placeholder dicts")
            metadatas = [{} for _ in range(n)]
        if len(metadatas) != n:
            raise ValueError(
                f"vec_add: metadatas length ({len(metadatas)}) must match texts length ({n})."
            )

        coll_suffix = _collection_suffix(collection)
        _source_pat = re.compile(rf"^{SOURCE_SESSIONS_DIR}/[^\s:]+:\d+$")
        normalised: list[dict[str, Any]] = []
        n_empty = 0
        n_partial = 0
        for i, m in enumerate(metadatas):
            if not isinstance(m, dict) or not m:
                n_empty += 1
                normalised.append({
                    "type": "auto",
                    "subject": coll_suffix or "unknown",
                    "auto_filled": True,
                })
                continue
            src = m.get("source")
            if isinstance(src, str) and src.startswith(f"{SOURCE_SESSIONS_DIR}/"):
                if not _source_pat.match(src):
                    raise ValueError(
                        f"vec_add: metadatas[{i}].source must match "
                        f"'{SOURCE_SESSIONS_DIR}/<sid>.jsonl:<line>' with numeric line number, got: {src!r}"
                    )
            if not any(k in m for k in ("source", "type", "subject", "topic")):
                m = dict(m)
                m["auto_filled"] = True
                m.setdefault("subject", coll_suffix or "unknown")
                m.setdefault("type", "auto")
                n_partial += 1
            normalised.append(m)

        if n_empty:
            warnings.append(
                f"{n_empty}/{n} entries had empty/missing metadata — "
                f"auto-filled with {{type:'auto', subject:'{coll_suffix or 'unknown'}', "
                f"auto_filled:true}}; please supply real metadata next time"
            )
        if n_partial:
            warnings.append(
                f"{n_partial}/{n} entries lacked any of (source/type/subject/topic) — "
                f"auto-tagged with type='auto'"
            )

        if collection not in self._collections:
            self._collections[collection] = []
        metadatas = normalised
        self._last_add_warnings = warnings

        embeddings = await _compute_embeddings(self.embedder, texts)

        # 如果未指定 record_id，自动生成当前毫秒时间戳
        if record_id is None:
            import time
            record_id = int(time.time() * 1000)

        added_ids = []
        for i, text in enumerate(texts):
            self._id_counter += 1
            entry_id = ids[i] if ids and i < len(ids) else f"vec_{self._id_counter}"
            self._collections[collection].append({
                "id": entry_id,
                "text": text,
                "embedding": embeddings[i] if i < len(embeddings) else [],
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "record_id": record_id,
            })
            added_ids.append(entry_id)
        return added_ids

    async def search(
        self,
        collection: str,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        logger.info(f"VectorStore.search of  {collection} for {query}")
        if collection not in self._collections:
            return []
        entries = self._collections[collection]
        if metadata_filter:
            entries = [
                e for e in entries
                if all(e["metadata"].get(k) == v for k, v in metadata_filter.items())
            ]
        if not entries:
            return []

        query_embedding = await _compute_single_embedding(self.embedder, query)
        if query_embedding:
            scored = [
                (_cosine_similarity(query_embedding, e["embedding"]), e)
                for e in entries
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            results = scored[:top_k]
        else:
            results = [(0.0, e) for e in entries[:top_k]]

        return [
            {"id": e["id"], "text": e["text"], "score": s, "metadata": e["metadata"]}
            for s, e in results
        ]

    async def search_all(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        all_results = []
        for collection in self._collections:
            results = await self.search(collection, query, top_k=top_k, metadata_filter=metadata_filter)
            for r in results:
                r["collection"] = collection
            all_results.extend(results)
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    async def search_all_with_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """使用预计算的 embedding 跨所有 collection 检索（避免重复 embed 调用）。

        Args:
            query_embedding: 预计算的 query 向量。
            top_k: 返回条数。
            metadata_filter: 元数据过滤条件。

        Returns:
            检索结果列表 [{id, text, score, metadata, collection}]。
        """
        all_results = []
        for coll_name, entries in self._collections.items():
            filtered = entries
            if metadata_filter:
                filtered = [
                    e for e in entries
                    if all(e["metadata"].get(k) == v for k, v in metadata_filter.items())
                ]
            if not filtered:
                continue

            for entry in filtered:
                entry_emb = entry.get("embedding")
                if not entry_emb:
                    continue
                score = _cosine_similarity(query_embedding, entry_emb)
                all_results.append({
                    "id": entry["id"],
                    "text": entry["text"],
                    "score": score,
                    "metadata": entry["metadata"],
                    "collection": coll_name,
                })

        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    def delete(self, collection: str, ids: list[str]) -> str:
        if collection not in self._collections:
            return f"Collection '{collection}' not found."
        before = len(self._collections[collection])
        self._collections[collection] = [
            e for e in self._collections[collection] if e["id"] not in set(ids)
        ]
        return f"Deleted {before - len(self._collections[collection])} entries from '{collection}'."

    def update(
        self,
        collection: str,
        entry_id: str,
        new_text: str | None = None,
        new_metadata: dict[str, Any] | None = None,
    ) -> str:
        if collection not in self._collections:
            return f"Collection '{collection}' not found."
        for entry in self._collections[collection]:
            if entry["id"] == entry_id:
                if new_text is not None:
                    entry["text"] = new_text
                if new_metadata is not None:
                    entry["metadata"].update(new_metadata)
                return f"Updated entry '{entry_id}'."
        return f"Entry '{entry_id}' not found."

    def get_stats(self) -> dict[str, Any]:
        return {
            "collections": {name: len(entries) for name, entries in self._collections.items()},
            "total_entries": sum(len(e) for e in self._collections.values()),
        }


# ---------------------------------------------------------------------------
# PG 版实现（延迟导入 psycopg / pgvector，不影响内存版用户）
# ---------------------------------------------------------------------------

class PgVectorStore(VectorStoreBase):
    """基于 pgvector 的向量存储（生产级）。

    额外实现 :class:`stores_base.SqlCapableMixin` 和
    :class:`stores_base.ScrollCapableMixin`，由 T2 task 层按 ``isinstance``
    检测后选择是否暴露对应工具给 LLM。

    隔离策略：每个用户拥有独立的 PostgreSQL schema，表结构相同但数据完全隔离。

    Args:
        backend: :class:`stores_pg.PgBackend` 共享实例。
        schema: 用户独立的 PostgreSQL schema 名称（如 'u_user_42'）。
        embedder: ``EmbeddingInterface`` 实例。
    """

    def __init__(self, backend: PgBackend, schema: str, embedder:EmbeddingInterface=None):
        from .pg_store_backend import _check_pg_available, _check_pgvector_available, _check_safe_ident
        _check_pg_available()
        _check_pgvector_available()
        _check_safe_ident(schema, "schema")
        from .stores_base import SqlCapableMixin, ScrollCapableMixin
        # 动态注册 mixin（避免多重继承声明时的循环依赖）
        self.__class__ = type(
            "PgVectorStore",
            (PgVectorStore, SqlCapableMixin, ScrollCapableMixin),
            {},
        )
        self.backend = backend
        self.schema = schema
        self.embedder = embedder
        self._id_counter = 0
        self._last_add_warnings: list[str] = []
        self._known_dim: int | None = None

    # ------------------------------------------------------------------
    # 序列化 / 反序列化
    # ------------------------------------------------------------------

    def serialize(self) -> dict[str, Any]:
        """将 PgVectorStore 序列化为可 JSON 化的配置字典。

        PG 版只需序列化连接参数、schema 和 embedding 配置，数据存储在 PG 中无需序列化。

        Returns:
            包含重建实例所需全部参数的字典。
        """
        data: dict[str, Any] = {
            "backend": "pg",
            "dsn": self.backend.dsn,
            "schema": self.schema,
        }
        if self.embedder is not None and hasattr(self.embedder, "serialize"):
            data["embedding_config"] = self.embedder.serialize()
        return data

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PgVectorStore":
        """从配置字典反序列化创建 PgVectorStore 实例。

        Args:
            config: 序列化配置字典，包含 'dsn'、'schema' 和可选的 'embedding_config' 字段。

        Returns:
            PgVectorStore 实例。
        """
        from .pg_store_backend import PgBackend
        dsn = config.get("dsn", "")
        schema = config.get("schema", "")
        backend = PgBackend(dsn)
        embedder = None
        embedding_config = config.get("embedding_config")
        if embedding_config:
            from utils.memory_llm_interface import EmbeddingInterface
            embedder = EmbeddingInterface(embedding_config)
        return cls(backend=backend, schema=schema, embedder=embedder)

    def list_collections(self) -> list[str]:
        with self.backend.conn_sync() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT DISTINCT collection FROM {self.schema}.t2_vec_entries ORDER BY collection",
                )
                return [r["collection"] for r in cur.fetchall()]

    def create_collection(self, name: str) -> str:
        return f"Collection '{name}' ready (pg backend creates lazily on first add)."

    def delete_collection(self, name: str) -> str:
        with self.backend.conn_sync() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self.schema}.t2_vec_entries WHERE collection = %s",
                    (name,),
                )
                return f"Collection '{name}' deleted ({cur.rowcount} entries)."

    async def add(
        self,
        collection: str,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        record_id: int | None = None,
    ) -> list[str]:
        import json
        n = len(texts)
        warnings: list[str] = []
        if metadatas is None:
            warnings.append(f"metadatas was missing — auto-filled {n} placeholder dicts")
            metadatas = [{} for _ in range(n)]
        if len(metadatas) != n:
            warnings.append(
                f"vec_add: metadatas length ({len(metadatas)}) != texts length ({n}), auto-adjusting."
            )
            metadatas = (metadatas + [{} for _ in range(n)])[:n]

        coll_suffix = _collection_suffix(collection)
        _source_pat = re.compile(rf"^{SOURCE_SESSIONS_DIR}/[^\s:]+:\d+$")
        normalised: list[dict[str, Any]] = []
        n_empty = 0
        for i, m in enumerate(metadatas):
            if not isinstance(m, dict) or not m:
                n_empty += 1
                normalised.append({"type": "auto", "subject": coll_suffix or "unknown", "auto_filled": True})
                continue
            src = m.get("source")
            if isinstance(src, str) and src.startswith(f"{SOURCE_SESSIONS_DIR}/"):
                if not _source_pat.match(src):
                    raise ValueError(
                        f"vec_add: metadatas[{i}].source must match "
                        f"'{SOURCE_SESSIONS_DIR}/<sid>.jsonl:<line>' with numeric line number, got: {src!r}"
                    )
            if not any(k in m for k in ("source", "type", "subject", "topic")):
                m = dict(m)
                m["auto_filled"] = True
                m.setdefault("subject", coll_suffix or "unknown")
                m.setdefault("type", "auto")
            normalised.append(m)
        if n_empty:
            warnings.append(
                f"{n_empty}/{n} entries had empty/missing metadata — auto-filled"
            )
        self._last_add_warnings = warnings
        metadatas = normalised

        embeddings = await _compute_embeddings(self.embedder, texts)

        # 如果未指定 record_id，自动生成当前毫秒时间戳
        if record_id is None:
            import time
            record_id = int(time.time() * 1000)

        async with await self.backend.conn() as conn:
            async with conn.cursor() as cur:
                if self._id_counter == 0:
                    await cur.execute(
                        f"SELECT COUNT(*) AS c FROM {self.schema}.t2_vec_entries"
                    )
                    self._id_counter = (await cur.fetchone())["c"]

                added_ids: list[str] = []
                rows: list[tuple] = []
                for i, text in enumerate(texts):
                    self._id_counter += 1
                    eid = ids[i] if ids and i < len(ids) else f"vec_{self._id_counter}"
                    added_ids.append(eid)
                    emb = embeddings[i] if i < len(embeddings) else []
                    rows.append((
                        collection, eid, text,
                        emb if emb else None,
                        json.dumps(metadatas[i], ensure_ascii=False),
                        record_id,
                    ))
                await cur.executemany(
                    f"""
                    INSERT INTO {self.schema}.t2_vec_entries (collection, entry_id, text, embedding, metadata, record_id)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                    ON CONFLICT (collection, entry_id) DO UPDATE
                      SET text = EXCLUDED.text,
                          embedding = COALESCE(EXCLUDED.embedding, {self.schema}.t2_vec_entries.embedding),
                          metadata = EXCLUDED.metadata,
                          record_id = EXCLUDED.record_id
                    """,
                    rows,
                )

        if embeddings and embeddings[0] and self._known_dim is None:
            self._known_dim = len(embeddings[0])
            try:
                await self.backend.lazy_create_hnsw(self.schema, self._known_dim)
            except Exception as e:
                logger.debug("lazy_create_hnsw failed (ignored): %s", e)

        return added_ids

    async def search(
        self,
        collection: str,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return await self._search_impl(query, top_k, metadata_filter, collection=collection)

    async def search_all(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return await self._search_impl(query, top_k, metadata_filter, collection=None)

    async def search_all_with_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """使用预计算的 embedding 跨所有 collection 检索（PG 版）。

        Args:
            query_embedding: 预计算的 query 向量。
            top_k: 返回条数。
            metadata_filter: 元数据过滤条件。

        Returns:
            检索结果列表 [{id, text, score, metadata, collection}]。
        """
        import json as _json

        where_parts: list[str] = []
        params: list[Any] = []
        if metadata_filter:
            where_parts.append("metadata @> %s::jsonb")
            params.append(_json.dumps(metadata_filter, ensure_ascii=False))

        where_clause = (" AND ".join(where_parts)) if where_parts else "TRUE"

        if query_embedding:
            sql_q = f"""
                SELECT entry_id, text, metadata, collection,
                       1 - (embedding <=> %s::vector) AS score
                FROM {self.schema}.t2_vec_entries
                WHERE {where_clause} AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            params = [query_embedding, *params, query_embedding, top_k]
        else:
            sql_q = f"""
                SELECT entry_id, text, metadata, collection, 0.0 AS score
                FROM {self.schema}.t2_vec_entries
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT %s
            """
            params.append(top_k)

        async with await self.backend.conn() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql_q, params)
                rows = await cur.fetchall()

        results = []
        for r in rows:
            md = r["metadata"]
            if isinstance(md, str):
                try: md = _json.loads(md)
                except Exception: md = {}
            results.append({
                "id": r["entry_id"],
                "text": r["text"],
                "score": float(r["score"] or 0.0),
                "metadata": md or {},
                "collection": r["collection"],
            })
        return results

    async def _search_impl(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None,
        collection: str | None,
    ) -> list[dict[str, Any]]:
        import json
        q_emb = await _compute_single_embedding(self.embedder, query)

        where_parts: list[str] = []
        params: list[Any] = []
        if collection is not None:
            where_parts.append("collection = %s")
            params.append(collection)
        if metadata_filter:
            where_parts.append("metadata @> %s::jsonb")
            params.append(json.dumps(metadata_filter, ensure_ascii=False))

        where_clause = (" AND ".join(where_parts)) if where_parts else "TRUE"
        if q_emb:
            sql_q = f"""
                SELECT entry_id, text, metadata, collection,
                       1 - (embedding <=> %s::vector) AS score
                FROM {self.schema}.t2_vec_entries
                WHERE {where_clause} AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            params = [q_emb, *params, q_emb, top_k]
        else:
            sql_q = f"""
                SELECT entry_id, text, metadata, collection, 0.0 AS score
                FROM {self.schema}.t2_vec_entries
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT %s
            """
            params.append(top_k)

        async with await self.backend.conn() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql_q, params)
                rows = await cur.fetchall()

        results = []
        for r in rows:
            md = r["metadata"]
            if isinstance(md, str):
                try: md = json.loads(md)
                except Exception: md = {}
            results.append({
                "id": r["entry_id"],
                "text": r["text"],
                "score": float(r["score"] or 0.0),
                "metadata": md or {},
                "collection": r["collection"],
            })
        return results

    def delete(self, collection: str, ids: list[str]) -> str:
        with self.backend.conn_sync() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self.schema}.t2_vec_entries WHERE collection = %s AND entry_id = ANY(%s)",
                    (collection, list(ids)),
                )
                return f"Deleted {cur.rowcount} entries from '{collection}'."

    def update(
        self,
        collection: str,
        entry_id: str,
        new_text: str | None = None,
        new_metadata: dict[str, Any] | None = None,
    ) -> str:
        import json
        sets = []
        params: list[Any] = []
        if new_text is not None:
            sets.append("text = %s")
            params.append(new_text)
        if new_metadata is not None:
            sets.append("metadata = metadata || %s::jsonb")
            params.append(json.dumps(new_metadata, ensure_ascii=False))
        if not sets:
            return f"Entry '{entry_id}' unchanged (no fields to update)."
        params += [collection, entry_id]
        with self.backend.conn_sync() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {self.schema}.t2_vec_entries SET {', '.join(sets)} "
                    f"WHERE collection = %s AND entry_id = %s",
                    params,
                )
                return f"Updated entry '{entry_id}' ({cur.rowcount} row affected)."

    def get_stats(self) -> dict[str, Any]:
        with self.backend.conn_sync() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT collection, COUNT(*) AS c FROM {self.schema}.t2_vec_entries GROUP BY collection",
                )
                colls = {r["collection"]: r["c"] for r in cur.fetchall()}
        return {"total_entries": sum(colls.values()), "collections": colls}

    @property
    def _collections(self) -> dict[str, list[dict[str, Any]]]:
        """快照式快读（兼容内存版接口，供状态快照 / export 使用）。"""
        import json
        with self.backend.conn_sync() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT collection, entry_id AS id, text, metadata "
                    f"FROM {self.schema}.t2_vec_entries ORDER BY collection, id",
                )
                rows = cur.fetchall()
        out: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            md = r["metadata"]
            if isinstance(md, str):
                try: md = json.loads(md)
                except Exception: md = {}
            out.setdefault(r["collection"], []).append({
                "id": r["id"], "text": r["text"], "metadata": md or {}, "embedding": [],
            })
        return out

    # ------------------------------------------------------------------
    # 扩展能力（SqlCapableMixin / ScrollCapableMixin）
    # ------------------------------------------------------------------

    async def sql_query_read(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        limit: int = 200,
        timeout_s: float = 5.0,
    ) -> list[dict[str, Any]]:
        """只读 SQL 查询（强制 read-only transaction + statement_timeout）。

        自动将 search_path 设置为用户 schema，确保查询限定在用户数据范围内。
        """
        import json
        bad = re.search(
            r"\b(insert|update|delete|drop|truncate|alter|create|grant|revoke|copy)\b",
            query, re.IGNORECASE,
        )
        if bad:
            raise PermissionError(
                f"sql_query_read: write keyword '{bad.group(0)}' not allowed."
            )
        q = query.rstrip().rstrip(";")
        if not re.search(r"\blimit\s+\d+\b", q, re.IGNORECASE):
            q = f"{q} LIMIT {int(limit)}"
        async with await self.backend.conn() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"SET LOCAL statement_timeout = {int(timeout_s * 1000)}")
                await cur.execute("SET LOCAL transaction_read_only = on")
                await cur.execute(f"SET LOCAL search_path = {self.schema}, public")
                merged = params or {}
                await cur.execute(q, merged)
                if cur.description is None:
                    return []
                rows = await cur.fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k, v in list(d.items()):
                if isinstance(v, str) and (v.startswith("{") or v.startswith("[")):
                    try: d[k] = json.loads(v)
                    except Exception: pass
            out.append(d)
        return out

    async def scroll(
        self,
        collection: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        cursor: str | None = None,
        page_size: int = 100,
    ) -> dict[str, Any]:
        """按 filter 翻页流式遍历（避免 search top_k 上限）。"""
        import json
        where: list[str] = []
        params: list[Any] = []
        if collection is not None:
            where.append("collection = %s")
            params.append(collection)
        if metadata_filter:
            where.append("metadata @> %s::jsonb")
            params.append(json.dumps(metadata_filter, ensure_ascii=False))
        if cursor:
            where.append("entry_id > %s")
            params.append(cursor)
        params.append(page_size)
        where_clause = (" AND ".join(where)) if where else "TRUE"
        sql_q = (
            f"SELECT entry_id, text, metadata, collection "
            f"FROM {self.schema}.t2_vec_entries "
            f"WHERE {where_clause} "
            f"ORDER BY entry_id ASC LIMIT %s"
        )
        async with await self.backend.conn() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql_q, params)
                rows = await cur.fetchall()
        items = []
        for r in rows:
            md = r["metadata"]
            if isinstance(md, str):
                try: md = json.loads(md)
                except Exception: md = {}
            items.append({
                "id": r["entry_id"], "text": r["text"],
                "metadata": md or {}, "collection": r["collection"],
            })
        next_cursor = items[-1]["id"] if len(items) == page_size else None
        return {"items": items, "next_cursor": next_cursor}


# ---------------------------------------------------------------------------
# 私有工具函数（两个实现类共用）
# ---------------------------------------------------------------------------

def _collection_suffix(collection: str) -> str:
    """从 collection 名提取 subject 后缀（如 facts_alex → alex）。"""
    for prefix in ("facts_", "events_", "preferences_", "skills_", "insights_"):
        if collection.startswith(prefix):
            return collection[len(prefix):]
    return ""


async def _compute_embeddings(embedder: Any, texts: list[str]) -> list[list[float]]:
    """批量计算 embedding，失败时返回空列表兜底。"""
    if not embedder:
        logger.error("_compute_embeddings: embedder is None/empty! texts=%d items will have no embedding.", len(texts))
        return [[] for _ in texts]
    try:
        return await embedder.embed(texts)
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        return [[] for _ in texts]


async def _compute_single_embedding(embedder: Any, text: str) -> list[float]:
    """计算单条 embedding，失败时返回空列表。"""
    if not embedder:
        logger.error("_compute_single_embedding: embedder is None/empty! query will have no embedding.")
        return []
    try:
        return await embedder.embed_single(text)
    except Exception as e:
        logger.error("Query embedding failed: %s", e)
        return []


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """计算余弦相似度。"""
    import math
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a)) or 1e-10
    norm_b = math.sqrt(sum(b * b for b in vec_b)) or 1e-10
    return dot / (norm_a * norm_b)
