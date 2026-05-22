-- 用户 schema 内的表结构 DDL 模板
-- 由 PgBackend.ensure_user_schema(schema_name) 动态执行
-- 所有表都在用户独立的 schema 下，无需 ns 列进行隔离
--
-- 占位符 {schema} 会在运行时被替换为实际的 schema 名称

-- 创建用户 schema（幂等）
CREATE SCHEMA IF NOT EXISTS {schema};

-- vec entries：用户独享，无需 ns 列
CREATE TABLE IF NOT EXISTS {schema}.t2_vec_entries (
    id           BIGSERIAL PRIMARY KEY,
    collection   TEXT       NOT NULL,
    entry_id     TEXT       NOT NULL,            -- 对外暴露的 id（vec_<n>）
    text         TEXT       NOT NULL,
    embedding    vector,                         -- 维度按写入决定（pgvector 0.7 支持任意维度）
    metadata     JSONB      NOT NULL DEFAULT '{{}}'::jsonb,
    record_id    BIGINT,                         -- 操作记录ID（毫秒时间戳，标识本次操作批次）
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (collection, entry_id)
);
CREATE INDEX IF NOT EXISTS t2_vec_coll_idx
    ON {schema}.t2_vec_entries (collection);
CREATE INDEX IF NOT EXISTS t2_vec_meta_gin_idx
    ON {schema}.t2_vec_entries USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS t2_vec_text_trgm_idx
    ON {schema}.t2_vec_entries USING gin (text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS t2_vec_created_idx
    ON {schema}.t2_vec_entries (created_at DESC);
CREATE INDEX IF NOT EXISTS t2_vec_record_id_idx
    ON {schema}.t2_vec_entries (record_id);
-- 注意：HNSW 向量索引维度未知时不能预建，等 add() 看到首条 embedding 后再 lazy 建。

-- graph_name 注册表（记录本 schema 对应的 AGE graph 名）
CREATE TABLE IF NOT EXISTS {schema}.t2_graph_registry (
    graph_name  TEXT NOT NULL UNIQUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
