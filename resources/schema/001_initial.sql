-- PostgreSQL 数据库初始化DDL语句
-- 用于 llm_gateway 项目的表结构和索引创建
-- 隔离策略：每个用户拥有独立的 PostgreSQL schema，schema 内表结构相同
--
-- 本文件仅负责全局扩展安装。
-- 用户 schema 及其内部表由 PgBackend.ensure_user_schema() 动态创建。

-- 扩展（可能已被 init script 装好，这里幂等再保险）
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;
CREATE EXTENSION IF NOT EXISTS pg_trgm;