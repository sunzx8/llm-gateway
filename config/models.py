"""Pydantic models for structured configuration.

- ``GatewayConfig``: fields consumed by llm_gateway itself.
- ``AppConfig``: top-level wrapper; the ``litellm`` section is kept as a
  plain ``dict[str, Any]`` so it can be passed through to LiteLLM unchanged.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ContextTaskMode(str, Enum):
    """ContextTask 的运行模式。

    - T2: 使用 LLM agent loop 进行摄入/演进/检索
    - Code_T2: 完全使用代码生成的摄入和消费脚本
    - Multi_Code_T1: 多策略消费代码生成（不含摄入代码生成），摄入/检索复用 T2
    - Multi_Code_T2: 多策略代码生成 + 多子类选择/生成检索
    - Atomic_Code_T2: 在 Code_T2 基础上将摄入/检索拆解为更细粒度的原子步骤
    """

    T2 = "T2"
    Code_T2 = "Code_T2"
    Multi_Code_T1 = "Multi_Code_T1"
    Multi_Code_T2 = "Multi_Code_T2"
    Atomic_Code_T2 = "Atomic_Code_T2"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file_enabled: bool = True
    file_path: str = "./logs"
    console_enabled: bool = True
    console_colored: bool = False


class GatewayConfig(BaseModel):
    """Configuration for the llm_gateway section."""

    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Gateway's logging configuration",
    )
    llm_hook_handler: str = Field(default="", description="Path to LLM hook handler module")


# ============================================================
# Memory Config sub-models
# ============================================================

class LLMConfig(BaseModel):
    """Configuration for the Memory LLM."""

    provider: str = Field(default="openai", description="LLM provider name (openai / anthropic / openai_compat)")
    model: str = Field(default="", description="LLM model name")
    base_url: Optional[str] = Field(default=None, description="Base URL for the LLM API")
    api_key: Optional[str] = Field(default=None, description="API key for the LLM")
    temperature: float = Field(default=1.0, description="Sampling temperature")
    max_tokens: int = Field(default=4096, description="Maximum tokens for generation")
    timeout: float = Field(default=600.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")


class EmbeddingConfig(BaseModel):
    """Configuration for the Embedding model."""

    provider: str = Field(default="openai", description="Embedding provider (openai / openai_compat)")
    model: str = Field(default="text-embedding-3-small", description="Embedding model name")
    api_key: Optional[str] = Field(default=None, description="API key (null reads from env EMBEDDING_API_KEY or OPENAI_API_KEY)")
    base_url: Optional[str] = Field(default=None, description="Base URL (null uses OpenAI official endpoint)")
    batch_size: int = Field(default=64, description="Batch request size")
    dimensions: Optional[int] = Field(default=None, description="Vector dimensions (null uses model default)")
    timeout: float = Field(default=120.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")


class IngestContextTaskConfig(BaseModel):
    """Configuration for the ingest_context_task."""

    token_budget_per_ingest: int = Field(default=2000, description="Token budget per ingest")
    max_messages_per_ingest: int = Field(default=10, description="Hard limit on message count per ingest")
    max_turns: int = Field(default=15, description="Maximum turns for agent loop")


class ConsolidateContextTaskConfig(BaseModel):
    """Configuration for the consolidate_context_task."""

    min_items_for_evolution: int = Field(default=5, description="Skip evolution when memory items below this threshold")
    max_turns: int = Field(default=20, description="Maximum turns for agent loop")


class RetrieveContextTaskConfig(BaseModel):
    """Configuration for the retrieve_context_task."""

    max_turns: int = Field(default=20, description="Maximum turns for agent loop")


class ContextTaskConfig(BaseModel):
    """Configuration for all context tasks."""

    context_task_mode: ContextTaskMode = Field(
        default=ContextTaskMode.T2,
        description="ContextTask 运行模式: T2 / Code_T2 / Multi_Code_T1 / Multi_Code_T2",
    )

    ingest_context_task: IngestContextTaskConfig = Field(
        default_factory=IngestContextTaskConfig,
        description="Ingest context task configuration",
    )
    consolidate_context_task: ConsolidateContextTaskConfig = Field(
        default_factory=ConsolidateContextTaskConfig,
        description="Consolidate context task configuration",
    )
    retrieve_context_task: RetrieveContextTaskConfig = Field(
        default_factory=RetrieveContextTaskConfig,
        description="Retrieve context task configuration",
    )
    consolidate_trigger_threshold: int = Field(
        default=5,
        description="累积 ingest 次数达到此阈值后触发 consolidate",
    )
    consolidate_retrieve_trigger_threshold: int = Field(
        default=10,
        description="累积 retrieve 次数达到此阈值后异步触发 consolidate",
    )
    compress_session: bool = Field(
        default=False,
        description="是否压缩 session",
    )
    scheduled_ingest_enabled: bool = Field(
        default=False,
        description="是否启用定时记忆摄入任务",
    )
    scheduled_ingest_interval_seconds: float = Field(
        default=30.0,
        description="定时记忆摄入任务的执行间隔（秒）",
    )
    scheduled_ingest_max_workers: int = Field(
        default=10,
        description="定时记忆摄入任务的最大并发进程数",
    )


class PgConfig(BaseModel):
    """PostgreSQL connection configuration."""

    dsn: Optional[str] = Field(default=None, description="Full DSN (highest priority)")
    host: str = Field(default="127.0.0.1", description="PostgreSQL host")
    port: int = Field(default=5544, description="PostgreSQL port")
    user: str = Field(default="t2", description="PostgreSQL user")
    password: str = Field(default="t2", description="PostgreSQL password")
    database: str = Field(default="t2", description="PostgreSQL database name")
    pool_min_size: int = Field(default=2, description="Connection pool minimum size")
    pool_max_size: int = Field(default=16, description="Connection pool maximum size")

class MemoryFsConfig(BaseModel):
    """Memory filesystem configuration."""

    type: str = Field(default="localfs", description="Storage backend type: localfs")
    root_dir: str = Field(default="/tmp/llm_gateway", description="Root directory for localfs")
    enable_git: bool = Field(default=True, description="Enable git for localfs")

class StorageConfig(BaseModel):
    """Configuration for the memory storage backend."""

    backend: str = Field(default="memory", description="Storage backend type: memory / pg")
    memory_fs: MemoryFsConfig = Field(
        default_factory=MemoryFsConfig,
        description="Memory filesystem config (only effective when backend=memory)",
    )
    pg: PgConfig = Field(
        default_factory=PgConfig,
        description="PostgreSQL connection config (only effective when backend=pg)",
    )


class MemoryConfig(BaseModel):
    """Configuration for the memory_config section."""

    for_evaluation: bool = Field(
        default=False,
        description="是否开启评测模式",
    )
    memory_llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Memory LLM configuration",
    )
    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig,
        description="Embedding model configuration",
    )
    context_task: ContextTaskConfig = Field(
        default_factory=ContextTaskConfig,
        description="Context task configuration",
    )
    storage: StorageConfig = Field(
        default_factory=StorageConfig,
        description="Memory storage backend configuration",
    )


class AppConfig(BaseModel):
    """Top-level configuration mapping for config.yaml.

    Attributes:
        litellm: Raw dict passed through to LiteLLM without modification.
        llm_gateway: Structured gateway-specific configuration.
        memory_config: Memory-related configuration.
    """

    litellm: dict[str, Any] = Field(
        default_factory=dict,
        description="LiteLLM config, forwarded as-is",
    )
    llm_gateway: GatewayConfig = Field(
        default_factory=GatewayConfig,
        description="Gateway's own configuration",
    )
    memory_config: MemoryConfig = Field(
        default_factory=MemoryConfig,
        description="Memory configuration",
    )
