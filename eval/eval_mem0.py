"""Mem0 评测脚本（Baseline 对比）。

使用 mem0 开源记忆框架作为记忆后端，评测流程：
1. 对每个 conversation 的 session 消息调用 mem0.add() 摄入记忆
2. 对每个问题调用 mem0.search() 检索相关记忆
3. 将检索到的记忆作为上下文 + 问题一起发送给 LLM 回答

用于和当前 memory 服务、full context baseline 进行对比测试。

评测指标与 eval_script.py / eval_full_context.py 完全对齐：
- benchmark.evaluate() 计算的 metrics（MCQ accuracy 等）
- LLM-as-Judge 评分（yes/no、rubrics）
- 耗时统计（ingest / search / answer 各阶段耗时）

结果格式与 eval_script.py 兼容，可在同一 Dashboard 中对比查看。

依赖安装
~~~~~~~~
::

    pip install mem0ai

使用示例
~~~~~~~~
::

    python -m eval.eval_mem0 \\
        --benchmarks personamem \\
        --model gpt-5.4 \\
        --conv-concurrency 5 \\
        --subset-size 5
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import functools
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# 确保 eval/ 目录在搜索路径中
_EVAL_DIR = str(Path(__file__).resolve().parent)
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

# 确保项目根目录也在搜索路径中
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from benchmarks.base_benchmark import BaseBenchmark, Conversation, QAPair
from config.models import LLMConfig
from judge import LLMJudge
from utils.memory_llm_interface import LLMInterface

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mem0 GPT-5 兼容补丁
# ---------------------------------------------------------------------------
# mem0 的 LLMBase._is_reasoning_model 只识别 {gpt-5, gpt-5o, gpt-5o-mini,
# gpt-5o-micro, o1*, o3*}，不识别 gpt-5.x 系列（如 gpt-5.4 / gpt-5.4-mini）。
# 当 base_url 指向需要 max_completion_tokens 的 Azure endpoint 时，
# mem0 会传 max_tokens 给 chat.completions，触发 400 错误。
# 修复策略：monkey-patch _is_reasoning_model，让 gpt-5.x / o-series 全部走
# reasoning 分支，mem0 会自动剔除 max_tokens / temperature / top_p。
def _patch_mem0_reasoning_detection() -> None:
    try:
        from mem0.llms.base import LLMBase
    except ImportError:
        return  # mem0 未安装，留给后续 import 报错

    if getattr(LLMBase, "_eval_reasoning_patch_applied", False):
        return

    _orig = LLMBase._is_reasoning_model

    def _patched_is_reasoning_model(self, model: str) -> bool:
        """识别完整的 GPT-5 家族 (gpt-5, gpt-5.x, gpt-5o*) 和 o-series。"""
        if not model:
            return False
        base = model.lower().rsplit("/", 1)[-1]
        # gpt-5, gpt-5o, gpt-5.4, gpt-5.4-mini, gpt-5-turbo, ...
        if base.startswith("gpt-5"):
            return True
        # o1 / o3 / o4 / future o-series
        if base in {"o1", "o3", "o4"}:
            return True
        if any(base.startswith(p) for p in ("o1-", "o1.", "o3-", "o3.", "o4-", "o4.")):
            return True
        return _orig(self, model)

    LLMBase._is_reasoning_model = _patched_is_reasoning_model
    LLMBase._eval_reasoning_patch_applied = True
    logger.info("Mem0 LLMBase._is_reasoning_model patched: GPT-5.x family now treated as reasoning model")


_patch_mem0_reasoning_detection()


# ---------------------------------------------------------------------------
# 模块级 asyncio.Lock：保护 mem0 初始化过程
# ---------------------------------------------------------------------------
_mem0_init_lock: asyncio.Lock | None = None


def _get_mem0_init_lock() -> asyncio.Lock:
    """延迟创建 asyncio.Lock（需要在事件循环中创建）。"""
    global _mem0_init_lock
    if _mem0_init_lock is None:
        _mem0_init_lock = asyncio.Lock()
    return _mem0_init_lock


# ---------------------------------------------------------------------------
# Embedding 维度探测
# ---------------------------------------------------------------------------
# 缓存已探测的维度，避免重复探测
_probed_embedding_dims: int | None = None


def _probe_embedding_dims(
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> int | None:
    """发送一个测试请求探测 embedding 模型的实际输出维度。

    当 embedding 维度未知时调用。通过向 embedding API 发送一个短文本
    来获取实际向量维度，避免 Qdrant 使用默认 1536 维导致维度不匹配。
    """
    global _probed_embedding_dims
    if _probed_embedding_dims is not None:
        return _probed_embedding_dims
    try:
        import openai
        client = openai.OpenAI(
            api_key=api_key or "none",
            base_url=base_url,
            timeout=30.0,
        )
        resp = client.embeddings.create(
            model=model or "text-embedding-3-small",
            input="dimension probe",
        )
        dims = len(resp.data[0].embedding)
        _probed_embedding_dims = dims
        logger.info("Probed embedding dims: model=%s, dims=%d", model, dims)
        return dims
    except Exception as e:
        logger.warning(
            "Failed to probe embedding dims (model=%s): %s. "
            "Falling back to default 1536.",
            model, e,
        )
        return None


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class Mem0RunConfig:
    """Mem0 评测运行配置。"""

    benchmarks: list[str] = field(default_factory=lambda: ["personamem"])
    """要评测的 benchmark 名称列表。"""

    subset_size: int | None = None
    """每个 benchmark 取前 N 条 conversation（None = 全量）。"""

    conv_concurrency: int = 5
    """同一 benchmark 内并行处理的 conversation 数量。
    注意：mem0 的并发能力受限于底层 LLM API 和向量数据库，建议不要设太高。"""

    # --- LLM 配置（用于回答问题） ---
    model: str = "ep-2bjbej6a"
    """用于回答问题的 LLM 模型名称（tokenhub 上 ep-2bjbej6a 背后即 gpt-5.4）。"""

    provider: str = "openai"
    """LLM provider（openai / openai_compat / anthropic）。"""

    base_url: str | None = None
    """LLM API 的 base URL（None 使用 provider 默认值）。"""

    api_key: str | None = None
    """LLM API Key（None 从环境变量读取）。"""

    temperature: float = 0.0
    """LLM 采样温度（baseline 评测建议用 0）。"""

    max_tokens: int = 4096
    """LLM 最大生成 token 数。"""

    # --- Mem0 配置 ---
    mem0_provider: str = "openai_compat"
    """Mem0 使用的 LLM provider。"""

    mem0_model: str = "ep-2bjbej6a"
    """Mem0 使用的 LLM 模型（用于记忆提取和摘要，tokenhub 上 ep-2bjbej6a 背后即 gpt-5.4）。"""

    mem0_base_url: str | None = None
    """Mem0 LLM API 的 base URL。"""

    mem0_api_key: str | None = None
    """Mem0 LLM API Key（None 从环境变量 OPENAI_API_KEY 读取）。"""

    mem0_embedding_model: str = "text-embedding-3-small"
    """Mem0 使用的 embedding 模型。"""

    mem0_embedding_base_url: str | None = None
    """Mem0 embedding API 的 base URL。"""

    mem0_embedding_api_key: str | None = None
    """Mem0 embedding API Key。"""

    mem0_search_limit: int = 20
    """mem0.search() 返回的最大记忆条数。"""

    eval_run_id: str = field(default_factory=lambda: f"mem0_{datetime.now().strftime('%Y%m%d%H%M%S')}")
    """本次评测的时间标识，前缀 mem0_ 表示 mem0 baseline。"""

    # --- Judge 配置 ---
    judge_model: str = "gpt-5.5"
    judge_base_url: str = "https://vdbteam.cognitiveservices.azure.com/openai/v1/"
    judge_api_key: str = "d0924051350c460295107e33772e1e49"
    no_judge: bool = False


# ---------------------------------------------------------------------------
# 结果数据结构（与 eval_script.py / eval_full_context.py 对齐）
# ---------------------------------------------------------------------------


@dataclass
class ConversationResult:
    """单个 conversation 的评测结果。"""

    conv_id: str
    predictions: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    qa_pairs: list[QAPair] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: str = ""
    # Mem0 操作耗时统计
    ingest_elapsed_list: list[float] = field(default_factory=list)
    """每次 mem0.add() 的耗时（秒）"""
    search_elapsed_list: list[float] = field(default_factory=list)
    """每次 mem0.search() 的耗时（秒）"""
    llm_elapsed_list: list[float] = field(default_factory=list)
    """每次纯 LLM 生成回答的耗时（秒）"""
    answer_elapsed_list: list[float] = field(default_factory=list)
    """每次回答问题的完整耗时（search + llm_generate）（秒）"""
    # TTFT（首 token 时延）统计
    ttft_list: list[float] = field(default_factory=list)
    """每次 LLM 调用的首 token 时延（秒），与 predictions 一一对应"""
    # LLM 使用统计
    input_tokens_list: list[int] = field(default_factory=list)
    """每次 LLM 调用的输入 token 数"""
    output_tokens_list: list[int] = field(default_factory=list)
    """每次 LLM 调用的输出 token 数"""
    # 记忆统计
    memories_retrieved_list: list[int] = field(default_factory=list)
    """每次 search 返回的记忆条数"""
    memories_added_list: list[int] = field(default_factory=list)
    """每次 add 新增的记忆条数"""
    # 错误统计
    ingest_errors: list[dict[str, Any]] = field(default_factory=list)
    answer_errors: list[dict[str, Any]] = field(default_factory=list)
    # 召回记忆内容
    retrieved_contexts: list[str] = field(default_factory=list)
    """每个 QA 对应的召回记忆文本，与 predictions 一一对应"""
    # Conversation 级别累计 token
    conv_total_tokens: int = 0
    """该 conversation 内所有 QA 的累计 token（input + output）"""
    # QA 正确率统计
    qa_pass_flags: list[bool] = field(default_factory=list)
    mcq_accuracy: float = 0.0


@dataclass
class Mem0BenchmarkResult:
    """单个 benchmark 的评测结果。"""

    benchmark_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    judge_scores: list[float] = field(default_factory=list)
    predictions: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    qa_pairs: list[QAPair] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    per_conversation_results: list[ConversationResult] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Mem0Runner
# ---------------------------------------------------------------------------


class Mem0Runner:
    """Mem0 评测 Runner。

    对每个 conversation：
    1. 创建独立的 mem0 用户（通过 user_id 隔离）
    2. 按 session 顺序调用 mem0.add() 摄入记忆
    3. 对每个 QAPair 调用 mem0.search() 检索相关记忆
    4. 将检索到的记忆 + 问题发送给 LLM 回答
    5. 收集 predictions / references，计算 metrics

    与 FullContextRunner 的区别：
    - 使用 mem0 进行记忆管理（add / search）
    - 不传入完整上下文，而是依赖 mem0 提取和检索的记忆
    - 用于评估 "mem0 记忆框架的记忆能力" 作为 baseline
    """

    # Embedding 模型的 token 上限（text-embedding-3-small = 8191）
    # mem0 内部会把消息拼接成字符串并添加格式化开销（role 前缀、换行符等），
    # 实际 token 数比原始消息多 15-30%，因此需要留足余量。按 5500 切分确保安全。
    EMBEDDING_TOKEN_LIMIT = 5500

    def __init__(self, config: Mem0RunConfig):
        self.config = config
        self._benchmarks: dict[str, BaseBenchmark] = {}
        self._results: list[Mem0BenchmarkResult] = []
        self._llm: LLMInterface | None = None
        # 每个 Runner 实例分配唯一 ID，确保所有路径完全隔离
        self._instance_id: str = uuid.uuid4().hex[:8]

    def register_benchmark(self, benchmark: BaseBenchmark) -> None:
        """注册一个 benchmark adapter。"""
        self._benchmarks[benchmark.name] = benchmark

    def _get_llm(self) -> LLMInterface:
        """获取或创建 LLM 接口实例（用于回答问题）。"""
        if self._llm is None:
            config = {
                "provider": self.config.provider,
                "model": self.config.model,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "timeout": 600.0,
                "max_retries": 3,
            }
            if self.config.base_url:
                config["base_url"] = self.config.base_url
            if self.config.api_key:
                config["api_key"] = self.config.api_key
            self._llm = LLMInterface(config)
        return self._llm

    def _build_mem0_config(self, conv_id: str = "") -> dict[str, Any]:
        """构建 mem0 的完整配置字典（LLM + Embedder + Vector Store）。

        每个 conversation 使用独立的 qdrant 存储路径和 history db 路径，
        避免并发时文件锁冲突。
        """
        # mem0 不支持 openai_compat，需要映射为 openai（mem0 的 openai 本身支持自定义 base_url）
        _PROVIDER_MAP = {"openai_compat": "openai"}
        llm_provider = _PROVIDER_MAP.get(self.config.mem0_provider, self.config.mem0_provider)

        # ── LLM 配置 ──
        llm_config: dict[str, Any] = {
            "model": self.config.mem0_model,
            "temperature": 0.0,
            "max_tokens": 4096,
        }
        if self.config.mem0_base_url:
            llm_config["openai_base_url"] = self.config.mem0_base_url
        # api_key 回退链：mem0_api_key → 通用 api_key → 环境变量 OPENAI_API_KEY
        llm_api_key = (
            self.config.mem0_api_key
            or self.config.api_key
            or os.environ.get("OPENAI_API_KEY")
        )
        if llm_api_key:
            llm_config["api_key"] = llm_api_key

        # ── Embedder 配置 ──
        embedder_config: dict[str, Any] = {
            "model": self.config.mem0_embedding_model,
        }
        if self.config.mem0_embedding_base_url:
            embedder_config["openai_base_url"] = self.config.mem0_embedding_base_url
        # embedding api_key 回退链
        embedding_api_key = (
            self.config.mem0_embedding_api_key
            or self.config.mem0_api_key
            or self.config.api_key
            or os.environ.get("OPENAI_API_KEY")
        )
        if embedding_api_key:
            embedder_config["api_key"] = embedding_api_key

        # ── Vector Store 配置（每个实例+conv 独立路径） ──
        qdrant_path = os.path.join(
            tempfile.gettempdir(),
            "mem0_eval_qdrant",
            f"{self._instance_id}_{conv_id or 'default'}",
        )
        os.makedirs(qdrant_path, exist_ok=True)

        # 自动探测 embedding 维度，避免硬编码 1536
        embed_dims = _probe_embedding_dims(
            model=self.config.mem0_embedding_model,
            api_key=embedding_api_key,
            base_url=self.config.mem0_embedding_base_url,
        )
        vector_store_cfg: dict[str, Any] = {
            "collection_name": "mem0_eval",
            "path": qdrant_path,
        }
        if embed_dims:
            vector_store_cfg["embedding_model_dims"] = embed_dims
            logger.info("Mem0 Qdrant embedding_model_dims set to %d", embed_dims)
        else:
            vector_store_cfg["embedding_model_dims"] = 1536  # 默认回退

        # ── History DB 路径（每个实例+conv 独立） ──
        db_path = os.path.join(
            tempfile.gettempdir(),
            f"mem0_eval_{self._instance_id}_{conv_id or 'default'}",
            "history.db",
        )
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        mem0_config: dict[str, Any] = {
            "llm": {
                "provider": llm_provider,
                "config": llm_config,
            },
            "embedder": {
                "provider": "openai",
                "config": embedder_config,
            },
            "vector_store": {
                "provider": "qdrant",
                "config": vector_store_cfg,
            },
            "history_db_path": db_path,
            "version": "v1.1",
        }

        return mem0_config

    async def _create_mem0_client(self, conv_id: str = ""):
        """创建 mem0 客户端实例（使用 asyncio.Lock 串行化）。

        mem0 内部会创建 _telemetry_vector_store，使用 MEM0_DIR/migrations_qdrant
        路径。通过以下策略彻底避免并发冲突：
        1. asyncio.Lock 串行化初始化过程
        2. 每个实例+conv 使用独立的 MEM0_DIR
        3. 重新加载 mem0 模块缓存的 mem0_dir 值
        """
        try:
            from mem0 import Memory
        except ImportError:
            raise ImportError(
                "mem0 未安装，请运行: pip install mem0ai\n"
                "详见: https://github.com/mem0ai/mem0"
            )

        mem0_config = self._build_mem0_config(conv_id)

        # 用锁保护 mem0 初始化：mem0 内部会创建一个 telemetry vector store
        # 指向 MEM0_DIR/migrations_qdrant，并发初始化会导致 Qdrant 文件锁冲突。
        # 通过串行化初始化 + 临时设置 MEM0_DIR 为每个实例独立路径来解决。
        async with _get_mem0_init_lock():
            # 每个实例+conv 使用独立的 MEM0_DIR，彻底避免 telemetry migrations 目录冲突
            mem0_dir = os.path.join(
                tempfile.gettempdir(),
                "mem0_eval_home",
                f"{self._instance_id}_{conv_id or 'default'}",
            )
            os.makedirs(mem0_dir, exist_ok=True)
            old_mem0_dir = os.environ.get("MEM0_DIR")
            os.environ["MEM0_DIR"] = mem0_dir
            try:
                # 重新加载 mem0 模块中缓存的 mem0_dir 值
                import mem0.memory.main as _mem0_main
                _mem0_main.mem0_dir = mem0_dir

                loop = asyncio.get_event_loop()
                mem0_client = await loop.run_in_executor(
                    None,
                    functools.partial(Memory.from_config, mem0_config),
                )
            finally:
                # 恢复原始 MEM0_DIR
                if old_mem0_dir is None:
                    os.environ.pop("MEM0_DIR", None)
                else:
                    os.environ["MEM0_DIR"] = old_mem0_dir

        logger.info(
            "Mem0 initialized: conv_id=%s, qdrant_path=%s, mem0_dir=%s",
            conv_id,
            mem0_config["vector_store"]["config"]["path"],
            mem0_dir,
        )
        return mem0_client

    @staticmethod
    def _close_mem0_client(mem0_client) -> None:
        """关闭 mem0 实例，释放所有 Qdrant 文件锁。

        mem0 内部会创建多个 Qdrant client：
        1. vector_store.client — 业务数据存储
        2. _telemetry_vector_store.client — telemetry migrations 存储
        必须全部关闭，否则重新初始化时会因文件锁冲突而失败。
        """
        if mem0_client is None:
            return
        # 关闭 telemetry vector store 的 Qdrant client
        try:
            tvs = getattr(mem0_client, '_telemetry_vector_store', None)
            if tvs is not None:
                client = getattr(tvs, 'client', None)
                if client is not None and hasattr(client, 'close'):
                    client.close()
                    logger.debug("Closed mem0 telemetry qdrant client")
        except Exception as e:
            logger.debug("Error closing mem0 telemetry qdrant client: %s", e)
        # 关闭业务数据的 Qdrant client
        try:
            vs = getattr(mem0_client, 'vector_store', None)
            if vs is not None:
                client = getattr(vs, 'client', None)
                if client is not None and hasattr(client, 'close'):
                    client.close()
                    logger.debug("Closed mem0 vector_store qdrant client")
        except Exception as e:
            logger.debug("Error closing mem0 vector_store qdrant client: %s", e)

    async def run(self) -> list[Mem0BenchmarkResult]:
        """运行所有已注册 benchmark 的评测。"""
        self._results = []
        run_start = time.monotonic()

        logger.info(
            "=== Mem0Runner: %d benchmarks, conv_concurrency=%d ===",
            len(self.config.benchmarks),
            self.config.conv_concurrency,
        )

        bench_names = [
            name for name in self.config.benchmarks
            if name in self._benchmarks
        ]
        skipped = [n for n in self.config.benchmarks if n not in self._benchmarks]
        for name in skipped:
            logger.warning("Benchmark %s not registered, skipping", name)

        for name in bench_names:
            result = await self._evaluate_benchmark(self._benchmarks[name])
            self._results.append(result)

        total_elapsed = time.monotonic() - run_start
        logger.info(
            "=== Mem0Runner done: %d results, total %.1fs ===",
            len(self._results), total_elapsed,
        )

        # 关闭 LLM 客户端
        if self._llm:
            await self._llm.aclose()

        return self._results

    # ------------------------------------------------------------------
    # 单个 benchmark 评测
    # ------------------------------------------------------------------

    async def _evaluate_benchmark(
        self,
        benchmark: BaseBenchmark,
    ) -> Mem0BenchmarkResult:
        """评测单个 benchmark。"""
        start_time = time.monotonic()
        bench_name = benchmark.name

        logger.info("=" * 60)
        logger.info("Mem0Runner: evaluating benchmark=%s", bench_name)

        # 加载数据
        conversations = benchmark.load_data()
        if self.config.subset_size:
            conversations = benchmark.subset(conversations, self.config.subset_size)
        logger.info(
            "Loaded %d conversations (subset=%s)",
            len(conversations), self.config.subset_size or "full",
        )

        # 并发处理 conversations
        conv_results = await self._run_conversations(benchmark, conversations)

        # 汇总 predictions / references / qa_pairs
        all_predictions: list[str] = []
        all_references: list[str] = []
        all_qa_pairs: list[QAPair] = []
        for cr in conv_results:
            all_predictions.extend(cr.predictions)
            all_references.extend(cr.references)
            all_qa_pairs.extend(cr.qa_pairs)

        # 计算 metrics
        metrics = benchmark.evaluate(
            all_predictions, all_references, qa_pairs=all_qa_pairs,
        )
        logger.info(
            "Mem0Runner: benchmark=%s metrics=%s",
            bench_name,
            {k: f"{v:.3f}" for k, v in metrics.items()},
        )

        # 为每个 conversation 计算 MCQ 准确率
        self._compute_mcq_accuracy(benchmark, conv_results)

        # LLM-as-Judge
        judge_scores: list[float] = []
        if bench_name in ("longmemeval", "locomo") and not self.config.no_judge:
            judge_scores = await self._run_judge(
                bench_name, all_predictions, all_references, all_qa_pairs, metrics,
            )

        elapsed = time.monotonic() - start_time

        # 汇总统计
        stats = self._aggregate_stats(conv_results)

        return Mem0BenchmarkResult(
            benchmark_name=bench_name,
            metrics=metrics,
            judge_scores=judge_scores,
            predictions=all_predictions,
            references=all_references,
            qa_pairs=all_qa_pairs,
            elapsed_seconds=elapsed,
            per_conversation_results=conv_results,
            stats={
                "num_conversations": len(conversations),
                "num_predictions": len(all_predictions),
                "elapsed_seconds": round(elapsed, 1),
                "eval_mode": "mem0",
                **stats,
            },
        )

    # ------------------------------------------------------------------
    # MCQ 准确率计算
    # ------------------------------------------------------------------

    def _compute_mcq_accuracy(
        self,
        benchmark: BaseBenchmark,
        conv_results: list[ConversationResult],
    ) -> None:
        """为每个 ConversationResult 计算 MCQ 准确率。"""
        for cr in conv_results:
            if not cr.predictions:
                cr.qa_pass_flags = []
                cr.mcq_accuracy = 0.0
                continue

            flags = benchmark.check_correctness(
                cr.predictions, cr.references, cr.qa_pairs,
            )
            cr.qa_pass_flags = flags
            passed = sum(1 for f in flags if f)
            cr.mcq_accuracy = passed / len(flags) if flags else 0.0

    # ------------------------------------------------------------------
    # Conversation 并发处理
    # ------------------------------------------------------------------

    async def _run_conversations(
        self,
        benchmark: BaseBenchmark,
        conversations: list[Conversation],
    ) -> list[ConversationResult]:
        """并发处理多个 conversations（同一 user_id 内串行，不同 user_id 间并发）。

        conv_concurrency 控制的是同时活跃的 user 数量，而非 conversation 数量。
        同一 user_id 下的多个 conversation 严格按顺序串行执行，避免记忆库并发写入冲突。
        """
        from collections import OrderedDict

        # Step 1: 按 user_id 分组，保持组内顺序
        user_groups: OrderedDict[str, list[Conversation]] = OrderedDict()
        for conv in conversations:
            uid = conv.metadata.get("user_id", conv.conv_id)
            user_groups.setdefault(uid, []).append(conv)

        sem = asyncio.Semaphore(self.config.conv_concurrency)
        total = len(conversations)
        completed_count = 0
        lock = asyncio.Lock()
        # 保存 (conv_id -> result) 的映射，最终按原始顺序返回
        result_map: dict[str, ConversationResult] = {}

        async def _run_user_group(uid: str, convs: list[Conversation]):
            """同一 user_id 下的 conversations 串行执行，整个 group 占用一个并发槽位。"""
            nonlocal completed_count
            async with sem:
                for conv in convs:
                    result = await self._evaluate_conversation(benchmark, conv)
                    result_map[conv.conv_id] = result
                    async with lock:
                        completed_count += 1
                        logger.info(
                            "  进度: %d/%d (%.1f%%) 已完成, 当前完成: conv=%s (user=%s)",
                            completed_count, total, completed_count / total * 100,
                            conv.conv_id, uid,
                        )

        # Step 2: 不同 user_id 的 group 并发执行
        logger.info(
            "  并发调度: %d 个 user group, conv_concurrency=%d",
            len(user_groups), self.config.conv_concurrency,
        )
        await asyncio.gather(
            *[_run_user_group(uid, convs) for uid, convs in user_groups.items()],
            return_exceptions=False,
        )

        # Step 3: 按原始 conversations 顺序返回结果
        return [result_map[conv.conv_id] for conv in conversations]

    # ------------------------------------------------------------------
    # 单个 conversation 评测
    # ------------------------------------------------------------------

    async def _evaluate_conversation(
        self,
        benchmark: BaseBenchmark,
        conv: Conversation,
    ) -> ConversationResult:
        """评测单个 conversation。

        流程：
        1. 为该 conversation 创建独立的 mem0 实例（内存隔离）
        2. 按 session 顺序调用 mem0.add() 摄入记忆
        3. 对每个 QAPair 调用 mem0.search() + LLM 回答
        4. 返回 ConversationResult
        """
        start_time = time.monotonic()
        cr = ConversationResult(conv_id=conv.conv_id)
        qdrant_path = os.path.join(
            tempfile.gettempdir(),
            f"mem0_eval_qdrant_{self.config.eval_run_id}_{conv.conv_id}",
        )

        mem0_client = None
        try:
            # 每个 conversation 创建独立的 mem0 实例，确保记忆隔离
            # 使用 asyncio.Lock 串行化创建过程 + MEM0_DIR 隔离，
            # 彻底避免 mem0 内部 telemetry qdrant 路径并发冲突
            mem0_client = await self._create_mem0_client(conv.conv_id)
            user_id = conv.metadata.get("user_id", conv.conv_id)
            # 拼接评测时间标识后缀，区分不同评测轮次
            user_id = f"{user_id}_{self.config.eval_run_id}"

            # 检测是否为 interleaved 模式
            interleaved = any(
                hasattr(s, "questions") and s.questions
                for s in conv.sessions
            )

            if interleaved:
                # Interleaved 模式：ingest session → 立即回答该 session 的问题
                for sess_idx, session in enumerate(conv.sessions):
                    # Phase 1: 摄入该 session 的消息到 mem0
                    messages = [
                        {"role": m.role, "content": m.content}
                        for m in session.messages
                    ]
                    logger.debug(
                        "  [conv=%s] mem0.add session %d/%d (%d msgs, interleaved)",
                        conv.conv_id, sess_idx + 1, len(conv.sessions), len(messages),
                    )
                    ingest_elapsed, memories_added, ingest_error = await self._mem0_add(
                        mem0_client, user_id, messages,
                    )
                    cr.ingest_elapsed_list.append(ingest_elapsed)
                    cr.memories_added_list.append(memories_added)
                    if ingest_error:
                        cr.ingest_errors.append(ingest_error)

                    # Phase 2: 回答该 session 的附属问题
                    sess_questions: list[QAPair] = getattr(session, "questions", []) or []
                    for qa in sess_questions:
                        prediction, answer_elapsed, search_elapsed, llm_elapsed, \
                            memories_found, input_tokens, output_tokens, \
                            ttft, retrieved_context, answer_error = (
                                await self._search_and_answer(
                                    mem0_client, user_id, qa.question,
                                )
                            )
                        if answer_error:
                            cr.answer_errors.append(answer_error)
                        cr.predictions.append(prediction)
                        cr.references.append(qa.reference_answer)
                        cr.qa_pairs.append(qa)
                        cr.answer_elapsed_list.append(answer_elapsed)
                        cr.search_elapsed_list.append(search_elapsed)
                        cr.llm_elapsed_list.append(llm_elapsed)
                        cr.memories_retrieved_list.append(memories_found)
                        cr.input_tokens_list.append(input_tokens)
                        cr.output_tokens_list.append(output_tokens)
                        cr.ttft_list.append(ttft)
                        cr.retrieved_contexts.append(retrieved_context)
            else:
                # 标准模式：先 ingest 所有 sessions，再统一回答问题
                for sess_idx, session in enumerate(conv.sessions):
                    messages = [
                        {"role": m.role, "content": m.content}
                        for m in session.messages
                    ]
                    logger.debug(
                        "  [conv=%s] mem0.add session %d/%d (%d msgs)",
                        conv.conv_id, sess_idx + 1, len(conv.sessions), len(messages),
                    )
                    ingest_elapsed, memories_added, ingest_error = await self._mem0_add(
                        mem0_client, user_id, messages,
                    )
                    cr.ingest_elapsed_list.append(ingest_elapsed)
                    cr.memories_added_list.append(memories_added)
                    if ingest_error:
                        cr.ingest_errors.append(ingest_error)

                # 回答所有问题
                questions = benchmark.get_questions(conv)
                for qa in questions:
                    prediction, answer_elapsed, search_elapsed, llm_elapsed, \
                        memories_found, input_tokens, output_tokens, \
                        ttft, retrieved_context, answer_error = (
                            await self._search_and_answer(
                                mem0_client, user_id, qa.question,
                            )
                        )
                    if answer_error:
                        cr.answer_errors.append(answer_error)
                    cr.predictions.append(prediction)
                    cr.references.append(qa.reference_answer)
                    cr.qa_pairs.append(qa)
                    cr.answer_elapsed_list.append(answer_elapsed)
                    cr.search_elapsed_list.append(search_elapsed)
                    cr.llm_elapsed_list.append(llm_elapsed)
                    cr.memories_retrieved_list.append(memories_found)
                    cr.input_tokens_list.append(input_tokens)
                    cr.output_tokens_list.append(output_tokens)
                    cr.ttft_list.append(ttft)
                    cr.retrieved_contexts.append(retrieved_context)

        except Exception as e:
            logger.exception(
                "Mem0Runner: conversation=%s failed: %s",
                conv.conv_id, e,
            )
            cr.error = str(e)
            # 对失败的 conversation，用空字符串填充 predictions 和 retrieved_contexts
            questions = benchmark.get_questions(conv)
            missing = len(questions) - len(cr.predictions)
            if missing > 0:
                cr.predictions.extend([""] * missing)
                cr.references.extend(
                    [qa.reference_answer for qa in questions[len(cr.predictions) - missing:]]
                )
                cr.qa_pairs.extend(questions[len(cr.qa_pairs):])
            ctx_missing = len(cr.predictions) - len(cr.retrieved_contexts)
            if ctx_missing > 0:
                cr.retrieved_contexts.extend([""] * ctx_missing)

        finally:
            # 计算 conversation 级别累计 token
            cr.conv_total_tokens = sum(cr.input_tokens_list) + sum(cr.output_tokens_list)
            cr.elapsed_seconds = time.monotonic() - start_time
            logger.info(
                "  [conv=%s] done: %d predictions, %.1fs, conv_total_tokens=%d, error=%s",
                conv.conv_id, len(cr.predictions), cr.elapsed_seconds,
                cr.conv_total_tokens, cr.error or "none",
            )
            # 关闭 mem0 实例，释放 Qdrant 文件锁
            self._close_mem0_client(mem0_client)
            # 清理该 conversation 的所有临时存储目录
            for cleanup_dir in [
                qdrant_path,
                os.path.join(
                    tempfile.gettempdir(),
                    "mem0_eval_qdrant",
                    f"{self._instance_id}_{conv.conv_id}",
                ),
                os.path.join(
                    tempfile.gettempdir(),
                    f"mem0_eval_{self._instance_id}_{conv.conv_id}",
                ),
                os.path.join(
                    tempfile.gettempdir(),
                    "mem0_eval_home",
                    f"{self._instance_id}_{conv.conv_id}",
                ),
            ]:
                try:
                    if cleanup_dir and os.path.isdir(cleanup_dir):
                        shutil.rmtree(cleanup_dir, ignore_errors=True)
                except Exception:
                    pass

        return cr

    # ------------------------------------------------------------------
    # Mem0 操作：add（摄入记忆）
    # ------------------------------------------------------------------

    # 每批 add 的最大字符数（保守估计：1 token ≈ 4 字符，8192 tokens ≈ 32K 字符，
    # 留余量设为 20K 字符，避免 embedding 模型 token 超限）
    _MEM0_ADD_MAX_CHARS_PER_BATCH = 20_000

    @staticmethod
    def _split_messages_into_batches(
        messages: list[dict[str, str]],
        max_chars: int,
    ) -> list[list[dict[str, str]]]:
        """将消息列表按字符数上限分批，确保每批不超过 max_chars。

        分割策略：
        - 按消息逐条累加字符数，超过阈值时切分为新批次
        - 尽量在 assistant 消息之后切分（对话轮次边界）
        - 单条消息超过阈值时独立成一批
        """
        if not messages:
            return []

        batches: list[list[dict[str, str]]] = []
        current_batch: list[dict[str, str]] = []
        current_chars = 0

        for msg in messages:
            msg_chars = len(msg.get("content", ""))

            # 如果当前批次加上这条消息会超限，且当前批次非空，则先切分
            if current_batch and current_chars + msg_chars > max_chars:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0

            current_batch.append(msg)
            current_chars += msg_chars

        # 最后一批
        if current_batch:
            batches.append(current_batch)

        return batches

    async def _mem0_add(
        self,
        mem0_client,
        user_id: str,
        messages: list[dict[str, str]],
    ) -> tuple[float, int, dict[str, Any] | None]:
        """调用 mem0.add() 摄入消息到记忆。

        当消息总字符数超过 embedding 模型限制时，自动分批调用 mem0.add()，
        避免 "maximum context length" 错误。

        Args:
            mem0_client: mem0 Memory 实例。
            user_id: 用户 ID。
            messages: 消息列表。

        Returns:
            元组 (耗时秒数, 新增记忆条数, 错误信息或None)。
        """
        t0 = time.monotonic()
        total_memories_added = 0
        last_error_info = None

        # 按字符数分批，避免 embedding 模型 token 超限
        batches = self._split_messages_into_batches(
            messages, self._MEM0_ADD_MAX_CHARS_PER_BATCH,
        )
        logger.debug(
            "  [mem0.add] user_id=%s, total_messages=%d, batches=%d",
            user_id, len(messages), len(batches),
        )

        loop = asyncio.get_event_loop()
        for batch_idx, batch in enumerate(batches):
            try:
                result = await loop.run_in_executor(
                    None,
                    functools.partial(
                        mem0_client.add,
                        batch,
                        user_id=user_id,
                    ),
                )

                # 统计新增记忆条数
                batch_added = 0
                if isinstance(result, dict):
                    # mem0 v1.1 返回格式: {"results": [{"id": ..., "memory": ..., "event": "ADD"}, ...]}
                    results_list = result.get("results", [])
                    batch_added = sum(
                        1 for r in results_list
                        if r.get("event") in ("ADD", "UPDATE")
                    )
                elif isinstance(result, list):
                    batch_added = len(result)

                total_memories_added += batch_added
                logger.debug(
                    "  [mem0.add] batch %d/%d: msgs=%d, memories_added=%d",
                    batch_idx + 1, len(batches), len(batch), batch_added,
                )

            except Exception as e:
                logger.error(
                    "mem0.add batch %d/%d failed: user_id=%s, msgs=%d, error=%s",
                    batch_idx + 1, len(batches), user_id, len(batch), e,
                )
                last_error_info = {
                    "type": type(e).__name__,
                    "detail": str(e)[:500],
                    "batch": f"{batch_idx + 1}/{len(batches)}",
                }

        elapsed = time.monotonic() - t0
        logger.debug(
            "  [mem0.add] user_id=%s, total_memories_added=%d, elapsed=%.2fs",
            user_id, total_memories_added, elapsed,
        )
        return elapsed, total_memories_added, last_error_info

    # ------------------------------------------------------------------
    # Mem0 操作：search + LLM 回答
    # ------------------------------------------------------------------

    async def _search_and_answer(
        self,
        mem0_client,
        user_id: str,
        question: str,
    ) -> tuple[str, float, float, float, int, int, int, float, str, dict[str, Any] | None]:
        """调用 mem0.search() 检索记忆，然后用 LLM 回答问题。

        Args:
            mem0_client: mem0 Memory 实例。
            user_id: 用户 ID。
            question: 完整问题文本（含 MCQ options）。

        Returns:
            元组 (答案, answer_elapsed(search+llm完整耗时), search_elapsed,
                   llm_elapsed(纯LLM生成耗时), 检索到的记忆条数,
                   输入token数, 输出token数, TTFT秒数, 召回记忆文本, 错误信息或None)。
        """
        # 整体计时起点（search + answer 完整耗时）
        total_t0 = time.monotonic()
        # Phase 1: 检索相关记忆
        search_t0 = time.monotonic()
        memories_text = ""
        memories_count = 0
        try:
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                None,
                functools.partial(
                    mem0_client.search,
                    question,
                    filters={"user_id": user_id},
                    limit=self.config.mem0_search_limit,
                ),
            )
            search_elapsed = time.monotonic() - search_t0

            # 解析搜索结果
            if isinstance(search_results, dict):
                results_list = search_results.get("results", [])
            elif isinstance(search_results, list):
                results_list = search_results
            else:
                results_list = []

            memories_count = len(results_list)

            # 将记忆格式化为文本
            memory_lines = []
            for i, mem in enumerate(results_list):
                if isinstance(mem, dict):
                    memory_text = mem.get("memory", mem.get("text", mem.get("content", "")))
                    score = mem.get("score", 0)
                    memory_lines.append(f"[Memory {i + 1}] (relevance: {score:.2f}) {memory_text}")
                else:
                    memory_lines.append(f"[Memory {i + 1}] {str(mem)}")
            memories_text = "\n".join(memory_lines)

            logger.debug(
                "  [mem0.search] user_id=%s, query_len=%d, memories_found=%d, elapsed=%.2fs",
                user_id, len(question), memories_count, search_elapsed,
            )

        except Exception as e:
            search_elapsed = time.monotonic() - search_t0
            logger.error(
                "mem0.search failed: user_id=%s, error=%s, elapsed=%.2fs",
                user_id, e, search_elapsed,
            )
            # 搜索失败时继续用空记忆回答
            memories_text = ""
            memories_count = 0

        # Phase 2: 用 LLM 回答问题
        llm_t0 = time.monotonic()
        try:
            llm = self._get_llm()

            # PersonaMem 官方 instructions
            instructions = (
                "Find the most appropriate model response and give your final answer "
                "(a), (b), (c), or (d) after the special token <final_answer>."
            )

            # 构建 system prompt：包含检索到的记忆
            if memories_text:
                system_prompt = (
                    "You are a helpful assistant with access to the user's personal memories. "
                    "Use the following memories to answer the user's question accurately.\n\n"
                    "=== Retrieved Memories ===\n"
                    f"{memories_text}\n"
                    "=== End of Memories ===\n\n"
                    "Based on these memories, answer the following question."
                )
            else:
                system_prompt = (
                    "You are a helpful assistant. Answer the user's question to the best of your ability."
                )

            messages = [
                {"role": "user", "content": f"{question}\n\n{instructions}"},
            ]

            # 使用 streaming 模式一次调用同时获取答案和 TTFT（首 token 时延）
            response, ttft = await llm.generate_with_ttft(
                system=system_prompt,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            llm_elapsed = time.monotonic() - llm_t0
            answer_elapsed = time.monotonic() - total_t0  # search + llm 完整耗时
            content = response.content or ""

            logger.info(
                "  [mem0_answer] model=%s input_tokens=%d output_tokens=%d "
                "content_len=%d answer_elapsed=%.2fs (search=%.2fs + llm=%.2fs) "
                "ttft=%.3fs memories=%d",
                self.config.model, response.input_tokens, response.output_tokens,
                len(content), answer_elapsed, search_elapsed, llm_elapsed,
                ttft, memories_count,
            )
            return (
                content, answer_elapsed, search_elapsed, llm_elapsed, memories_count,
                response.input_tokens, response.output_tokens, ttft, memories_text, None,
            )

        except Exception as e:
            llm_elapsed = time.monotonic() - llm_t0
            answer_elapsed = time.monotonic() - total_t0
            logger.error(
                "Mem0 answer failed: model=%s, error=%s, answer_elapsed=%.2fs",
                self.config.model, e, answer_elapsed,
            )
            error_info = {
                "type": type(e).__name__,
                "detail": str(e)[:500],
            }
            return (
                f"[ERROR] {type(e).__name__}: {str(e)}",
                answer_elapsed, search_elapsed, llm_elapsed, memories_count,
                0, 0, 0.0, memories_text, error_info,
            )

    # ------------------------------------------------------------------
    # 统计汇总
    # ------------------------------------------------------------------

    def _aggregate_stats(self, conv_results: list[ConversationResult]) -> dict[str, Any]:
        """汇总所有 conversation 的统计信息。"""
        all_ingest_elapsed: list[float] = []
        all_search_elapsed: list[float] = []
        all_llm_elapsed: list[float] = []
        all_answer_elapsed: list[float] = []
        all_ttft: list[float] = []
        all_input_tokens: list[int] = []
        all_output_tokens: list[int] = []
        all_conv_total_tokens: list[int] = []
        all_memories_retrieved: list[int] = []
        all_memories_added: list[int] = []
        total_ingest_errors = 0
        total_answer_errors = 0

        for cr in conv_results:
            all_ingest_elapsed.extend(cr.ingest_elapsed_list)
            all_search_elapsed.extend(cr.search_elapsed_list)
            all_llm_elapsed.extend(cr.llm_elapsed_list)
            all_answer_elapsed.extend(cr.answer_elapsed_list)
            all_ttft.extend(cr.ttft_list)
            all_input_tokens.extend(cr.input_tokens_list)
            all_output_tokens.extend(cr.output_tokens_list)
            if cr.conv_total_tokens > 0:
                all_conv_total_tokens.append(cr.conv_total_tokens)
            all_memories_retrieved.extend(cr.memories_retrieved_list)
            all_memories_added.extend(cr.memories_added_list)
            total_ingest_errors += len(cr.ingest_errors)
            total_answer_errors += len(cr.answer_errors)

        # 辅助函数：计算百分位
        def _percentile(sorted_values: list[float], p: float) -> float:
            if not sorted_values:
                return 0.0
            idx = int(len(sorted_values) * p)
            idx = min(idx, len(sorted_values) - 1)
            return sorted_values[idx]

        # Ingest 统计
        ingest_count = len(all_ingest_elapsed)
        ingest_total_s = sum(all_ingest_elapsed)
        ingest_avg_s = (ingest_total_s / ingest_count) if ingest_count > 0 else 0.0
        ingest_max_s = max(all_ingest_elapsed) if all_ingest_elapsed else 0.0
        ingest_p90_s = _percentile(sorted(all_ingest_elapsed), 0.9)

        # Search 统计
        search_count = len(all_search_elapsed)
        search_total_s = sum(all_search_elapsed)
        search_avg_s = (search_total_s / search_count) if search_count > 0 else 0.0
        search_max_s = max(all_search_elapsed) if all_search_elapsed else 0.0
        search_p90_s = _percentile(sorted(all_search_elapsed), 0.9)

        # LLM Generate 统计（纯 LLM 生成耗时）
        llm_count = len(all_llm_elapsed)
        llm_total_s = sum(all_llm_elapsed)
        llm_avg_s = (llm_total_s / llm_count) if llm_count > 0 else 0.0
        llm_max_s = max(all_llm_elapsed) if all_llm_elapsed else 0.0
        llm_p90_s = _percentile(sorted(all_llm_elapsed), 0.9)

        # Answer 统计（search + llm 完整耗时）
        answer_count = len(all_answer_elapsed)
        answer_total_s = sum(all_answer_elapsed)
        answer_avg_s = (answer_total_s / answer_count) if answer_count > 0 else 0.0
        answer_max_s = max(all_answer_elapsed) if all_answer_elapsed else 0.0
        sorted_answer = sorted(all_answer_elapsed) if all_answer_elapsed else []
        answer_p50_s = _percentile(sorted_answer, 0.5)
        answer_p90_s = _percentile(sorted_answer, 0.9)
        answer_p99_s = _percentile(sorted_answer, 0.99)

        # TTFT 统计
        valid_ttft = [t for t in all_ttft if t > 0]
        sorted_ttft = sorted(valid_ttft) if valid_ttft else []
        ttft_count = len(valid_ttft)
        ttft_avg = (sum(valid_ttft) / ttft_count) if ttft_count > 0 else 0.0
        ttft_p50 = _percentile(sorted_ttft, 0.5)
        ttft_p90 = _percentile(sorted_ttft, 0.9)
        ttft_p99 = _percentile(sorted_ttft, 0.99)

        # 单次对话 in token 统计
        total_input_tokens = sum(all_input_tokens)
        total_output_tokens = sum(all_output_tokens)
        avg_input_tokens = (total_input_tokens / answer_count) if answer_count > 0 else 0
        sorted_input_tokens = sorted(all_input_tokens) if all_input_tokens else []
        p50_input_tokens = _percentile([float(x) for x in sorted_input_tokens], 0.5)

        # Conversation 累计 token 统计
        conv_tokens_avg = (
            round(sum(all_conv_total_tokens) / len(all_conv_total_tokens))
            if all_conv_total_tokens else 0
        )
        conv_tokens_max = max(all_conv_total_tokens) if all_conv_total_tokens else 0
        sorted_conv_tokens = sorted(all_conv_total_tokens) if all_conv_total_tokens else []
        conv_tokens_p50 = _percentile([float(x) for x in sorted_conv_tokens], 0.5)

        # 记忆统计
        total_memories_added = sum(all_memories_added)
        total_memories_retrieved = sum(all_memories_retrieved)
        avg_memories_retrieved = (
            total_memories_retrieved / search_count
        ) if search_count > 0 else 0.0

        # 错误统计
        total_errors = total_ingest_errors + total_answer_errors
        total_requests = ingest_count + answer_count
        error_rate = (total_errors / total_requests) if total_requests > 0 else 0.0

        return {
            # Ingest (mem0.add) 统计
            "ingest_count": ingest_count,
            "ingest_total_s": round(ingest_total_s, 2),
            "ingest_avg_s": round(ingest_avg_s, 2),
            "ingest_max_s": round(ingest_max_s, 2),
            "ingest_p90_s": round(ingest_p90_s, 2),
            # Search (mem0.search) 统计
            "search_count": search_count,
            "search_total_s": round(search_total_s, 2),
            "search_avg_s": round(search_avg_s, 2),
            "search_max_s": round(search_max_s, 2),
            "search_p90_s": round(search_p90_s, 2),
            # LLM Generate 统计（纯 LLM 生成耗时）
            "llm_generate_count": llm_count,
            "llm_generate_total_s": round(llm_total_s, 2),
            "llm_generate_avg_s": round(llm_avg_s, 2),
            "llm_generate_max_s": round(llm_max_s, 2),
            "llm_generate_p90_s": round(llm_p90_s, 2),
            # Answer 统计（search + llm 完整耗时，用于和 memory 服务对比）
            "answer_api_count": answer_count,
            "answer_api_total_s": round(answer_total_s, 2),
            "answer_api_avg_s": round(answer_avg_s, 2),
            "answer_api_max_s": round(answer_max_s, 2),
            "answer_api_p50_s": round(answer_p50_s, 2),
            "answer_api_p90_s": round(answer_p90_s, 2),
            "answer_api_p99_s": round(answer_p99_s, 2),
            # TTFT 统计
            "ttft_count": ttft_count,
            "ttft_avg_s": round(ttft_avg, 4),
            "ttft_p50_s": round(ttft_p50, 4),
            "ttft_p90_s": round(ttft_p90, 4),
            "ttft_p99_s": round(ttft_p99, 4),
            # 单次对话 in token 统计
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "avg_input_tokens_per_question": round(avg_input_tokens),
            "p50_input_tokens": round(p50_input_tokens),
            # Conversation 累计 token 统计
            "conv_total_tokens_avg": conv_tokens_avg,
            "conv_total_tokens_max": conv_tokens_max,
            "conv_total_tokens_p50": round(conv_tokens_p50),
            # 记忆统计
            "total_memories_added": total_memories_added,
            "total_memories_retrieved": total_memories_retrieved,
            "avg_memories_per_search": round(avg_memories_retrieved, 1),
            # 错误统计
            "ingest_error_count": total_ingest_errors,
            "answer_error_count": total_answer_errors,
            "total_errors": total_errors,
            "total_requests": total_requests,
            "error_rate": round(error_rate, 4),
        }

    # ------------------------------------------------------------------
    # LLM-as-Judge
    # ------------------------------------------------------------------

    async def _run_judge(
        self,
        bench_name: str,
        predictions: list[str],
        references: list[str],
        qa_pairs: list[QAPair],
        metrics: dict[str, float],
    ) -> list[float]:
        """运行 LLM-as-Judge 评分（与 eval_script.py 保持一致）。"""
        judge_config = LLMConfig(
            provider="openai",
            model=self.config.judge_model,
            base_url=self.config.judge_base_url,
            api_key=self.config.judge_api_key,
        )
        judge = LLMJudge(judge_config)
        questions = [qa.question for qa in qa_pairs]
        question_types = [qa.question_type for qa in qa_pairs]

        sample_size = len(predictions) if bench_name == "locomo" else min(50, len(predictions))
        judge_scores: list[float] = []

        try:
            if bench_name == "locomo":
                locomo_results = await judge.score_locomo_batch(
                    questions[:sample_size],
                    references[:sample_size],
                    predictions[:sample_size],
                    question_types[:sample_size],
                )
                judge_scores = [r.score for r in locomo_results]
                if judge_scores:
                    metrics["judge_accuracy"] = sum(judge_scores) / len(judge_scores)
            else:
                judge_results = await judge.score_yesno_batch(
                    questions[:sample_size],
                    references[:sample_size],
                    predictions[:sample_size],
                    question_types[:sample_size],
                    prompt_set="longmemeval",
                )
                judge_scores = [r.score for r in judge_results]
                if judge_scores:
                    metrics["judge_accuracy"] = sum(judge_scores) / len(judge_scores)
        except Exception as e:
            logger.error("Mem0Runner: judge failed for %s: %s", bench_name, e)
            metrics["judge_accuracy"] = -1.0
            metrics["judge_error"] = 1.0

        return judge_scores


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Mem0 评测脚本 —— 使用 mem0 记忆框架作为 baseline 对比",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--benchmarks",
        type=str,
        default="personamem",
        help="要评测的 benchmark 名称，多个用逗号分隔",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(_EVAL_DIR, "benchmarks", "data", "personamem"),
        help="Benchmark 数据目录路径",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="32k",
        help="PersonaMem 数据集 split（32k / 128k / 1M）",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="每个 benchmark 取前 N 条 conversation（不指定则全量）",
    )
    parser.add_argument(
        "--conv-concurrency",
        type=int,
        default=5,
        help="同一 benchmark 内并行处理的 conversation 数量（mem0 建议不要太高）",
    )

    # LLM 配置（回答问题）
    parser.add_argument(
        "--model",
        type=str,
        default="ep-2bjbej6a",
        help="用于回答问题的 LLM 模型名称（tokenhub 上 ep-2bjbej6a 背后即 gpt-5.4）",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="LLM provider（openai / openai_compat / anthropic）",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="LLM API 的 base URL",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="LLM API Key",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM 采样温度",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="LLM 最大生成 token 数",
    )

    # Mem0 配置
    parser.add_argument(
        "--mem0-provider",
        type=str,
        default="openai_compat",
        help="Mem0 使用的 LLM provider",
    )
    parser.add_argument(
        "--mem0-model",
        type=str,
        default="ep-2bjbej6a",
        help="Mem0 使用的 LLM 模型（用于记忆提取和摘要，tokenhub 上 ep-2bjbej6a 背后即 gpt-5.4）",
    )
    parser.add_argument(
        "--mem0-base-url",
        type=str,
        default=None,
        help="Mem0 LLM API 的 base URL",
    )
    parser.add_argument(
        "--mem0-api-key",
        type=str,
        default=None,
        help="Mem0 LLM API Key",
    )
    parser.add_argument(
        "--mem0-embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="Mem0 使用的 embedding 模型",
    )
    parser.add_argument(
        "--mem0-embedding-base-url",
        type=str,
        default=None,
        help="Mem0 embedding API 的 base URL",
    )
    parser.add_argument(
        "--mem0-embedding-api-key",
        type=str,
        default=None,
        help="Mem0 embedding API Key",
    )
    parser.add_argument(
        "--mem0-search-limit",
        type=int,
        default=20,
        help="mem0.search() 返回的最大记忆条数",
    )

    # 评测标识
    parser.add_argument(
        "--eval-run-id",
        type=str,
        default=None,
        help="本次评测的时间标识（默认自动生成 mem0_ 前缀时间戳）",
    )

    # Judge 配置
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-5.5",
        help="LLM Judge 评分模型",
    )
    parser.add_argument(
        "--judge-base-url",
        type=str,
        default="https://vdbteam.cognitiveservices.azure.com/openai/v1/",
        help="Judge 模型的 API 地址",
    )
    parser.add_argument(
        "--judge-api-key",
        type=str,
        default="d0924051350c460295107e33772e1e49",
        help="Judge 模型的 API Key",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        default=False,
        help="跳过 LLM Judge 评分",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )

    return parser.parse_args()


def _serialize_result(obj: Any) -> Any:
    """将评测结果对象递归序列化为可 JSON 化的字典。"""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _serialize_result(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_serialize_result(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialize_result(v) for k, v in obj.items()}
    return obj


def _save_results_json(
    results: list[Mem0BenchmarkResult],
    eval_run_id: str,
    config: Mem0RunConfig | None = None,
) -> None:
    """将评测结果保存为 JSON 文件到 eval/results/ 目录。"""
    results_dir = os.path.join(_EVAL_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    filename = f"eval_{eval_run_id}.json"
    filepath = os.path.join(results_dir, filename)

    data = {
        "eval_run_id": eval_run_id,
        "eval_mode": "mem0",
        "created_at": datetime.now().isoformat(),
        "base_model": config.model if config else "unknown",
        "memory_model": config.mem0_model if config else "unknown",
        "results": [_serialize_result(r) for r in results],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("评测结果已保存: %s", filepath)
    print(f"\n📁 评测结果已保存: {filepath}")
    print(f"   启动 Dashboard 查看: python -m eval.dashboard.server")


async def async_main() -> None:
    """异步主函数，解析参数并运行评测。"""
    args = parse_args()

    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 构建配置
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]

    config_kwargs: dict[str, Any] = {
        "benchmarks": benchmarks,
        "subset_size": args.subset_size,
        "conv_concurrency": args.conv_concurrency,
        "model": args.model,
        "provider": args.provider,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "mem0_provider": args.mem0_provider,
        "mem0_model": args.mem0_model,
        "mem0_embedding_model": args.mem0_embedding_model,
        "mem0_search_limit": args.mem0_search_limit,
        "judge_model": args.judge_model,
        "judge_base_url": args.judge_base_url,
        "judge_api_key": args.judge_api_key,
        "no_judge": args.no_judge,
    }
    if args.base_url:
        config_kwargs["base_url"] = args.base_url
    if args.api_key:
        config_kwargs["api_key"] = args.api_key
    if args.mem0_base_url:
        config_kwargs["mem0_base_url"] = args.mem0_base_url
    if args.mem0_api_key:
        config_kwargs["mem0_api_key"] = args.mem0_api_key
    if args.mem0_embedding_base_url:
        config_kwargs["mem0_embedding_base_url"] = args.mem0_embedding_base_url
    if args.mem0_embedding_api_key:
        config_kwargs["mem0_embedding_api_key"] = args.mem0_embedding_api_key
    if args.eval_run_id:
        config_kwargs["eval_run_id"] = args.eval_run_id
    config = Mem0RunConfig(**config_kwargs)

    # 创建 Runner
    runner = Mem0Runner(config)

    # 注册 benchmark adapter
    for bench_name in benchmarks:
        if bench_name == "personamem":
            from benchmarks.personamem_adapter import PersonaMemAdapter
            runner.register_benchmark(PersonaMemAdapter(args.data_dir, split=args.split))
        else:
            logger.warning(
                "未知的 benchmark '%s'，请手动注册 adapter 或扩展此脚本",
                bench_name,
            )

    # 运行评测
    results = await runner.run()

    # 保存结果
    _save_results_json(results, config.eval_run_id, config)

    # 输出结果摘要
    print("\n" + "=" * 70)
    print("Mem0 评测结果摘要（Baseline）")
    print("=" * 70)
    for result in results:
        print(f"\nBenchmark: {result.benchmark_name}")
        print(f"  评测模式: Mem0 (mem0 记忆框架)")
        print(f"  回答模型: {config.model}")
        print(f"  Mem0 模型: {config.mem0_model}")
        print(f"  耗时: {result.elapsed_seconds:.1f}s")
        print(f"  Metrics:")
        for k, v in result.metrics.items():
            print(f"    {k}: {v:.4f}")
        if result.judge_scores:
            avg_judge = sum(result.judge_scores) / len(result.judge_scores)
            print(f"  Judge 平均分: {avg_judge:.4f}")
        print(f"  Stats:")
        for k, v in result.stats.items():
            print(f"    {k}: {v}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(async_main())
