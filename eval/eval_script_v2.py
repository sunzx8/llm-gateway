"""Context Task 评测 Runner v2（基于 eval_script.py 的改进版，独立保留，原版不动）。

v2 相对 v1 的差异：
1. **ingest 改成 batch 喂入**（与 gateway 端 `max_messages_per_ingest` 对齐）：
   一个 session 内若消息数 > `--ingest-batch-size`（默认 10），会被切成多个 batch
   顺序调 ingest API；每个 batch 的 session_id 自动加 `__bN of total` 后缀。
2. **per-conversation 进度条**：每条 conv 启动时打印总工作量（sessions/ingest_batches/questions），
   ingest 期间和 retrieve 期间各自实时刷新单行进度；conv 完成换行进入下一条。
   `--no-progress` 关闭。

其他逻辑（接口签名、判分流程、结果输出）均与 v1 一致，可直接替换 entry point 使用。

基于 benchmark 调用 context_task（IngestContextTask + RetrieveContextTask）
完成端到端评测。评测结果仅保留在内存中，不持久化到磁盘。

并发策略
~~~~~~~~
- ``benchmark_concurrency``: 同时运行的 benchmark 数量（默认 1，顺序执行）。
- ``conv_concurrency``: 同一 benchmark 内并行处理的 conversation 数量（默认 1）。
  每个 conversation 拥有独立的 stores（fs/vec/graph），互不干扰。
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# 确保 eval/ 目录在搜索路径中，以便 benchmarks 等子模块可被正确导入
_EVAL_DIR = str(Path(__file__).resolve().parent)
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

# 确保项目根目录也在搜索路径中，以便 config 等模块可被正确导入
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import httpx

from benchmarks.base_benchmark import BaseBenchmark, Conversation, QAPair
from config.models import LLMConfig
from judge import LLMJudge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class ContextTaskRunConfig:
    """Context Task 评测运行配置。"""

    benchmarks: list[str] = field(default_factory=lambda: ["personamem"])
    """要评测的 benchmark 名称列表。"""

    subset_size: int | None = None
    """每个 benchmark 取前 N 条 conversation（None = 全量）。"""

    benchmark_concurrency: int = 1
    """同时运行的 benchmark 数量。"""

    conv_concurrency: int = 20
    """同一 benchmark 内并行处理的 conversation 数量。"""

    model: str = "ep-2bjbej6a"
    """用于 ingest / retrieve 的 LLM 模型名称（tokenhub 上 ep-2bjbej6a 背后即 gpt-5.4）。"""

    # Gateway 接口配置（用于 ingest 调用）
    gateway_base_url: str = "http://127.0.0.1:8000"
    """LLM Gateway 服务地址。"""

    gateway_api_key: str = "sk-my-test-key-123"
    """Gateway 接口的 API Key。"""

    ingest_model: str = "memory-initialize"
    """摄入接口使用的模型名称。"""

    eval_run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S"))
    """本次评测的时间标识，用作 user_id 后缀以区分不同评测轮次。"""

    # Judge 配置（与 EvalRunner 保持一致）
    judge_model: str = "gpt-5.5"
    judge_base_url: str = "https://vdbteam.cognitiveservices.azure.com/openai/v1/"
    judge_api_key: str = "d0924051350c460295107e33772e1e49"
    no_judge: bool = False

    api_timeout: float = 3600.0
    """调用 Gateway 接口（ingest / answer）的超时时间（秒），默认 3600。"""

    ingest_append_questions: int = 3
    """摄入记忆时，从当前 session 的 questions 中取前 n 个拼接到 messages 末尾（默认 3）。"""

    # ── v2 新增 ──
    ingest_batch_size: int = 10
    """每次调用 ingest API 一次发送的最大消息数。
    一个 session 内若消息数 > 此值，会被切成多个 batch 顺序发送。
    与 gateway 端 `max_messages_per_ingest` 对齐时记忆质量最佳。"""

    show_progress: bool = True
    """是否在控制台打印 ingest / retrieve 的 per-conversation 进度条。"""


# ---------------------------------------------------------------------------
# 结果数据结构
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
    # 统计信息
    ingest_stats: list[dict[str, Any]] = field(default_factory=list)
    """每次 ingest 调用返回的 task_results 中 task_type='ingest' 的结果列表"""
    retrieve_stats: list[dict[str, Any]] = field(default_factory=list)
    """每次 answer 调用返回的 task_results 中 task_type='retrieve' 的结果列表"""
    consolidate_stats: list[dict[str, Any]] = field(default_factory=list)
    """所有 task_type='consolidate' 的结果列表"""
    # 错误统计
    ingest_errors: list[dict[str, Any]] = field(default_factory=list)
    """Ingest API 请求失败的错误信息列表"""
    answer_errors: list[dict[str, Any]] = field(default_factory=list)
    """Answer API 请求失败的错误信息列表"""
    # Answer API 耗时统计
    answer_elapsed_list: list[float] = field(default_factory=list)
    """每次 _call_answer_api 的端到端耗时（秒），与 predictions 一一对应"""
    # TTFT（首 token 时延）统计
    ttft_list: list[float] = field(default_factory=list)
    """每次 answer API 调用的首 token 时延（秒），与 predictions 一一对应"""
    # Token 统计
    input_tokens_list: list[int] = field(default_factory=list)
    """每次 answer API 调用的输入 token 数"""
    output_tokens_list: list[int] = field(default_factory=list)
    """每次 answer API 调用的输出 token 数"""
    # Conversation 级别累计 token
    conv_total_tokens: int = 0
    """该 conversation 内所有 QA 的累计 token（input + output）"""
    # 召回记忆内容
    retrieved_contexts: list[str] = field(default_factory=list)
    """每个 QA 对应的召回记忆内容，与 predictions 一一对应"""
    # QA 正确率统计
    qa_pass_flags: list[bool] = field(default_factory=list)
    """每个 QA 是否回答正确的布尔列表，True 表示正确"""
    mcq_accuracy: float = 0.0
    """该 conversation 的 MCQ 准确率（0.0 ~ 1.0）"""


@dataclass
class ContextTaskBenchmarkResult:
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
# ContextTaskRunner
# ---------------------------------------------------------------------------


class ContextTaskRunner:
    """基于 context_task 的评测 Runner。

    对每个 conversation：
    1. 创建独立的临时 stores（fs / vec / graph）
    2. 按 session 顺序调用 IngestContextTask
    3. （可选）调用 ConsolidateContextTask
    4. 对每个 QAPair 调用 RetrieveContextTask + 主 Agent 生成答案
    5. 收集 predictions / references，计算 metrics

    评测结果仅保留在内存中，不写入磁盘。
    """

    def __init__(self, config: ContextTaskRunConfig):
        self.config = config
        self._benchmarks: dict[str, BaseBenchmark] = {}
        self._results: list[ContextTaskBenchmarkResult] = []
        # 实时进度追踪
        self._progress_completed: int = 0
        self._progress_passed: int = 0
        self._progress_lock = asyncio.Lock()
        self._progress_file: str = os.path.join(
            _EVAL_DIR, "results", f"eval_{config.eval_run_id}_progress.json",
        )
        # 增量过程数据文件（每完成一个 conversation 就追加写入）
        self._incremental_file: str = os.path.join(
            _EVAL_DIR, "results", f"eval_{config.eval_run_id}_incremental.jsonl",
        )
        self._incremental_lock = asyncio.Lock()

    def register_benchmark(self, benchmark: BaseBenchmark) -> None:
        """注册一个 benchmark adapter。"""
        self._benchmarks[benchmark.name] = benchmark

    async def run(self) -> list[ContextTaskBenchmarkResult]:
        """运行所有已注册 benchmark 的评测。

        Returns:
            每个 benchmark 的 ContextTaskBenchmarkResult 列表。
        """
        self._results = []
        run_start = time.monotonic()

        logger.info(
            "=== ContextTaskRunner: %d benchmarks, "
            "benchmark_concurrency=%d, conv_concurrency=%d ===",
            len(self.config.benchmarks),
            self.config.benchmark_concurrency,
            self.config.conv_concurrency,
        )

        bench_names = [
            name for name in self.config.benchmarks
            if name in self._benchmarks
        ]
        skipped = [n for n in self.config.benchmarks if n not in self._benchmarks]
        for name in skipped:
            logger.warning("Benchmark %s not registered, skipping", name)

        if self.config.benchmark_concurrency > 1 and len(bench_names) > 1:
            # 并发运行多个 benchmark
            bench_sem = asyncio.Semaphore(self.config.benchmark_concurrency)

            async def _run_bench_with_sem(name: str) -> ContextTaskBenchmarkResult:
                async with bench_sem:
                    return await self._evaluate_benchmark(self._benchmarks[name])

            results = await asyncio.gather(*[_run_bench_with_sem(n) for n in bench_names])
            self._results.extend(results)
        else:
            # 顺序运行
            for name in bench_names:
                result = await self._evaluate_benchmark(self._benchmarks[name])
                self._results.append(result)

        total_elapsed = time.monotonic() - run_start
        logger.info(
            "=== ContextTaskRunner done: %d results, total %.1fs ===",
            len(self._results), total_elapsed,
        )
        return self._results

    # ------------------------------------------------------------------
    # 单个 benchmark 评测
    # ------------------------------------------------------------------

    async def _evaluate_benchmark(
        self,
        benchmark: BaseBenchmark,
    ) -> ContextTaskBenchmarkResult:
        """评测单个 benchmark。"""
        start_time = time.monotonic()
        bench_name = benchmark.name

        logger.info("=" * 60)
        logger.info("ContextTaskRunner: evaluating benchmark=%s", bench_name)

        # 加载数据
        conversations = benchmark.load_data()
        if self.config.subset_size:
            conversations = benchmark.subset(conversations, self.config.subset_size)
        logger.info(
            "Loaded %d conversations (subset=%s)",
            len(conversations), self.config.subset_size or "full",
        )

        # 并发或顺序处理 conversations
        if self.config.conv_concurrency > 1:
            conv_results = await self._run_conversations_concurrent(
                benchmark, conversations,
            )
        else:
            conv_results = await self._run_conversations_sequential(
                benchmark, conversations,
            )

        # 汇总 predictions / references / qa_pairs
        all_predictions: list[str] = []
        all_references: list[str] = []
        all_qa_pairs: list[QAPair] = []
        for cr in conv_results:
            all_predictions.extend(cr.predictions)
            all_references.extend(cr.references)
            all_qa_pairs.extend(cr.qa_pairs)

        # 计算 metrics
        try:
            metrics = benchmark.evaluate(
                all_predictions, all_references, qa_pairs=all_qa_pairs,
            )
        except Exception as e:
            logger.exception(
                "ContextTaskRunner: benchmark.evaluate failed: %s", e,
            )
            metrics = {"evaluate_error": -1.0}
        logger.info(
            "ContextTaskRunner: benchmark=%s metrics=%s",
            bench_name,
            {k: f"{v:.3f}" for k, v in metrics.items()},
        )

        # 为每个 conversation 计算 MCQ 准确率
        try:
            self._compute_mcq_accuracy(benchmark, conv_results)
        except Exception as e:
            logger.exception(
                "ContextTaskRunner: _compute_mcq_accuracy failed: %s", e,
            )

        # LLM-as-Judge（与 EvalRunner 保持一致）
        judge_scores: list[float] = []
        if bench_name in ("longmemeval", "locomo") and not self.config.no_judge:
            try:
                judge_scores = await self._run_judge(
                    bench_name, all_predictions, all_references, all_qa_pairs, metrics,
                )
            except Exception as e:
                logger.exception(
                    "ContextTaskRunner: _run_judge failed: %s", e,
                )
                judge_scores = []

        elapsed = time.monotonic() - start_time

        # 汇总 benchmark 级别的任务统计
        try:
            task_stats_summary = self._aggregate_task_stats(conv_results)
        except Exception as e:
            logger.exception(
                "ContextTaskRunner: _aggregate_task_stats failed: %s", e,
            )
            task_stats_summary = {"aggregate_error": str(e)}

        return ContextTaskBenchmarkResult(
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
                **task_stats_summary,
            },
        )

    # ------------------------------------------------------------------
    # 计算每个 Conversation 的 MCQ 准确率
    # ------------------------------------------------------------------

    def _compute_mcq_accuracy(
        self,
        benchmark: BaseBenchmark,
        conv_results: list[ConversationResult],
    ) -> None:
        """为每个 ConversationResult 计算 MCQ 准确率。

        利用 benchmark 的 check_correctness 方法逐条判断每个 QA 是否正确，
        并将结果写入 ConversationResult 的 qa_pass_flags 和 mcq_accuracy 字段。

        Args:
            benchmark: 当前 benchmark adapter 实例。
            conv_results: 所有 conversation 的评测结果列表。
        """
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

    async def _run_conversations_concurrent(
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
                    result = await self._evaluate_conversation(benchmark, conv, total=total)
                    result_map[conv.conv_id] = result
                    # 增量保存过程数据
                    await self._save_conversation_incremental(result)
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

    async def _run_conversations_sequential(
        self,
        benchmark: BaseBenchmark,
        conversations: list[Conversation],
    ) -> list[ConversationResult]:
        """顺序处理 conversations。"""
        results: list[ConversationResult] = []
        total = len(conversations)
        for idx, conv in enumerate(conversations):
            logger.info(
                "  [%d/%d] conversation=%s",
                idx + 1, total, conv.conv_id,
            )
            result = await self._evaluate_conversation(benchmark, conv, total=total)
            results.append(result)
            # 增量保存过程数据
            await self._save_conversation_incremental(result)
            logger.info(
                "  进度: %d/%d (%.1f%%) 已完成",
                idx + 1, total, (idx + 1) / total * 100,
            )
        return results

    # ------------------------------------------------------------------
    # 单个 conversation 评测
    # ------------------------------------------------------------------

    async def _evaluate_conversation(
        self,
        benchmark: BaseBenchmark,
        conv: Conversation,
        total: int = 0,
    ) -> ConversationResult:
        """评测单个 conversation。

        流程：
        1. 创建独立临时 stores
        2. 按 session 顺序 ingest
        3. （可选）consolidate
        4. 对每个 QAPair 执行 retrieve + 主 Agent 生成答案
        5. 返回 ConversationResult
        """
        start_time = time.monotonic()
        cr = ConversationResult(conv_id=conv.conv_id)

        try:
            user_id = conv.metadata.get("user_id", conv.conv_id)
            # 拼接评测时间标识后缀，区分不同评测轮次
            user_id = f"{user_id}_{self.config.eval_run_id}"

            # ── 预计算 ingest / retrieve 的总工作量，用于进度显示 ──
            bsz = max(1, int(self.config.ingest_batch_size))
            n_append = self.config.ingest_append_questions
            total_ingest_batches = 0
            for s in conv.sessions:
                sess_qs = getattr(s, "questions", []) or []
                n_msgs = len(s.messages) + min(n_append, len(sess_qs))
                total_ingest_batches += self._count_ingest_batches(n_msgs, bsz)
            total_questions = sum(
                len(getattr(s, "questions", []) or []) for s in conv.sessions
            )
            if total_questions == 0:
                # 非 interleaved 模式：所有 question 挂在 conv 上
                total_questions = len(benchmark.get_questions(conv))

            ingest_progress = {"done": 0, "total": total_ingest_batches}
            retrieve_progress = {"done": 0, "total": total_questions}

            if self.config.show_progress:
                total_label = f"/{total}" if total else ""
                self._progress_print(
                    f"\n[conv={conv.conv_id}{total_label}] start: "
                    f"sessions={len(conv.sessions)}, "
                    f"ingest_batches={total_ingest_batches}, "
                    f"questions={total_questions}, batch_size={bsz}\n"
                )

            # ── Phase 1: Ingest & consolidate ──
            # 检测是否为 interleaved 模式（PersonaMem：每个 session 有附属问题）
            interleaved = any(
                hasattr(s, "questions") and s.questions
                for s in conv.sessions
            )

            if interleaved:
                # Interleaved：ingest session → 立即回答该 session 的问题
                for sess_idx, session in enumerate(conv.sessions):
                    messages = [
                        {"role": m.role, "content": m.content}
                        for m in session.messages
                    ]
                    # 拼接当前 session 的前 n 个 question 到 messages 末尾
                    sess_questions: list[QAPair] = getattr(session, "questions", []) or []
                    for qa in sess_questions[:n_append]:
                        messages.append({"role": "user", "content": qa.question})

                    await self._ingest_messages_in_batches(
                        cr=cr,
                        conv_id=conv.conv_id,
                        user_id=user_id,
                        session_id=session.session_id,
                        messages=messages,
                        sess_idx=sess_idx,
                        sess_total=len(conv.sessions),
                        ingest_progress=ingest_progress,
                    )

                    # 回答该 session 的附属问题
                    if sess_questions:
                        for qa in sess_questions:
                            prediction, answer_task_results, answer_error, answer_elapsed, input_tokens, output_tokens, ttft = await self._call_answer_api(
                                question=qa.question,
                                session_id=session.session_id,
                                user_id=user_id,
                                benchmark_name=benchmark.name,
                            )
                            if answer_error:
                                cr.answer_errors.append(answer_error)
                            cr.predictions.append(prediction)
                            cr.references.append(qa.reference_answer)
                            cr.qa_pairs.append(qa)
                            cr.answer_elapsed_list.append(answer_elapsed)
                            cr.ttft_list.append(ttft)
                            cr.input_tokens_list.append(input_tokens)
                            cr.output_tokens_list.append(output_tokens)
                            cr.retrieved_contexts.append(self._extract_retrieved_context(answer_task_results))
                            self._classify_task_results(cr, answer_task_results)

                            retrieve_progress["done"] += 1
                            if self.config.show_progress:
                                self._progress_print(
                                    f"\r[conv={conv.conv_id}] retrieve "
                                    f"{retrieve_progress['done']}/{retrieve_progress['total']}   "
                                )
                            # 实时更新 benchmark 级别进度
                            await self._update_progress(benchmark, prediction, qa.reference_answer, qa)
            else:
                # 标准模式：先 ingest & consolidate 所有 sessions，再统一回答问题
                for sess_idx, session in enumerate(conv.sessions):
                    messages = [
                        {"role": m.role, "content": m.content}
                        for m in session.messages
                    ]
                    sess_questions_std: list[QAPair] = getattr(session, "questions", []) or []
                    for qa in sess_questions_std[:n_append]:
                        messages.append({"role": "user", "content": qa.question})

                    await self._ingest_messages_in_batches(
                        cr=cr,
                        conv_id=conv.conv_id,
                        user_id=user_id,
                        session_id=session.session_id,
                        messages=messages,
                        sess_idx=sess_idx,
                        sess_total=len(conv.sessions),
                        ingest_progress=ingest_progress,
                    )

                # ── Phase 2: Retrieve + Answer ──
                if self.config.show_progress:
                    self._progress_print(
                        f"\n[conv={conv.conv_id}] ingest done, "
                        f"start retrieve ({retrieve_progress['total']} questions)\n"
                    )
                questions = benchmark.get_questions(conv)
                for qa in questions:
                    last_session_id = (
                        conv.sessions[-1].session_id if conv.sessions else ""
                    )
                    prediction, answer_task_results, answer_error, answer_elapsed, input_tokens, output_tokens, ttft = await self._call_answer_api(
                        question=qa.question,
                        session_id=last_session_id,
                        user_id=user_id,
                        benchmark_name=benchmark.name,
                    )
                    if answer_error:
                        cr.answer_errors.append(answer_error)
                    cr.predictions.append(prediction)
                    cr.references.append(qa.reference_answer)
                    cr.qa_pairs.append(qa)
                    cr.answer_elapsed_list.append(answer_elapsed)
                    cr.ttft_list.append(ttft)
                    cr.input_tokens_list.append(input_tokens)
                    cr.output_tokens_list.append(output_tokens)
                    cr.retrieved_contexts.append(self._extract_retrieved_context(answer_task_results))
                    self._classify_task_results(cr, answer_task_results)

                    retrieve_progress["done"] += 1
                    if self.config.show_progress:
                        self._progress_print(
                            f"\r[conv={conv.conv_id}] retrieve "
                            f"{retrieve_progress['done']}/{retrieve_progress['total']}   "
                        )
                    await self._update_progress(benchmark, prediction, qa.reference_answer, qa)

            if self.config.show_progress:
                # 本 conv 结束，换行以便下一 conv 的进度从新行开始
                self._progress_print("\n")

        except Exception as e:
            logger.exception(
                "ContextTaskRunner: conversation=%s failed: %s",
                conv.conv_id, e,
            )
            cr.error = str(e)
            # 对失败的 conversation，用空字符串填充 predictions
            questions = benchmark.get_questions(conv)
            missing = len(questions) - len(cr.predictions)
            if missing > 0:
                cr.predictions.extend([""] * missing)
                cr.references.extend(
                    [qa.reference_answer for qa in questions[len(cr.predictions) - missing:]]
                )
                cr.qa_pairs.extend(questions[len(cr.qa_pairs):])
                cr.retrieved_contexts.extend([""] * missing)

        finally:
            # 计算 conversation 级别累计 token
            cr.conv_total_tokens = sum(cr.input_tokens_list) + sum(cr.output_tokens_list)
            cr.elapsed_seconds = time.monotonic() - start_time
            logger.info(
                "  [conv=%s] done: %d predictions, %.1fs, conv_total_tokens=%d, error=%s",
                conv.conv_id, len(cr.predictions), cr.elapsed_seconds,
                cr.conv_total_tokens, cr.error or "none",
            )

        return cr

    # ------------------------------------------------------------------
    # v2: batch 化 ingest + 进度打印
    # ------------------------------------------------------------------

    @staticmethod
    def _progress_print(text: str) -> None:
        """直接打到 stdout（不走 logger，避免被等级过滤；用 \\r 实现单行刷新）。"""
        sys.stdout.write(text)
        sys.stdout.flush()

    async def _ingest_messages_in_batches(
        self,
        *,
        cr: ConversationResult,
        conv_id: str,
        user_id: str,
        session_id: str,
        messages: list[dict[str, str]],
        sess_idx: int,
        sess_total: int,
        ingest_progress: dict[str, int],
    ) -> None:
        """把一个 session 的 messages 按 batch_size 切片，串行调 ingest API。

        Args:
            cr: 当前 conversation 的 result 对象，用于累计错误/任务统计
            session_id: 上层 session id；若切了多个 batch，会自动加 `__bN` 后缀
            messages: 该 session 完整 messages（含可能 append 的 question）
            sess_idx / sess_total: 当前是第几个 session，共几个
            ingest_progress: { "done": int, "total": int }，本 conv 累计已 ingest 的 batch 计数器
        """
        bsz = max(1, int(self.config.ingest_batch_size))
        total_batches = (len(messages) + bsz - 1) // bsz if messages else 1

        for batch_idx in range(total_batches):
            chunk = messages[batch_idx * bsz : (batch_idx + 1) * bsz]
            # 多 batch 时给 session_id 加后缀，便于 gateway 端区分
            sid = (
                session_id
                if total_batches == 1
                else f"{session_id}__b{batch_idx + 1}of{total_batches}"
            )
            # message_offset: 该 batch 第一条消息在完整对话中的全局编号（从 1 开始）
            message_offset = batch_idx * bsz + 1

            ingest_result = await self._call_ingest_api(
                user_id=user_id,
                session_id=sid,
                messages=chunk,
                message_offset=message_offset,
            )
            if "error" in ingest_result:
                cr.ingest_errors.append({
                    "type": ingest_result.get("error_type", "unknown"),
                    "detail": ingest_result["error"],
                    "user_id": user_id,
                    "session_id": sid,
                })
            self._classify_task_results(cr, ingest_result.get("task_results", []))

            ingest_progress["done"] += 1
            if self.config.show_progress:
                self._progress_print(
                    f"\r[conv={conv_id}] ingest "
                    f"sess {sess_idx + 1}/{sess_total} "
                    f"batch {batch_idx + 1}/{total_batches} "
                    f"(total ingested: {ingest_progress['done']}/{ingest_progress['total']})   "
                )

    @staticmethod
    def _count_ingest_batches(messages_total: int, bsz: int) -> int:
        bsz = max(1, bsz)
        return max(1, (messages_total + bsz - 1) // bsz)

    # ------------------------------------------------------------------
    # Benchmark 级别任务统计汇总
    # ------------------------------------------------------------------

    def _aggregate_task_stats(
        self,
        conv_results: list[ConversationResult],
    ) -> dict[str, Any]:
        """汇总所有 conversation 的任务统计信息到 benchmark 级别。

        从每个 ConversationResult 的 ingest_stats / retrieve_stats / consolidate_stats
        中提取 TaskStats 字段，计算总计和最大值。

        Args:
            conv_results: 所有 conversation 的评测结果列表。

        Returns:
            benchmark 级别的任务统计字典。
        """
        # 收集各类型任务的统计列表
        all_ingest: list[dict[str, Any]] = []
        all_retrieve: list[dict[str, Any]] = []
        all_consolidate: list[dict[str, Any]] = []

        for cr in conv_results:
            all_ingest.extend(cr.ingest_stats)
            all_retrieve.extend(cr.retrieve_stats)
            all_consolidate.extend(cr.consolidate_stats)

        def _get_result(item: dict[str, Any]) -> dict[str, Any]:
            """从完整 task_results item 中提取 result 字段用于统计。"""
            if not isinstance(item, dict):
                return {}
            result = item.get("result")
            return result if isinstance(result, dict) else {}

        def _safe_num(val: Any) -> float | int:
            """安全地将值转为数值，None 或非数值类型返回 0。"""
            if val is None:
                return 0
            if isinstance(val, (int, float)):
                return val
            return 0

        def _summarize(stats_list: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
            """对一组任务统计计算总计和最大值。

            单位约定：耗时使用秒（s），Token 使用千（k）。
            """
            results = [_get_result(s) for s in stats_list]
            count = len(results)
            total_latency = sum(_safe_num(r.get("total_latency_s")) for r in results)
            total_llm_calls = sum(_safe_num(r.get("llm_calls")) for r in results)
            total_llm_latency = sum(_safe_num(r.get("llm_total_latency_s")) for r in results)
            total_llm_tokens_k = sum(_safe_num(r.get("llm_total_tokens_k")) for r in results)
            total_tool_calls = sum(_safe_num(r.get("tool_calls")) for r in results)
            total_tool_latency = sum(_safe_num(r.get("tool_total_latency_s")) for r in results)
            total_messages = sum(_safe_num(r.get("messages_count")) for r in results)

            # 存储访问次数汇总
            total_fs_access = sum(_safe_num(r.get("fs_access_count")) for r in results)
            total_vec_access = sum(_safe_num(r.get("vec_access_count")) for r in results)
            total_graph_access = sum(_safe_num(r.get("graph_access_count")) for r in results)

            max_latency = max((_safe_num(r.get("total_latency_s")) for r in results), default=0)
            max_llm_calls = max((_safe_num(r.get("llm_calls")) for r in results), default=0)
            max_llm_latency = max((_safe_num(r.get("llm_total_latency_s")) for r in results), default=0)
            max_llm_tokens_k = max((_safe_num(r.get("llm_total_tokens_k")) for r in results), default=0)
            max_tool_calls = max((_safe_num(r.get("tool_calls")) for r in results), default=0)
            max_tool_latency = max((_safe_num(r.get("tool_total_latency_s")) for r in results), default=0)

            # 找到最大耗时任务对应的 messages_count
            max_latency_messages_count = 0
            if results:
                max_latency_item = max(results, key=lambda r: _safe_num(r.get("total_latency_s")))
                max_latency_messages_count = _safe_num(max_latency_item.get("messages_count"))

            return {
                f"{prefix}_count": count,
                f"{prefix}_total_latency_s": round(total_latency, 2),
                f"{prefix}_total_llm_calls": total_llm_calls,
                f"{prefix}_total_llm_latency_s": round(total_llm_latency, 2),
                f"{prefix}_total_llm_tokens_k": round(total_llm_tokens_k, 2),
                f"{prefix}_total_tool_calls": total_tool_calls,
                f"{prefix}_total_tool_latency_s": round(total_tool_latency, 2),
                f"{prefix}_total_messages_count": total_messages,
                f"{prefix}_total_fs_access": total_fs_access,
                f"{prefix}_total_vec_access": total_vec_access,
                f"{prefix}_total_graph_access": total_graph_access,
                f"{prefix}_max_latency_s": round(max_latency, 2),
                f"{prefix}_max_llm_calls": max_llm_calls,
                f"{prefix}_max_llm_latency_s": round(max_llm_latency, 2),
                f"{prefix}_max_llm_tokens_k": round(max_llm_tokens_k, 2),
                f"{prefix}_max_tool_calls": max_tool_calls,
                f"{prefix}_max_tool_latency_s": round(max_tool_latency, 2),
                f"{prefix}_max_latency_messages_count": max_latency_messages_count,
            }

        ingest_summary = _summarize(all_ingest, "ingest")
        retrieve_summary = _summarize(all_retrieve, "retrieve")
        consolidate_summary = _summarize(all_consolidate, "consolidate")

        # 全部任务汇总
        all_tasks = all_ingest + all_retrieve + all_consolidate
        all_results = [_get_result(s) for s in all_tasks]
        all_count = len(all_results)
        all_total_latency = sum(_safe_num(r.get("total_latency_s")) for r in all_results)
        all_total_llm_calls = sum(_safe_num(r.get("llm_calls")) for r in all_results)
        all_total_llm_latency = sum(_safe_num(r.get("llm_total_latency_s")) for r in all_results)
        all_total_llm_tokens_k = sum(_safe_num(r.get("llm_total_tokens_k")) for r in all_results)
        all_total_tool_calls = sum(_safe_num(r.get("tool_calls")) for r in all_results)
        all_total_tool_latency = sum(_safe_num(r.get("tool_total_latency_s")) for r in all_results)
        all_total_fs_access = sum(_safe_num(r.get("fs_access_count")) for r in all_results)
        all_total_vec_access = sum(_safe_num(r.get("vec_access_count")) for r in all_results)
        all_total_graph_access = sum(_safe_num(r.get("graph_access_count")) for r in all_results)

        overall_summary = {
            "all_task_count": all_count,
            "all_task_total_latency_s": round(all_total_latency, 2),
            "all_task_total_llm_calls": all_total_llm_calls,
            "all_task_total_llm_latency_s": round(all_total_llm_latency, 2),
            "all_task_total_llm_tokens_k": round(all_total_llm_tokens_k, 2),
            "all_task_total_tool_calls": all_total_tool_calls,
            "all_task_total_tool_latency_s": round(all_total_tool_latency, 2),
            "all_task_total_fs_access": all_total_fs_access,
            "all_task_total_vec_access": all_total_vec_access,
            "all_task_total_graph_access": all_total_graph_access,
        }

        # Answer API 端到端耗时汇总（包含网络传输 + Gateway 处理全流程）
        all_answer_elapsed: list[float] = []
        all_ttft: list[float] = []
        all_input_tokens: list[int] = []
        all_output_tokens: list[int] = []
        all_conv_total_tokens: list[int] = []
        for cr in conv_results:
            all_answer_elapsed.extend(cr.answer_elapsed_list)
            all_ttft.extend(cr.ttft_list)
            all_input_tokens.extend(cr.input_tokens_list)
            all_output_tokens.extend(cr.output_tokens_list)
            if cr.conv_total_tokens > 0:
                all_conv_total_tokens.append(cr.conv_total_tokens)

        def _percentile(sorted_values: list[float], p: float) -> float:
            """计算第 p 百分位值（0 < p < 1）。"""
            if not sorted_values:
                return 0.0
            idx = int(len(sorted_values) * p)
            idx = min(idx, len(sorted_values) - 1)
            return sorted_values[idx]

        answer_api_count = len(all_answer_elapsed)
        answer_api_total_s = sum(all_answer_elapsed)
        answer_api_max_s = max(all_answer_elapsed) if all_answer_elapsed else 0.0
        answer_api_avg_s = (answer_api_total_s / answer_api_count) if answer_api_count > 0 else 0.0

        # 端到端时延百分位
        sorted_elapsed = sorted(all_answer_elapsed) if all_answer_elapsed else []
        answer_api_p50_s = _percentile(sorted_elapsed, 0.5)
        answer_api_p90_s = _percentile(sorted_elapsed, 0.9)
        answer_api_p99_s = _percentile(sorted_elapsed, 0.99)

        # TTFT 统计（首 token 时延，通过 streaming 模式获取）
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
        avg_input_tokens = (total_input_tokens / answer_api_count) if answer_api_count > 0 else 0
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

        answer_api_summary = {
            "answer_api_count": answer_api_count,
            "answer_api_total_s": round(answer_api_total_s, 2),
            "answer_api_max_s": round(answer_api_max_s, 2),
            "answer_api_avg_s": round(answer_api_avg_s, 2),
            "answer_api_p50_s": round(answer_api_p50_s, 2),
            "answer_api_p90_s": round(answer_api_p90_s, 2),
            "answer_api_p99_s": round(answer_api_p99_s, 2),
            # TTFT 统计（通过 streaming 模式获取的真实首 token 时延）
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
        }

        # 错误统计汇总
        error_summary = self._aggregate_error_stats(conv_results)

        return {
            **ingest_summary,
            **retrieve_summary,
            **consolidate_summary,
            **overall_summary,
            **answer_api_summary,
            **error_summary,
        }

    # ------------------------------------------------------------------
    # 错误统计汇总
    # ------------------------------------------------------------------

    def _aggregate_error_stats(
        self,
        conv_results: list[ConversationResult],
    ) -> dict[str, Any]:
        """汇总所有 conversation 的 API 请求错误统计。

        Args:
            conv_results: 所有 conversation 的评测结果列表。

        Returns:
            错误统计字典。
        """
        all_ingest_errors: list[dict[str, Any]] = []
        all_answer_errors: list[dict[str, Any]] = []

        for cr in conv_results:
            all_ingest_errors.extend(cr.ingest_errors)
            all_answer_errors.extend(cr.answer_errors)

        # 按错误类型分类计数
        ingest_error_types: dict[str, int] = {}
        for err in all_ingest_errors:
            err_type = err.get("type", "unknown")
            ingest_error_types[err_type] = ingest_error_types.get(err_type, 0) + 1

        answer_error_types: dict[str, int] = {}
        for err in all_answer_errors:
            err_type = err.get("type", "unknown")
            answer_error_types[err_type] = answer_error_types.get(err_type, 0) + 1

        total_ingest_requests = sum(
            len(cr.ingest_stats) + len(cr.ingest_errors) for cr in conv_results
        )
        total_answer_requests = sum(
            len(cr.retrieve_stats) + len(cr.answer_errors) for cr in conv_results
        )
        total_requests = total_ingest_requests + total_answer_requests
        total_errors = len(all_ingest_errors) + len(all_answer_errors)

        return {
            "ingest_error_count": len(all_ingest_errors),
            "ingest_timeout_count": ingest_error_types.get("timeout", 0),
            "ingest_connect_error_count": ingest_error_types.get("connect_error", 0),
            "ingest_http_error_count": ingest_error_types.get("http_error", 0),
            "answer_error_count": len(all_answer_errors),
            "answer_timeout_count": answer_error_types.get("timeout", 0),
            "answer_connect_error_count": answer_error_types.get("connect_error", 0),
            "answer_http_error_count": answer_error_types.get("http_error", 0),
            "answer_content_policy_count": answer_error_types.get("content_policy", 0),
            "total_error_count": total_errors,
            "total_requests": total_requests,
            "error_rate": round(total_errors / total_requests, 4) if total_requests > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Task Results 分类辅助方法
    # ------------------------------------------------------------------

    def _classify_task_results(
        self,
        cr: ConversationResult,
        task_results: list[dict[str, Any]],
    ) -> None:
        """将接口返回的 task_results 按 task_type 分类存入 ConversationResult。

        接口返回的 task_results 协议格式（每个元素）：
        {
            "task_type": "ingest" | "retrieve" | "consolidate",
            "input_params": { ... },
            "result": { ... }  // 合并后的统计字典（业务统计 + 通用统计）
        }

        完整保留原始 item（包含 task_type、input_params、result），方便排查问题。

        Args:
            cr: 当前 conversation 的结果对象。
            task_results: 接口返回的 task_results 列表。
        """
        for item in task_results:
            task_type = item.get("task_type", "")
            if task_type == "ingest":
                cr.ingest_stats.append(item)
            elif task_type == "retrieve":
                cr.retrieve_stats.append(item)
            elif task_type == "consolidate":
                cr.consolidate_stats.append(item)
            else:
                logger.warning(
                    "Unknown task_type in task_results: %s", task_type,
                )

    @staticmethod
    def _extract_retrieved_context(task_results: list[dict[str, Any]]) -> str:
        """从 task_results 中提取 retrieve 任务的 retrieved_context。

        Args:
            task_results: 接口返回的 task_results 列表。

        Returns:
            检索到的记忆上下文字符串，若无则返回空字符串。
        """
        for item in task_results:
            if item.get("task_type") == "retrieve":
                return item.get("result", {}).get("retrieved_context", "")
        return ""

    async def _update_progress(
        self,
        benchmark: BaseBenchmark,
        prediction: str,
        reference: str,
        qa: QAPair,
    ) -> None:
        """每次回答完一个问题后，更新临时通过率并覆盖写入进度文件。

        Args:
            benchmark: 当前 benchmark adapter 实例。
            prediction: 模型生成的答案。
            reference: 参考答案。
            qa: 当前 QAPair 对象。
        """
        # 判断当前题目是否通过
        flags = benchmark.check_correctness([prediction], [reference], [qa])
        is_passed = flags[0] if flags else False

        async with self._progress_lock:
            self._progress_completed += 1
            if is_passed:
                self._progress_passed += 1

            completed = self._progress_completed
            passed = self._progress_passed
            accuracy = passed / completed if completed > 0 else 0.0

            # 覆盖写入进度文件
            progress_data = {
                "eval_run_id": self.config.eval_run_id,
                "completed": completed,
                "passed": passed,
                "accuracy": round(accuracy, 4),
                "accuracy_pct": f"{accuracy * 100:.2f}%",
                "updated_at": datetime.now().isoformat(),
            }

            os.makedirs(os.path.dirname(self._progress_file), exist_ok=True)
            with open(self._progress_file, "w", encoding="utf-8") as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)

            # 同时输出到日志
            logger.info(
                "  [进度] 已完成: %d, 已通过: %d, 临时通过率: %.2f%%",
                completed, passed, accuracy * 100,
            )

    async def _save_conversation_incremental(self, cr: ConversationResult) -> None:
        """将单个 conversation 的评测结果增量追加写入 JSONL 文件。

        每完成一个 conversation 就调用此方法，确保过程数据不会因最终汇总失败而丢失。
        """
        try:
            record = {
                "conv_id": cr.conv_id,
                "elapsed_seconds": round(cr.elapsed_seconds, 2),
                "error": cr.error,
                "num_predictions": len(cr.predictions),
                "predictions": cr.predictions,
                "references": cr.references,
                "mcq_accuracy": cr.mcq_accuracy,
                "qa_pass_flags": cr.qa_pass_flags,
                "ingest_stats": cr.ingest_stats,
                "retrieve_stats": cr.retrieve_stats,
                "consolidate_stats": cr.consolidate_stats,
                "ingest_errors": cr.ingest_errors,
                "answer_errors": cr.answer_errors,
                "answer_elapsed_list": cr.answer_elapsed_list,
                "retrieved_contexts": cr.retrieved_contexts,
                "saved_at": datetime.now().isoformat(),
            }
            async with self._incremental_lock:
                os.makedirs(os.path.dirname(self._incremental_file), exist_ok=True)
                with open(self._incremental_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("增量保存 conversation %s 失败: %s", cr.conv_id, e)

    # ------------------------------------------------------------------
    # Ingest 接口调用
    # ------------------------------------------------------------------

    async def _call_ingest_api(
        self,
        *,
        user_id: str,
        session_id: str,
        messages: list[dict[str, str]],
        message_offset: int = 1,
    ) -> dict[str, Any]:
        """通过 HTTP 接口调用 Gateway 完成摄入（ingest）。

        调用 /llm/v1/chat/completions 接口，model 使用配置的 ingest_model，
        通过请求体 metadata 字段传递 user_id 和 session_id。

        Args:
            user_id: 用户 ID，通过请求体 metadata.user_id 传递。
            session_id: 会话 ID，通过请求体 metadata.session_id 传递。
            messages: 待摄入的消息列表。
            message_offset: 消息编号起始值（全局偏移量）。

        Returns:
            接口响应的 JSON 字典；若请求失败则返回包含 error 信息的字典。
        """
        url = f"{self.config.gateway_base_url.rstrip('/')}/llm/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.gateway_api_key}",
        }
        payload = {
            "model": "memory-initialize",
            "messages": messages,
            "metadata": {
                "user_id": user_id,
                "session_id": session_id,
                "message_offset": message_offset,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.config.api_timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                result = resp.json()
                logger.debug(
                    "  [ingest] user_id=%s session_id=%s status=%d",
                    user_id, session_id, resp.status_code,
                )
                # 从接口返回中提取 task_results
                task_results = result.get("task_results", [])
                return {"response": result, "task_results": task_results}
        except httpx.HTTPStatusError as e:
            logger.error(
                "Ingest API HTTP error: user_id=%s, session_id=%s, status=%d, response=%s",
                user_id, session_id, e.response.status_code, e.response.text[:500],
            )
            return {
                "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}",
                "error_type": "http_error",
            }
        except httpx.TimeoutException as e:
            logger.error(
                "Ingest API timeout: user_id=%s, session_id=%s, type=%s, detail=%s",
                user_id, session_id, type(e).__name__, str(e) or "no detail",
            )
            return {
                "error": f"Timeout ({type(e).__name__}): {str(e) or 'request timed out'}",
                "error_type": "timeout",
            }
        except httpx.ConnectError as e:
            logger.error(
                "Ingest API connection error: user_id=%s, session_id=%s, url=%s, detail=%s",
                user_id, session_id, url, str(e) or "connection refused",
            )
            return {
                "error": f"ConnectError: {str(e) or 'connection refused'}",
                "error_type": "connect_error",
            }
        except Exception as e:
            logger.error(
                "Ingest API request failed: user_id=%s, session_id=%s, type=%s, detail=%s",
                user_id, session_id, type(e).__name__, repr(e),
            )
            return {
                "error": f"{type(e).__name__}: {str(e) or repr(e)}",
                "error_type": "unknown",
            }

    # ------------------------------------------------------------------
    # 单个问题的 Retrieve + Answer（通过接口调用）
    # ------------------------------------------------------------------

    async def _call_answer_api(
        self,
        *,
        question: str,
        session_id: str,
        user_id: str,
        benchmark_name: str = "",
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any] | None, float, int, int, float]:
        """通过 HTTP 接口调用 Gateway 完成 retrieve + 提问 LLM（streaming 模式）。

        使用 streaming 模式调用 Gateway，解析 SSE chunks 获取：
        - 真实 TTFT（首 token 时延）
        - 完整回答内容（拼接所有 delta.content）
        - task_results（从 Gateway 中间件注入的 __gateway_meta__ chunk 中提取）
        - usage（从最后一个包含 usage 的 chunk 中提取）

        Args:
            question: 完整问题文本。
            session_id: 当前 session ID，通过请求体 metadata.session_id 传递。
            user_id: 用户 ID，通过请求体 metadata.user_id 传递。

        Returns:
            元组 (答案, task_results, 错误信息或None, 耗时秒数, 输入token数, 输出token数, TTFT秒数)。
        """
        url = f"{self.config.gateway_base_url.rstrip('/')}/llm/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.gateway_api_key}",
        }
        # 仅对 PersonaMem 评测添加 MCQ 格式约束
        if benchmark_name == "personamem":
            formatted_question = (
                f"# Question\n\n{question}\n\n"
                f"**Important**: End your response with a line in exactly this format: "
                f"**Correct Answer: (X)** — where X is a, b, c, or d."
            )
        else:
            formatted_question = question

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "user", "content": formatted_question},
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
            "metadata": {
                "user_id": user_id,
                "session_id": session_id,
            },
        }

        t0 = time.monotonic()  # 记录 answer API 调用开始时间
        try:
            async with httpx.AsyncClient(timeout=self.config.api_timeout) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()

                    content_parts: list[str] = []
                    task_results: list[dict[str, Any]] = []
                    input_tokens = 0
                    output_tokens = 0
                    ttft = 0.0
                    first_token_received = False

                    async for line in resp.aiter_lines():
                        # SSE 格式：每行以 "data: " 开头
                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:]  # 去掉 "data: " 前缀

                        # 检测 [DONE] 标记
                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            chunk_data = json.loads(data_str)
                        except (json.JSONDecodeError, TypeError):
                            continue

                        # 检测 Gateway 中间件注入的元数据 chunk
                        if chunk_data.get("__gateway_meta__"):
                            task_results = chunk_data.get("task_results", [])
                            continue

                        # 提取 usage（通常在最后一个 chunk 中，stream_options.include_usage=True）
                        usage = chunk_data.get("usage")
                        if usage:
                            input_tokens = usage.get("prompt_tokens", 0) or 0
                            output_tokens = usage.get("completion_tokens", 0) or 0

                        # 提取 delta.content
                        choices = chunk_data.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        delta_content = delta.get("content", "")
                        if delta_content:
                            if not first_token_received:
                                ttft = time.monotonic() - t0
                                first_token_received = True
                            content_parts.append(delta_content)

                    content = "".join(content_parts)
                    elapsed = time.monotonic() - t0
                    logger.info(
                        "  [answer] user_id=%s session_id=%s status=%d content_len=%d "
                        "task_results_count=%d input_tokens=%d output_tokens=%d "
                        "elapsed=%.2fs ttft=%.3fs",
                        user_id, session_id, resp.status_code, len(content),
                        len(task_results), input_tokens, output_tokens, elapsed, ttft,
                    )
                    return content, task_results, None, elapsed, input_tokens, output_tokens, ttft

        except httpx.HTTPStatusError as e:
            elapsed = time.monotonic() - t0
            # streaming response 的 body 必须先 aread() 才能拿 .text，否则会触发
            # "Attempted to access streaming response content, without having called `read()`"
            resp_text = ""
            try:
                await e.response.aread()
                resp_text = e.response.text[:500] if e.response.text else ""
            except Exception:
                resp_text = str(e)
            # 检测是否为内容策略拦截
            error_type = "http_error"
            if "ContentPolicyViolation" in resp_text or "content_policy" in resp_text:
                error_type = "content_policy"
            logger.error(
                "Answer API HTTP error: user_id=%s, session_id=%s, status=%d, elapsed=%.2fs, response=%s",
                user_id, session_id, e.response.status_code, elapsed, resp_text,
            )
            error_info = {
                "type": error_type,
                "detail": f"HTTP {e.response.status_code}: {resp_text[:200]}",
                "user_id": user_id,
                "session_id": session_id,
            }
            return f"[ERROR] HTTP {e.response.status_code}: {resp_text[:200]}", [], error_info, elapsed, 0, 0, 0.0
        except httpx.TimeoutException as e:
            elapsed = time.monotonic() - t0
            logger.error(
                "Answer API timeout: user_id=%s, session_id=%s, type=%s, elapsed=%.2fs, question=%.80s, detail=%s",
                user_id, session_id, type(e).__name__, elapsed, question, str(e) or "no detail",
            )
            error_info = {
                "type": "timeout",
                "detail": f"Timeout ({type(e).__name__}): {str(e) or 'request timed out'}",
                "user_id": user_id,
                "session_id": session_id,
            }
            return f"[ERROR] Timeout ({type(e).__name__}): {str(e) or 'request timed out'}", [], error_info, elapsed, 0, 0, 0.0
        except httpx.ConnectError as e:
            elapsed = time.monotonic() - t0
            logger.error(
                "Answer API connection error: user_id=%s, session_id=%s, url=%s, elapsed=%.2fs, detail=%s",
                user_id, session_id, url, elapsed, str(e) or "connection refused",
            )
            error_info = {
                "type": "connect_error",
                "detail": f"ConnectError: {str(e) or 'connection refused'}",
                "user_id": user_id,
                "session_id": session_id,
            }
            return f"[ERROR] ConnectError: {str(e) or 'connection refused'}", [], error_info, elapsed, 0, 0, 0.0
        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.error(
                "Answer API request failed: user_id=%s, session_id=%s, type=%s, elapsed=%.2fs, question=%.80s, detail=%s",
                user_id, session_id, type(e).__name__, elapsed, question, repr(e),
            )
            error_info = {
                "type": "unknown",
                "detail": f"{type(e).__name__}: {str(e) or repr(e)}",
                "user_id": user_id,
                "session_id": session_id,
            }
            return f"[ERROR] {type(e).__name__}: {str(e) or repr(e)}", [], error_info, elapsed, 0, 0, 0.0


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
        """运行 LLM-as-Judge 评分（与 EvalRunner 保持一致）。"""

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
            logger.error("ContextTaskRunner: judge failed for %s: %s", bench_name, e)
            metrics["judge_accuracy"] = -1.0
            metrics["judge_error"] = 1.0

        return judge_scores


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Context Task 评测 Runner —— 基于 LLM Gateway 接口的端到端评测脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--benchmarks",
        type=str,
        default="personamem",
        help="要评测的 benchmark 名称，多个用逗号分隔（如 personamem,locomo）",
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
        default=20,
        help="同一 benchmark 内并行处理的 conversation 数量",
    )
    parser.add_argument(
        "--benchmark-concurrency",
        type=int,
        default=1,
        help="同时运行的 benchmark 数量",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ep-2bjbej6a",
        help="用于 retrieve + 回答的 LLM 模型名称（tokenhub 上 ep-2bjbej6a 背后即 gpt-5.4）",
    )
    parser.add_argument(
        "--ingest-model",
        type=str,
        default="memory-initialize",
        help="摄入接口使用的模型名称",
    )
    parser.add_argument(
        "--gateway-base-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="LLM Gateway 服务地址",
    )
    parser.add_argument(
        "--gateway-api-key",
        type=str,
        default="sk-my-test-key-123",
        help="Gateway 接口的 API Key",
    )
    parser.add_argument(
        "--eval-run-id",
        type=str,
        default=None,
        help="本次评测的时间标识（默认自动生成当前时间戳）",
    )
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
        "--api-timeout",
        type=float,
        default=3600.0,
        help="调用 Gateway 接口（ingest / answer）的超时时间（秒）",
    )
    parser.add_argument(
        "--ingest-append-questions",
        type=int,
        default=3,
        help="摄入记忆时，从当前 session 的 questions 中取前 n 个拼接到 messages 末尾",
    )
    parser.add_argument(
        "--ingest-batch-size",
        type=int,
        default=10,
        help="每次调用 ingest API 一次发送的最大消息数。"
             "session 内若超过此值，会被切成多个 batch 顺序发送。"
             "建议与 gateway 端 max_messages_per_ingest 对齐（默认 10）",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭控制台的 per-conversation 进度条打印",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )

    return parser.parse_args()


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
        "benchmark_concurrency": args.benchmark_concurrency,
        "conv_concurrency": args.conv_concurrency,
        "model": args.model,
        "gateway_base_url": args.gateway_base_url,
        "gateway_api_key": args.gateway_api_key,
        "ingest_model": args.ingest_model,
        "judge_model": args.judge_model,
        "judge_base_url": args.judge_base_url,
        "judge_api_key": args.judge_api_key,
        "no_judge": args.no_judge,
        "api_timeout": args.api_timeout,
        "ingest_append_questions": args.ingest_append_questions,
        "ingest_batch_size": args.ingest_batch_size,
        "show_progress": not args.no_progress,
    }
    if args.eval_run_id:
        config_kwargs["eval_run_id"] = args.eval_run_id

    config = ContextTaskRunConfig(**config_kwargs)

    # 创建 Runner
    runner = ContextTaskRunner(config)

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

    # 将结果保存为 JSON 文件供 Dashboard 展示
    try:
        _save_results_json(results, config.eval_run_id, config)
    except Exception as e:
        logger.exception("保存评测结果 JSON 失败: %s", e)
        print(f"\n⚠️ 保存评测结果 JSON 失败: {e}")
        print(f"   增量过程数据仍可在 eval/results/eval_{config.eval_run_id}_incremental.jsonl 中查看")

    # 输出结果摘要
    print("\n" + "=" * 70)
    print("评测结果摘要")
    print("=" * 70)
    for result in results:
        try:
            print(f"\nBenchmark: {result.benchmark_name}")
            print(f"  耗时: {result.elapsed_seconds:.1f}s")
            print(f"  Metrics:")
            for k, v in result.metrics.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")
            if result.judge_scores:
                avg_judge = sum(result.judge_scores) / len(result.judge_scores)
                print(f"  Judge 平均分: {avg_judge:.4f}")
            print(f"  Stats:")
            for k, v in result.stats.items():
                print(f"    {k}: {v}")
        except Exception as e:
            logger.exception("输出 benchmark %s 摘要失败: %s", result.benchmark_name, e)
    print("\n" + "=" * 70)
    print(f"\n📁 增量过程数据: eval/results/eval_{config.eval_run_id}_incremental.jsonl")


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
    results: list[ContextTaskBenchmarkResult],
    eval_run_id: str,
    config: ContextTaskRunConfig | None = None,
) -> None:
    """将评测结果保存为 JSON 文件到 eval/results/ 目录。

    文件名格式: eval_{eval_run_id}.json
    """
    results_dir = os.path.join(_EVAL_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    filename = f"eval_{eval_run_id}.json"
    filepath = os.path.join(results_dir, filename)

    data = {
        "eval_run_id": eval_run_id,
        "eval_mode": "memory",
        "created_at": datetime.now().isoformat(),
        "base_model": config.model if config else "unknown",
        "memory_model": config.model if config else "unknown",
        "results": [_serialize_result(r) for r in results],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("评测结果已保存: %s", filepath)
    print(f"\n📁 评测结果已保存: {filepath}")
    print(f"   启动 Dashboard 查看: python -m eval.dashboard.server")


if __name__ == "__main__":
    asyncio.run(async_main())

