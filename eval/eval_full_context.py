"""Full Context 评测脚本（Baseline 对比）。

直接将完整的对话上下文（所有 session 的消息）拼接后传给 LLM 回答问题，
不经过 memory 的 ingest/retrieve 流程。用于和 memory 方案进行对比测试。

评测指标与 eval_script.py 完全对齐：
- benchmark.evaluate() 计算的 metrics（MCQ accuracy 等）
- LLM-as-Judge 评分（yes/no、rubrics）
- 耗时统计（answer API 端到端耗时）

结果格式与 eval_script.py 兼容，可在同一 Dashboard 中对比查看。

使用示例
~~~~~~~~
::

    python -m eval.eval_full_context \\
        --benchmarks personamem \\
        --model gpt-5.4 \\
        --conv-concurrency 10 \\
        --subset-size 5
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import sys
import time
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
# 配置
# ---------------------------------------------------------------------------


@dataclass
class FullContextRunConfig:
    """Full Context 评测运行配置。"""

    benchmarks: list[str] = field(default_factory=lambda: ["personamem"])
    """要评测的 benchmark 名称列表。"""

    subset_size: int | None = None
    """每个 benchmark 取前 N 条 conversation（None = 全量）。"""

    conv_concurrency: int = 10
    """同一 benchmark 内并行处理的 conversation 数量。"""

    model: str = "ep-2bjbej6a"
    """用于回答问题的 LLM 模型名称。"""

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

    eval_run_id: str = field(default_factory=lambda: f"fc_{datetime.now().strftime('%Y%m%d%H%M%S')}")
    """本次评测的时间标识，前缀 fc_ 表示 full context。"""

    # Judge 配置
    judge_model: str = "gpt-5.5"
    judge_base_url: str = "https://vdbteam.cognitiveservices.azure.com/openai/v1/"
    judge_api_key: str = "d0924051350c460295107e33772e1e49"
    no_judge: bool = False

    max_context_tokens: int | None = None
    """最大上下文 token 数限制（None = 不限制，传入全部）。"""


# ---------------------------------------------------------------------------
# 结果数据结构（与 eval_script.py 的 ConversationResult 对齐）
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
    # Answer API 耗时统计
    answer_elapsed_list: list[float] = field(default_factory=list)
    """每次 LLM 调用的端到端耗时（秒），与 predictions 一一对应"""
    # TTFT（首 token 时延）统计
    ttft_list: list[float] = field(default_factory=list)
    """每次 LLM 调用的首 token 时延（秒），与 predictions 一一对应"""
    # LLM 使用统计
    input_tokens_list: list[int] = field(default_factory=list)
    """每次 LLM 调用的输入 token 数"""
    output_tokens_list: list[int] = field(default_factory=list)
    """每次 LLM 调用的输出 token 数"""
    # Conversation 级别累计 token
    conv_total_tokens: int = 0
    """该 conversation 内所有 QA 的累计 token（input + output）"""
    # 错误统计
    answer_errors: list[dict[str, Any]] = field(default_factory=list)
    """LLM 调用失败的错误信息列表"""
    # QA 正确率统计
    qa_pass_flags: list[bool] = field(default_factory=list)
    """每个 QA 是否回答正确的布尔列表"""
    mcq_accuracy: float = 0.0
    """该 conversation 的 MCQ 准确率（0.0 ~ 1.0）"""


@dataclass
class FullContextBenchmarkResult:
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
# FullContextRunner
# ---------------------------------------------------------------------------


class FullContextRunner:
    """Full Context 评测 Runner。

    对每个 conversation：
    1. 收集所有 session 的消息作为完整上下文
    2. 对每个 QAPair，将上下文 + 问题一起发送给 LLM
    3. 收集 predictions / references，计算 metrics

    与 ContextTaskRunner 的区别：
    - 不调用 ingest/retrieve/consolidate
    - 直接将原始对话历史作为 context 传给 LLM
    - 用于评估 "有完整上下文时 LLM 的回答能力" 作为 baseline
    """

    def __init__(self, config: FullContextRunConfig):
        self.config = config
        self._benchmarks: dict[str, BaseBenchmark] = {}
        self._results: list[FullContextBenchmarkResult] = []
        self._llm: LLMInterface | None = None

    def register_benchmark(self, benchmark: BaseBenchmark) -> None:
        """注册一个 benchmark adapter。"""
        self._benchmarks[benchmark.name] = benchmark

    def _get_llm(self) -> LLMInterface:
        """获取或创建 LLM 接口实例。"""
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

    async def run(self) -> list[FullContextBenchmarkResult]:
        """运行所有已注册 benchmark 的评测。"""
        self._results = []
        run_start = time.monotonic()

        logger.info(
            "=== FullContextRunner: %d benchmarks, conv_concurrency=%d ===",
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
            "=== FullContextRunner done: %d results, total %.1fs ===",
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
    ) -> FullContextBenchmarkResult:
        """评测单个 benchmark。"""
        start_time = time.monotonic()
        bench_name = benchmark.name

        logger.info("=" * 60)
        logger.info("FullContextRunner: evaluating benchmark=%s", bench_name)

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
            "FullContextRunner: benchmark=%s metrics=%s",
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

        return FullContextBenchmarkResult(
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
                "eval_mode": "full_context",
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
        """并发处理多个 conversations。"""
        sem = asyncio.Semaphore(self.config.conv_concurrency)
        total = len(conversations)
        completed_count = 0
        lock = asyncio.Lock()

        async def _run_one(conv: Conversation) -> ConversationResult:
            nonlocal completed_count
            async with sem:
                result = await self._evaluate_conversation(benchmark, conv)
            async with lock:
                completed_count += 1
                logger.info(
                    "  进度: %d/%d (%.1f%%) 已完成, 当前完成: conv=%s",
                    completed_count, total, completed_count / total * 100, conv.conv_id,
                )
            return result

        results = await asyncio.gather(
            *[_run_one(conv) for conv in conversations],
            return_exceptions=False,
        )
        return list(results)

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
        1. 收集所有 session 的消息作为完整上下文
        2. 对每个 QAPair，构建 [context + question] 发送给 LLM
        3. 返回 ConversationResult
        """
        start_time = time.monotonic()
        cr = ConversationResult(conv_id=conv.conv_id)

        try:
            # 检测是否为 interleaved 模式
            interleaved = any(
                hasattr(s, "questions") and s.questions
                for s in conv.sessions
            )

            if interleaved:
                # Interleaved 模式：每个 session 有附属问题
                # 问题只能看到当前 session 及之前的上下文
                accumulated_messages: list[dict[str, str]] = []
                for session in conv.sessions:
                    # 累积当前 session 的消息
                    for m in session.messages:
                        accumulated_messages.append({"role": m.role, "content": m.content})

                    # 回答该 session 的附属问题
                    sess_questions: list[QAPair] = getattr(session, "questions", []) or []
                    for qa in sess_questions:
                        prediction, elapsed, input_tokens, output_tokens, ttft, error = (
                            await self._answer_with_context(
                                context_messages=accumulated_messages,
                                question=qa.question,
                            )
                        )
                        if error:
                            cr.answer_errors.append(error)
                        cr.predictions.append(prediction)
                        cr.references.append(qa.reference_answer)
                        cr.qa_pairs.append(qa)
                        cr.answer_elapsed_list.append(elapsed)
                        cr.ttft_list.append(ttft)
                        cr.input_tokens_list.append(input_tokens)
                        cr.output_tokens_list.append(output_tokens)
            else:
                # 标准模式：收集所有 session 消息作为完整上下文
                all_messages: list[dict[str, str]] = []
                for session in conv.sessions:
                    for m in session.messages:
                        all_messages.append({"role": m.role, "content": m.content})

                # 回答所有问题
                questions = benchmark.get_questions(conv)
                for qa in questions:
                    prediction, elapsed, input_tokens, output_tokens, ttft, error = (
                        await self._answer_with_context(
                            context_messages=all_messages,
                            question=qa.question,
                        )
                    )
                    if error:
                        cr.answer_errors.append(error)
                    cr.predictions.append(prediction)
                    cr.references.append(qa.reference_answer)
                    cr.qa_pairs.append(qa)
                    cr.answer_elapsed_list.append(elapsed)
                    cr.ttft_list.append(ttft)
                    cr.input_tokens_list.append(input_tokens)
                    cr.output_tokens_list.append(output_tokens)

        except Exception as e:
            logger.exception(
                "FullContextRunner: conversation=%s failed: %s",
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
    # 核心：带完整上下文的 LLM 调用（streaming 模式，获取 TTFT）
    # ------------------------------------------------------------------

    async def _answer_with_context(
        self,
        *,
        context_messages: list[dict[str, str]],
        question: str,
    ) -> tuple[str, float, int, int, float, dict[str, Any] | None]:
        """将完整上下文 + 问题发送给 LLM 获取答案。

        与 PersonaMem 官方评测协议对齐：
        - Context messages 直接作为 chat messages 传递（保留原始角色）
        - 问题 + instructions + options 作为最后一条 user message 追加
        - 即: messages = context_messages + [{"role": "user", "content": question + instructions}]
        - system 消息转换为 user 消息以兼容各种模型

        Args:
            context_messages: 完整的对话上下文消息列表。
            question: 要回答的问题（已包含 MCQ options）。

        Returns:
            元组 (答案, 耗时秒数, 输入token数, 输出token数, TTFT秒数, 错误信息或None)。
        """
        llm = self._get_llm()

        # PersonaMem 官方 instructions
        instructions = (
            "Find the most appropriate model response and give your final answer "
            "(a), (b), (c), or (d) after the special token <final_answer>."
        )

        # 构建消息列表：context history + 最后一条问题消息
        messages = list(context_messages) + [
            {"role": "user", "content": f"{question}\n\n{instructions}"},
        ]

        # 将 system 消息转换为 user 消息以兼容各种模型
        # （与官方 PersonaMem 的 convert_role_system_to_user 对齐）
        messages = self._convert_system_to_user(messages)

        t0 = time.monotonic()
        try:
            # 使用 streaming 模式获取真实 TTFT（首 token 时延）
            # context messages 中已包含 persona 信息（原 system 消息已转为 user 消息）
            response, ttft = await llm.generate_with_ttft(
                system="",
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            elapsed = time.monotonic() - t0
            content = response.content or ""
            logger.info(
                "  [full_context_answer] model=%s input_tokens=%d output_tokens=%d "
                "content_len=%d elapsed=%.2fs ttft=%.3fs n_messages=%d",
                self.config.model, response.input_tokens, response.output_tokens,
                len(content), elapsed, ttft, len(messages),
            )
            return content, elapsed, response.input_tokens, response.output_tokens, ttft, None

        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.error(
                "Full context answer failed: model=%s, error=%s, elapsed=%.2fs",
                self.config.model, e, elapsed,
            )
            error_info = {
                "type": type(e).__name__,
                "detail": str(e)[:500],
            }
            return f"[ERROR] {type(e).__name__}: {str(e)}", elapsed, 0, 0, 0.0, error_info

    @staticmethod
    def _convert_system_to_user(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """将 system 角色的消息转换为 user 角色。

        与官方 PersonaMem 的 convert_role_system_to_user 对齐：
        某些模型（如 o-series）不支持 system 角色，需要转换为 user 消息。
        """
        converted = []
        for msg in messages:
            if msg.get("role") == "system":
                converted.append({"role": "user", "content": msg.get("content", "")})
            else:
                converted.append(msg)
        return converted

    # ------------------------------------------------------------------
    # 统计汇总
    # ------------------------------------------------------------------

    @staticmethod
    def _percentile(sorted_values: list[float], p: float) -> float:
        """计算第 p 百分位值（0 < p < 1）。"""
        if not sorted_values:
            return 0.0
        idx = int(len(sorted_values) * p)
        idx = min(idx, len(sorted_values) - 1)
        return sorted_values[idx]

    def _aggregate_stats(self, conv_results: list[ConversationResult]) -> dict[str, Any]:
        """汇总所有 conversation 的统计信息。"""
        all_elapsed: list[float] = []
        all_ttft: list[float] = []
        all_input_tokens: list[int] = []
        all_output_tokens: list[int] = []
        all_conv_total_tokens: list[int] = []
        total_errors = 0

        for cr in conv_results:
            all_elapsed.extend(cr.answer_elapsed_list)
            all_ttft.extend(cr.ttft_list)
            all_input_tokens.extend(cr.input_tokens_list)
            all_output_tokens.extend(cr.output_tokens_list)
            if cr.conv_total_tokens > 0:
                all_conv_total_tokens.append(cr.conv_total_tokens)
            total_errors += len(cr.answer_errors)

        count = len(all_elapsed)
        total_elapsed = sum(all_elapsed)
        max_elapsed = max(all_elapsed) if all_elapsed else 0.0
        avg_elapsed = (total_elapsed / count) if count > 0 else 0.0

        # 端到端时延百分位
        sorted_elapsed = sorted(all_elapsed) if all_elapsed else []
        p50_elapsed = self._percentile(sorted_elapsed, 0.5)
        p90_elapsed = self._percentile(sorted_elapsed, 0.9)
        p99_elapsed = self._percentile(sorted_elapsed, 0.99)

        # TTFT 统计
        valid_ttft = [t for t in all_ttft if t > 0]
        sorted_ttft = sorted(valid_ttft) if valid_ttft else []
        ttft_count = len(valid_ttft)
        ttft_avg = (sum(valid_ttft) / ttft_count) if ttft_count > 0 else 0.0
        ttft_p50 = self._percentile(sorted_ttft, 0.5)
        ttft_p90 = self._percentile(sorted_ttft, 0.9)
        ttft_p99 = self._percentile(sorted_ttft, 0.99)

        # 单次对话 in token 统计
        total_input_tokens = sum(all_input_tokens)
        total_output_tokens = sum(all_output_tokens)
        max_input_tokens = max(all_input_tokens) if all_input_tokens else 0
        avg_input_tokens = (total_input_tokens / count) if count > 0 else 0
        sorted_input_tokens = sorted(all_input_tokens) if all_input_tokens else []
        p50_input_tokens = self._percentile([float(x) for x in sorted_input_tokens], 0.5)

        # Conversation 累计 token 统计
        conv_tokens_avg = (
            round(sum(all_conv_total_tokens) / len(all_conv_total_tokens))
            if all_conv_total_tokens else 0
        )
        conv_tokens_max = max(all_conv_total_tokens) if all_conv_total_tokens else 0
        sorted_conv_tokens = sorted(all_conv_total_tokens) if all_conv_total_tokens else []
        conv_tokens_p50 = self._percentile([float(x) for x in sorted_conv_tokens], 0.5)

        return {
            "answer_api_count": count,
            "answer_api_total_s": round(total_elapsed, 2),
            "answer_api_max_s": round(max_elapsed, 2),
            "answer_api_avg_s": round(avg_elapsed, 2),
            "answer_api_p50_s": round(p50_elapsed, 2),
            "answer_api_p90_s": round(p90_elapsed, 2),
            "answer_api_p99_s": round(p99_elapsed, 2),
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
            "max_input_tokens": max_input_tokens,
            "p50_input_tokens": round(p50_input_tokens),
            # Conversation 累计 token 统计
            "conv_total_tokens_avg": conv_tokens_avg,
            "conv_total_tokens_max": conv_tokens_max,
            "conv_total_tokens_p50": round(conv_tokens_p50),
            # 错误统计
            "total_errors": total_errors,
            "error_rate": round(total_errors / count, 4) if count > 0 else 0.0,
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
            logger.error("FullContextRunner: judge failed for %s: %s", bench_name, e)
            metrics["judge_accuracy"] = -1.0
            metrics["judge_error"] = 1.0

        return judge_scores



# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Full Context 评测脚本 —— 直接将完整上下文传给 LLM 回答问题（Baseline 对比）",
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
        default=10,
        help="同一 benchmark 内并行处理的 conversation 数量",
    )
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
    parser.add_argument(
        "--eval-run-id",
        type=str,
        default=None,
        help="本次评测的时间标识（默认自动生成 fc_ 前缀时间戳）",
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
    results: list[FullContextBenchmarkResult],
    eval_run_id: str,
    config: FullContextRunConfig | None = None,
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
        "eval_mode": "full_context",
        "created_at": datetime.now().isoformat(),
        "base_model": config.model if config else "unknown",
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
        "judge_model": args.judge_model,
        "judge_base_url": args.judge_base_url,
        "judge_api_key": args.judge_api_key,
        "no_judge": args.no_judge,
    }
    if args.base_url:
        config_kwargs["base_url"] = args.base_url
    if args.api_key:
        config_kwargs["api_key"] = args.api_key
    if args.eval_run_id:
        config_kwargs["eval_run_id"] = args.eval_run_id
    config = FullContextRunConfig(**config_kwargs)

    # 创建 Runner
    runner = FullContextRunner(config)

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
    print("Full Context 评测结果摘要（Baseline）")
    print("=" * 70)
    for result in results:
        print(f"\nBenchmark: {result.benchmark_name}")
        print(f"  评测模式: Full Context (直接传入完整上下文)")
        print(f"  模型: {config.model}")
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
