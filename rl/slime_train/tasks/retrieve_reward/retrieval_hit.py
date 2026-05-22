"""Retrieval-hit reward: real retrieval + frozen-model QA scoring."""

from __future__ import annotations

import asyncio
import logging

try:  # pragma: no cover - fallback for direct script-style imports
    from .answer_scorer import build_qa_prompt, score_fill, score_mcq
    from .frozen_qa_client import call_frozen_model_async
    from .parser import strip_think_wrapper, try_parse_json
    from .snapshot_cache import get_loaded_env
except ImportError:  # pragma: no cover
    from answer_scorer import build_qa_prompt, score_fill, score_mcq
    from frozen_qa_client import call_frozen_model_async
    from parser import strip_think_wrapper, try_parse_json
    from snapshot_cache import get_loaded_env

logger = logging.getLogger(__name__)


async def compute_r_retrieval_hit(
    response: str,
    ground_truth: dict,
    metadata: dict,
) -> float:
    """Score rollout-produced retrieval context with frozen-model QA.

    Retrieve RL now trains the retrieve task itself. The rollout response must
    carry the final context; reward must not execute retrieval a second time.
    """
    text = strip_think_wrapper(response)
    parsed = try_parse_json(text)
    retrieved_context = _extract_retrieved_context(parsed if isinstance(parsed, dict) else None, text)
    if not retrieved_context.strip():
        return 0.0
    return await score_context_for_ground_truth(retrieved_context, ground_truth)


async def score_context_for_ground_truth(context: str, ground_truth: dict) -> float:
    main_question = ground_truth.get("question", "")
    alt_queries = ground_truth.get("alt_queries", [])

    all_questions = [main_question] + [q for q in alt_queries if q]
    if not all_questions or not all_questions[0]:
        return 0.0

    tasks = []
    for question in all_questions:
        q_ground_truth = {**ground_truth, "question": question}
        tasks.append(score_context_against_gold_async(context, q_ground_truth))

    scores = await asyncio.gather(*tasks, return_exceptions=True)
    valid_scores = [score for score in scores if isinstance(score, (int, float))]
    if not valid_scores:
        return 0.0

    if all(score >= 0.99 for score in valid_scores):
        return 1.0
    return sum(valid_scores) / len(valid_scores)


def _extract_retrieved_context(parsed: dict | None, raw_text: str) -> str:
    if isinstance(parsed, dict):
        for key in ("retrieved_context", "context"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value
        task_result = parsed.get("task_result")
        if isinstance(task_result, dict):
            for key in ("retrieved_context", "final_output", "finish_summary"):
                value = task_result.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            stats = task_result.get("stats")
            if isinstance(stats, dict):
                for key in ("retrieved_context", "query_memory"):
                    value = stats.get(key)
                    if isinstance(value, str) and value.strip():
                        return value
    stripped = raw_text.strip()
    if stripped and not stripped.startswith("{"):
        return stripped
    return ""


async def execute_real_retrieval(queries: list[dict], metadata: dict) -> str:
    """Run generated queries against the real snapshot-backed retrieve stack.

    Retained for legacy debugging only; retrieve reward no longer calls this.
    """
    snapshot_id = metadata.get("snapshot_id", "")
    traj_id = metadata.get("traj_id", "")
    if not snapshot_id:
        return ""

    try:
        loaded = await asyncio.to_thread(get_loaded_env, snapshot_id, traj_id)

        retrieve_task = loaded.env.triad.retrieve if loaded.env.triad else None
        if retrieve_task is None:
            return ""
        return await run_retrieval_with_queries(retrieve_task, queries)

    except Exception as e:
        logger.warning("真实检索执行失败: %s", e)
        return ""


async def run_retrieval_with_queries(retrieve_task, queries: list[dict]) -> str:
    """Run retrieval using the best available method on the task, without LLM."""
    # Path 1: task has retrieve_with_queries (e.g. RetrieveT3Task)
    if hasattr(retrieve_task, "retrieve_with_queries"):
        return await retrieve_task.retrieve_with_queries(queries)

    # Path 2: task has _rrf_fuse_results (e.g. older T2/T3 tasks)
    if hasattr(retrieve_task, "_rrf_fuse_results"):
        fs_results = retrieve_task._search_fs(queries)
        vec_results = await retrieve_task._search_vec(queries)
        graph_results = await retrieve_task._search_graph(queries)
        fused_results = retrieve_task._rrf_fuse_results(fs_results, vec_results, graph_results)
        return retrieve_task._format_results(fused_results, fs_results, vec_results, graph_results)

    # Path 3: RetrieveT2AgentLoopTask — use raw search + _build_candidates_from_raw + _format_final_output
    if hasattr(retrieve_task, "_build_candidates_from_raw") and hasattr(retrieve_task, "_format_final_output"):
        fs_results = retrieve_task._search_fs(queries)
        vec_results = await retrieve_task._search_vec(queries)
        graph_results = await retrieve_task._search_graph(queries)
        candidates = retrieve_task._build_candidates_from_raw(fs_results, vec_results, graph_results)
        return retrieve_task._format_final_output(candidates)

    return ""


async def score_context_against_gold_async(context: str, ground_truth: dict) -> float:
    """Ask the frozen model to answer from retrieved context, then score it."""
    question = ground_truth.get("question", "")
    if not question:
        return 0.0

    answer_type = ground_truth.get("answer_type", "fill")

    if not context or not context.strip():
        context = "(No relevant memories found)"

    qa_prompt = build_qa_prompt(question, context, ground_truth)
    model_answer = await call_frozen_model_async(qa_prompt)
    if not model_answer:
        return 0.0

    if answer_type == "mcq":
        return score_mcq(model_answer, ground_truth)
    return score_fill(model_answer, ground_truth)


# Backward-compatible private aliases.
_execute_real_retrieval = execute_real_retrieval
_run_retrieval_with_queries = run_retrieval_with_queries
_score_context_against_gold_async = score_context_against_gold_async
