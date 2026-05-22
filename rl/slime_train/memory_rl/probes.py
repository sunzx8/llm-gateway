from __future__ import annotations

import asyncio
import os
import re
from typing import Any

try:
    from llm_gateway.rl.slime_train.tasks.retrieve_reward.retrieval_hit import score_context_against_gold_async
except ImportError:  # pragma: no cover - legacy retrieve-only PYTHONPATH
    from tasks.retrieve_reward.retrieval_hit import score_context_against_gold_async


def select_task_probes(
    record: dict[str, Any],
    task: str,
    *,
    fallback_all_when_untyped: bool = False,
) -> list[dict[str, Any]]:
    """Select query probes for one RL task from either new or legacy schemas.

    Preferred schema is ``probes_by_task.{ingest,retrieve,consolidate}``, as in
    ``rl_data_test_2``. Legacy datasets may only have ``probes`` without
    ``task_target``; those are returned only when ``fallback_all_when_untyped``
    is enabled.
    """
    by_task = record.get("probes_by_task")
    if isinstance(by_task, dict):
        probes = by_task.get(task)
        if isinstance(probes, list):
            return [p for p in probes if isinstance(p, dict)]

    raw_probes = record.get("probes", [])
    if not isinstance(raw_probes, list):
        return []
    typed = [p for p in raw_probes if isinstance(p, dict) and p.get("task_target") == task]
    if typed:
        return typed
    if fallback_all_when_untyped and not any(isinstance(p, dict) and p.get("task_target") for p in raw_probes):
        return [p for p in raw_probes if isinstance(p, dict)]
    return []


def normalize_ground_truth(probe: dict[str, Any]) -> dict[str, Any]:
    """Map raw probe schema to retrieve reward ground-truth schema."""
    question_type = probe.get("question_type", "fill_in_the_blank")
    answer_type = "mcq" if question_type == "multiple_choice" else "fill"
    raw_gt = probe.get("ground_truth", "")
    if answer_type == "mcq":
        answer = raw_gt if isinstance(raw_gt, str) else str(raw_gt)
    elif isinstance(raw_gt, list):
        answer = raw_gt[0] if raw_gt else ""
    else:
        answer = str(raw_gt)

    ground_truth: dict[str, Any] = {
        "question": probe.get("probe_query", ""),
        "answer": answer,
        "answer_type": answer_type,
        "alt_queries": probe.get("alt_queries", []),
        "source_evidence": probe.get("source_evidence", ""),
        "probe_type": probe.get("probe_type", ""),
    }
    if answer_type == "fill" and isinstance(raw_gt, list):
        ground_truth["acceptable_answers"] = raw_gt
    if answer_type == "mcq":
        options = probe.get("options", {})
        if isinstance(options, dict):
            ground_truth["options"] = [f"{k}. {v}" for k, v in sorted(options.items())]
        else:
            ground_truth["options"] = options
        ground_truth["correct_option"] = str(answer).upper().strip()
    return ground_truth


def build_probe_queries(probe: dict[str, Any]) -> list[dict[str, str]]:
    """Build deterministic retrieve queries from one probe."""
    texts = [probe.get("probe_query", "")]
    texts.extend(q for q in probe.get("alt_queries", []) if q)
    seen: set[str] = set()
    queries: list[dict[str, str]] = []
    for text in texts:
        text = str(text).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        queries.append({"type": "semantic", "text": text})
        keyword = _keyword_query(text)
        if keyword and keyword != text:
            queries.append({"type": "keyword", "text": keyword})
    return queries[:8]


async def evaluate_probe_set(env, probes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retrieve with the bound query task, then QA-score every probe.

    This intentionally runs ``env.step_query`` instead of feeding deterministic
    rewritten queries into lower-level retrieval: ingest/consolidate rewards must
    evaluate the actual query task implementation selected by ``task_version``
    (atomic rewrite+return, t2_agent_loop agentic retrieve, etc.).
    """
    if not probes:
        return []

    async def _score_one(index: int, probe: dict[str, Any]) -> dict[str, Any]:
        question = str(probe.get("probe_query", "")).strip()
        context = await _retrieve_probe_context(env, question)
        if not context.strip():
            context = "(No relevant memories found)"
        score = await score_context_against_gold_async(context, normalize_ground_truth(probe))
        return {
            "index": index,
            "probe_id": probe.get("id", ""),
            "question": probe.get("probe_query", ""),
            "score": float(score),
            "correct": float(score) >= 0.5,
            "context": context,
        }

    results = await _gather_limited(
        [_score_one(i, probe) for i, probe in enumerate(probes)],
        # Query tasks are stateful (shared task/result stats inside one env), so
        # default to sequential probe retrieval. Override only after confirming
        # the selected task_version is concurrency-safe.
        limit=int(os.environ.get("PROBE_TASK_MAX_CONCURRENCY", "8")),
    )
    return [r for r in results if isinstance(r, dict)]


async def _retrieve_probe_context(env, question: str) -> str:
    if not question:
        return ""
    step = await env.step_query(query=question, session_id="rl_probe_eval")
    stats = getattr(step.task_result, "stats", None)
    if isinstance(stats, dict):
        for key in ("retrieved_context", "query_memory"):
            value = stats.get(key)
            if value:
                return str(value)
    return str(step.task_result.final_output or step.task_result.finish_summary or "")


async def score_probe_set(
    env,
    probes: list[dict[str, Any]],
    *,
    return_contexts: bool = False,
) -> tuple[float, list[str]] | float:
    """Score current env by query probes and frozen-model QA judging."""
    evals = await evaluate_probe_set(env, probes)
    if not evals:
        return (0.0, []) if return_contexts else 0.0
    avg = average_probe_score(evals)
    contexts = [str(e.get("context", "")) for e in evals]
    return (avg, contexts) if return_contexts else avg


def average_probe_score(evals: list[dict[str, Any]]) -> float:
    if not evals:
        return 0.0
    return sum(float(e.get("score", 0.0)) for e in evals) / len(evals)


def probe_accuracy(evals: list[dict[str, Any]]) -> float:
    if not evals:
        return 0.0
    return sum(1.0 for e in evals if e.get("correct")) / len(evals)


def positive_probe_score_delta(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> float:
    """Average positive per-probe score gain, clipped to [0, 1]."""
    if not after:
        return 0.0
    total = 0.0
    for i, after_eval in enumerate(after):
        before_score = float(before[i].get("score", 0.0)) if i < len(before) else 0.0
        total += max(0.0, float(after_eval.get("score", 0.0)) - before_score)
    return min(1.0, total / len(after))


def positive_probe_accuracy_delta(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> float:
    """Positive delta in probe correctness rate after evolution."""
    return max(0.0, probe_accuracy(after) - probe_accuracy(before))


def context_diff_score(
    before_contexts: list[str],
    after_contexts: list[str],
    *,
    before_scores: list[float] | None = None,
    after_scores: list[float] | None = None,
) -> float:
    """Reward retrieval context changes, gated by non-decreasing probe scores."""
    if not after_contexts:
        return 0.0
    scores: list[float] = []
    for i, after in enumerate(after_contexts):
        before = before_contexts[i] if i < len(before_contexts) else ""
        before_score = before_scores[i] if before_scores and i < len(before_scores) else 0.0
        after_score = after_scores[i] if after_scores and i < len(after_scores) else before_score
        if after_score < before_score:
            scores.append(0.0)
            continue
        if not after.strip() or after == "(No relevant memories found)":
            scores.append(0.0)
            continue
        if before == after:
            scores.append(0.0)
            continue
        before_tokens = set(_tokens(before))
        after_tokens = set(_tokens(after))
        if not before_tokens and after_tokens:
            scores.append(1.0)
        else:
            union = before_tokens | after_tokens
            inter = before_tokens & after_tokens
            raw_diff = 1.0 - (len(inter) / len(union) if union else 1.0)
            scores.append(raw_diff if after_score > before_score else 0.25 * raw_diff)
    return sum(scores) / len(scores)


def operation_diff_score(calls: list[dict[str, Any]], expected_ops: list[dict[str, Any]] | None = None) -> float:
    """Legacy non-primary op overlap score retained for diagnostics/tests."""
    meaningful = [c for c in calls if c.get("tool") and c.get("tool") != "finish"]
    if not meaningful:
        return 0.0
    base = min(1.0, len(meaningful) / 3.0)
    if not expected_ops:
        return base

    generated_sig = {_call_signature(c) for c in meaningful}
    expected_sig = {_call_signature({"tool": op.get("tool"), "arguments": op.get("arguments", {})}) for op in expected_ops}
    generated_sig.discard("")
    expected_sig.discard("")
    if not expected_sig:
        return base
    overlap = len(generated_sig & expected_sig) / len(expected_sig)
    return 0.4 * base + 0.6 * overlap


async def _gather_limited(coros, *, limit: int):
    if limit <= 0:
        return await asyncio.gather(*coros, return_exceptions=True)
    sem = asyncio.Semaphore(limit)

    async def _run(coro):
        async with sem:
            return await coro

    return await asyncio.gather(*[_run(coro) for coro in coros], return_exceptions=True)


def _keyword_query(text: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z0-9_./:-]+", text) if len(w) > 2]
    return " ".join(words[:12])


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_./:-]+", text.lower())


def _call_signature(call: dict[str, Any]) -> str:
    tool = call.get("tool", "")
    args = call.get("arguments", {}) if isinstance(call.get("arguments"), dict) else {}
    key = args.get("path") or args.get("collection") or args.get("node_id") or args.get("source") or ""
    content = args.get("content") or args.get("text") or args.get("relation") or ""
    toks = " ".join(_tokens(str(content))[:10])
    return f"{tool}:{key}:{toks}"
