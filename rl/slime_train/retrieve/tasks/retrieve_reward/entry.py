"""Slime reward entrypoint and total reward composition."""

from __future__ import annotations

import logging

try:  # pragma: no cover - fallback for direct script-style imports
    from .metrics import log_debug_response, log_rollout_record, log_single_metric
    from .parser import strip_think_wrapper, try_parse_json
    from .retrieval_hit import compute_r_retrieval_hit
except ImportError:  # pragma: no cover
    from metrics import log_debug_response, log_rollout_record, log_single_metric
    from parser import strip_think_wrapper, try_parse_json
    from retrieval_hit import compute_r_retrieval_hit

logger = logging.getLogger(__name__)


async def reward_func(args, sample, **kwargs) -> float:
    """Slime custom reward-model entrypoint."""
    response = sample.response or ""
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}

    log_debug_response(response, metadata)

    try:
        reward, sub_rewards = await compute_single_reward_async(response, metadata)
    except Exception as e:
        logger.warning("reward computation failed: %s", e)
        reward = 0.0
        sub_rewards = {"r_retrieval_hit": 0.0, "r_context_payload": 0.0}

    log_single_metric(reward, sub_rewards, response)
    log_rollout_record(sample, reward, sub_rewards)

    return reward


async def compute_single_reward_async(response: str, metadata: dict) -> tuple[float, dict]:
    """Compute total reward and sub-reward breakdown for one rollout sample."""
    if not response or not response.strip():
        return 0.0, {"r_retrieval_hit": 0.0, "r_context_payload": 0.0}

    ground_truth = metadata.get("ground_truth", {})

    r_context_payload = compute_r_context_payload(response)
    r_retrieval_hit = await compute_r_retrieval_hit(response, ground_truth, metadata)

    r_total = 0.90 * r_retrieval_hit + 0.10 * r_context_payload

    think_count = response.count("</think>")
    if think_count > 1:
        think_penalty = max(0.3, 1.0 - 0.2 * (think_count - 1))
        r_total *= think_penalty

    sub_rewards = {
        "r_retrieval_hit": r_retrieval_hit,
        "r_context_payload": r_context_payload,
    }
    return r_total, sub_rewards


def compute_r_context_payload(response: str) -> float:
    text = strip_think_wrapper(response)
    parsed = try_parse_json(text)
    if not isinstance(parsed, dict):
        return 0.25 if text.strip() else 0.0
    context = parsed.get("retrieved_context")
    task_result = parsed.get("task_result") if isinstance(parsed.get("task_result"), dict) else {}
    if not context and isinstance(task_result, dict):
        context = task_result.get("retrieved_context") or task_result.get("final_output")
    trace = parsed.get("agentic_trace")
    has_context = isinstance(context, str) and bool(context.strip())
    has_trace = isinstance(trace, list)
    return 1.0 if has_context and has_trace else 0.7 if has_context else 0.0


# Backward-compatible private alias.
_compute_single_reward_async = compute_single_reward_async
