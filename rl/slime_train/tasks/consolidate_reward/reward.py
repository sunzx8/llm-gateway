"""Slime reward for T3 consolidate/evolve RLVR.

The primary signal is hidden query-probe correctness before vs. after evolution.
A small retrieval-context diff reward is kept only when probe scores do not
regress, so superficial rewrites are not rewarded over correctness.
"""

from __future__ import annotations

import asyncio
import logging
import os

from llm_gateway.rl.slime_train.memory_rl.paths import ensure_workspace_paths

ensure_workspace_paths(__file__)

from llm_gateway.rl.rl_env.snapshot_session import SnapshotSession
from llm_gateway.rl.slime_train.memory_rl.probes import (
    average_probe_score,
    context_diff_score,
    evaluate_probe_set,
    positive_probe_accuracy_delta,
    positive_probe_score_delta,
    probe_accuracy,
)
from llm_gateway.rl.slime_train.memory_rl.response_parser import format_reward, parse_tool_calls_response

logger = logging.getLogger(__name__)

_snapshot_session: SnapshotSession | None = None


def _get_snapshot_session() -> SnapshotSession:
    import os

    global _snapshot_session
    data_root = os.environ.get("CONSOLIDATE_SNAPSHOT_DATA_ROOT") or os.environ.get("SNAPSHOT_DATA_ROOT")
    if not data_root:
        raise RuntimeError("CONSOLIDATE_SNAPSHOT_DATA_ROOT or SNAPSHOT_DATA_ROOT is required")
    task_version = os.environ.get("MEMORY_RL_TASK_VERSION", "atomic_code_t2")
    if (
        _snapshot_session is None
        or _snapshot_session.data_root != data_root
        or _snapshot_session.task_version != task_version
    ):
        _snapshot_session = SnapshotSession(data_root=data_root, enable_git=False, task_version=task_version)
    return _snapshot_session


async def reward_func(args, sample, **kwargs) -> float:
    import os, json as _json, time as _time

    response = sample.response or ""
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    try:
        reward, sub = await compute_single_reward_async(response, metadata)
    except Exception as exc:
        logger.warning("consolidate reward failed: %s", exc)
        reward, sub = 0.0, {"r_format": 0.0, "error": str(exc)}

    # 持久化所有 rollout 记录
    log_path = os.environ.get("REWARD_METRICS_LOG", "")
    if log_path:
        log_dir = os.path.dirname(log_path)
        rollout_file = os.path.join(log_dir, "rollout_records.jsonl")
        try:
            record = {
                "timestamp": _time.time(),
                "reward": reward,
                "sub_rewards": sub,
                "response": response,
                "response_len": len(response),
                "metadata": {
                    "traj_id": metadata.get("traj_id", ""),
                    "snapshot_id": metadata.get("snapshot_id", ""),
                    "session_id": metadata.get("session_id", ""),
                    "probes_count": len(metadata.get("probes", [])),
                },
            }
            with open(rollout_file, "a") as f:
                f.write(_json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # SwanLab 监控读取的精简 metrics 文件
        try:
            metrics_record = {
                "timestamp": _time.time(),
                "r_total": reward,
                "r_format": sub.get("r_format", 0),
                "r_after_score": sub.get("r_after_score", 0),
                "r_acc_delta": sub.get("r_acc_delta", 0),
                "r_score_delta": sub.get("r_score_delta", 0),
                "r_context_diff": sub.get("r_context_diff", 0),
                "response_len": len(response),
                "is_truncated": 0,
            }
            with open(log_path, "a") as f:
                f.write(_json.dumps(metrics_record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    return reward


async def compute_single_reward_async(response: str, metadata: dict) -> tuple[float, dict[str, float]]:
    calls, parsed = parse_tool_calls_response(response)
    task_loop = os.environ.get("MEMORY_RL_APPLY_MODE", "tool_calls").strip().lower() == "task_loop"
    replay_task_loop_response = task_loop and isinstance(parsed, dict) and "agentic_trace" in parsed and "tool_calls" in parsed
    r_format = 1.0 if task_loop and not calls else format_reward(response)
    probes = metadata.get("probes", [])

    r_before_score = 0.0
    r_after_score = 0.0
    r_before_acc = 0.0
    r_after_acc = 0.0
    r_acc_delta = 0.0
    r_score_delta = 0.0
    r_context_diff = 0.0

    if (calls or task_loop) and metadata.get("snapshot_id") and probes:
        session = _get_snapshot_session()
        loaded = await asyncio.to_thread(
            session.load,
            traj_id=metadata.get("traj_id") or None,
            snapshot_id=metadata["snapshot_id"],
        )
        try:
            before_evals = await evaluate_probe_set(loaded.env, probes)
            r_before_score = average_probe_score(before_evals)
            r_before_acc = probe_accuracy(before_evals)

            await loaded.env.apply_consolidate_tool_calls(
                calls,
                extras={
                    "session_id": metadata.get("session_id", ""),
                    "replay_task_loop_response": replay_task_loop_response,
                },
            )

            after_evals = await evaluate_probe_set(loaded.env, probes)
            r_after_score = average_probe_score(after_evals)
            r_after_acc = probe_accuracy(after_evals)
            r_acc_delta = positive_probe_accuracy_delta(before_evals, after_evals)
            r_score_delta = positive_probe_score_delta(before_evals, after_evals)
            r_context_diff = context_diff_score(
                [str(e.get("context", "")) for e in before_evals],
                [str(e.get("context", "")) for e in after_evals],
                before_scores=[float(e.get("score", 0.0)) for e in before_evals],
                after_scores=[float(e.get("score", 0.0)) for e in after_evals],
            )
        finally:
            await asyncio.to_thread(loaded.__exit__, None, None, None)

    sub = {
        "r_before_score": r_before_score,
        "r_after_score": r_after_score,
        "r_before_acc": r_before_acc,
        "r_after_acc": r_after_acc,
        "r_acc_delta": r_acc_delta,
        "r_score_delta": r_score_delta,
        "r_context_diff": r_context_diff,
        "r_format": r_format,
    }
    total = (
        0.50 * r_acc_delta
        + 0.25 * r_score_delta
        + 0.15 * r_after_score
        + 0.05 * r_context_diff
        + 0.05 * r_format
    )
    return total, sub


_compute_single_reward_async = compute_single_reward_async
