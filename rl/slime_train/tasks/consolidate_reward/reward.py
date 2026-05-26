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
from llm_gateway.rl.slime_train.memory_rl.env_acquire import acquire_loaded_env
from llm_gateway.rl.slime_train.memory_rl.probes import (
    average_probe_score,
    context_diff_score,
    evaluate_probe_set,
    headroom_after_score,
    positive_probe_accuracy_delta,
    positive_probe_score_delta,
    probe_accuracy,
    signed_probe_accuracy_delta,
    signed_probe_score_delta,
)
from llm_gateway.rl.slime_train.memory_rl.response_parser import (
    format_reward,
    parse_task_loop_payload,
    parse_tool_calls_response,
)

logger = logging.getLogger(__name__)


def _f(env: str, default: float) -> float:
    """Read float from env with fallback (silently use default on parse error)."""
    val = os.environ.get(env, "")
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        return default


# Reward weights — overridable from launch script via env vars so we can A/B
# test reward shaping without code changes.
#
# Defaults (signed-delta variant, recommended for consolidate):
#   * r_acc_delta_signed:   net change in probe correctness rate, in [-1, 1]
#   * r_score_delta_signed: net change in avg probe score, in [-1, 1]
#   * r_after_headroom:     score improvement / remaining headroom (0 for
#                           probes that were already correct → no free points)
#   * r_context_diff:       only counted when scores don't regress
#   * r_format:             structural sanity of tool-call output
#   * action_cost:          subtract small constant per tool call beyond
#                           ``CONSOLIDATE_REWARD_FREE_CALLS`` to discourage
#                           "spam tool calls" exploits (default OFF: 0.0)
#
# Setting any *_legacy=1 flag falls back to the original positive-only
# formulation if you want to reproduce the old behavior.
W_ACC_DELTA = _f("CONSOLIDATE_REWARD_W_ACC_DELTA", 0.50)
W_SCORE_DELTA = _f("CONSOLIDATE_REWARD_W_SCORE_DELTA", 0.25)
W_AFTER = _f("CONSOLIDATE_REWARD_W_AFTER", 0.15)
W_CONTEXT_DIFF = _f("CONSOLIDATE_REWARD_W_CONTEXT_DIFF", 0.05)
W_FORMAT = _f("CONSOLIDATE_REWARD_W_FORMAT", 0.05)
ACTION_COST_PER_CALL = _f("CONSOLIDATE_REWARD_ACTION_COST", 0.0)
ACTION_COST_FREE_CALLS = int(_f("CONSOLIDATE_REWARD_FREE_CALLS", 8))
USE_LEGACY_POSITIVE_ONLY = os.environ.get("CONSOLIDATE_REWARD_LEGACY_POSITIVE", "0") == "1"
USE_HEADROOM_AFTER = os.environ.get("CONSOLIDATE_REWARD_USE_HEADROOM", "1") == "1"

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

    # prompt_uid 用于 swanlab_monitor 精确按 prompt 分组（GRPO/GSPO 同组的 N 个
    # rollout 应聚合到一起算 std/advantage，而不是按落盘顺序流式凑数）。
    # 同一 prompt 的 N 个 sample 共享同一份 metadata，所以三者拼接是稳定的 key。
    prompt_uid = "|".join(
        [
            str(metadata.get("traj_id", "")),
            str(metadata.get("snapshot_id", "")),
            str(metadata.get("session_id", "")),
        ]
    )

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
                "prompt_uid": prompt_uid,
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
                "prompt_uid": prompt_uid,
                "r_total": reward,
                "r_format": sub.get("r_format", 0),
                "r_after_score": sub.get("r_after_score", 0),
                "r_after_headroom": sub.get("r_after_headroom", 0),
                "r_acc_delta": sub.get("r_acc_delta", 0),
                "r_acc_delta_signed": sub.get("r_acc_delta_signed", 0),
                "r_score_delta": sub.get("r_score_delta", 0),
                "r_score_delta_signed": sub.get("r_score_delta_signed", 0),
                "r_context_diff": sub.get("r_context_diff", 0),
                "n_tool_calls": sub.get("n_tool_calls", 0),
                "action_cost": sub.get("action_cost", 0),
                "used_post_snapshot": sub.get("used_post_snapshot", 0),
                "response_len": len(response),
                "is_truncated": 0,
            }
            with open(log_path, "a") as f:
                f.write(_json.dumps(metrics_record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    return reward


async def compute_single_reward_async(response: str, metadata: dict) -> tuple[float, dict[str, float]]:
    task_loop = os.environ.get("MEMORY_RL_APPLY_MODE", "tool_calls").strip().lower() == "task_loop"
    task_payload = parse_task_loop_payload(response) if task_loop else None
    if task_payload is not None:
        raw_calls = task_payload.get("tool_calls", [])
        calls = [c for c in raw_calls if isinstance(c, dict)] if isinstance(raw_calls, list) else []
        parsed = task_payload
    else:
        calls, parsed = parse_tool_calls_response(response)
    replay_task_loop_response = task_payload is not None or (
        task_loop and isinstance(parsed, dict) and "agentic_trace" in parsed and "tool_calls" in parsed
    )
    r_format = 1.0 if task_payload is not None else (1.0 if task_loop and not calls else format_reward(response))
    probes = metadata.get("probes", [])

    r_before_score = 0.0
    r_after_score = 0.0
    r_before_acc = 0.0
    r_after_acc = 0.0
    r_acc_delta = 0.0
    r_acc_delta_signed = 0.0
    r_score_delta = 0.0
    r_score_delta_signed = 0.0
    r_after_headroom = 0.0
    r_context_diff = 0.0
    n_tool_calls_executed = 0

    used_post_snapshot = 0.0
    post_snapshot_path = ""
    post_snapshot_id = ""
    if isinstance(parsed, dict):
        post_snapshot = parsed.get("post_consolidate_snapshot")
        if isinstance(post_snapshot, dict):
            post_snapshot_path = str(post_snapshot.get("snapshot_path", "") or "")
            post_snapshot_id = str(post_snapshot.get("snapshot_id", "") or "")

    if (calls or task_loop) and probes:
        session = _get_snapshot_session()
        traj_id = metadata.get("traj_id") or None
        source_snapshot_id = metadata.get("snapshot_id") or ""
        # NOTE: empty source_snapshot_id is the trajectory-first-step case
        # produced by build_mixed_data.shift_pre_ingest_snapshot. We start
        # from a freshly-reset MemoryEnv via acquire_loaded_env in that case.

        async def _load_before():
            return await acquire_loaded_env(
                {"snapshot_id": source_snapshot_id, "traj_id": traj_id},
                session,
            )

        before_evals = await evaluate_probe_set(None, probes, env_factory=_load_before)
        r_before_score = average_probe_score(before_evals)
        r_before_acc = probe_accuracy(before_evals)

        if post_snapshot_path and os.path.exists(post_snapshot_path):
            used_post_snapshot = 1.0

            async def _load_after_post():
                return await asyncio.to_thread(
                    session.load_path,
                    post_snapshot_path,
                    traj_id=traj_id,
                    snapshot_id=post_snapshot_id or None,
                )

            after_evals = await evaluate_probe_set(None, probes, env_factory=_load_after_post)
        else:
            # Fallback for old rollouts without persisted post snapshot: replay tool calls.
            loaded = await acquire_loaded_env(
                {"snapshot_id": source_snapshot_id, "traj_id": traj_id},
                session,
            )
            try:
                await loaded.env.apply_consolidate_tool_calls(
                    calls,
                    extras={
                        "session_id": metadata.get("session_id", ""),
                        "replay_task_loop_response": replay_task_loop_response,
                    },
                )
                after_evals = await evaluate_probe_set(loaded.env, probes)
            finally:
                await asyncio.to_thread(loaded.__exit__, None, None, None)

        r_after_score = average_probe_score(after_evals)
        r_after_acc = probe_accuracy(after_evals)
        r_acc_delta = positive_probe_accuracy_delta(before_evals, after_evals)
        r_acc_delta_signed = signed_probe_accuracy_delta(before_evals, after_evals)
        r_score_delta = positive_probe_score_delta(before_evals, after_evals)
        r_score_delta_signed = signed_probe_score_delta(before_evals, after_evals)
        r_after_headroom = headroom_after_score(before_evals, after_evals)
        r_context_diff = context_diff_score(
            [str(e.get("context", "")) for e in before_evals],
            [str(e.get("context", "")) for e in after_evals],
            before_scores=[float(e.get("score", 0.0)) for e in before_evals],
            after_scores=[float(e.get("score", 0.0)) for e in after_evals],
        )

        # tool_calls 数量来自 task_loop 的 trace 或外层的 calls
        if replay_task_loop_response and isinstance(parsed, dict):
            n_tool_calls_executed = len(parsed.get("tool_calls", []) or [])
        else:
            n_tool_calls_executed = len(calls)

    # ----------------------------------------------------------------------
    # Reward shaping
    # ----------------------------------------------------------------------
    if USE_LEGACY_POSITIVE_ONLY:
        # 旧公式：只 reward 正向变化，不惩罚 regression（保留作为对比 baseline）
        acc_term = r_acc_delta
        score_term = r_score_delta
        after_term = r_after_score
    else:
        # 新公式：用 signed delta，让 regression 真正扣分；
        # after term 改成 headroom-normalized，避免"原本就对"白送 0.15
        acc_term = r_acc_delta_signed
        score_term = r_score_delta_signed
        after_term = r_after_headroom if USE_HEADROOM_AFTER else r_after_score

    # 动作成本：超过 free 额度后，每多一个 tool call 扣 ACTION_COST_PER_CALL
    action_cost = 0.0
    if ACTION_COST_PER_CALL > 0 and n_tool_calls_executed > ACTION_COST_FREE_CALLS:
        action_cost = ACTION_COST_PER_CALL * (n_tool_calls_executed - ACTION_COST_FREE_CALLS)

    total = (
        W_ACC_DELTA * acc_term
        + W_SCORE_DELTA * score_term
        + W_AFTER * after_term
        + W_CONTEXT_DIFF * r_context_diff
        + W_FORMAT * r_format
        - action_cost
    )

    sub = {
        "r_before_score": r_before_score,
        "r_after_score": r_after_score,
        "r_before_acc": r_before_acc,
        "r_after_acc": r_after_acc,
        "r_acc_delta": r_acc_delta,
        "r_acc_delta_signed": r_acc_delta_signed,
        "r_score_delta": r_score_delta,
        "r_score_delta_signed": r_score_delta_signed,
        "r_after_headroom": r_after_headroom,
        "r_context_diff": r_context_diff,
        "r_format": r_format,
        "n_tool_calls": float(n_tool_calls_executed),
        "action_cost": action_cost,
        "used_post_snapshot": used_post_snapshot,
    }
    return total, sub


_compute_single_reward_async = compute_single_reward_async
