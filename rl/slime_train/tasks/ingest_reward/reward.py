"""Slime reward for T3 ingest RLVR.

The policy outputs JSON tool calls. Reward applies them to the pre-ingest
snapshot, then measures whether generated memories answer the probes.
"""

from __future__ import annotations

import asyncio
import logging
import os

from llm_gateway.rl.slime_train.memory_rl.paths import ensure_workspace_paths

ensure_workspace_paths(__file__)

from llm_gateway.rl.rl_env.snapshot_session import SnapshotSession
from llm_gateway.rl.slime_train.memory_rl.env_acquire import acquire_loaded_env
from llm_gateway.rl.slime_train.memory_rl.probes import score_probe_set
from llm_gateway.rl.slime_train.memory_rl.response_parser import format_reward, parse_tool_calls_response

logger = logging.getLogger(__name__)

_snapshot_session: SnapshotSession | None = None


def _get_snapshot_session() -> SnapshotSession:
    import os

    global _snapshot_session
    data_root = os.environ.get("INGEST_SNAPSHOT_DATA_ROOT") or os.environ.get("SNAPSHOT_DATA_ROOT")
    if not data_root:
        raise RuntimeError("INGEST_SNAPSHOT_DATA_ROOT or SNAPSHOT_DATA_ROOT is required")
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
    timings: dict[str, float] = {}
    _t_reward_start = _time.time()
    try:
        reward, sub = await compute_single_reward_async(response, metadata, timings=timings)
    except Exception as exc:
        logger.warning("ingest reward failed: %s", exc)
        reward, sub = 0.0, {"r_probe": 0.0, "r_format": 0.0, "error": str(exc)}
    timings["reward_total_s"] = round(_time.time() - _t_reward_start, 4)

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
                "reward_timings": timings,
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
                "r_probe": sub.get("r_probe", 0),
                "response_len": len(response),
                "is_truncated": 0,
            }
            with open(log_path, "a") as f:
                f.write(_json.dumps(metrics_record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    return reward


async def compute_single_reward_async(
    response: str,
    metadata: dict,
    *,
    timings: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    import time as _time
    _t0 = _time.time()
    calls, parsed = parse_tool_calls_response(response)
    task_loop = os.environ.get("MEMORY_RL_APPLY_MODE", "tool_calls").strip().lower() == "task_loop"
    replay_task_loop_response = task_loop and isinstance(parsed, dict) and "agentic_trace" in parsed and "tool_calls" in parsed
    r_format = 1.0 if task_loop and not calls else format_reward(response)
    r_probe = 0.0
    if timings is not None:
        timings["parse_and_format_s"] = round(_time.time() - _t0, 4)

    # NOTE: we now also score samples whose ``snapshot_id`` is empty: that's
    # the trajectory-first-step case introduced by the PRE-ingest shift in
    # build_mixed_data.shift_pre_ingest_snapshot. acquire_loaded_env returns
    # a freshly-reset MemoryEnv for those.
    if calls or task_loop:
        session = _get_snapshot_session()
        _t_env_start = _time.time()
        loaded = await acquire_loaded_env(metadata, session)
        if timings is not None:
            timings["env_acquire_s"] = round(_time.time() - _t_env_start, 4)
        try:
            _t_apply_start = _time.time()
            await loaded.env.apply_ingest_tool_calls(
                calls,
                session_time=metadata.get("session_time", ""),
                extras={
                    "session_id": metadata.get("session_id", ""),
                    "session_time": metadata.get("session_time", ""),
                    "pending_messages": metadata.get("pending_messages", []),
                    "replay_task_loop_response": replay_task_loop_response,
                },
            )
            if timings is not None:
                timings["apply_ingest_tool_calls_s"] = round(_time.time() - _t_apply_start, 4)
                timings["ingest_tool_calls_count"] = len(calls) if isinstance(calls, list) else 0

            _t_probe_start = _time.time()
            probes_list = metadata.get("probes", [])
            r_probe = float(await score_probe_set(loaded.env, probes_list))
            if timings is not None:
                timings["score_probe_set_s"] = round(_time.time() - _t_probe_start, 4)
                timings["probes_count"] = len(probes_list) if isinstance(probes_list, list) else 0
        finally:
            _t_release_start = _time.time()
            await asyncio.to_thread(loaded.__exit__, None, None, None)
            if timings is not None:
                timings["env_release_s"] = round(_time.time() - _t_release_start, 4)

    sub = {
        "r_probe": r_probe,
        "r_format": r_format,
    }
    total = 0.90 * r_probe + 0.10 * r_format
    return total, sub


# Backward-compatible aliases for tests/debugging.
_compute_single_reward_async = compute_single_reward_async
