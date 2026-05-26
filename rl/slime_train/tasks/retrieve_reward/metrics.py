"""JSONL logging helpers for Retrieve reward metrics and rollout records."""

from logging import Logger


from __future__ import annotations

import json
import logging
import os
import time as _time

try:  # pragma: no cover - fallback for direct script-style imports
    from .parser import strip_think_wrapper, try_parse_json
except ImportError:  # pragma: no cover
    from parser import strip_think_wrapper, try_parse_json

logger: Logger = logging.getLogger(__name__)


def _reward_metrics_log_path() -> str:
    return os.environ.get(
        "REWARD_METRICS_LOG",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs", "reward_metrics.jsonl"),
    )


def log_debug_response(response: str, metadata: dict) -> None:
    """Append full response/debug parse info to ``reward_debug.jsonl``."""
    log_path = _reward_metrics_log_path()
    debug_path = log_path.replace("reward_metrics.jsonl", "reward_debug.jsonl")
    os.makedirs(os.path.dirname(debug_path), exist_ok=True)

    stripped = strip_think_wrapper(response)
    parsed = try_parse_json(stripped)

    record = {
        "response_full": response,
        "response_len": len(response),
        "has_think_open": "<think>" in response,
        "has_think_close": "</think>" in response,
        "stripped_prefix": stripped[:500] if stripped else "(empty)",
        "parsed_ok": parsed is not None,
        "parsed_queries_count": len(parsed.get("queries", [])) if parsed else 0,
        "traj_id": metadata.get("traj_id", ""),
        "snapshot_id": metadata.get("snapshot_id", "")[:12],
    }

    try:
        with open(debug_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def log_rollout_record(sample, reward: float, sub_rewards: dict) -> None:
    """Append full prompt/response/reward rollout record to JSONL."""
    log_path = _reward_metrics_log_path()
    rollout_path = log_path.replace("reward_metrics.jsonl", "rollout_records.jsonl")
    os.makedirs(os.path.dirname(rollout_path), exist_ok=True)

    prompt = sample.prompt
    if isinstance(prompt, list):
        prompt_data = prompt
    else:
        prompt_data = str(prompt)[:2000]

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}

    record = {
        "timestamp": _time.time(),
        "prompt": prompt_data,
        "response": sample.response or "",
        "reward": reward,
        "sub_rewards": sub_rewards,
        "metadata": metadata,
        "response_len": len(sample.response or ""),
    }

    try:
        with open(rollout_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("写入 rollout record 失败: %s", e)


def log_single_metric(reward: float, sub_rewards: dict, response: str) -> None:
    """Append one compact reward metric record to JSONL."""
    log_path = _reward_metrics_log_path()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    is_trunc = 1 if ("<think>" in response and "</think>" not in response) else 0
    record = {
        "timestamp": _time.time(),
        "r_retrieval_hit": sub_rewards.get("r_retrieval_hit", 0.0),
        "r_format": sub_rewards.get("r_format", 0.0),
        "r_query_quality": sub_rewards.get("r_query_quality", 0.0),
        "r_total": reward,
        "response_len": len(response),
        "is_truncated": is_trunc,
    }

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning("写入 reward metrics 日志失败: %s", e)


# Backward-compatible private aliases.
_log_debug_response = log_debug_response
_log_rollout_record = log_rollout_record
_log_single_metric = log_single_metric
