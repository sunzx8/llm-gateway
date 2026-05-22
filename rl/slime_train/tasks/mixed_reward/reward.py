"""Unified reward dispatcher for mixed memory RL training."""

from __future__ import annotations

import logging
from typing import Any

from llm_gateway.rl.slime_train.memory_rl.paths import ensure_workspace_paths

ensure_workspace_paths(__file__)

logger = logging.getLogger(__name__)


def infer_task_from_metadata(metadata: dict[str, Any]) -> str:
    task = str(metadata.get("task", "")).lower()
    if task.startswith("ingest"):
        return "ingest"
    if task.startswith("consolidate") or task.startswith("evolve"):
        return "consolidate"
    if task.startswith("retrieve") or task.startswith("query") or task.startswith("consume"):
        return "retrieve"
    if isinstance(metadata.get("ground_truth"), dict):
        return "retrieve"
    if metadata.get("probes"):
        # Ingest/consolidate samples converted by current converters always set task.
        # If absent, prefer ingest as the safer write-task fallback.
        return "ingest"
    return "retrieve"


async def reward_func(args, sample, **kwargs) -> float:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    task = infer_task_from_metadata(metadata)
    try:
        if task == "ingest":
            from llm_gateway.rl.slime_train.tasks.ingest_reward.reward import reward_func as _reward
        elif task == "consolidate":
            from llm_gateway.rl.slime_train.tasks.consolidate_reward.reward import reward_func as _reward
        elif task == "retrieve":
            from llm_gateway.rl.slime_train.tasks.retrieve_reward.reward import reward_func as _reward
        else:  # pragma: no cover - defensive
            raise ValueError(f"unsupported mixed task: {task}")
        return float(await _reward(args, sample, **kwargs))
    except Exception as exc:  # noqa: BLE001 - reward must be robust in Slime workers
        logger.warning("mixed reward failed for task=%s: %s", task, exc)
        return 0.0


__all__ = ["reward_func", "infer_task_from_metadata"]
