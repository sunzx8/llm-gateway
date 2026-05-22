"""Compatibility wrapper for retrieve training snapshot loading.

The implementation now lives in :mod:`llm_gateway.rl.rl_env.snapshot_session`
so it can be shared by other training/evaluation pipelines.
"""

from __future__ import annotations

from llm_gateway.rl.slime_train.memory_rl.paths import ensure_workspace_paths

ensure_workspace_paths(__file__)

from llm_gateway.rl.rl_env.snapshot_session import LoadedEnv, SnapshotSession  # noqa: E402

__all__ = ["SnapshotSession", "LoadedEnv"]
