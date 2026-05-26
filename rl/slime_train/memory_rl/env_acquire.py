"""Helpers to acquire a MemoryEnv for reward / rollout, with proper handling
of the empty-snapshot case introduced by the PRE-ingest snapshot shift.

When build_mixed_data shifts each ingest/consolidate record's snapshot_id by
-1 within its trajectory, the first record of every trajectory ends up with
``snapshot_id == ""``. Downstream code historically did
``snapshot_id and session.load(...)`` and silently skipped, which dropped
the entire first step from training signal. These helpers explicitly start a
fresh MemoryEnv in that case.

Usage::

    async with acquire_env(metadata, snapshot_session) as loaded:
        await loaded.env.apply_ingest_tool_calls(...)
        score = await score_probe_set(loaded.env, probes)

``loaded`` exposes ``.env`` and a context-manager interface compatible with
``SnapshotSession.LoadedEnv``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from typing import Any

from llm_gateway.rl.rl_env.env import MemoryEnv
from llm_gateway.rl.rl_env.snapshot import InMemorySnapshotBackend
from llm_gateway.rl.rl_env.snapshot_session import LoadedEnv, SnapshotSession

logger = logging.getLogger(__name__)


class _EphemeralLoadedEnv:
    """LoadedEnv-shaped wrapper around a freshly-reset MemoryEnv."""

    def __init__(self, env: MemoryEnv, env_dir: str, traj_id: str) -> None:
        self.env = env
        self.env_dir = env_dir
        self.traj_id = traj_id
        self.snapshot_id = ""
        self._released = False

    def __enter__(self) -> "_EphemeralLoadedEnv":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()

    def release(self) -> None:
        if self._released:
            return
        try:
            self.env._owns_base_dir = False  # we'll clean it ourselves
            self.env.close()
        finally:
            self._released = True
            shutil.rmtree(self.env_dir, ignore_errors=True)


def _build_default_llm_from_env() -> Any | None:
    """Mirror SnapshotSession's lazy LLM construction so empty-snap envs
    behave identically (some retrieve/ingest tasks need an LLM in their loop)."""
    from llm_gateway.rl.rl_env.snapshot_session import _build_default_llm_from_env as _impl
    return _impl()


def _make_empty_env(*, traj_id: str, task_version: str, llm: Any | None) -> _EphemeralLoadedEnv:
    tmp_base = os.environ.get("MEMORY_RL_TMP_BASE", tempfile.gettempdir())
    os.makedirs(tmp_base, exist_ok=True)
    safe = (traj_id or "anon").replace(os.sep, "_")
    env_dir = os.path.join(tmp_base, f"empty_{safe}_{uuid.uuid4().hex[:12]}")
    os.makedirs(env_dir, exist_ok=False)

    if llm is None:
        llm = _build_default_llm_from_env()

    env = MemoryEnv(
        llm=llm,
        embedder=None,
        base_dir=env_dir,
        backend="memory",
        enable_git=False,
        snapshot_backend=InMemorySnapshotBackend(),
        task_version=task_version,
    )

    def _sync_reset() -> None:
        asyncio.run(env.reset(user_id=traj_id or "rl_empty", wipe_base_dir=True))

    try:
        asyncio.get_running_loop()
        # Fall through; caller is in async context, do the reset on a worker.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            pool.submit(_sync_reset).result(timeout=30)
    except RuntimeError:
        asyncio.run(env.reset(user_id=traj_id or "rl_empty", wipe_base_dir=True))

    return _EphemeralLoadedEnv(env=env, env_dir=env_dir, traj_id=traj_id)


async def acquire_loaded_env(
    metadata: dict,
    session: SnapshotSession,
    *,
    llm: Any | None = None,
) -> LoadedEnv | _EphemeralLoadedEnv:
    """Async-load a MemoryEnv for a sample. Caller is responsible for
    ``__exit__`` / ``release`` when done.

    - If ``metadata['snapshot_id']`` is non-empty: load from .cbsnap via the
      shared SnapshotSession (post-ingest snapshot path or, after shift, the
      previous step's post-ingest snapshot which serves as the current step's
      true PRE-ingest state).
    - If empty: spin up a freshly-reset MemoryEnv. This is the trajectory
      first-step path after applying the snapshot shift.
    """
    snap_id = metadata.get("snapshot_id") or ""
    traj_id = metadata.get("traj_id") or metadata.get("trajectory_id") or ""
    if snap_id:
        return await asyncio.to_thread(
            session.load,
            traj_id=traj_id or None,
            snapshot_id=snap_id,
            llm=llm,
        )
    # Empty snapshot — first record of trajectory after the PRE-ingest shift.
    task_version = session.task_version if session is not None else os.environ.get(
        "MEMORY_RL_TASK_VERSION", "atomic_code_t2"
    )
    return await asyncio.to_thread(
        _make_empty_env,
        traj_id=traj_id,
        task_version=task_version,
        llm=llm,
    )


@asynccontextmanager
async def acquire_env(metadata: dict, session: SnapshotSession, *, llm: Any | None = None):
    """Async context-manager flavour of acquire_loaded_env."""
    loaded = await acquire_loaded_env(metadata, session, llm=llm)
    try:
        yield loaded
    finally:
        try:
            await asyncio.to_thread(loaded.__exit__, None, None, None)
        except Exception as exc:  # pragma: no cover - cleanup best-effort
            logger.warning("acquire_env cleanup failed: %s", exc)


__all__ = ["acquire_loaded_env", "acquire_env"]
