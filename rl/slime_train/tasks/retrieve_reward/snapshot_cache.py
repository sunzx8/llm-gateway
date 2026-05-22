"""Snapshot loading/cache helpers for Retrieve reward computation."""

from __future__ import annotations

import logging
import os
import threading

from llm_gateway.rl.rl_env import SnapshotSession

logger = logging.getLogger(__name__)

_snapshot_session: SnapshotSession | None = None
_loaded_env_cache: dict[str, object] = {}
_CACHE_MAX_SIZE = 32
_cache_lock = threading.Lock()


def get_snapshot_session() -> SnapshotSession:
    """Return the process-global ``SnapshotSession`` configured by env vars."""
    global _snapshot_session
    data_root = (
        os.environ.get("RETRIEVE_SNAPSHOT_DATA_ROOT")
        or os.environ.get("SNAPSHOT_DATA_ROOT")
        or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ingest_snapshots")
    )
    with _cache_lock:
        task_version = os.environ.get("MEMORY_RL_TASK_VERSION", "atomic_code_t2")
        if (
            _snapshot_session is None
            or _snapshot_session.data_root != data_root
            or _snapshot_session.task_version != task_version
        ):
            _snapshot_session = SnapshotSession(data_root=data_root, task_version=task_version)
            _loaded_env_cache.clear()
        return _snapshot_session


def get_loaded_env(snapshot_id: str, traj_id: str):
    """Return a cached LoadedEnv for read-only retrieve reward execution."""
    global _loaded_env_cache
    session = get_snapshot_session()
    if not traj_id:
        traj_id = session.get_traj_id(snapshot_id)
    cache_key = f"{traj_id}_{snapshot_id}"

    with _cache_lock:
        if cache_key in _loaded_env_cache:
            return _loaded_env_cache[cache_key]

    loaded = session.load(traj_id=traj_id, snapshot_id=snapshot_id)

    with _cache_lock:
        if cache_key in _loaded_env_cache:
            session.release(loaded)
            return _loaded_env_cache[cache_key]

        if len(_loaded_env_cache) >= _CACHE_MAX_SIZE:
            oldest_key = next(iter(_loaded_env_cache))
            old_loaded = _loaded_env_cache.pop(oldest_key)
            session.release(old_loaded)

        _loaded_env_cache[cache_key] = loaded
        return loaded


# Backward-compatible private aliases.
_get_snapshot_session = get_snapshot_session
_get_loaded_env = get_loaded_env
