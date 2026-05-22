from __future__ import annotations

import asyncio
import os
from typing import Any

from llm_gateway.rl.rl_env.snapshot_session import LoadedEnv, SnapshotSession


class EnvToolExecutor:
    """Stateful task tool executor backed by one isolated ``MemoryEnv`` copy."""

    def __init__(self, task: str, metadata: dict[str, Any], *, data_root: str | None = None):
        self.task = task
        self.metadata = metadata
        self.data_root = data_root or _snapshot_root_for_task(task)
        self.session = SnapshotSession(
            data_root=self.data_root,
            enable_git=False,
            task_version=os.environ.get("MEMORY_RL_TASK_VERSION", "atomic_code_t2"),
        )
        self.loaded: LoadedEnv | None = None

    async def __aenter__(self) -> "EnvToolExecutor":
        self.loaded = await asyncio.to_thread(
            self.session.load,
            traj_id=self.metadata.get("traj_id") or None,
            snapshot_id=self.metadata["snapshot_id"],
        )
        return self

    async def __aexit__(self, *exc) -> None:
        if self.loaded is not None:
            await asyncio.to_thread(self.loaded.__exit__, None, None, None)
            self.loaded = None

    async def __call__(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.loaded is None:
            raise RuntimeError("EnvToolExecutor must be used as an async context manager")

        if self.task == "ingest":
            step = await self.loaded.env.apply_ingest_tool_calls(
                calls,
                session_time=self.metadata.get("session_time", ""),
                extras={
                    "session_id": self.metadata.get("session_id", ""),
                    "session_time": self.metadata.get("session_time", ""),
                    "pending_messages": self.metadata.get("pending_messages", []),
                },
            )
        elif self.task == "consolidate":
            step = await self.loaded.env.apply_consolidate_tool_calls(
                calls,
                extras={"session_id": self.metadata.get("session_id", "")},
            )
        else:
            return [{"tool": call.get("tool", ""), "arguments": call.get("arguments", {}), "result": "no executor"} for call in calls]

        trace = step.task_result.stats.get("tool_calls_trace", []) if step.task_result.stats else []
        if isinstance(trace, list) and trace:
            return trace
        return [{"tool": call.get("tool", ""), "arguments": call.get("arguments", {}), "result": step.task_result.error or "ok"} for call in calls]


def _snapshot_root_for_task(task: str) -> str:
    env_name = f"{task.upper()}_SNAPSHOT_DATA_ROOT"
    data_root = os.environ.get(env_name) or os.environ.get("SNAPSHOT_DATA_ROOT")
    if not data_root:
        raise RuntimeError(f"{env_name} or SNAPSHOT_DATA_ROOT is required")
    return data_root
