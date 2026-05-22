"""llm_gateway.rl.rl_env — RL Sandbox for llm_gateway memory tasks.

对外公开 API 与老仓库 ``rl_env`` 一一对齐，迁移侧仅需改 import 包名即可：

    from llm_gateway.rl.rl_env import (
        MemoryEnv, StepResult, SnapshotSession, LoadedEnv,
        Snapshot, SnapshotBackend,
        InMemorySnapshotBackend, EncodedSnapshotBackend,
        FilesystemSnapshotBackend, GitFsSnapshotBackend,
        CASBlobStore, EncodedSnapshot,
        BaseScorer, ScoreResult,
        dump_snapshot, load_snapshot,
    )
"""

from llm_gateway.rl.rl_env.env import MemoryEnv, StepResult
from llm_gateway.rl.rl_env.git_backend import GitFsSnapshotBackend
from llm_gateway.rl.rl_env.scorer import (
    BaseScorer,
    ConsumeScorer,
    EvolveScorer,
    IngestScorer,
    ScoreResult,
)
from llm_gateway.rl.rl_env.serialize import CASBlobStore, EncodedSnapshot
from llm_gateway.rl.rl_env.snapshot import (
    EncodedSnapshotBackend,
    FilesystemSnapshotBackend,
    InMemorySnapshotBackend,
    Snapshot,
    SnapshotBackend,
    dump_snapshot,
    load_snapshot,
)
from llm_gateway.rl.rl_env.snapshot_session import LoadedEnv, SnapshotSession

__all__ = [
    "MemoryEnv",
    "StepResult",
    "SnapshotSession",
    "LoadedEnv",
    "Snapshot",
    "SnapshotBackend",
    "InMemorySnapshotBackend",
    "EncodedSnapshotBackend",
    "FilesystemSnapshotBackend",
    "GitFsSnapshotBackend",
    "CASBlobStore",
    "EncodedSnapshot",
    "BaseScorer",
    "ScoreResult",
    "ConsumeScorer",
    "EvolveScorer",
    "IngestScorer",
    "dump_snapshot",
    "load_snapshot",
]
