"""Snapshot layer for :class:`llm_gateway.rl.rl_env.MemoryEnv`.

A snapshot captures the *full* state of the three memory backends at a
given point in time, so that :meth:`MemoryEnv.restore` can rewind the
env to that state and replay alternative actions.

Three concrete backends are shipped today:

- :class:`InMemorySnapshotBackend` — fastest, biggest footprint
- :class:`EncodedSnapshotBackend` — compressed in-RAM (zstd + numpy float32)
- :class:`FilesystemSnapshotBackend` — persist as ``.cbsnap`` on disk

A 4th option, :class:`llm_gateway.rl.rl_env.git_backend.GitFsSnapshotBackend`,
is cheapest for branching rollouts when ``enable_git=True`` (FS state goes
through git refs; only Vec/Graph go through the inner backend).
"""

from __future__ import annotations

import copy
import logging
import os
import shutil
import tempfile
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llm_gateway.rl.rl_env.serialize import (
    CASBlobStore,
    EncodedSnapshot,
    decode_snapshot,
    dump_encoded,
    encode_snapshot,
    load_encoded,
)

if TYPE_CHECKING:
    from storage.file_system_store import FileSystemStore
    from storage.graph_stores import GraphStore
    from storage.vector_stores import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Snapshot handle
# ---------------------------------------------------------------------------


@dataclass
class Snapshot:
    """Opaque snapshot handle.

    Produced by :meth:`SnapshotBackend.capture` and consumed by
    :meth:`SnapshotBackend.restore`. The ``payload`` shape is owned by
    the backend — treat it as opaque.
    """

    snapshot_id: str
    backend_name: str
    payload: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return (
            f"Snapshot(id={self.snapshot_id[:8]}, "
            f"backend={self.backend_name}, meta={self.meta})"
        )

    def size_on_disk(self) -> int:
        """Best-effort size estimate in bytes (0 if backend doesn't track)."""
        enc: EncodedSnapshot | None = self.payload.get("encoded")
        if enc is not None:
            return enc.size_bytes
        fs_dir = self.payload.get("fs_dir")
        if fs_dir and os.path.isdir(fs_dir):
            total = 0
            for root, _dirs, files in os.walk(fs_dir):
                for fn in files:
                    try:
                        total += os.path.getsize(os.path.join(root, fn))
                    except OSError:  # pragma: no cover
                        pass
            return total
        return 0


# ---------------------------------------------------------------------------
# Backend contract
# ---------------------------------------------------------------------------


class SnapshotBackend(ABC):
    """Pluggable snapshot/restore strategy."""

    name: str = "abstract"

    @abstractmethod
    def capture(
        self,
        fs: "FileSystemStore",
        vec: "VectorStore",
        graph: "GraphStore",
        *,
        meta: dict[str, Any] | None = None,
    ) -> Snapshot:
        """Freeze the current state of the three stores into a Snapshot."""

    @abstractmethod
    def restore(
        self,
        snapshot: Snapshot,
        fs: "FileSystemStore",
        vec: "VectorStore",
        graph: "GraphStore",
    ) -> None:
        """Rewind the three stores to the state captured in ``snapshot``."""

    def release(self, snapshot: Snapshot) -> None:
        """Best-effort cleanup of any on-disk artefacts."""
        return None


# ---------------------------------------------------------------------------
# In-memory backend — fastest, no serialization overhead
# ---------------------------------------------------------------------------


class InMemorySnapshotBackend(SnapshotBackend):
    """Fastest backend: deepcopy Vec/Graph + copytree FS to a scratch dir."""

    name = "in_memory"

    def __init__(self, snapshot_root: str | None = None) -> None:
        if snapshot_root is None:
            snapshot_root = tempfile.mkdtemp(prefix="rl_env_snap_")
        os.makedirs(snapshot_root, exist_ok=True)
        self.snapshot_root = snapshot_root

    def capture(
        self,
        fs: "FileSystemStore",
        vec: "VectorStore",
        graph: "GraphStore",
        *,
        meta: dict[str, Any] | None = None,
    ) -> Snapshot:
        snap_id = uuid.uuid4().hex
        fs_dst = os.path.join(self.snapshot_root, snap_id, "fs")
        os.makedirs(os.path.dirname(fs_dst), exist_ok=True)

        if os.path.isdir(fs.base_path):
            shutil.copytree(
                fs.base_path, fs_dst, symlinks=False, ignore_dangling_symlinks=True,
            )
        else:
            os.makedirs(fs_dst, exist_ok=True)

        payload: dict[str, Any] = {
            "fs_dir": fs_dst,
            "fs_base_path_at_capture": fs.base_path,
            "vec": {
                "collections": copy.deepcopy(vec._collections),
                "id_counter": vec._id_counter,
            },
            "graph": {
                "nodes": copy.deepcopy(graph._nodes),
                "edges": copy.deepcopy(graph._edges),
                "edge_counter": graph._edge_counter,
                # llm_gateway storage 多了 node/edge embedding，兼容性 deepcopy
                "node_embeddings": copy.deepcopy(getattr(graph, "_node_embeddings", {})),
                "edge_embeddings": copy.deepcopy(getattr(graph, "_edge_embeddings", {})),
            },
        }
        return Snapshot(
            snapshot_id=snap_id,
            backend_name=self.name,
            payload=payload,
            meta=dict(meta or {}),
        )

    def restore(
        self,
        snapshot: Snapshot,
        fs: "FileSystemStore",
        vec: "VectorStore",
        graph: "GraphStore",
    ) -> None:
        if snapshot.backend_name != self.name:
            raise ValueError(
                f"InMemorySnapshotBackend cannot restore snapshot from "
                f"backend={snapshot.backend_name!r}"
            )
        payload = snapshot.payload

        fs_src = payload["fs_dir"]
        target = fs.base_path
        if os.path.isdir(target):
            shutil.rmtree(target)
        shutil.copytree(fs_src, target, symlinks=False, ignore_dangling_symlinks=True)

        vec_p = payload["vec"]
        vec._collections = copy.deepcopy(vec_p["collections"])
        vec._id_counter = vec_p["id_counter"]
        if hasattr(vec, "_last_add_warnings"):
            vec._last_add_warnings = []

        graph_p = payload["graph"]
        graph._nodes = copy.deepcopy(graph_p["nodes"])
        graph._edges = copy.deepcopy(graph_p["edges"])
        graph._edge_counter = graph_p["edge_counter"]
        if hasattr(graph, "_node_embeddings"):
            graph._node_embeddings = copy.deepcopy(graph_p.get("node_embeddings", {}))
        if hasattr(graph, "_edge_embeddings"):
            graph._edge_embeddings = copy.deepcopy(graph_p.get("edge_embeddings", {}))

    def release(self, snapshot: Snapshot) -> None:
        fs_src = snapshot.payload.get("fs_dir")
        if fs_src and os.path.isdir(fs_src):
            try:
                shutil.rmtree(os.path.dirname(fs_src), ignore_errors=True)
            except OSError as e:  # pragma: no cover
                logger.warning("release(%s): %s", snapshot.snapshot_id[:8], e)


# ---------------------------------------------------------------------------
# Encoded-in-memory backend — serialize once, hold bytes in RAM
# ---------------------------------------------------------------------------


class EncodedSnapshotBackend(SnapshotBackend):
    """Keep snapshots as :class:`EncodedSnapshot` blobs in process memory."""

    name = "encoded"

    def __init__(
        self,
        *,
        fs_mode: str = "tar_zst",
        cas: CASBlobStore | None = None,
        zstd_level: int = 3,
    ) -> None:
        if fs_mode == "cas" and cas is None:
            raise ValueError("fs_mode='cas' requires a CASBlobStore")
        self.fs_mode = fs_mode
        self.cas = cas
        self.zstd_level = zstd_level

    def capture(
        self,
        fs: "FileSystemStore",
        vec: "VectorStore",
        graph: "GraphStore",
        *,
        meta: dict[str, Any] | None = None,
    ) -> Snapshot:
        snap_id = uuid.uuid4().hex
        encoded = encode_snapshot(
            snap_id,
            meta=dict(meta or {}),
            fs=fs, vec=vec, graph=graph,
            fs_mode=self.fs_mode,
            cas=self.cas,
            zstd_level=self.zstd_level,
        )
        logger.debug(
            "EncodedSnapshotBackend: captured %s (blob=%d bytes, fs_mode=%s)",
            snap_id[:8], len(encoded.blob), self.fs_mode,
        )
        return Snapshot(
            snapshot_id=snap_id,
            backend_name=self.name,
            payload={"encoded": encoded},
            meta=dict(meta or {}),
        )

    def restore(
        self,
        snapshot: Snapshot,
        fs: "FileSystemStore",
        vec: "VectorStore",
        graph: "GraphStore",
    ) -> None:
        enc: EncodedSnapshot = snapshot.payload["encoded"]
        decode_snapshot(enc, fs=fs, vec=vec, graph=graph, cas=self.cas)


# ---------------------------------------------------------------------------
# Filesystem backend — persist encoded snapshots to disk
# ---------------------------------------------------------------------------


class FilesystemSnapshotBackend(SnapshotBackend):
    """Persist snapshots as ``.cbsnap`` files on disk."""

    name = "filesystem"

    def __init__(
        self,
        root: str,
        *,
        fs_mode: str = "tar_zst",
        cas_subdir: str = "cas",
        zstd_level: int = 3,
    ) -> None:
        os.makedirs(root, exist_ok=True)
        self.root = root
        self.fs_mode = fs_mode
        self.zstd_level = zstd_level
        self.cas: CASBlobStore | None = None
        if fs_mode == "cas":
            self.cas = CASBlobStore(os.path.join(root, cas_subdir))
        elif fs_mode != "tar_zst":
            raise ValueError(f"invalid fs_mode={fs_mode!r}")

    def _snap_path(self, snap_id: str) -> str:
        return os.path.join(self.root, f"{snap_id}.cbsnap")

    def capture(
        self,
        fs: "FileSystemStore",
        vec: "VectorStore",
        graph: "GraphStore",
        *,
        meta: dict[str, Any] | None = None,
    ) -> Snapshot:
        snap_id = uuid.uuid4().hex
        encoded = encode_snapshot(
            snap_id,
            meta=dict(meta or {}),
            fs=fs, vec=vec, graph=graph,
            fs_mode=self.fs_mode,
            cas=self.cas,
            zstd_level=self.zstd_level,
        )
        path = self._snap_path(snap_id)
        enc_to_dump = encoded
        if self.fs_mode == "cas":
            enc_to_dump = EncodedSnapshot(
                version=encoded.version,
                snapshot_id=encoded.snapshot_id,
                blob=encoded.blob,
                fs_mode="cas",
                fs_blobs={},
            )
        dump_encoded(enc_to_dump, path)

        size_on_disk = os.path.getsize(path)
        logger.debug(
            "FilesystemSnapshotBackend: saved %s (%d bytes, fs_mode=%s)",
            snap_id[:8], size_on_disk, self.fs_mode,
        )
        return Snapshot(
            snapshot_id=snap_id,
            backend_name=self.name,
            payload={
                "path": path,
                "fs_mode": self.fs_mode,
                "size_on_disk": size_on_disk,
            },
            meta=dict(meta or {}),
        )

    def restore(
        self,
        snapshot: Snapshot,
        fs: "FileSystemStore",
        vec: "VectorStore",
        graph: "GraphStore",
    ) -> None:
        path = snapshot.payload["path"]
        encoded = load_encoded(path)
        decode_snapshot(encoded, fs=fs, vec=vec, graph=graph, cas=self.cas)

    def release(self, snapshot: Snapshot) -> None:
        # 持久化产物：release 不删除磁盘文件。
        pass

    def attach(self, path: str) -> Snapshot:
        """Wrap an existing ``.cbsnap`` file as a Snapshot handle."""
        encoded = load_encoded(path)
        return Snapshot(
            snapshot_id=encoded.snapshot_id,
            backend_name=self.name,
            payload={
                "path": path,
                "fs_mode": encoded.fs_mode,
                "size_on_disk": os.path.getsize(path),
            },
            meta={},
        )


# ---------------------------------------------------------------------------
# Cross-backend dump / load conveniences
# ---------------------------------------------------------------------------


def dump_snapshot(
    snapshot: Snapshot,
    path: str,
    *,
    fs: "FileSystemStore | None" = None,
    vec: "VectorStore | None" = None,
    graph: "GraphStore | None" = None,
    cas: CASBlobStore | None = None,
    fs_mode: str = "tar_zst",
    zstd_level: int = 3,
) -> None:
    """Write any snapshot to a single ``.cbsnap`` file."""
    if snapshot.backend_name == EncodedSnapshotBackend.name:
        enc: EncodedSnapshot = snapshot.payload["encoded"]
        dump_encoded(enc, path)
        return
    if snapshot.backend_name == FilesystemSnapshotBackend.name:
        src = snapshot.payload["path"]
        if src != path:
            shutil.copyfile(src, path)
        return
    if snapshot.backend_name == "git_fs":
        raise ValueError(
            "dump_snapshot(): git_fs snapshots must be persisted via "
            "GitFsSnapshotBackend.export_bundle(snapshot, dir, fs, vec, graph). "
            "A single-file .cbsnap cannot represent the git ref."
        )

    if fs is None or vec is None or graph is None:
        raise ValueError(
            "dump_snapshot(): InMemorySnapshotBackend snapshots require "
            "live `fs`, `vec`, `graph` so their state can be encoded."
        )
    backend = InMemorySnapshotBackend()
    backend.restore(snapshot, fs, vec, graph)
    encoded = encode_snapshot(
        snapshot.snapshot_id,
        meta=snapshot.meta,
        fs=fs, vec=vec, graph=graph,
        fs_mode=fs_mode,
        cas=cas,
        zstd_level=zstd_level,
    )
    dump_encoded(encoded, path)


def load_snapshot(path: str) -> Snapshot:
    """Load a ``.cbsnap`` file as a :class:`FilesystemSnapshotBackend` snapshot handle."""
    encoded = load_encoded(path)
    return Snapshot(
        snapshot_id=encoded.snapshot_id,
        backend_name=FilesystemSnapshotBackend.name,
        payload={
            "path": path,
            "fs_mode": encoded.fs_mode,
            "size_on_disk": os.path.getsize(path),
        },
        meta={},
    )


__all__ = [
    "Snapshot",
    "SnapshotBackend",
    "InMemorySnapshotBackend",
    "EncodedSnapshotBackend",
    "FilesystemSnapshotBackend",
    "dump_snapshot",
    "load_snapshot",
]
