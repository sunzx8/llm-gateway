"""Git-backed snapshot backend for the FS layer.

``FileSystemStore(enable_git=True)`` already runs ``git add -A && git commit``
at every task boundary. That makes the FS history a perfect content-addressable,
delta-compressed log — *every commit is already a snapshot*. This backend leans
on that and only takes ~10ms per snapshot (one commit + rev-parse).

Composition: the FS layer = git ref/SHA; the Vec/Graph layer is delegated to
an inner :class:`SnapshotBackend` (defaults to :class:`InMemorySnapshotBackend`).

Cross-process / cross-machine portability is via :meth:`export_bundle` /
:meth:`import_bundle`, which produce a small directory holding a git bundle +
encoded Vec/Graph cbsnap.
"""

from __future__ import annotations

import logging
import os
import subprocess
import uuid
from typing import TYPE_CHECKING, Any

from llm_gateway.rl.rl_env.serialize import EncodedSnapshot, dump_encoded, load_encoded
from llm_gateway.rl.rl_env.snapshot import (
    InMemorySnapshotBackend,
    Snapshot,
    SnapshotBackend,
)

if TYPE_CHECKING:
    from storage.file_system_store import FileSystemStore
    from storage.graph_stores import GraphStore
    from storage.vector_stores import VectorStore

logger = logging.getLogger(__name__)


_SNAP_COMMIT_PREFIX = "rl_env/snap"


def _git(repo: str, *args: str, timeout: int = 30) -> tuple[int, str, str]:
    """Run ``git`` inside ``repo`` and return ``(rc, stdout, stderr)``."""
    try:
        proc = subprocess.run(  # noqa: S603 — argv is fully internal
            ["git", *args],
            capture_output=True, text=True, timeout=timeout, cwd=repo,
        )
    except subprocess.TimeoutExpired:
        return (124, "", f"git {' '.join(args)} timed out after {timeout}s")
    except FileNotFoundError:
        return (127, "", "git binary not found on PATH")
    return (proc.returncode, proc.stdout, proc.stderr)


def _is_git_repo(path: str) -> bool:
    return os.path.isdir(os.path.join(path, ".git"))


class GitFsSnapshotBackend(SnapshotBackend):
    """FS layer = git ref/SHA; Vec/Graph layer = inner backend."""

    name = "git_fs"

    def __init__(
        self,
        *,
        inner: SnapshotBackend | None = None,
        commit_prefix: str = _SNAP_COMMIT_PREFIX,
        require_git: bool = True,
        keep_clean_on_restore: bool = True,
    ) -> None:
        self.inner: SnapshotBackend = inner or InMemorySnapshotBackend()
        self.commit_prefix = commit_prefix
        self.require_git = require_git
        self.keep_clean_on_restore = keep_clean_on_restore

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _ensure_git(self, fs: "FileSystemStore") -> bool:
        if not getattr(fs, "enable_git", False) or not _is_git_repo(fs.base_path):
            if self.require_git:
                raise RuntimeError(
                    "GitFsSnapshotBackend requires FileSystemStore(enable_git=True). "
                    "Either enable git on the env (MemoryEnv(enable_git=True)) or pass "
                    "require_git=False to fall through to the inner backend."
                )
            return False
        return True

    def _commit_or_use_head(self, fs: "FileSystemStore", tag: str) -> str:
        repo = fs.base_path
        rc, _, err = _git(repo, "add", "-A")
        if rc != 0:
            raise RuntimeError(f"GitFsSnapshotBackend: git add failed: {err.strip()}")
        rc_diff, out, _ = _git(repo, "diff", "--cached", "--name-only")
        has_changes = rc_diff == 0 and bool(out.strip())
        if has_changes:
            msg = f"{self.commit_prefix}: {tag}"[:4000]
            rc, out2, err2 = _git(repo, "commit", "-m", msg)
            if rc != 0:
                raise RuntimeError(
                    f"GitFsSnapshotBackend: git commit failed: "
                    f"{(err2 or out2).strip()}"
                )
        rc, sha_full, err = _git(repo, "rev-parse", "HEAD")
        if rc != 0:
            raise RuntimeError(
                f"GitFsSnapshotBackend: rev-parse HEAD failed: {err.strip()}"
            )
        return sha_full.strip()

    # ------------------------------------------------------------------
    # capture / restore
    # ------------------------------------------------------------------

    def capture(
        self,
        fs: "FileSystemStore",
        vec: "VectorStore",
        graph: "GraphStore",
        *,
        meta: dict[str, Any] | None = None,
    ) -> Snapshot:
        snap_id = uuid.uuid4().hex
        meta = dict(meta or {})

        use_git = self._ensure_git(fs)
        sha: str | None = None

        if use_git:
            tag = meta.get("step_counter", "?")
            sha = self._commit_or_use_head(fs, f"{snap_id[:8]} step={tag}")
            logger.debug(
                "GitFsSnapshotBackend: captured %s @ %s",
                snap_id[:8], sha[:10],
            )

        inner_snap: Snapshot = self.inner.capture(fs, vec, graph, meta=meta)

        return Snapshot(
            snapshot_id=snap_id,
            backend_name=self.name,
            payload={
                "git_sha": sha,
                "use_git": use_git,
                "inner_snapshot": inner_snap,
                "fs_base_path_at_capture": fs.base_path,
            },
            meta=meta,
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
                f"GitFsSnapshotBackend cannot restore snapshot from "
                f"backend={snapshot.backend_name!r}"
            )
        payload = snapshot.payload
        inner_snap: Snapshot = payload["inner_snapshot"]

        if payload.get("use_git"):
            sha = payload["git_sha"]
            repo = fs.base_path
            if not _is_git_repo(repo):
                raise RuntimeError(
                    "GitFsSnapshotBackend.restore: target FS is no longer a "
                    "git repo. Did the env get reset()? Snapshot was captured "
                    f"at base_path={payload.get('fs_base_path_at_capture')!r}"
                )
            rc, _, err = _git(repo, "reset", "--hard", sha)
            if rc != 0:
                raise RuntimeError(
                    f"GitFsSnapshotBackend: reset --hard {sha[:10]} failed: "
                    f"{err.strip()}"
                )
            if self.keep_clean_on_restore:
                rc, _, err = _git(repo, "clean", "-ffdx")
                if rc != 0:
                    logger.warning(
                        "GitFsSnapshotBackend: git clean failed (non-fatal): %s",
                        err.strip(),
                    )

        inner = self._resolve_inner_for(inner_snap)
        inner.restore(inner_snap, fs, vec, graph)

    def release(self, snapshot: Snapshot) -> None:
        # 不删除 git commit；inner snapshot 还可能持有 Vec/Graph 临时产物。
        inner_snap = snapshot.payload.get("inner_snapshot")
        if inner_snap is not None:
            inner = self._resolve_inner_for(inner_snap)
            inner.release(inner_snap)

    def _resolve_inner_for(self, inner_snap: Snapshot) -> SnapshotBackend:
        if inner_snap.backend_name == self.inner.name:
            return self.inner
        from llm_gateway.rl.rl_env.snapshot import EncodedSnapshotBackend

        if inner_snap.backend_name == EncodedSnapshotBackend.name:
            return EncodedSnapshotBackend()
        raise ValueError(
            f"GitFsSnapshotBackend: inner snapshot from "
            f"backend={inner_snap.backend_name!r} cannot be routed; "
            f"configured inner is {self.inner.name!r}"
        )

    # ------------------------------------------------------------------
    # cross-process / cross-machine helpers
    # ------------------------------------------------------------------

    def export_bundle(
        self,
        snapshot: Snapshot,
        path: str,
        fs: "FileSystemStore",
        vec: "VectorStore",
        graph: "GraphStore",
    ) -> None:
        """Write a portable representation of the snapshot to ``path``.

        Layout (directory):
        - ``fs.bundle``       — git bundle covering the snapshot commit (delta packfile)
        - ``vec_graph.cbsnap`` — encoded Vec/Graph payload only
        - ``manifest.json``   — links the two together
        """
        import json

        if not snapshot.payload.get("use_git"):
            raise RuntimeError(
                "export_bundle: snapshot has no git SHA (was captured "
                "without git). Use dump_snapshot() instead."
            )
        sha = snapshot.payload["git_sha"]
        os.makedirs(path, exist_ok=True)

        bundle_path = os.path.join(path, "fs.bundle")
        tag_name = f"refs/rl_env/snap_{snapshot.snapshot_id[:12]}"
        rc, _, err = _git(fs.base_path, "update-ref", tag_name, sha)
        if rc != 0:
            raise RuntimeError(
                f"export_bundle: update-ref failed: {err.strip()}"
            )
        try:
            rc, _, err = _git(
                fs.base_path, "bundle", "create", bundle_path, tag_name,
            )
            if rc != 0:
                raise RuntimeError(
                    f"export_bundle: git bundle create failed: {err.strip()}"
                )
        finally:
            _git(fs.base_path, "update-ref", "-d", tag_name)

        from llm_gateway.rl.rl_env.serialize import encode_snapshot

        cbsnap_path = os.path.join(path, "vec_graph.cbsnap")
        encoded = encode_snapshot(
            snapshot.snapshot_id,
            meta=snapshot.meta,
            fs=fs, vec=vec, graph=graph,
            fs_mode="tar_zst",
            cas=None,
            zstd_level=3,
        )
        slim = EncodedSnapshot(
            version=encoded.version,
            snapshot_id=encoded.snapshot_id,
            blob=encoded.blob,
            fs_mode="git_external",
            fs_blobs={},
        )
        dump_encoded(slim, cbsnap_path)

        manifest = {
            "version": 1,
            "snapshot_id": snapshot.snapshot_id,
            "git_sha": sha,
            "fs_bundle": "fs.bundle",
            "vec_graph_cbsnap": "vec_graph.cbsnap",
            "meta": snapshot.meta,
        }
        with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            "GitFsSnapshotBackend.export_bundle: wrote %s "
            "(bundle=%d B, vec_graph=%d B)",
            path,
            os.path.getsize(bundle_path),
            os.path.getsize(cbsnap_path),
        )

    def import_bundle(
        self,
        path: str,
        fs: "FileSystemStore",
    ) -> Snapshot:
        """Import a bundle created by :meth:`export_bundle` into ``fs``."""
        import json

        with open(os.path.join(path, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        bundle_path = os.path.join(path, manifest["fs_bundle"])
        cbsnap_path = os.path.join(path, manifest["vec_graph_cbsnap"])
        sha = manifest["git_sha"]

        if not _is_git_repo(fs.base_path):
            raise RuntimeError(
                "import_bundle: target FS is not a git repo; call reset() "
                "with enable_git=True first."
            )
        rc, _, err = _git(
            fs.base_path, "fetch", bundle_path, "refs/*:refs/rl_env/imported/*",
        )
        if rc != 0:
            raise RuntimeError(
                f"import_bundle: git fetch from bundle failed: {err.strip()}"
            )
        rc, out, err = _git(fs.base_path, "cat-file", "-t", sha)
        if rc != 0 or out.strip() != "commit":
            raise RuntimeError(
                f"import_bundle: commit {sha[:10]} missing after fetch "
                f"(rc={rc}, err={err.strip()})"
            )

        from llm_gateway.rl.rl_env.snapshot import EncodedSnapshotBackend

        encoded = load_encoded(cbsnap_path)
        inner_snap = Snapshot(
            snapshot_id=encoded.snapshot_id,
            backend_name=EncodedSnapshotBackend.name,
            payload={"encoded": encoded},
            meta=manifest.get("meta", {}),
        )
        return Snapshot(
            snapshot_id=manifest["snapshot_id"],
            backend_name=self.name,
            payload={
                "git_sha": sha,
                "use_git": True,
                "inner_snapshot": inner_snap,
                "fs_base_path_at_capture": fs.base_path,
                "_inner_backend_override": EncodedSnapshotBackend.name,
            },
            meta=manifest.get("meta", {}),
        )


__all__ = ["GitFsSnapshotBackend"]
