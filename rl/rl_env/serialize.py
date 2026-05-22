"""Efficient serialization primitives for :mod:`llm_gateway.rl.rl_env.snapshot`.

Goals
-----
- **Small on disk.** Embeddings dominate snapshot size (each vec entry
  carries a ``dim``-sized ``list[float]``). We pack them as
  ``numpy.float32`` into a single buffer per collection, then zstd the
  lot. Typical compression: ~5-10x vs pickled deepcopies.
- **Fast.** numpy frombuffer / zstd one-shot compress. No per-entry
  Python object dance.
- **Portable.** Only JSON + numpy arrays + raw FS tarball go to disk.
  No ``pickle`` anywhere (cross-Python-version / security friendly).
- **Deduplicated.** Optional content-addressable storage (CAS): FS
  files are sha256-keyed so multiple snapshots of the same user share
  unchanged blobs (critical for branching rollouts where ~95% of FS
  content is identical between siblings).

Layout of an encoded snapshot (versioned)::

    {
        "version": 1,
        "snapshot_id": "<uuid>",
        "meta": {...},                  # caller metadata
        "fs": {...},                    # see _encode_fs
        "vec": {
            "id_counter": int,
            "collections": {
                "<name>": {
                    "entries": [{"id", "text", "metadata"}],
                    "dim": int,
                    "embeddings": <numpy bytes>,
                    "has_embedding": <numpy bytes>,
                },
                ...
            }
        },
        "graph": {
            "nodes": {...},
            "edges": [...],
            "edge_counter": int,
            "node_embeddings": {...},   # optional, since llm_gateway>=v?
            "edge_embeddings": {...},
        }
    }
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import stat
import tarfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import orjson
import zstandard as zstd

if TYPE_CHECKING:
    from storage.file_system_store import FileSystemStore
    from storage.graph_stores import GraphStore
    from storage.vector_stores import VectorStore

logger = logging.getLogger(__name__)

SERIALIZE_VERSION = 1
_BYTES_SENTINEL = "__b64__"
_DEFAULT_ZSTD_LEVEL = 3


# ---------------------------------------------------------------------------
# Public dataclass — what callers see after encoding
# ---------------------------------------------------------------------------


@dataclass
class EncodedSnapshot:
    """A fully serialized snapshot, ready to persist or ship over the wire."""

    version: int
    snapshot_id: str
    blob: bytes
    fs_mode: str  # "tar_zst" | "cas" | "git_external"
    fs_blobs: dict[str, bytes] = field(default_factory=dict)
    size_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.size_bytes:
            self.size_bytes = len(self.blob) + sum(len(v) for v in self.fs_blobs.values())


# ---------------------------------------------------------------------------
# Encode / decode API
# ---------------------------------------------------------------------------


def encode_snapshot(
    snapshot_id: str,
    meta: dict[str, Any],
    fs: "FileSystemStore",
    vec: "VectorStore",
    graph: "GraphStore",
    *,
    fs_mode: str = "tar_zst",
    cas: "CASBlobStore | None" = None,
    zstd_level: int = _DEFAULT_ZSTD_LEVEL,
) -> EncodedSnapshot:
    """Capture live state and return a portable :class:`EncodedSnapshot`."""
    if fs_mode not in {"tar_zst", "cas"}:
        raise ValueError(f"invalid fs_mode={fs_mode!r}")
    if fs_mode == "cas" and cas is None:
        raise ValueError("fs_mode='cas' requires a CASBlobStore")

    fs_payload, fs_blobs = _encode_fs(fs, mode=fs_mode, cas=cas, zstd_level=zstd_level)
    vec_payload = _encode_vec(vec)
    graph_payload = _encode_graph(graph)

    doc: dict[str, Any] = {
        "version": SERIALIZE_VERSION,
        "snapshot_id": snapshot_id,
        "meta": meta,
        "fs": fs_payload,
        "vec": vec_payload,
        "graph": graph_payload,
    }
    raw_json = _dumps_json_with_bytes(doc)
    blob = _zstd_compress(raw_json, level=zstd_level)

    return EncodedSnapshot(
        version=SERIALIZE_VERSION,
        snapshot_id=snapshot_id,
        blob=blob,
        fs_mode=fs_mode,
        fs_blobs=fs_blobs if fs_mode == "cas" else {},
    )


def decode_snapshot(
    encoded: EncodedSnapshot,
    fs: "FileSystemStore",
    vec: "VectorStore",
    graph: "GraphStore",
    *,
    cas: "CASBlobStore | None" = None,
) -> dict[str, Any]:
    """Rewind ``fs / vec / graph`` to the state captured in ``encoded``."""
    raw_json = _zstd_decompress(encoded.blob)
    doc = _loads_json_with_bytes(raw_json)
    if doc.get("version") != SERIALIZE_VERSION:
        raise ValueError(f"unsupported snapshot version: {doc.get('version')}")

    _decode_fs(doc["fs"], fs, cas=cas, fallback_blobs=encoded.fs_blobs)
    _decode_vec(doc["vec"], vec)
    _decode_graph(doc["graph"], graph)
    return doc.get("meta", {})


# ---------------------------------------------------------------------------
# File-level dump / load helpers (single-file .cbsnap format)
# ---------------------------------------------------------------------------


_MAGIC = b"CBSNAP01"


def dump_encoded(encoded: EncodedSnapshot, path: str) -> None:
    """Write a self-contained ``.cbsnap`` file to ``path``.

    Format:
        [8 magic][1 fs_mode_flag][8 snap_id_len][snap_id]
        [8 blob_len][blob]
        [8 num_fs_blobs][ for each: 32 sha256 + 8 len + bytes ]

    fs_mode_flag legend:
        0 = tar_zst       (self-contained tar)
        1 = cas           (FS blobs live in an external CASBlobStore)
        2 = git_external  (FS lives in an external git bundle/repo)
    """
    snap_id_bytes = encoded.snapshot_id.encode("utf-8")
    fs_mode_flag = {"tar_zst": 0, "cas": 1, "git_external": 2}[encoded.fs_mode]

    with open(path, "wb") as f:
        f.write(_MAGIC)
        f.write(bytes([fs_mode_flag]))
        f.write(len(snap_id_bytes).to_bytes(8, "big"))
        f.write(snap_id_bytes)
        f.write(len(encoded.blob).to_bytes(8, "big"))
        f.write(encoded.blob)
        f.write(len(encoded.fs_blobs).to_bytes(8, "big"))
        for sha_hex, content in encoded.fs_blobs.items():
            sha_bytes = bytes.fromhex(sha_hex)
            if len(sha_bytes) != 32:
                raise ValueError(f"invalid sha256 length for {sha_hex}")
            f.write(sha_bytes)
            f.write(len(content).to_bytes(8, "big"))
            f.write(content)


def load_encoded(path: str) -> EncodedSnapshot:
    """Load a ``.cbsnap`` file produced by :func:`dump_encoded`."""
    with open(path, "rb") as f:
        if f.read(len(_MAGIC)) != _MAGIC:
            raise ValueError(f"not a .cbsnap file: {path}")
        fs_mode_flag = f.read(1)[0]
        snap_id_len = int.from_bytes(f.read(8), "big")
        snap_id = f.read(snap_id_len).decode("utf-8")
        blob_len = int.from_bytes(f.read(8), "big")
        blob = f.read(blob_len)
        num_blobs = int.from_bytes(f.read(8), "big")
        fs_blobs: dict[str, bytes] = {}
        for _ in range(num_blobs):
            sha = f.read(32).hex()
            size = int.from_bytes(f.read(8), "big")
            fs_blobs[sha] = f.read(size)

    return EncodedSnapshot(
        version=SERIALIZE_VERSION,
        snapshot_id=snap_id,
        blob=blob,
        fs_mode={0: "tar_zst", 1: "cas", 2: "git_external"}[fs_mode_flag],
        fs_blobs=fs_blobs,
    )


# ---------------------------------------------------------------------------
# FS encoding — two modes: tar_zst (self-contained) / cas (dedup-friendly)
# ---------------------------------------------------------------------------


def _encode_fs(
    fs: "FileSystemStore",
    *,
    mode: str,
    cas: "CASBlobStore | None",
    zstd_level: int,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    base = fs.base_path
    if not os.path.isdir(base):
        return {"mode": mode, "tree": [], "empty_dirs": []}, {}

    if mode == "tar_zst":
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            for root, dirs, files in os.walk(base):
                rel_root = os.path.relpath(root, base)
                if rel_root == ".":
                    rel_root = ""
                if not files and not dirs and rel_root:
                    tf.add(root, arcname=rel_root, recursive=False)
                for fn in files:
                    abs_p = os.path.join(root, fn)
                    arc_p = os.path.join(rel_root, fn) if rel_root else fn
                    tf.add(abs_p, arcname=arc_p, recursive=False)
        tar_bytes = buf.getvalue()
        return {"mode": "tar_zst", "tar_zst": _zstd_compress(tar_bytes, level=zstd_level)}, {}

    # mode == "cas"
    assert cas is not None
    tree: list[dict[str, Any]] = []
    empty_dirs: list[str] = []
    blobs: dict[str, bytes] = {}
    for root, dirs, files in os.walk(base):
        rel_root = os.path.relpath(root, base)
        if rel_root == ".":
            rel_root = ""
        if not files and not dirs and rel_root:
            empty_dirs.append(rel_root)
        for fn in files:
            abs_p = os.path.join(root, fn)
            rel_p = os.path.join(rel_root, fn) if rel_root else fn
            try:
                with open(abs_p, "rb") as f:
                    content = f.read()
            except (OSError, IOError) as e:  # pragma: no cover
                logger.warning("_encode_fs: skipping unreadable %s: %s", abs_p, e)
                continue
            sha = hashlib.sha256(content).hexdigest()
            st_mode = os.stat(abs_p).st_mode
            tree.append({
                "path": rel_p,
                "sha256": sha,
                "size": len(content),
                "mode": stat.S_IMODE(st_mode),
            })
            if not cas.has(sha):
                cas.put(sha, content)
            blobs[sha] = content

    return {"mode": "cas", "tree": tree, "empty_dirs": empty_dirs}, blobs


def _decode_fs(
    payload: dict[str, Any],
    fs: "FileSystemStore",
    *,
    cas: "CASBlobStore | None",
    fallback_blobs: dict[str, bytes] | None = None,
) -> None:
    base = fs.base_path
    if os.path.isdir(base):
        for entry in os.listdir(base):
            p = os.path.join(base, entry)
            if os.path.isdir(p) and not os.path.islink(p):
                _rmtree(p)
            else:
                os.remove(p)
    else:
        os.makedirs(base, exist_ok=True)

    mode = payload.get("mode", "tar_zst")
    if mode == "tar_zst":
        tar_bytes = _zstd_decompress(payload["tar_zst"])
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tf:
            try:
                tf.extractall(path=base, filter="data")
            except TypeError:  # pragma: no cover - py<3.12
                tf.extractall(path=base)
        return

    # mode == "cas"
    fallback_blobs = fallback_blobs or {}
    for rel_dir in payload.get("empty_dirs", []):
        os.makedirs(os.path.join(base, rel_dir), exist_ok=True)
    for node in payload.get("tree", []):
        rel_p = node["path"]
        sha = node["sha256"]
        dst = os.path.join(base, rel_p)
        os.makedirs(os.path.dirname(dst) or base, exist_ok=True)
        content: bytes | None = None
        if cas is not None and cas.has(sha):
            content = cas.get(sha)
        if content is None:
            content = fallback_blobs.get(sha)
        if content is None:
            raise FileNotFoundError(
                f"CAS blob {sha[:12]} not found for {rel_p}; "
                f"attach a CASBlobStore or use a self-contained .cbsnap"
            )
        with open(dst, "wb") as f:
            f.write(content)
        try:
            os.chmod(dst, node.get("mode", 0o644))
        except OSError:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# Vec encoding — split embeddings into float32 numpy blob
# ---------------------------------------------------------------------------


def _encode_vec(vec: "VectorStore") -> dict[str, Any]:
    collections: dict[str, Any] = {}
    for name, entries in vec._collections.items():
        # find first non-empty embedding to infer dim
        dim = 0
        for e in entries:
            emb = e.get("embedding") or []
            if emb:
                dim = len(emb)
                break

        if dim > 0 and entries:
            arr = np.zeros((len(entries), dim), dtype=np.float32)
            has_emb = np.zeros(len(entries), dtype=np.uint8)
            for i, e in enumerate(entries):
                emb = e.get("embedding") or []
                if emb and len(emb) == dim:
                    arr[i] = np.asarray(emb, dtype=np.float32)
                    has_emb[i] = 1
            embeddings_blob = _ndarray_to_bytes(arr)
            has_emb_blob = _ndarray_to_bytes(has_emb)
        else:
            embeddings_blob = b""
            has_emb_blob = b""

        stripped = [
            {"id": e.get("id"), "text": e.get("text", ""), "metadata": e.get("metadata", {})}
            for e in entries
        ]
        collections[name] = {
            "entries": stripped,
            "dim": dim,
            "embeddings": embeddings_blob,
            "has_embedding": has_emb_blob,
        }

    return {
        "id_counter": vec._id_counter,
        "collections": collections,
    }


def _decode_vec(payload: dict[str, Any], vec: "VectorStore") -> None:
    collections: dict[str, list[dict[str, Any]]] = {}
    for name, c in payload["collections"].items():
        entries_meta = c["entries"]
        dim = c.get("dim", 0)
        arr: np.ndarray | None = None
        has_emb: np.ndarray | None = None
        if dim > 0 and entries_meta and c.get("embeddings"):
            arr = _ndarray_from_bytes(c["embeddings"])
            if c.get("has_embedding"):
                has_emb = _ndarray_from_bytes(c["has_embedding"])

        rebuilt: list[dict[str, Any]] = []
        for i, em in enumerate(entries_meta):
            emb: list[float] = []
            if arr is not None and i < arr.shape[0]:
                if has_emb is None or has_emb[i]:
                    emb = arr[i].tolist()
            rebuilt.append({
                "id": em["id"],
                "text": em.get("text", ""),
                "embedding": emb,
                "metadata": em.get("metadata", {}),
            })
        collections[name] = rebuilt

    vec._collections = collections
    vec._id_counter = int(payload.get("id_counter", 0))
    if hasattr(vec, "_last_add_warnings"):
        vec._last_add_warnings = []


# ---------------------------------------------------------------------------
# Graph encoding — plain JSON + node/edge embeddings
# ---------------------------------------------------------------------------


def _encode_graph(graph: "GraphStore") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "nodes": graph._nodes,
        "edges": graph._edges,
        "edge_counter": graph._edge_counter,
    }
    # llm_gateway storage 比老 agent_memory 多了 node/edge embedding，
    # 缺失时静默忽略以兼容老 .cbsnap 解码。
    node_embs = getattr(graph, "_node_embeddings", None)
    edge_embs = getattr(graph, "_edge_embeddings", None)
    if node_embs is not None:
        payload["node_embeddings"] = node_embs
    if edge_embs is not None:
        payload["edge_embeddings"] = edge_embs
    return payload


def _decode_graph(payload: dict[str, Any], graph: "GraphStore") -> None:
    graph._nodes = dict(payload.get("nodes", {}))
    graph._edges = list(payload.get("edges", []))
    graph._edge_counter = int(payload.get("edge_counter", 0))
    if hasattr(graph, "_node_embeddings"):
        graph._node_embeddings = dict(payload.get("node_embeddings", {}))
    if hasattr(graph, "_edge_embeddings"):
        graph._edge_embeddings = dict(payload.get("edge_embeddings", {}))


# ---------------------------------------------------------------------------
# CAS (content-addressable blob store) — optional
# ---------------------------------------------------------------------------


class CASBlobStore:
    """Minimal on-disk content-addressable store for FS file blobs."""

    def __init__(self, root: str):
        os.makedirs(root, exist_ok=True)
        self.root = root

    def _path(self, sha: str) -> str:
        return os.path.join(self.root, sha[:2], sha[2:])

    def has(self, sha: str) -> bool:
        return os.path.isfile(self._path(sha))

    def put(self, sha: str, content: bytes) -> None:
        p = self._path(sha)
        if os.path.isfile(p):
            return
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "wb") as f:
            f.write(content)
        os.replace(tmp, p)

    def get(self, sha: str) -> bytes | None:
        p = self._path(sha)
        if not os.path.isfile(p):
            return None
        with open(p, "rb") as f:
            return f.read()

    def size(self) -> int:
        total = 0
        for root, _dirs, files in os.walk(self.root):
            for fn in files:
                try:
                    total += os.path.getsize(os.path.join(root, fn))
                except OSError:  # pragma: no cover
                    pass
        return total


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _zstd_compress(data: bytes, *, level: int = _DEFAULT_ZSTD_LEVEL) -> bytes:
    return zstd.ZstdCompressor(level=level).compress(data)


def _zstd_decompress(data: bytes) -> bytes:
    dctx = zstd.ZstdDecompressor()
    return dctx.decompress(data)


def _ndarray_to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _ndarray_from_bytes(b: bytes) -> np.ndarray:
    return np.load(io.BytesIO(b), allow_pickle=False)


# --- JSON with embedded bytes -------------------------------------------


def _dumps_json_with_bytes(obj: Any) -> bytes:
    return orjson.dumps(_encode_bytes(obj))


def _loads_json_with_bytes(data: bytes) -> Any:
    return _decode_bytes(orjson.loads(data))


def _encode_bytes(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return {_BYTES_SENTINEL: base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, dict):
        return {k: _encode_bytes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_encode_bytes(v) for v in obj]
    if isinstance(obj, tuple):
        return [_encode_bytes(v) for v in obj]
    return obj


def _decode_bytes(obj: Any) -> Any:
    if isinstance(obj, dict):
        if _BYTES_SENTINEL in obj and len(obj) == 1:
            return base64.b64decode(obj[_BYTES_SENTINEL])
        return {k: _decode_bytes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_bytes(v) for v in obj]
    return obj


def _rmtree(path: str) -> None:
    """Stripped-down rmtree we can swap for ``shutil.rmtree`` if needed."""
    import shutil as _shutil
    _shutil.rmtree(path, ignore_errors=True)
