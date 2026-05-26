"""Dataset snapshot loader for restoring ``.cbsnap`` files into ``MemoryEnv``.

Expected dataset layout::

    data_root/
    ├── ingest_snapshots.jsonl
    └── snapshots/{traj_id}/{snapshot_id}.cbsnap

This module is intentionally generic and lives in
:mod:`llm_gateway.rl.rl_env` so training packages can reuse the same
lifecycle/cache wrapper instead of reaching into
:mod:`llm_gateway.rl.rl_env.serialize` directly.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import shutil
import threading
import uuid
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

from llm_gateway.rl.rl_env.env import MemoryEnv
from llm_gateway.rl.rl_env.serialize import decode_snapshot, dump_encoded, encode_snapshot, load_encoded
from llm_gateway.rl.rl_env.snapshot import InMemorySnapshotBackend

if TYPE_CHECKING:
    from storage.file_system_store import FileSystemStore
    from storage.graph_stores import GraphStore
    from storage.vector_stores import VectorStore

logger = logging.getLogger(__name__)


class SnapshotSession:
    """Manage loading and releasing dataset snapshots as ``MemoryEnv`` objects."""

    def __init__(
        self,
        data_root: str,
        tmp_base: str | None = None,
        *,
        task_version: str = "atomic_code_t2",
        backend: str | None = "memory",
        enable_git: bool = False,
    ):
        """
        Args:
            data_root: Dataset root containing ``snapshots/`` and
                ``ingest_snapshots.jsonl``.
            tmp_base: Parent directory for restored env workdirs. Defaults to
                ``data_root/_env_tmp``.
            task_version: Task implementation family to bind in ``MemoryEnv``.
                ``"atomic_code_t2"`` (默认) / ``"code_t2"`` /
                ``"multi_code_t2"`` / ``"t2"``。
            backend: Store backend passed to ``MemoryEnv`` (``"memory"`` /
                ``"pg"``)。
            enable_git: Whether restored env FS should initialise git.
        """
        self.data_root = data_root
        env_tmp_base = (
            os.environ.get("MEMORY_RL_ENV_TMP_BASE")
            or os.environ.get("MEMORY_RL_ENV_BASE_DIR")
            or os.environ.get("RL_ENV_BASE_DIR")
        )
        self.tmp_base = tmp_base or env_tmp_base or os.path.join(data_root, "_env_tmp")
        self.task_version = task_version
        self.backend = backend
        self.enable_git = enable_git
        os.makedirs(self.tmp_base, exist_ok=True)

        self._snapshot_index: dict[str, dict] | None = None
        self._active_dirs: list[str] = []
        self._lock = threading.Lock()

    def _load_snapshot_index(self) -> dict[str, dict]:
        """Load and cache ``snapshot_id -> record`` from ``ingest_snapshots.jsonl``."""
        if self._snapshot_index is not None:
            return self._snapshot_index

        index_path = os.path.join(self.data_root, "ingest_snapshots.jsonl")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"找不到索引文件: {index_path}")

        self._snapshot_index = {}
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                sid = record.get("snapshot_id", "")
                if sid:
                    self._snapshot_index[sid] = record
        logger.info("加载了 %d 条快照索引", len(self._snapshot_index))
        return self._snapshot_index

    def get_traj_id(self, snapshot_id: str) -> str:
        """Return trajectory id for a snapshot id using the dataset index."""
        index = self._load_snapshot_index()
        record = index.get(snapshot_id)
        if record is None:
            raise KeyError(f"索引中找不到 snapshot_id={snapshot_id}")
        traj_id = record.get("trajectory_id", "")
        if not traj_id:
            raise ValueError(f"snapshot_id={snapshot_id} 没有 trajectory_id")
        return traj_id

    def load(
        self,
        traj_id: str | None = None,
        snapshot_id: str = "",
        *,
        llm: Any = None,
        embedder: Any = None,
    ) -> "LoadedEnv":
        """Load one ``.cbsnap`` into a fresh ``MemoryEnv``.

        If ``traj_id`` is omitted, it is resolved from ``ingest_snapshots.jsonl``.
        """
        if not snapshot_id:
            raise ValueError("必须提供 snapshot_id")

        if not traj_id:
            traj_id = self.get_traj_id(snapshot_id)

        if llm is None:
            llm = _build_default_llm_from_env()

        if embedder is None:
            embedder = _build_default_embedder()

        cbsnap_path = os.path.join(
            self.data_root, "snapshots", traj_id, f"{snapshot_id}.cbsnap"
        )
        if not os.path.exists(cbsnap_path):
            raise FileNotFoundError(f".cbsnap 文件不存在: {cbsnap_path}")

        env_dir = self._get_or_create_env_dir(traj_id, snapshot_id)
        env = MemoryEnv(
            llm=llm,
            embedder=embedder,
            base_dir=env_dir,
            backend=self.backend,
            enable_git=self.enable_git,
            snapshot_backend=InMemorySnapshotBackend(),
            task_version=self.task_version,
        )

        def _sync_reset() -> None:
            asyncio.run(env.reset(user_id=traj_id, wipe_base_dir=True))

        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                pool.submit(_sync_reset).result(timeout=30)
        except RuntimeError:
            asyncio.run(env.reset(user_id=traj_id, wipe_base_dir=True))

        try:
            encoded = load_encoded(cbsnap_path)
            decode_snapshot(encoded, env.fs, env.vec, env.graph)
        except Exception as e:
            shutil.rmtree(env_dir, ignore_errors=True)
            raise RuntimeError(f"从 .cbsnap 恢复失败: {e}") from e

        return LoadedEnv(
            env=env,
            env_dir=env_dir,
            traj_id=traj_id,
            snapshot_id=snapshot_id,
            session=self,
        )

    def save_env_snapshot(
        self,
        env: MemoryEnv,
        *,
        traj_id: str,
        source_snapshot_id: str = "",
        snapshot_id: str | None = None,
        subdir: str = "rollout_snapshots",
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist the current live ``MemoryEnv`` state as a standalone ``.cbsnap``.

        This is used by agentic rollout: the policy mutates a restored env in the
        rollout worker, then reward workers load this exact post-action state for
        probe evaluation instead of replaying lossy tool traces.
        """
        env._require_reset()
        if env.fs is None or env.vec is None or env.graph is None:
            raise RuntimeError("MemoryEnv stores are not initialized")

        snap_id = snapshot_id or uuid.uuid4().hex
        safe_traj = (traj_id or "unknown_traj").replace(os.sep, "_")
        root_dir = subdir if os.path.isabs(subdir) else os.path.join(self.data_root, subdir)
        out_dir = os.path.join(root_dir, safe_traj)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{snap_id}.cbsnap")
        encoded = encode_snapshot(
            snap_id,
            meta={
                "trajectory_id": traj_id,
                "source_snapshot_id": source_snapshot_id,
                "task_version": self.task_version,
                **(meta or {}),
            },
            fs=env.fs,
            vec=env.vec,
            graph=env.graph,
            fs_mode="tar_zst",
        )
        tmp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
        dump_encoded(encoded, tmp_path)
        os.replace(tmp_path, path)
        return {
            "snapshot_id": snap_id,
            "snapshot_path": path,
            "traj_id": traj_id,
            "source_snapshot_id": source_snapshot_id,
        }

    def load_path(
        self,
        snapshot_path: str,
        *,
        traj_id: str | None = None,
        snapshot_id: str | None = None,
        llm: Any = None,
        embedder: Any = None,
    ) -> "LoadedEnv":
        """Load a standalone ``.cbsnap`` path into a fresh isolated ``MemoryEnv``."""
        if not snapshot_path:
            raise ValueError("必须提供 snapshot_path")
        if not os.path.exists(snapshot_path):
            raise FileNotFoundError(f".cbsnap 文件不存在: {snapshot_path}")
        encoded = load_encoded(snapshot_path)
        sid = snapshot_id or encoded.snapshot_id
        tid = traj_id or "rollout_post"

        if llm is None:
            llm = _build_default_llm_from_env()

        if embedder is None:
            embedder = _build_default_embedder()

        env_dir = self._get_or_create_env_dir(tid, sid)
        env = MemoryEnv(
            llm=llm,
            embedder=embedder,
            base_dir=env_dir,
            backend=self.backend,
            enable_git=self.enable_git,
            snapshot_backend=InMemorySnapshotBackend(),
            task_version=self.task_version,
        )

        def _sync_reset() -> None:
            asyncio.run(env.reset(user_id=tid, wipe_base_dir=True))

        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                pool.submit(_sync_reset).result(timeout=30)
        except RuntimeError:
            asyncio.run(env.reset(user_id=tid, wipe_base_dir=True))

        try:
            decode_snapshot(encoded, env.fs, env.vec, env.graph)
        except Exception as e:
            shutil.rmtree(env_dir, ignore_errors=True)
            raise RuntimeError(f"从 .cbsnap 恢复失败: {e}") from e

        return LoadedEnv(
            env=env,
            env_dir=env_dir,
            traj_id=tid,
            snapshot_id=sid,
            session=self,
        )

    def _get_or_create_env_dir(self, traj_id: str, snapshot_id: str) -> str:
        safe_traj = traj_id.replace(os.sep, "_")
        dir_name = f"{safe_traj}_{snapshot_id[:12]}_{uuid.uuid4().hex[:12]}"
        env_dir = os.path.join(self.tmp_base, dir_name)

        with self._lock:
            os.makedirs(env_dir, exist_ok=False)
            self._active_dirs.append(env_dir)
            logger.debug("创建独立 env_dir: %s", env_dir)
            return env_dir

    def release(self, loaded: "LoadedEnv") -> None:
        """Release a loaded env and clean its isolated workdir."""
        if loaded._released:
            return

        loaded.env._owns_base_dir = False
        loaded.env.close()
        loaded._released = True

        with self._lock:
            env_dir = loaded.env_dir
            shutil.rmtree(env_dir, ignore_errors=True)
            self._active_dirs = [d for d in self._active_dirs if d != env_dir]
            logger.debug("释放独立 env_dir: %s", env_dir)

        # 断开引用以加速 GC 回收内存
        loaded.env = None  # type: ignore[assignment]
        loaded._session = None  # type: ignore[assignment]

    def cleanup_all(self) -> None:
        """Best-effort cleanup for all active env workdirs."""
        for env_dir in list(self._active_dirs):
            if os.path.isdir(env_dir):
                shutil.rmtree(env_dir, ignore_errors=True)
        self._active_dirs = []

        if os.path.isdir(self.tmp_base):
            shutil.rmtree(self.tmp_base, ignore_errors=True)


def _build_default_llm_from_env() -> Any | None:
    """Build an OpenAI-compatible LLMInterface from RL smoke/train env vars."""
    api_url = (
        os.environ.get("MEMORY_RL_RETRIEVE_LLM_API_URL")
        or os.environ.get("RETRIEVE_LLM_API_URL")
        or os.environ.get("FROZEN_MODEL_URL")
        or os.environ.get("MEMORY_RL_LLM_API_URL")
        or os.environ.get("VLLM_API_URL")
        or os.environ.get("ROLLOUT_MODEL_URL")
    )
    model = (
        os.environ.get("MEMORY_RL_RETRIEVE_LLM_MODEL")
        or os.environ.get("RETRIEVE_LLM_MODEL")
        or os.environ.get("FROZEN_MODEL_NAME")
        or os.environ.get("MEMORY_RL_LLM_MODEL")
        or os.environ.get("VLLM_MODEL")
        or os.environ.get("ROLLOUT_MODEL_NAME")
    )
    if not api_url or not model:
        return None

    try:
        from utils.memory_llm_interface import LLMInterface
    except Exception as exc:  # pragma: no cover - defensive import guard
        logger.warning("无法导入 LLMInterface: %s", exc)
        return None

    return LLMInterface({
        "provider": "openai_compat",
        "model": model,
        "base_url": _openai_base_url(api_url),
        "api_key": os.environ.get("MEMORY_RL_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "none",
        "temperature": float(os.environ.get("MEMORY_RL_LLM_TEMPERATURE", "0")),
        "max_tokens": int(os.environ.get("MEMORY_RL_LLM_MAX_TOKENS", "65536")),
        "timeout": float(os.environ.get("MEMORY_RL_LLM_TIMEOUT", "300")),
        "max_retries": int(os.environ.get("MEMORY_RL_LLM_MAX_RETRIES", "1")),
    })


def _build_default_embedder() -> Any | None:
    """Build an EmbeddingInterface for RL env from env vars or config.yaml.

    优先级：
    1. 环境变量 MEMORY_RL_EMBEDDING_* 系列
    2. config.yaml 中的 memory_config.embedding 配置
    3. 硬编码的 bge-m3 默认配置（兜底）
    """
    try:
        from utils.memory_llm_interface import EmbeddingInterface
    except Exception as exc:  # pragma: no cover
        logger.warning("无法导入 EmbeddingInterface: %s", exc)
        return None

    # 优先从环境变量构造
    env_base_url = os.environ.get("MEMORY_RL_EMBEDDING_BASE_URL") or os.environ.get("EMBEDDING_BASE_URL")
    env_model = os.environ.get("MEMORY_RL_EMBEDDING_MODEL") or os.environ.get("EMBEDDING_MODEL")
    if env_base_url and env_model:
        config = {
            "provider": "openai_compat",
            "model": env_model,
            "base_url": env_base_url,
            "api_key": os.environ.get("MEMORY_RL_EMBEDDING_API_KEY") or os.environ.get("EMBEDDING_API_KEY") or "EMPTY",
            "batch_size": int(os.environ.get("MEMORY_RL_EMBEDDING_BATCH_SIZE", "64")),
            "dimensions": None,
            "timeout": float(os.environ.get("MEMORY_RL_EMBEDDING_TIMEOUT", "120")),
            "max_retries": int(os.environ.get("MEMORY_RL_EMBEDDING_MAX_RETRIES", "3")),
        }
        logger.info("_build_default_embedder: 从环境变量构造 (model=%s, base_url=%s)", env_model, env_base_url)
        return EmbeddingInterface(config)

    # 尝试从 config.yaml 读取
    try:
        from config.loader import get_config
        cfg = get_config()
        embedding_config = cfg.memory_config.embedding.model_dump()
        logger.info(
            "_build_default_embedder: 从 config.yaml 加载 (model=%s, base_url=%s)",
            embedding_config.get("model"), embedding_config.get("base_url"),
        )
        return EmbeddingInterface(embedding_config)
    except Exception:
        pass

    # 硬编码兜底（与 config.yaml 中当前配置保持一致）
    fallback_config = {
        "provider": "openai_compat",
        "model": "bge-m3",
        "api_key": "EMPTY",
        "base_url": "http://81.70.50.154:8082/v1",
        "batch_size": 64,
        "dimensions": None,
        "timeout": 120.0,
        "max_retries": 3,
    }
    logger.warning("_build_default_embedder: 使用硬编码兜底配置 (bge-m3)")
    return EmbeddingInterface(fallback_config)


def _openai_base_url(url: str) -> str:
    """Accept either /v1 or /v1/chat/completions and return the OpenAI base URL."""
    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    suffix = "/chat/completions"
    if path.endswith(suffix):
        path = path[: -len(suffix)] or "/"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


class LoadedEnv:
    """Wrapper around a restored ``MemoryEnv`` with context-manager cleanup."""

    def __init__(
        self,
        env: MemoryEnv,
        env_dir: str,
        traj_id: str,
        snapshot_id: str,
        session: SnapshotSession,
    ):
        self.env = env
        self.env_dir = env_dir
        self.traj_id = traj_id
        self.snapshot_id = snapshot_id
        self._session = session
        self._released = False

    def __enter__(self) -> "LoadedEnv":
        return self

    def __exit__(self, *exc) -> None:
        self._session.release(self)

    @property
    def fs(self) -> "FileSystemStore":
        return self.env.fs  # type: ignore[return-value]

    @property
    def vec(self) -> "VectorStore":
        return self.env.vec  # type: ignore[return-value]

    @property
    def graph(self) -> "GraphStore":
        return self.env.graph  # type: ignore[return-value]


__all__ = ["SnapshotSession", "LoadedEnv", "_build_default_embedder"]
