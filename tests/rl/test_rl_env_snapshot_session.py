from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from llm_gateway.rl.rl_env import FilesystemSnapshotBackend, MemoryEnv, SnapshotSession  # noqa: E402


class _StubLLM:
    async def generate(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError("stub LLM should not be invoked")


class _StubEmbedder:
    async def embed(self, texts):
        if isinstance(texts, str):
            return [0.1, 0.2, 0.3, 0.4]
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    async def embed_single(self, text):
        return [0.1, 0.2, 0.3, 0.4]


def test_snapshot_session_loads_cbsnap_dataset_layout():
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            data_root = os.path.join(tmp, "dataset")
            traj_id = "traj_test"
            snap_dir = os.path.join(data_root, "snapshots", traj_id)
            os.makedirs(snap_dir, exist_ok=True)

            producer = MemoryEnv(
                llm=_StubLLM(),
                embedder=_StubEmbedder(),
                base_dir=os.path.join(tmp, "producer_fs"),
                backend="memory",
                enable_git=False,
                snapshot_backend=FilesystemSnapshotBackend(snap_dir, fs_mode="tar_zst"),
                task_version="atomic_code_t2",
            )
            await producer.reset(user_id=traj_id)
            producer.fs.write_file("people/alex/preferences.md", "Alex likes handmade pasta")
            snap = producer.snapshot(meta={"trajectory_id": traj_id})

            with open(os.path.join(data_root, "ingest_snapshots.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"snapshot_id": snap.snapshot_id, "trajectory_id": traj_id}) + "\n")

            session = SnapshotSession(data_root=data_root, task_version="atomic_code_t2")
            loaded = session.load(snapshot_id=snap.snapshot_id, llm=_StubLLM(), embedder=_StubEmbedder())

            try:
                assert loaded.traj_id == traj_id
                assert loaded.env.task_version == "atomic_code_t2"
                assert loaded.env.triad.retrieve.task_name == "retrieve_context_atomic_code_t2"
                assert loaded.fs.read_file("people/alex/preferences.md") == "Alex likes handmade pasta"
                assert session.get_traj_id(snap.snapshot_id) == traj_id

                loaded2 = session.load(snapshot_id=snap.snapshot_id, llm=_StubLLM(), embedder=_StubEmbedder())
                try:
                    assert loaded2.env_dir != loaded.env_dir
                    loaded.fs.write_file("people/alex/preferences.md", "mutated in rollout A")
                    assert loaded2.fs.read_file("people/alex/preferences.md") == "Alex likes handmade pasta"
                finally:
                    session.release(loaded2)
            finally:
                session.release(loaded)
                session.cleanup_all()
                producer.close()

    asyncio.run(run())
