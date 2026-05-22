"""Smoke test for rl_env.MemoryEnv snapshot / restore semantics.

We do not exercise a real LLM here — the test focuses on the env's
state-management contract:

1. reset() builds live stores + tasks.
2. snapshot() captures the full (FS + vec + graph) state.
3. Manual mutations to all three stores are visible.
4. restore() rewinds all three stores to the captured state.
5. Snapshots taken at different moments are independent (restoring
   one does not affect the other).
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

import pytest

# Ensure src/ is on sys.path (mirrors the project's pytest conftest style).
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from llm_gateway.rl.rl_env import MemoryEnv  # noqa: E402


class _StubEmbedder:
    """Minimal embedding stub matching VectorStore's usage (batch ``embed``
    and ``embed_single``)."""

    def _one(self, text: str) -> list[float]:
        return [float((hash(text) >> (8 * i)) & 0xFF) / 255.0 for i in range(4)]

    async def embed(self, texts):
        if isinstance(texts, str):
            return self._one(texts)
        return [self._one(t) for t in texts]

    async def embed_single(self, text):
        return self._one(text)


class _StubLLM:
    """LLM stub — never actually called in this test."""

    async def generate(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError("stub LLM should not be invoked")


@pytest.mark.asyncio
async def test_snapshot_restore_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        env = MemoryEnv(
            llm=_StubLLM(),
            embedder=_StubEmbedder(),
            base_dir=tmp,
            backend="memory",
            enable_git=False,  # faster; git not needed for this assertion
        )
        await env.reset(user_id="u1", namespace="rl__u1")

        # --- mutate all three stores directly (bypassing the agent) ---
        fs = env.fs
        vec = env.vec
        graph = env.graph
        assert fs is not None and vec is not None and graph is not None

        fs.write_file("notes/a.md", "hello A")
        vec.create_collection("facts")
        await vec.add("facts", ["fact one"], [{"source": "test"}])
        graph.add_node("user:u1", label="User", properties={"name": "u1"})
        graph.add_node("topic:rust", label="Topic")
        graph.add_edge("user:u1", "topic:rust", relation="learning")

        snap0 = env.snapshot(meta={"label": "after_initial_mutations"})

        # FS observation right after snap0
        files_at_snap0 = set(fs.list_files())
        vec_entries_at_snap0 = len(vec._collections["facts"])
        graph_nodes_at_snap0 = dict(graph._nodes)
        graph_edges_at_snap0 = list(graph._edges)

        # --- further mutations that should disappear after restore ---
        fs.write_file("notes/b.md", "hello B")
        await vec.add("facts", ["fact two"], [{"source": "test2"}])
        graph.add_node("topic:python", label="Topic")
        graph.add_edge("user:u1", "topic:python", relation="dislikes")

        assert "notes/b.md" in set(fs.list_files())
        assert len(vec._collections["facts"]) == vec_entries_at_snap0 + 1
        assert "topic:python" in graph._nodes
        assert len(graph._edges) == len(graph_edges_at_snap0) + 1

        # --- restore ---
        env.restore(snap0)

        assert set(fs.list_files()) == files_at_snap0
        assert len(vec._collections["facts"]) == vec_entries_at_snap0
        assert set(graph._nodes.keys()) == set(graph_nodes_at_snap0.keys())
        assert len(graph._edges) == len(graph_edges_at_snap0)

        # --- a second snapshot/restore cycle stays independent ---
        fs.write_file("notes/c.md", "hello C")
        snap1 = env.snapshot(meta={"label": "after_c"})

        # go back further to snap0
        env.restore(snap0)
        assert "notes/c.md" not in set(fs.list_files())

        # then forward to snap1
        env.restore(snap1)
        assert "notes/c.md" in set(fs.list_files())

        env.close()


@pytest.mark.asyncio
async def test_reset_wipes_previous_state():
    with tempfile.TemporaryDirectory() as tmp:
        env = MemoryEnv(
            llm=_StubLLM(),
            embedder=_StubEmbedder(),
            base_dir=tmp,
            backend="memory",
            enable_git=False,
        )
        await env.reset(user_id="u1")
        env.fs.write_file("dirty.md", "leftover")
        assert "dirty.md" in set(env.fs.list_files())

        await env.reset(user_id="u2")
        assert "dirty.md" not in set(env.fs.list_files())
        assert env.user_id == "u2"
        env.close()


def test_observe_shape_before_reset_raises():
    env = MemoryEnv(llm=_StubLLM(), embedder=_StubEmbedder())
    with pytest.raises(RuntimeError):
        env.observe()
    env.close()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(test_snapshot_restore_round_trip())
    asyncio.run(test_reset_wipes_previous_state())
    print("OK")
