"""Core :class:`MemoryEnv` — RL-style sandbox around llm_gateway memory tasks.

老仓库 ``rl_env.env.MemoryEnv`` 直连 ``agent_memory.{core,tasks}``；本迁移版完全
对接 llm_gateway 自身：

- stores 由 ``llm_gateway.storage.{file_system_store,vector_stores,graph_stores}`` 直接构造
- LLM/Embedding 接口走 ``llm_gateway.utils.memory_llm_interface``
- task 实现通过 ``task_version`` + :mod:`llm_gateway.rl.rl_env.task_factory` 工厂化绑定
  到 ``atomic_code_t2`` / ``context_task`` 的 4 套 ingest/consolidate/retrieve

对外 API（``reset/step_ingest/step_consolidate/step_query/snapshot/restore/observe/close``）与
老版完全保持兼容；``apply_*_tool_calls`` 在 llm_gateway 当前阶段没有 T3 风格 task，
默认抛 NotImplementedError，等 T3 task 接入后再实现。
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from llm_gateway.rl.rl_env._models import Event, EventType, TaskResult, ToolCall
from llm_gateway.rl.rl_env.scorer import BaseScorer, ScoreResult
from llm_gateway.rl.rl_env.snapshot import (
    InMemorySnapshotBackend,
    Snapshot,
    SnapshotBackend,
)
from llm_gateway.rl.rl_env.task_factory import (
    TaskTriad,
    build_task_triad,
    dispatch_event,
)

if TYPE_CHECKING:
    from storage.file_system_store import FileSystemStore
    from storage.stores_base import GraphStoreBase, VectorStoreBase
    from utils.memory_llm_interface import EmbeddingInterface, LLMInterface

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# StepResult — what every step_* call returns
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Bundled output of one env step."""

    task_name: str
    task_result: TaskResult
    pre_snapshot: Snapshot | None = None
    post_snapshot: Snapshot | None = None
    score: ScoreResult | None = None
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MemoryEnv
# ---------------------------------------------------------------------------


_VALID_BACKENDS = {"memory", "pg"}


class MemoryEnv:
    """RL sandbox env exposing reset / snapshot / restore + 三个 step_* 方法。

    Args (constructor):
        llm: ``utils.memory_llm_interface.LLMInterface``。
        embedder: ``utils.memory_llm_interface.EmbeddingInterface``，默认 None
            （memory 后端 + 不写 vec 时可省）。
        base_dir: FS 根目录。omit 时自动建 ``/tmp/rl_env_<ts>_<rand>/``，
            ``close()`` 时自动清理。
        backend: ``"memory"`` (默认) / ``"pg"``。pg 模式下 vec/graph 接 PG。
        enable_git: 是否在 base_dir 上 ``git init``（默认 True，对应
            ``GitFsSnapshotBackend`` 的前置条件）。
        snapshot_backend: 自定义快照策略；默认 :class:`InMemorySnapshotBackend`。
        max_turns: 覆盖三件套 task 的 ``max_turns``（若它们有该属性）。
        task_version: ``"atomic_code_t2"`` (默认) / ``"code_t2"`` /
            ``"multi_code_t2"`` / ``"t2"``。
    """

    def __init__(
        self,
        *,
        llm: "LLMInterface",
        embedder: "EmbeddingInterface | None" = None,
        base_dir: str | None = None,
        backend: str | None = None,
        enable_git: bool = True,
        snapshot_backend: SnapshotBackend | None = None,
        max_turns: int | None = None,
        task_version: str = "atomic_code_t2",
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.backend = (backend or "memory").lower()
        if self.backend not in _VALID_BACKENDS:
            raise ValueError(
                f"unknown backend={backend!r}; expected one of {sorted(_VALID_BACKENDS)}"
            )
        self.enable_git = enable_git
        self.max_turns = max_turns
        self.task_version = task_version
        self.snapshot_backend: SnapshotBackend = (
            snapshot_backend or InMemorySnapshotBackend()
        )

        # base_dir 生命周期：env 内部建的 temp dir 在 close 时清理；外部指定的不动
        self._owns_base_dir = base_dir is None
        if base_dir is None:
            base_dir = tempfile.mkdtemp(
                prefix=f"rl_env_{datetime.now().strftime('%Y%m%d_%H%M%S')}_",
            )
        self.base_dir = base_dir

        # 由 reset() 填充
        self.id: str = ""
        self.user_id: str = ""
        self.namespace: str = ""
        self.fs: "FileSystemStore | None" = None
        self.vec: "VectorStoreBase | None" = None
        self.graph: "GraphStoreBase | None" = None
        self.triad: TaskTriad | None = None
        self.backend_info: dict[str, Any] = {}
        self._step_counter: int = 0
        self._managed_snapshots: list[Snapshot] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def reset(
        self,
        *,
        user_id: str,
        namespace: str | None = None,
        wipe_base_dir: bool = True,
    ) -> None:
        """重新初始化 stores 并绑定一组 fresh task 三件套。"""
        self.user_id = user_id
        self.namespace = namespace or f"rl__{user_id}"

        if wipe_base_dir and os.path.isdir(self.base_dir):
            shutil.rmtree(self.base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

        # 直接构造 stores —— 不调用 storage.stores_factory.make_stores，
        # 因为后者强制 fs_path = {root_dir}/{user_id}/memory_repo 且会缓存到
        # 全局 _memory_store_cache。RL 环境需要任意 base_dir 与隔离实例。
        from storage.file_system_store import FileSystemStore
        from storage.graph_stores import GraphStore
        from storage.vector_stores import VectorStore

        fs = FileSystemStore(self.base_dir, enable_git=self.enable_git)

        if self.backend == "pg":
            from storage.graph_stores import AgePgGraphStore
            from storage.pg_store_backend import PgBackend
            from storage.stores_factory import _user_id_to_schema
            from storage.vector_stores import PgVectorStore
            from config.loader import get_config

            cfg = get_config()
            pg = PgBackend.from_config(cfg.memory_config.storage.pg)
            pg.ensure_schema_sync()
            schema = _user_id_to_schema(user_id)
            pg.ensure_user_schema_sync(schema)
            vec: "VectorStoreBase" = PgVectorStore(pg, schema=schema, embedder=self.embedder)
            graph: "GraphStoreBase" = AgePgGraphStore(pg, schema=schema)
            info = {"backend": "pg", "schema": schema}
        else:
            vec = VectorStore(embedding_interface=self.embedder)
            graph = GraphStore(embedding_interface=self.embedder)
            info = {"backend": "memory"}

        self.fs, self.vec, self.graph = fs, vec, graph
        self.backend_info = info

        self.triad = build_task_triad(
            self.task_version,
            llm=self.llm,
            fs=fs,
            vec=vec,
            graph=graph,
            max_turns=self.max_turns,
        )

        self._step_counter = 0
        logger.info(
            "MemoryEnv.reset: user_id=%s namespace=%s backend=%s task_version=%s base_dir=%s",
            user_id, self.namespace, info.get("backend"), self.task_version, self.base_dir,
        )

    def close(self) -> None:
        """释放快照资源 + 清理自有的 temp base_dir。"""
        for snap in self._managed_snapshots:
            try:
                self.snapshot_backend.release(snap)
            except Exception as e:  # pragma: no cover - best effort
                logger.warning("close(): release snapshot failed: %s", e)
        self._managed_snapshots = []

        if self._owns_base_dir and os.path.isdir(self.base_dir):
            shutil.rmtree(self.base_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------

    def snapshot(self, *, meta: dict[str, Any] | None = None) -> Snapshot:
        self._require_reset()
        snap = self.snapshot_backend.capture(
            self.fs, self.vec, self.graph,  # type: ignore[arg-type]
            meta={
                "user_id": self.user_id,
                "namespace": self.namespace,
                "step_counter": self._step_counter,
                **(meta or {}),
            },
        )
        self._managed_snapshots.append(snap)
        return snap

    def restore(self, snapshot: Snapshot) -> None:
        self._require_reset()
        self.snapshot_backend.restore(
            snapshot,
            self.fs, self.vec, self.graph,  # type: ignore[arg-type]
        )
        self._step_counter = snapshot.meta.get("step_counter", 0)

    def release_snapshot(self, snapshot: Snapshot) -> None:
        try:
            self.snapshot_backend.release(snapshot)
        finally:
            self._managed_snapshots = [
                s for s in self._managed_snapshots
                if s.snapshot_id != snapshot.snapshot_id
            ]

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(self) -> dict[str, Any]:
        self._require_reset()
        fs_files = self.fs.list_files() if self.fs else []  # type: ignore[union-attr]
        vec_stats = self.vec.get_stats() if self.vec else {}  # type: ignore[union-attr]
        graph_stats = self.graph.get_stats() if self.graph else {}  # type: ignore[union-attr]
        return {
            "user_id": self.user_id,
            "namespace": self.namespace,
            "backend": self.backend_info.get("backend"),
            "task_version": self.task_version,
            "step_counter": self._step_counter,
            "fs": {
                "base_path": self.fs.base_path if self.fs else None,
                "num_files": len(fs_files),
                "files": fs_files[:50],
            },
            "vec": vec_stats,
            "graph": graph_stats,
        }

    # ------------------------------------------------------------------
    # Step methods
    # ------------------------------------------------------------------

    async def step_ingest(
        self,
        *,
        session_id: str,
        messages: list[dict[str, Any]] | None = None,
        content: str | None = None,
        extra_payload: dict[str, Any] | None = None,
        auto_snapshot: bool = False,
        scorer: BaseScorer | None = None,
        gold: Any = None,
    ) -> StepResult:
        """跑一次完整 ingest agent loop。

        与老版兼容：``messages`` / ``content`` 二选一；llm_gateway 的 task
        实际只接受 ``messages``，所以传入 ``content`` 时会 wrap 成单条
        user 消息。
        """
        self._require_reset()
        assert self.triad is not None

        if messages is not None and content is not None:
            raise ValueError("pass either `messages` or `content`, not both")
        if messages is None:
            if content is None:
                content = ""
            messages = [{"role": "user", "content": content}] if content else []

        payload: dict[str, Any] = {
            "user_id": self.user_id,
            "session_id": session_id,
            "messages": messages,
            "ingest_number": self._step_counter + 1,
            "message_count": len(messages),
        }
        if extra_payload:
            payload.update(extra_payload)
        event = Event(type=EventType.MESSAGE, payload=payload)

        return await self._step_generic(
            event=event,
            auto_snapshot=auto_snapshot,
            scorer=scorer,
            gold=gold,
            extras={"session_id": session_id, "phase": "ingest"},
        )

    async def step_consolidate(
        self,
        *,
        session_id: str = "",
        extra_payload: dict[str, Any] | None = None,
        auto_snapshot: bool = False,
        scorer: BaseScorer | None = None,
        gold: Any = None,
    ) -> StepResult:
        """跑一次完整 consolidate（演进）agent loop。"""
        self._require_reset()
        assert self.triad is not None

        payload: dict[str, Any] = {
            "user_id": self.user_id,
            "session_id": session_id,
        }
        if extra_payload:
            payload.update(extra_payload)
        event = Event(type=EventType.MEMORY_CONSOLIDATION, payload=payload)

        return await self._step_generic(
            event=event,
            auto_snapshot=auto_snapshot,
            scorer=scorer,
            gold=gold,
            extras={"session_id": session_id, "phase": "evolve"},
        )

    async def step_query(
        self,
        *,
        query: str,
        session_id: str = "",
        extra_payload: dict[str, Any] | None = None,
        auto_snapshot: bool = False,
        scorer: BaseScorer | None = None,
        gold: Any = None,
    ) -> StepResult:
        """跑一次完整 retrieve agent loop，得到一个 memory context。"""
        self._require_reset()
        assert self.triad is not None

        payload: dict[str, Any] = {
            "user_id": self.user_id,
            "session_id": session_id,
            "query": query,
        }
        if extra_payload:
            payload.update(extra_payload)
        event = Event(type=EventType.MEMORY_QUERY, payload=payload)

        return await self._step_generic(
            event=event,
            auto_snapshot=auto_snapshot,
            scorer=scorer,
            gold=gold,
            extras={"session_id": session_id, "phase": "consume", "query": query},
        )

    # ------------------------------------------------------------------
    # RL rollout apply_*_tool_calls
    # ------------------------------------------------------------------

    async def apply_ingest_tool_calls(
        self,
        tool_calls: list[Any],
        *,
        session_time: str = "",
        auto_snapshot: bool = False,
        scorer: BaseScorer | None = None,
        gold: Any = None,
        extras: dict[str, Any] | None = None,
    ) -> StepResult:
        """Apply policy-emitted memory write calls to this env.

        Two modes are supported:
        - ``MEMORY_RL_APPLY_MODE=task_loop``: ignore external calls and run the
          bound ingest task's own loop using ``extras['pending_messages']``.
        - default: execute normalized T3/atomic/t2-agent write tool calls against
          FS/Vec/Graph directly.
        """
        extras = extras or {}
        if _apply_mode() == "task_loop" and not tool_calls and not extras.get("replay_task_loop_response"):
            payload = {"session_time": session_time or extras.get("session_time", "")}
            return await self.step_ingest(
                session_id=str(extras.get("session_id", "")),
                messages=extras.get("pending_messages") or [],
                extra_payload=payload,
                auto_snapshot=auto_snapshot,
                scorer=scorer,
                gold=gold,
            )
        return await self._apply_memory_tool_calls(
            tool_calls,
            phase="ingest",
            session_time=session_time or str(extras.get("session_time", "")),
            auto_snapshot=auto_snapshot,
            scorer=scorer,
            gold=gold,
            extras=extras,
        )

    async def apply_consolidate_tool_calls(
        self,
        tool_calls: list[Any],
        *,
        auto_snapshot: bool = False,
        scorer: BaseScorer | None = None,
        gold: Any = None,
        extras: dict[str, Any] | None = None,
    ) -> StepResult:
        """Apply policy-emitted consolidation calls or run native task loop."""
        extras = extras or {}
        if _apply_mode() == "task_loop" and not tool_calls and not extras.get("replay_task_loop_response"):
            return await self.step_consolidate(
                session_id=str(extras.get("session_id", "")),
                auto_snapshot=auto_snapshot,
                scorer=scorer,
                gold=gold,
            )
        return await self._apply_memory_tool_calls(
            tool_calls,
            phase="consolidate",
            session_time=str(extras.get("session_time", "")),
            auto_snapshot=auto_snapshot,
            scorer=scorer,
            gold=gold,
            extras=extras,
        )

    async def retrieve_with_queries(
        self,
        queries: list[dict[str, str]],
        *,
        original_query: str | None = None,
        use_reranker: bool | None = None,
    ) -> str:
        """直接把已重写好的 queries 传给 retrieve task（跳过 LLM 重写）。

        要求 retrieve task 实现 ``retrieve_with_queries`` 方法；llm_gateway 中
        目前只有 RetrieveT2/T3Task 旧版实现了，atomic_code_t2 / context_task 没有。
        """
        self._require_reset()
        assert self.triad is not None
        retrieve_task = self.triad.retrieve
        if hasattr(retrieve_task, "retrieve_with_queries"):
            return await retrieve_task.retrieve_with_queries(  # type: ignore[attr-defined]
                queries,
                original_query=original_query,
                use_reranker=use_reranker,
            )

        query_text = original_query or _queries_to_text(queries)
        step = await self.step_query(query=query_text, session_id="rl_retrieve_with_queries")
        return step.task_result.final_output or step.task_result.finish_summary

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _step_generic(
        self,
        *,
        event: Event,
        auto_snapshot: bool,
        scorer: BaseScorer | None,
        gold: Any,
        extras: dict[str, Any],
    ) -> StepResult:
        assert self.triad is not None
        pre_snap = self.snapshot(meta={"point": "pre", **extras}) if auto_snapshot else None

        result: TaskResult = await dispatch_event(self.triad, event)

        self._step_counter += 1
        post_snap = self.snapshot(meta={"point": "post", **extras}) if auto_snapshot else None

        score: ScoreResult | None = None
        if scorer is not None:
            score = scorer.score(
                pre_snapshot=pre_snap,
                post_snapshot=post_snap,
                task_result=result,
                gold=gold,
                extras=extras,
            )

        return StepResult(
            task_name=result.task_name,
            task_result=result,
            pre_snapshot=pre_snap,
            post_snapshot=post_snap,
            score=score,
            extras=extras,
        )

    async def _apply_memory_tool_calls(
        self,
        tool_calls: list[Any],
        *,
        phase: str,
        session_time: str,
        auto_snapshot: bool,
        scorer: BaseScorer | None,
        gold: Any,
        extras: dict[str, Any],
    ) -> StepResult:
        self._require_reset()
        assert self.fs is not None and self.vec is not None and self.graph is not None

        pre_snap = self.snapshot(meta={"point": "pre", "phase": phase, **extras}) if auto_snapshot else None
        trace: list[dict[str, Any]] = []
        finish_summary = ""
        error_count = 0

        for i, raw in enumerate(tool_calls):
            try:
                call = _coerce_tool_call(raw, i)
                result = await _execute_memory_tool_call(
                    fs=self.fs,
                    vec=self.vec,
                    graph=self.graph,
                    call=call,
                    session_time=session_time,
                )
                if call.name == "finish":
                    finish_summary = str(call.arguments.get("summary", "")) or result
                trace.append({
                    "tool": call.name,
                    "arguments": call.arguments,
                    "result": result,
                    "is_error": result.startswith("ERROR:"),
                })
                if result.startswith("ERROR:"):
                    error_count += 1
            except Exception as exc:  # noqa: BLE001 - keep rollout robust
                error_count += 1
                trace.append({
                    "tool": getattr(raw, "name", ""),
                    "arguments": getattr(raw, "arguments", raw if isinstance(raw, dict) else {}),
                    "result": f"ERROR: {exc}",
                    "is_error": True,
                })

        self._step_counter += 1
        post_snap = self.snapshot(meta={"point": "post", "phase": phase, **extras}) if auto_snapshot else None
        task_result = TaskResult(
            task_name=f"{phase}_{self.task_version}_tool_apply",
            finish_summary=finish_summary,
            finish_reason="finish" if finish_summary else "applied",
            error="" if error_count == 0 else f"{error_count} tool calls failed",
            stats={
                "tool_calls_trace": trace,
                "tool_calls": len(trace),
                "tool_errors": error_count,
                "success": error_count == 0,
            },
        )

        score: ScoreResult | None = None
        if scorer is not None:
            score = scorer.score(
                pre_snapshot=pre_snap,
                post_snapshot=post_snap,
                task_result=task_result,
                gold=gold,
                extras=extras,
            )
        return StepResult(
            task_name=task_result.task_name,
            task_result=task_result,
            pre_snapshot=pre_snap,
            post_snapshot=post_snap,
            score=score,
            extras=extras,
        )

    def _require_reset(self) -> None:
        if self.fs is None or self.triad is None:
            raise RuntimeError("MemoryEnv: call reset() before using the env")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_tool_call(raw: Any, index: int = 0) -> ToolCall:
    """规范化 dict / OpenAI 风格 tool call payloads 到 :class:`ToolCall`。"""
    if isinstance(raw, ToolCall):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"tool call must be dict or ToolCall, got {type(raw)!r}")

    function = raw.get("function") if isinstance(raw.get("function"), dict) else {}
    name = raw.get("tool") or raw.get("name") or function.get("name")
    if not name:
        raise ValueError(f"tool call #{index} missing tool/name")

    arguments = raw.get("arguments", function.get("arguments", {}))
    if isinstance(arguments, str):
        import json as _json

        try:
            parsed = _json.loads(arguments)
        except _json.JSONDecodeError as exc:
            raise ValueError(f"tool call #{index} arguments is not JSON") from exc
        arguments = parsed
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise TypeError(f"tool call #{index} arguments must be an object")

    return ToolCall(
        id=str(raw.get("id") or f"rollout_call_{index}"),
        name=str(name),
        arguments=arguments,
        raw=raw,
    )


async def _execute_memory_tool_call(*, fs: Any, vec: Any, graph: Any, call: ToolCall, session_time: str) -> str:
    name = call.name
    args = call.arguments

    if name == "finish":
        return str(args.get("summary", args.get("changes_summary", "finished")))

    if name in {
        "fs_grep", "fs_bm25_search", "fs_read_file", "fs_read_lines", "fs_tree",
        "vec_search", "vec_search_all", "graph_search_nodes", "graph_entity_search",
        "vec_semantic_search", "fs_execute_bash",
    }:
        return f"SKIPPED: read/search tool {name} is not replayed during reward apply"

    if name in {"fs_write", "write_file"}:
        path = _safe_rel_path(fs.base_path, str(args.get("path", "memory/notes.md")))
        content = str(args.get("content", ""))
        action = str(args.get("action", "add_file" if name == "fs_write" else "write"))
        if action == "append":
            return fs.append_file(path, _ensure_leading_newline_if_needed(fs, path, content))
        if action.startswith("update:"):
            return _update_line(fs, path, action, content)
        if action == "update_meta":
            return _update_meta(fs, path, content or str(args.get("meta_json", "")))
        return fs.write_file(path, content)

    if name == "fs_append":
        path = _safe_rel_path(fs.base_path, str(args.get("path", "memory/notes.md")))
        return fs.append_file(path, _ensure_leading_newline_if_needed(fs, path, str(args.get("content", ""))))

    if name == "fs_update_line":
        path = _safe_rel_path(fs.base_path, str(args.get("path", "")))
        return _update_line(fs, path, f"update:{args.get('line_number', 0)}", str(args.get("new_content", "")))

    if name == "fs_update_meta":
        path = _safe_rel_path(fs.base_path, str(args.get("path", "")))
        return _update_meta(fs, path, str(args.get("meta_json", "")))

    if name == "edit_file":
        path = _safe_rel_path(fs.base_path, str(args.get("path", "")))
        old = str(args.get("old_string", ""))
        new = str(args.get("new_string", ""))
        if not old:
            return "ERROR: edit_file requires old_string"
        content = fs.read_file(path)
        if content.startswith("ERROR"):
            return content
        expected = int(args.get("expected_replacements", 1))
        count = content.count(old)
        if count != expected:
            return f"ERROR: expected {expected} replacements, found {count}"
        return fs.write_file(path, content.replace(old, new))

    if name == "fs_delete":
        path = _safe_rel_path(fs.base_path, str(args.get("path", "")))
        full = os.path.join(fs.base_path, path)
        if os.path.isdir(full):
            shutil.rmtree(full)
            return f"Deleted directory: {path}"
        if os.path.exists(full):
            os.remove(full)
            return f"Deleted file: {path}"
        return f"Not found: {path}"

    if name in {"vec_write", "vec_add"}:
        collection = str(args.get("collection") or "memory")
        if name == "vec_add" or isinstance(args.get("items"), list):
            items = args.get("items") or []
            texts = [str(item.get("text", "")) for item in items if isinstance(item, dict) and item.get("text")]
            metas = [_vec_metadata(item.get("metadata"), session_time, i + 1) for i, item in enumerate(items) if isinstance(item, dict) and item.get("text")]
        else:
            text = str(args.get("text", ""))
            if not text:
                return "ERROR: vec_write requires text"
            entry_type = str(args.get("entry_type") or args.get("type") or "fact")
            texts = [text]
            metas = [_vec_metadata({"type": entry_type}, session_time, 1)]
        if not texts:
            return "ERROR: vec write requires non-empty text/items"
        try:
            vec.create_collection(collection)
        except Exception:
            pass
        action = str(args.get("action", "add"))
        if action.startswith("update:") or args.get("entry_id"):
            entry_id = str(args.get("entry_id") or action.split(":", 1)[1])
            return vec.update(collection, entry_id, new_text=texts[0], new_metadata=metas[0])
        ids = await vec.add(collection=collection, texts=texts, metadatas=metas)
        return f"added {len(ids)} entries to '{collection}': {ids}"

    if name in {"graph_write", "graph_add_node", "graph_add_edge"}:
        action = str(args.get("action", ""))
        if name == "graph_add_node" or action == "add_node" or args.get("node_id"):
            node_id = str(args.get("node_id", ""))
            if not node_id:
                return "ERROR: graph node requires node_id"
            props = _as_dict(args.get("properties"))
            props.setdefault("ingest_time", session_time)
            return graph.add_node(node_id=node_id, label=str(args.get("label", "")), properties=props)
        source = str(args.get("source", ""))
        target = str(args.get("target", ""))
        relation = str(args.get("relation", ""))
        if not source or not target or not relation:
            return "ERROR: graph edge requires source, target, relation"
        props = _as_dict(args.get("properties"))
        props.setdefault("ingest_time", session_time)
        return graph.add_edge(source=source, target=target, relation=relation, properties=props)

    if name == "graph_delete_node":
        return graph.delete_node(str(args.get("node_id", "")))
    if name == "graph_delete_edge":
        return graph.delete_edge(str(args.get("edge_id", "")))

    return f"ERROR: unsupported tool: {name}"


def _safe_rel_path(base_path: str, path: str) -> str:
    if not path:
        raise ValueError("path is required")
    base = os.path.realpath(base_path)
    if os.path.isabs(path):
        full = os.path.realpath(path)
    else:
        full = os.path.realpath(os.path.join(base, path))
    if full != base and not full.startswith(base + os.sep):
        raise ValueError(f"path escapes memory root: {path}")
    rel = os.path.relpath(full, base)
    return "" if rel == "." else rel


def _ensure_leading_newline_if_needed(fs: Any, path: str, content: str) -> str:
    full = os.path.join(fs.base_path, path)
    if os.path.exists(full) and os.path.getsize(full) > 0 and content and not content.startswith("\n"):
        return "\n" + content
    return content


def _update_line(fs: Any, path: str, action: str, content: str) -> str:
    try:
        line_no = int(action.split(":", 1)[1])
    except Exception:
        return f"ERROR: invalid update action: {action}"
    old = fs.read_file(path)
    if old.startswith("ERROR"):
        return old
    lines = old.split("\n")
    if line_no < 1 or line_no > len(lines):
        return f"ERROR: line {line_no} out of range"
    lines[line_no - 1] = content
    return fs.write_file(path, "\n".join(lines))


def _update_meta(fs: Any, path: str, meta_json: str) -> str:
    if not meta_json:
        return "ERROR: update_meta requires content/meta_json"
    old = fs.read_file(path)
    if old.startswith("ERROR"):
        return fs.write_file(path, meta_json + "\n")
    lines = old.split("\n")
    if lines and lines[0].lstrip().startswith("{"):
        lines[0] = meta_json
    else:
        lines.insert(0, meta_json)
    return fs.write_file(path, "\n".join(lines))


def _vec_metadata(raw: Any, session_time: str, turn: int) -> dict[str, Any]:
    meta = _as_dict(raw)
    meta.setdefault("ingest_time", session_time)
    meta.setdefault("ingest_turn", turn)
    return meta


def _as_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        import json as _json
        try:
            parsed = _json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except _json.JSONDecodeError:
            return {"note": raw}
    return {}


def _apply_mode() -> str:
    return os.environ.get("MEMORY_RL_APPLY_MODE", "tool_calls").strip().lower()


def _queries_to_text(queries: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for q in queries:
        if not isinstance(q, dict):
            continue
        text = str(q.get("query") or q.get("text") or q.get("content") or "").strip()
        q_type = str(q.get("type") or q.get("query_type") or "").strip()
        if text:
            parts.append(f"[{q_type}] {text}" if q_type else text)
    return "\n".join(parts)


def _format_messages_as_content(buffer: list[dict[str, Any]]) -> str:
    """老 ingest 时把 messages buffer 拼成 ``[i] role: content`` 文本块。

    迁移后 llm_gateway 的 ingest task 直接接受 messages list，不再需要这种
    平铺文本；保留此 helper 用于回退场景。
    """
    lines = []
    for i, msg in enumerate(buffer):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")
        tool_name = msg.get("tool_name", "")

        if role == "tool" and tool_name:
            lines.append(f"[{i + 1}] tool({tool_name}): {content}")
        elif role == "assistant" and tool_calls:
            parts: list[str] = []
            if content:
                parts.append(content)
            for tc in tool_calls:
                tc_name = tc.get("name", "unknown_tool")
                tc_args = tc.get("arguments", "")
                if isinstance(tc_args, str) and len(tc_args) > 500:
                    tc_args = tc_args[:500] + "..."
                parts.append(f"[tool_call: {tc_name}({tc_args})]")
            lines.append(
                f"[{i + 1}] {role}: {' '.join(parts) if parts else '(empty)'}"
            )
        else:
            lines.append(f"[{i + 1}] {role}: {content}")
    return "\n".join(lines)


__all__ = ["MemoryEnv", "StepResult"]
