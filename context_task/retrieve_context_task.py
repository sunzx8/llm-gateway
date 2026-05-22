"""
Retrieve Context Task — 完备封装 T2 检索流程。

将 RetrieveT2Task 的核心 agent loop 逻辑和 query_memory
的上层调用逻辑统一封装为可独立调用的 ContextTask。

使用方式：
    task = RetrieveContextTask(llm, fs_store, vec_store, graph_store)
    result = await task.execute(
        query="用户的编程语言偏好是什么？",
        session_id="session_001",
        user_id="user_001",
    )
    # result["retrieved_context"] 即为检索到的记忆上下文

设计原则：
- 继承 BaseContextTask，复用 llm_generate_with_stat / tool_with_stat / save_checkpoint
- 内置完整的 retrieve agent loop（多轮 LLM tool-calling）
- 内置 quick_search 预检索（BM25 + vector + graph 并行）
- 内置 user prompt 构建（index.md、fs/vec/graph 状态注入）
- 所有 LLM 调用和工具操作自动统计
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from context_task.base_context_task import BaseContextTask, TaskStats
from context_task.prompt.chinese_prompt import (
    _T2_QUERY_GUIDANCE_TEMPLATE,
    RETRIEVE_SYSTEM_PROMPT,
    RETRIEVE_USER_TEMPLATE,
    PG_BACKEND_NOTICE,
)
from context_task.tool import (
    RETRIEVE_TOOLS,
    _PG_RETRIEVE_TOOLS,
    get_retrieve_tools,
    get_pg_retrieve_tools,
)

if TYPE_CHECKING:
    from utils.memory_llm_interface import LLMInterface
    from storage.file_system_store import FileSystemStore
    from storage.stores_base import GraphStoreBase
    from storage.stores_base import VectorStoreBase
import logger.logger as logger

from storage.file_system_store import (
    INDEX_FILE_PATH,
    INDEX_FILE_PATH_FALLBACK,
)


# ---------------------------------------------------------------------------
# 业务统计数据结构
# ---------------------------------------------------------------------------


@dataclass
class RetrieveContextTaskStats(TaskStats):
    """RetrieveContextTask 的任务级业务统计数据。

    继承 TaskStats 获得 LLM/工具通用统计字段，
    同时扩展 retrieve 任务特有的业务指标。
    采用 dataclass 形式，字段即指标，便于查看、维护和序列化。
    """

    # ── 任务结果状态 ──
    status: str = ""
    """任务执行状态，取值: 'success' | 'empty' | 'error'"""

    reason: str = ""
    """失败或空结果的原因"""

    # ── 上下文标识 ──
    user_id: str = ""
    """用户 ID"""

    session_id: str = ""
    """当前会话 ID"""

    query: str = ""
    """检索查询文本"""

    # ── 预检索统计 ──
    quick_search_backends: list[str] = field(default_factory=list)
    """quick_search 使用的后端列表"""

    quick_search_result_length: int = 0
    """quick_search 返回的结果长度"""

    # ── 执行统计 ──
    tools_called: int = 0
    """agent loop 中工具调用总次数"""

    # ── 检索结果 ──
    retrieved_context: str = ""
    """最终检索到的上下文内容"""

    retrieved_context_length: int = 0
    """检索上下文内容的长度"""

    finish_result: str = ""
    """finish 工具返回的完整结果文本"""

    query_memory: str = ""
    """完整的记忆上下文（带prompt解释）"""

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典，合并基类统计和业务统计。"""
        base = super().to_dict()
        base.update({
            "status": self.status,
            "reason": self.reason,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "query": self.query,
            "quick_search_backends": self.quick_search_backends,
            "quick_search_result_length": self.quick_search_result_length,
            "tools_called": self.tools_called,
            "retrieved_context": self.retrieved_context,
            "retrieved_context_length": self.retrieved_context_length,
            "finish_result": self.finish_result,
            "query_memory": self.query_memory,
        })
        return base


# ---------------------------------------------------------------------------
# 工具定义（使用统一的工具模块）
# ---------------------------------------------------------------------------

# 基础检索工具集（从统一的工具模块导入）
RETRIEVE_TOOLS = RETRIEVE_TOOLS

# PostgreSQL 后端专用的检索工具集
_PG_RETRIEVE_TOOLS = _PG_RETRIEVE_TOOLS


# ---------------------------------------------------------------------------
# RetrieveContextTask
# ---------------------------------------------------------------------------


class RetrieveContextTask(BaseContextTask):
    """完备封装 T2 检索流程的 ContextTask。

    整合了以下逻辑：
    1. **quick_search 预检索**：BM25 + vector + graph 并行预取
    2. **user prompt 构建**：注入 index.md、fs/vec/graph 状态、预检索结果
    3. **agent loop**：多轮 LLM tool-calling 循环，执行检索操作
    4. **结果组装**：提取 finish 结果或纯文本回复作为 retrieved_context
    5. **统计**：所有 LLM 调用和工具操作自动记录到 TaskStats

    生命周期：
        execute(query, session_id, user_id) →
            pre_run: 重置统计
            run: quick_search → 构建 prompt → agent loop → 组装结果
            post_run: 记录完成日志
    """

    task_name = "retrieve_context"

    # ── 配置（从全局 YAML 配置读取，构造参数可覆盖） ──
    _CFG_SECTION = "retrieve_context_task"

    MAX_TURNS: int = 20
    """agent loop 的最大轮次"""

    def __init__(
        self,
        llm: "LLMInterface",
        fs_store: "FileSystemStore",
        vec_store: "VectorStoreBase",
        graph_store: "GraphStoreBase",
        *,
        max_turns: int,
    ):
        """初始化 RetrieveContextTask。

        配置优先级：构造参数 > 全局 YAML 配置 > 类属性默认值。

        Args:
            llm: LLM 接口实例。
            fs_store: 文件系统存储后端。
            vec_store: 向量数据库存储后端。
            graph_store: 图数据库存储后端。
            max_turns: 覆盖 MAX_TURNS。
        """
        super().__init__(llm)
        self.fs = fs_store
        self.vec = vec_store
        self.graph = graph_store

        # 配置优先级：构造参数 > 全局 YAML 配置 > 类属性默认值
        self.MAX_TURNS = max_turns

        # 运行时状态
        self._current_question: str = ""
        self._current_session_id: str = ""

    # ------------------------------------------------------------------
    # 生命周期实现
    # ------------------------------------------------------------------

    async def pre_run(self, query: str = "", session_id: str = "", **kwargs: Any) -> None:
        """前置处理：初始化业务统计和运行时状态。"""
        # 保留基类 execute() 中已设置的 start_time
        _start_time = getattr(self._stats, "start_time", 0.0) if hasattr(self, "_stats") else 0.0

        # 使用 RetrieveContextTaskStats 替代基类的 TaskStats，
        # 这样 llm_generate_with_stat / tool_with_stat 的统计会自动记录到同一实例
        self._stats = RetrieveContextTaskStats(task_name=self.task_name)
        self._stats.start_time = _start_time

        self._current_question = query
        self._current_session_id = session_id

    async def run(self, query: str = "", session_id: str = "", user_id: str = "default_user", messages: list[dict[str, Any]] | None = None, **kwargs: Any) -> None:
        """核心逻辑：quick_search → 构建 prompt → agent loop → 组装结果。

        将业务结果直接写入 self._stats（RetrieveContextTaskStats 实例）。

        Args:
            query: 检索查询文本（必需）。
            session_id: 当前 session ID（用于日志和 bash 环境变量）。
            user_id: 用户 ID（默认 "default_user"）。
            messages: 当前对话的完整消息列表（可选，传递给 prompt 构建）。
        """
        biz_stats: RetrieveContextTaskStats = self._stats  # type: ignore[assignment]

        if not query:
            biz_stats.status = "empty"
            biz_stats.reason = "empty query"
            biz_stats.user_id = user_id
            biz_stats.session_id = session_id
            return

        self._current_question = query
        self._current_session_id = session_id

        logger.info(
            "RetrieveContextTask: run start "
            "(query_len=%d, user_id=%s, session_id=%s)",
            len(query), user_id, session_id,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "RetrieveContextTask: query_content=%.200s",
                query[:200],
            )

        # Phase 1: quick_search 预检索
        logger.info("RetrieveContextTask: Phase 1 - quick_search start")
        quick_results, quick_backends = await self._quick_search(query)
        logger.info(
            "RetrieveContextTask: Phase 1 - quick_search done "
            "(backends=%s, result_len=%d)",
            quick_backends, len(quick_results),
        )

        # Phase 2: 构建 user prompt
        logger.info("RetrieveContextTask: Phase 2 - building user prompt...")
        user_prompt = self._build_user_prompt(json.dumps(messages), quick_results)
        logger.info(
            "RetrieveContextTask: Phase 2 - user prompt built (len=%d)",
            len(user_prompt),
        )

        # Phase 3: 执行 agent loop
        logger.info(
            "RetrieveContextTask: Phase 3 - agent loop start (max_turns=%d)",
            self.MAX_TURNS,
        )
        loop_stats = await self._run_agent_loop(
            user_prompt=user_prompt,
            session_id=session_id,
        )

        # 填充上下文标识和预检索统计
        biz_stats.user_id = user_id
        biz_stats.session_id = session_id
        biz_stats.query = query
        biz_stats.quick_search_backends = quick_backends
        biz_stats.quick_search_result_length = len(quick_results)
        biz_stats.tools_called = loop_stats.tools_called
        biz_stats.finish_result = loop_stats.finish_result

        # 确定最终 retrieved_context
        context = biz_stats.finish_result or ""
        biz_stats.retrieved_context = context
        biz_stats.retrieved_context_length = len(context)
        biz_stats.status = "success" if context else "empty"
        if not context:
            biz_stats.reason = "no relevant context found"

        logger.info(
            "RetrieveContextTask: run completed "
            "(status=%s, context_len=%d, tools_called=%d)",
            biz_stats.status, biz_stats.retrieved_context_length,
            biz_stats.tools_called,
        )

        # 拼接 user_msg（参考 AgentMemoryT2System.query 的拼接方式）
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        memory_body = (context or "").strip() or "(No relevant memories found)"
        t2_schema_notes = _T2_QUERY_GUIDANCE_TEMPLATE.format(now=now)
        biz_stats.query_memory = (
            f"# Memory Context\n\n"
            f"{memory_body}\n\n"
            f"---\n\n"
            f"{t2_schema_notes}\n"
        )

    async def post_run(self, *, session_id: str = "", **kwargs: Any) -> None:
        """后置处理：记录完成日志。"""
        biz_stats: RetrieveContextTaskStats = self._stats  # type: ignore[assignment]
        logger.info(
            "RetrieveContextTask: completed, "
            "status=%s, tools=%d, context_len=%d",
            biz_stats.status,
            biz_stats.tools_called,
            biz_stats.retrieved_context_length,
        )

    # ------------------------------------------------------------------
    # Quick Search 预检索
    # ------------------------------------------------------------------

    async def _quick_search(self, question: str) -> tuple[str, list[str]]:
        """跨三个后端执行快速预检索。

        Args:
            question: 检索查询文本。

        Returns:
            (预检索结果文本, 命中的后端列表)
        """
        results_parts: list[str] = []
        hit_backends: list[str] = []
        fs_results: list[tuple[str, float, str]] = []
        vec_results: list[dict[str, Any]] = []
        graph_results: list[dict[str, Any]] = []
        try:
            fs_results = await self.tool_with_stat(
                "quick_fs_search",
                self.fs.search_bm25,
                question, 5,
                arguments_summary={"query": question[:100], "top_k": 5},
            )
            if fs_results:
                hit_backends.append("fs")
                fs_lines = [
                    f"  [{score:.2f}] {path}\n      {snippet[:300].replace(chr(10), ' ')}"
                    for path, score, snippet in fs_results
                ]
                results_parts.append("### File System (BM25):\n" + "\n".join(fs_lines))
        except Exception as e:
            logger.error("Quick fs search failed: %s", e)

        # 2. Vector semantic search
        try:
            vec_results = await self.tool_with_stat(
                "quick_vec_search",
                self.vec.search_all,
                question, top_k=10,
                arguments_summary={"query": question[:100], "top_k": 10},
            )
            if vec_results:
                hit_backends.append("vec")
                vec_lines: list[str] = []
                for r in vec_results:
                    meta = r.get("metadata", {}) or {}
                    meta_bits: list[str] = []
                    for key in ("source", "source_session", "session", "path", "type", "status"):
                        if meta.get(key):
                            meta_bits.append(f"{key}={meta[key]}")
                    meta_str = (" | " + ", ".join(meta_bits)) if meta_bits else ""
                    vec_lines.append(
                        f"  [{r['score']:.3f}] id={r.get('id', '?')} "
                        f"coll={r.get('collection', '?')}{meta_str}\n"
                        f"      {r.get('text', '')[:250]}"
                    )
                results_parts.append("### Vector DB (Semantic):\n" + "\n".join(vec_lines))
        except Exception as e:
            logger.error("Quick vector search failed: %s", e)

        # 3. Graph keyword search
        try:
            keywords = self._extract_keywords(question)
            seen_nodes: set[str] = set()
            for kw in keywords[:5]:
                try:
                    nodes = await self.tool_with_stat(
                        "quick_graph_search",
                        self.graph.search_nodes,
                        keyword=kw,
                        arguments_summary={"keyword": kw},
                    )
                except Exception as e:
                    logger.warning("graph.search_nodes failed for kw=%r: %s", kw, e)
                    continue
                for node in nodes[:3]:
                    if node["id"] in seen_nodes:
                        continue
                    seen_nodes.add(node["id"])
                    try:
                        neighbors = self.graph.get_neighbors(node["id"])
                    except Exception:
                        neighbors = []
                    graph_results.append({"node": node, "neighbors": neighbors[:5]})
            if graph_results:
                hit_backends.append("graph")
                graph_lines: list[str] = []
                for gr in graph_results:
                    node = gr["node"]
                    graph_lines.append(
                        f"  Node: id={node['id']} label={node.get('label') or '(none)'} "
                        f"props={node.get('properties', {})}"
                    )
                    for nb in gr["neighbors"]:
                        edge = nb["edge"]
                        target = nb["node"]
                        graph_lines.append(
                            f"    --[{edge['relation']}]--> id={target['id']} "
                            f"label={target.get('label') or '(none)'}"
                        )
                results_parts.append("### Graph DB (Entity Relations):\n" + "\n".join(graph_lines))
        except Exception as e:
            logger.error("Quick graph search failed: %s", e)

        # ── 3b. User stance overview（不依赖 query 关键字）──────────────────
        # 把所有 Person / user 节点出发的**立场类**边汇总——这是 ingest 阶段
        # 归纳出的"用户态度 tag"鸟瞰图，对 stance / evolution / suggest 类 query
        # 几乎总是有价值；即使 query 关键字未命中具体 Topic 节点，这些边也能提示
        # "用户在哪些 topic 上已有明确态度"，避免 retrieve agent 因 keyword miss 而
        # 完全忽略 graph 信号。
        stance_relations = {
            "prefers", "enjoys", "likes", "tried", "engaged_in",
            "participated_in", "joined", "created", "committed_to",
            "interested_in", "curious_about",
            "avoids", "withdrew_from", "struggled_in", "dislikes",
        }
        stance_lines: list[str] = []
        try:
            person_nodes = self.graph.search_nodes(label="Person")
            if not person_nodes:
                person_nodes = self.graph.search_nodes(keyword="user")
            person_ids = {n["id"] for n in person_nodes}
            for pid in list(person_ids)[:4]:
                try:
                    nbrs = self.graph.get_neighbors(pid, direction="out")
                except Exception:
                    nbrs = []
                for nb in nbrs:
                    edge = nb["edge"]
                    rel = edge.get("relation", "")
                    if rel not in stance_relations:
                        continue
                    target = nb["node"]
                    stance_lines.append(
                        f"  ({pid}) --[{rel}]--> id={target['id']} "
                        f"label={target.get('label') or '(none)'}"
                    )
                    if len(stance_lines) >= 30:
                        break
                if len(stance_lines) >= 30:
                    break
        except Exception as e:
            logger.warning("user stance overview failed: %s", e)
        if stance_lines:
            results_parts.append(
                "### Graph: User Stance Overview "
                "(ingest-derived attitude edges; query-agnostic):\n"
                + "\n".join(stance_lines)
            )

        # ── 4. RRF fusion (cross-backend rank aggregation) ────────────
        fs_rank = [
            (f"fs:{path}", f"fs:{path} — {snippet[:180].replace(chr(10), ' ')}")
            for path, _, snippet in (fs_results or [])
        ]
        vec_rank = [
            (
                f"vec:{r.get('id', '?')}",
                f"vec:{r.get('id', '?')} (coll={r.get('collection', '?')}) — "
                f"{r.get('text', '')[:180].replace(chr(10), ' ')}",
            )
            for r in (vec_results or [])
        ]
        graph_rank = [
            (
                f"graph:{gr['node']['id']}",
                f"graph:{gr['node']['id']} (label={gr['node'].get('label') or '(none)'}) — "
                f"{str(gr['node'].get('properties', {}))[:160]}",
            )
            for gr in graph_results
        ]
        fused = self._rrf_fuse([fs_rank, vec_rank, graph_rank], k=60, top_k=10)
        if fused:
            fused_lines = [
                f"  [rrf={score:.4f}] {summary}"
                for score, _key, summary in fused
            ]
            # Fused block 放最前面——对 LLM 而言"三后端一致同意的候选"比任何
            # 单后端 top-N 更具指向性。
            results_parts.insert(
                0, "### Fused Top Candidates (RRF across FS / Vec / Graph):\n" + "\n".join(fused_lines)
            )

        return "\n\n".join(results_parts) if results_parts else "", hit_backends

    @staticmethod
    def _rrf_fuse(
        ranked_lists: list[list[tuple[str, str]]],
        k: int = 60,
        top_k: int = 10,
    ) -> list[tuple[float, str, str]]:
        """Reciprocal Rank Fusion.

        Args:
            ranked_lists: each element is a per-backend ranked list of
                (doc_key, display_summary) tuples, ordered best→worst.
            k: RRF smoothing constant (default 60 per Cormack et al.).
            top_k: number of fused candidates to return.

        Returns:
            Top-k fused (score, doc_key, display_summary) tuples, best first.
        """
        scores: dict[str, float] = {}
        summaries: dict[str, str] = {}
        for lst in ranked_lists:
            for rank, (key, summary) in enumerate(lst, start=1):
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
                summaries.setdefault(key, summary)
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [(score, key, summaries[key]) for key, score in ranked]

    def _build_user_prompt(self, question: str, quick_results: str) -> str:
        """构建 retrieve 的 user prompt。

        读取当前记忆库状态（index.md、fs 结构、vec_stats、graph_stats），
        注入预检索结果，生成完整的 user prompt。

        Args:
            question: 检索查询文本。
            quick_results: 预检索结果文本。

        Returns:
            完整的 user prompt。
        """
        fs_files = self.fs.list_files()
        vec_stats = self.vec.get_stats()
        graph_stats = self.graph.get_stats()

        # --- index.md 路径对齐（优先 .meta/index.md，回退 index.md） ---
        index_path, index_content = self.fs.read_index()

        # --- 可读的后端明细 ---
        vec_collections_map = (
            vec_stats.get("collections", {}) if isinstance(vec_stats, dict) else {}
        )
        if vec_collections_map:
            vec_collections_detail = "\n".join(
                f"    - {name}: {size} entries"
                for name, size in sorted(vec_collections_map.items())
            )
        else:
            vec_collections_detail = "    (no collections)"

        graph_labels_map = graph_stats.get("node_labels", {}) if isinstance(graph_stats, dict) else {}
        graph_rel_map = graph_stats.get("relation_types", {}) if isinstance(graph_stats, dict) else {}
        graph_labels_str = (
            ", ".join(f"{k}:{v}" for k, v in sorted(graph_labels_map.items()))
            or "(none)"
        )
        graph_relations_str = (
            ", ".join(f"{k}:{v}" for k, v in sorted(graph_rel_map.items()))
            or "(none)"
        )

        # --- 记忆库结构摘要 ---
        try:
            fs_structure = self.fs.get_structure_summary(max_depth=2)
        except Exception as e:
            logger.warning("get_structure_summary failed: %s", e)
            fs_structure = "(unavailable)"

        user_msg = RETRIEVE_USER_TEMPLATE.format(
            question=question,
            index_path=index_path,
            index_content=index_content,
            fs_structure=fs_structure,
            fs_file_count=len(fs_files),
            vec_total=vec_stats.get("total_entries", 0) if isinstance(vec_stats, dict) else 0,
            vec_collections_detail=vec_collections_detail,
            graph_nodes=graph_stats.get("total_nodes", 0) if isinstance(graph_stats, dict) else 0,
            graph_edges=graph_stats.get("total_edges", 0) if isinstance(graph_stats, dict) else 0,
            graph_labels=graph_labels_str,
            graph_relations=graph_relations_str,
        )
        if quick_results:
            user_msg += (
                f"\n\n## Quick Search Results (pre-fetched):\n{quick_results}"
            )
        # PG-only：告诉 agent 它额外有 SQL / Cypher 能力
        if hasattr(self.vec, "sql_query_read"):
            user_msg += "\n\n" + PG_BACKEND_NOTICE
        return user_msg

    # ------------------------------------------------------------------
    # Agent Loop
    # ------------------------------------------------------------------

    async def _run_agent_loop(
        self,
        user_prompt: str,
        session_id: str,
    ) -> RetrieveContextTaskStats:
        """执行多轮 LLM tool-calling agent loop。

        Args:
            user_prompt: 构建好的 user prompt。
            session_id: 当前 session ID（仅用于日志和 bash 环境变量）。

        Returns:
            本次检索的业务统计数据。
        """
        system_prompt = RETRIEVE_SYSTEM_PROMPT
        # 基础工具集 + 按后端能力扩展（PG 后端会额外暴露 sql_query_read /
        # graph_cypher_read / vec_scroll，这些工具在 in-memory 后端会返回 ERROR
        # 但 LLM 看不到则不会调用）。
        tools: list[dict] = list(RETRIEVE_TOOLS)
        if hasattr(self.vec, "sql_query_read"):
            tools.extend(_PG_RETRIEVE_TOOLS)
        messages_history: list[dict[str, Any]] = [
            {"role": "user", "content": user_prompt}
        ]

        biz_stats = RetrieveContextTaskStats()
        finish_result: str = ""

        for turn in range(self.MAX_TURNS):
            logger.info(
                "RetrieveContextTask: agent loop turn %d/%d start",
                turn + 1, self.MAX_TURNS,
            )
            # 调用 LLM（自动统计）
            response = await self.llm_generate_with_stat(
                system_prompt,
                messages_history,
                tools=tools,
                label=f"retrieve_loop_turn_{turn + 1}",
            )

            if not response.tool_calls:
                # 模型返回纯文本 → 视为最终回答（assembled context）
                if response.content:
                    finish_result = finish_result or response.content
                logger.info(
                    "RetrieveContextTask: agent loop turn %d/%d "
                    "ended with text response (len=%d)",
                    turn + 1, self.MAX_TURNS,
                    len(response.content or ""),
                )
                break

            # 将模型回复加入历史
            messages_history.append(response.to_message())
            should_break = False

            logger.info(
                "RetrieveContextTask: agent loop turn %d/%d "
                "got %d tool_calls",
                turn + 1, self.MAX_TURNS, len(response.tool_calls),
            )

            for tc in response.tool_calls:
                if tc.name == "finish":
                    finish_result = tc.arguments.get("result", "")
                    messages_history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "(task finished)",
                    })
                    should_break = True
                    continue

                # 执行工具调用（自动统计）
                result_str = await self._execute_tool_with_stat(
                    tc.name, tc.arguments, biz_stats, session_id,
                )
                messages_history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

            if should_break:
                logger.info(
                    "RetrieveContextTask: agent loop finished at turn %d/%d "
                    "(finish tool called, result_len=%d)",
                    turn + 1, self.MAX_TURNS, len(finish_result),
                )
                break
        else:
            logger.warning(
                "RetrieveContextTask: reached max_turns=%d without finish",
                self.MAX_TURNS,
            )

        logger.info(
            "RetrieveContextTask: agent loop summary "
            "(total_turns=%d, tools_called=%d, finish_result_len=%d)",
            self.MAX_TURNS if self.MAX_TURNS > 0 else 0,
            biz_stats.tools_called,
            len(finish_result),
        )

        biz_stats.finish_result = finish_result
        return biz_stats

    # ------------------------------------------------------------------
    # 工具执行
    # ------------------------------------------------------------------

    async def _execute_tool_with_stat(
        self,
        tool_name: str,
        args: dict[str, Any],
        biz_stats: RetrieveContextTaskStats,
        session_id: str,
    ) -> str:
        """执行单个工具调用，自动记录统计。

        通过 BaseContextTask.dispatch_tool 统一分派（只读模式）。

        Args:
            tool_name: 工具名称。
            args: 工具参数。
            biz_stats: 业务统计数据实例。
            session_id: 当前 session ID。

        Returns:
            工具执行结果字符串。
        """
        try:
            result = await self.dispatch_tool(
                tool_name, args,
                session_id=session_id,
                allow_write=False,
            )
            biz_stats.tools_called += 1
            return result
        except Exception as e:
            logger.error(
                "RetrieveContextTask: tool %s failed: %s", tool_name, e,
            )
            return f"ERROR: {e}"

    # ------------------------------------------------------------------
    # 便捷方法：query_memory
    # ------------------------------------------------------------------

    async def query_memory(
        self,
        query: str,
        session_id: str = "",
        user_id: str = "default_user",
    ) -> str:
        """查询记忆，返回检索到的上下文字符串。

        对齐 EventDispatcher.query_memory 的接口语义，
        便于上层系统直接替换调用。

        Args:
            query: 查询文本。
            session_id: 当前 session ID。
            user_id: 用户 ID。

        Returns:
            检索到的记忆上下文字符串，无结果时返回空字符串。
        """
        logger.info("RetrieveContextTask.query_memory: query=%s...", query[:80])
        result = await self.execute(
            query=query,
            session_id=session_id,
            user_id=user_id,
        )
        # execute 直接返回合并后的统计字典（业务 + 通用统计）
        context = result.get("retrieved_context", "")
        if context:
            logger.info(
                "RetrieveContextTask.query_memory: retrieved %d chars", len(context),
            )
        else:
            logger.info("RetrieveContextTask.query_memory: no relevant context found")
        return context

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """抽取用于 graph_search_nodes 的关键词。

        同时处理英文与中日韩文本：

        - 英文：按 ASCII 单词切分，去停用词，保留长度 > 2 的词。
        - CJK：按 Unicode 块提取连续 CJK 字符段，保留长度 ≥ 2 的段作为候选；
          短段（单字）不做 bigram 爆炸，避免产生大量噪声 keyword。
        - 都抽不到时，fallback 用原文前 30 个字符作为整体 keyword。
        """
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "out", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "just", "because",
            "but", "and", "or", "if", "while", "about", "what", "which",
            "who", "whom", "this", "that", "these", "those", "it", "its",
            "i", "me", "my", "myself", "we", "our", "ours", "you", "your",
            "he", "him", "his", "she", "her", "they", "them", "their",
        }

        # 1) 英文 token
        en_words = re.findall(r"\b[a-zA-Z]+\b", text)
        en_keywords = [w for w in en_words if w.lower() not in stop_words and len(w) > 2]

        # 2) CJK 连续段（含中/日韩常用 Unicode 块）
        cjk_runs = re.findall(
            r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]{2,}",
            text,
        )

        # 3) 去重保序
        seen: set[str] = set()
        unique: list[str] = []
        for kw in en_keywords + cjk_runs:
            lk = kw.lower()
            if lk in seen:
                continue
            seen.add(lk)
            unique.append(kw)

        # 4) fallback：一个都没抽到 → 用原文前 30 字符
        if not unique:
            trimmed = text.strip()[:30]
            if trimmed:
                unique.append(trimmed)

        return unique
