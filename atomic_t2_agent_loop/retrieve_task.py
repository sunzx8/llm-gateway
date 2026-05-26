"""T3 retrieve task — 三模式检索（工程化 / LLM query 改写 / Agent 自主检索）。

设计思想：
- 提供三种可切换的检索模式：
  a. 工程化改写：基于 jieba 分词 + 同义词扩展 + 规则化 query 生成
  b. LLM 改写：一次 LLM 调用生成多个适合检索的 query
  c. Agent 自主检索（AGENT 模式）：两轮 LLM 调用，最大化模型策略自由度
     - Round 1 Plan+Retrieve: LLM 在 content 中输出检索策略，同时发出并发 tool calls
     - Round 2 Submit: LLM 审视检索结果，纯 text 输出精筛结果

核心流程（ENGINEERING / LLM 模式）：
1. Query 改写（工程化 or LLM）→ 生成多个检索 query
2. 并行检索三后端：fs_bm25 / vec_search / graph_vector_search
3. Submit 精筛 → 结构化输出

核心流程（AGENT 模式）：
1. Round 1: 开启 reasoning → LLM 在思维链中输出检索策略 + 同时发出并发 tool calls
2. Round 2: 续接 messages（含 tool results），仅提供 submit tool → LLM 调用 submit 精筛
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import TYPE_CHECKING, Any

from llm_gateway.rl.rl_env._models import Event, EventType, TaskResult
from context_task.base_context_task import BaseContextTask

if TYPE_CHECKING:
    from utils.memory_llm_interface import LLMInterface
    from storage.file_system_store import FileSystemStore
    from storage.stores_base import GraphStoreBase as GraphStore, VectorStoreBase as VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 检索模式枚举
# ---------------------------------------------------------------------------

class RetrieveMode(str, Enum):
    """检索模式（仅保留 agent loop — 其它模式已物理移除）。"""
    AGENT_LOOP = "agent_loop"  # Budget-constrained agent loop（默认 max_turns=5）


# ---------------------------------------------------------------------------
# AGENT_LOOP Mode — budget-constrained agent loop prompts
# ---------------------------------------------------------------------------

AGENT_LOOP_SYSTEM_PROMPT = """\
You are a Memory Retrieval Agent. Find relevant memories for the question and submit \
a focused selection. Do NOT output explanation text — only make tool calls.

## Turn 1: Cast a Wide Net (RECALL)
Issue MANY search tool calls at once (they run concurrently):
- `fs_bm25_search`: 2-3 calls with different keyword combinations
- `vec_semantic_search`: 2-3 calls with different semantic phrasings
- `graph_entity_search`: 1-2 calls for key entities
- `fs_read_file`: if you see a relevant path in the structure below

Aim for 5-10 tool calls. More is better for recall.

### Temporal retrieval
If the question contains time cues (recently, last year, in 2022, before/after, earlier/later, current vs past), include temporal terms in search queries and prefer results with matching `[YYYY]`, `[YYYY-MM]`, `[YYYY-MM-DD]`, or `occurred_at` metadata. For evolution questions, retrieve BOTH past and current entries plus their reasons.

## Turn 2: Submit with Restraint (PRECISION)
Review results. Call `submit` with an ADAPTIVE but FOCUSED selection:
- **Keep**: Direct facts, relevant preferences, attitude history, negative experiences, reasons
- **Drop**: Tangential topics, duplicates saying the same thing differently, vague entries, generic profiles not tied to the question
- **Order**: Most relevant first; concrete evidence before synthesized profiles

Selection budget by question type:
- Direct fact recall: 3-8 entries
- Reasons / preference evolution: 8-16 entries (include past + current + WHY)
- Recommendation / suggestion: 6-10 entries normally; max 12 only when needed. Include:
  concrete positive preferences + negative constraints + 0-2 directly relevant profile/constraint entries.
  Do NOT submit broad preference profiles if concrete memories already answer the question.

Never submit 0 entries unless truly no memories exist. If results are insufficient, do targeted follow-up searches before submitting.

## Memory Structure

### File System
{fs_structure}

### Vector Collections
{vec_collections}

### Graph Schema
{graph_schema}

## Tools
- `fs_bm25_search`: BM25 keyword search. Best for names, dates, specific terms.
- `vec_semantic_search`: Semantic vector search. Best for meaning, paraphrases.
- `graph_entity_search`: Entity graph traversal. Best for relationships.
- `fs_read_file`: Read a known file path.
- `fs_execute_bash`: Shell command (read-only). Use grep/find for advanced searches.
- `submit`: Final selection. `keep: ["R1", "R3", ...]` in relevance order.

## Tips
- `source_sessions/` has raw conversation logs — search for verbatim quotes when needed.
- If the question asks for suggestions/recommendations or references "we discussed", search assistant-sourced memories and source_sessions too; past assistant suggestions may be the best answer.
- Entries get IDs (R1, R2, ...) in the order they appear across tool results.
"""

AGENT_LOOP_USER_TEMPLATE = """\
{question}
"""

# ---------------------------------------------------------------------------
# Retrieve agent loop tool definitions (alive — used by AGENT_LOOP_TOOLS below)
# ---------------------------------------------------------------------------

AGENT_RETRIEVE_TOOLS: list[dict] = [
    {"type": "function", "function": {
        "name": "fs_bm25_search",
        "description": (
            "BM25 keyword search over the memory filesystem. Returns matching files "
            "with scores and snippets. Use for exact keyword matches, specific names, "
            "dates, topics mentioned in files."
        ),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search query (keywords or natural language)"},
            "top_k": {"type": "integer", "description": "Max results to return (default: 8)"},
            "scope": {
                "type": "array", "items": {"type": "string"},
                "description": "Optional directory prefixes to narrow search (e.g. ['people/alex/', 'events/'])",
            },
            "temporal_filter": {
                "type": "string",
                "description": "Optional time filter (e.g. '2017', '2019-03', '2018~2020', 'recent', 'current', 'past')",
            },
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "vec_semantic_search",
        "description": (
            "Semantic vector search across all vector collections. Returns text chunks "
            "ranked by cosine similarity. Use for meaning-based retrieval, conceptual "
            "queries, paraphrased searches."
        ),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Natural language search query"},
            "top_k": {"type": "integer", "description": "Max results to return (default: 15)"},
            "temporal_filter": {
                "type": "string",
                "description": "Optional time filter (e.g. '2017', '2019-03', '2018~2020', 'recent', 'current', 'past')",
            },
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "graph_entity_search",
        "description": (
            "Entity-based graph search with subgraph expansion. Finds entities (people, "
            "places, topics) and their relationships via multi-hop graph traversal."
        ),
        "parameters": {"type": "object", "properties": {
            "keyword": {
                "type": "string",
                "description": "Entity name or keyword to search for in graph nodes",
            },
            "semantic_query": {
                "type": "string",
                "description": "Optional semantic query for vector-based node search",
            },
            "max_hops": {"type": "integer", "description": "Max graph traversal depth (default: 2)"},
        }},
    }},
    {"type": "function", "function": {
        "name": "fs_read_file",
        "description": (
            "Read the full content of a specific file in the memory filesystem. "
            "Use when you know the exact file path from the filesystem structure."
        ),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative file path in the memory filesystem"},
        }, "required": ["path"]},
    }},
]


AGENT_LOOP_TOOLS: list[dict] = [
    *AGENT_RETRIEVE_TOOLS,
    {"type": "function", "function": {
        "name": "fs_execute_bash",
        "description": (
            "Execute shell command in memory store root. "
            "Allowed: cat/ls/head/tail/wc/sort/uniq/tree/find/grep/awk/sed/cut/tr/diff/jq/date. "
            "Forbidden: write commands, `..` escape, git mutations."
        ),
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default: 15)"},
        }, "required": ["command"]}}},
    {
        "type": "function", "function": {
            "name": "submit",
            "description": (
                "Submit the final filtered memory entries. Select ONLY entries relevant "
                "to the user's question, ordered by relevance (most relevant first). "
                "Be INCLUSIVE — when in doubt, keep the entry."
            ),
            "parameters": {"type": "object", "properties": {
                "keep": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of entry IDs to keep, in relevance order (e.g. ['R1', 'R3', 'R5'])",
                },
            }, "required": ["keep"]},
        },
    },
]


# ---------------------------------------------------------------------------
# Retrieve T3 Task
# ---------------------------------------------------------------------------


class RetrieveT2AgentLoopTask(BaseContextTask):
    """T3 retrieve task — 三模式检索。

    支持三种检索模式：
    - ENGINEERING：纯工程化（jieba 分词 + 停用词过滤 + 同义词扩展）
    - LLM：一次 LLM 调用生成多个检索 query
    - AGENT：两轮 LLM 调用（Plan+Retrieve → Submit），最大化模型策略自由度

    ENGINEERING / LLM 模式检索流程：
    1. Query 改写 → 多个 query
    2. 并行检索 fs/vec/graph
    3. Submit 精筛 → 结构化输出

    AGENT 模式检索流程：
    1. Round 1 (Plan + Retrieve): 开启 reasoning/thinking，LLM 在思维链中输出
       检索策略，同时发出并发 tool calls（fs_bm25_search / vec_search / graph_search 等）
    2. Round 2 (Submit): 续接 R1 的 messages + tool results，仅提供 submit tool，
       LLM 调用 submit({keep: [...]}) 输出精筛结果
    """

    task_name = "retrieve_t2_agent_loop"
    log_event_type = "memory_query"
    max_turns = 2  # AGENT 模式 2 轮

    # 配置参数
    retrieve_mode: RetrieveMode = RetrieveMode.AGENT_LOOP
    # Graph 检索参数
    graph_similarity_threshold: float = 0.5  # 节点相似度阈值
    graph_subgraph_depth: int = 2  # 子图召回深度
    # 分跳数的 top_k 配置：key=跳数, value=该跳保留的最大边数（按相似度降序）
    # 例如 {1: 5, 2: 10} 表示 1 跳保留 5 条边，2 跳保留 10 条边
    graph_hop_top_k: dict[int, int] | None = None  # None 使用默认值 {1: 10, 2: 15, 3: 5}
    # 每跳边的最小相似度阈值（低于此值的边不展示）
    graph_edge_min_similarity: float = 0.1
    # 检索 top_k
    fs_top_k: int = 8
    vec_top_k: int = 15
    graph_top_k: int = 3
    # RRF 参数
    rrf_k: int = 60
    # Reranker 精排参数
    reranker_url: str = ""  # BGE Reranker 服务地址（空字符串=禁用 reranker，走 RRF fallback）
    rerank_score_threshold: float = -2.0  # rerank 分数阈值（BGE logits，>0 表示相关）
    rerank_top_k: int = 15  # rerank 后最终保留的最大条目数
    rerank_dedup_threshold: float = 0.75  # Jaccard 去重阈值
    # 多跳保护配额：精排后至少保留 N 条 2+跳的边（防止多跳信息被全部排低）
    rerank_multihop_min_keep: int = 3
    # 最终输出各后端 top_k（兜底防过长）
    output_fs_top_k: int = 6
    output_vec_top_k: int = 8
    output_graph_edges_top_k: int = 15
    # Agent loop 默认 budget 上限
    RETRIEVE_AGENT_MAX_TURNS: int = 5

    def __init__(
        self,
        llm: LLMInterface,
        fs_store: FileSystemStore,
        vec_store: VectorStore,
        graph_store: GraphStore,
        *,
        mode: RetrieveMode = RetrieveMode.AGENT_LOOP,
        max_turns: int | None = None,
        **kwargs: Any,
    ):
        # 对齐 BaseContextTask 子类约定：自己持有 stores
        super().__init__(llm)
        self.fs = fs_store
        self.vec = vec_store
        self.graph = graph_store
        # 兼容老接口字段（仍会被一些 helper 引用）
        self.retrieve_mode = mode
        if max_turns is not None:
            self.RETRIEVE_AGENT_MAX_TURNS = max_turns
        self.extra_kwargs = kwargs
        # 业务结果（agent loop 摘要 / 检索上下文 / 工具轨迹），由 run() 写入
        self.result_extra: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 核心执行逻辑（agent loop 入口）
    # ------------------------------------------------------------------

    async def run(
        self,
        user_id: str = "default_user",
        session_id: str = "",
        query: str = "",
        **kwargs: Any,
    ) -> None:
        """执行检索（agent loop 模式）。

        BaseContextTask 子类约定：``run()`` 不返回值，结果写入 ``self.result_extra``，
        由 :meth:`BaseContextTask.execute` 包装并合并到 ``self._stats.to_dict()``。
        """
        question = (query or "").strip()
        if not question:
            self.result_extra["retrieved_context"] = ""
            self.result_extra["query_memory"] = ""
            self.result_extra["error"] = "Empty query"
            return

        # 构造内部 Event 兼容 _execute_agent_loop（其内部读取 event.payload）
        event = Event(
            type=EventType.MEMORY_QUERY,
            payload={
                "query": question,
                "user_id": user_id,
                "session_id": session_id,
                **{k: v for k, v in kwargs.items() if k not in ("query",)},
            },
        )

        # 仅支持 AGENT_LOOP 模式（迁移时已锁死）
        result: TaskResult = await self._execute_agent_loop(event, question, session_id)

        # 把 TaskResult 字段平展到 result_extra，方便 dispatcher / 调用方
        self.result_extra["retrieved_context"] = result.retrieved_context or result.final_output or ""
        self.result_extra["query_memory"] = self.result_extra["retrieved_context"]
        self.result_extra["finish_summary"] = result.finish_summary or result.finish_result or ""
        self.result_extra["error"] = result.error or ""
        self.result_extra["reasoning"] = result.reasoning or ""
        # 把 TaskResult.stats 中的轨迹字段也透出来
        if isinstance(result.stats, dict):
            for k, v in result.stats.items():
                self.result_extra.setdefault(k, v)

    # ------------------------------------------------------------------
    # Agent loop 实现 — RetrieveT2AgentLoopTask 的核心
    # ------------------------------------------------------------------

    async def _execute_agent_loop(
        self, event: Event, question: str, session_id: str,
    ) -> TaskResult:
        """Budget-constrained agent loop 检索。

        与固定 AGENT 模式的核心区别：
        - LLM 自主决定每一轮搜什么、何时 submit
        - search + submit 工具统一暴露
        - 最多 max_turns 轮，每轮可输出多个并发 tool calls
        - 详细统计 token 开销和轮次分布
        """
        import asyncio as _aio
        import json as _json
        def _get_evt(): return None  # noqa: E731 (stub: RL env 不需要结构化事件日志)

        evt = _get_evt()
        max_turns = self.RETRIEVE_AGENT_MAX_TURNS

        # 准备 FS 结构信息
        fs_structure = self._read_index_md()
        if fs_structure == "(empty memory — no index available)":
            fs_structure = self._build_fs_structure_with_meta()

        # 预注入三后端结构快照
        vec_stats = self.vec.get_stats() if self.vec else {}
        vec_collections_map = vec_stats.get("collections", {}) if isinstance(vec_stats, dict) else {}
        vec_collections = "\n".join(
            f"  - {name}: {size} entries"
            for name, size in sorted(vec_collections_map.items())
        ) if vec_collections_map else "(no collections yet)"
        graph_stats = self.graph.get_stats() if self.graph else {}
        graph_labels = graph_stats.get("node_labels", {}) if isinstance(graph_stats, dict) else {}
        graph_rels = graph_stats.get("relation_types", {}) if isinstance(graph_stats, dict) else {}
        graph_schema = (
            f"Nodes: {', '.join(f'{k}:{v}' for k, v in sorted(graph_labels.items())) or '(none)'}\n"
            f"Relations: {', '.join(f'{k}:{v}' for k, v in sorted(graph_rels.items())) or '(none)'}"
        )

        system_prompt = (
            AGENT_LOOP_SYSTEM_PROMPT
            .replace("{fs_structure}", fs_structure)
            .replace("{vec_collections}", vec_collections)
            .replace("{graph_schema}", graph_schema)
        )
        user_prompt = AGENT_LOOP_USER_TEMPLATE.replace("{question}", question)
        tools = AGENT_LOOP_TOOLS

        if evt:
            evt.log("task_prompt", task=self.task_name, session_id=session_id,
                     system_prompt="(agent_loop retrieve)", user_prompt=question,
                     tools_count=len(tools), max_steps=max_turns)

        # --- Token 统计结构 ---
        import json as _json_local
        tool_defs_chars = len(_json_local.dumps(tools, ensure_ascii=False))
        token_stats: dict[str, Any] = {
            "system_prompt_chars": len(system_prompt),
            "tool_defs_chars": tool_defs_chars,
            "initial_user_prompt_chars": len(user_prompt),
            "tool_defs_count": len(tools),
            "per_turn": [],
            "total_llm_calls": 0,
            "total_input_chars": 0,    # 含 system + tools + messages（每轮完整输入）
            "total_output_chars": 0,
            # essential = 所有 tool call 参数（无论读写，workflow 也必须输出）
            "essential_output_chars": 0,
            # overhead = LLM content 文本 + tool call 格式标记
            "overhead_output_chars": 0,
            "tool_call_type_counts": {},
        }

        messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        agent_trace: list[dict[str, Any]] = []
        all_candidates: list[dict[str, Any]] = []
        candidate_idx = 1
        final_candidates: list[dict[str, Any]] = []
        steps_used = 0
        submit_raw = ""

        for turn in range(max_turns):
            steps_used = turn + 1

            # 注入轮次 budget 提示（不改 system prompt，保持 prefix caching）
            if turn > 0:
                remaining = max_turns - turn
                budget_hint = (
                    f"[Turn {turn + 1}/{max_turns}, {remaining} remaining] "
                    f"{'You MUST call `submit` now with the relevant entries.' if remaining <= 1 else 'Continue searching or call `submit` when ready.'}"
                )
                messages.append({"role": "user", "content": budget_hint})

            # 计算本轮输入 chars（system + tools + messages 内容 + assistant tool_calls）
            turn_input_chars = len(system_prompt) + tool_defs_chars
            for m in messages:
                turn_input_chars += len(m.get("content", "") or "")
                if m.get("tool_calls"):
                    turn_input_chars += sum(
                        len(tc_m.get("function", {}).get("name", ""))
                        + len(tc_m.get("function", {}).get("arguments", ""))
                        + len(tc_m.get("id", "")) + 20
                        for tc_m in m["tool_calls"]
                        if isinstance(tc_m, dict)
                    )

            response = await self.llm_generate_with_stat(
                system=system_prompt,
                messages=messages,
                tools=tools,
                label=f"retrieve_agent_turn{turn + 1}",
            )
            token_stats["total_llm_calls"] += 1
            token_stats["total_input_chars"] += turn_input_chars

            turn_output_chars = len(response.content or "")
            turn_essential_chars = 0
            turn_overhead_chars = len(response.content or "")

            step_trace: dict[str, Any] = {
                "step": turn + 1,
                "model_content": (response.content or "")[:500],
                "reasoning_content": (response.reasoning_content or "")[:500] if response.reasoning_content else "",
                "tool_calls_count": len(response.tool_calls or []),
                "tool_calls": [],
            }

            if not response.tool_calls:
                # 纯文本 — 无法继续，尝试自动检索
                if not all_candidates:
                    logger.warning("AGENT_LOOP R%d: no tool calls, auto-retrieving", turn + 1)
                    # fallback: 工程化检索
                    auto_queries = self._rewrite_queries_engineering(question)

                    async def _fs_async():
                        return self._search_fs(auto_queries)

                    fs_r, vec_r, graph_r = await _aio.gather(
                        _fs_async(),
                        self._search_vec(auto_queries),
                        self._search_graph(auto_queries),
                    )
                    all_candidates = self._build_candidates_from_raw(fs_r, vec_r, graph_r)
                    final_candidates = all_candidates
                # 统计
                token_stats["total_output_chars"] += turn_output_chars
                token_stats["overhead_output_chars"] += turn_overhead_chars
                token_stats["per_turn"].append({
                    "turn": turn + 1, "type": "text_only",
                    "input_chars": turn_input_chars, "output_chars": turn_output_chars,
                    "tool_calls": 0,
                })
                agent_trace.append(step_trace)
                break

            # 处理 tool calls
            messages.append(response.to_message())
            should_break = False
            turn_tool_calls = response.tool_calls or []

            # 先并发执行所有非 submit 的 tool calls
            search_calls = [tc for tc in turn_tool_calls if tc.name != "submit"]
            submit_calls = [tc for tc in turn_tool_calls if tc.name == "submit"]

            if search_calls:
                async def _run_tool(tc):
                    result = await self._execute_agent_tool(tc.name, tc.arguments)
                    return tc, result

                gather_results = await _aio.gather(
                    *[_run_tool(tc) for tc in search_calls],
                    return_exceptions=True,
                )

                for gr in gather_results:
                    if isinstance(gr, Exception):
                        logger.warning("AGENT_LOOP tool error: %s", gr)
                        continue
                    tc, result_data = gr

                    tc_args_str = _json.dumps(tc.arguments, ensure_ascii=False)
                    tc_args_chars = len(tc_args_str)
                    # tool call 格式开销：function name + call_id + 协议结构标记
                    tc_format_chars = len(tc.name) + len(getattr(tc, "id", "") or "") + 20
                    tc_output_chars = tc_args_chars + tc_format_chars
                    turn_output_chars += tc_output_chars
                    # essential = tool call 参数；overhead = 格式标记
                    turn_essential_chars += tc_args_chars
                    turn_overhead_chars += tc_format_chars

                    token_stats["tool_call_type_counts"][tc.name] = \
                        token_stats["tool_call_type_counts"].get(tc.name, 0) + 1

                    # 将检索结果转换为候选并构建 tool result message
                    # bash 工具不生成候选，直接返回输出文本
                    if tc.name == "fs_execute_bash":
                        bash_output = ""
                        for item in (result_data if isinstance(result_data, list) else []):
                            if isinstance(item, dict):
                                bash_output = item.get("output", item.get("error", ""))
                        result_text = bash_output[:3000] if bash_output else "(empty output)"
                        new_cands = []
                    else:
                        new_cands = self._build_candidates_from_single_tool(
                            tc.name, result_data, candidate_idx,
                        )
                        result_lines = [f"{c['id']}: {c['display']}" for c in new_cands]
                        result_text = "\n".join(result_lines) if result_lines else "(no results)"
                    candidate_idx += len(new_cands)
                    all_candidates.extend(new_cands)
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": result_text,
                    })
                    step_trace["tool_calls"].append({
                        "tool": tc.name, "arguments": tc.arguments,
                        "result_count": len(new_cands),
                        "result": result_text[:2000],
                    })

            # 处理 submit
            for tc in submit_calls:
                kept_ids = tc.arguments.get("keep", [])
                submit_raw = _json.dumps({"keep": kept_ids})
                tc_args_str = _json.dumps(tc.arguments, ensure_ascii=False)
                tc_args_chars = len(tc_args_str)
                tc_format_chars = len(tc.name) + len(getattr(tc, "id", "") or "") + 20
                tc_output_chars = tc_args_chars + tc_format_chars
                turn_output_chars += tc_output_chars
                turn_essential_chars += tc_args_chars  # submit 参数是 essential
                turn_overhead_chars += tc_format_chars  # 格式是 overhead
                token_stats["tool_call_type_counts"]["submit"] = \
                    token_stats["tool_call_type_counts"].get("submit", 0) + 1

                if kept_ids:
                    id_set = set(kept_ids)
                    id_order = {rid: i for i, rid in enumerate(kept_ids)}
                    final_candidates = [c for c in all_candidates if c["id"] in id_set]
                    final_candidates.sort(key=lambda c: id_order.get(c["id"], 999))
                else:
                    final_candidates = all_candidates

                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": f"Submitted {len(final_candidates)} entries.",
                })
                step_trace["tool_calls"].append({
                    "tool": "submit", "arguments": tc.arguments,
                    "keep_count": len(kept_ids),
                    "result": f"Submitted {len(final_candidates)} entries.",
                })
                should_break = True

            # 统计
            token_stats["total_output_chars"] += turn_output_chars
            token_stats["essential_output_chars"] += turn_essential_chars
            token_stats["overhead_output_chars"] += turn_overhead_chars
            token_stats["per_turn"].append({
                "turn": turn + 1, "type": "tool_calls",
                "input_chars": turn_input_chars,
                "output_chars": turn_output_chars,
                "essential_chars": turn_essential_chars,
                "overhead_chars": turn_overhead_chars,
                "tool_calls": len(turn_tool_calls),
                "tools": [tc.name for tc in turn_tool_calls],
            })

            agent_trace.append(step_trace)
            if should_break:
                break
        else:
            logger.warning("Retrieve agent loop reached max_turns=%d without submit", max_turns)
            # 未 submit，返回全部候选
            final_candidates = all_candidates

        # 格式化输出
        context = self._format_final_output(final_candidates)

        if evt:
            evt.log("task_trace", task=self.task_name, session_id=session_id,
                     steps_used=steps_used, finish_result=context,
                     agent_loop_trace=agent_trace,
                     token_stats=token_stats)

        return TaskResult(
            task_name=self.task_name,
            retrieved_context=context,
            finish_result=context,
            stats={
                "mode": "agent_loop",
                "tool_calls_count": sum(len(t.get("tool_calls", [])) for t in agent_trace),
                "candidates_total": len(all_candidates),
                "candidates_submitted": len(final_candidates),
                "turns_used": steps_used,
                "max_turns": max_turns,
                "submit_raw": submit_raw,
                "token_stats": token_stats,
            },
        )

    def _build_candidates_from_single_tool(
        self, tool_name: str, result_data: list, start_idx: int,
    ) -> list[dict[str, Any]]:
        """从单个 tool call 的检索结果构建候选列表（复用 _build_candidates_from_agent_tools 逻辑）。"""
        # 包装为 tool_results 格式然后调用已有方法
        fake_tool_results = [{"tool": tool_name, "args": {}, "results": result_data}]
        raw_candidates = self._build_candidates_from_agent_tools(fake_tool_results)
        # 重新编号
        for i, c in enumerate(raw_candidates):
            c["id"] = f"R{start_idx + i}"
        return raw_candidates

    # ------------------------------------------------------------------
    # AGENT 模式：Plan → Tool Call → Submit 三轮
    # ------------------------------------------------------------------

    def _format_tool_result_for_message(
        self,
        tool_name: str,
        result_data: list,
        candidates: list[dict[str, Any]],
    ) -> str:
        """将单个 tool 的检索结果格式化为 tool message content。

        在 content 中包含候选 R{n} ID，让 R2 的 LLM 能引用。
        """
        lines: list[str] = []

        if tool_name == "fs_bm25_search":
            for item in result_data:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    path, score, snippet = item[0], item[1], item[2]
                    # 查找对应的 candidate ID
                    rid = self._find_candidate_id(candidates, "fs", path=path)
                    snippet_short = snippet[:200].replace("\n", " ")
                    lines.append(f"{rid}: [FS] {path} (score={score:.2f}): {snippet_short}")

        elif tool_name == "vec_semantic_search":
            for r in result_data:
                if not isinstance(r, dict):
                    continue
                rid = self._find_candidate_id(candidates, "vec", vec_id=r.get("id"))
                text = r.get("text", "")[:200]
                score = r.get("score", 0)
                lines.append(f"{rid}: [VEC] (score={score:.3f}): {text}")

        elif tool_name == "graph_entity_search":
            for gr in result_data:
                if not isinstance(gr, dict):
                    continue
                subgraph = gr.get("subgraph", {})
                hops = subgraph.get("hops", {})
                if hops:
                    for d in sorted(hops.keys()):
                        for entry in hops[d]:
                            edge = entry.get("edge", {})
                            src = edge.get("source", "")
                            rel = edge.get("relation", "").replace("_", " ")
                            tgt = edge.get("target", "")
                            ek = f"{src}|{edge.get('relation', '')}|{tgt}"
                            rid = self._find_candidate_id(candidates, "graph", edge_key=ek)
                            lines.append(f"{rid}: [GRAPH] {src} --[{rel}]--> {tgt} (hop={d})")

        elif tool_name == "fs_read_file":
            for item in result_data:
                if not isinstance(item, dict):
                    continue
                path = item.get("path", "")
                rid = self._find_candidate_id(candidates, "fs", path=path)
                content = item.get("content", "")
                lines.append(f"{rid}: [FS] {path} ({len(content)} chars)")

        return "\n".join(lines) if lines else "(no results)"

    @staticmethod
    def _find_candidate_id(
        candidates: list[dict[str, Any]],
        source: str,
        path: str = "",
        vec_id: str = "",
        edge_key: str = "",
    ) -> str:
        """根据检索结果的标识信息，查找对应的 candidate R{n} ID。"""
        for c in candidates:
            if c["source"] != source and not (source == "graph" and c["source"] in ("graph", "graph_edge")):
                continue
            orig = c.get("original", {})
            if source == "fs" and path and orig.get("path") == path:
                return c["id"]
            if source == "vec" and vec_id and orig.get("id") == vec_id:
                return c["id"]
            if source == "graph" and edge_key:
                edge = orig.get("edge", {})
                ek = f"{edge.get('source', '')}|{edge.get('relation', '')}|{edge.get('target', '')}"
                if ek == edge_key:
                    return c["id"]
        return "R?"

    def _normalize_fs_path(self, path: Any) -> str:
        """Normalize paths copied from displayed trees such as `filesystem/...`."""
        p = str(path or "").strip()
        while p.startswith("./"):
            p = p[2:]
        p = p.lstrip("/")
        if p == "filesystem":
            return ""
        if p.startswith("filesystem/"):
            return p[len("filesystem/"):]
        return p

    def _normalize_fs_paths(self, paths: Any) -> Any:
        if isinstance(paths, list):
            return [self._normalize_fs_path(p) for p in paths]
        return self._normalize_fs_path(paths)

    # ------------------------------------------------------------------
    # AGENT 模式：Tool 执行器
    # ------------------------------------------------------------------

    async def _execute_agent_tool(
        self, tool_name: str, args: dict[str, Any],
    ) -> list[dict[str, Any]] | list[tuple]:
        """执行 AGENT 模式的单个检索 tool call。

        将 LLM 发出的 tool call 映射到内部检索方法。
        """
        if tool_name == "fs_bm25_search":
            return self._agent_tool_fs_search(args)
        elif tool_name == "vec_semantic_search":
            return await self._agent_tool_vec_search(args)
        elif tool_name == "graph_entity_search":
            return await self._agent_tool_graph_search(args)
        elif tool_name in {"fs_read_file", "fs_read", "read_file"}:
            return self._agent_tool_fs_read(args)
        elif tool_name == "fs_execute_bash":
            return self._agent_tool_bash(args)
        else:
            logger.warning("AGENT: unknown tool %r", tool_name)
            return []

    def _agent_tool_fs_search(self, args: dict[str, Any]) -> list[tuple[str, float, str]]:
        """AGENT tool: fs_bm25_search → 内部 BM25 检索。"""
        query_text = args.get("query", "")
        top_k = args.get("top_k", self.fs_top_k)
        scope = args.get("scope", None)
        temporal_filter = args.get("temporal_filter", None)

        if not query_text:
            return []

        # 构建 queries 格式以复用 _search_fs
        queries: list[dict[str, str]] = [
            {"type": "keyword", "text": query_text},
            {"type": "semantic", "text": query_text},
        ]
        if temporal_filter:
            queries.append({"type": "temporal", "text": temporal_filter})
        if scope:
            norm_scope = self._normalize_fs_paths(scope)
            if isinstance(norm_scope, list):
                queries.append({"type": "fs_scope", "text": ",".join(norm_scope)})
            elif norm_scope:
                queries.append({"type": "fs_scope", "text": norm_scope})

        # 暂存并恢复 fs_top_k
        orig_top_k = self.fs_top_k
        self.fs_top_k = top_k
        try:
            results = self._search_fs(queries)
        finally:
            self.fs_top_k = orig_top_k

        return results

    async def _agent_tool_vec_search(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        """AGENT tool: vec_semantic_search → 内部向量检索。"""
        query_text = args.get("query", "")
        top_k = args.get("top_k", self.vec_top_k)
        temporal_filter = args.get("temporal_filter", None)

        if not query_text:
            return []

        queries: list[dict[str, str]] = [
            {"type": "semantic", "text": query_text},
        ]
        if temporal_filter:
            queries.append({"type": "temporal", "text": temporal_filter})

        orig_top_k = self.vec_top_k
        self.vec_top_k = top_k
        try:
            results = await self._search_vec(queries)
        finally:
            self.vec_top_k = orig_top_k

        return results

    async def _agent_tool_graph_search(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        """AGENT tool: graph_entity_search → 内部图检索。"""
        keyword = args.get("keyword", "")
        semantic_query = args.get("semantic_query", "")
        max_hops = args.get("max_hops", self.graph_subgraph_depth)

        if not keyword and not semantic_query:
            return []

        queries: list[dict[str, str]] = []
        if keyword:
            queries.append({"type": "entity", "text": keyword})
        if semantic_query:
            queries.append({"type": "semantic", "text": semantic_query})

        orig_depth = self.graph_subgraph_depth
        self.graph_subgraph_depth = max_hops
        try:
            results = await self._search_graph(queries)
        finally:
            self.graph_subgraph_depth = orig_depth

        return results

    def _agent_tool_fs_read(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        """AGENT tool: fs_read_file → 读取单个文件。"""
        path = self._normalize_fs_path(args.get("path", ""))
        if not path:
            return []

        try:
            content = self.fs.read_file(path)
            if content and not content.startswith("ERROR"):
                meta_desc, body_start = self._parse_file_metadata(content)
                return [{"path": path, "content": content, "meta": meta_desc}]
            else:
                return [{"path": path, "error": content or "file not found"}]
        except Exception as e:
            return [{"path": path, "error": str(e)}]

    def _agent_tool_bash(self, args: dict[str, Any]) -> list[dict[str, Any]]:
        """AGENT tool: fs_execute_bash → 执行 shell 命令。"""
        command = args.get("command", "")
        if not command:
            return [{"error": "fs_execute_bash requires 'command'"}]
        try:
            output = self.fs.execute_bash(
                command=command,
                timeout=args.get("timeout", 15),
                allow_write=False,  # retrieve 只读
            )
            return [{"command": command, "output": output[:3000]}]
        except Exception as e:
            return [{"command": command, "error": str(e)}]

    # ------------------------------------------------------------------
    # AGENT 模式：候选构建
    # ------------------------------------------------------------------

    def _build_candidates_from_agent_tools(
        self, tool_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """从 AGENT tool call 结果构建统一候选列表。"""
        candidates: list[dict[str, Any]] = []
        idx = 1

        for tr in tool_results:
            tool_name = tr.get("tool", "")
            results = tr.get("results", [])

            if tool_name == "fs_bm25_search":
                for item in results:
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        path, score, snippet = item[0], item[1], item[2]
                        rid = f"R{idx}"
                        display, detail = self._fs_to_display(path, score, snippet)
                        candidates.append({
                            "id": rid, "source": "fs",
                            "display": display, "detail": detail,
                            "original": {"path": path, "score": score, "snippet": snippet},
                        })
                        idx += 1

            elif tool_name == "vec_semantic_search":
                for r in results:
                    if isinstance(r, dict):
                        rid = f"R{idx}"
                        display, detail = self._vec_to_display(r)
                        candidates.append({
                            "id": rid, "source": "vec",
                            "display": display, "detail": detail, "original": r,
                        })
                        idx += 1

            elif tool_name == "graph_entity_search":
                seen_edges: set[str] = set()
                for gr in results:
                    if not isinstance(gr, dict):
                        continue
                    subgraph = gr.get("subgraph", {})
                    hops = subgraph.get("hops", {})
                    seed_node = gr.get("node", {})
                    if hops:
                        for d in sorted(hops.keys()):
                            for entry in hops[d]:
                                edge = entry.get("edge", {})
                                ek = f"{edge.get('source', '')}|{edge.get('relation', '')}|{edge.get('target', '')}"
                                if ek in seen_edges:
                                    continue
                                seen_edges.add(ek)
                                rid = f"R{idx}"
                                display, detail = self._graph_edge_to_display(edge, d, seed_node)
                                candidates.append({
                                    "id": rid, "source": "graph",
                                    "display": display, "detail": detail,
                                    "original": {"edge": edge, "hop": d, "node": seed_node},
                                })
                                idx += 1
                    else:
                        for edge in subgraph.get("edges", []):
                            if not isinstance(edge, dict):
                                continue
                            ek = f"{edge.get('source', '')}|{edge.get('relation', '')}|{edge.get('target', '')}"
                            if ek in seen_edges:
                                continue
                            seen_edges.add(ek)
                            rid = f"R{idx}"
                            display, detail = self._graph_edge_to_display(edge, 0, seed_node)
                            candidates.append({
                                "id": rid, "source": "graph",
                                "display": display, "detail": detail,
                                "original": {"edge": edge, "hop": 0, "node": seed_node},
                            })
                            idx += 1

            elif tool_name == "fs_read_file":
                for item in results:
                    if not isinstance(item, dict):
                        continue
                    path = item.get("path", "")
                    content = item.get("content", "")
                    error = item.get("error", "")
                    if error:
                        continue
                    meta_desc = item.get("meta", "")
                    rid = f"R{idx}"
                    desc_tag = f" ({meta_desc})" if meta_desc and meta_desc != "(no meta info)" else ""
                    display = f"[FS] {path}{desc_tag}: {content[:150].replace(chr(10), ' ')}"
                    # 截断内容
                    body = content[:self.FS_DETAIL_BUDGET]
                    if len(content) > self.FS_DETAIL_BUDGET:
                        body += f"\n... ({len(content)} chars total)"
                    detail = f"[FS] {path}{desc_tag}\n{body}"
                    candidates.append({
                        "id": rid, "source": "fs",
                        "display": display, "detail": detail,
                        "original": {"path": path, "score": 1.0, "snippet": content[:300]},
                    })
                    idx += 1

        return candidates

    def _rewrite_queries_engineering(self, question: str) -> list[dict[str, str]]:
        """工程化 query 改写：分词 + 停用词过滤 + 同义词扩展。

        生成三类 query：
        - semantic: 原始问题 + 变体（用于 vec 语义检索）
        - keyword: 关键词组合（用于 fs BM25）
        - entity: 实体名（用于 graph 检索）
        """
        queries: list[dict[str, str]] = []

        # 1. 语义 query：直接使用原始问题
        queries.append({"type": "semantic", "text": question})

        # 2. 关键词提取
        keywords = self._extract_keywords_engineering(question)
        if keywords:
            # 关键词组合作为 BM25 query
            queries.append({"type": "keyword", "text": " ".join(keywords)})
            # 每个重要关键词也单独作为 query
            for kw in keywords[:5]:
                queries.append({"type": "entity", "text": kw})

        # 3. 尝试 jieba 分词（如果可用）
        jieba_keywords = self._jieba_segment(question)
        if jieba_keywords:
            queries.append({"type": "keyword", "text": " ".join(jieba_keywords)})
            # jieba 提取的名词/实体
            for kw in jieba_keywords:
                if kw not in [q["text"] for q in queries if q["type"] == "entity"]:
                    queries.append({"type": "entity", "text": kw})

        # 4. 同义词扩展（简单规则）
        expanded = self._expand_synonyms(question)
        if expanded and expanded != question:
            queries.append({"type": "semantic", "text": expanded})

        # 5. 时间提取（temporal query）
        temporal = self._extract_temporal(question)
        if temporal:
            queries.append({"type": "temporal", "text": temporal})

        # 6. FS 路径范围推断（fs_scope）
        fs_scope = self._infer_fs_scope(queries)
        if fs_scope:
            queries.append({"type": "fs_scope", "text": ",".join(fs_scope)})

        return queries

    def _infer_fs_scope(self, queries: list[dict[str, str]]) -> list[str] | None:
        """基于已提取的实体名，推断 FS 搜索路径范围。

        策略：检查 entity query 中的名称是否对应已有的目录/文件路径。
        例如 entity="alex" → 如果存在 "people/alex/" 目录则加入 scope。
        """
        try:
            all_files = self.fs.list_files()
        except Exception:
            return None

        if not all_files:
            return None

        # 提取所有一级目录
        top_dirs: set[str] = set()
        for f in all_files:
            parts = f.split("/")
            if len(parts) > 1:
                top_dirs.add(parts[0])

        # 从 entity queries 中提取实体名
        entities: list[str] = []
        for q in queries:
            if q["type"] == "entity":
                entities.append(q["text"].lower())

        if not entities:
            return None

        # 匹配策略：检查实体名是否出现在文件路径中
        matched_paths: set[str] = set()
        for entity in entities:
            for f in all_files:
                f_lower = f.lower()
                # 匹配目录名或文件名中包含实体名的路径
                parts = f_lower.split("/")
                for i, part in enumerate(parts):
                    # 去掉扩展名后比较
                    name = part.rsplit(".", 1)[0] if "." in part else part
                    if entity == name or entity in name:
                        # 匹配到了，加入对应的目录前缀
                        if i == 0:
                            matched_paths.add(f"{parts[0]}/")
                        elif i == 1:
                            matched_paths.add(f"{parts[0]}/{parts[1]}/")
                        break

        # 如果匹配到的路径太多（>5），说明推断不够精确，放弃
        if len(matched_paths) > 5:
            return None

        return list(matched_paths) if matched_paths else None

    @staticmethod
    def _extract_temporal(text: str) -> str | None:
        """从文本中提取时间表达式，转为 graph 时间查询格式。

        支持的模式：
        - "in 2017", "2017年", "2019-03" → "2017", "2019-03"
        - "from 2018 to 2020", "2018到2020" → "2018~2020"
        - "between 2017 and 2019" → "2017~2019"
        """
        import re as _re

        # 范围模式：from X to Y / between X and Y / X到Y / X~Y / X-Y年
        range_patterns = [
            r'(?:from|between)\s+(\d{4}(?:-\d{2})?(?:-\d{2})?)\s+(?:to|and)\s+(\d{4}(?:-\d{2})?(?:-\d{2})?)',
            r'(\d{4}(?:-\d{2})?(?:-\d{2})?)\s*[到至~]\s*(\d{4}(?:-\d{2})?(?:-\d{2})?)',
        ]
        for pattern in range_patterns:
            m = _re.search(pattern, text, _re.IGNORECASE)
            if m:
                return f"{m.group(1)}~{m.group(2)}"

        # 单值模式：in 2017 / 2017年 / 2019-03 / 2019年3月
        single_patterns = [
            r'(?:in|around|circa|during)\s+(\d{4}-\d{2}-\d{2})',
            r'(?:in|around|circa|during)\s+(\d{4}-\d{2})',
            r'(?:in|around|circa|during)\s+(\d{4})',
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{4}-\d{2})(?!\d)',
            r'(\d{4})年(\d{1,2})月',
            r'(\d{4})年',
            r'\b(\d{4})\b',
        ]
        for pattern in single_patterns:
            m = _re.search(pattern, text, _re.IGNORECASE)
            if m:
                groups = m.groups()
                if len(groups) == 2 and groups[1]:
                    # "2019年3月" → "2019-03"
                    return f"{groups[0]}-{int(groups[1]):02d}"
                return groups[0]

        return None

    def _extract_keywords_engineering(self, text: str) -> list[str]:
        """工程化关键词提取。"""
        stop_words_en = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "then", "when",
            "where", "why", "how", "all", "each", "every", "both",
            "few", "more", "most", "other", "some", "no", "not", "only",
            "very", "just", "but", "and", "or", "if", "about", "what",
            "which", "who", "this", "that", "these", "those", "it", "its",
            "i", "me", "my", "we", "our", "you", "your", "he", "him",
            "she", "her", "they", "them", "their", "what's", "how's",
            "does", "don't", "doesn't", "didn't", "won't", "wouldn't",
            "tell", "know", "think", "like", "want", "need", "please",
            "can", "could", "would", "should",
        }
        stop_words_zh = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
            "吗", "什么", "那", "还", "能", "把", "让", "给", "从", "们",
            "呢", "吧", "啊", "哦", "嗯", "呀", "哈", "嘛",
        }

        # 英文 token
        en_words = re.findall(r"\b[a-zA-Z']+\b", text)
        en_keywords = [w for w in en_words if w.lower() not in stop_words_en and len(w) > 2]

        # CJK 连续段
        cjk_runs = re.findall(
            r"[\u4e00-\u9fff]{2,}",
            text,
        )
        # 过滤中文停用词
        cjk_keywords = [w for w in cjk_runs if w not in stop_words_zh]

        # 去重保序
        seen: set[str] = set()
        unique: list[str] = []
        for kw in en_keywords + cjk_keywords:
            lk = kw.lower()
            if lk in seen:
                continue
            seen.add(lk)
            unique.append(kw)

        return unique[:15]

    @staticmethod
    def _jieba_segment(text: str) -> list[str]:
        """使用 jieba 分词提取关键词（如果 jieba 可用）。"""
        try:
            import jieba
            import jieba.analyse
            # 使用 TF-IDF 提取关键词
            keywords = jieba.analyse.extract_tags(text, topK=10, withWeight=False)
            return [kw for kw in keywords if len(kw) > 1]
        except ImportError:
            # jieba 不可用，返回空
            return []

    @staticmethod
    def _expand_synonyms(text: str) -> str:
        """简单的同义词扩展规则。"""
        # 常见同义词映射
        synonym_map = {
            "like": "enjoy prefer love",
            "dislike": "hate avoid don't like",
            "hobby": "interest activity pastime",
            "food": "cuisine dish meal cooking",
            "movie": "film cinema show",
            "music": "song album artist band",
            "book": "novel reading literature",
            "sport": "exercise fitness workout",
            "travel": "trip journey vacation",
            "work": "job career profession",
            "喜欢": "爱好 偏好 热爱",
            "讨厌": "不喜欢 厌恶 反感",
            "爱好": "兴趣 喜好 偏好",
        }

        expanded_parts = [text]
        text_lower = text.lower()
        for key, synonyms in synonym_map.items():
            if key in text_lower:
                expanded_parts.append(synonyms)
                break  # 只扩展第一个匹配

        return " ".join(expanded_parts) if len(expanded_parts) > 1 else text

    # ------------------------------------------------------------------
    # Query 改写方法 B：LLM
    # ------------------------------------------------------------------

    def _read_index_md(self) -> str:
        """读取 .meta/index.md 作为提示词级目录。"""
        try:
            if hasattr(self.fs, "read_index"):
                _, content = self.fs.read_index()
            else:
                content = self.fs.read_file(".meta/index.md")
            if isinstance(content, str) and content.startswith("ERROR"):
                return "(empty memory — no index available)"
            return content
        except Exception:
            return "(empty memory — no index available)"

    def _build_fs_structure_with_meta(self) -> str:
        """构建带 meta 描述信息的文件系统结构摘要。

        遍历所有文件，读取每个文件的 meta description，构建类似：
        filesystem/
          people/
            alex/
              preferences.md — Stores Alex's preferences about food, music, and hobbies
              activities.md — Records Alex's regular activities and routines
              events.md — Notable events and experiences in Alex's life
        """
        try:
            all_files = self.fs.list_files("")
        except Exception:
            # fallback 到简单的 tree
            try:
                tree = self.fs.tree(max_depth=2)
                return tree if tree else "(not available)"
            except Exception:
                return "(not available)"

        if not all_files:
            return "(empty filesystem)"

        # 限制文件数量，避免 prompt 过长
        if len(all_files) > 50:
            all_files = all_files[:50]
            truncated = True
        else:
            truncated = False

        # 按目录分组
        dir_files: dict[str, list[tuple[str, str]]] = {}  # dir -> [(filename, meta_desc)]
        for fpath in all_files:
            try:
                content = self.fs.read_file(fpath)
                if content.startswith("ERROR"):
                    meta_desc = ""
                else:
                    meta_desc, _ = self._parse_file_metadata(content)
            except Exception:
                meta_desc = ""

            parts = fpath.split("/")
            if len(parts) > 1:
                dir_key = "/".join(parts[:-1])
                filename = parts[-1]
            else:
                dir_key = ""
                filename = fpath

            if dir_key not in dir_files:
                dir_files[dir_key] = []
            dir_files[dir_key].append((filename, meta_desc))

        # 构建输出
        lines: list[str] = ["filesystem/"]
        for dir_path in sorted(dir_files.keys()):
            if dir_path:
                # 展示目录层级
                depth = dir_path.count("/") + 1
                indent = "  " * depth
                lines.append(f"{indent}{dir_path}/")
            else:
                indent = "  "

            file_indent = "  " * (dir_path.count("/") + 2) if dir_path else "  "
            for filename, meta_desc in sorted(dir_files[dir_path]):
                if meta_desc and meta_desc != "(no meta info)":
                    lines.append(f"{file_indent}{filename} — {meta_desc}")
                else:
                    lines.append(f"{file_indent}{filename}")

        result = "\n".join(lines)
        if truncated:
            result += "\n  ... (truncated, showing first 50 files)"
        return result

    def _search_fs(self, queries: list[dict[str, str]]) -> list[tuple[str, float, str]]:
        """在文件系统中执行 BM25 检索。

        支持：
        - fs_scope query：指定搜索路径范围（目录前缀或文件路径），避免全量搜索
        - temporal query：对结果按行级时间前缀过滤
        没有时间前缀（[undated]）的行默认保留。
        """
        # 提取时间过滤条件
        temporal_filter: str | None = None
        for q in queries:
            if q["type"] == "temporal":
                temporal_filter = q["text"]
                break

        # 提取 fs_scope（搜索路径范围）
        fs_scope: list[str] | None = None
        for q in queries:
            if q["type"] == "fs_scope":
                scope_text = q["text"].strip()
                if scope_text:
                    # 支持逗号分隔的多个路径
                    fs_scope = [self._normalize_fs_path(s.strip()) for s in scope_text.split(",") if s.strip()]
                break

        all_results: dict[str, tuple[float, str]] = {}  # path -> (best_score, snippet)

        for q in queries:
            if q["type"] in ("semantic", "keyword"):
                try:
                    results = self.fs.search_bm25(q["text"], top_k=self.fs_top_k, scope=fs_scope)
                    for path, score, snippet in results:
                        if path not in all_results or score > all_results[path][0]:
                            all_results[path] = (score, snippet)
                except Exception as e:
                    logger.warning("FS BM25 search failed for query=%r: %s", q["text"], e)

        # 按分数排序
        sorted_results = sorted(
            [(path, score, snippet) for path, (score, snippet) in all_results.items()],
            key=lambda x: -x[1],
        )

        # 时间过滤：对文件内容按行级时间前缀过滤
        if temporal_filter:
            filtered = []
            for path, score, snippet in sorted_results:
                # 读取文件内容，按行过滤
                try:
                    content = self.fs.read_file(path)
                    if content.startswith("ERROR"):
                        filtered.append((path, score, snippet))
                        continue
                    # 按行过滤：保留匹配时间的行和无时间标记的行
                    lines = content.split("\n")
                    kept_lines = []
                    for line in lines:
                        line_time = self._extract_fs_line_time(line)
                        if line_time is None:
                            # 无时间前缀的行（标题、空行等）默认保留
                            kept_lines.append(line)
                        elif line_time == "":
                            # [undated] 标记的行默认保留
                            kept_lines.append(line)
                        elif self._time_matches_filter(line_time, temporal_filter):
                            kept_lines.append(line)
                    if kept_lines:
                        filtered_snippet = "\n".join(kept_lines[:5])
                        filtered.append((path, score, filtered_snippet))
                except Exception:
                    filtered.append((path, score, snippet))
            sorted_results = filtered

        return sorted_results[:self.fs_top_k]

    @staticmethod
    def _extract_fs_line_time(line: str) -> str | None:
        """从 FS 行中提取时间前缀。

        格式：[YYYY-MM-DD] 或 [YYYY-MM] 或 [YYYY] 或 [undated]
        返回：时间字符串（如 "2019-03"），"" 表示 undated，None 表示无时间前缀。
        """
        import re as _re
        stripped = line.strip()
        if not stripped:
            return None
        # 匹配 [时间] 前缀
        m = _re.match(r'^\[(\d{4}(?:-\d{2})?(?:-\d{2})?)\]', stripped)
        if m:
            return m.group(1)
        if stripped.startswith("[undated]"):
            return ""
        # 也匹配 "- [时间]" 格式（列表项）
        m = _re.match(r'^-\s*\[(\d{4}(?:-\d{2})?(?:-\d{2})?)\]', stripped)
        if m:
            return m.group(1)
        if _re.match(r'^-\s*\[undated\]', stripped):
            return ""
        return None

    async def _search_vec(self, queries: list[dict[str, str]]) -> list[dict[str, Any]]:
        """在向量 DB 中执行语义检索。

        优化：将所有 semantic query 去重后一次性批量 embed（单次 API 调用），
        然后用预计算的 embedding 在本地做余弦相似度计算，避免 N×M 次重复
        embedding API 调用（N=semantic query 数, M=collection 数）。

        时间过滤：如果 queries 中包含 temporal query，则对结果按时间过滤。
        没有 occurred_at 的记录默认保留（不被过滤掉）。
        """
        # 提取时间过滤条件
        temporal_filter: str | None = None
        for q in queries:
            if q["type"] == "temporal":
                temporal_filter = q["text"]
                break

        # 收集所有 semantic query text 并去重
        semantic_texts: list[str] = []
        seen_texts: set[str] = set()
        for q in queries:
            if q["type"] == "semantic" and q["text"] not in seen_texts:
                semantic_texts.append(q["text"])
                seen_texts.add(q["text"])

        if not semantic_texts:
            return []

        # 一次性批量 embed 所有 query（单次 API 调用）
        query_embeddings: dict[str, list[float]] = {}
        embedder = getattr(self.vec, "embedder", None)
        if embedder:
            try:
                all_embs = await embedder.embed(semantic_texts)
                for text, emb in zip(semantic_texts, all_embs):
                    query_embeddings[text] = emb
            except Exception as e:
                logger.warning("Batch embedding failed, falling back to sequential: %s", e)
                # Fallback: 逐个 embed（降级但不中断）
                for text in semantic_texts:
                    try:
                        emb = await embedder.embed_single(text)
                        query_embeddings[text] = emb
                    except Exception as e2:
                        logger.warning("embed_single failed for %r: %s", text[:50], e2)
        else:
            logger.error("vec.embedder is None — vec semantic search disabled, embedding 服务无法访问")

        # 用预计算的 embedding 在本地做余弦相似度，避免重复 API 调用
        all_results: dict[str, dict[str, Any]] = {}  # id -> result

        if query_embeddings and hasattr(self.vec, "_collections"):
            # 内存后端：直接访问 _collections 做本地余弦相似度
            for q_text, q_emb in query_embeddings.items():
                if not q_emb:
                    continue
                for coll_name, entries in self.vec._collections.items():
                    for entry in entries:
                        entry_emb = entry.get("embedding")
                        if not entry_emb:
                            continue
                        score = VectorStore._cosine_similarity(q_emb, entry_emb)
                        rid = entry.get("id", "")
                        if rid and (rid not in all_results or score > all_results[rid].get("score", 0)):
                            all_results[rid] = {
                                "id": rid,
                                "text": entry.get("text", ""),
                                "score": score,
                                "metadata": entry.get("metadata", {}),
                                "collection": coll_name,
                            }
        else:
            # PG 后端：使用 search_all_with_embedding 避免重复 embed
            if query_embeddings and hasattr(self.vec, "search_all_with_embedding"):
                for q_text, q_emb in query_embeddings.items():
                    if not q_emb:
                        continue
                    try:
                        results = await self.vec.search_all_with_embedding(q_emb, top_k=self.vec_top_k)
                        for r in results:
                            rid = r.get("id", "")
                            if rid and (rid not in all_results or r.get("score", 0) > all_results[rid].get("score", 0)):
                                all_results[rid] = r
                    except Exception as e:
                        logger.warning("Vec search_all_with_embedding failed: %s", e)
            else:
                # 最终 fallback：无 embedder 或无 search_all_with_embedding 方法
                for q_text in semantic_texts:
                    try:
                        results = await self.vec.search_all(q_text, top_k=self.vec_top_k)
                        for r in results:
                            rid = r.get("id", "")
                            if rid and (rid not in all_results or r.get("score", 0) > all_results[rid].get("score", 0)):
                                all_results[rid] = r
                    except Exception as e:
                        logger.warning("Vec search failed for query=%r: %s", q_text, e)

        # 按分数排序
        sorted_results = sorted(all_results.values(), key=lambda x: -x.get("score", 0))

        # 时间过滤：如果有 temporal query，过滤结果
        # 规则：没有 occurred_at 或 occurred_at 为空的记录默认保留（不被过滤）
        if temporal_filter:
            filtered = []
            for r in sorted_results:
                meta = r.get("metadata", {}) or {}
                occurred_at = meta.get("occurred_at", "")
                if not occurred_at:
                    # 无时间信息的记录默认保留
                    filtered.append(r)
                elif self._time_matches_filter(occurred_at, temporal_filter):
                    filtered.append(r)
            sorted_results = filtered

        return sorted_results[:self.vec_top_k]

    @staticmethod
    def _time_matches_filter(occurred_at: str, time_filter: str) -> bool:
        """判断 occurred_at 是否匹配时间过滤条件。

        支持：
        - "2017" → occurred_at 以 "2017" 开头
        - "2018~2020" → occurred_at 在范围内
        - "recent" / "current" → 非 deprecated/past 的 dated/current 记录优先保留
        - "past" / "previous" → past/deprecated/used_to 记录
        """
        if not occurred_at or not time_filter:
            return True

        occurred_l = str(occurred_at).strip().lower()
        filter_l = time_filter.strip().lower()

        if filter_l in {"recent", "current", "now", "latest"}:
            return not any(x in occurred_l for x in ("past", "previous", "deprecated", "used_to"))
        if filter_l in {"past", "previous", "former", "deprecated", "used_to"}:
            return any(x in occurred_l for x in ("past", "previous", "deprecated", "used_to"))

        if "~" in time_filter:
            parts = time_filter.split("~", 1)
            time_start = parts[0].strip()
            time_end = parts[1].strip()
            # 范围匹配
            min_len = min(len(time_start), len(time_end))
            occurred_prefix = occurred_at[:min_len]
            return time_start <= occurred_prefix <= time_end
        else:
            # 前缀匹配
            return occurred_at.startswith(time_filter)

    async def _search_graph(self, queries: list[dict[str, str]]) -> list[dict[str, Any]]:
        """在图 DB 中执行向量增强检索。

        策略：
        1. 用 semantic query 计算查询向量，通过向量相似度找种子节点
        2. 用 entity query 做关键词节点搜索（补充）
        3. 对种子节点，按跳数展开子图，每跳按相似度排序并截断
        """
        all_results: list[dict[str, Any]] = []
        seen_nodes: set[str] = set()

        # 收集所有 semantic query 的文本，用于计算查询向量
        semantic_texts = [q["text"] for q in queries if q["type"] == "semantic"]
        query_embedding: list[float] | None = None

        # 1. 向量相似度检索种子节点（优先）
        if not (semantic_texts and self.graph.embedder):
            if semantic_texts and not self.graph.embedder:
                logger.error("graph.embedder is None — graph vector search disabled, embedding 服务无法访问")
        if semantic_texts and self.graph.embedder:
            try:
                # 用第一个 semantic query 作为主查询向量
                if hasattr(self.graph.embedder, 'embed_query_single'):
                    query_embedding = await self.graph.embedder.embed_query_single(semantic_texts[0])
                else:
                    query_embedding = await self.graph.embedder.embed_single(semantic_texts[0])
                nodes = self.graph.search_nodes_by_embedding(
                    query_embedding=query_embedding,
                    top_k=self.graph_top_k,
                    threshold=self.graph_similarity_threshold,
                )
                for node in nodes:
                    if node["id"] in seen_nodes:
                        continue
                    seen_nodes.add(node["id"])
                    # 获取子图（带相似度排序）
                    subgraph = self._get_node_subgraph_with_similarity(
                        node["id"], query_embedding
                    )
                    all_results.append({
                        "node": node,
                        "subgraph": subgraph,
                        "seed_similarity": node.get("similarity", 0.0),
                    })
            except Exception as e:
                logger.warning("Graph vector search failed: %s", e)

        # 2. 关键词节点搜索（补充种子节点）
        for q in queries:
            if q["type"] == "entity":
                try:
                    nodes = self.graph.search_nodes(keyword=q["text"])
                    for node in nodes[:self.graph_top_k]:
                        if node["id"] in seen_nodes:
                            continue
                        seen_nodes.add(node["id"])
                        subgraph = self._get_node_subgraph_with_similarity(
                            node["id"], query_embedding
                        )
                        # 计算种子节点与查询的相似度
                        seed_sim = 0.0
                        if query_embedding:
                            node_emb = self.graph.get_node_embedding(node["id"])
                            if node_emb:
                                seed_sim = self._cosine_sim(query_embedding, node_emb)
                        all_results.append({
                            "node": node,
                            "subgraph": subgraph,
                            "seed_similarity": seed_sim,
                        })
                except Exception as e:
                    logger.warning("Graph search failed for entity=%r: %s", q["text"], e)

        # 3. 补充：用 keyword query 也搜一遍 graph
        for q in queries:
            if q["type"] == "keyword":
                words = q["text"].split()
                for word in words[:3]:
                    if len(word) < 2:
                        continue
                    try:
                        nodes = self.graph.search_nodes(keyword=word)
                        for node in nodes[:2]:
                            if node["id"] in seen_nodes:
                                continue
                            seen_nodes.add(node["id"])
                            subgraph = self._get_node_subgraph_with_similarity(
                                node["id"], query_embedding
                            )
                            seed_sim = 0.0
                            if query_embedding:
                                node_emb = self.graph.get_node_embedding(node["id"])
                                if node_emb:
                                    seed_sim = self._cosine_sim(query_embedding, node_emb)
                            all_results.append({
                                "node": node,
                                "subgraph": subgraph,
                                "seed_similarity": seed_sim,
                            })
                    except Exception:
                        pass

        # 4. 时间过滤检索（temporal query）
        for q in queries:
            if q["type"] == "temporal":
                try:
                    time_results = self.graph.search_by_time(time_query=q["text"])
                    # 将时间匹配的节点加入结果
                    for node in time_results.get("nodes", []):
                        if node["id"] in seen_nodes:
                            continue
                        seen_nodes.add(node["id"])
                        subgraph = self._get_node_subgraph_with_similarity(
                            node["id"], query_embedding
                        )
                        seed_sim = 0.0
                        if query_embedding:
                            node_emb = self.graph.get_node_embedding(node["id"])
                            if node_emb:
                                seed_sim = self._cosine_sim(query_embedding, node_emb)
                        all_results.append({
                            "node": node,
                            "subgraph": subgraph,
                            "seed_similarity": seed_sim,
                            "temporal_match": q["text"],
                        })
                    # 将时间匹配的边的端点节点也加入结果
                    for edge in time_results.get("edges", []):
                        for endpoint in (edge.get("source", ""), edge.get("target", "")):
                            if not endpoint or endpoint in seen_nodes:
                                continue
                            node = self.graph.get_node(endpoint)
                            if not node:
                                continue
                            seen_nodes.add(endpoint)
                            subgraph = self._get_node_subgraph_with_similarity(
                                endpoint, query_embedding
                            )
                            seed_sim = 0.0
                            if query_embedding:
                                node_emb = self.graph.get_node_embedding(endpoint)
                                if node_emb:
                                    seed_sim = self._cosine_sim(query_embedding, node_emb)
                            all_results.append({
                                "node": node,
                                "subgraph": subgraph,
                                "seed_similarity": seed_sim,
                                "temporal_match": q["text"],
                            })
                except Exception as e:
                    logger.warning("Graph temporal search failed for %r: %s", q["text"], e)

        # 按种子节点相似度降序排列
        all_results.sort(key=lambda x: x.get("seed_similarity", 0.0), reverse=True)
        return all_results[:self.graph_top_k * 2]

    def _get_node_subgraph_with_similarity(
        self, node_id: str, query_embedding: list[float] | None
    ) -> dict[str, Any]:
        """获取节点的子图，按跳数分层并按相似度排序截断。

        返回结构：
        {
            "nodes": [...],
            "edges": [...],
            "hops": {
                1: [{"edge": {...}, "neighbor_id": "...", "similarity": 0.85}, ...],
                2: [{"edge": {...}, "neighbor_id": "...", "similarity": 0.72}, ...],
            }
        }
        """
        try:
            raw_subgraph = self.graph.get_subgraph(node_id, depth=self.graph_subgraph_depth)
        except Exception as e:
            logger.warning("get_subgraph failed for %s: %s", node_id, e)
            # Fallback: 只获取直接邻居
            try:
                neighbors = self.graph.get_neighbors(node_id)
                return {"nodes": [], "edges": [], "neighbors": neighbors}
            except Exception:
                return {"nodes": [], "edges": []}

        edges = raw_subgraph.get("edges", [])
        nodes = raw_subgraph.get("nodes", [])

        if not edges:
            return raw_subgraph

        # BFS 确定每条边的跳数
        adj: dict[str, list[dict]] = {}
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            adj.setdefault(src, []).append(edge)
            adj.setdefault(tgt, []).append(edge)

        node_depth: dict[str, int] = {node_id: 0}
        queue = [node_id]
        edge_hop: dict[str, int] = {}  # edge_id -> hop_number
        edge_neighbor: dict[str, str] = {}  # edge_id -> 该边连接的远端节点

        while queue:
            current = queue.pop(0)
            for edge in adj.get(current, []):
                eid = edge.get("id", "")
                if eid in edge_hop:
                    continue
                src = edge.get("source", "")
                tgt = edge.get("target", "")
                neighbor = tgt if src == current else src
                depth = node_depth[current] + 1
                edge_hop[eid] = depth
                edge_neighbor[eid] = neighbor
                if neighbor not in node_depth:
                    node_depth[neighbor] = depth
                    queue.append(neighbor)

        # 按跳数分组，每跳内按相似度排序
        hops: dict[int, list[dict[str, Any]]] = {}
        max_depth = max(edge_hop.values()) if edge_hop else 0

        for d in range(1, max_depth + 1):
            hop_entries: list[dict[str, Any]] = []
            for edge in edges:
                if not isinstance(edge, dict):
                    continue
                eid = edge.get("id", "")
                if edge_hop.get(eid) != d:
                    continue
                neighbor_id = edge_neighbor.get(eid, "")

                # 分跳策略：
                # - 1跳：优先用边 embedding（更精确），fallback 到邻居节点 embedding
                # - 2+跳：不做 embedding 筛选（语义偏移导致无效），仅靠数量截断
                sim = 0.0
                if query_embedding and d == 1:
                    # 1跳：优先使用边的全信息 embedding（source relation target）
                    edge_emb = self.graph.get_edge_embedding(eid) if hasattr(self.graph, 'get_edge_embedding') else None
                    if edge_emb:
                        sim = self._cosine_sim(query_embedding, edge_emb)
                    elif neighbor_id:
                        # Fallback: 使用邻居节点 embedding
                        neighbor_emb = self.graph.get_node_embedding(neighbor_id)
                        if neighbor_emb:
                            sim = self._cosine_sim(query_embedding, neighbor_emb)
                # 2+跳：sim 保持 0.0，不做 embedding 筛选

                hop_entries.append({
                    "edge": edge,
                    "neighbor_id": neighbor_id,
                    "similarity": sim,
                })

            # 1跳：按相似度降序排列（embedding 筛选有效）
            # 2+跳：保持原始顺序（不做 embedding 排序，靠数量截断 + 精排兜底）
            if d == 1:
                hop_entries.sort(key=lambda x: x["similarity"], reverse=True)

            # 按配置的 hop_top_k 截断
            effective_hop_top_k = self.graph_hop_top_k or {1: 10, 2: 15, 3: 5}
            if d in effective_hop_top_k:
                hop_entries = hop_entries[:effective_hop_top_k[d]]

            # 过滤低相似度的边（仅对 1 跳有效，2+跳 sim=0 不会被过滤）
            if query_embedding and d == 1:
                hop_entries = [
                    e for e in hop_entries
                    if e["similarity"] >= self.graph_edge_min_similarity or e["similarity"] == 0.0
                ]

            hops[d] = hop_entries

        return {
            "nodes": nodes,
            "edges": edges,
            "hops": hops,
        }

    def _cosine_sim(vec_a: list[float], vec_b: list[float]) -> float:
        """计算两个向量的余弦相似度。"""
        import math
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a)) or 1e-10
        norm_b = math.sqrt(sum(b * b for b in vec_b)) or 1e-10
        return dot / (norm_a * norm_b)

    # ------------------------------------------------------------------
    # Cross-Encoder 精排（BGE Reranker v2 M3）
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_file_metadata(content: str) -> tuple[str, int]:
        """解析文件第一行的 JSON 元数据。

        元数据格式：文件第一行是一个 JSON 对象，如 {"description": "..."}
        同时兼容旧格式（---\ndescription: ...\n---）。

        Returns:
            (description_text, body_start_line_index)
            body_start_line_index 是元数据之后正文开始的行索引（0-based）
        """
        import json as _json
        lines = content.split("\n")
        if not lines:
            return ("", 0)

        first_line = lines[0].strip()

        # 新格式：第一行是 JSON 对象
        if first_line.startswith("{"):
            try:
                meta = _json.loads(first_line)
                if isinstance(meta, dict):
                    description = meta.get("description", "")
                    return (description, 1)
            except _json.JSONDecodeError:
                pass

        # 兼容旧格式：---\ndescription: ...\n---
        if first_line == "---":
            for i in range(1, len(lines)):
                if lines[i].strip() == "---":
                    meta_lines = lines[1:i]
                    description = ""
                    for ml in meta_lines:
                        ml_stripped = ml.strip()
                        if ml_stripped.startswith("description:"):
                            description = ml_stripped[len("description:"):].strip()
                    return (description, i + 1)

        # 无元数据
        return ("", 0)

    # ------------------------------------------------------------------
    # Step 4 & 5: 统一候选构建 + Submit Filter
    # ------------------------------------------------------------------

    def _build_candidates_from_raw(
        self,
        fs_results: list[tuple[str, float, str]],
        vec_results: list[dict[str, Any]],
        graph_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """直接从三后端原始检索结果构建带 ID 的统一候选列表。

        不经过 RRF/Rerank，把排序和过滤完全交给 Round 2 submit LLM。
        每个候选: {"id": "R1", "source": "fs"|"vec"|"graph", "display": str, "detail": str, "original": Any}
        """
        candidates: list[dict[str, Any]] = []
        idx = 1

        # FS 条目（按 BM25 分数已排序，取 top_k）
        for path, score, snippet in fs_results[:self.fs_top_k]:
            rid = f"R{idx}"
            display, detail = self._fs_to_display(path, score, snippet)
            candidates.append({
                "id": rid, "source": "fs",
                "display": display, "detail": detail,
                "original": {"path": path, "score": score, "snippet": snippet},
            })
            idx += 1

        # Vec 条目（按相似度已排序，取 top_k）
        for r in vec_results[:self.vec_top_k]:
            rid = f"R{idx}"
            display, detail = self._vec_to_display(r)
            candidates.append({
                "id": rid, "source": "vec",
                "display": display, "detail": detail, "original": r,
            })
            idx += 1

        # Graph 条目（展开为边，去重）
        seen_edges: set[str] = set()
        edge_count = 0
        for gr in graph_results:
            subgraph = gr.get("subgraph", {})
            hops = subgraph.get("hops", {})
            seed_node = gr.get("node", {})
            if hops:
                for d in sorted(hops.keys()):
                    for entry in hops[d]:
                        if edge_count >= self.output_graph_edges_top_k:
                            break
                        edge = entry.get("edge", {})
                        ek = f"{edge.get('source', '')}|{edge.get('relation', '')}|{edge.get('target', '')}"
                        if ek in seen_edges:
                            continue
                        seen_edges.add(ek)
                        rid = f"R{idx}"
                        display, detail = self._graph_edge_to_display(edge, d, seed_node)
                        candidates.append({
                            "id": rid, "source": "graph",
                            "display": display, "detail": detail,
                            "original": {"edge": edge, "hop": d, "node": seed_node},
                        })
                        idx += 1
                        edge_count += 1
            else:
                # Fallback: 直接从 edges 列表提取
                for edge in subgraph.get("edges", []):
                    if not isinstance(edge, dict) or edge_count >= self.output_graph_edges_top_k:
                        break
                    ek = f"{edge.get('source', '')}|{edge.get('relation', '')}|{edge.get('target', '')}"
                    if ek in seen_edges:
                        continue
                    seen_edges.add(ek)
                    rid = f"R{idx}"
                    display, detail = self._graph_edge_to_display(edge, 0, seed_node)
                    candidates.append({
                        "id": rid, "source": "graph",
                        "display": display, "detail": detail,
                        "original": {"edge": edge, "hop": 0, "node": seed_node},
                    })
                    idx += 1
                    edge_count += 1

        return candidates

    # FS 内容展示的预算参数
    FS_DETAIL_BUDGET: int = 2000        # detail 正文总字符预算
    FS_HEAD_BUDGET: int = 600           # 文件头部（标题/偏好摘要）保留预算
    FS_MATCH_CONTEXT_LINES: int = 5     # 每个匹配行向上/下扩展的上下文行数

    def _fs_to_display(self, path: str, score: float, snippet: str) -> tuple[str, str]:
        """FS 条目 → (submit 用一行摘要, main agent 用完整内容)。

        策略：以 BM25 匹配行为锚点的段落级上下文窗口，而非盲截前 N 字符。
        1. 保留文件头部结构信息（标题、偏好摘要），预算 FS_HEAD_BUDGET
        2. 从 snippet 中提取匹配关键词，在文件中定位匹配行
        3. 以匹配行为中心，向上/下扩展 FS_MATCH_CONTEXT_LINES 行，组成上下文窗口
        4. 按出现顺序去重合并窗口，保留 section header
        5. 总预算 FS_DETAIL_BUDGET
        """
        meta_desc = ""
        full_content = ""
        try:
            full_content = self.fs.read_file(path)
            if full_content and full_content.startswith("ERROR"):
                full_content = ""
        except Exception:
            pass

        if full_content:
            meta_desc, body_start = self._parse_file_metadata(full_content)
        else:
            body_start = 0

        desc_tag = f" ({meta_desc})" if meta_desc and meta_desc != "(no meta info)" else ""
        display = f"[FS] {path}{desc_tag}: {snippet[:150].replace(chr(10), ' ')}"

        if not full_content:
            detail = f"[FS] {path}{desc_tag}\n{snippet[:500]}"
            return (display, detail)

        all_lines = full_content.split("\n")
        body_lines = all_lines[body_start:]

        # 短文件：直接全量展示
        full_body = "\n".join(body_lines)
        if len(full_body) <= self.FS_DETAIL_BUDGET:
            detail = f"[FS] {path}{desc_tag}\n{full_body}"
            return (display, detail)

        # --- 长文件：段落级上下文窗口 ---
        # Step 1: 提取头部（到第一个 ## 或前 FS_HEAD_BUDGET 字符）
        head_lines: list[str] = []
        head_chars = 0
        for i, line in enumerate(body_lines):
            # 遇到二级标题且已有内容 → 头部结束
            if i > 0 and line.startswith("## ") and head_chars > 0:
                break
            head_lines.append(line)
            head_chars += len(line) + 1
            if head_chars >= self.FS_HEAD_BUDGET:
                break
        head_end_idx = len(head_lines)  # body_lines 中头部结束的索引

        # Step 2: 从 snippet 提取匹配关键词，定位匹配行
        match_tokens = self._extract_snippet_tokens(snippet)
        matched_indices: list[int] = []
        for i, line in enumerate(body_lines):
            if i < head_end_idx:
                continue  # 头部已包含，跳过
            line_lower = line.lower()
            if any(t in line_lower for t in match_tokens):
                matched_indices.append(i)

        # Step 3: 以匹配行为中心，扩展上下文窗口
        keep_set: set[int] = set()
        ctx = self.FS_MATCH_CONTEXT_LINES
        for mi in matched_indices:
            # 向上扩展时包含最近的 section header（# 或 ## 开头的行）
            start = max(head_end_idx, mi - ctx)
            # 再往上找 section header
            for j in range(start, -1, -1):
                if body_lines[j].startswith("#"):
                    start = j
                    break
            end = min(len(body_lines), mi + ctx + 1)
            for k in range(start, end):
                keep_set.add(k)

        # Step 4: 如果没有匹配行（snippet 关键词未命中），回退到原始策略
        if not matched_indices:
            truncated = full_body[:self.FS_DETAIL_BUDGET]
            if len(full_body) > self.FS_DETAIL_BUDGET:
                truncated += f"\n... ({len(full_body)} chars total)"
            detail = f"[FS] {path}{desc_tag}\n{truncated}"
            return (display, detail)

        # Step 5: 按顺序合并头部 + 匹配窗口，控制总预算
        kept_indices = sorted(keep_set)
        parts: list[str] = []
        chars_used = 0

        # 先放头部
        head_text = "\n".join(head_lines)
        parts.append(head_text)
        chars_used += len(head_text)

        # 再放匹配窗口（按原文顺序，连续区间之间用 "..." 分隔）
        prev_idx = head_end_idx - 1
        for ki in kept_indices:
            line = body_lines[ki]
            if chars_used + len(line) + 1 > self.FS_DETAIL_BUDGET:
                parts.append("... (truncated)")
                break
            if ki > prev_idx + 1:
                parts.append("...")  # 不连续区间的分隔
            parts.append(line)
            chars_used += len(line) + 1
            prev_idx = ki

        # 附加总长度提示
        if chars_used < len(full_body):
            parts.append(f"... ({len(full_body)} chars total)")

        detail = f"[FS] {path}{desc_tag}\n" + "\n".join(parts)
        return (display, detail)

    @staticmethod
    def _extract_snippet_tokens(snippet: str) -> list[str]:
        """从 BM25 snippet 中提取有意义的匹配关键词（用于定位匹配行）。"""
        import re as _re
        # snippet 格式: "line1 | line2 | line3" (BM25 匹配行拼接)
        # 提取所有长度 >= 3 的非停用词 token
        stop_words = {
            "the", "and", "for", "are", "but", "not", "you", "all",
            "was", "her", "his", "has", "had", "its", "our", "she",
            "him", "how", "who", "that", "this", "with", "from",
            "they", "been", "have", "will", "more", "when", "what",
            "some", "than", "them", "into", "each", "just", "also",
            "over", "such", "very", "your", "about", "which",
        }
        words = _re.findall(r"[a-zA-Z\u4e00-\u9fff]{2,}", snippet.lower())
        tokens = [w for w in words if w not in stop_words and len(w) >= 3]
        # 去重保序，最多取 10 个
        seen: set[str] = set()
        unique: list[str] = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                unique.append(t)
            if len(unique) >= 10:
                break
        return unique

    def _vec_to_display(self, r: dict[str, Any]) -> tuple[str, str]:
        """Vec 条目 → (submit 用一行摘要, main agent 用完整信息)。"""
        text = r.get("text", "")
        meta = r.get("metadata", {}) or {}
        occurred_at = meta.get("occurred_at", "")
        topic = meta.get("topic", "")
        etype = meta.get("type", "")

        time_tag = f" @{occurred_at}" if occurred_at else ""
        topic_tag = f" topic={topic}" if topic else ""
        type_tag = f" type={etype}" if etype else ""

        display = f"[VEC]{time_tag}{topic_tag}{type_tag}: {text[:180]}"
        detail = f"[VEC] id={r.get('id', '?')} coll={r.get('collection', '?')}{time_tag}{topic_tag}{type_tag}\n{text}"

        return (display, detail)

    def _graph_edge_to_display(
        self, edge: dict, hop: int, seed_node: dict
    ) -> tuple[str, str]:
        """Graph 边 → (submit 用一行摘要, main agent 用完整信息)。"""
        src = edge.get("source", "")
        rel = edge.get("relation", "").replace("_", " ")
        tgt = edge.get("target", "")
        props = edge.get("properties", {})
        occurred_at = props.get("occurred_at", "")
        reason = props.get("reason", "")

        time_tag = f" @{occurred_at}" if occurred_at else ""
        reason_tag = f" reason=\"{reason}\"" if reason else ""
        hop_tag = f" (hop={hop})" if hop > 1 else ""

        display = f"[GRAPH] {src} --[{rel}]--> {tgt}{time_tag}{reason_tag}{hop_tag}"
        detail = display  # graph 边本身就很简洁

        return (display, detail)

    def _format_final_output(self, candidates: list[dict[str, Any]]) -> str:
        """将 submit 后的候选格式化为 main agent 的 memory context。

        格式：每个条目以 [R{n}] 开头，使用 detail 字段的完整内容。
        按 source 分组展示。
        """
        if not candidates:
            return "(No relevant memories found)"

        # 按 source 分组但保持整体的相关度顺序
        parts: list[str] = []

        # 收集各类型
        fs_items = [c for c in candidates if c["source"] == "fs"]
        vec_items = [c for c in candidates if c["source"] == "vec"]
        graph_items = [c for c in candidates if c["source"] in ("graph", "graph_edge")]

        if fs_items:
            lines = []
            for c in fs_items:
                lines.append(f"  {c['id']}: {c['detail']}")
            parts.append("## File System\n" + "\n\n".join(lines))

        if vec_items:
            lines = []
            for c in vec_items:
                lines.append(f"  {c['id']}: {c['detail']}")
            parts.append("## Facts & Preferences\n" + "\n".join(lines))

        if graph_items:
            lines = []
            for c in graph_items:
                lines.append(f"  {c['id']}: {c['detail']}")
            parts.append("## Entity Relations\n" + "\n".join(lines))

        return "\n\n".join(parts)

# ---------------------------------------------------------------------------
# 公开常量别名（agent loop 模式）
# ---------------------------------------------------------------------------

RETRIEVE_T2_AGENT_SYSTEM_PROMPT = AGENT_LOOP_SYSTEM_PROMPT
RETRIEVE_T2_AGENT_USER_TEMPLATE = AGENT_LOOP_USER_TEMPLATE
RETRIEVE_T2_AGENT_TOOLS = AGENT_LOOP_TOOLS


__all__ = [
    "RetrieveT2AgentLoopTask",
    "RETRIEVE_T2_AGENT_TOOLS",
    "RETRIEVE_T2_AGENT_SYSTEM_PROMPT",
    "RETRIEVE_T2_AGENT_USER_TEMPLATE",
    # 兼容老命名
    "AGENT_LOOP_TOOLS",
    "AGENT_LOOP_SYSTEM_PROMPT",
    "AGENT_LOOP_USER_TEMPLATE",
    "RetrieveMode",
]
