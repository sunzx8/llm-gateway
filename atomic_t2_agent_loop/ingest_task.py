"""Ingest T2 Agent Loop Task.

Budget-constrained agent loop 摄入：LLM 自主决定每一轮做什么（读/写/完成），
读写工具集统一暴露，最多 ``max_turns`` 轮（默认 5）。

设计来源：``dev-0421-t3`` 分支的 ``IngestT3Task._execute_agent_loop``。
迁移时移除了 fast/slow 两轮模式（``IngestMode.FAST/SLOW``）以及对应的
round1/round2 prompt 和 ``_execute_two_round`` 路径，仅保留 agent loop。

接口对齐 :class:`BaseContextTask`：``run()`` 不返回值，结果写入
``self.result_extra``，由 ``execute()`` 包装并合并到 ``self._stats.to_dict()``。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime
from typing import Any

from context_task.base_context_task import BaseContextTask
from storage.file_system_store import FileSystemStore
from storage.stores_base import GraphStoreBase, VectorStoreBase
from utils.memory_llm_interface import LLMInterface

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------

# Read-only tools
_READ_ONLY_TOOLS: list[dict] = [
    {"type": "function", "function": {
        "name": "fs_grep",
        "description": "Regex/substring search returning matching lines with context. Use for dedup checks and finding existing content.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Regex or substring pattern"},
            "paths": {
                "oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}],
                "description": "File or directory path(s) to search in (use '.' for all files)",
            },
        }, "required": ["pattern", "paths"]}}},
    {"type": "function", "function": {
        "name": "fs_bm25_search",
        "description": "BM25 full-text search across all memory files. Returns top-k matching files with scores and snippets. Better than fs_grep for natural language queries.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search query (keywords or natural language)"},
            "top_k": {"type": "integer", "description": "Max results to return (default: 5)"},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fs_read_file",
        "description": "Read the full content of a specific file. Use when you know the exact path.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative file path"},
        }, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "fs_read_lines",
        "description": "Read a specific line range from a file (1-indexed, inclusive).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative file path"},
            "start": {"type": "integer", "description": "Start line number (1-indexed)"},
            "end": {"type": "integer", "description": "End line number (inclusive)"},
        }, "required": ["path", "start"]}}},
    {"type": "function", "function": {
        "name": "fs_tree",
        "description": "Show directory tree structure.",
        "parameters": {"type": "object", "properties": {
            "max_depth": {"type": "integer", "description": "Max depth (default: 3)"},
        }}}},
    {"type": "function", "function": {
        "name": "vec_search",
        "description": "Semantic search in a specific vector collection.",
        "parameters": {"type": "object", "properties": {
            "collection": {"type": "string", "description": "Collection name"},
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "description": "Number of results (default: 5)"},
        }, "required": ["collection", "query"]}}},
    {"type": "function", "function": {
        "name": "vec_search_all",
        "description": "Semantic search across ALL vector collections.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "description": "Number of results (default: 10)"},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "graph_search_nodes",
        "description": "Search graph nodes by keyword.",
        "parameters": {"type": "object", "properties": {
            "keyword": {"type": "string", "description": "Fuzzy search in node_id, label, properties"},
        }, "required": ["keyword"]}}},
    {"type": "function", "function": {
        "name": "fs_execute_bash",
        "description": (
            "Execute shell command in memory store root. "
            "Allowed: cat/ls/head/tail/wc/sort/uniq/tree/find/grep/awk/sed/cut/tr/diff/jq/"
            "mkdir/touch/mv/cp/rm/chmod/tee/echo/printf/python3/date, "
            "git add/commit/rm/mv/log/show/diff/status. "
            "Forbidden: `..` escape, git checkout/branch/merge/rebase/push/pull/fetch/remote/clone."
        ),
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default: 15)"},
        }, "required": ["command"]}}},
]


# Write tools
_WRITE_TOOLS: list[dict] = [
    {"type": "function", "function": {
        "name": "fs_write",
        "description": "Create or overwrite a file with full content. Auto-creates parent dirs.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative file path"},
            "content": {"type": "string", "description": "Full file content"},
        }, "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "fs_append",
        "description": "Append content to a file (creates if not exists).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative file path"},
            "content": {"type": "string", "description": "Content to append (one or more lines)"},
        }, "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "fs_update_line",
        "description": "Replace a specific line in a file (1-indexed). Use for UPDATE operations.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative file path"},
            "line_number": {"type": "integer", "description": "Line number to replace (1-indexed)"},
            "new_content": {"type": "string", "description": "New content for that line"},
        }, "required": ["path", "line_number", "new_content"]}}},
    {"type": "function", "function": {
        "name": "fs_update_meta",
        "description": "Update a file's first-line JSON metadata (e.g. description).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative file path"},
            "meta_json": {"type": "string", "description": "JSON string for metadata, e.g. {\"description\":\"...\"}"},
        }, "required": ["path", "meta_json"]}}},
    {"type": "function", "function": {
        "name": "vec_add",
        "description": "Add entries to a vector collection (auto-embed + index). Prefer first-person phrasing for text.",
        "parameters": {"type": "object", "properties": {
            "collection": {"type": "string", "description": "Collection name (e.g. facts_alex, preferences_alex)"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Atomic, independently understandable statement"},
                        "metadata": {"type": "object", "description": "Optional metadata (e.g. type, subject, occurred_at, topic, etc.)"},
                    },
                    "required": ["text"],
                },
                "description": "List of entries. Each has text (required) and optional metadata dict.",
            },
        }, "required": ["collection", "items"]}}},
    {"type": "function", "function": {
        "name": "graph_add_node",
        "description": "Add or update a graph node.",
        "parameters": {"type": "object", "properties": {
            "node_id": {"type": "string", "description": "Node ID (lowercase slug, e.g. italian_food)"},
            "label": {"type": "string", "description": "Node type: Person/Topic/Activity/Genre/Preference"},
            "properties": {"type": "object", "description": "Node properties (include occurred_at if applicable)"},
        }, "required": ["node_id"]}}},
    {"type": "function", "function": {
        "name": "graph_add_edge",
        "description": "Add a relation edge between two nodes (auto-creates nodes if missing).",
        "parameters": {"type": "object", "properties": {
            "source": {"type": "string", "description": "Source node ID"},
            "target": {"type": "string", "description": "Target node ID"},
            "relation": {"type": "string", "description": "Relation type (e.g. prefers, avoids, tried, enjoys)"},
            "properties": {"type": "object", "description": "Edge properties (include occurred_at if applicable)"},
        }, "required": ["source", "target", "relation"]}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "Signal completion. Provide a brief summary of what was extracted and stored.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "description": "Brief summary of extractions performed"},
        }, "required": ["summary"]}}},
]


# Public: agent-loop 完整工具集（read + write + finish）
INGEST_T2_AGENT_TOOLS: list[dict] = [*_READ_ONLY_TOOLS, *_WRITE_TOOLS]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

INGEST_T2_AGENT_SYSTEM_PROMPT = """\
You are a Memory Extraction Agent. Given a conversation, identify what is worth \
remembering and persist it into the memory system.

## Methodology

### Step 1: Identify What Matters
Read the conversation and extract ALL facts worth remembering:
- Concrete preferences, opinions, experiences, life events
- Relationships between people, places, activities
- Attitude changes ("used to X, now Y because Z") — capture the FULL chain with REASONS
- Negative experiences (dislikes, quits, aversions — equally important)
- Specific details: names, dates, places, companions, reasons

**CRITICAL — Reasons are first-class facts.** When the user explains WHY they did \
something, changed their mind, or quit an activity — the reason itself must be stored \
as a dedicated vec entry AND in the FS file. "I quit X" without "because Y" is incomplete.

**CRITICAL — Extract EVERY specific fact, even minor ones.** If the user mentions \
a book title, a friend's name, a place they visited, a class they took — extract it. \
Missing a fact causes worse errors downstream than having a redundant entry.

### Assistant-side Memory (IMPORTANT)
The assistant messages are also part of the conversation. Store assistant content ONLY when it can affect future answers:
- Recommendations, suggestions, activity/book/film/place ideas given to the user
- Explanations or reasons the assistant gave that the user may later refer to
- Options the assistant proposed and the user's reaction/acceptance/rejection

Store these as assistant-sourced memories, e.g.:
- "The assistant suggested a soundscape recording activity as a creatively fulfilling weekend idea."
- "The assistant recommended a film-industry behind-the-scenes book for learning about filmmakers."
Do NOT store generic praise, empathy, filler, or long assistant explanations unless the user reacts to them.

Do NOT store: greetings, filler, chitchat, or information already in memory.

### Step 2: Check Existing Memory
Use read tools (fs_bm25_search, vec_search, fs_grep) to check what already exists:
- **Dedup**: Skip facts already stored
- **Update**: If new info contradicts/extends existing records, APPEND the new stance \
while PRESERVING the old one and its reason. Never overwrite — build a timeline:
  "Used to X because [reason1], then switched to Y because [reason2]"
- **Skip exploration if memory is empty** (first ingest) — go straight to writing

### Step 3: Choose Storage Form Based on Memory Shape
Look at the current memory structure and decide the best fit for each fact:
- **FS files**: Topical markdown files for structured, human-readable knowledge \
(group related facts under the right file; create new files for new topics)
- **Vec entries**: Atomic, independently retrievable statements \
(one fact per entry; include metadata: subject, occurred_at, type, source_role). \
Use first-person for user facts; use "The assistant suggested/recommended..." for assistant-sourced facts.
- **Graph nodes+edges**: Entity relationships and stance tracking \
(person→topic edges with relation types: prefers, avoids, tried, etc.)

Write to ALL THREE backends for important facts (redundancy improves retrieval). \
For minor facts, FS + Vec is sufficient.

## Vec Entry Quality (CRITICAL for downstream retrieval)

Each vec entry must be **specific and self-contained**. Bad vs Good examples:
- BAD: "User likes food" → GOOD: "I love authentic Sichuan hotpot, especially the mala flavor"
- BAD: "User had a bad experience" → GOOD: "I dropped out of a salsa class because close physical contact with strangers caused me anxiety"
- BAD: "User changed hobbies" → GOOD: "I stopped running marathons after a knee injury in 2022 and switched to swimming"

## Rules
1. **File metadata**: First line of every file: `{"description":"..."}`
2. **Temporal prefix**: Each FS line starts with `[YYYY-MM-DD]`, `[YYYY-MM]`, `[YYYY]`, or `[undated]`
3. **Relative time normalization**: The user prompt provides CURRENT DATE. If the conversation contains explicit relative time (e.g. today, yesterday, last week, last month, recently, this weekend), convert it to the best absolute date/month/year using CURRENT DATE. Use `[undated]` only when no time cue exists.
4. **Vec entries must be atomic**: Self-contained, independently understandable
5. **Preserve specifics**: Keep the user's concrete details — names, places, reasons
6. **occurred_at**: Include normalized time info in vec metadata and graph edge properties when available
7. **Language consistency**: Match the conversation's dominant language
8. Do not fabricate; only extract what is explicitly stated. Do NOT invent precise dates when only a month/year is inferable.
9. **Be concise in your text output** — skip explanations, go straight to tool calls
10. Call `finish` with a brief summary when done
"""

INGEST_T2_AGENT_USER_TEMPLATE = """\
## Current Date
{current_date}

Use this ONLY to normalize explicit relative time expressions in the conversation. If no time cue exists, use `[undated]`.

## Memory Index
{index_content}

## Current Memory Structure (pre-loaded — no need to call fs_tree/vec_search_all/graph_search_nodes for structure)

### File System
{fs_tree}

### Vector Collections
{vec_collections}

### Graph Schema
{graph_schema}

---

## Conversation to Process (session: {session_id}):

{conversation}

---

Extract and store all valuable information. You already have the memory structure above — \
go directly to targeted dedup checks (fs_grep/vec_search) if needed, then write. Call `finish` when done.
"""


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class IngestT2AgentLoopTask(BaseContextTask):
    """Atomic T2 多后端 ingest 任务（agent loop 模式）。"""

    task_name = "ingest_t2_agent_loop"

    def __init__(
        self,
        llm: LLMInterface,
        fs_store: FileSystemStore,
        vec_store: VectorStoreBase,
        graph_store: GraphStoreBase,
        *,
        max_turns: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm)
        self.fs = fs_store
        self.vec = vec_store
        self.graph = graph_store
        self.max_turns = max_turns
        self.extra_kwargs = kwargs
        # 业务结果（agent loop 摘要 / 工具轨迹 / 计数），由 run() 写入
        self.result_extra: dict[str, Any] = {}
        # 跨 session 去重哈希（同一 task 实例多次调用时使用）
        self._seen_hashes: set[str] = set()
        # 跨 session 累积的 finish summary
        self._session_extractions: list[str] = []

    # ------------------------------------------------------------------
    # 主 run（BaseContextTask 钩子）
    # ------------------------------------------------------------------

    async def run(
        self,
        user_id: str = "default_user",
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """从 messages 抽取信息，按 agent loop 写入记忆库。"""
        messages = messages or []
        if not messages:
            logger.warning(
                "[%s] run: messages 为空，跳过 (user_id=%s, session_id=%s)",
                self.task_name, user_id, session_id,
            )
            return

        # 时间锚：优先 kwargs[session_time]，否则系统时间
        session_time_str = kwargs.get("session_time") or datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S, %a"
        )
        # current_date 用于 prompt 注入（仅 YYYY-MM-DD）
        try:
            current_date = session_time_str.split()[0]
        except Exception:
            current_date = datetime.now().date().isoformat()

        # 拼对话文本
        message_offset: int = kwargs.get("message_offset", 1)
        conversation = self._format_conversation(messages, start_index=message_offset)

        # 去重（基于本次对话内容哈希）
        content_hash = hashlib.md5(conversation.encode()).hexdigest()
        if content_hash in self._seen_hashes:
            logger.info("[%s] skipping duplicate session content", self.task_name)
            self.result_extra["ingest_finish_summary"] = "skipping duplicate session content"
            self.result_extra["ingest_finish_reason"] = "duplicate"
            return
        self._seen_hashes.add(content_hash)

        # 构建 user prompt — 预注入三后端结构快照
        index_content = self._read_index_md()
        fs_tree = self.fs.tree(max_depth=3)

        vec_stats = self.vec.get_stats() if self.vec else {}
        vec_collections_map = (
            vec_stats.get("collections", {}) if isinstance(vec_stats, dict) else {}
        )
        vec_collections = "\n".join(
            f"  - {name}: {size} entries"
            for name, size in sorted(vec_collections_map.items())
        ) if vec_collections_map else "(no collections yet)"

        graph_stats = self.graph.get_stats() if self.graph else {}
        graph_labels = (
            graph_stats.get("node_labels", {}) if isinstance(graph_stats, dict) else {}
        )
        graph_rels = (
            graph_stats.get("relation_types", {}) if isinstance(graph_stats, dict) else {}
        )
        graph_schema = (
            f"Nodes: {', '.join(f'{k}:{v}' for k, v in sorted(graph_labels.items())) or '(none)'}\n"
            f"Relations: {', '.join(f'{k}:{v}' for k, v in sorted(graph_rels.items())) or '(none)'}"
        )

        user_prompt = INGEST_T2_AGENT_USER_TEMPLATE.format(
            current_date=current_date,
            index_content=index_content,
            fs_tree=fs_tree,
            vec_collections=vec_collections,
            graph_schema=graph_schema,
            session_id=session_id,
            conversation=conversation,
        )

        system_prompt = INGEST_T2_AGENT_SYSTEM_PROMPT
        tools = INGEST_T2_AGENT_TOOLS
        max_turns = self.max_turns

        # ---- agent loop ----
        messages_history: list[dict[str, Any]] = [
            {"role": "user", "content": user_prompt},
        ]
        agent_loop_trace: list[dict[str, Any]] = []
        finish_summary = ""
        steps_used = 0

        # 业务统计
        stats: dict[str, Any] = {
            "facts_added": 0,
            "facts_updated": 0,
            "vec_added": 0,
            "vec_updated": 0,
            "graph_ops": 0,
        }

        for turn in range(max_turns):
            steps_used = turn + 1

            # 注入轮次 budget 提示（不改 system prompt，保持 prefix caching）
            if turn > 0:
                remaining = max_turns - turn
                budget_hint = (
                    f"[Turn {turn + 1}/{max_turns}, {remaining} remaining] "
                    f"{'You MUST call `finish` now.' if remaining <= 1 else 'Continue or call `finish` when done.'}"
                )
                messages_history.append({"role": "user", "content": budget_hint})

            response = await self.llm_generate_with_stat(
                system=system_prompt,
                messages=messages_history,
                tools=tools,
                label=f"ingest_agent_turn{turn + 1}",
            )

            step_trace: dict[str, Any] = {
                "step": turn + 1,
                "model_content": (response.content or "")[:500],
                "tool_calls_count": len(response.tool_calls or []),
                "tool_calls": [],
            }

            if not response.tool_calls:
                # 纯文本回复 — 视为完成
                finish_summary = response.content or ""
                agent_loop_trace.append(step_trace)
                break

            # 处理 tool calls
            messages_history.append(response.to_message())
            should_break = False

            for tc in response.tool_calls:
                if tc.name == "finish":
                    finish_summary = (tc.arguments or {}).get("summary", "")
                    step_trace["tool_calls"].append({
                        "tool": "finish",
                        "arguments": tc.arguments,
                        "result": finish_summary[:2000],
                    })
                    messages_history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "(task finished)",
                    })
                    should_break = True
                    continue

                # 执行工具（统一 dispatcher）— 走 self.tool_with_stat 自动统计
                try:
                    result_str = await self.tool_with_stat(
                        tc.name,
                        self._execute_agent_tool,
                        tc.name, tc.arguments or {}, stats,
                        arguments_summary={"tool": tc.name, **{
                            k: str(v)[:80] for k, v in (tc.arguments or {}).items()
                        }},
                    )
                except Exception as e:
                    logger.exception("Agent tool %s failed: %s", tc.name, e)
                    result_str = f"ERROR: {e}"

                messages_history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result_str),
                })
                step_trace["tool_calls"].append({
                    "tool": tc.name,
                    "arguments": tc.arguments,
                    "result": str(result_str)[:2000],
                })

            agent_loop_trace.append(step_trace)
            if should_break:
                break
        else:
            logger.warning(
                "[%s] agent loop reached max_turns=%d without finish",
                self.task_name, max_turns,
            )

        # Git commit（best-effort）
        try:
            self.fs.execute_bash(
                command=f'git add -A && git commit -m "ingest_t2_agent: {finish_summary[:50]}" --allow-empty',
                timeout=10,
                allow_write=True,
            )
        except Exception as e:
            logger.warning("Git commit failed: %s", e)

        if finish_summary:
            self._session_extractions.append(f"[{session_id}] {finish_summary[:200]}")

        # 写入 result_extra（dispatcher / 调用方读取）
        self.result_extra["ingest_finish_summary"] = finish_summary
        self.result_extra["ingest_finish_reason"] = (
            "finish" if finish_summary and steps_used <= max_turns else "max_turns"
        )
        self.result_extra["ingest_turns_used"] = steps_used
        self.result_extra["ingest_max_turns"] = max_turns
        self.result_extra["facts_added"] = stats["facts_added"]
        self.result_extra["facts_updated"] = stats["facts_updated"]
        self.result_extra["vec_added"] = stats["vec_added"]
        self.result_extra["vec_updated"] = stats["vec_updated"]
        self.result_extra["graph_ops"] = stats["graph_ops"]
        self.result_extra["new_items"] = (
            stats["facts_added"] + stats["vec_added"] + stats["graph_ops"]
        )
        self.result_extra["agent_loop_trace_summary"] = [
            {
                "step": st["step"],
                "tool_calls_count": st["tool_calls_count"],
                "tools": [tc["tool"] for tc in st["tool_calls"]],
            }
            for st in agent_loop_trace
        ]

        logger.info(
            "[%s] ingest done (turns=%d, summary=%.200s, fs_add=%d, vec_add=%d, graph_ops=%d)",
            self.task_name, steps_used, finish_summary,
            stats["facts_added"], stats["vec_added"], stats["graph_ops"],
        )

    # ------------------------------------------------------------------
    # Tool dispatcher（read + write 合并）
    # ------------------------------------------------------------------

    async def _execute_agent_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        stats: dict[str, Any],
    ) -> str:
        """统一 dispatcher — read-only 与 write 工具全部在此。"""
        try:
            # ---- Read-only ----
            if tool_name == "fs_grep":
                pattern = args.get("pattern", "")
                if not pattern:
                    return "ERROR: fs_grep requires 'pattern'"
                results = self.fs.grep(
                    pattern=pattern,
                    paths=args.get("paths", "."),
                    context_lines=2,
                    max_matches=20,
                    case_insensitive=True,
                )
                return str(results)[:3000]

            if tool_name == "fs_bm25_search":
                query = args.get("query", "")
                if not query:
                    return "ERROR: fs_bm25_search requires 'query'"
                results = self.fs.search_bm25(query, top_k=int(args.get("top_k", 5)))
                return str(results)[:3000]

            if tool_name == "fs_read_file":
                path = args.get("path", "")
                if not path:
                    return "ERROR: fs_read_file requires 'path'"
                return self.fs.read_file(path)[:3000]

            if tool_name == "fs_read_lines":
                path = args.get("path", "")
                if not path:
                    return "ERROR: fs_read_lines requires 'path'"
                return self.fs.read_lines(
                    rel_path=path,
                    start=int(args.get("start", 1)),
                    end=int(args["end"]) if args.get("end") is not None else None,
                )

            if tool_name == "fs_tree":
                return self.fs.tree(int(args.get("max_depth", 3)))

            if tool_name == "vec_search":
                results = await self.vec.search(
                    collection=args["collection"],
                    query=args["query"],
                    top_k=int(args.get("top_k", 5)),
                )
                return str(results)[:3000]

            if tool_name == "vec_search_all":
                results = await self.vec.search_all(
                    query=args["query"],
                    top_k=int(args.get("top_k", 10)),
                )
                return str(results)[:3000]

            if tool_name == "graph_search_nodes":
                results = self.graph.search_nodes(keyword=args.get("keyword", ""))
                return str(results)[:3000]

            if tool_name == "fs_execute_bash":
                command = args.get("command", "")
                if not command:
                    return "ERROR: fs_execute_bash requires 'command'"
                return self.fs.execute_bash(
                    command=command,
                    timeout=int(args.get("timeout", 15)),
                    allow_write=True,
                )

            # ---- Write ----
            if tool_name == "fs_write":
                path = args.get("path", "")
                content = args.get("content", "")
                if not path:
                    return "ERROR: fs_write requires 'path'"
                result = self.fs.write_file(path, content)
                stats["facts_added"] = stats.get("facts_added", 0) + 1
                return result

            if tool_name == "fs_append":
                path = args.get("path", "")
                content = args.get("content", "")
                if not path:
                    return "ERROR: fs_append requires 'path'"
                result = self.fs.append_file(path, content)
                stats["facts_added"] = stats.get("facts_added", 0) + 1
                return result

            if tool_name == "fs_update_line":
                path = args.get("path", "")
                line_number = int(args.get("line_number", 0))
                new_content = args.get("new_content", "")
                if not path or line_number <= 0:
                    return "ERROR: fs_update_line requires valid 'path' and 'line_number'"
                # FileSystemStore 没有 update_line API；用 read+replace+write 实现
                old_content = self.fs.read_file(path)
                if old_content.startswith("ERROR"):
                    return old_content
                lines = old_content.split("\n")
                if line_number > len(lines):
                    return f"ERROR: line {line_number} out of range (file has {len(lines)} lines)"
                lines[line_number - 1] = new_content
                result = self.fs.write_file(path, "\n".join(lines))
                stats["facts_updated"] = stats.get("facts_updated", 0) + 1
                return result

            if tool_name == "fs_update_meta":
                path = args.get("path", "")
                meta_json = args.get("meta_json", "")
                if not path or not meta_json:
                    return "ERROR: fs_update_meta requires 'path' and 'meta_json'"
                # 校验 JSON
                try:
                    json.loads(meta_json)
                except (json.JSONDecodeError, TypeError) as e:
                    return f"ERROR: invalid meta_json: {e}"
                old_content = self.fs.read_file(path)
                if old_content.startswith("ERROR"):
                    # 创建新文件，第一行就是 meta
                    return self.fs.write_file(path, meta_json + "\n")
                lines = old_content.split("\n")
                # 替换或添加首行 meta
                if lines and lines[0].lstrip().startswith("{"):
                    lines[0] = meta_json
                else:
                    lines.insert(0, meta_json)
                result = self.fs.write_file(path, "\n".join(lines))
                stats["facts_updated"] = stats.get("facts_updated", 0) + 1
                return result

            if tool_name == "vec_add":
                collection = args.get("collection", "")
                items = args.get("items", [])
                if not collection or not items:
                    return "ERROR: vec_add requires 'collection' and 'items'"
                # 自动 create_collection
                try:
                    self.vec.create_collection(collection)
                except Exception:
                    pass
                added = 0
                for item in items:
                    text = item.get("text", "") if isinstance(item, dict) else ""
                    if not text:
                        continue
                    metadata = item.get("metadata") if isinstance(item, dict) else None
                    try:
                        await self.vec.add(
                            collection=collection,
                            text=text,
                            metadata=metadata if isinstance(metadata, dict) else None,
                        )
                        added += 1
                    except Exception as e:
                        logger.warning("vec.add failed: %s", e)
                stats["vec_added"] = stats.get("vec_added", 0) + added
                return f"added {added}/{len(items)} entries to '{collection}'"

            if tool_name == "graph_add_node":
                node_id = args.get("node_id", "")
                if not node_id:
                    return "ERROR: graph_add_node requires 'node_id'"
                properties = args.get("properties")
                if isinstance(properties, str):
                    try:
                        properties = json.loads(properties)
                    except (json.JSONDecodeError, TypeError):
                        properties = {"note": properties}
                result = self.graph.add_node(
                    node_id=node_id,
                    label=args.get("label", ""),
                    properties=properties if isinstance(properties, dict) else None,
                )
                stats["graph_ops"] = stats.get("graph_ops", 0) + 1
                # 计算 node embedding（best-effort）
                embedder = getattr(self.graph, "embedder", None)
                if embedder is not None:
                    embed_text = self._build_node_embed_text(
                        node_id, args.get("label", ""),
                        properties if isinstance(properties, dict) else None,
                    )
                    try:
                        emb = await embedder.embed_single(embed_text)
                        self.graph.set_node_embedding(node_id, emb)
                    except Exception:
                        pass
                return result

            if tool_name == "graph_add_edge":
                source = args.get("source", "")
                target = args.get("target", "")
                relation = args.get("relation", "")
                if not source or not target or not relation:
                    return "ERROR: graph_add_edge requires 'source', 'target', 'relation'"
                properties = args.get("properties")
                if isinstance(properties, str):
                    try:
                        properties = json.loads(properties)
                    except (json.JSONDecodeError, TypeError):
                        properties = {"note": properties}
                result = self.graph.add_edge(
                    source=source,
                    target=target,
                    relation=relation,
                    properties=properties if isinstance(properties, dict) else None,
                )
                stats["graph_ops"] = stats.get("graph_ops", 0) + 1
                # 计算 edge embedding（best-effort）
                embedder = getattr(self.graph, "embedder", None)
                if embedder is not None:
                    edge_text = self._build_edge_embed_text(source, relation, target)
                    edge_id = self._extract_edge_id_from_result(result)
                    if edge_id:
                        try:
                            emb = await embedder.embed_single(edge_text)
                            self.graph.set_edge_embedding(edge_id, emb)
                        except Exception:
                            pass
                return result

            return f"Unknown tool: {tool_name}"

        except Exception as e:
            logger.exception("Tool %s failed: %s", tool_name, e)
            return f"ERROR: {e}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_index_md(self) -> str:
        """读取 .meta/index.md 作为提示词级目录。"""
        try:
            content = self.fs.read_file(".meta/index.md")
            if isinstance(content, str) and content.startswith("ERROR"):
                return "(empty memory — no index available)"
            return content or "(empty memory — no index available)"
        except Exception:
            return "(empty memory — no index available)"

    @staticmethod
    def _build_node_embed_text(
        node_id: str,
        label: str,
        properties: dict | None,
    ) -> str:
        """构建节点的文本表示，用于计算嵌入向量。"""
        parts = []
        readable_id = node_id.replace("_", " ").replace("-", " ")
        parts.append(readable_id)
        if label:
            parts.append(f"({label})")
        if properties:
            for key, value in properties.items():
                if value and key not in ("created_at", "updated_at", "timestamp"):
                    parts.append(f"{key}: {value}")
        return " ".join(parts)

    @staticmethod
    def _build_edge_embed_text(source: str, relation: str, target: str) -> str:
        """构建边的文本表示，用于计算嵌入向量。"""
        src = source.replace("_", " ").replace("-", " ")
        rel = relation.replace("_", " ").replace("-", " ")
        tgt = target.replace("_", " ").replace("-", " ")
        return f"{src} {rel} {tgt}"

    @staticmethod
    def _extract_edge_id_from_result(result_msg: str) -> str | None:
        """从 add_edge 返回消息中提取 edge_id（用于绑定 embedding）。"""
        m = re.search(r"\(id=(edge_\d+)\)", result_msg or "")
        return m.group(1) if m else None

    @staticmethod
    def _format_conversation(
        messages: list[dict[str, Any]],
        start_index: int = 1,
    ) -> str:
        """格式化对话为可读文本。"""
        lines: list[str] = []
        for i, m in enumerate(messages, start=start_index):
            role = m.get("role", "?")
            if role not in ("user", "assistant"):
                continue
            content = m.get("content", "")
            if isinstance(content, list):
                # OpenAI 多模态格式
                text_parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                content = "\n".join(text_parts)
            lines.append(f"[{i}] {role}: {content}")
        return "\n".join(lines)


__all__ = [
    "IngestT2AgentLoopTask",
    "INGEST_T2_AGENT_TOOLS",
    "INGEST_T2_AGENT_SYSTEM_PROMPT",
    "INGEST_T2_AGENT_USER_TEMPLATE",
]
