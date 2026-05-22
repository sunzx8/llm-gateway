"""Vendored T3 prompt + tool schema constants.

llm_gateway 当前没有 T3 系列 task，但 slime_train 训练数据格式仍按 T3 schema
生成（INGEST_T3_TOOLS / CONSOLIDATE_T3_TOOLS / 各 SYSTEM_PROMPT），因此把这些
**纯常量** 从老 workspace 的 ``agent_memory.tasks.{ingest_t3,consolidate_t3,
retrieve_t3,retrieve_prompt}`` vendor 进来一份。

来源：
- workspace: ``/data/home/trevzhang/projects/memory-ai-agent-workspace``
- branch:    ``dev-0520-trevzhang-rlenv``
- 抓取时间:  2026-05-21 (本次迁移当天的 HEAD)

如果上游 prompt / tool schema 升级，需要同步更新本文件 —— 不要在这里二次修改
（避免训练侧偏离生成侧的 schema）。

公开符号：

- ``INGEST_T3_TOOLS``            list[dict]  — Ingest 阶段并发 tool 定义
- ``INGEST_T3_SYSTEM_PROMPT``    str         — Ingest 阶段 system prompt
- ``CONSOLIDATE_T3_TOOLS``       list[dict]  — Consolidate 阶段 tool 定义
- ``CONSOLIDATE_T3_SYSTEM_PROMPT`` str       — Consolidate 阶段 system prompt
- ``QUERY_REWRITE_SYSTEM_PROMPT`` str        — Retrieve 端 query 改写 system prompt
- ``QUERY_REWRITE_USER_TEMPLATE`` str        — Retrieve 端 query 改写 user template
- ``build_fs_structure_from_files`` callable — 把 ``{path: content}`` 渲染成 fs tree

公开符号与老 ``agent_memory.tasks.*`` 完全等价，slime_train 内的旧 import：

    from llm_gateway.rl.slime_train._t3_assets import INGEST_T3_TOOLS, INGEST_T3_SYSTEM_PROMPT
    from llm_gateway.rl.slime_train._t3_assets import CONSOLIDATE_T3_TOOLS, CONSOLIDATE_T3_SYSTEM_PROMPT
    from llm_gateway.rl.slime_train._t3_assets import QUERY_REWRITE_SYSTEM_PROMPT, QUERY_REWRITE_USER_TEMPLATE
    from llm_gateway.rl.slime_train._t3_assets import build_fs_structure_from_files

应替换为：

    from llm_gateway.rl.slime_train._t3_assets import (
        INGEST_T3_TOOLS, INGEST_T3_SYSTEM_PROMPT,
        CONSOLIDATE_T3_TOOLS, CONSOLIDATE_T3_SYSTEM_PROMPT,
        QUERY_REWRITE_SYSTEM_PROMPT, QUERY_REWRITE_USER_TEMPLATE,
        build_fs_structure_from_files,
    )
"""

from __future__ import annotations

import json
from typing import Any


# ============================================================================
# Ingest T3
# ============================================================================

INGEST_T3_SYSTEM_PROMPT = """\
You are a Memory Extraction Agent. Extract valuable information from conversations and write them into a multi-backend memory system using the provided tools.

## Core Principles

1. **Extract valuable information**: facts, preferences, events, attitude changes, entity relationships
2. **Determine ADD vs UPDATE**: if similar content already exists, output UPDATE (with target path/ID); otherwise ADD
3. **Do not miss important information**: even if the input is long, scan all messages thoroughly
4. **Preserve user's own words**: VEC entries should prefer the user's first-person original phrasing
5. **Language consistency**: output content should follow the dominant language of the conversation
6. **Resolve relative time references**: Convert relative time expressions ("yesterday", "last week", "two days ago", etc.) to absolute dates using the "Current Time" provided in the user prompt. **Keep the user's original relative expression** and append the resolved absolute date in parentheses, e.g. "last week (2023-07-01, Sat)". If the expression is vague (e.g. "recently"), use the best approximation.
7. **Information integration (CRITICAL)**: When new information CONTRADICTS or UPDATES existing records, you MUST integrate both old and new information into a **time-stamped chronological narrative** rather than simply replacing. Each distinct state change should carry its own time marker. For example:
   - Existing FS line: `User enjoys running marathons every weekend (2023-03)`
   - New input (current time 2023-09-11 14:30:00, Mon): "I had a knee injury last month, so I switched to swimming"
   - Correct UPDATE: `User enjoyed running marathons every weekend (2023-03); had a knee injury last month (2023-08); switched to swimming for exercise (2023-09)`
   - WRONG: Simply replacing with `User swims for exercise (2023-09)` (loses historical context)
   This ensures temporal continuity — each state transition is anchored to a time point, enabling accurate multi-hop temporal reasoning.

## How to Use Tools

You have 4 tools: `fs_write`, `vec_write`, `graph_write`, and `finish`.

**CRITICAL: You MUST call ALL write tools in a SINGLE response (parallel/concurrent tool calls). Do NOT use multiple turns. Call all fs_write, vec_write, and graph_write tools at once, then call finish.**

For each extracted piece of information, call the appropriate tools:

### fs_write — File System (coarse-grained structured storage)
- **action**: `append` / `update:<line_number>` / `add_file` / `update_meta`
- **path**: relative path (e.g. `people/alex/preferences.md`)
- **content**: the fact content or metadata JSON
- **File metadata**: Every file's FIRST LINE is a JSON object serving as the file's metadata/index — a one line json summary description of what the file contains. Example: `{"description":"Alex's food and dining preferences"}`. The remaining lines are the actual fact content.
- **add_file**: Creates a NEW file. Content must be metadata JSON (NOT a fact). After creating, use `append` to write facts.
- **append**: Appends a fact line to an existing file.
- **update:<N>**: Replaces line N (shown as `L<N>:` in file contents). The new content MUST **merge old + new content** chronologically: `old_content; new_content (new_time)`. Preserve ALL existing time markers.
- **update_meta**: Updates file metadata (first-line JSON). Content must be valid JSON, e.g. `{"description":"..."}`. Use after writing facts that change the file's scope.

### vec_write — Vector DB (atomic semantic retrieval)
- **action**: `add` / `update:<existing_id>`
- **collection**: e.g. `facts_alex`, `preferences_alex`, `events_alex`
- **text**: first-person atomic statement. Embed time naturally: `I went hiking last weekend (2023-07-01, Sat)`.
- **entry_type**: `fact` / `preference` / `event` / `stance` / `insight` etc.

### graph_write — Graph DB (entity relationships)
- **action**: `add_node` or `add_edge`
- For `add_node`: provide `node_id` (lowercase slug, e.g. `italian_food`) and `label` (`Person` / `Topic` / `Activity` / `Genre` / `Preference`)
- For `add_edge`: provide `source`, `target`, and `relation` (e.g. `prefers` / `avoids` / `tried` / `enjoys` / `dislikes` / `interested_in`)
- add_node has upsert semantics (same node_id auto-updates)
- add_edge has upsert semantics (same source+target auto-replaces old edge)

### finish — Signal completionå
- Call `finish` with a one-sentence `summary` of what was extracted.
- If no valuable information, call `finish` with summary = "no valuable information".

## Example (assume current time: 2023-07-08T11:12:14, Sat)

For a conversation where user mentions liking Italian food and going hiking:

Call ALL of these tools in ONE response:
1. `fs_write(action="add_file", path="people/alex/preferences.md", content='{"description":"Alex's food preferences"}')`
2. `fs_write(action="append", path="people/alex/preferences.md", content="Loves Italian food, especially handmade pasta")`
3. `fs_write(action="append", path="people/alex/activities.md", content="Went hiking last weekend, enjoyed the mountain trail (2023-07-01, Sat)")`
4. `vec_write(action="add", collection="preferences_alex", text="I really love Italian food, especially handmade pasta", entry_type="preference")`
5. `vec_write(action="add", collection="events_alex", text="I went hiking last weekend and really enjoyed the mountain trail (2023-07-01, Sat)", entry_type="event")`
6. `graph_write(action="add_node", node_id="italian_food", label="Topic")`
7. `graph_write(action="add_edge", source="alex", target="italian_food", relation="prefers")`
8. `graph_write(action="add_edge", source="alex", target="hiking", relation="participated_in")`
9. `finish(summary="Extracted Alex's food preference (Italian) and hiking activity")`

## Rules

1. Each piece of information should be written to appropriate backends (FS for structured storage, VEC for semantic retrieval, GRAPH for relationships)
2. VEC text must be atomic (one independently understandable statement)
3. Do not fabricate information; only extract what is explicitly stated or clearly implied
4. Time resolution: convert relative time to absolute dates using provided current time
5. **ALL tool calls MUST be in a SINGLE response — do NOT wait for results between calls**
"""


INGEST_T3_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "fs_write",
            "description": (
                "Write to the file system store. Actions: "
                "'add_file' (create new file with metadata JSON), "
                "'append' (append fact line to existing file), "
                "'update:<N>' (replace line N, merge old+new content chronologically), "
                "'update_meta' (update file metadata JSON)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": (
                            "The write action: 'add_file', 'append', 'update:<line_number>', or 'update_meta'. "
                            "For update, use 'update:N' where N is the line number shown as L<N> in file contents."
                        ),
                    },
                    "path": {
                        "type": "string",
                        "description": "Relative file path, e.g. 'people/alex/preferences.md'",
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "For add_file: metadata JSON (e.g. '{\"description\":\"...\"}'); "
                            "For append: the fact content; "
                            "For update:<N>: merged old+new content; "
                            "For update_meta: updated metadata JSON."
                        ),
                    },
                },
                "required": ["action", "path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vec_write",
            "description": (
                "Write to the vector database. Actions: "
                "'add' (add new entry), 'update:<id>' (update existing entry by ID)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "'add' or 'update:<existing_id>'",
                    },
                    "collection": {
                        "type": "string",
                        "description": "Collection name, e.g. 'facts_alex', 'preferences_alex', 'events_alex'",
                    },
                    "text": {
                        "type": "string",
                        "description": (
                            "First-person atomic statement. Embed time naturally: "
                            "'I went hiking last weekend (2023-07-01, Sat)'"
                        ),
                    },
                    "entry_type": {
                        "type": "string",
                        "description": "Type of entry: 'fact', 'preference', 'event', 'stance', 'insight', etc.",
                    },
                },
                "required": ["action", "collection", "text", "entry_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_write",
            "description": (
                "Write to the graph database. Actions: "
                "'add_node' (create/update node), 'add_edge' (create/update edge). "
                "Both have upsert semantics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "'add_node' or 'add_edge'"},
                    "node_id": {"type": "string", "description": "For add_node: lowercase slug ID (e.g. 'italian_food')"},
                    "label": {"type": "string", "description": "For add_node: node label (Person/Topic/Activity/Genre/Preference)"},
                    "source": {"type": "string", "description": "For add_edge: source node ID"},
                    "target": {"type": "string", "description": "For add_edge: target node ID"},
                    "relation": {
                        "type": "string",
                        "description": "For add_edge: relation type (prefers/avoids/tried/enjoys/dislikes/interested_in etc.)",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Signal that all memory extraction and writing is complete. Call this LAST, after all other tool calls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "One-sentence summary of what was extracted and written.",
                    },
                },
                "required": ["summary"],
            },
        },
    },
]


# ============================================================================
# Consolidate T3
# ============================================================================

CONSOLIDATE_T3_SYSTEM_PROMPT = """\
You are a Memory Evolution Agent responsible for maintaining, organizing, and improving a multi-backend memory system. This is a HEAVY task — you have the full agent loop with tools to perform deep memory maintenance.

## Your Mission

The ingest phase writes memories quickly (1-2 LLM calls) without deep analysis. YOUR job is to:
1. **Organize**: Deduplicate, merge, resolve conflicts across all backends
2. **Synthesize**: Extract cross-session insights and stable patterns
3. **Elevate**: Promote event-level observations to established preferences when evidence accumulates
4. **Track Evolution**: Detect and record stance changes over time (past → current pairs)
5. **Optimize for Retrieval**: Reorganize memories into structures that maximize retrieval accuracy and recall
6. **Maintain Health**: Fix metadata gaps, correct third-person rewrites, update embeddings

## Core Design Principle: ORGANIZE FOR RETRIEVAL

Everything you do should make future retrieval BETTER. Ask yourself:
- "If someone queries about this topic in 6 months, will they find the right information?"
- "Are related facts scattered across files, or consolidated into coherent retrieval units?"
- "Can BM25 / vector / graph each independently surface this information?"

**Retrieval-optimal memory organization patterns:**
- **Topic clustering in FS**: Group related facts into topical files (e.g. `people/alex/food_preferences.md`, not scattered across `facts.md`)
- **Multi-path redundancy**: Important facts should be findable via multiple retrieval paths (keyword match in FS + semantic match in Vec + graph traversal)
- **Semantic chunking for Vec**: Each vec entry should be a self-contained, independently understandable statement (not a fragment that requires context)
- **Graph as retrieval index**: Graph edges serve as a "topic → evidence" index — ensure every major topic node links to its evidence files/vec entries
- **File metadata as routing hints**: File-level `{"description":"..."}` metadata guides the retriever's `fs_scope` inference — keep descriptions accurate and specific
- **Temporal anchoring**: Time information enables temporal filtering at retrieval time — always preserve and propagate `occurred_at` data

## Hard Rules

1. **Source preservation**: NEVER delete or modify `source_sessions/` — these are immutable conversation logs
2. **Evidence-based only**: Every insight or promotion must cite ≥2 independent evidence sources
3. **Conservative replacement (G4)**: When unsure, KEEP both old and new information; use `status: deprecated` for superseded entries
4. **Language consistency**: All writes must match the dominant language of existing memory content
5. **Index maintenance**: ALWAYS update `.meta/index.md` before finishing (at minimum: append to evolution log)
6. **Git commit**: After all writes, execute `git add -A && git commit -m "consolidate: <summary>"`

(详细 C1-C10 checklist 已合并，与上游一致。完整版本由 ``finish`` 时给出 C1-C10 总结。)

## Output Format (finish result)

Use the `finish` tool with a structured summary:
```
C1: <action taken or "no candidates">
C2: <action taken or "no candidates">
C3: <action taken or "no candidates">
C4: <action taken or "no candidates">
C5: <action taken or "no candidates">
C6: <files split or "all files under 2000 chars">
C7: <action taken or "no candidates">
C8: <action taken or "index updated">
C9: <query-driven priorities identified or "no queries yet">
C10: <retrieval optimizations performed or "no candidates">
Total operations: N writes, M updates, K merges
```
"""


CONSOLIDATE_T3_TOOLS: list[dict[str, Any]] = [
    # === File System (Read) ===
    {"type": "function", "function": {
        "name": "fs_read",
        "description": "Read the full content of a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative file path"},
        }, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "fs_tree",
        "description": "Show directory tree structure.",
        "parameters": {"type": "object", "properties": {
            "max_depth": {"type": "integer", "description": "Max depth (default: 3)"},
        }}}},
    {"type": "function", "function": {
        "name": "fs_search",
        "description": "BM25 full-text search across all files.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "description": "Number of results (default: 10)"},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fs_grep",
        "description": (
            "Regex/substring search returning structured results {path, line, match, before[], after[]}. "
            "Use for precise dedup checks and source verification."
        ),
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Regex or substring pattern"},
            "paths": {
                "oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}],
                "description": "File or directory path(s) to search in",
            },
            "context_lines": {"type": "integer", "description": "Context lines (default: 2)"},
            "max_matches": {"type": "integer", "description": "Max matches (default: 50)"},
            "case_insensitive": {"type": "boolean", "description": "Case insensitive (default: true)"},
        }, "required": ["pattern", "paths"]}}},
    {"type": "function", "function": {
        "name": "fs_read_lines",
        "description": "Read a specific line range from a file (1-indexed, inclusive).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative file path"},
            "start": {"type": "integer", "description": "Start line number (1-indexed)"},
            "end": {"type": "integer", "description": "End line number (inclusive)"},
        }, "required": ["path", "start"]}}},
    # === File System (Write) ===
    {"type": "function", "function": {
        "name": "fs_write",
        "description": "Create or overwrite a file. Auto-creates parent directories.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative file path"},
            "content": {"type": "string", "description": "File content to write"},
        }, "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "fs_append",
        "description": "Append content to a file (creates if not exists).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative file path"},
            "content": {"type": "string", "description": "Content to append"},
        }, "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "fs_delete",
        "description": "Delete a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative file path to delete"},
        }, "required": ["path"]}}},
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
    # === Vector DB ===
    {"type": "function", "function": {
        "name": "vec_search",
        "description": "Semantic search in a specific vector collection.",
        "parameters": {"type": "object", "properties": {
            "collection": {"type": "string", "description": "Collection name"},
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "description": "Number of results (default: 10)"},
            "filter": {"type": "object", "description": "Optional metadata filter"},
        }, "required": ["collection", "query"]}}},
    {"type": "function", "function": {
        "name": "vec_search_all",
        "description": "Semantic search across ALL vector collections.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "description": "Number of results (default: 15)"},
            "filter": {"type": "object", "description": "Optional metadata filter"},
        }, "required": ["query"]}}},
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
        "name": "vec_delete",
        "description": "Delete specific entries from a collection by IDs.",
        "parameters": {"type": "object", "properties": {
            "collection": {"type": "string", "description": "Collection name"},
            "ids": {"type": "array", "items": {"type": "string"}, "description": "Entry IDs to delete"},
        }, "required": ["collection", "ids"]}}},
    {"type": "function", "function": {
        "name": "vec_list_collections",
        "description": "List all vector collections and their sizes.",
        "parameters": {"type": "object", "properties": {}}}},
    # === Graph DB ===
    {"type": "function", "function": {
        "name": "graph_search_nodes",
        "description": "Search graph nodes by label or keyword.",
        "parameters": {"type": "object", "properties": {
            "label": {"type": "string", "description": "Filter by node label (Person/Topic/Activity/Genre)"},
            "keyword": {"type": "string", "description": "Fuzzy search in node_id, label, properties"},
        }}}},
    {"type": "function", "function": {
        "name": "graph_get_neighbors",
        "description": "Get all neighbors of a node (connected entities and relations).",
        "parameters": {"type": "object", "properties": {
            "node_id": {"type": "string", "description": "Node to query"},
            "relation": {"type": "string", "description": "Filter by relation type"},
            "direction": {"type": "string", "description": "'out', 'in', or 'both' (default: 'both')"},
        }, "required": ["node_id"]}}},
    {"type": "function", "function": {
        "name": "graph_get_subgraph",
        "description": "Get subgraph centered on a node (expand by depth).",
        "parameters": {"type": "object", "properties": {
            "node_id": {"type": "string", "description": "Center node"},
            "depth": {"type": "integer", "description": "Max traversal depth (default: 2)"},
        }, "required": ["node_id"]}}},
    {"type": "function", "function": {
        "name": "graph_add_node",
        "description": "Add or update a graph node.",
        "parameters": {"type": "object", "properties": {
            "node_id": {"type": "string", "description": "Node ID (lowercase slug)"},
            "label": {"type": "string", "description": "Node type (Person/Topic/Activity/Genre/Insight)"},
            "properties": {"type": "object", "description": "Node properties"},
            "timestamp": {"type": "string", "description": "Timestamp"},
        }, "required": ["node_id"]}}},
    {"type": "function", "function": {
        "name": "graph_add_edge",
        "description": "Add a relation edge between two nodes (auto-creates nodes if missing).",
        "parameters": {"type": "object", "properties": {
            "source": {"type": "string", "description": "Source node ID"},
            "target": {"type": "string", "description": "Target node ID"},
            "relation": {"type": "string", "description": "Relation type (lowercase verb phrase)"},
            "properties": {"type": "object", "description": "Edge properties"},
            "timestamp": {"type": "string", "description": "Relation timestamp"},
        }, "required": ["source", "target", "relation"]}}},
    {"type": "function", "function": {
        "name": "graph_delete_node",
        "description": "Delete a graph node and all its edges.",
        "parameters": {"type": "object", "properties": {
            "node_id": {"type": "string", "description": "Node to delete"},
        }, "required": ["node_id"]}}},
    {"type": "function", "function": {
        "name": "graph_delete_edge",
        "description": "Delete a specific edge by ID.",
        "parameters": {"type": "object", "properties": {
            "edge_id": {"type": "string", "description": "Edge ID to delete"},
        }, "required": ["edge_id"]}}},
    {"type": "function", "function": {
        "name": "graph_stats",
        "description": "Get graph statistics (node count, edge count, label/relation distributions).",
        "parameters": {"type": "object", "properties": {}}}},
    # === Finish ===
    {"type": "function", "function": {
        "name": "finish",
        "description": (
            "Complete the consolidation round. Result should be a structured summary "
            "following the C1-C8 format.\n\n"
            "IMPORTANT: Before calling finish, ensure you have:\n"
            "1. Updated .meta/index.md (at minimum: evolution log entry)\n"
            "2. Run `git add -A && git commit -m \"consolidate: <summary>\"`"
        ),
        "parameters": {"type": "object", "properties": {
            "result": {"type": "string", "description": "Structured C1-C8 summary of actions taken"},
        }, "required": ["result"]}}},
]


# ============================================================================
# Retrieve T3 — Query Rewrite
# ============================================================================

QUERY_REWRITE_SYSTEM_PROMPT = """\
You are a Query Rewriting Agent. Rewrite the user's conversation/question into multiple queries suitable for retrieving from a memory store.

## Output Format

Output a JSON object:
```json
{
  "queries": [
    {"type": "semantic", "text": "natural language query suitable for vector semantic retrieval"},
    {"type": "keyword", "text": "core keywords separated by spaces, suitable for BM25 matching"},
    {"type": "entity", "text": "single entity name (person, place, activity, topic) for graph retrieval"},
    {"type": "temporal", "text": "time expression for temporal filtering (e.g. '2017', '2019-03', '2018~2020')"},
    {"type": "fs_scope", "text": "comma-separated file paths or directory prefixes to narrow BM25 search scope"}
  ],
  "graph_config": {
    "depth": 3,
    "hop_top_k": {"1": 10, "2": 15, "3": 5}
  }
}
```

## MANDATORY Requirements

You MUST generate ALL of the following query types (minimum counts):
- **semantic**: At least 2 queries (natural language, different angles)
- **keyword**: At least 1 query (core nouns/verbs extracted from the conversation, NO stop words, space-separated)
- **entity**: At least 1 query (extract person names, activity names, topic names, place names — one entity per query entry)

The following are OPTIONAL (only generate when clearly applicable):
- **temporal**: Only if there is a clear time reference
- **fs_scope**: Only if you can confidently infer relevant paths from the filesystem structure

## graph_config (MANDATORY)

You MUST output a `graph_config` field to control graph traversal depth and per-hop retrieval limits based on query complexity:
- **depth**: How many hops to traverse in the knowledge graph. Minimum is 3. Use higher values (4-6) for multi-hop reasoning questions that involve chains of relationships (e.g. "What does the friend of Alex's colleague like?"). Use 3 for simple factual questions.
- **hop_top_k**: A mapping from hop number (as string key) to the maximum number of edges to keep at that hop. Closer hops should generally have more results. Example: {"1": 10, "2": 15, "3": 8, "4": 5}

Guidelines for setting graph_config:
- Simple factual query (e.g. "What is Alex's favorite food?") → depth=3, hop_top_k={"1": 10, "2": 10, "3": 5}
- Multi-entity query (e.g. "What do Alex and Bob have in common?") → depth=3, hop_top_k={"1": 12, "2": 15, "3": 8}
- Multi-hop reasoning (e.g. "What hobby does Alex's friend's sister enjoy?") → depth=5, hop_top_k={"1": 8, "2": 12, "3": 15, "4": 10, "5": 5}
- Temporal chain (e.g. "What happened after Alex met Bob?") → depth=4, hop_top_k={"1": 10, "2": 15, "3": 10, "4": 5}

## Rewriting Rules

1. **semantic query**: Keep natural language form, suitable for vector similarity matching. Generate 2-3 semantic queries from different angles. Focus on the core intent and information need.
2. **keyword query**: Extract the most important nouns, verbs, and proper nouns. Remove ALL stop words (the, a, is, are, was, I, my, etc.). Combine into space-separated keywords. Generate 1-2 keyword queries.
3. **entity query**: Extract ALL identifiable entities: person names, place names, activity names (e.g. "podcasting", "hiking"), topic names (e.g. "music theory", "literature"), genre names, etc. Generate ONE entity query per entity found. Even common topics count as entities.
4. **temporal query**: If the question mentions or implies a specific time period, extract it as a temporal filter. Use formats like "2017", "2019-03", "2018~2020". Only generate if there is a clear temporal reference. **Important**: When the question contains relative time expressions (e.g. "yesterday", "last week", "two months ago"), convert them to absolute dates using the "Current Time" provided in the user prompt before generating the temporal query.
5. **fs_scope query**: Based on the question's topic, infer which directories/files in the memory filesystem are most likely to contain relevant information. Use directory prefixes like "people/alex/", "events/", "preferences/". Generate 1 fs_scope query if you can reasonably infer the relevant paths. If unsure, do NOT generate this type.

## Memory Filesystem Structure

{fs_structure}

## Notes

- Extract retrieval intent from the last few turns of conversation
- If there is an explicit question in the conversation, rewrite around the question
- If it's an implicit memory need (e.g. recommendation, recall), infer what information the user wants
- Pay attention to temporal cues: words like "in 2017", "last year", "recently", "back in college" should be converted to temporal queries when possible
- For fs_scope: use the filesystem structure above to determine which paths are relevant. Common patterns:
  - Questions about a person → "people/<name>/"
  - Questions about events/timeline → "people/<name>/timeline.md" or "events/"
  - Questions about preferences → "people/<name>/preferences.md" or "preferences/"
  - General questions → omit fs_scope to search everything
- Output pure JSON, no markdown code blocks
- IMPORTANT: Do NOT output only semantic queries. You MUST include keyword and entity types.
- IMPORTANT: Always include graph_config in your output. Analyze the query complexity to determine appropriate depth and hop_top_k values.
"""

QUERY_REWRITE_USER_TEMPLATE = """\
## Current Time: {current_time}

## Conversation context to retrieve for:

{conversation}

---

Rewrite into multiple queries suitable for retrieving from the memory store. Output pure JSON.
Remember: only include fs_scope if you can confidently infer relevant paths from the filesystem structure.
If the conversation contains relative time expressions (e.g. "last week", "yesterday"), convert them to absolute dates based on the Current Time above when generating temporal queries.
"""


# ============================================================================
# Retrieve Prompt — fs structure renderer
# ============================================================================


def _extract_file_description(content: str) -> str:
    """Extract file metadata description from JSON-first-line or YAML front matter."""
    if not content:
        return ""

    lines = content.split("\n")
    first_line = lines[0].strip() if lines else ""

    if first_line.startswith("{"):
        try:
            meta = json.loads(first_line)
            if isinstance(meta, dict):
                return meta.get("description", "") or ""
        except json.JSONDecodeError:
            pass

    if first_line == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                for meta_line in lines[1:idx]:
                    stripped = meta_line.strip()
                    if stripped.startswith("description:"):
                        return stripped[len("description:"):].strip()
                break

    return ""


def build_fs_structure_from_files(
    files: dict[str, str],
    *,
    max_files: int = 50,
    include_truncation_note: bool = True,
) -> str:
    """Build the filesystem tree string injected into query rewrite prompts."""
    if not files:
        return "(empty filesystem)"

    items = sorted(files.items())
    truncated = len(items) > max_files
    if truncated:
        items = items[:max_files]

    dir_files: dict[str, list[tuple[str, str]]] = {}
    for file_path, content in items:
        parts = file_path.split("/")
        if len(parts) > 1:
            dir_key = "/".join(parts[:-1])
            filename = parts[-1]
        else:
            dir_key = ""
            filename = file_path

        meta_desc = _extract_file_description(content)
        dir_files.setdefault(dir_key, []).append((filename, meta_desc))

    lines: list[str] = ["filesystem/"]
    for dir_path in sorted(dir_files.keys()):
        if dir_path:
            depth = dir_path.count("/") + 1
            indent = "  " * depth
            lines.append(f"{indent}{dir_path}/")

        file_indent = "  " * (dir_path.count("/") + 2) if dir_path else "  "
        for filename, meta_desc in sorted(dir_files[dir_path]):
            if meta_desc and meta_desc != "(no meta info)":
                lines.append(f"{file_indent}{filename} — {meta_desc}")
            else:
                lines.append(f"{file_indent}{filename}")

    result = "\n".join(lines)
    if truncated and include_truncation_note:
        result += f"\n  ... (truncated, showing first {max_files} files)"
    return result


__all__ = [
    "INGEST_T3_TOOLS",
    "INGEST_T3_SYSTEM_PROMPT",
    "CONSOLIDATE_T3_TOOLS",
    "CONSOLIDATE_T3_SYSTEM_PROMPT",
    "QUERY_REWRITE_SYSTEM_PROMPT",
    "QUERY_REWRITE_USER_TEMPLATE",
    "build_fs_structure_from_files",
]
