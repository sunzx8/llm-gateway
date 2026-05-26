"""Consolidate T2 Agent Loop Task.

重型 Agent Loop 演进任务（最多 50 轮）。设计来源：``dev-0421-t3`` 分支的
``ConsolidateT3Task``。原本就只有 agent loop 一种模式（继承 T2BaseTask 的
execute() 共享 driver），本次迁移改为继承 :class:`BaseContextTask`，把
agent loop driver 内联到 :meth:`run`。

执行触发：由外部 :class:`ConsolidateTrigger` 决定（保留原工具类）。
本 task 只负责执行：注入完整的当前记忆状态，让 LLM 通过 agent loop 中
fs/vec/graph 全套读写工具去合并、去重、提升、生成 insight。
"""

from __future__ import annotations

import logging
from typing import Any

from context_task.base_context_task import BaseContextTask
from storage.file_system_store import FileSystemStore
from storage.stores_base import GraphStoreBase, VectorStoreBase
from utils.memory_llm_interface import LLMInterface

logger = logging.getLogger(__name__)


class ConsolidateTrigger:
    """Consolidate 触发策略管理。

    跟踪 ingest/query 调用次数，判断是否应该触发演进。
    """

    def __init__(
        self,
        ingest_interval: int = 5,   # 每 N 次 ingest 后触发
        query_interval: int = 5,    # 每 M 次 query 后触发
        min_memory_items: int = 5,  # 记忆条目数低于此阈值时不触发
        enabled: bool = True,       # 是否启用
    ):
        self.ingest_interval = ingest_interval
        self.query_interval = query_interval
        self.min_memory_items = min_memory_items
        self.enabled = enabled

        # 计数器
        self._ingest_count: int = 0
        self._query_count: int = 0
        self._last_consolidate_at_ingest: int = 0
        self._last_consolidate_at_query: int = 0
        self._consolidate_count: int = 0

    def record_ingest(self) -> None:
        """记录一次 ingest 调用。"""
        self._ingest_count += 1

    def record_query(self) -> None:
        """记录一次 query 调用。"""
        self._query_count += 1

    def should_trigger(self) -> bool:
        """判断是否应该触发 consolidate。

        Returns:
            True if consolidate should be triggered.
        """
        if not self.enabled:
            return False

        # 按 ingest 间隔触发
        ingests_since_last = self._ingest_count - self._last_consolidate_at_ingest
        if self.ingest_interval > 0 and ingests_since_last >= self.ingest_interval:
            return True

        # 按 query 间隔触发
        queries_since_last = self._query_count - self._last_consolidate_at_query
        if self.query_interval > 0 and queries_since_last >= self.query_interval:
            return True

        return False

    def mark_triggered(self) -> None:
        """标记已触发一次 consolidate。"""
        self._last_consolidate_at_ingest = self._ingest_count
        self._last_consolidate_at_query = self._query_count
        self._consolidate_count += 1

    @property
    def consolidate_count(self) -> int:
        return self._consolidate_count

    @property
    def ingest_count(self) -> int:
        return self._ingest_count

    @property
    def query_count(self) -> int:
        return self._query_count


# ---------------------------------------------------------------------------
# System Prompt — 演进 Agent
# ---------------------------------------------------------------------------

CONSOLIDATE_T2_AGENT_SYSTEM_PROMPT = """\
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
4. **NEVER delete reasons**: When merging or deduplicating, the WHY behind a stance \
change or decision must be preserved. "I quit X because Y" — the "because Y" part \
must survive every merge. If a vec entry has a reason and its duplicate doesn't, \
keep the one WITH the reason.
5. **NEVER break evolution chains**: If entries form a temporal sequence \
(liked → stopped → resumed), do NOT merge them into one. The chain IS the information.
6. **Language consistency**: All writes must match the dominant language of existing memory content
7. **Index maintenance**: ALWAYS update `.meta/index.md` before finishing (at minimum: append to evolution log)
8. **Git commit**: After all writes, execute `git add -A && git commit -m "consolidate: <summary>"`

## Checklist (C1–C9, review each — "no candidates" is a valid conclusion)

### C1. Deduplication & Conflict Resolution
- Semantic duplicates (same direction) → MERGE (keep most specific verbatim, add `merged_from:` list)
- Same topic with opposing stances → PAIR as past_stance + current_stance with `stance_anchor_id`
- Different facets of same topic → COEXIST (do not merge)

### C2. Cross-Session Insight Extraction
- ≥2 sessions showing same pattern → Create insight with `evidence: [source1, source2, ...]`
- Behavioral patterns (e.g. "always X when Y") → Extract as `user_behavior_pattern`
- Single-session observations → DO NOT promote (leave as `recent_event`)

### C3. Event → Preference Promotion
- Event mentioned across ≥2 sessions + category-level phrasing → Promote to `evidence_type: recurring`
- Single session + category statement → Mark as `single_session_declaration: true`
- Promote criteria: same direction + same category + multiple sessions

### C4. Stance Evolution Tracking
- "Used to X, now Y" patterns → Create paired entries with shared `stance_anchor_id`
- Temporal ordering via `narrative_time` / `occurred_at`
- Both graph edges AND vec entries should reflect evolution
- **Build explicit evolution chains**: When multiple entries about the same topic show
  different stances at different times, construct a full chronological chain and record
  it as a dedicated vec entry with `type: stance_evolution`:
  "My relationship with [topic] evolved: initially [state1] because [reason1],
  then [state2] because [reason2], now [state3] because [reason3]."
- Also add a chronological section in the relevant FS topical file:
  ```
  ## Stance Evolution
  - Phase 1 [YYYY]: description + reason
  - Phase 2 [YYYY]: changed because trigger + new stance
  - Phase 3 [YYYY]: current stance + reason
  ```
- Graph edges should form a chain: `liked → stopped (reason) → resumed (reason)`
  with each edge carrying `occurred_at` and `reason` properties

### C5. Graph Structure Optimization
- Merge near-duplicate nodes (e.g. "italian_food" and "italian_cuisine")
- Add missing edges when evidence supports relationships
- Update node/edge embeddings after content changes
- Ensure temporal info (`occurred_at`) is present on edges

### C6. File System Reorganization (CRITICAL for retrieval quality)

**Why this matters**: The retriever shows each file as a snippet to the main
agent. Files larger than ~2000 characters get TRUNCATED — content beyond the
cutoff is INVISIBLE to the answering model. This is the #1 cause of missed
facts in downstream QA.

**File size audit** (run `fs_execute_bash` with `wc -c` on each file):
- Files **> 2000 chars** → MUST split by sub-topic into smaller files
  (e.g. `music.md` → `music_preferences.md` + `music_events.md` + `music_stance_evolution.md`)
- Files **> 3000 chars** → URGENT split, key information is certainly being lost
- After splitting, update file metadata `{"description":"..."}` for each new file

**Splitting strategy**:
- Each sub-file should cover ONE retrievable topic / aspect
- Put Stance Evolution sections into their own file (they are high-value for QA)
- Put negative experiences / things the user quit into a dedicated file
  (e.g. `people/alex/negative_experiences.md`) — these are frequently queried
- Keep each sub-file under 1500 chars if possible
- Update `.meta/index.md` after every split

### C7. Vector DB Health
- Sample entries: check for missing metadata (source, type, subject, occurred_at)
- Fix third-person rewrites → first-person original phrasing
- Remove true duplicates (same text, same collection)
- Limit: ~30 fixes per consolidation round
- **Specificity audit**: Sample 10 entries. If any entry is an overly abstract
  restatement that lost concrete details (names, places, companions, reasons),
  check FS/graph for the original details and create a more specific companion entry.
  Example: abstract "I explored social activities" → specific "I signed up for a
  salsa dancing class with a friend to step out of my comfort zone"
- **Negative preference coverage**: For each `avoids`, `stopped`, `dropped_out_of`,
  or `dislikes` edge in the graph, verify a corresponding first-person vec entry
  exists that explains WHY (the reason is critical for downstream reasoning).
  If missing, reconstruct from FS content or graph edge properties and add it.

### C8. Index Self-Check (CRITICAL — used by ingest/retrieve as prompt-level directory)
- `.meta/index.md` is injected directly into ingest and retrieve prompts as context
- It MUST be accurate and comprehensive. Structure:
  ```
  # Memory Index

  ## File System
  <list each file with path and 1-line description>

  ## Source Sessions (raw conversation logs — ground truth for retrieval)
  <list source_sessions/ files with session_id and brief topic summary>
  (These are the original conversations; retrieval can search them for verbatim user quotes)

  ## Vector Collections
  <list each collection name, entry count, and topic hint>

  ## Graph Database
  <node label distribution, relation types, total counts>

  ## Evolution Log
  <append this round's actions summary>
  ```
- Keep total length under ~2000 tokens (truncate directory tail if needed)
- Update after ANY structural change (new files, new collections, merged nodes, etc.)
- **Source sessions listing**: Read `source_sessions/` directory, list each file with
  its session_id and a 5-10 word topic summary (inferred from filename or first few lines).
  This enables the retrieve agent to search raw conversations when extracted memories
  are insufficient.

### C9. Query History Reflection (ACTIONABLE — guides C1-C8 + C10 priorities)
- Read `.meta/queries.jsonl` if it exists (each line: {question, rewritten_queries, retrieved_summary})
- **Identify retrieval gaps**:
  - Questions where `retrieved_summary` is very short or "(No relevant memories found)"
    → The topic exists in conversations but was NOT stored or is NOT retrievable
    → Action: check ingest output for this topic; if stored, check whether FS file
    names / vec metadata / graph nodes contain the right keywords for retrieval
  - Questions where `retrieved_summary` shows truncated FS content (`... (N chars total)`)
    → The file is too large and critical info was cut off
    → Action: SPLIT the file per C6 rules (this is the highest-priority action)
- **Identify high-frequency topics**: Topics queried multiple times → ensure they
  have excellent retrieval paths (all three backends, keyword-rich FS content,
  specific vec entries)
- **Do NOT delete** the queries.jsonl file, just use it to prioritize your C1-C8 work

### C10. Retrieval Path Optimization (CRITICAL for downstream quality)

This is the **most impactful** checklist item — directly determines whether future queries find the right memories.

**9a. Topic Consolidation Files (FS → BM25 retrieval)**
- Identify scattered facts about the same topic across multiple files
- Consolidate into dedicated topical files with clear, keyword-rich content
- Example: if "cooking" facts are in `facts.md` L12, `events.md` L7, `preferences.md` L3 → create/update `people/<name>/cooking.md` aggregating all cooking-related info
- Each topical file should be self-contained: reading just that file gives a complete picture of the topic
- File names should be descriptive and searchable (avoid generic names like `misc.md`)

**9b. Cross-Backend Linkage (redundancy for recall)**
- For every IMPORTANT fact (recurring preference, strong stance, life event):
  - FS: fact exists in the right topical file with keywords that BM25 can match
  - Vec: an atomic first-person statement exists with proper metadata
  - Graph: relevant entity nodes + stance/relation edges exist
- If any backend is MISSING coverage for an important fact → ADD it
- This ensures retrieval succeeds regardless of which backend the retriever queries first

**9c. Semantic Quality of Vec Entries**
- Each vec entry should answer: "What would someone type to find this information?"
- Bad: "mentioned something about food" (too vague for semantic match)
- Good: "I love Italian food, especially handmade pasta from small trattorias" (specific, first-person, keyword-rich)
- Rewrite low-quality entries to maximize semantic retrieval hit rate
- Add SYNONYM-RICH entries for concepts that can be queried in multiple ways
  - e.g. if user likes "podcasts", also ensure the vec entry contains "audio shows" / "listening"

**9d. Graph as Retrieval Router**
- Graph should serve as a "topic discovery" layer — when retriever finds a node, its neighbors reveal WHAT to search for in FS/Vec
- Ensure every Person node has edges to their major topics/activities
- Add `evidence_path` property to edges: `{"fs_path": "people/alex/cooking.md", "vec_ids": ["vec_1", "vec_2"]}`
- This lets the retriever do: graph_search → find related topics → targeted FS/Vec search

**9e. File Metadata for Scope Inference**
- The retriever uses file metadata `{"description":"..."}` to infer `fs_scope` (which files to search)
- Ensure every file's description accurately reflects its CURRENT content (not stale from creation time)
- Descriptions should contain the KEY TERMS a user might query:
  - Bad: `{"description": "Various preferences"}` (too generic)
  - Good: `{"description": "Alex's food and dining preferences: Italian cuisine, pasta, cooking classes, restaurant choices"}` (keyword-rich, specific)

**9f. Temporal Organization**
- Ensure all time-referenced facts have `occurred_at` in Vec metadata AND graph edge properties
- Create timeline entries in FS for life events (enables temporal queries like "what happened in 2019?")
- If a file mixes dated and undated entries, consider reordering: dated entries chronologically, undated at the end

### C11. Preference Profile Synthesis (CONSERVATIVE, evidence-bound)

This is only for recommendation / suggestion questions. Build or update a compact profile
ONLY when it improves retrieval and only when there is clear evidence.

Strict rules:
- Do NOT create broad generic profiles from sparse evidence.
- Do NOT let profiles replace concrete memories. Concrete facts remain primary.
- Only synthesize a profile bullet if it is supported by ≥2 concrete memories OR the user explicitly states a general preference.
- Each profile bullet must include a short evidence hint (file path, topic, or reason).
- Keep profiles compact: max 8 bullets for positives, max 8 bullets for constraints.
- Avoid duplicate vec entries for the same profile/constraint.

Recommended files:
- `people/<name>/preference_profile.md` — compact positives and decision criteria
- `people/<name>/recommendation_constraints.md` — hard avoids, stressors, safety/budget/time constraints

Vec entries may use metadata `type: preference_profile` or `type: recommendation_constraint`, but keep them concrete:
Good: "I prefer low-pressure creative activities because high-scrutiny performance settings have repeatedly made me anxious."
Bad: "I like creative things."

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
C11: <preference profiles/constraints synthesized or "no candidates">
Total operations: N writes, M updates, K merges
```
"""


# ---------------------------------------------------------------------------
# User Prompt Template
# ---------------------------------------------------------------------------

CONSOLIDATE_T2_AGENT_USER_TEMPLATE = """\
## Current Memory State

### File System Structure ({fs_file_count} files):
{fs_tree}


### Vector DB ({vec_total} entries across {vec_collections_count} collections):
{vec_collections_detail}

### Vector DB Sample Entries (up to 10 per collection):
{vec_samples}

### Graph DB ({graph_nodes} nodes, {graph_edges} edges):
Node labels: {graph_labels}
Relation types: {graph_relations}

### Recent Query History (.meta/queries.jsonl):
{query_history}

### Current Index (.meta/index.md):
{current_index}

---

## Changes Since Last Consolidation (REVIEW THESE FIRST)

{changes_since_last}

---

## Task

Review the above memory state and perform evolution operations following the C1-C10 checklist.

**Priorities for this round:**
1. **Start from "Changes Since Last Consolidation"** — these are NEW writes since last round, focus your dedup/merge/organize efforts here
2. Look for DUPLICATE or CONFLICTING information across backends (especially between new and old entries)
3. Check if any single-session events now have cross-session evidence (→ promote)
4. Identify stance evolution patterns (used to X → now Y)
5. Update index.md with any structural changes

**Remember:**
- Read files and search before writing (verify current state)
- Language must match existing content
- Conservative: when unsure, keep both versions
- Git commit before finish
- ALWAYS update .meta/index.md (at minimum: evolution log entry)
"""


# ---------------------------------------------------------------------------
# Tool Definitions — 完整读写工具集
# ---------------------------------------------------------------------------

CONSOLIDATE_T2_AGENT_TOOLS: list[dict] = [
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


# ---------------------------------------------------------------------------
# Consolidate T3 Task
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class ConsolidateT2AgentLoopTask(BaseContextTask):
    """Atomic T2 多后端 consolidate 任务（重型 agent loop 模式）。

    特点：
    - max_turns 默认 50（重型任务，允许较多轮次）
    - pre-check：若总条目数 < ``MIN_ITEMS_FOR_EVOLUTION``（默认 5），跳过
    - prompt 注入完整的当前记忆状态（FS 文件内容 + Vec 采样 + Graph edges 列表）
    """

    task_name = "consolidate_t2_agent_loop"

    # 记忆条目低于此阈值时跳过演进
    MIN_ITEMS_FOR_EVOLUTION = 5

    def __init__(
        self,
        llm: LLMInterface,
        fs_store: FileSystemStore,
        vec_store: VectorStoreBase,
        graph_store: GraphStoreBase,
        *,
        max_turns: int = 50,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm)
        self.fs = fs_store
        self.vec = vec_store
        self.graph = graph_store
        self.max_turns = max_turns
        self.extra_kwargs = kwargs
        self.result_extra: dict[str, Any] = {}

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
    # 主 run（BaseContextTask 钩子）
    # ------------------------------------------------------------------

    async def run(
        self,
        user_id: str = "default_user",
        session_id: str = "",
        **kwargs: Any,
    ) -> None:
        """执行 consolidate agent loop。"""
        # ---- pre-check：记忆太小时跳过 ----
        fs_files = self.fs.list_files() if self.fs else []
        vec_stats0 = self.vec.get_stats() if self.vec else {}
        graph_stats0 = self.graph.get_stats() if self.graph else {}
        total_items = (
            len(fs_files)
            + (vec_stats0.get("total_entries", 0) if isinstance(vec_stats0, dict) else 0)
            + (graph_stats0.get("total_nodes", 0) if isinstance(graph_stats0, dict) else 0)
        )
        if total_items < self.MIN_ITEMS_FOR_EVOLUTION:
            logger.info(
                "[%s] memory too small (%d items), skipping consolidation",
                self.task_name, total_items,
            )
            self.result_extra["consolidate_skipped"] = True
            self.result_extra["consolidate_skip_reason"] = (
                f"only {total_items} items, threshold={self.MIN_ITEMS_FOR_EVOLUTION}"
            )
            return

        # ---- 构建 user prompt ----
        user_prompt = self._build_user_prompt(kwargs.get("changes_since_last") or [])

        # ---- agent loop ----
        system_prompt = CONSOLIDATE_T2_AGENT_SYSTEM_PROMPT
        tools = CONSOLIDATE_T2_AGENT_TOOLS
        max_turns = self.max_turns

        messages_history: list[dict[str, Any]] = [
            {"role": "user", "content": user_prompt},
        ]
        agent_loop_trace: list[dict[str, Any]] = []
        finish_summary = ""
        steps_used = 0

        stats: dict[str, Any] = {
            "fs_writes": 0,
            "fs_updates": 0,
            "vec_adds": 0,
            "vec_deletes": 0,
            "graph_ops": 0,
            "merges": 0,
            "promotions": 0,
            "insights_created": 0,
        }

        for turn in range(max_turns):
            steps_used = turn + 1
            response = await self.llm_generate_with_stat(
                system=system_prompt,
                messages=messages_history,
                tools=tools,
                label=f"consolidate_agent_turn{turn + 1}",
            )

            step_trace: dict[str, Any] = {
                "step": turn + 1,
                "model_content": (response.content or "")[:500],
                "tool_calls_count": len(response.tool_calls or []),
                "tool_calls": [],
            }

            if not response.tool_calls:
                if response.content:
                    finish_summary = response.content[:1000]
                agent_loop_trace.append(step_trace)
                break

            messages_history.append(response.to_message())
            should_break = False

            for tc in response.tool_calls:
                if tc.name == "finish":
                    finish_summary = (
                        (tc.arguments or {}).get("result")
                        or (tc.arguments or {}).get("changes_summary")
                        or (tc.arguments or {}).get("summary")
                        or ""
                    )
                    step_trace["tool_calls"].append({
                        "tool": "finish",
                        "arguments": tc.arguments,
                        "result": finish_summary[:2000],
                    })
                    messages_history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "(consolidation finished)",
                    })
                    should_break = True
                    continue

                try:
                    result_str = await self.tool_with_stat(
                        tc.name,
                        self._execute_tool,
                        tc.name, tc.arguments or {}, stats,
                        arguments_summary={"tool": tc.name, **{
                            k: str(v)[:80] for k, v in (tc.arguments or {}).items()
                        }},
                    )
                except Exception as e:
                    logger.exception("Tool %s failed: %s", tc.name, e)
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
                "[%s] consolidate agent loop reached max_turns=%d without finish",
                self.task_name, max_turns,
            )

        # Git commit（best-effort）
        try:
            self.fs.execute_bash(
                command=f'git add -A && git commit -m "consolidate_t2_agent: {finish_summary[:50]}" --allow-empty',
                timeout=10,
                allow_write=True,
            )
        except Exception as e:
            logger.warning("Git commit failed: %s", e)

        # 写入 result_extra
        self.result_extra["consolidate_finish_summary"] = finish_summary
        self.result_extra["consolidate_turns_used"] = steps_used
        self.result_extra["consolidate_max_turns"] = max_turns
        self.result_extra["fs_writes"] = stats["fs_writes"]
        self.result_extra["fs_updates"] = stats["fs_updates"]
        self.result_extra["vec_adds"] = stats["vec_adds"]
        self.result_extra["vec_deletes"] = stats["vec_deletes"]
        self.result_extra["graph_ops"] = stats["graph_ops"]
        self.result_extra["merges"] = stats["merges"]
        self.result_extra["promotions"] = stats["promotions"]
        self.result_extra["insights_created"] = stats["insights_created"]
        self.result_extra["agent_loop_trace_summary"] = [
            {
                "step": st["step"],
                "tool_calls_count": st["tool_calls_count"],
                "tools": [tc["tool"] for tc in st["tool_calls"]],
            }
            for st in agent_loop_trace
        ]

        logger.info(
            "[%s] consolidate done (turns=%d, fs_w=%d, fs_u=%d, vec_a=%d, vec_d=%d, graph=%d, summary=%.200s)",
            self.task_name, steps_used,
            stats["fs_writes"], stats["fs_updates"], stats["vec_adds"],
            stats["vec_deletes"], stats["graph_ops"], finish_summary,
        )

    # ------------------------------------------------------------------
    # Prompt 构建
    # ------------------------------------------------------------------

    def _build_user_prompt(self, changes_list: list[str]) -> str:
        """构建 user prompt：注入完整的当前记忆状态。"""
        fs_files = self.fs.list_files()
        vec_stats = self.vec.get_stats()
        graph_stats = self.graph.get_stats()

        fs_tree = self.fs.tree(max_depth=4)

        fs_contents_parts: list[str] = []
        for f in fs_files:
            content = self.fs.read_file(f)
            if isinstance(content, str) and content.startswith("ERROR"):
                continue
            lines = content.split("\n")
            char_count = len(content)
            size_warning = ""
            if char_count > 3000:
                size_warning = " ⚠️ OVERSIZED — MUST SPLIT per C6"
            elif char_count > 2000:
                size_warning = " ⚠️ LARGE — consider splitting per C6"
            numbered = []
            for i, line in enumerate(lines[:50], 1):
                numbered.append(f"  L{i}: {line}")
            if len(lines) > 50:
                numbered.append(f"  ... ({len(lines) - 50} more lines)")
            fs_contents_parts.append(
                f"#### {f} ({len(lines)} lines, {char_count} chars{size_warning})\n"
                + "\n".join(numbered)
            )
        fs_contents = "\n\n".join(fs_contents_parts) if fs_contents_parts else "(no files)"

        vec_collections_map = (
            vec_stats.get("collections", {}) if isinstance(vec_stats, dict) else {}
        )
        vec_collections_detail = "\n".join(
            f"  - {name}: {size} entries"
            for name, size in sorted(vec_collections_map.items())
        ) if vec_collections_map else "  (no collections)"

        vec_samples_parts: list[str] = []
        if hasattr(self.vec, "_collections"):
            for coll_name, entries in self.vec._collections.items():
                sample_lines = []
                for entry in entries[:10]:
                    meta = entry.get("metadata", {}) or {}
                    meta_str = ", ".join(f"{k}={v}" for k, v in meta.items() if v)
                    sample_lines.append(
                        f"    [id={entry.get('id', '?')}] {entry.get('text', '')[:200]}"
                        + (f"\n      metadata: {meta_str}" if meta_str else "")
                    )
                if len(entries) > 10:
                    sample_lines.append(f"    ... ({len(entries) - 10} more entries)")
                vec_samples_parts.append(
                    f"  Collection '{coll_name}' ({len(entries)} total):\n"
                    + "\n".join(sample_lines)
                )
        vec_samples = "\n\n".join(vec_samples_parts) if vec_samples_parts else "  (no entries)"

        graph_labels_map = (
            graph_stats.get("node_labels", {}) if isinstance(graph_stats, dict) else {}
        )
        graph_rel_map = (
            graph_stats.get("relation_types", {}) if isinstance(graph_stats, dict) else {}
        )
        graph_labels_str = (
            ", ".join(f"{k}:{v}" for k, v in sorted(graph_labels_map.items())) or "(none)"
        )
        graph_relations_str = (
            ", ".join(f"{k}:{v}" for k, v in sorted(graph_rel_map.items())) or "(none)"
        )

        graph_edges_parts: list[str] = []
        if hasattr(self.graph, "_edges"):
            for edge in self.graph._edges[:100]:
                props = edge.get("properties", {})
                props_str = f" {props}" if props else ""
                graph_edges_parts.append(
                    f"  [{edge.get('id', '?')}] {edge.get('source', '')} "
                    f"--[{edge.get('relation', '')}]--> {edge.get('target', '')}{props_str}"
                )
            if len(self.graph._edges) > 100:
                graph_edges_parts.append(
                    f"  ... ({len(self.graph._edges) - 100} more edges)"
                )
        graph_edges_detail = "\n".join(graph_edges_parts) if graph_edges_parts else "  (no edges)"

        if changes_list:
            changes_text = "\n".join(f"- {c}" for c in changes_list)
        else:
            changes_text = "(no ingest changes since last consolidation)"

        return CONSOLIDATE_T2_AGENT_USER_TEMPLATE.format(
            fs_file_count=len(fs_files),
            fs_tree=fs_tree,
            fs_contents=fs_contents,
            vec_total=vec_stats.get("total_entries", 0) if isinstance(vec_stats, dict) else 0,
            vec_collections_count=len(vec_collections_map),
            vec_collections_detail=vec_collections_detail,
            vec_samples=vec_samples,
            graph_nodes=graph_stats.get("total_nodes", 0) if isinstance(graph_stats, dict) else 0,
            graph_edges=graph_stats.get("total_edges", 0) if isinstance(graph_stats, dict) else 0,
            graph_labels=graph_labels_str,
            graph_relations=graph_relations_str,
            graph_edges_detail=graph_edges_detail,
            query_history=self._read_query_history(),
            current_index=self._read_current_index(),
            changes_since_last=changes_text,
        )

    def _read_query_history(self) -> str:
        """读取 .meta/queries.jsonl 的最近 20 条记录。"""
        try:
            content = self.fs.read_file(".meta/queries.jsonl")
            if isinstance(content, str) and content.startswith("ERROR"):
                return "(no query history yet)"
            lines = (content or "").strip().split("\n")
            recent = lines[-20:] if len(lines) > 20 else lines
            return "\n".join(recent)
        except Exception:
            return "(no query history yet)"

    def _read_current_index(self) -> str:
        """读取当前 .meta/index.md 内容。"""
        try:
            if hasattr(self.fs, "read_index"):
                _, content = self.fs.read_index()
            else:
                content = self.fs.read_file(".meta/index.md")
            if isinstance(content, str) and content.startswith("ERROR"):
                return "(no index yet)"
            return content or "(no index yet)"
        except Exception:
            return "(no index yet)"

    # ------------------------------------------------------------------
    # Tool dispatch（read + write 全套，从 dev-0421-t3 ConsolidateT3Task.execute_tool 直接保留）
    # ------------------------------------------------------------------

    async def _execute_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        stats: dict[str, Any],
    ) -> str:
        try:
            # === File System ===
            if tool_name in {"fs_read", "fs_read_file", "read_file"}:
                path = self._normalize_fs_path(args.get("path", ""))
                if not path:
                    return "ERROR: fs_read requires 'path'"
                return self.fs.read_file(path)
            elif tool_name == "fs_tree":
                return self.fs.tree(int(args.get("max_depth", 3)))
            elif tool_name == "fs_search":
                query = args.get("query", "")
                if not query:
                    return "ERROR: fs_search requires 'query'"
                results = self.fs.search_bm25(query, int(args.get("top_k", 10)))
                return str(results)
            elif tool_name == "fs_grep":
                pattern = args.get("pattern", "")
                if not pattern:
                    return "ERROR: fs_grep requires 'pattern'"
                results = self.fs.grep(
                    pattern=pattern,
                    paths=self._normalize_fs_paths(args.get("paths", ".")),
                    context_lines=int(args.get("context_lines", 2)),
                    max_matches=int(args.get("max_matches", 50)),
                    case_insensitive=args.get("case_insensitive", True),
                    regex=args.get("regex", True),
                )
                return str(results)
            elif tool_name == "fs_read_lines":
                path = self._normalize_fs_path(args.get("path", ""))
                if not path:
                    return "ERROR: fs_read_lines requires 'path'"
                return self.fs.read_lines(
                    rel_path=path,
                    start=int(args.get("start", 1)),
                    end=int(args["end"]) if args.get("end") is not None else None,
                )
            elif tool_name == "fs_write":
                path = self._normalize_fs_path(args.get("path", ""))
                if not path:
                    return "ERROR: fs_write requires 'path'"
                result = self.fs.write_file(path, args.get("content", ""))
                stats["fs_writes"] = stats.get("fs_writes", 0) + 1
                return result
            elif tool_name == "fs_append":
                path = self._normalize_fs_path(args.get("path", ""))
                if not path:
                    return "ERROR: fs_append requires 'path'"
                result = self.fs.append_file(path, args.get("content", ""))
                stats["fs_updates"] = stats.get("fs_updates", 0) + 1
                return result
            elif tool_name == "fs_delete":
                path = self._normalize_fs_path(args.get("path", ""))
                if not path:
                    return "ERROR: fs_delete requires 'path'"
                return self.fs.delete_file(path)
            elif tool_name == "fs_execute_bash":
                command = args.get("command", "")
                if not command:
                    return "ERROR: missing required arg 'command'"
                return self.fs.execute_bash(
                    command=command,
                    timeout=args.get("timeout", 15),
                    allow_write=True,
                )
            # === Vector DB ===
            elif tool_name == "vec_search":
                collection = args.get("collection", "")
                query = args.get("query", "")
                if not query:
                    return "ERROR: vec_search requires 'query'"
                if not collection:
                    results = await self.vec.search_all(
                        query=query,
                        top_k=int(args.get("top_k", 10)),
                        metadata_filter=args.get("filter") or None,
                    )
                    return str(results)
                results = await self.vec.search(
                    collection=collection,
                    query=query,
                    top_k=int(args.get("top_k", 10)),
                    metadata_filter=args.get("filter") or None,
                )
                return str(results)
            elif tool_name == "vec_search_all":
                query = args.get("query", "")
                if not query:
                    return "ERROR: vec_search_all requires 'query'"
                results = await self.vec.search_all(
                    query=query,
                    top_k=int(args.get("top_k", 15)),
                    metadata_filter=args.get("filter") or None,
                )
                return str(results)
            elif tool_name == "vec_add":
                # 适配两种参数格式：
                # 新格式（schema）: items=[{text, metadata}, ...]
                # 旧格式（兼容）: texts=[...], metadatas=[...]
                if "items" in args:
                    items = [it for it in args["items"] if isinstance(it, dict) and it.get("text")]
                    texts = [item["text"] for item in items]
                    metadatas = [item.get("metadata") for item in items]
                    # 去掉全 None 的 metadatas
                    if all(m is None for m in metadatas):
                        metadatas = None
                else:
                    texts = args.get("texts") or []
                    metadatas = args.get("metadatas")
                if not texts:
                    return "vec_add: no valid texts provided"
                collection = args.get("collection", "")
                if not collection:
                    return "ERROR: vec_add requires 'collection'"
                ids = await self.vec.add(
                    collection=collection,
                    texts=texts,
                    metadatas=metadatas,
                )
                stats["vec_adds"] = stats.get("vec_adds", 0) + len(texts)
                return f"Added {len(ids)} entries to '{collection}': {ids}"
            elif tool_name == "vec_delete":
                collection = args.get("collection", "")
                ids_to_del = args.get("ids", [])
                if not collection or not ids_to_del:
                    return "ERROR: vec_delete requires 'collection' and 'ids'"
                result = self.vec.delete(collection, ids_to_del)
                stats["vec_deletes"] = stats.get("vec_deletes", 0) + len(ids_to_del)
                return str(result)
            elif tool_name == "vec_list_collections":
                return str(self.vec.get_stats())
            # === Graph DB ===
            elif tool_name == "graph_search_nodes":
                results = self.graph.search_nodes(
                    label=args.get("label"),
                    keyword=args.get("keyword"),
                )
                return str(results)
            elif tool_name == "graph_get_neighbors":
                node_id = args.get("node_id", "")
                if not node_id:
                    return "ERROR: graph_get_neighbors requires 'node_id'"
                results = self.graph.get_neighbors(
                    node_id,
                    relation=args.get("relation"),
                    direction=args.get("direction", "both"),
                )
                return str(results)
            elif tool_name == "graph_get_subgraph":
                node_id = args.get("node_id", "")
                if not node_id:
                    return "ERROR: graph_get_subgraph requires 'node_id'"
                result = self.graph.get_subgraph(
                    node_id,
                    depth=args.get("depth", 2),
                )
                return str(result)
            elif tool_name == "graph_add_node":
                node_id = args.get("node_id", "")
                if not node_id:
                    return "ERROR: graph_add_node requires 'node_id'"
                result = self.graph.add_node(
                    node_id=node_id,
                    label=args.get("label", ""),
                    properties=args.get("properties"),
                    timestamp=args.get("timestamp", ""),
                )
                stats["graph_ops"] = stats.get("graph_ops", 0) + 1
                return result
            elif tool_name == "graph_add_edge":
                source = args.get("source", "")
                target = args.get("target", "")
                relation = args.get("relation", "")
                if not source or not target or not relation:
                    return "ERROR: graph_add_edge requires 'source', 'target', 'relation'"
                result = self.graph.add_edge(
                    source=source,
                    target=target,
                    relation=relation,
                    properties=args.get("properties"),
                    timestamp=args.get("timestamp", ""),
                )
                stats["graph_ops"] = stats.get("graph_ops", 0) + 1
                return result
            elif tool_name == "graph_delete_node":
                node_id = args.get("node_id", "")
                if not node_id:
                    return "ERROR: graph_delete_node requires 'node_id'"
                return self.graph.delete_node(node_id)
            elif tool_name == "graph_delete_edge":
                edge_id = args.get("edge_id", "")
                if not edge_id:
                    return "ERROR: graph_delete_edge requires 'edge_id'"
                return self.graph.delete_edge(edge_id)
            elif tool_name == "graph_stats":
                return str(self.graph.get_stats())
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            logger.error("ConsolidateT3Task tool error (%s): %s", tool_name, e)
            return f"ERROR: {e}"

__all__ = [
    "ConsolidateT2AgentLoopTask",
    "ConsolidateTrigger",
    "CONSOLIDATE_T2_AGENT_TOOLS",
    "CONSOLIDATE_T2_AGENT_SYSTEM_PROMPT",
    "CONSOLIDATE_T2_AGENT_USER_TEMPLATE",
]
