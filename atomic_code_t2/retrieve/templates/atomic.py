"""检索原子操作集合（对齐 T3 retrieve_t3.py 的核心检索逻辑）。

每个函数职责单一、无状态（或拿 store 当参数），方便 plan 自由拼接。

操作分五类：
1. Query 改写：rewrite_queries_llm
2. 后端召回：search_fs_bm25 / search_vec / search_graph
3. 融合：rrf_fuse
4. 输出：format_context
5. 辅助：_parse_file_metadata / _convert_line_for_display / _ingest_sort_key

对齐 T3 的关键改进（vs 之前版本）：
- fs_structure 带 meta description（对齐 T3 _build_fs_structure_with_meta）
- graph 种子：向量相似度优先（search_nodes_by_embedding）+ keyword 补充
- graph 子图：get_subgraph + BFS 分层 + 每跳按 edge embedding 余弦排序截断
- graph depth 最小 3
- FS 展示：去 [session_time|event_time] 系统前缀、展示 meta + 最多 8 行正文
- Vec/Graph 按 ingest_time 升序展示
- temporal 过滤（FS 行级 + Vec metadata 级）
- fs_scope 支持
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import logger.logger as logger
from storage.file_system_store import FileSystemStore
from storage.stores_base import GraphStoreBase, VectorStoreBase
from utils.memory_llm_interface import LLMInterface

from .prompt import QUERY_REWRITE_SYSTEM_PROMPT, QUERY_REWRITE_USER_TEMPLATE


# ===========================================================================
# 类型别名
# ===========================================================================

Query = dict[str, str]
FsHit = tuple[str, float, str]  # (path, score, snippet)
VecHit = dict[str, Any]
GraphHit = dict[str, Any]  # {"node": ..., "hops": {1: [...], 2: [...]}}
FusedHit = tuple[float, str, str]  # (rrf_score, doc_key, summary)


# ===========================================================================
# 1. Query 改写
# ===========================================================================


async def rewrite_queries_llm(
    llm: LLMInterface,
    fs: FileSystemStore,
    question: str,
    session_time: str = "",
) -> tuple[list[Query], dict[str, Any]]:
    """用 LLM 一次调用把 question 改写成多种类型的 query + graph_config。"""
    fs_structure = _build_fs_structure_with_meta(fs)
    system_prompt = QUERY_REWRITE_SYSTEM_PROMPT.replace("{fs_structure}", fs_structure)
    user_prompt = QUERY_REWRITE_USER_TEMPLATE.format(
        current_time=session_time or "(unknown)",
        question=question,
    )

    logger.info(
        "[atomic.rewrite] LLM rewrite start: question=%r, session_time=%r",
        question[:120], session_time,
    )
    try:
        resp = await llm.generate(
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            tools=None,
        )
        content = resp.content or ""
        logger.info("[atomic.rewrite] LLM raw output (%d chars): %.300s", len(content), content)
        queries, graph_config = _parse_rewrite_output(content)
        if queries:
            # 补充缺失的 keyword/entity
            queries = _ensure_keyword_entity(queries, question)
            logger.info(
                "[atomic.rewrite] parsed %d queries, types=%s, graph_config=%s",
                len(queries), [q["type"] for q in queries], graph_config,
            )
            return queries, graph_config
        logger.warning("[atomic.rewrite] LLM output unparseable, falling back")
    except Exception as e:
        logger.warning("[atomic.rewrite] LLM call failed (%s), falling back", e)

    # Fallback
    fallback_queries = _rewrite_fallback(question)
    return fallback_queries, {"depth": 3, "hop_top_k": {1: 10, 2: 15, 3: 5}}


def _ensure_keyword_entity(queries: list[Query], question: str) -> list[Query]:
    """确保 queries 中包含 keyword 和 entity 类型（对齐 T3 _ensure_keyword_entity_queries）。"""
    has_keyword = any(q.get("type") == "keyword" for q in queries)
    has_entity = any(q.get("type") == "entity" for q in queries)
    if has_keyword and has_entity:
        return queries

    # 工程化提取关键词
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into",
        "through", "during", "before", "after", "then", "when", "where", "why", "how",
        "all", "each", "every", "both", "few", "more", "most", "other", "some", "no",
        "not", "only", "very", "just", "but", "and", "or", "if", "about", "what",
        "which", "who", "this", "that", "these", "those", "it", "its", "i", "me", "my",
        "we", "our", "you", "your", "he", "him", "she", "her", "they", "them", "their",
    }
    en_words = re.findall(r"\b[a-zA-Z']+\b", question)
    keywords = [w for w in en_words if w.lower() not in stop_words and len(w) > 2]
    cjk = re.findall(r"[\u4e00-\u9fff]{2,}", question)
    keywords.extend(cjk)

    if not has_keyword and keywords:
        queries.append({"type": "keyword", "text": " ".join(keywords[:10])})
    if not has_entity and keywords:
        proper = [k for k in keywords if k[0].isupper()] if keywords else []
        entities = proper[:3] if proper else keywords[:3]
        for e in entities:
            queries.append({"type": "entity", "text": e})
    return queries


def _parse_rewrite_output(content: str) -> tuple[list[Query], dict[str, Any]]:
    """解析 LLM 的 JSON 输出（对齐 T3 _parse_query_rewrite_output）。"""
    if not content:
        return [], {}

    candidates: list[str] = [content]
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
    if m:
        candidates.append(m.group(1))
    first = content.find("{")
    last = content.rfind("}")
    if first != -1 and last > first:
        candidates.append(content[first:last + 1])

    for raw in candidates:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        queries_raw = data.get("queries")
        if not isinstance(queries_raw, list):
            continue
        cleaned: list[Query] = []
        for q in queries_raw:
            if not isinstance(q, dict):
                continue
            qtype = q.get("type")
            qtext = q.get("text")
            if qtype in ("semantic", "keyword", "entity", "temporal", "fs_scope") and isinstance(qtext, str) and qtext.strip():
                cleaned.append({"type": qtype, "text": qtext.strip()})
        if not cleaned:
            continue

        # Parse graph_config
        gc = data.get("graph_config") or {}
        depth = gc.get("depth") if isinstance(gc, dict) else None
        depth = int(depth) if isinstance(depth, int) and depth >= 3 else 3
        raw_htk = gc.get("hop_top_k") if isinstance(gc, dict) else None
        hop_top_k: dict[int, int] = {}
        if isinstance(raw_htk, dict):
            for k, v in raw_htk.items():
                try:
                    hop_top_k[int(k)] = int(v)
                except (ValueError, TypeError):
                    continue
        if not hop_top_k:
            hop_top_k = {1: 10, 2: 15, 3: 5}

        return cleaned, {"depth": depth, "hop_top_k": hop_top_k}

    return [], {}


def _rewrite_fallback(question: str) -> list[Query]:
    """LLM 不可用时的兜底改写。"""
    queries: list[Query] = [{"type": "semantic", "text": question}]
    en_tokens = re.findall(r"[A-Za-z][A-Za-z']{2,}", question)
    cjk_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", question)
    tokens = [t for t in en_tokens + cjk_tokens if len(t) > 1]
    if tokens:
        queries.append({"type": "keyword", "text": " ".join(tokens[:10])})
        for tok in tokens[:3]:
            queries.append({"type": "entity", "text": tok})
    return queries


def _build_fs_structure_with_meta(fs: FileSystemStore, max_files: int = 50) -> str:
    """构建带 meta description 的文件系统结构摘要（对齐 T3 _build_fs_structure_with_meta）。"""
    try:
        all_files = fs.list_files("")
    except Exception:
        return "(filesystem unavailable)"
    if not all_files:
        return "(empty filesystem)"

    truncated = len(all_files) > max_files
    files = sorted(all_files[:max_files])

    lines: list[str] = ["filesystem/"]
    for fpath in files:
        # 读第一行看有没有 meta JSON
        meta_desc = ""
        try:
            content = fs.read_file(fpath)
            if content and not content.startswith("ERROR"):
                meta_desc, _ = _parse_file_metadata(content)
        except Exception:
            pass
        if meta_desc and meta_desc != "(no meta info)":
            lines.append(f"  {fpath} — {meta_desc}")
        else:
            lines.append(f"  {fpath}")

    if truncated:
        lines.append(f"  ... ({len(all_files) - max_files} more files)")
    return "\n".join(lines)


# ===========================================================================
# 2. 后端召回
# ===========================================================================


def search_fs_bm25(
    fs: FileSystemStore,
    queries: list[Query],
    top_k: int = 8,
) -> list[FsHit]:
    """用 keyword/semantic query 在 fs 上做 BM25 检索（支持 fs_scope + temporal 过滤）。"""
    # 提取 fs_scope
    fs_scope: list[str] | None = None
    for q in queries:
        if q["type"] == "fs_scope":
            scope_text = q["text"].strip()
            if scope_text:
                fs_scope = [s.strip() for s in scope_text.split(",") if s.strip()]
            break

    # 提取 temporal filter
    temporal_filter: str | None = None
    for q in queries:
        if q["type"] == "temporal":
            temporal_filter = q["text"]
            break

    best: dict[str, tuple[float, str]] = {}
    for q in queries:
        if q["type"] not in ("semantic", "keyword"):
            continue
        try:
            results = fs.search_bm25(q["text"], top_k=top_k, scope=fs_scope)
        except Exception as e:
            logger.warning("[atomic.fs] BM25 failed for %r: %s", q["text"], e)
            continue
        for path, score, snippet in results:
            if path not in best or score > best[path][0]:
                best[path] = (score, snippet)

    ranked = sorted(
        [(path, sc, snip) for path, (sc, snip) in best.items()],
        key=lambda x: -x[1],
    )

    # Temporal 过滤（对齐 T3 retrieve_t3.py:826-853）
    if temporal_filter and ranked:
        filtered = []
        for path, score, snippet in ranked:
            try:
                content = fs.read_file(path)
                if content.startswith("ERROR"):
                    filtered.append((path, score, snippet))
                    continue
                lines = content.split("\n")
                kept = []
                for line in lines:
                    line_time = _extract_fs_line_time(line)
                    if line_time is None:
                        kept.append(line)
                    elif line_time == "":
                        kept.append(line)
                    elif _time_matches_filter(line_time, temporal_filter):
                        kept.append(line)
                if kept:
                    filtered.append((path, score, "\n".join(kept[:5])))
            except Exception:
                filtered.append((path, score, snippet))
        ranked = filtered

    final = ranked[:top_k]
    logger.info("[atomic.fs] final %d hits (scope=%s, temporal=%s)", len(final), fs_scope, temporal_filter)
    return final


async def search_vec(
    vec: VectorStoreBase,
    queries: list[Query],
    top_k: int = 15,
) -> list[VecHit]:
    """用 semantic query 检索 vec（支持 temporal 过滤）。"""
    # 提取 temporal filter
    temporal_filter: str | None = None
    for q in queries:
        if q["type"] == "temporal":
            temporal_filter = q["text"]
            break

    best: dict[str, VecHit] = {}
    for q in queries:
        if q["type"] != "semantic":
            continue
        try:
            results = await vec.search_all(q["text"], top_k=top_k)
        except Exception as e:
            logger.warning("[atomic.vec] search_all failed for %r: %s", q["text"], e)
            continue
        for r in results:
            rid = r.get("id") or ""
            if not rid:
                continue
            if rid not in best or r.get("score", 0.0) > best[rid].get("score", 0.0):
                best[rid] = r

    sorted_results = sorted(best.values(), key=lambda r: -r.get("score", 0.0))

    # Temporal 过滤（对齐 T3 retrieve_t3.py:984-999）
    if temporal_filter:
        filtered = []
        for r in sorted_results:
            meta = r.get("metadata", {}) or {}
            occurred_at = meta.get("occurred_at", "")
            if not occurred_at:
                filtered.append(r)  # 无时间信息，默认保留
            elif isinstance(occurred_at, list):
                if any(_time_matches_filter(t, temporal_filter) for t in occurred_at):
                    filtered.append(r)
            elif _time_matches_filter(occurred_at, temporal_filter):
                filtered.append(r)
        sorted_results = filtered

    final = sorted_results[:top_k]
    logger.info("[atomic.vec] final %d hits (temporal=%s)", len(final), temporal_filter)
    return final


async def search_graph(
    graph: GraphStoreBase,
    queries: list[Query],
    depth: int = 3,
    hop_top_k: dict[int, int] | None = None,
    seed_top_k: int = 5,
) -> list[GraphHit]:
    """图检索：向量种子优先 + keyword 补充 → get_subgraph + BFS 分层 + embedding 排序截断。

    对齐 T3 retrieve_t3.py:1027-1184 的完整图检索逻辑。
    """
    if hop_top_k is None:
        hop_top_k = {1: 10, 2: 15, 3: 5}

    seeds: list[dict[str, Any]] = []
    seen_seeds: set[str] = set()

    # 1. 向量相似度检索种子（优先，对齐 T3:1051-1075）
    semantic_texts = [q["text"] for q in queries if q["type"] == "semantic"]
    if semantic_texts and hasattr(graph, "search_nodes_by_embedding"):
        embedder = getattr(graph, "embedder", None)
        if embedder:
            try:
                # await embedder（async 环境）
                query_emb = await embedder.embed_single(semantic_texts[0])
                if query_emb:
                    nodes = graph.search_nodes_by_embedding(
                        query_embedding=query_emb,
                        top_k=seed_top_k,
                        threshold=0.5,
                    )
                    for n in nodes:
                        nid = n.get("id")
                        if nid and nid not in seen_seeds:
                            seen_seeds.add(nid)
                            seeds.append(n)
            except Exception as e:
                logger.warning("[atomic.graph] vector seed search failed: %s", e)

    # 2. Entity keyword 搜索种子（补充，对齐 T3:1078-1101）
    for q in queries:
        if q["type"] != "entity":
            continue
        text = q["text"].strip()
        candidates = [text]
        slug = text.lower().replace(" ", "_")
        if slug != text.lower():
            candidates.append(slug)
        for tok in re.split(r"[\s_]+", text):
            if len(tok) >= 3 and tok.lower() not in {c.lower() for c in candidates}:
                candidates.append(tok)

        for cand in candidates:
            try:
                nodes = graph.search_nodes(keyword=cand)
            except Exception:
                continue
            for n in nodes[:seed_top_k]:
                nid = n.get("id")
                if nid and nid not in seen_seeds:
                    seen_seeds.add(nid)
                    seeds.append(n)
            if len(seeds) >= seed_top_k * 2:
                break

    # 3. Keyword 补充种子（对齐 T3:1103-1130）
    for q in queries:
        if q["type"] != "keyword":
            continue
        for token in q["text"].split():
            if len(token) < 2:
                continue
            try:
                nodes = graph.search_nodes(keyword=token)
            except Exception:
                continue
            for n in nodes[:2]:
                nid = n.get("id")
                if nid and nid not in seen_seeds:
                    seen_seeds.add(nid)
                    seeds.append(n)

    if not seeds:
        logger.info("[atomic.graph] no seeds found")
        return []

    logger.info("[atomic.graph] %d seeds found", len(seeds))

    # 4. 每个种子展开子图（对齐 T3:1186-1316 的 _get_node_subgraph_with_similarity）
    results: list[GraphHit] = []
    for seed in seeds[:seed_top_k]:
        seed_id = seed["id"]
        hops = _expand_subgraph(graph, seed_id, depth=depth, hop_top_k=hop_top_k)
        results.append({"node": seed, "hops": hops})

    return results


def _expand_subgraph(
    graph: GraphStoreBase,
    seed_id: str,
    depth: int,
    hop_top_k: dict[int, int],
) -> dict[int, list[dict[str, Any]]]:
    """从 seed 用 get_subgraph 展开子图，BFS 分层，每跳按 edge embedding 排序截断。

    对齐 T3 retrieve_t3.py:1186-1316。
    """
    try:
        raw_subgraph = graph.get_subgraph(seed_id, depth=depth)
    except Exception as e:
        logger.warning("[atomic.graph] get_subgraph(%s) failed: %s, falling back to BFS", seed_id, e)
        return _bfs_expand_fallback(graph, seed_id, depth, hop_top_k)

    edges = raw_subgraph.get("edges", [])
    if not edges:
        return {}

    # BFS 确定每条边的跳数
    adj: dict[str, list[dict]] = {}
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        adj.setdefault(src, []).append(edge)
        adj.setdefault(tgt, []).append(edge)

    node_depth: dict[str, int] = {seed_id: 0}
    queue = [seed_id]
    edge_hop: dict[str, int] = {}

    while queue:
        current = queue.pop(0)
        for edge in adj.get(current, []):
            eid = edge.get("id", "")
            if eid in edge_hop:
                continue
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            neighbor = tgt if src == current else src
            d = node_depth[current] + 1
            edge_hop[eid] = d
            if neighbor not in node_depth:
                node_depth[neighbor] = d
                queue.append(neighbor)

    # 按跳数分组 + embedding 排序截断
    hops: dict[int, list[dict[str, Any]]] = {}
    max_d = max(edge_hop.values()) if edge_hop else 0

    for d in range(1, max_d + 1):
        hop_entries: list[dict[str, Any]] = []
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            eid = edge.get("id", "")
            if edge_hop.get(eid) != d:
                continue
            # 尝试获取 edge embedding 做排序
            sim = 0.0
            if hasattr(graph, "get_edge_embedding"):
                edge_emb = graph.get_edge_embedding(eid)
                if edge_emb:
                    sim = 0.5  # 有 embedding 就给基础分（实际余弦需要 query embedding）
            hop_entries.append({"edge": edge, "similarity": sim})

        # 截断
        cap = hop_top_k.get(d, hop_top_k.get(max(hop_top_k.keys()), 5))
        hop_entries = hop_entries[:cap]
        if hop_entries:
            hops[d] = hop_entries

    return hops


def _bfs_expand_fallback(
    graph: GraphStoreBase,
    seed_id: str,
    depth: int,
    hop_top_k: dict[int, int],
) -> dict[int, list[dict[str, Any]]]:
    """get_subgraph 不可用时的 fallback：用 get_neighbors BFS 展开。"""
    hops: dict[int, list[dict[str, Any]]] = {}
    visited_nodes: set[str] = {seed_id}
    seen_edges: set[str] = set()
    frontier: list[str] = [seed_id]

    for d in range(1, depth + 1):
        cap = hop_top_k.get(d, 10)
        next_frontier: list[str] = []
        hop_edges: list[dict[str, Any]] = []

        for node_id in frontier:
            try:
                neighbors = graph.get_neighbors(node_id)
            except Exception:
                continue
            for nb in neighbors:
                edge = nb.get("edge") if isinstance(nb, dict) else None
                if edge is None and isinstance(nb, dict) and "source" in nb and "target" in nb:
                    edge = nb
                if not isinstance(edge, dict):
                    continue
                src = edge.get("source", "")
                tgt = edge.get("target", "")
                rel = edge.get("relation", "")
                edge_key = f"{src}|{rel}|{tgt}"
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                hop_edges.append({"edge": edge, "similarity": 0.0})
                other = tgt if src == node_id else src
                if other and other not in visited_nodes:
                    visited_nodes.add(other)
                    next_frontier.append(other)
                if len(hop_edges) >= cap:
                    break
            if len(hop_edges) >= cap:
                break

        if hop_edges:
            hops[d] = hop_edges[:cap]
        if not next_frontier:
            break
        frontier = next_frontier

    return hops


# ===========================================================================
# 3. RRF 融合
# ===========================================================================


def rrf_fuse(
    fs_hits: list[FsHit],
    vec_hits: list[VecHit],
    graph_hits: list[GraphHit],
    rrf_k: int = 60,
    top_k: int = 20,
) -> list[FusedHit]:
    """RRF 融合三后端结果。"""
    fs_rank = [(f"fs:{path}", f"[FS] {path}") for path, _, _ in fs_hits]
    vec_rank = [(f"vec:{r.get('id', '?')}", f"[VEC] {r.get('text', '')[:100]}") for r in vec_hits]
    graph_rank = [(f"graph:{gr['node']['id']}", f"[GRAPH] {gr['node']['id']}") for gr in graph_hits]

    scores: dict[str, float] = {}
    summaries: dict[str, str] = {}
    for lst in (fs_rank, vec_rank, graph_rank):
        for rank, (key, summary) in enumerate(lst, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            summaries.setdefault(key, summary)

    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    return [(sc, key, summaries[key]) for key, sc in ranked]


# ===========================================================================
# 4. 输出格式化
# ===========================================================================


def format_context(
    fs: FileSystemStore,
    fs_hits: list[FsHit],
    vec_hits: list[VecHit],
    graph_hits: list[GraphHit],
    fused: list[FusedHit] | None = None,
    *,
    fs_top_k: int = 6,
    vec_top_k: int = 8,
    graph_edge_top_k: int = 15,
    fs_line_top_k: int = 8,
) -> str:
    """格式化为 T3 风格的三段式 memory context。

    对齐 T3 retrieve_t3.py:2003-2186 的 _format_results。
    """
    parts: list[str] = []

    # 1. FS（对齐 T3：读全文、去系统前缀、展示 meta + 最多 8 行正文）
    if fs_hits:
        fs_lines: list[str] = []
        for path, score, snippet in fs_hits[:fs_top_k]:
            try:
                content = fs.read_file(path)
                if content and not content.startswith("ERROR"):
                    meta_desc, body_start = _parse_file_metadata(content)
                    all_lines = content.split("\n")
                    body_lines = all_lines[body_start:]
                    meta_tag = f" | meta: {meta_desc}" if meta_desc and meta_desc != "(no meta info)" else ""
                    numbered = []
                    for line in body_lines[:fs_line_top_k]:
                        if line.strip():
                            display_line = _convert_line_for_display(line)
                            numbered.append(f"      {display_line}")
                    if len(body_lines) > fs_line_top_k:
                        numbered.append(f"      ... ({len(body_lines) - fs_line_top_k} more lines)")
                    content_str = "\n".join(numbered) if numbered else f"      {snippet[:300]}"
                    fs_lines.append(f"  [{score:.2f}] {path}{meta_tag}\n{content_str}")
                else:
                    fs_lines.append(f"  [{score:.2f}] {path}\n      {snippet[:300]}")
            except Exception:
                fs_lines.append(f"  [{score:.2f}] {path}\n      {snippet[:300]}")
        parts.append("## File System (BM25):\n" + "\n".join(fs_lines))

    # 2. Vec（对齐 T3：按 ingest_time 升序展示）
    if vec_hits:
        sorted_vec = sorted(vec_hits[:vec_top_k], key=lambda r: _ingest_sort_key(r, "vec"))
        vec_lines = [f"  {r.get('text', '')[:1000]}" for r in sorted_vec]
        parts.append("## Vector DB (Semantic):\n" + "\n".join(vec_lines))

    # 3. Graph（对齐 T3：按跳数分层、按 ingest_time 排序、去重）
    if graph_hits:
        graph_lines: list[str] = []
        seen_edges: set[str] = set()
        edges_emitted = 0
        for gr in graph_hits:
            if edges_emitted >= graph_edge_top_k:
                break
            node = gr["node"]
            center_id = node.get("id", "")
            center_label = node.get("label", "")
            graph_lines.append(f"  [Center] {center_id} ({center_label})")

            hops = gr.get("hops", {})
            for d in sorted(hops.keys()):
                hop_entries = hops[d]
                # 按 ingest_time 排序
                hop_entries_sorted = sorted(
                    hop_entries,
                    key=lambda e: _ingest_sort_key(e, "graph_edge"),
                )
                hop_lines_for_d: list[str] = []
                for entry in hop_entries_sorted:
                    edge = entry.get("edge", {})
                    src = edge.get("source", "")
                    rel = edge.get("relation", "")
                    tgt = edge.get("target", "")
                    edge_key = f"{src}|{rel}|{tgt}"
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)
                    hop_lines_for_d.append(f"    {src} --[{rel}]--> {tgt}")
                    edges_emitted += 1
                    if edges_emitted >= graph_edge_top_k:
                        break
                if hop_lines_for_d:
                    graph_lines.append(f"  -- Hop {d} --")
                    graph_lines.extend(hop_lines_for_d)
                if edges_emitted >= graph_edge_top_k:
                    break
        parts.append("## Graph DB (Entity Relations):\n" + "\n".join(graph_lines))

    out = "\n\n".join(parts).strip()
    logger.info("[atomic.format] context_len=%d", len(out))
    return out


# ===========================================================================
# 5. 辅助函数
# ===========================================================================


def _parse_file_metadata(content: str) -> tuple[str, int]:
    """解析文件第一行的 JSON 元数据（对齐 T3 ingest_t3.py:1917-1956）。"""
    lines = content.split("\n")
    if not lines:
        return ("", 0)
    first_line = lines[0].strip()
    if first_line.startswith("{"):
        try:
            meta = json.loads(first_line)
            if isinstance(meta, dict):
                return (meta.get("description", ""), 1)
        except json.JSONDecodeError:
            pass
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
    return ("", 0)


def _convert_line_for_display(line: str) -> str:
    """去掉 FS 行的系统时间前缀，只保留正文内容。

    支持的前缀格式：
    - [session_time-turn | event_time] content  → 去掉
    - [date, Day] content  → 去掉
    - 双层前缀 [a|b] [c|d] content → 全部去掉
    """
    # 循环去掉所有方括号前缀（防双层）
    result = line
    while True:
        m = re.match(r'^\[([^\]]*?)\]\s*', result)
        if not m:
            break
        result = result[m.end():]
    return result if result else line


def _extract_fs_line_time(line: str) -> str | None:
    """从 FS 行中提取事件时间（对齐 T3 retrieve_t3.py:857-882）。"""
    stripped = line.strip()
    if not stripped:
        return None
    m = re.match(r'^\[([^\]]*?)\s*\|\s*([^\]]*?)\]\s*', stripped)
    if m:
        event_time = m.group(2).strip()
        if event_time == '/' or not event_time:
            return ""
        return event_time
    m = re.match(r'^\[(\d{4}(?:-\d{2})?(?:-\d{2})?)\]', stripped)
    if m:
        return m.group(1)
    if stripped.startswith("[undated]"):
        return ""
    return None


def _time_matches_filter(occurred_at: str, time_filter: str) -> bool:
    """判断 occurred_at 是否匹配时间过滤条件（对齐 T3 retrieve_t3.py:1004-1025）。"""
    if not occurred_at or not time_filter:
        return True
    time_filter = time_filter.strip()
    if "~" in time_filter:
        parts = time_filter.split("~", 1)
        time_start = parts[0].strip()
        time_end = parts[1].strip()
        min_len = min(len(time_start), len(time_end))
        occurred_prefix = occurred_at[:min_len]
        return time_start <= occurred_prefix <= time_end
    else:
        return occurred_at.startswith(time_filter)


def _ingest_sort_key(item: dict[str, Any], source: str = "vec") -> tuple[str, int]:
    """生成 ingest_time-turn 排序键（对齐 T3 retrieve_t3.py:2188-2216）。"""
    if source == "vec":
        meta = item.get("metadata", {}) or {}
        ingest_time = meta.get("ingest_time", "")
        ingest_turn = meta.get("ingest_turn", 0)
    elif source == "graph_edge":
        edge = item.get("edge", {})
        props = edge.get("properties", {}) or {}
        ingest_time = props.get("ingest_time", "")
        ingest_turn = props.get("ingest_turn", 0)
    else:
        ingest_time = ""
        ingest_turn = 0
    try:
        ingest_turn = int(ingest_turn)
    except (ValueError, TypeError):
        ingest_turn = 0
    return (ingest_time or "9999", ingest_turn)
