"""预置消费子类 — T3 双模式检索（工程化 query 改写 / LLM query 改写）。

特点：
- 双模式 Query 改写（工程化 jieba 分词 / LLM 改写）
- 三后端并行检索（FS BM25 with scope / Vec 语义 with batch embed / Graph 向量+关键词+时序）
- Graph 多跳子图召回（动态 depth + 分跳 top_k + 边 embedding 排序）
- Cross-Encoder 精排（可选，带 embedding fallback + 多跳保护配额）
- RRF 融合排序（fallback）

本文件是代码仓库中的预置子类，与演进阶段 LLM 生成的子类共同供
RetrieveContextMultiCodeTask 选择使用。
"""

from __future__ import annotations

import json
import logging
import math
import re
from enum import Enum
from typing import Any

from context_task.codegen.base_memory_consumer import BaseMemoryConsumer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 检索模式枚举
# ---------------------------------------------------------------------------

class RetrieveMode(str, Enum):
    """Query 改写模式。"""
    ENGINEERING = "engineering"  # 工程化改写（jieba 分词 + 规则）
    LLM = "llm"  # LLM 改写（一次调用生成多 query）


# ---------------------------------------------------------------------------
# LLM Query 改写 Prompt
# ---------------------------------------------------------------------------

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
- **depth**: How many hops to traverse in the knowledge graph. Minimum is 3. Use higher values (4-6) for multi-hop reasoning questions.
- **hop_top_k**: A mapping from hop number (as string key) to the maximum number of edges to keep at that hop.

Guidelines for setting graph_config:
- Simple factual query → depth=3, hop_top_k={"1": 10, "2": 10, "3": 5}
- Multi-entity query → depth=3, hop_top_k={"1": 12, "2": 15, "3": 8}
- Multi-hop reasoning → depth=5, hop_top_k={"1": 8, "2": 12, "3": 15, "4": 10, "5": 5}
- Temporal chain → depth=4, hop_top_k={"1": 10, "2": 15, "3": 10, "4": 5}

## Rewriting Rules

1. **semantic query**: Keep natural language form, suitable for vector similarity matching. Generate 2-3 from different angles.
2. **keyword query**: Extract the most important nouns, verbs, and proper nouns. Remove ALL stop words. Space-separated.
3. **entity query**: Extract ALL identifiable entities: person names, place names, activity names, topic names, etc. One entity per query entry.
4. **temporal query**: If the question mentions or implies a specific time period, extract it. Convert relative time expressions to absolute dates.
5. **fs_scope query**: Based on the question's topic, infer which directories/files are most likely relevant.

## Memory Filesystem Structure

{fs_structure}

## Notes

- Extract retrieval intent from the last few turns of conversation
- Output pure JSON, no markdown code blocks
- IMPORTANT: Do NOT output only semantic queries. You MUST include keyword and entity types.
- IMPORTANT: Always include graph_config in your output.
"""

QUERY_REWRITE_USER_TEMPLATE = """\
## Current Time: {current_time}

## Conversation context to retrieve for:

{conversation}

---

Rewrite into multiple queries suitable for retrieving from the memory store. Output pure JSON.
"""


# ---------------------------------------------------------------------------
# PresetRetrieveT3Consumer
# ---------------------------------------------------------------------------


class PresetRetrieveT3Consumer(BaseMemoryConsumer):
    """预置的 T3 检索消费子类 — 双模式检索。

    支持两种 query 改写模式：
    - ENGINEERING：纯工程化（jieba 分词 + 停用词过滤 + 同义词扩展）
    - LLM：一次 LLM 调用生成多个检索 query

    检索流程：
    1. Query 改写 → 多个 query
    2. 并行检索 fs/vec/graph
    3. Cross-Encoder 精排（可选）或 RRF 融合
    4. 格式化输出
    """

    # 配置参数
    retrieve_mode: RetrieveMode = RetrieveMode.ENGINEERING
    # Graph 检索参数
    graph_similarity_threshold: float = 0.5
    graph_subgraph_depth: int = 3
    graph_hop_top_k: dict[int, int] | None = None  # None 使用默认值 {1: 10, 2: 15, 3: 5}
    graph_edge_min_similarity: float = 0.1
    # 检索 top_k
    fs_top_k: int = 8
    vec_top_k: int = 15
    graph_top_k: int = 3
    # RRF 参数
    rrf_k: int = 60
    # Reranker 精排参数
    use_reranker: bool = False
    reranker_url: str = "http://127.0.0.1:8900"
    rerank_score_threshold: float = -2.0
    rerank_top_k: int = 15
    rerank_dedup_threshold: float = 0.75
    rerank_multihop_min_keep: int = 3
    # 最终输出各后端 top_k
    output_fs_top_k: int = 6
    output_vec_top_k: int = 8
    output_graph_edges_top_k: int = 15

    async def retrieve_memory(
        self,
        query: str,
        messages: list[dict[str, Any]],
        user_id: str,
        session_id: str,
    ) -> str:
        """根据用户查询从记忆库中检索相关记忆。

        核心流程：Query 改写 → 并行检索三后端 → 精排/融合 → 格式化输出。
        """
        if not query:
            return ""

        # 获取当前时间
        from datetime import datetime as _dt
        session_time = _dt.now().strftime("%Y-%m-%d %H:%M:%S, %a")

        # 初始化动态配置
        self._dynamic_graph_depth = 3
        self._dynamic_hop_top_k: dict[int, int] | None = None

        # Step 1: Query 改写
        if self.retrieve_mode == RetrieveMode.LLM:
            queries = await self._rewrite_queries_llm(query, session_time=session_time)
        else:
            queries = self._rewrite_queries_engineering(query)

        logger.info(
            "PresetRetrieveT3Consumer: mode=%s, generated %d queries",
            self.retrieve_mode, len(queries),
        )

        # Step 2: 并行检索三后端
        fs_results = self._search_fs(queries)
        vec_results = await self._search_vec(queries)
        graph_results = await self._search_graph(queries)

        # Step 3: Cross-Encoder 精排（优先）或 RRF 融合（fallback）
        reranked_results = await self._rerank_and_filter(
            query, fs_results, vec_results, graph_results
        )

        if reranked_results is not None:
            context = self._format_results_from_reranked(
                reranked_results, fs_results, vec_results, graph_results
            )
        else:
            fused_results = self._rrf_fuse_results(fs_results, vec_results, graph_results)
            context = self._format_results(fused_results, fs_results, vec_results, graph_results)

        return context

    # ------------------------------------------------------------------
    # Query 改写方法 A：工程化
    # ------------------------------------------------------------------

    def _rewrite_queries_engineering(self, question: str) -> list[dict[str, str]]:
        """工程化 query 改写：分词 + 停用词过滤 + 同义词扩展。"""
        queries: list[dict[str, str]] = []

        # 1. 语义 query：直接使用原始问题
        queries.append({"type": "semantic", "text": question})

        # 2. 关键词提取
        keywords = self._extract_keywords_engineering(question)
        if keywords:
            queries.append({"type": "keyword", "text": " ".join(keywords)})
            for kw in keywords[:5]:
                queries.append({"type": "entity", "text": kw})

        # 3. 尝试 jieba 分词
        jieba_keywords = self._jieba_segment(question)
        if jieba_keywords:
            queries.append({"type": "keyword", "text": " ".join(jieba_keywords)})
            for kw in jieba_keywords:
                if kw not in [q["text"] for q in queries if q["type"] == "entity"]:
                    queries.append({"type": "entity", "text": kw})

        # 4. 同义词扩展
        expanded = self._expand_synonyms(question)
        if expanded and expanded != question:
            queries.append({"type": "semantic", "text": expanded})

        # 5. 时间提取
        temporal = self._extract_temporal(question)
        if temporal:
            queries.append({"type": "temporal", "text": temporal})

        # 6. FS 路径范围推断
        fs_scope = self._infer_fs_scope(queries)
        if fs_scope:
            queries.append({"type": "fs_scope", "text": ",".join(fs_scope)})

        return queries

    def _infer_fs_scope(self, queries: list[dict[str, str]]) -> list[str] | None:
        """基于已提取的实体名，推断 FS 搜索路径范围。"""
        try:
            all_files = self.fs.list_files()
        except Exception:
            return None

        if not all_files:
            return None

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
                parts = f_lower.split("/")
                for i, part in enumerate(parts):
                    name = part.rsplit(".", 1)[0] if "." in part else part
                    if entity == name or entity in name:
                        if i == 0:
                            matched_paths.add(f"{parts[0]}/")
                        elif i == 1:
                            matched_paths.add(f"{parts[0]}/{parts[1]}/")
                        break

        if len(matched_paths) > 5:
            return None

        return list(matched_paths) if matched_paths else None

    @staticmethod
    def _extract_temporal(text: str) -> str | None:
        """从文本中提取时间表达式。"""
        # 范围模式
        range_patterns = [
            r'(?:from|between)\s+(\d{4}(?:-\d{2})?(?:-\d{2})?)\s+(?:to|and)\s+(\d{4}(?:-\d{2})?(?:-\d{2})?)',
            r'(\d{4}(?:-\d{2})?(?:-\d{2})?)\s*[到至~]\s*(\d{4}(?:-\d{2})?(?:-\d{2})?)',
        ]
        for pattern in range_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return f"{m.group(1)}~{m.group(2)}"

        # 单值模式
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
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                groups = m.groups()
                if len(groups) == 2 and groups[1]:
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

        en_words = re.findall(r"\b[a-zA-Z']+\b", text)
        en_keywords = [w for w in en_words if w.lower() not in stop_words_en and len(w) > 2]

        cjk_runs = re.findall(r"[\u4e00-\u9fff]{2,}", text)
        cjk_keywords = [w for w in cjk_runs if w not in stop_words_zh]

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
            keywords = jieba.analyse.extract_tags(text, topK=10, withWeight=False)
            return [kw for kw in keywords if len(kw) > 1]
        except ImportError:
            return []

    @staticmethod
    def _expand_synonyms(text: str) -> str:
        """简单的同义词扩展规则。"""
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
                break

        return " ".join(expanded_parts) if len(expanded_parts) > 1 else text

    # ------------------------------------------------------------------
    # Query 改写方法 B：LLM
    # ------------------------------------------------------------------

    async def _rewrite_queries_llm(
        self, question: str, session_time: str = ""
    ) -> list[dict[str, str]]:
        """使用 LLM 改写 query。一次调用生成多个检索 query。"""
        user_prompt = QUERY_REWRITE_USER_TEMPLATE.format(
            conversation=question, current_time=session_time
        )

        # 获取 FS 结构摘要
        fs_structure = self._build_fs_structure_with_meta()
        system_prompt = QUERY_REWRITE_SYSTEM_PROMPT.replace("{fs_structure}", fs_structure)

        try:
            response = await self.llm.generate(
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                tools=None,
            )

            content = response.content or ""
            queries = self._parse_query_rewrite_output(content)
            if queries:
                queries = self._ensure_keyword_entity_queries(queries, question)
                return queries
        except Exception as e:
            logger.error("LLM query rewrite failed: %s", e)

        # Fallback 到工程化方法
        logger.warning("LLM query rewrite failed, falling back to engineering mode")
        return self._rewrite_queries_engineering(question)

    def _ensure_keyword_entity_queries(
        self, queries: list[dict[str, str]], question: str
    ) -> list[dict[str, str]]:
        """确保 queries 中包含 keyword 和 entity 类型。"""
        has_keyword = any(q.get("type") == "keyword" for q in queries)
        has_entity = any(q.get("type") == "entity" for q in queries)

        if has_keyword and has_entity:
            return queries

        keywords = self._extract_keywords_engineering(question)

        if not has_keyword and keywords:
            queries.append({"type": "keyword", "text": " ".join(keywords[:10])})

        if not has_entity and keywords:
            proper_nouns = [kw for kw in keywords if kw[0].isupper()]
            entities = proper_nouns[:3] if proper_nouns else keywords[:3]
            for entity in entities:
                queries.append({"type": "entity", "text": entity})

        return queries

    def _build_fs_structure_with_meta(self) -> str:
        """构建带 meta 描述信息的文件系统结构摘要。"""
        try:
            all_files = self.fs.list_files("")
        except Exception:
            try:
                tree = self.fs.tree(max_depth=2)
                return tree if tree else "(not available)"
            except Exception:
                return "(not available)"

        if not all_files:
            return "(empty filesystem)"

        if len(all_files) > 50:
            all_files = all_files[:50]
            truncated = True
        else:
            truncated = False

        dir_files: dict[str, list[tuple[str, str]]] = {}
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
        if truncated:
            result += "\n  ... (truncated, showing first 50 files)"
        return result

    def _parse_query_rewrite_output(self, content: str) -> list[dict[str, str]] | None:
        """解析 LLM query 改写输出，同时提取 graph_config。"""
        if not content:
            return None

        def _extract_graph_config(data: dict) -> None:
            gc = data.get("graph_config")
            if isinstance(gc, dict):
                depth = gc.get("depth")
                if isinstance(depth, int) and depth >= 3:
                    self._dynamic_graph_depth = depth
                else:
                    self._dynamic_graph_depth = 3
                hop_top_k = gc.get("hop_top_k")
                if isinstance(hop_top_k, dict):
                    self._dynamic_hop_top_k = {
                        int(k): int(v) for k, v in hop_top_k.items()
                        if str(k).isdigit() and isinstance(v, (int, float))
                    }
                else:
                    self._dynamic_hop_top_k = None
            else:
                self._dynamic_graph_depth = 3
                self._dynamic_hop_top_k = None

        # 尝试直接解析
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "queries" in data:
                _extract_graph_config(data)
                return data["queries"]
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown 代码块中提取
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict) and "queries" in data:
                    _extract_graph_config(data)
                    return data["queries"]
            except json.JSONDecodeError:
                pass

        # 尝试找 JSON 对象
        first_brace = content.find("{")
        last_brace = content.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            try:
                data = json.loads(content[first_brace:last_brace + 1])
                if isinstance(data, dict) and "queries" in data:
                    _extract_graph_config(data)
                    return data["queries"]
            except json.JSONDecodeError:
                pass

        self._dynamic_graph_depth = 3
        self._dynamic_hop_top_k = None
        return None

    # ------------------------------------------------------------------
    # 检索执行
    # ------------------------------------------------------------------

    def _search_fs(self, queries: list[dict[str, str]]) -> list[tuple[str, float, str]]:
        """在文件系统中执行 BM25 检索。支持 fs_scope 和 temporal 过滤。"""
        # 提取时间过滤条件
        temporal_filter: str | None = None
        for q in queries:
            if q["type"] == "temporal":
                temporal_filter = q["text"]
                break

        # 提取 fs_scope
        fs_scope: list[str] | None = None
        for q in queries:
            if q["type"] == "fs_scope":
                scope_text = q["text"].strip()
                if scope_text:
                    fs_scope = [s.strip() for s in scope_text.split(",") if s.strip()]
                break

        all_results: dict[str, tuple[float, str]] = {}

        for q in queries:
            if q["type"] in ("semantic", "keyword"):
                try:
                    results = self.fs.search_bm25(q["text"], top_k=self.fs_top_k, scope=fs_scope)
                    for path, score, snippet in results:
                        if path not in all_results or score > all_results[path][0]:
                            all_results[path] = (score, snippet)
                except Exception as e:
                    logger.warning("FS BM25 search failed for query=%r: %s", q["text"], e)

        sorted_results = sorted(
            [(path, score, snippet) for path, (score, snippet) in all_results.items()],
            key=lambda x: -x[1],
        )

        # 时间过滤
        if temporal_filter:
            filtered = []
            for path, score, snippet in sorted_results:
                try:
                    content = self.fs.read_file(path)
                    if content.startswith("ERROR"):
                        filtered.append((path, score, snippet))
                        continue
                    lines = content.split("\n")
                    kept_lines = []
                    for line in lines:
                        line_time = self._extract_fs_line_time(line)
                        if line_time is None:
                            kept_lines.append(line)
                        elif line_time == "":
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
        """从 FS 行中提取事件时间。"""
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

    async def _search_vec(self, queries: list[dict[str, str]]) -> list[dict[str, Any]]:
        """在向量 DB 中执行语义检索。支持批量 embed 优化和时间过滤。"""
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

        # 一次性批量 embed 所有 query
        query_embeddings: dict[str, list[float]] = {}
        embedder = getattr(self.vec, "embedder", None)
        if embedder:
            try:
                all_embs = await embedder.embed(semantic_texts)
                for text, emb in zip(semantic_texts, all_embs):
                    query_embeddings[text] = emb
            except Exception as e:
                logger.warning("Batch embedding failed, falling back to sequential: %s", e)
                for text in semantic_texts:
                    try:
                        emb = await embedder.embed_single(text)
                        query_embeddings[text] = emb
                    except Exception as e2:
                        logger.warning("embed_single failed for %r: %s", text[:50], e2)

        # 用预计算的 embedding 检索
        all_results: dict[str, dict[str, Any]] = {}

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
            # Fallback：使用 search_all（会内部做 embed）
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

        # 时间过滤
        if temporal_filter:
            filtered = []
            for r in sorted_results:
                meta = r.get("metadata", {}) or {}
                occurred_at = meta.get("occurred_at", "")
                if not occurred_at:
                    filtered.append(r)
                elif isinstance(occurred_at, list):
                    if any(self._time_matches_filter(t, temporal_filter) for t in occurred_at):
                        filtered.append(r)
                elif self._time_matches_filter(occurred_at, temporal_filter):
                    filtered.append(r)
            sorted_results = filtered

        return sorted_results[:self.vec_top_k]

    @staticmethod
    def _time_matches_filter(occurred_at: str, time_filter: str) -> bool:
        """判断 occurred_at 是否匹配时间过滤条件。"""
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

    async def _search_graph(self, queries: list[dict[str, str]]) -> list[dict[str, Any]]:
        """在图 DB 中执行向量增强检索。

        策略：
        1. 用 semantic query 计算查询向量，通过向量相似度找种子节点
        2. 用 entity query 做关键词节点搜索（补充）
        3. 对种子节点，按跳数展开子图，每跳按相似度排序并截断
        4. 时间过滤检索
        """
        all_results: list[dict[str, Any]] = []
        seen_nodes: set[str] = set()

        effective_depth = getattr(self, '_dynamic_graph_depth', None) or self.graph_subgraph_depth
        logger.info(
            "Graph search: depth=%d, hop_top_k=%s",
            effective_depth,
            getattr(self, '_dynamic_hop_top_k', None) or self.graph_hop_top_k or {1: 10, 2: 15, 3: 5},
        )

        # 收集 semantic query 文本
        semantic_texts = [q["text"] for q in queries if q["type"] == "semantic"]
        query_embedding: list[float] | None = None

        # 1. 向量相似度检索种子节点
        graph_embedder = getattr(self.graph, "embedder", None)
        if semantic_texts and graph_embedder:
            try:
                query_embedding = await graph_embedder.embed_single(semantic_texts[0])
                nodes = self.graph.search_nodes_by_embedding(
                    query_embedding=query_embedding,
                    top_k=self.graph_top_k,
                    threshold=self.graph_similarity_threshold,
                )
                for node in nodes:
                    if node["id"] in seen_nodes:
                        continue
                    seen_nodes.add(node["id"])
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

        # 2. 关键词节点搜索
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

        # 3. keyword query 补充搜索
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

        # 4. 时间过滤检索
        for q in queries:
            if q["type"] == "temporal":
                try:
                    time_results = self.graph.search_by_time(time_query=q["text"])
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

        all_results.sort(key=lambda x: x.get("seed_similarity", 0.0), reverse=True)
        return all_results[:self.graph_top_k * 2]

    def _get_node_subgraph_with_similarity(
        self, node_id: str, query_embedding: list[float] | None
    ) -> dict[str, Any]:
        """获取节点的子图，按跳数分层并按相似度排序截断。"""
        effective_depth = getattr(self, '_dynamic_graph_depth', None) or self.graph_subgraph_depth
        try:
            raw_subgraph = self.graph.get_subgraph(node_id, depth=effective_depth)
        except Exception as e:
            logger.warning("get_subgraph failed for %s: %s", node_id, e)
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
        edge_hop: dict[str, int] = {}
        edge_neighbor: dict[str, str] = {}

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

                sim = 0.0
                if query_embedding:
                    edge_emb = self.graph.get_edge_embedding(eid)
                    if edge_emb:
                        sim = self._cosine_sim(query_embedding, edge_emb)
                    elif neighbor_id:
                        neighbor_emb = self.graph.get_node_embedding(neighbor_id)
                        if neighbor_emb:
                            sim = self._cosine_sim(query_embedding, neighbor_emb)

                hop_entries.append({
                    "edge": edge,
                    "neighbor_id": neighbor_id,
                    "similarity": sim,
                })

            if query_embedding:
                hop_entries.sort(key=lambda x: x["similarity"], reverse=True)

            # 按配置的 hop_top_k 截断
            dynamic_htk = getattr(self, '_dynamic_hop_top_k', None)
            effective_hop_top_k = dynamic_htk or self.graph_hop_top_k or {1: 10, 2: 15, 3: 5}
            if d in effective_hop_top_k:
                hop_entries = hop_entries[:effective_hop_top_k[d]]
            else:
                max_configured_hop = max(effective_hop_top_k.keys()) if effective_hop_top_k else 0
                if max_configured_hop > 0:
                    hop_entries = hop_entries[:effective_hop_top_k[max_configured_hop]]

            # 过滤低相似度的边
            if query_embedding:
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

    @staticmethod
    def _cosine_sim(vec_a: list[float], vec_b: list[float]) -> float:
        """计算两个向量的余弦相似度。"""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a)) or 1e-10
        norm_b = math.sqrt(sum(b * b for b in vec_b)) or 1e-10
        return dot / (norm_a * norm_b)

    # ------------------------------------------------------------------
    # Cross-Encoder 精排
    # ------------------------------------------------------------------

    async def _rerank_and_filter(
        self,
        query: str,
        fs_results: list[tuple[str, float, str]],
        vec_results: list[dict[str, Any]],
        graph_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """Cross-Encoder 精排，带 embedding fallback。"""
        if not self.use_reranker:
            return None

        candidates = self._build_rerank_candidates(fs_results, vec_results, graph_results)
        if not candidates:
            return None

        candidates = self._deduplicate_candidates(candidates)

        documents = [c["rerank_text"] for c in candidates]
        rerank_mode = "cross-encoder"
        try:
            rerank_results = await self._call_reranker_service(query, documents)
            for rr in rerank_results:
                idx = rr["index"]
                if 0 <= idx < len(candidates):
                    candidates[idx]["rerank_score"] = rr["score"]
        except Exception as e:
            logger.warning("Reranker 服务调用失败，尝试 embedding fallback: %s", e)
            fallback_ok = await self._embedding_rerank_fallback(query, candidates)
            if not fallback_ok:
                return None
            rerank_mode = "embedding-fallback"

        candidates.sort(key=lambda x: x.get("rerank_score", -999), reverse=True)

        threshold = self.rerank_score_threshold if rerank_mode == "cross-encoder" else 0.3
        filtered = [
            c for c in candidates
            if c.get("rerank_score", -999) >= threshold
        ]
        filtered = filtered[:self.rerank_top_k]

        filtered = self._apply_multihop_protection(filtered, candidates)

        logger.info(
            "Rerank 完成 [%s]: 候选 %d → 去重后 %d → 精排后 %d",
            rerank_mode, len(documents), len(candidates), len(filtered),
        )

        return filtered if filtered else None

    async def _embedding_rerank_fallback(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> bool:
        """Embedding-based rerank fallback。"""
        embedder = getattr(self.graph, "embedder", None) or getattr(self.vec, "embedder", None)
        if not embedder:
            return False

        try:
            documents = [c["rerank_text"] for c in candidates]
            all_texts = [query] + documents
            all_embeddings = await embedder.embed(all_texts)

            if not all_embeddings or len(all_embeddings) < len(all_texts):
                return False

            query_emb = all_embeddings[0]
            doc_embeddings = all_embeddings[1:]

            for i, candidate in enumerate(candidates):
                if i < len(doc_embeddings):
                    sim = self._cosine_sim(query_emb, doc_embeddings[i])
                    candidate["rerank_score"] = sim
                else:
                    candidate["rerank_score"] = 0.0

            return True
        except Exception as e:
            logger.warning("Embedding rerank fallback 失败: %s", e)
            return False

    def _build_rerank_candidates(
        self,
        fs_results: list[tuple[str, float, str]],
        vec_results: list[dict[str, Any]],
        graph_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """将三后端结果统一文本化，构建 rerank 候选列表。"""
        candidates: list[dict[str, Any]] = []

        # FS 结果
        for path, score, snippet in fs_results:
            rerank_text = f"{path}: {snippet[:400]}"
            candidates.append({
                "source": "fs",
                "rerank_text": rerank_text,
                "original_score": score,
                "original": {"path": path, "score": score, "snippet": snippet},
            })

        # Vec 结果
        for r in vec_results:
            text = r.get("text", "")
            if not text:
                continue
            candidates.append({
                "source": "vec",
                "rerank_text": text[:500],
                "original_score": r.get("score", 0.0),
                "original": r,
            })

        # Graph 结果：每条边拆开
        for gr in graph_results:
            subgraph = gr.get("subgraph", {})
            hops = subgraph.get("hops", {})
            seed_sim = gr.get("seed_similarity", 0.0)
            seed_node = gr.get("node", {})

            if hops:
                for d in sorted(hops.keys()):
                    hop_entries = hops[d]
                    hop_decay = {1: 1.0, 2: 0.7, 3: 0.5}.get(d, 0.3)
                    for entry in hop_entries:
                        edge = entry.get("edge", {})
                        edge_text = self._graph_edge_to_text(edge)
                        if not edge_text.strip():
                            continue

                        if d == 1:
                            rerank_text = edge_text
                        else:
                            rerank_text = self._build_path_text(edge, d, hops, seed_node.get("id", ""))

                        edge_sim = entry.get("similarity", 0.0)
                        original_score = seed_sim * hop_decay * max(edge_sim, 0.1)
                        candidates.append({
                            "source": "graph_edge",
                            "rerank_text": rerank_text,
                            "original_score": original_score,
                            "original": {
                                "edge": edge,
                                "node": seed_node,
                                "hop": d,
                                "seed_similarity": seed_sim,
                                "edge_similarity": edge_sim,
                            },
                        })
            else:
                edges = subgraph.get("edges", [])
                for edge in edges:
                    if not isinstance(edge, dict):
                        continue
                    edge_text = self._graph_edge_to_text(edge)
                    if not edge_text.strip():
                        continue
                    candidates.append({
                        "source": "graph_edge",
                        "rerank_text": edge_text,
                        "original_score": seed_sim * 0.5,
                        "original": {
                            "edge": edge,
                            "node": seed_node,
                            "hop": 0,
                            "seed_similarity": seed_sim,
                            "edge_similarity": 0.0,
                        },
                    })

        return candidates

    @staticmethod
    def _graph_edge_to_text(edge: dict) -> str:
        """将 Graph 边转为三元组文本。"""
        src = edge.get("source", "")
        rel = edge.get("relation", "").replace("_", " ")
        tgt = edge.get("target", "")
        return f"{src} {rel} {tgt}"

    def _build_path_text(
        self,
        target_edge: dict,
        target_hop: int,
        hops: dict[int, list[dict[str, Any]]],
        seed_node_id: str,
    ) -> str:
        """为多跳边构建路径级文本。"""
        path_edges: list[str] = []
        target_text = self._graph_edge_to_text(target_edge)
        target_src = target_edge.get("source", "")
        target_tgt = target_edge.get("target", "")

        current_nodes = {target_src, target_tgt}

        for d in range(target_hop - 1, 0, -1):
            hop_entries = hops.get(d, [])
            for entry in hop_entries:
                pred_edge = entry.get("edge", {})
                pred_src = pred_edge.get("source", "")
                pred_tgt = pred_edge.get("target", "")
                if pred_tgt in current_nodes or pred_src in current_nodes:
                    pred_text = self._graph_edge_to_text(pred_edge)
                    path_edges.insert(0, pred_text)
                    current_nodes = {pred_src, pred_tgt}
                    break

        path_edges.append(target_text)
        return "; ".join(path_edges)

    def _apply_multihop_protection(
        self,
        filtered: list[dict[str, Any]],
        all_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """多跳保护配额：确保精排后至少保留 N 条 2+跳的 Graph 边。"""
        min_keep = self.rerank_multihop_min_keep
        if min_keep <= 0:
            return filtered

        multihop_in_filtered = [
            c for c in filtered
            if c.get("source") == "graph_edge"
            and c.get("original", {}).get("hop", 0) >= 2
        ]

        if len(multihop_in_filtered) >= min_keep:
            return filtered

        need = min_keep - len(multihop_in_filtered)
        filtered_ids = {id(c) for c in filtered}
        multihop_candidates = [
            c for c in all_candidates
            if id(c) not in filtered_ids
            and c.get("source") == "graph_edge"
            and c.get("original", {}).get("hop", 0) >= 2
            and c.get("rerank_score", -999) > -999
        ]
        multihop_candidates.sort(key=lambda x: x.get("rerank_score", -999), reverse=True)

        supplemented = multihop_candidates[:need]
        if supplemented:
            filtered = filtered + supplemented

        return filtered

    def _deduplicate_candidates(
        self, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """跨路去重：Jaccard token overlap > threshold 的候选只保留分数最高的。"""
        sorted_candidates = sorted(
            candidates, key=lambda x: x.get("original_score", 0.0), reverse=True
        )

        kept: list[dict[str, Any]] = []
        kept_token_sets: list[set[str]] = []

        for candidate in sorted_candidates:
            tokens = set(candidate["rerank_text"].lower().split())
            if not tokens:
                continue

            is_duplicate = False
            for existing_tokens in kept_token_sets:
                intersection = tokens & existing_tokens
                union = tokens | existing_tokens
                jaccard = len(intersection) / len(union) if union else 0.0
                if jaccard >= self.rerank_dedup_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(candidate)
                kept_token_sets.append(tokens)

        return kept

    async def _call_reranker_service(
        self, query: str, documents: list[str],
        max_retries: int = 3, base_delay: float = 1.0,
    ) -> list[dict[str, Any]]:
        """调用 BGE Reranker 服务进行精排（带指数退避重试）。"""
        import asyncio
        import aiohttp

        last_exception: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=30)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{self.reranker_url}/rerank",
                        json={
                            "query": query,
                            "documents": documents,
                            "top_k": None,
                        },
                    ) as resp:
                        if resp.status == 503:
                            text = await resp.text()
                            raise RuntimeError(f"Reranker 服务过载(503): {text[:200]}")
                        if resp.status != 200:
                            text = await resp.text()
                            raise RuntimeError(f"Reranker 服务返回 {resp.status}: {text[:200]}")
                        data = await resp.json()
                        return data.get("results", [])
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    delay = base_delay * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)

        raise last_exception  # type: ignore[misc]

    # ------------------------------------------------------------------
    # RRF 融合（Fallback）
    # ------------------------------------------------------------------

    def _rrf_fuse_results(
        self,
        fs_results: list[tuple[str, float, str]],
        vec_results: list[dict[str, Any]],
        graph_results: list[dict[str, Any]],
    ) -> list[tuple[float, str, str]]:
        """RRF 融合三后端结果。"""
        fs_rank = [
            (f"fs:{path}", f"[FS] {path} — {snippet[:150]}")
            for path, _, snippet in fs_results
        ]
        vec_rank = [
            (
                f"vec:{r.get('id', '?')}",
                f"[VEC] id={r.get('id', '?')} coll={r.get('collection', '?')} — {r.get('text', '')[:150]}",
            )
            for r in vec_results
        ]
        graph_rank = [
            (
                f"graph:{gr['node']['id']}",
                f"[GRAPH] {gr['node']['id']} ({gr['node'].get('label', '')})",
            )
            for gr in graph_results
        ]

        scores: dict[str, float] = {}
        summaries: dict[str, str] = {}
        for lst in [fs_rank, vec_rank, graph_rank]:
            for rank, (key, summary) in enumerate(lst, start=1):
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
                summaries.setdefault(key, summary)

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:20]
        return [(score, key, summaries[key]) for key, score in ranked]

    # ------------------------------------------------------------------
    # 结果格式化
    # ------------------------------------------------------------------

    def _format_results_from_reranked(
        self,
        reranked: list[dict[str, Any]],
        fs_results: list[tuple[str, float, str]],
        vec_results: list[dict[str, Any]],
        graph_results: list[dict[str, Any]],
    ) -> str:
        """基于 rerank 结果格式化输出。"""
        fs_kept: list[dict[str, Any]] = []
        vec_kept: list[dict[str, Any]] = []
        graph_edges_kept: list[dict[str, Any]] = []

        for item in reranked:
            source = item.get("source", "")
            if source == "fs":
                fs_kept.append(item)
            elif source == "vec":
                vec_kept.append(item)
            elif source == "graph_edge":
                graph_edges_kept.append(item)

        fs_kept = fs_kept[:self.output_fs_top_k]
        vec_kept = vec_kept[:self.output_vec_top_k]
        graph_edges_kept = graph_edges_kept[:self.output_graph_edges_top_k]

        parts: list[str] = []

        # 1. Rerank 总览
        overview_lines = []
        for item in reranked[:self.rerank_top_k]:
            score = item.get("rerank_score", 0.0)
            source = item.get("source", "?")
            text_preview = item.get("rerank_text", "")[:120]
            overview_lines.append(f"  [{score:.2f}] [{source.upper()}] {text_preview}")
        if overview_lines:
            parts.append("## Reranked Top Candidates:\n" + "\n".join(overview_lines))

        # 2. 文件系统详情
        if fs_kept:
            fs_lines = []
            for item in fs_kept:
                orig = item["original"]
                path = orig["path"]
                score = item.get("rerank_score", orig["score"])
                try:
                    full_content = self.fs.read_file(path)
                    if full_content and not full_content.startswith("ERROR"):
                        meta_desc, body_start = self._parse_file_metadata(full_content)
                        all_lines = full_content.split("\n")
                        body_lines = all_lines[body_start:]
                        meta_tag = f" | meta: {meta_desc}" if meta_desc and meta_desc != "(no meta info)" else ""
                        numbered = []
                        for line in body_lines[:8]:
                            if line.strip():
                                display_line = re.sub(r'^\[([^\]]*?)\s*\|\s*([^\]]*?)\]\s*', '', line.strip())
                                numbered.append(f"      {display_line}")
                        if len(body_lines) > 8:
                            numbered.append(f"      ... ({len(body_lines) - 8} more lines)")
                        content_str = "\n".join(numbered) if numbered else f"      {orig['snippet'][:300]}"
                        fs_lines.append(f"  [{score:.2f}] {path}{meta_tag}\n{content_str}")
                    else:
                        fs_lines.append(f"  [{score:.2f}] {path}\n      {orig['snippet'][:300]}")
                except Exception:
                    fs_lines.append(f"  [{score:.2f}] {path}\n      {orig['snippet'][:300]}")
            parts.append("## File System (BM25 + Reranked):\n" + "\n".join(fs_lines))

        # 3. 向量 DB 详情（按 ingest_time 排序）
        if vec_kept:
            vec_kept_sorted = sorted(
                vec_kept,
                key=lambda item: self._ingest_sort_key(item.get("original", {}), "vec"),
            )
            vec_lines = []
            for item in vec_kept_sorted:
                r = item["original"]
                vec_lines.append(f"  {r.get('text', '')[:250]}")
            parts.append("## Vector DB (Semantic + Reranked):\n" + "\n".join(vec_lines))

        # 4. 图 DB 详情
        if graph_edges_kept:
            graph_edges_sorted = sorted(
                graph_edges_kept,
                key=lambda item: self._ingest_sort_key(item.get("original", {}), "graph_edge"),
            )
            graph_lines = []
            seen_edges: set[str] = set()
            for item in graph_edges_sorted:
                orig = item["original"]
                edge = orig.get("edge", {})
                src = edge.get("source", "")
                rel = edge.get("relation", "")
                tgt = edge.get("target", "")
                edge_key = f"{src}|{rel}|{tgt}"
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                hop = orig.get("hop", 0)
                edge_props = edge.get("properties", {})
                hop_tag = f" (hop={hop})" if hop > 0 else ""
                props_tag = self._format_edge_props(edge_props)
                graph_lines.append(f"    {src} --[{rel}]--> {tgt}{props_tag}{hop_tag}")
            if graph_lines:
                parts.append("## Graph DB (Entity Relations, reflects latest state):\n" + "\n".join(graph_lines))

        if not parts:
            return "(No relevant memories found)"

        return "\n\n".join(parts)

    def _format_results(
        self,
        fused: list[tuple[float, str, str]],
        fs_results: list[tuple[str, float, str]],
        vec_results: list[dict[str, Any]],
        graph_results: list[dict[str, Any]],
    ) -> str:
        """格式化检索结果为结构化文本（RRF fallback 模式）。"""
        parts: list[str] = []

        # 文件系统详情
        if fs_results:
            fs_lines = []
            for path, score, snippet in fs_results[:8]:
                try:
                    full_content = self.fs.read_file(path)
                    if full_content and not full_content.startswith("ERROR"):
                        meta_desc, body_start = self._parse_file_metadata(full_content)
                        all_lines = full_content.split("\n")
                        body_lines = all_lines[body_start:]
                        meta_tag = f" | meta: {meta_desc}" if meta_desc and meta_desc != "(no meta info)" else ""
                        numbered = []
                        for line in body_lines[:8]:
                            if line.strip():
                                display_line = re.sub(r'^\[([^\]]*?)\s*\|\s*([^\]]*?)\]\s*', '', line.strip())
                                numbered.append(f"      {display_line}")
                        if len(body_lines) > 8:
                            numbered.append(f"      ... ({len(body_lines) - 8} more lines)")
                        content_str = "\n".join(numbered) if numbered else f"      {snippet[:300]}"
                        fs_lines.append(f"  [{score:.2f}] {path}{meta_tag}\n{content_str}")
                    else:
                        fs_lines.append(f"  [{score:.2f}] {path}\n      {snippet[:300]}")
                except Exception:
                    fs_lines.append(f"  [{score:.2f}] {path}\n      {snippet[:300]}")
            parts.append("## File System (BM25):\n" + "\n".join(fs_lines))

        # 向量 DB 详情（按 ingest_time 排序）
        if vec_results:
            sorted_vec = sorted(vec_results[:10], key=lambda r: self._ingest_sort_key(r, "vec"))
            vec_lines = []
            for r in sorted_vec:
                vec_lines.append(f"  {r.get('text', '')[:250]}")
            parts.append("## Vector DB (Semantic):\n" + "\n".join(vec_lines))

        # 图 DB 详情 — 按跳数分层展示
        if graph_results:
            graph_lines = []
            seen_edges: set[str] = set()
            for gr in graph_results:
                node = gr["node"]
                center_id = node["id"]
                center_label = node.get("label", "")
                props = node.get("properties", {})
                props_str = self._format_edge_props(props)
                temporal_tag = f" [temporal:{gr['temporal_match']}]" if gr.get("temporal_match") else ""
                graph_lines.append(
                    f"  [Center] {center_id} ({center_label}){props_str}{temporal_tag}"
                )

                subgraph = gr.get("subgraph", {})
                hops = subgraph.get("hops", {})

                if hops:
                    for d in sorted(hops.keys()):
                        hop_entries = hops[d]
                        if not hop_entries:
                            continue
                        hop_entries_sorted = sorted(
                            hop_entries,
                            key=lambda e: self._ingest_sort_key(e, "graph_edge"),
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
                            edge_props = edge.get("properties", {})
                            props_tag = self._format_edge_props(edge_props)
                            hop_lines_for_d.append(f"    {src} --[{rel}]--> {tgt}{props_tag}")
                        if hop_lines_for_d:
                            graph_lines.append(f"  -- Hop {d} --")
                            graph_lines.extend(hop_lines_for_d)
                else:
                    edges = subgraph.get("edges", [])
                    if edges:
                        graph_lines.append(f"  -- Edges --")
                        for edge in edges[:10]:
                            if not isinstance(edge, dict):
                                continue
                            src = edge.get("source", "")
                            rel = edge.get("relation", "")
                            tgt = edge.get("target", "")
                            edge_key = f"{src}|{rel}|{tgt}"
                            if edge_key in seen_edges:
                                continue
                            seen_edges.add(edge_key)
                            graph_lines.append(f"    {src} --[{rel}]--> {tgt}")

            if graph_lines:
                parts.append("## Graph DB (Entity Relations, reflects latest state):\n" + "\n".join(graph_lines))

        if not parts:
            return "(No relevant memories found)"

        return "\n\n".join(parts)

    @staticmethod
    def _ingest_sort_key(item: dict[str, Any], source: str = "vec") -> tuple[str, int]:
        """生成 ingest_time-turn 排序键。"""
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

    @staticmethod
    def _format_edge_props(props: dict) -> str:
        """将边/节点的 properties 格式化为紧凑 JSON 字符串。"""
        if not props:
            return ""
        _system_keys = {"ingest_time", "ingest_turn", "occurred_at"}
        display = {}
        for k, v in props.items():
            if k in _system_keys:
                continue
            if v is None or v == "" or v == "/":
                continue
            display[k] = v
        if not display:
            return ""
        return " " + json.dumps(display, ensure_ascii=False, separators=(',', ':'))

    @staticmethod
    def _parse_file_metadata(content: str) -> tuple[str, int]:
        """解析文件第一行的 JSON 元数据。"""
        lines = content.split("\n")
        if not lines:
            return ("", 0)

        first_line = lines[0].strip()

        # 新格式：第一行是 JSON 对象
        if first_line.startswith("{"):
            try:
                meta = json.loads(first_line)
                if isinstance(meta, dict):
                    description = meta.get("description", "")
                    return (description, 1)
            except json.JSONDecodeError:
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

        return ("", 0)
