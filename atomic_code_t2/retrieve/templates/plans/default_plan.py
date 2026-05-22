"""Atomic_Code_T2 默认检索方案（对齐 T3 retrieve_t3.py 的参数和流程）。

工作流：
1. query 改写（LLM）→ semantic / keyword / entity / temporal / fs_scope + graph_config
2. 三后端并行召回（fs BM25 / vec semantic / graph 向量+keyword）
3. RRF 融合（当前仅用于日志，不直接进 context）
4. T3 风格三段式格式化输出（FS / Vec / Graph）
"""

from __future__ import annotations

import asyncio
from typing import Any

import logger.logger as logger

from atomic_code_t2.retrieve.base_plan import BasePlan

from ..atomic import (
    format_context,
    rewrite_queries_llm,
    rrf_fuse,
    search_fs_bm25,
    search_graph,
    search_vec,
)


class DefaultPlan(BasePlan):
    """T3 风格默认检索方案（参数对齐 T3 retrieve_t3.py）。"""

    plan_name = "default"

    # 召回配额（对齐 T3 类属性）
    fs_top_k: int = 8
    vec_top_k: int = 15
    graph_seed_top_k: int = 5

    # 输出配额
    output_fs_top_k: int = 6
    output_vec_top_k: int = 8
    output_graph_edge_top_k: int = 15
    output_fs_line_top_k: int = 8  # T3 展示最多 8 行正文

    # RRF
    rrf_k: int = 60
    rrf_top_k: int = 20

    async def run(
        self,
        query: str,
        session_time: str = "",
        **kwargs: Any,
    ) -> str:
        if not query or not query.strip():
            return ""

        # Step 1: query 改写（完整版 prompt，含 temporal/fs_scope）
        queries, graph_config = await rewrite_queries_llm(
            self.llm, self.fs, query, session_time=session_time
        )
        depth = int(graph_config.get("depth", 3))
        depth = max(depth, 3)  # 最小 3 跳（对齐 T3）
        hop_top_k = graph_config.get("hop_top_k") or {1: 10, 2: 15, 3: 5}

        logger.info(
            "[retrieve/default] queries=%d (types=%s) graph depth=%d hop_top_k=%s",
            len(queries),
            sorted({q["type"] for q in queries}),
            depth,
            hop_top_k,
        )

        # Step 2: 三后端并行召回
        fs_task = asyncio.to_thread(
            search_fs_bm25, self.fs, queries, self.fs_top_k
        )
        vec_task = search_vec(self.vec, queries, self.vec_top_k)
        graph_task = search_graph(
            self.graph,
            queries,
            depth,
            hop_top_k,
            self.graph_seed_top_k,
        )

        fs_hits, vec_hits, graph_hits = await asyncio.gather(
            fs_task, vec_task, graph_task
        )

        logger.info(
            "[retrieve/default] recall: fs=%d vec=%d graph=%d",
            len(fs_hits), len(vec_hits), len(graph_hits),
        )

        # Step 3: RRF 融合（仅日志/调试）
        fused = rrf_fuse(
            fs_hits, vec_hits, graph_hits,
            rrf_k=self.rrf_k, top_k=self.rrf_top_k,
        )

        # Step 4: 格式化输出（T3 风格）
        context = format_context(
            self.fs,
            fs_hits, vec_hits, graph_hits,
            fused=fused,
            fs_top_k=self.output_fs_top_k,
            vec_top_k=self.output_vec_top_k,
            graph_edge_top_k=self.output_graph_edge_top_k,
            fs_line_top_k=self.output_fs_line_top_k,
        )
        return context
