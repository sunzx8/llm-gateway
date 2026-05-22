# T3 双模式检索策略参考文档

> 本文档从 tmp/retrieve_t3.py 源码中提取，描述其记忆消费（Retrieve）的完整方案。
> 格式遵循 CODEGEN_RETRIEVE_STRATEGY_SYSTEM_PROMPT 中的"结构化伪代码 + 决策表"要求。

---

## Part 1: 检索路由决策表

T3 采用 **双模式 Query 改写 + 三后端并行检索 + RRF/Rerank 融合** 架构。
支持两种 Query 改写模式：工程化（jieba 分词 + 规则）和 LLM（一次调用生成多 query）。

### Query 改写模式

| 模式 | 触发条件 | 改写方法 | 生成的 query 类型 | 降级策略 |
|------|---------|---------|------------------|---------|
| ENGINEERING | `retrieve_mode == "engineering"` | jieba 分词 + 停用词过滤 + 同义词扩展 + 时间提取 + FS路径推断 | semantic, keyword, entity, temporal, fs_scope | — |
| LLM | `retrieve_mode == "llm"` | 一次 LLM 调用，输出 JSON（含 queries + graph_config） | semantic(≥2), keyword(≥1), entity(≥1), temporal(可选), fs_scope(可选) | LLM 失败时 fallback 到 ENGINEERING 模式 |

### 检索路由决策表

| query 类型 | 首选后端 | 检索方法 | 降级策略 | 输出格式 |
|-----------|---------|---------|---------|----------|
| semantic | vec (语义) | `await self.vec.search_all(query, top_k=15)` 或 `search_all_with_embedding(emb, top_k=15)` | 批量 embed 失败时逐个 embed_single | `[{id, text, score, metadata, collection}]` |
| keyword | fs (BM25) | `self.fs.search_bm25(query, top_k=8, scope=fs_scope)` | — | `[(path, score, snippet)]` |
| entity | graph (节点搜索) | `self.graph.search_nodes(keyword=entity)` → `get_subgraph(node_id, depth)` | 向量搜索种子节点补充 | `[{node, subgraph, seed_similarity}]` |
| semantic (graph) | graph (向量) | `self.graph.search_nodes_by_embedding(query_emb, top_k=3, threshold=0.5)` | 关键词搜索补充 | 同上 |
| temporal | fs + vec + graph | fs: 行级时间前缀过滤; vec: metadata.occurred_at 过滤; graph: `search_by_time(time_query)` | 无时间信息的记录默认保留 | 过滤后的子集 |
| fs_scope | fs | 限定 BM25 搜索的路径范围（目录前缀） | 推断不精确(>5路径)时放弃 | 传入 `scope` 参数 |

### 融合排序策略

| 融合方式 | 触发条件 | 方法 | 参数 |
|---------|---------|------|------|
| Cross-Encoder 精排 | `use_reranker=True` 且 reranker 服务可用 | 统一文本化 → Jaccard 去重 → rerank → 阈值截断 | threshold=-2.0, top_k=15 |
| Embedding Rerank (fallback) | Cross-Encoder 不可用 | 批量 embed 候选文本 → 余弦相似度排序 | threshold=0.3, top_k=15 |
| RRF 融合 (fallback) | reranker 完全不可用 | `1/(k+rank)` 加权融合三后端排名 | k=60 |

---

## Part 2: 处理流程伪代码

```python
async def retrieve_memory(self, event: Event) -> TaskResult:
    """T3 检索流程 — Query 改写 → 并行检索 → 融合排序"""
    
    question = event.payload.get("query", "")
    session_time = event.payload.get("session_time", "") or datetime.now().strftime(...)
    
    # ===== Phase 1: Query 改写 =====
    if self.retrieve_mode == RetrieveMode.LLM:
        # LLM 改写：一次调用生成多个检索 query + graph_config
        queries = await self._rewrite_queries_llm(question, session_time)
        # LLM 输出 JSON: {"queries": [...], "graph_config": {"depth": N, "hop_top_k": {...}}}
        # 自动提取 graph_config 设置动态深度和分跳 top_k
        # Fallback 补充：如果 LLM 未生成 keyword/entity，自动从文本中提取
    else:
        # 工程化改写：分词 + 停用词过滤 + 同义词扩展
        queries = self._rewrite_queries_engineering(question)
        # 生成: semantic(原文) + keyword(关键词组合) + entity(每个关键词)
        #       + temporal(时间表达式) + fs_scope(路径推断)
    
    # ===== Phase 2: 并行检索三后端 =====
    fs_results = self._search_fs(queries)
    # - 遍历 semantic/keyword query → fs.search_bm25(query, top_k=8, scope=fs_scope)
    # - 按 path 去重保留最高分
    # - 如有 temporal query：读取文件内容，按行级时间前缀过滤
    
    vec_results = await self._search_vec(queries)
    # - 收集所有 semantic query 文本并去重
    # - 批量 embed（单次 API 调用）→ query_embeddings
    # - 用预计算 embedding 在本地做余弦相似度（内存后端）
    #   或调用 search_all_with_embedding（PG 后端）
    # - 按 id 去重保留最高分
    # - 如有 temporal query：按 metadata.occurred_at 过滤
    
    graph_results = await self._search_graph(queries)
    # - 向量相似度检索种子节点（semantic query → embed → search_nodes_by_embedding）
    # - 关键词节点搜索补充（entity query → search_nodes）
    # - keyword query 也搜 graph（每个词取前2个节点）
    # - temporal query → search_by_time（节点+边端点）
    # - 每个种子节点展开子图：_get_node_subgraph_with_similarity
    #   - BFS 确定每条边的跳数
    #   - 每跳内按边 embedding 与 query 的余弦相似度排序
    #   - 按 hop_top_k 配置截断（LLM 可动态覆盖）
    #   - 过滤低相似度边（< graph_edge_min_similarity）
    
    # ===== Phase 3: 融合排序 =====
    if self.use_reranker:
        reranked = await self._rerank_and_filter(question, fs_results, vec_results, graph_results)
        # Step 1: 统一文本化 → _build_rerank_candidates
        #   - fs: "path: snippet[:400]"
        #   - vec: text[:500]
        #   - graph: 1跳边直接三元组文本; 2+跳边附带完整路径文本
        # Step 2: Jaccard 去重（threshold=0.75）
        # Step 3: Cross-Encoder 精排（带指数退避重试 3 次）
        #   - 失败时 embedding fallback（批量 embed + 余弦相似度）
        # Step 4: 阈值截断 + top_k 限制
        # Step 5: 多跳保护配额（至少保留 N 条 2+跳边）
        
        if reranked:
            context = self._format_results_from_reranked(reranked, ...)
        else:
            # RRF fallback
            fused = self._rrf_fuse_results(fs_results, vec_results, graph_results)
            context = self._format_results(fused, ...)
    else:
        fused = self._rrf_fuse_results(fs_results, vec_results, graph_results)
        context = self._format_results(fused, ...)
    
    # ===== Phase 4: 格式化输出 =====
    # 按后端分组输出：
    # - "## File System (BM25):" — 带元数据描述 + 内容行（去掉系统前缀）
    # - "## Vector DB (Semantic):" — 按 ingest_time-turn 排序
    # - "## Graph DB (Entity Relations):" — 按跳数分层展示三元组
    
    return TaskResult(task_name="retrieve_t3", retrieved_context=context)
```

### 工程化 Query 改写子流程

```python
def _rewrite_queries_engineering(self, question: str) -> list[dict]:
    queries = []
    
    # 1. 语义 query：原始问题
    queries.append({"type": "semantic", "text": question})
    
    # 2. 关键词提取（英文停用词过滤 + CJK 连续段提取）
    keywords = _extract_keywords_engineering(question)  # 最多15个
    queries.append({"type": "keyword", "text": " ".join(keywords)})
    for kw in keywords[:5]:
        queries.append({"type": "entity", "text": kw})
    
    # 3. jieba 分词（TF-IDF 提取 top10 关键词）
    jieba_keywords = jieba.analyse.extract_tags(question, topK=10)
    queries.append({"type": "keyword", "text": " ".join(jieba_keywords)})
    for kw in jieba_keywords:
        queries.append({"type": "entity", "text": kw})  # 去重
    
    # 4. 同义词扩展（规则映射表：like→enjoy prefer love）
    expanded = _expand_synonyms(question)
    if expanded != question:
        queries.append({"type": "semantic", "text": expanded})
    
    # 5. 时间提取（正则匹配：in 2017 / 2018~2020 / 2019年3月）
    temporal = _extract_temporal(question)
    if temporal:
        queries.append({"type": "temporal", "text": temporal})
    
    # 6. FS 路径推断（entity 名匹配文件路径中的目录/文件名）
    fs_scope = _infer_fs_scope(queries)
    if fs_scope:
        queries.append({"type": "fs_scope", "text": ",".join(fs_scope)})
    
    return queries
```

### LLM Query 改写子流程

```python
async def _rewrite_queries_llm(self, question: str, session_time: str) -> list[dict]:
    # 构建 FS 结构摘要（带 meta description）
    fs_structure = _build_fs_structure_with_meta()
    # 注入到 system prompt 的 {fs_structure} 占位符
    
    system_prompt = QUERY_REWRITE_SYSTEM_PROMPT.replace("{fs_structure}", fs_structure)
    user_prompt = QUERY_REWRITE_USER_TEMPLATE.format(
        conversation=question, current_time=session_time
    )
    
    response = await self.llm.generate(system=system_prompt, messages=[...], tools=None)
    
    # 解析 JSON 输出：{"queries": [...], "graph_config": {"depth": N, "hop_top_k": {...}}}
    queries = _parse_query_rewrite_output(response.content)
    # 提取 graph_config → self._dynamic_graph_depth, self._dynamic_hop_top_k
    
    # Fallback 补充：确保有 keyword 和 entity 类型
    queries = _ensure_keyword_entity_queries(queries, question)
    
    return queries
```

### Graph 子图展开子流程

```python
def _get_node_subgraph_with_similarity(self, node_id: str, query_embedding) -> dict:
    """按跳数分层展开子图，每跳按相似度排序截断。"""
    
    effective_depth = self._dynamic_graph_depth or self.graph_subgraph_depth  # 最小3跳
    raw_subgraph = self.graph.get_subgraph(node_id, depth=effective_depth)
    edges = raw_subgraph.get("edges", [])
    
    # BFS 确定每条边的跳数
    node_depth = {node_id: 0}
    queue = [node_id]
    edge_hop = {}  # edge_id → hop_number
    # ... BFS 遍历 ...
    
    # 按跳数分组
    hops = {}
    for d in range(1, max_depth + 1):
        hop_entries = []
        for edge in edges_at_hop_d:
            # 计算相似度：优先边 embedding，fallback 邻居节点 embedding
            sim = cosine_sim(query_embedding, edge_embedding or neighbor_embedding)
            hop_entries.append({"edge": edge, "neighbor_id": ..., "similarity": sim})
        
        # 按相似度降序排列
        hop_entries.sort(key=lambda x: x["similarity"], reverse=True)
        
        # 按 hop_top_k 截断（LLM 动态覆盖 > 类级别配置 > 默认 {1:10, 2:15, 3:5}）
        effective_hop_top_k = self._dynamic_hop_top_k or self.graph_hop_top_k or {1:10, 2:15, 3:5}
        hop_entries = hop_entries[:effective_hop_top_k.get(d, 5)]
        
        # 过滤低相似度边（< graph_edge_min_similarity=0.1）
        hop_entries = [e for e in hop_entries if e["similarity"] >= 0.1 or e["similarity"] == 0.0]
        
        hops[d] = hop_entries
    
    return {"nodes": nodes, "edges": edges, "hops": hops}
```

---

## Part 3: 数据格式兼容性表

| 后端 | 摄入写入格式 | 消费解析方式 | 关键字段 |
|------|------------|------------|----------|
| fs 文件 | 第一行 JSON 元数据 `{"description": "..."}` + 正文行 `[session_time \| event_time] content` | `read_file(path)` → `_parse_file_metadata()` 提取 description + body_start; 正文按行解析，`_extract_fs_line_time()` 提取行级时间 | description, session_time, event_time |
| fs BM25 搜索 | — | `self.fs.search_bm25(query, top_k=8, scope=fs_scope)` → `[(path, score, snippet)]` | path, score, snippet |
| vec metadata | `{"occurred_at": "2023-07-01" 或 ["2023", "2024"], "ingest_time": "ISO", "ingest_turn": int, ...}` | `search_all(query, top_k)` → `[{id, text, score, metadata, collection}]`; 时间过滤: `metadata.get("occurred_at")` | occurred_at, ingest_time, ingest_turn |
| vec embedding | 批量 embed 后存储 | 消费时批量 embed query → `search_all_with_embedding(emb, top_k)` 或本地余弦相似度 | embedding 向量 |
| graph 节点 | `{id, label, properties, embedding}` | `search_nodes(keyword)` / `search_nodes_by_embedding(emb, top_k, threshold)` | id, label, embedding |
| graph 边 | `{id, source, relation, target, properties: {ingest_time, ingest_turn, occurred_at, ...}, embedding}` | `get_subgraph(node_id, depth)` → BFS 分跳; `get_edge_embedding(eid)` 获取边向量 | source, relation, target, properties |
| graph 时间搜索 | 边/节点的 occurred_at 属性 | `search_by_time(time_query)` → `{"nodes": [...], "edges": [...]}` | occurred_at |

### 时间格式兼容

| 时间表达式 | 解析方式 | 匹配规则 |
|-----------|---------|---------|
| `"2017"` | 单值 | `occurred_at.startswith("2017")` |
| `"2019-03"` | 单值 | `occurred_at.startswith("2019-03")` |
| `"2018~2020"` | 范围 | `"2018" <= occurred_at[:4] <= "2020"` |
| `["2023", "2024"]` (数组) | 任一匹配 | `any(time_matches(t, filter) for t in occurred_at)` |
| 空/None | 默认保留 | 无时间信息的记录不被过滤 |

### FS 行级时间格式

| 行格式 | 解析结果 | 过滤行为 |
|--------|---------|---------|
| `[2024-01-15 10:30 \| 2023-07-01] content` | event_time = "2023-07-01" | 按 temporal filter 匹配 |
| `[2024-01-15 10:30 \| /] content` | event_time = "" (无时间) | 默认保留 |
| `# 标题行` (无前缀) | None | 默认保留 |
| `[undated] content` | "" | 默认保留 |

---

## Part 4: 性能优化策略

| 优化点 | 策略 | 实现方式 | 效果 |
|-------|------|----------|------|
| 批量 Embedding | 所有 semantic query 一次性 embed | `embedder.embed(semantic_texts)` 单次 API 调用 | 避免 N 次重复 embed 调用 |
| 预计算 Embedding 复用 | embed 后用本地余弦相似度搜索 | `search_all_with_embedding(emb)` 或内存后端直接计算 | 避免 N×M 次 API 调用 |
| FS 路径范围限定 | fs_scope query 限定搜索目录 | `search_bm25(query, scope=["people/alex/"])` | 减少 BM25 搜索范围 |
| Graph 分跳截断 | 每跳按相似度排序后截断 | `hop_top_k = {1:10, 2:15, 3:5}` | 控制子图展开规模 |
| Graph 动态深度 | LLM 根据 query 复杂度决定深度 | 简单事实=3跳, 多跳推理=5跳 | 按需深挖，避免过度展开 |
| 低相似度过滤 | 边相似度低于阈值不展示 | `graph_edge_min_similarity=0.1` | 减少噪声边 |
| Jaccard 去重 | rerank 前跨路去重 | `jaccard(tokens_a, tokens_b) >= 0.75` 则去重 | 避免重复候选浪费 rerank 配额 |
| Reranker 指数退避 | 服务过载时自动重试 | 3 次重试，延迟 1s→2s→4s | 提高 reranker 调用成功率 |
| Embedding Rerank Fallback | Cross-Encoder 不可用时降级 | 批量 embed 候选 + 余弦相似度 | 保留路径级语义匹配能力 |
| 多跳保护配额 | 精排后保留最少 N 条 2+跳边 | `rerank_multihop_min_keep=3` | 防止多跳信息被全部排低 |
| 输出 top_k 兜底 | 各后端最终输出有独立限制 | `output_fs_top_k=6, output_vec_top_k=8, output_graph_edges_top_k=15` | 防止单后端输出过长 |
| 时间过滤保守策略 | 无时间信息的记录默认保留 | `if not occurred_at: keep` | 避免误删无时间标记的有效记忆 |

### 关键参数配置表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `fs_top_k` | 8 | FS BM25 检索返回数 |
| `vec_top_k` | 15 | 向量检索返回数 |
| `graph_top_k` | 3 | Graph 种子节点数 |
| `graph_similarity_threshold` | 0.5 | 向量搜索种子节点的最低相似度 |
| `graph_subgraph_depth` | 3 | 子图展开深度（最小值） |
| `graph_edge_min_similarity` | 0.1 | 边的最低相似度阈值 |
| `rrf_k` | 60 | RRF 融合参数 |
| `use_reranker` | False | 是否启用 Cross-Encoder 精排 |
| `reranker_url` | `http://127.0.0.1:8900` | BGE Reranker 服务地址 |
| `rerank_score_threshold` | -2.0 | Cross-Encoder 分数阈值 |
| `rerank_top_k` | 15 | 精排后保留的最大条目数 |
| `rerank_dedup_threshold` | 0.75 | Jaccard 去重阈值 |
| `rerank_multihop_min_keep` | 3 | 多跳保护配额 |

---

## Part 5: 证据强度判断规则

| 强度 | 判断条件 | 对应场景 | 说明 |
|------|---------|---------|------|
| strong | 多后端交叉命中 + rerank 高分 | Cross-Encoder score > 0 或 embedding sim > 0.7 | 三路信号（fs+vec+graph）均有支持 |
| medium | 双后端命中 或 单后端高分 | vec score > 0.7 或 graph seed_similarity > 0.7 | 两路信号支持 |
| weak | 仅单后端低分命中 | vec score 0.3~0.7 或 graph 3+跳边 | 间接关联，需谨慎使用 |
| filtered | 低于阈值 | vec score < threshold 或 rerank score < -2.0 | 直接排除 |

### Graph 证据的跳数衰减

| 跳数 | 衰减因子 | 说明 |
|------|---------|------|
| 1 跳 | 1.0 | 直接关联，证据最强 |
| 2 跳 | 0.7 | 间接关联（如朋友的偏好） |
| 3 跳 | 0.5 | 远距关联 |
| 4+ 跳 | 0.3 | 弱关联，需多跳保护配额兜底 |

### Rerank 候选文本构建规则

| 来源 | 文本化方式 | 示例 |
|------|-----------|------|
| fs | `"{path}: {snippet[:400]}"` | `"people/alex/preferences.md: Alex likes jazz music..."` |
| vec | `text[:500]` | `"Alex mentioned enjoying jazz concerts in 2023"` |
| graph 1跳边 | 三元组文本 `"{src} {rel} {tgt}"` | `"alex likes jazz"` |
| graph 2+跳边 | 完整路径文本（回溯前驱边） | `"alex is_friend_of bob; bob likes jazz"` |

### 多跳路径回溯算法

```python
def _build_path_text(target_edge, target_hop, hops, seed_node_id) -> str:
    """为多跳边构建路径级文本：从种子节点到该边的完整路径。"""
    path_edges = []
    current_nodes = {target_src, target_tgt}
    
    # 从 target_hop-1 到 1，贪心找前驱边
    for d in range(target_hop - 1, 0, -1):
        for entry in hops[d]:
            pred_edge = entry["edge"]
            if pred_edge.target in current_nodes or pred_edge.source in current_nodes:
                path_edges.insert(0, edge_to_text(pred_edge))
                current_nodes = {pred_edge.source, pred_edge.target}
                break
    
    path_edges.append(edge_to_text(target_edge))
    return "; ".join(path_edges)
    # 输出示例: "alex is_friend_of bob; bob likes jazz; jazz originated_in new_orleans"
```

---

## 附录：完整的检索架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Query Input                                     │
│                   "What music does Alex's friend like?"                   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
         ┌──────────────────┐     ┌──────────────────────┐
         │  ENGINEERING Mode │     │      LLM Mode        │
         │  jieba + 规则     │     │  一次 LLM 调用        │
         │  同义词扩展       │     │  输出 JSON            │
         │  时间提取         │     │  含 graph_config      │
         └────────┬─────────┘     └──────────┬───────────┘
                  │                           │
                  └─────────┬─────────────────┘
                            ▼
              ┌─────────────────────────────┐
              │   Queries (多类型多条)        │
              │  semantic / keyword / entity │
              │  temporal / fs_scope         │
              └──────┬──────────┬──────┬────┘
                     │          │      │
         ┌───────────┘          │      └───────────┐
         ▼                      ▼                   ▼
┌─────────────────┐  ┌──────────────────┐  ┌────────────────────┐
│   FS (BM25)     │  │   Vec (Semantic) │  │   Graph (Vector+   │
│                 │  │                  │  │   Keyword+Time)    │
│ search_bm25()   │  │ 批量 embed       │  │                    │
│ scope 限定      │  │ search_all_with_ │  │ search_nodes_by_   │
│ temporal 行过滤  │  │ embedding()      │  │ embedding()        │
│                 │  │ temporal 过滤     │  │ search_nodes()     │
│ top_k=8         │  │ top_k=15         │  │ search_by_time()   │
└────────┬────────┘  └────────┬─────────┘  │                    │
         │                    │             │ 子图展开:           │
         │                    │             │ BFS分跳+相似度排序  │
         │                    │             │ hop_top_k截断       │
         │                    │             │ top_k=3×2=6 种子    │
         │                    │             └──────────┬─────────┘
         │                    │                        │
         └────────────────────┼────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────────┐
              │        融合排序（二选一）            │
              │                                   │
              │  ┌─────────────────────────────┐  │
              │  │ A. Rerank 精排               │  │
              │  │  1. 统一文本化               │  │
              │  │  2. Jaccard 去重(≥0.75)      │  │
              │  │  3. Cross-Encoder 精排       │  │
              │  │     (失败→Embedding fallback) │  │
              │  │  4. 阈值截断 + top_k         │  │
              │  │  5. 多跳保护配额             │  │
              │  └─────────────────────────────┘  │
              │                                   │
              │  ┌─────────────────────────────┐  │
              │  │ B. RRF 融合 (fallback)       │  │
              │  │  score = Σ 1/(k+rank)        │  │
              │  │  k=60, 取 top 20             │  │
              │  └─────────────────────────────┘  │
              └───────────────────┬───────────────┘
                                  │
                                  ▼
              ┌───────────────────────────────────┐
              │          格式化输出                 │
              │                                   │
              │  ## File System (BM25):           │
              │    [score] path | meta: desc      │
              │      content lines...             │
              │                                   │
              │  ## Vector DB (Semantic):          │
              │    text (按 ingest_time 排序)      │
              │                                   │
              │  ## Graph DB (Entity Relations):   │
              │    [Center] node_id (label)        │
              │    -- Hop 1 --                     │
              │      src --[rel]--> tgt {props}    │
              │    -- Hop 2 --                     │
              │      src --[rel]--> tgt {props}    │
              └───────────────────────────────────┘
```

---

## 附录：LLM Query 改写 Prompt 的 graph_config 动态配置

LLM 模式下，模型会根据 query 复杂度动态设置 graph 检索参数：

| Query 复杂度 | depth | hop_top_k 示例 | 适用场景 |
|-------------|-------|---------------|---------|
| 简单事实 | 3 | `{1:10, 2:10, 3:5}` | "What is Alex's favorite food?" |
| 多实体 | 3 | `{1:12, 2:15, 3:8}` | "What do Alex and Bob have in common?" |
| 多跳推理 | 5 | `{1:8, 2:12, 3:15, 4:10, 5:5}` | "What hobby does Alex's friend's sister enjoy?" |
| 时间链 | 4 | `{1:10, 2:15, 3:10, 4:5}` | "What happened after Alex met Bob?" |
