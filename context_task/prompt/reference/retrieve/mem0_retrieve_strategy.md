# Mem0 记忆消费策略参考文档

> 本文档从 mem0 开源项目源码中提取，描述其记忆消费（Retrieve/Search）的完整方案。
> 格式遵循 CODEGEN_RETRIEVE_STRATEGY_SYSTEM_PROMPT 中的"结构化伪代码 + 决策表"要求。

---

## Part 1: 检索路由决策表

mem0 采用 **混合检索（Hybrid Search）** 架构：语义搜索 + BM25关键词搜索 + 实体图谱boost，三路信号融合排序。

| query 类型 | 首选后端 | 检索方法 | 降级策略 | 输出格式 |
|-----------|---------|---------|---------|----------|
| 语义相关 | vec (semantic) | vector_store.search(query, vectors, top_k=limit*4, filters) | — | [{id, memory, score, hash, created_at, updated_at, metadata}] |
| 关键词精确 | vec (BM25) | vector_store.keyword_search(query_lemmatized, top_k=limit*4, filters) | 无BM25支持时跳过 | [{id, score}] → bm25_scores dict |
| 实体关联 | entity_store | entity_store.search(entity_text, entity_embedding, top_k=500, filters) | 无实体提取时跳过 | entity_boosts dict: {memory_id: boost_score} |
| 重排序 | reranker | reranker.rerank(query, results, limit) | reranker不可用时使用原始排序 | 重排后的结果列表 |

### 三路信号权重

| 信号 | 权重范围 | 说明 |
|------|---------|------|
| 语义相似度 (semantic_score) | [0, 1.0] | 向量余弦相似度，基础信号 |
| BM25关键词分数 (bm25_score) | [0, 1.0] (sigmoid归一化) | 词形还原后的关键词匹配 |
| 实体boost (entity_boost) | [0, 0.5] (ENTITY_BOOST_WEIGHT=0.5) | 实体图谱关联增强 |

### 最终得分计算

```
combined = (semantic_score + bm25_score + entity_boost) / max_possible
其中 max_possible = 1.0 + (1.0 if has_bm25) + (0.5 if has_entity)
```

---

## Part 2: 处理流程伪代码

```python
async def retrieve_memory(self, query: str, user_id: str, top_k: int = 20, threshold: float = 0.1) -> list[dict]:
    """mem0 混合检索流程 — 9步流水线"""
    
    # Step 1: Query预处理
    query_lemmatized = lemmatize_for_bm25(query)  # 词形还原（用于BM25）
    query_entities = extract_entities(query)       # 实体提取（用于entity boost）
    
    # Step 2: Query嵌入
    embeddings = self.embedding_model.embed(query, "search")
    
    # Step 3: 语义搜索（过度获取，4倍top_k）
    internal_limit = max(top_k * 4, 60)
    semantic_results = self.vector_store.search(
        query=query,
        vectors=embeddings,
        top_k=internal_limit,
        filters={"user_id": user_id}
    )
    
    # Step 4: BM25关键词搜索（如果向量库支持）
    keyword_results = self.vector_store.keyword_search(
        query=query_lemmatized,
        top_k=internal_limit,
        filters={"user_id": user_id}
    )
    
    # Step 5: BM25分数归一化（Sigmoid归一化到[0,1]）
    bm25_scores = {}
    if keyword_results:
        # 根据query长度自适应sigmoid参数
        num_terms = len(query_lemmatized.split())
        if num_terms <= 3:
            midpoint, steepness = 5.0, 0.7
        elif num_terms <= 6:
            midpoint, steepness = 7.0, 0.6
        elif num_terms <= 9:
            midpoint, steepness = 9.0, 0.5
        else:
            midpoint, steepness = 12.0, 0.5
        
        for mem in keyword_results:
            if mem.score > 0:
                # Sigmoid归一化: 1 / (1 + exp(-steepness * (score - midpoint)))
                bm25_scores[mem.id] = 1.0 / (1.0 + math.exp(-steepness * (mem.score - midpoint)))
    
    # Step 6: 实体Boost计算
    entity_boosts = {}
    if query_entities:
        # 去重实体（最多8个）
        seen = set()
        deduped_entities = []
        for entity_type, entity_text in query_entities[:8]:
            key = entity_text.strip().lower()
            if key and key not in seen:
                seen.add(key)
                deduped_entities.append((entity_type, entity_text))
        
        for _, entity_text in deduped_entities:
            entity_embedding = self.embedding_model.embed(entity_text, "search")
            matches = self.entity_store.search(
                query=entity_text,
                vectors=entity_embedding,
                top_k=500,
                filters={"user_id": user_id}
            )
            for match in matches:
                if match.score < 0.5:
                    continue  # 相似度阈值过滤
                
                linked_memory_ids = match.payload.get("linked_memory_ids", [])
                num_linked = max(len(linked_memory_ids), 1)
                
                # 扩散衰减：链接记忆越多的实体，单条boost越小
                memory_count_weight = 1.0 / (1.0 + 0.001 * ((num_linked - 1) ** 2))
                boost = match.score * 0.5 * memory_count_weight  # ENTITY_BOOST_WEIGHT = 0.5
                
                for memory_id in linked_memory_ids:
                    entity_boosts[memory_id] = max(entity_boosts.get(memory_id, 0.0), boost)
    
    # Step 7: 构建候选集（以语义搜索结果为基础）
    candidates = []
    for mem in semantic_results:
        candidates.append({
            "id": str(mem.id),
            "score": mem.score,
            "payload": mem.payload,
        })
    
    # Step 8: 融合评分 + 排序
    has_bm25 = bool(bm25_scores)
    has_entity = bool(entity_boosts)
    max_possible = 1.0 + (1.0 if has_bm25 else 0) + (0.5 if has_entity else 0)
    
    scored_results = []
    for candidate in candidates:
        semantic_score = candidate["score"]
        if semantic_score < threshold:  # 语义阈值门控
            continue
        
        mem_id = candidate["id"]
        bm25_score = bm25_scores.get(mem_id, 0.0)
        entity_boost = entity_boosts.get(mem_id, 0.0)
        
        combined = min((semantic_score + bm25_score + entity_boost) / max_possible, 1.0)
        scored_results.append({"id": mem_id, "score": combined, "payload": candidate["payload"]})
    
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    scored_results = scored_results[:top_k]
    
    # Step 9: 格式化输出
    results = []
    for scored in scored_results:
        payload = scored["payload"]
        results.append({
            "id": scored["id"],
            "memory": payload.get("data", ""),
            "score": scored["score"],
            "hash": payload.get("hash"),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
            "user_id": payload.get("user_id"),
            "metadata": {k: v for k, v in payload.items() if k not in CORE_KEYS},
        })
    
    return results
```

---

## Part 3: 数据格式兼容性表

| 后端 | 摄入写入格式 | 消费解析方式 | 关键字段 |
|------|------------|------------|----------|
| vec payload | `{"data": "<记忆文本>", "hash": "md5", "text_lemmatized": "<词形还原>", "user_id": "...", "created_at": "ISO", "updated_at": "ISO", "attributed_to": "user/assistant"}` | `payload.get("data")` 获取记忆文本；`payload.get("hash/created_at/updated_at")` 获取元信息 | data, hash, created_at |
| entity_store payload | `{"data": "<实体文本>", "entity_type": "<类型>", "linked_memory_ids": [...], "user_id": "..."}` | `match.payload.get("linked_memory_ids")` 获取关联记忆ID列表 | linked_memory_ids |
| BM25索引 | `text_lemmatized` 字段（词形还原后的文本） | `vector_store.keyword_search(query_lemmatized, ...)` 直接搜索 | text_lemmatized |
| 搜索结果 | `MemoryItem(id, memory, hash, created_at, updated_at, score)` | 标准化的结果对象 | id, memory, score |

---

## Part 4: 性能优化策略

| 优化点 | 策略 | 实现方式 | 效果 |
|-------|------|----------|------|
| 过度获取 | 语义搜索获取4倍候选 | `internal_limit = max(top_k * 4, 60)` | 扩大候选池，提高重排后精度 |
| 自适应BM25归一化 | 根据query词数调整sigmoid参数 | 短query(≤3词): midpoint=5, steepness=0.7; 长query(>15词): midpoint=12, steepness=0.5 | 不同长度query的BM25分数可比 |
| 实体扩散衰减 | 高扇出实体降低单条boost | `weight = 1/(1 + 0.001*(num_linked-1)²)` | 防止通用实体过度boost |
| 实体数量限制 | 最多处理8个去重实体 | `query_entities[:8]` + 去重 | 控制entity_store查询次数 |
| 语义阈值门控 | 低于threshold的候选直接排除 | `if semantic_score < threshold: continue` | 减少无关候选的后续计算 |
| 可选Reranker | 支持外部reranker重排 | `self.reranker.rerank(query, results, limit)` | 进一步提升排序质量 |
| 并行检索 | 语义搜索和BM25搜索可并行 | 两者独立查询同一向量库 | 减少总延迟 |

### BM25 Sigmoid参数自适应表

| Query词数 | midpoint | steepness | 说明 |
|-----------|----------|-----------|------|
| ≤3 | 5.0 | 0.7 | 短query，BM25原始分较低 |
| 4-6 | 7.0 | 0.6 | 中等query |
| 7-9 | 9.0 | 0.5 | 较长query |
| 10-15 | 10.0 | 0.5 | 长query |
| >15 | 12.0 | 0.5 | 超长query，BM25原始分较高 |

---

## Part 5: 证据强度判断规则

| 强度 | 判断条件 | 对应分数范围 | 说明 |
|------|---------|------------|------|
| strong | 三路信号均命中（语义+BM25+实体） | combined > 0.7 | 语义相关 + 关键词匹配 + 实体关联 |
| medium | 两路信号命中（语义+BM25 或 语义+实体） | 0.4 < combined ≤ 0.7 | 部分信号支持 |
| weak | 仅语义信号命中 | threshold < combined ≤ 0.4 | 仅向量相似度支持 |
| filtered | 语义分数低于阈值 | semantic_score < threshold (默认0.1) | 直接排除 |

### 实体Boost的证据增强逻辑

```python
# 实体boost的核心价值：当query中的实体与记忆关联时，即使语义相似度不高，
# 也能通过实体关联将相关记忆提升到结果中。

# 示例：
# Query: "Tell me about Poppy"
# 实体提取: [("ENTITY", "Poppy")]
# entity_store搜索: 找到实体"Poppy" → linked_memory_ids: ["mem_001", "mem_002", "mem_003"]
# 效果: mem_001/002/003 获得额外的 entity_boost，即使它们的语义相似度不是最高的

# 扩散衰减公式：
# boost = entity_similarity * 0.5 * (1.0 / (1.0 + 0.001 * (num_linked - 1)²))
# 
# 当实体链接1条记忆时: weight = 1.0 → boost最大
# 当实体链接10条记忆时: weight ≈ 0.92 → 轻微衰减
# 当实体链接100条记忆时: weight ≈ 0.09 → 显著衰减（通用实体）
```

---

## 附录：完整的检索架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Query Input                               │
│                    "Tell me about Poppy"                         │
└─────────────────────┬───────────────────────────────────────────┘
                      │
          ┌───────────┼───────────────────┐
          ▼           ▼                   ▼
┌─────────────┐ ┌──────────────┐ ┌────────────────┐
│  Embedding  │ │ Lemmatization│ │Entity Extraction│
│  embed()    │ │ lemmatize()  │ │ extract_entities│
└──────┬──────┘ └──────┬───────┘ └───────┬────────┘
       │               │                  │
       ▼               ▼                  ▼
┌─────────────┐ ┌──────────────┐ ┌────────────────┐
│  Semantic   │ │    BM25      │ │  Entity Store  │
│  Search     │ │   Search     │ │    Search      │
│ top_k=4*N   │ │ top_k=4*N    │ │  top_k=500     │
└──────┬──────┘ └──────┬───────┘ └───────┬────────┘
       │               │                  │
       │         ┌─────┘                  │
       │         │ Sigmoid归一化           │ 扩散衰减
       │         ▼                        ▼
       │    bm25_scores{}           entity_boosts{}
       │         │                        │
       ▼         ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Additive Scoring                              │
│  combined = (semantic + bm25 + entity_boost) / max_possible     │
│  Filter: semantic_score >= threshold                            │
│  Sort: by combined score descending                             │
│  Truncate: top_k results                                        │
└─────────────────────────────────────────────────────────────────┘
                      │
                      ▼ (optional)
┌─────────────────────────────────────────────────────────────────┐
│                    Reranker (optional)                           │
│  reranker.rerank(query, results, limit)                         │
└─────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Formatted Results                             │
│  [{id, memory, score, hash, created_at, updated_at, metadata}]  │
└─────────────────────────────────────────────────────────────────┘
```
