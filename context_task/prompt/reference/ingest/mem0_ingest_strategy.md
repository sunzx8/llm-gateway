# Mem0 消息摄入策略参考文档

> 本文档从 mem0 开源项目源码中提取，描述其消息摄入（Ingest）的完整方案。
> 格式遵循 CODEGEN_INGEST_STRATEGY_SYSTEM_PROMPT 中的"结构化伪代码 + 决策表"要求。

---

## Part 1: 信息提取与存储路由决策表

mem0 采用 **单后端（向量数据库）+ 实体图谱辅助** 的存储架构，所有记忆统一存入向量库，实体关系存入实体图谱用于检索增强。

| 信息类型 | 识别模式/关键词 | 存储后端 | collection/路径 | metadata 字段 | 示例 |
|---------|---------------|---------|----------------|--------------|------|
| 个人事实 | "我是/我在/我有/Name is..." | vec + entity_store | collection=主集合 | data, hash, text_lemmatized, created_at, updated_at, user_id, attributed_to="user" | "Name is John" |
| 偏好立场 | "我喜欢/讨厌/favourite..." | vec + entity_store | collection=主集合 | data, hash, text_lemmatized, attributed_to="user" | "Favourite movies are Inception and Interstellar" |
| 计划意图 | "打算/计划/next month..." | vec + entity_store | collection=主集合 | data, hash, text_lemmatized, attributed_to="user" | "Planning a trip to Japan in March" |
| 职业信息 | "工作/职位/engineer..." | vec + entity_store | collection=主集合 | data, hash, text_lemmatized, attributed_to="user" | "Is a Software engineer" |
| 助手推荐 | 助手给出的具体建议/方案 | vec + entity_store | collection=主集合 | data, hash, text_lemmatized, attributed_to="assistant" | "User was recommended 'Formula 1: Drive to Survive'" |
| 共享内容 | 用户分享的文档/数据/案例 | vec + entity_store | collection=主集合 | data, hash, text_lemmatized, attributed_to="user" | "Bajimaya v Reward Homes: construction began in 2014..." |
| 程序性记忆 | agent执行历史摘要 | vec | collection=主集合 | data, hash, memory_type="procedural_memory", agent_id | 完整的agent执行步骤摘要 |
| 噪声/问候 | "Hi/Hello/Thanks" | — (NOOP) | — | — | "Hey, good morning!" |

### 关键设计决策

1. **单一向量集合**：所有类型的记忆存入同一个 collection，通过 metadata 字段区分类型
2. **实体图谱辅助**：从记忆文本中提取实体，建立 `entity → [linked_memory_ids]` 的映射关系
3. **Hash 去重**：使用 MD5 hash 防止完全重复的记忆写入
4. **BM25 索引**：每条记忆同时存储 `text_lemmatized` 字段用于关键词检索

---

## Part 2: 处理流程伪代码

```python
async def ingest_memory(self, messages: list[dict], user_id: str, session_id: str) -> str:
    """mem0 V3 Phased Batch Pipeline — 8阶段批处理流水线"""
    
    # Phase 0: 上下文收集
    session_scope = build_session_scope(filters)  # 如 "user_id=xxx"
    last_messages = self.db.get_last_messages(session_scope, limit=10)  # 获取最近10条历史消息
    parsed_messages = parse_messages(messages)  # 格式化为 "role: content\n" 文本
    
    # Phase 1: 已有记忆检索（用于去重和关联）
    query_embedding = self.embedding_model.embed(parsed_messages, "search")
    existing_results = self.vector_store.search(
        query=parsed_messages,
        vectors=query_embedding,
        top_k=10,
        filters={"user_id": user_id}
    )
    # 映射 UUID → 整数ID（防止LLM幻觉生成假UUID）
    existing_memories = []
    uuid_mapping = {}
    for idx, mem in enumerate(existing_results):
        uuid_mapping[str(idx)] = mem.id
        existing_memories.append({"id": str(idx), "text": mem.payload.get("data", "")})
    
    # Phase 2: LLM 提取（单次调用，ADD-only模式）
    system_prompt = ADDITIVE_EXTRACTION_PROMPT  # 详细的记忆提取系统提示
    user_prompt = generate_additive_extraction_prompt(
        existing_memories=existing_memories,      # 已有记忆（用于去重）
        new_messages=parsed_messages,             # 新消息
        last_k_messages=last_messages,            # 最近历史（用于代词消解）
        custom_instructions=custom_instructions,  # 自定义指令
    )
    response = self.llm.generate_response(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    extracted_memories = json.loads(response).get("memory", [])
    # 输出格式: [{"id": "0", "text": "...", "attributed_to": "user/assistant", "linked_memory_ids": [...]}]
    
    if not extracted_memories:
        self.db.save_messages(messages, session_scope)
        return ""  # NOOP
    
    # Phase 3: 批量嵌入所有提取的记忆文本
    mem_texts = [m.get("text", "") for m in extracted_memories if m.get("text")]
    mem_embeddings_list = self.embedding_model.embed_batch(mem_texts, "add")
    embed_map = dict(zip(mem_texts, mem_embeddings_list))
    
    # Phase 4: 逐条CPU处理 + Phase 5: Hash去重
    existing_hashes = {mem.payload.get("hash") for mem in existing_results if mem.payload.get("hash")}
    seen_hashes = set()  # 批内去重
    records = []
    
    for mem in extracted_memories:
        text = mem.get("text")
        if not text or text not in embed_map:
            continue
        
        # Hash去重
        mem_hash = hashlib.md5(text.encode()).hexdigest()
        if mem_hash in existing_hashes or mem_hash in seen_hashes:
            continue  # 跳过重复
        seen_hashes.add(mem_hash)
        
        # 构建metadata
        text_lemmatized = lemmatize_for_bm25(text)  # BM25索引用
        memory_id = str(uuid.uuid4())
        metadata = {
            "data": text,
            "text_lemmatized": text_lemmatized,
            "hash": mem_hash,
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if mem.get("attributed_to"):
            metadata["attributed_to"] = mem["attributed_to"]
        
        records.append((memory_id, text, embed_map[text], metadata))
    
    if not records:
        self.db.save_messages(messages, session_scope)
        return ""
    
    # Phase 6: 批量持久化到向量库
    self.vector_store.insert(
        vectors=[r[2] for r in records],
        ids=[r[0] for r in records],
        payloads=[r[3] for r in records],
    )
    
    # 批量写入历史记录
    self.db.batch_add_history([
        {"memory_id": r[0], "old_memory": None, "new_memory": r[1], "event": "ADD", "created_at": r[3]["created_at"]}
        for r in records
    ])
    
    # Phase 7: 批量实体链接（Entity Linking）
    all_texts = [r[1] for r in records]
    all_entities = extract_entities_batch(all_texts)  # 从文本中提取实体
    
    # 7a: 全局去重 — 收集所有唯一实体
    global_entities = {}  # normalized_key -> (entity_type, entity_text, set_of_memory_ids)
    for idx, (memory_id, text, embedding, payload) in enumerate(records):
        entities = all_entities[idx]
        for entity_type, entity_text in entities:
            key = entity_text.strip().lower()
            if key in global_entities:
                global_entities[key][2].add(memory_id)
            else:
                global_entities[key] = [entity_type, entity_text, {memory_id}]
    
    # 7b: 批量嵌入所有唯一实体
    entity_texts = [global_entities[k][1] for k in global_entities]
    entity_embeddings = self.embedding_model.embed_batch(entity_texts, "add")
    
    # 7c: 批量搜索已有实体（相似度>=0.95视为同一实体）
    existing_matches = self.entity_store.search_batch(
        queries=entity_texts, vectors_list=entity_embeddings, top_k=1, filters={"user_id": user_id}
    )
    
    # 7d: 分流 — 更新已有实体 vs 插入新实体
    for j, key in enumerate(global_entities):
        entity_type, entity_text, memory_ids = global_entities[key]
        matches = existing_matches[j]
        if matches and matches[0].score >= 0.95:
            # 更新已有实体：追加 linked_memory_ids
            match = matches[0]
            linked = set(match.payload.get("linked_memory_ids", []))
            linked |= memory_ids
            self.entity_store.update(vector_id=match.id, payload={"linked_memory_ids": sorted(linked)})
        else:
            # 新实体：插入
            self.entity_store.insert(
                vectors=[entity_embeddings[j]],
                ids=[str(uuid.uuid4())],
                payloads=[{"data": entity_text, "entity_type": entity_type, "linked_memory_ids": sorted(memory_ids), "user_id": user_id}]
            )
    
    # Phase 8: 保存消息 + 返回
    self.db.save_messages(messages, session_scope)
    
    return f"摄入完成：新增 {len(records)} 条记忆"
```

---

## Part 3: 数据格式规范表

| 后端 | 数据格式 | 示例 |
|------|---------|------|
| vec 主集合 payload | `{"data": "<记忆文本>", "hash": "<md5>", "text_lemmatized": "<词形还原文本>", "user_id": "...", "created_at": "ISO8601", "updated_at": "ISO8601", "attributed_to": "user/assistant"}` | `{"data": "User's name is Marcus", "hash": "a1b2c3...", "text_lemmatized": "user name marcus", "user_id": "u1", "created_at": "2025-01-01T00:00:00+00:00", "attributed_to": "user"}` |
| entity_store payload | `{"data": "<实体文本>", "entity_type": "<类型>", "linked_memory_ids": ["uuid1", "uuid2"], "user_id": "..."}` | `{"data": "Marcus", "entity_type": "PERSON", "linked_memory_ids": ["mem_001", "mem_002"], "user_id": "u1"}` |
| history DB (SQLite) | `{memory_id, old_memory, new_memory, event, created_at, updated_at, is_deleted}` | `{"memory_id": "uuid", "old_memory": null, "new_memory": "User likes pizza", "event": "ADD"}` |

### 记忆文本质量标准（来自 ADDITIVE_EXTRACTION_PROMPT）

| 维度 | 要求 | 正例 | 反例 |
|------|------|------|------|
| 上下文丰富 | 事实+周围上下文合为一条 | "User has a dog named Poppy and their morning walks together are the highlight of their day" | "User has a dog" |
| 自包含 | 替换所有代词为具体名称 | "Marcus was promoted to Senior Engineer at Shopify" | "He got promoted" |
| 简洁完整 | 15-80词，最多100词 | 1-2句话 | 超长段落 |
| 时间锚定 | 相对时间→绝对时间 | "User went to Paris the week of May 15, 2023" | "User went to Paris last week" |
| 数值精确 | 保留原始数值 | "416 pages" | "about 400 pages" |
| 保留专有名词 | 书名/地名/品牌名完整保留 | "'Eternal Sunshine of the Spotless Mind'" | "a movie" |
| 第一人称 | 事件/立场类用用户原话 | "I stopped listening to podcasts" | "Alex stopped listening" |
| 变迁完整 | 捕获新旧状态的转换 | "User switched from almond milk to oat milk lattes after developing an almond sensitivity" | "User prefers oat milk lattes" |

---

## Part 4: LLM 调用策略

| 场景 | 是否调用 LLM | 目的 | 输入 | 输出格式 |
|------|------------|------|------|----------|
| 记忆提取（V3 Additive） | 是（1次） | 从对话中提取所有可记忆信息 | system=ADDITIVE_EXTRACTION_PROMPT + user=generate_additive_extraction_prompt(...) | `{"memory": [{"id": "0", "text": "...", "attributed_to": "user/assistant", "linked_memory_ids": [...]}]}` |
| 程序性记忆摘要 | 是（1次，仅agent_id+procedural模式） | 将agent执行历史压缩为结构化摘要 | system=PROCEDURAL_MEMORY_SYSTEM_PROMPT + messages + "Create procedural memory" | 自由文本（结构化摘要） |
| 实体提取 | 否（规则/NER） | 从记忆文本中提取实体 | 记忆文本 | [(entity_type, entity_text), ...] |
| 去重判断 | 否（Hash比对） | 防止完全重复写入 | MD5(text) | 布尔值 |
| 语义去重 | 否（向量相似度） | 实体级别的近似去重 | 实体embedding相似度>=0.95 | 合并/新建 |

### LLM 提取的关键设计

1. **ADD-only 模式**：V3版本的提取只产生ADD操作，不在提取阶段做UPDATE/DELETE（与旧版DEFAULT_UPDATE_MEMORY_PROMPT不同）
2. **防幻觉设计**：已有记忆的UUID映射为整数ID传给LLM，LLM输出的linked_memory_ids引用这些整数ID
3. **双角色提取**：同时从user和assistant消息中提取，通过`attributed_to`字段区分来源
4. **去重输入**：将已有记忆和最近提取的记忆作为输入，让LLM自行跳过已存在的信息

---

## Part 5: 索引维护规则

### 会话消息归档
- 每轮摄入结束后调用 `self.db.save_messages(messages, session_scope)` 保存原始消息
- 用于后续摄入时提供 `last_k_messages` 上下文（代词消解）

### 历史记录维护
- 每条记忆的ADD/UPDATE/DELETE操作都记录到SQLite history表
- 字段：memory_id, old_memory, new_memory, event, created_at, updated_at, actor_id, role, is_deleted

### 实体图谱维护
- 新记忆写入后自动提取实体并链接到entity_store
- 记忆更新时：先从旧实体中移除memory_id，再从新文本提取实体重新链接
- 记忆删除时：从所有关联实体中移除该memory_id
