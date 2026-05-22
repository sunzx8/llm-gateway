# Mem0 记忆演进策略参考文档

> 本文档从 mem0 开源项目源码中提取，描述其记忆演进（Evolve）的完整方案。
> 格式遵循 EVOLVE_STRATEGY_SYSTEM_PROMPT 中的输出格式要求。

---

## 概述

mem0 的记忆演进采用 **UPDATE_MEMORY_PROMPT 驱动的四操作模型**：
- ADD：新增记忆
- UPDATE：更新已有记忆（同实体同属性被修订）
- DELETE：删除矛盾/过时记忆
- NONE：无变更

演进发生在**摄入阶段的旧版流程中**（V2），由LLM对比新提取的事实与已有记忆来决定操作。
V3版本将演进简化为 ADD-only + Hash去重，将UPDATE/DELETE推迟到显式调用。

---

## C1: 同主题重复/冲突处理

### 策略

| 场景 | 判断条件 | 操作 | 示例 |
|------|---------|------|------|
| 语义重复（方向相同） | 新旧事实表达相同含义，信息量无差异 | NONE（保留已有） | 旧："Likes cheese pizza" + 新："Loves cheese pizza" → NONE |
| 语义重复（新更具体） | 新事实包含更多信息 | UPDATE（保留更具体的） | 旧："User likes to play cricket" + 新："Loves to play cricket with friends" → UPDATE |
| 同主题不同方面 | 新事实补充新维度 | ADD | 旧："Loves cheese pizza" + 新："Loves chicken pizza" → UPDATE合并为"Loves cheese and chicken pizza" |
| 同主题立场相反 | 新事实与旧事实矛盾 | DELETE旧 + ADD新 | 旧："Loves cheese pizza" + 新："Dislikes cheese pizza" → DELETE旧 |

### 去重机制

```python
# 1. Hash精确去重（Phase 5）
mem_hash = hashlib.md5(text.encode()).hexdigest()
if mem_hash in existing_hashes:
    continue  # 完全相同文本，跳过

# 2. 语义去重（通过LLM在提取阶段完成）
# ADDITIVE_EXTRACTION_PROMPT 中的去重指令：
# "Use these ONLY for deduplication and linking — do NOT extract new memories 
#  from Existing Memories. If new information in New Messages is semantically 
#  equivalent to an Existing Memory with no meaningful new context, skip it."

# 3. 实体级去重（entity_store，相似度>=0.95视为同一实体）
if matches and matches[0].score >= 0.95:
    # 合并linked_memory_ids而非创建新实体
    pass
```

---

## C2: 跨会话洞察提取

### 策略

mem0 V3 版本**不在摄入阶段做跨会话归纳**。其设计哲学是：
- 摄入阶段只做 ADD（原子事实提取）
- 跨会话模式识别留给检索阶段的 scoring 机制（BM25 + 语义 + 实体boost）

### 间接支持

通过 `last_k_messages`（最近10条历史消息）提供跨轮次上下文，帮助LLM：
- 消解代词引用
- 避免重复提取已有信息
- 识别信息的延续性（通过 linked_memory_ids 关联）

---

## C3: 事件 → 偏好提升

### 策略

mem0 不区分"事件"和"偏好"的存储层级。所有信息统一为原子记忆条目。
但通过 ADDITIVE_EXTRACTION_PROMPT 中的质量标准间接处理：

| 输入模式 | 提取结果 | 说明 |
|---------|---------|------|
| 单次事件 "I went to a cooking class yesterday" | "User went to a cooking class around [具体日期]" | 时间锚定，保留为事件 |
| 类目级表态 "I love comedy shows" | "User loves comedy shows" | 直接作为偏好记录 |
| 重复事件（跨session） | 由去重机制跳过重复，保留首次提取 | 不自动升格 |

---

## C4: 立场演化追踪

### 策略

mem0 通过 ADDITIVE_EXTRACTION_PROMPT 的"变迁完整性"规则处理：

```
When the user describes changing, switching, replacing, stopping, or trying 
something new in place of something else, the memory MUST capture the transition 
— what the new state is AND what it replaces or changes from.

Good: "User switched from almond milk to oat milk lattes after developing an almond sensitivity"
Bad: "User prefers oat milk lattes"
```

### linked_memory_ids 关联

```json 新记忆通过 linked_memory_ids 关联到旧的相关记忆
{
  "id": "0",
  "text": "User is switching teams at Shopify to the payments platform in April 2025",
  "linked_memory_ids": ["b2c3d4e5-6789-abcd-ef01-222222222222"]  // 关联到"User works at Shopify"
}
```

---

## C5: 实体图谱结构优化

### 策略

mem0 的 entity_store 维护规则：

| 操作 | 触发条件 | 实现方式 |
|------|---------|---------|
| 实体合并 | 新实体与已有实体相似度>=0.95 | 合并 linked_memory_ids 列表 |
| 实体创建 | 无匹配的已有实体 | 插入新实体记录 |
| 实体清理（记忆更新时） | 记忆文本变更 | 从旧实体移除memory_id，从新文本提取实体重新链接 |
| 实体清理（记忆删除时） | 记忆被删除 | 从所有关联实体移除该memory_id |

### 实体数据结构

```python
entity_payload = {
    "data": "Marcus",           # 实体文本
    "entity_type": "PERSON",    # 实体类型
    "linked_memory_ids": ["mem_001", "mem_002", "mem_003"],  # 关联的记忆ID列表
    "user_id": "u1",            # 所属用户
}
```

---

## C6: 文件系统重组

mem0 不使用文件系统存储记忆（纯向量库架构），此项不适用。

---

## C7: 向量库健康

### 策略

| 检查项 | 实现方式 | 说明 |
|-------|---------|------|
| Hash去重 | 写入前检查 `existing_hashes` | 防止完全重复 |
| 批内去重 | `seen_hashes` 集合 | 同一批次内不重复写入 |
| 嵌入失败容错 | 批量嵌入失败时逐条降级 | 确保部分失败不影响整体 |
| 写入失败容错 | 批量插入失败时逐条降级 | 确保部分失败不影响整体 |
| BM25索引同步 | 每条记忆同时存储 `text_lemmatized` | 支持关键词检索 |

---

## C8: 索引自检

### 策略

mem0 通过 SQLite history 表维护操作日志：

```sql
-- history 表结构
CREATE TABLE history (
    id INTEGER PRIMARY KEY,
    memory_id TEXT,
    old_memory TEXT,
    new_memory TEXT,
    event TEXT,        -- ADD / UPDATE / DELETE
    created_at TEXT,
    updated_at TEXT,
    actor_id TEXT,
    role TEXT,
    is_deleted INTEGER DEFAULT 0
);
```

每次操作后自动记录，支持通过 `memory.history(memory_id)` 查询变更历史。

---

## C9: 检索路径优化

### 策略

mem0 通过以下机制优化检索路径：

| 优化维度 | 实现方式 | 效果 |
|---------|---------|------|
| 混合检索 | 语义搜索 + BM25关键词搜索 + 实体boost | 多信号融合提高召回率 |
| 实体图谱路由 | 从query提取实体 → 搜索entity_store → boost关联记忆 | 精确实体匹配增强 |
| BM25词形还原 | 存储时预计算 `text_lemmatized` | 加速关键词匹配 |
| 过度获取+重排 | 语义搜索 top_k*4 → scoring重排 → 截断到top_k | 扩大候选池提高精度 |

---

## Total Plan

```
演进模型: 四操作模型 (ADD / UPDATE / DELETE / NONE)
V3简化: ADD-only + Hash去重 + 实体链接
去重层次: 3层（Hash精确 → LLM语义 → 实体相似度）
实体维护: 自动提取 + 相似度合并 + 生命周期清理
检索优化: 混合检索（语义 + BM25 + 实体boost）+ 过度获取重排
```

---

## 附录：DEFAULT_UPDATE_MEMORY_PROMPT 四操作决策规则（V2旧版，供参考）

```
操作选择规则：
1. ADD: 新信息不存在于已有记忆中 → 生成新ID添加
2. UPDATE: 新信息与已有记忆同主题但内容不同/更具体 → 保持原ID更新文本
3. DELETE: 新信息与已有记忆矛盾 → 标记删除
4. NONE: 新信息已存在于记忆中 → 不做任何操作

输出格式:
{
    "memory": [
        {"id": "<ID>", "text": "<内容>", "event": "ADD|UPDATE|DELETE|NONE", "old_memory": "<仅UPDATE时>"}
    ]
}
```
