# `llm_gateway.rl.rl_env` — RL Sandbox for llm_gateway memory tasks

轻量的环境/沙盒层，围绕 llm_gateway 的 4 套 memory task（atomic_code_t2 / code_t2 /
multi_code_t2 / t2）+ storage 三后端（FS + Vec + Graph），支撑 **RL rollout / 数据生产**
所需的两件核心能力：

1. **单步执行**：一次 `step_*` = 对应 task 的一次完整 agent loop
   （ingest / consolidate / query），直到 `finish` 或 `max_turns`。
2. **状态回退**：`snapshot()` / `restore(snap)` 可以在任意时刻冻结/回放三
   个后端的完整状态，支持对同一基态做多种 action 的分叉式 rollout。
3. **可持久化 / 可传输**：snapshot 可序列化为单文件 `.cbsnap`，跨进程加载，
   支持内容去重（CAS），适合大规模并行 rollout。

## 模块布局

```
llm_gateway/rl/rl_env/
├── __init__.py             # 对外 API
├── _models.py              # 本地化 Event/EventType/TaskResult/ToolCall
├── env.py                  # MemoryEnv（核心）、StepResult
├── task_factory.py         # 4 版本 task dispatcher
├── snapshot.py             # InMemory/Encoded/Filesystem 三个 SnapshotBackend
├── git_backend.py          # GitFsSnapshotBackend：FS 走 git ref
├── serialize.py            # 高效编解码（zstd + orjson + numpy float32 + CAS）
├── snapshot_session.py     # 加载老 .cbsnap 数据集到 MemoryEnv
├── scorer.py               # BaseScorer + 三阶段占位
└── README.md               # 本文件
```

## 快照后端选型

四种后端，同一个 `SnapshotBackend` 抽象：

| 后端 | FS 容量 | 速度 | 跨进程 | 适用场景 |
|---|---|---|---|---|
| `InMemorySnapshotBackend` | **最大**（原生 copytree + deepcopy） | 最快 | ❌ | 单进程内快速分叉 rollout |
| `EncodedSnapshotBackend` | **压缩 5-10×** | 每 MB 个位数 ms | ⚠ 仅内存 | 跨进程前内存中保存多个候选 |
| `FilesystemSnapshotBackend` | **同上，落盘** + 可选 CAS 去重 | I/O 主导 | ✅ `.cbsnap` 单文件 | 长时 rollout、分布式训练 |
| **`GitFsSnapshotBackend`** | **接近 git pack 极限** | **~10ms / step** | ✅ `export_bundle` | **最推荐用于 enable_git=True 的真实 rollout** |

## 多版本 Task 绑定

```python
env = MemoryEnv(
    llm=...,
    embedder=...,
    base_dir="/tmp/rl/u42",
    task_version="atomic_code_t2",  # 或 "code_t2" / "multi_code_t2" / "t2"
)
```

| task_version | ingest                              | consolidate                          | retrieve                              |
|--------------|-------------------------------------|--------------------------------------|---------------------------------------|
| `atomic_code_t2` | `IngestContextAtomicCodeT2Task` | `ConsolidateContextAtomicCodeT2Task` | `RetrieveContextAtomicCodeT2Task` |
| `code_t2`    | `IngestContextCodeTask`             | `ConsolidateContextTask`             | `RetrieveContextCodeTask`             |
| `multi_code_t2` | `IngestContextMultiCodeTask`     | `ConsolidateContextTask`             | `RetrieveContextMultiCodeTask`        |
| `t2`         | `IngestContextTask`                 | `ConsolidateContextTask`             | `RetrieveContextTask`                 |

## 最小示例

### 1) 内存内分叉

```python
from llm_gateway.rl.rl_env import MemoryEnv, InMemorySnapshotBackend

env = MemoryEnv(
    llm=..., embedder=..., base_dir="/tmp/rl/u42",
    task_version="atomic_code_t2",
    snapshot_backend=InMemorySnapshotBackend(),
)
await env.reset(user_id="u42")

await env.step_ingest(session_id="s0", messages=[...])
snap = env.snapshot()

await env.step_consolidate()
env.restore(snap)
ctx = await env.step_query(query="...")
```

### 2) 落盘 + 跨进程加载

```python
from llm_gateway.rl.rl_env import (
    MemoryEnv, FilesystemSnapshotBackend, load_snapshot,
)

env = MemoryEnv(..., snapshot_backend=FilesystemSnapshotBackend(
    "/shared/snaps", fs_mode="cas"))
await env.reset(user_id="u42")
snap = env.snapshot()

# consumer 进程
env2 = MemoryEnv(..., snapshot_backend=FilesystemSnapshotBackend(
    "/shared/snaps", fs_mode="cas"))
await env2.reset(user_id="u42")
env2.restore(env2.snapshot_backend.attach("/shared/snaps/abc.cbsnap"))
```

### 3) Git 后端：FS 走 git，Vec/Graph 走 inner backend

```python
from llm_gateway.rl.rl_env import (
    MemoryEnv, GitFsSnapshotBackend, EncodedSnapshotBackend,
)

env = MemoryEnv(
    llm=..., embedder=..., base_dir="/tmp/rl/u42",
    enable_git=True,
    snapshot_backend=GitFsSnapshotBackend(
        inner=EncodedSnapshotBackend(),
    ),
)
await env.reset(user_id="u42")

await env.step_ingest(session_id="s0", messages=[...])
snap = env.snapshot()  # FS 状态 = 40 字节 SHA + Vec/Graph deepcopy

# 跨机分发
env.snapshot_backend.export_bundle(snap, "/shared/snap_001/", env.fs, env.vec, env.graph)
```

### 4) 加载老 workspace 的 `.cbsnap` 数据集

```python
from llm_gateway.rl.rl_env import SnapshotSession

session = SnapshotSession(
    data_root="/data/home/trevzhang/projects/memory-ai-agent-workspace/rl_data_test_2",
    task_version="atomic_code_t2",
    enable_git=False,
)
with session.load(snapshot_id="...") as loaded:
    env = loaded.env
    await env.step_query(query="...", session_id="s0")
```

## 与 llm_gateway 主体的对接细节

- **Stores**：env 的 `reset()` 直接构造
  `storage.{file_system_store.FileSystemStore, vector_stores.VectorStore,
  graph_stores.GraphStore}`（memory backend）或对应 PG 实现，**不**走
  `storage.stores_factory.make_stores`，因为后者强制 `fs_path={root_dir}/{user_id}/memory_repo`
  且会缓存到全局 dict，与 RL 的"任意 base_dir + 隔离实例"语义冲突。
- **LLM/Embedder**：使用 `utils.memory_llm_interface.{LLMInterface, EmbeddingInterface}`。
- **Task**：通过 `task_factory.build_task_triad(task_version, ...)` 工厂化绑定到
  `atomic_code_t2.{ingest_task,consolidate_task,retrieve_task}` 或
  `context_task.*` 中的对应实现；调用走 `await task.execute(user_id, session_id, **kwargs)`，
  返回 dict 由 `dispatch_event` 反向映射为 `TaskResult`。

## 限制 / 已知差异

- `apply_*_tool_calls`（T3 风格 rollout）**当前阶段不可用**：llm_gateway 没有
  T3 系列 task。等 T3 task 接入后再启用。
- 原 `agent_memory.tasks` 风格的 `Event(payload={user_id,session_id,content,...})`
  入参在 `step_*` 方法内部转换为 llm_gateway task 的实际签名（messages/query/...），
  对外保持兼容；`content` 形参会自动 wrap 成单条 user message。
- llm_gateway storage 的 `GraphStore` 比老 `agent_memory` 多了 `_node_embeddings` /
  `_edge_embeddings`；snapshot/serialize 已适配（缺失字段静默忽略，向前兼容）。

## 测试

迁移测试位于 `llm_gateway/tests/rl/`。
