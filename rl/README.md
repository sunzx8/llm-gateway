# `llm_gateway/rl/` — 离线 RL 管线

源自老 workspace `memory-ai-agent-workspace/{src/rl_env,slime_train}` 的迁移版本。
本包**仅训练 / 评测 / 数据侧使用**，gateway 主服务运行时不导入，零启动开销。

## 子包

```
llm_gateway/rl/
├── rl_env/                # MemoryEnv + Snapshot/Restore 沙盒
├── slime_train/           # Slime 训练定制脚本
└── data_gen/              # 数据生成（占位；老 workspace 仍在跑）
```

## 与 llm_gateway 主体的对接

- **存储层**：直接使用 `llm_gateway.storage.{file_system_store, vector_stores, graph_stores}`
- **LLM 接口**：使用 `llm_gateway.utils.memory_llm_interface.{LLMInterface, EmbeddingInterface}`
- **Task 实现**：通过 `task_version` 参数绑定不同实现：
  - `atomic_code_t2`（默认）→ `llm_gateway.atomic_code_t2.{ingest_task, consolidate_task, retrieve_task}`
  - `code_t2` → `llm_gateway.context_task.{ingest_context_code_task, generate_code_context_code_t2_task, retrieve_context_code_task}`
  - `multi_code_t2` → `llm_gateway.context_task.{ingest_context_multi_code_task, generate_code_context_multi_code_t2_task, retrieve_context_multi_code_task}`
  - `t2` → `llm_gateway.context_task.{ingest_context_task, consolidate_context_task, retrieve_context_task}`
  - `t2_agent_loop` → `llm_gateway.atomic_t2_agent_loop.{ingest_task, consolidate_task, retrieve_task}`
    （来自 `memory-ai-agent-workspace` 仓库 `dev-0421-t3` 分支的 T3 任务，仅
    agent loop 模式；统一适配到 `BaseContextTask` 接口；ingest/consolidate/retrieve
    分别 17 / 23 / 6 tools，max_turns 默认 5 / 50 / 5）

## 最小用法

### 1) 单步 ingest + snapshot/restore

```python
import asyncio
from llm_gateway.rl.rl_env import MemoryEnv, InMemorySnapshotBackend
from llm_gateway.utils.memory_llm_interface import LLMInterface, EmbeddingInterface

async def main():
    env = MemoryEnv(
        llm=LLMInterface({"provider": "openai", "model": "gpt-5.4", ...}),
        embedder=EmbeddingInterface({"provider": "openai_compat", "model": "bge-m3", ...}),
        base_dir="/tmp/rl/u42",
        task_version="atomic_code_t2",
        snapshot_backend=InMemorySnapshotBackend(),
    )
    await env.reset(user_id="u42")

    await env.step_ingest(session_id="s0", messages=[
        {"role": "user", "content": "我是初二学生"},
        {"role": "assistant", "content": "好的，我记下了"},
    ])
    snap = env.snapshot()

    ctx = await env.step_query(query="我是几年级？", session_id="s0")
    print(ctx.task_result.final_output)

    env.restore(snap)
    env.close()

asyncio.run(main())
```

### 2) 加载老 workspace 产出的 `.cbsnap` 数据集

```python
from llm_gateway.rl.rl_env import SnapshotSession

session = SnapshotSession(
    data_root="/data/home/trevzhang/projects/memory-ai-agent-workspace/rl_data_test_2",
    task_version="atomic_code_t2",
)
with session.load(traj_id="0986", snapshot_id="...") as loaded:
    env = loaded.env
    ctx = await env.step_query(query="...", session_id="s0")
```

## 训练入口

### Reward / rollout 入口

单任务训练使用各自 reward；混合训练统一使用 `mixed_reward` 自动按 `metadata.task` 分发：

```text
# Slime --custom-rm-path
llm_gateway.rl.slime_train.tasks.ingest_reward.reward.reward_func
llm_gateway.rl.slime_train.tasks.consolidate_reward.reward.reward_func
llm_gateway.rl.slime_train.tasks.retrieve_reward.reward.reward_func
llm_gateway.rl.slime_train.tasks.mixed_reward.reward.reward_func

# Slime --custom-generate-function-path
llm_gateway.rl.slime_train.memory_rl.custom_generate.custom_generate
```

### 训练模式开关

| 场景 | 环境变量 |
|------|----------|
| `atomic_code_t2` 写入任务工具调用策略训练 | `MEMORY_RL_TASK_VERSION=atomic_code_t2 MEMORY_RL_APPLY_MODE=tool_calls` |
| `atomic_code_t2` 检索任务训练/评测 | `MEMORY_RL_TASK_VERSION=atomic_code_t2 MEMORY_RL_APPLY_MODE=task_loop MEMORY_RL_TRAIN_TASK_LOOP=1` |
| `t2_agent_loop` 原生 agent loop 训练 | `MEMORY_RL_TASK_VERSION=t2_agent_loop MEMORY_RL_APPLY_MODE=task_loop MEMORY_RL_TRAIN_TASK_LOOP=1` |
| 仅用 task loop 做 smoke/eval，不把内部 loop 接入 loss | `MEMORY_RL_APPLY_MODE=task_loop MEMORY_RL_TRAIN_TASK_LOOP=0` |

说明：`MEMORY_RL_TRAIN_TASK_LOOP=1` 时，`custom_generate` 会把任务自身的 `run()` 接到
Slime rollout router，并写入 `sample.tokens` / `sample.response_length` / `sample.loss_mask`；工具
observation 使用 `loss_mask=0`。当前支持：

- `t2_agent_loop`：`ingest` / `consolidate` / `retrieve` 都是原生 agent loop 训练。
- `atomic_code_t2`：`retrieve` 走原生 query task（LLM rewrite → 三后端召回 → context return）；`ingest` / `consolidate` 默认仍可走工具调用策略训练。

Probe reward 统一语义：对每个 probe 先调用当前 `task_version` 绑定的 query task 做检索，再把检索到的
context 交给 frozen QA 回答并打分；不再绕过 query task 直接调用底层 `_search_*`。

### 数据构建

单任务仍可使用原 converter：

```bash
python3 rl/slime_train/ingest/convert_to_slime_format.py \
  --input data/rl_data_test_2/rl_data.jsonl \
  --data-root data/rl_data_test_2 \
  --output /tmp/rl_smoke/ingest_train.jsonl

python3 rl/slime_train/consolidate/convert_to_slime_format.py \
  --input data/rl_data_test_2/rl_data.jsonl \
  --data-root data/rl_data_test_2 \
  --output /tmp/rl_smoke/consolidate_train.jsonl

python3 rl/slime_train/retrieve/convert_to_slime_format.py \
  --input data/rl_data_test_2/rl_data.jsonl \
  --data-root data/rl_data_test_2 \
  --output /tmp/rl_smoke/retrieve_train.jsonl
```

混合训练使用统一 builder：

```bash
python3 rl/slime_train/memory_rl/build_mixed_data.py \
  --input data/rl_data_test_2/rl_data.jsonl \
  --data-root data/rl_data_test_2 \
  --tasks ingest+consolidate+retrieve \
  --output /tmp/rl_smoke/mixed_train.jsonl \
  --eval-output /tmp/rl_smoke/mixed_val.jsonl
```

`--tasks` 支持以下别名和组合：

| 名称 | 含义 |
|------|------|
| `ingest` | 摄入 |
| `consolidate` / `evolve` | 演进 |
| `retrieve` / `query` / `consume` | 消费 |
| `ingest+consolidate` | 摄入 + 演进 |
| `ingest+retrieve` | 摄入 + 消费 |
| `consolidate+retrieve` | 演进 + 消费 |
| `ingest+consolidate+retrieve` | 摄入 + 演进 + 消费 |

每条混合样本会写入 `metadata.task`，`custom_generate` 与 `mixed_reward` 会据此自动选择对应逻辑。

### 本地 smoke

单任务：

```bash
python3 rl/slime_train/memory_rl/smoke_vllm_rollout_reward.py \
  --task ingest \
  --data /tmp/rl_smoke/ingest_train.jsonl \
  --snapshot-data-root data/rl_data_test_2 \
  --api-url http://<host>:<port>/v1/chat/completions \
  --model <model> \
  --task-version atomic_code_t2 \
  --apply-mode tool_calls \
  --limit 1
```

混合：

```bash
python3 rl/slime_train/memory_rl/smoke_vllm_rollout_reward.py \
  --task mixed \
  --data /tmp/rl_smoke/mixed_train.jsonl \
  --snapshot-data-root data/rl_data_test_2 \
  --api-url http://<host>:<port>/v1/chat/completions \
  --model <model> \
  --task-version atomic_code_t2 \
  --apply-mode tool_calls \
  --limit 4
```

### Slime 训练脚本

单任务脚本：

```bash
bash rl/slime_train/ingest/scripts/train_ingest_grpo.sh ...
bash rl/slime_train/consolidate/scripts/train_consolidate_grpo.sh ...
bash rl/slime_train/retrieve/scripts/train_retrieve_grpo.sh ...
```

混合训练脚本（若 `TRAIN_DATA` 与 `VAL_DATA`/`EVAL_DATA` 已存在，会直接使用预切分数据；否则自动构建）：

```bash
MIXED_TASKS=ingest+consolidate+retrieve \
INPUT_RL_DATA=data/rl_data_test_2/rl_data.jsonl \
SNAPSHOT_DATA_ROOT=data/rl_data_test_2 \
bash rl/slime_train/memory_rl/train_mixed_grpo.sh ...
```

传入已切分 train/val：

```bash
TRAIN_DATA=/path/to/train.jsonl \
VAL_DATA=/path/to/val.jsonl \
SNAPSHOT_DATA_ROOT=data/rl_data_test_2 \
bash rl/slime_train/memory_rl/train_mixed_grpo.sh ...
```

训练脚本启动时会打印数据摘要日志：`[data:train]` / `[data:val]`，包括样本数、任务分布和 prompt role 分布。

`t2_agent_loop` 内部 agent loop 可训练混合示例：

```bash
MIXED_TASKS=ingest+consolidate+retrieve \
MEMORY_RL_TASK_VERSION=t2_agent_loop \
MEMORY_RL_APPLY_MODE=task_loop \
MEMORY_RL_TRAIN_TASK_LOOP=1 \
FROZEN_MODEL_URL=http://<frozen-host>:<port>/v1/chat/completions \
FROZEN_MODEL_NAME=<qa-model> \
bash rl/slime_train/memory_rl/train_mixed_grpo.sh ...
```

`retrieve` 任务的 reward 不会在 reward 阶段二次检索：rollout 必须产出完整 retrieval context
（`retrieved_context` / `task_result.final_output`），reward 只基于这个 context 做 frozen QA。
`ingest` / `consolidate` 的 probe 评估会在 apply 写入/演进后，用外部 frozen query task 先检索再回答；
可用以下环境变量指定该检索模型：

```bash
MEMORY_RL_RETRIEVE_LLM_API_URL=http://<frozen-host>:<port>/v1/chat/completions
MEMORY_RL_RETRIEVE_LLM_MODEL=<qa-or-retrieve-model>
PROBE_TASK_MAX_CONCURRENCY=1  # 默认串行，避免 query task 共享状态并发污染
```

## 依赖

```
zstandard
orjson
numpy
```

已加入主 `requirements.txt`。Slime / SGLang / vLLM 训练侧依赖在
`slime_train/requirements-train.txt`（不入主服务）。

## 迁移参考

import 替换清单见 `_migration_import_map.md`。
