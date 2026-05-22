## 记忆增强的模型服务

整体目标：给用户提供一个记忆增强的模型服务。
详情参见：https://iwiki.woa.com/p/4020201525

## 调试与开发

### 1. 安装依赖

```bash
pip3 install -r requirements.txt
```

### 运行

```bash

python3 main.py -c config.yaml --reload --host 127.0.0.1 --port 8000

```

### 测试

**普通单轮测试**

```curl

curl -X POST http://127.0.0.1:8000/llm/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-my-test-key-123" \
  -d '{
    "model": "gpt-5.4",
    "messages": [
      {"role": "user", "content": "我是几年级的学生？"}
    ],
    "metadata": {
      "user_id": "alice",
      "session_id": "123"
    }
  }' 

```

**批量记忆导入**

```curl
curl -X POST http://127.0.0.1:8000/llm/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-my-test-key-123" \
  -d '{
    "model": "memory-initialize",
    "messages": [
      {"role": "user", "content": "我是一名初中二年级的学生，你是谁，可以帮我写作业吗？"},
      {"role": "assistant", "content": "我是一个人工智能助手，可以提供学习和写作方面的帮助。如果你有特定的作业问题或者需要辅导的内容，欢迎告诉我，我会尽力帮助你。请问你有什么具体的作业需要帮助吗？"}
    ],
    "metadata": {
      "user_id": "alice",
      "session_id": "123"
    }
  }' 

```

## memory数据存储

`config.yaml`配置中，将`memory_config.storage.backend` 设置为`pg`,同时通过 `docker/pg/` 目录的 `docker-compose.yaml`启动一个docker容器

```bash
cd docker/pg
docker-compose up -d
```

然后即可将数据存储在pg当中。

## RL 管线 (`llm_gateway/rl/`)

离线 RL 管线（`MemoryEnv` 沙盒 + Slime 训练定制脚本），**不参与主服务运行时**，
零启动开销。详见 [`rl/README.md`](rl/README.md)。

```
llm_gateway/rl/
├── rl_env/        # MemoryEnv + 4 种 SnapshotBackend（含 GitFs）
├── slime_train/   # Slime 训练定制脚本：reward / custom_generate / agentic rollout / smoke
└── data_gen/      # 数据生成（占位；当前仍在老 workspace 跑）
```

通过 `task_version` 在 5 套 ingest/consolidate/retrieve 实现间切换：

| `task_version`    | 绑定到的 task 实现                                              |
|-------------------|------------------------------------------------------------|
| `atomic_code_t2`  | `atomic_code_t2.{ingest,consolidate,retrieve}_task`（默认） |
| `code_t2`         | `context_task.{ingest_context_code_task, consolidate_context_task, retrieve_context_code_task}` |
| `multi_code_t2`   | `context_task.{ingest_context_multi_code_task, consolidate_context_task, retrieve_context_multi_code_task}` |
| `t2`              | `context_task.{ingest_context_task, consolidate_context_task, retrieve_context_task}` |
| `t2_agent_loop`   | `atomic_t2_agent_loop.{ingest,consolidate,retrieve}_task`（来自 `dev-0421-t3` 分支的 agent loop 模式 T3 任务，统一适配到 `BaseContextTask` 接口） |

最小用法：

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
ctx = await env.step_query(query="...")
env.restore(snap)
```

Slime 训练入口（在 `--custom-rm-path` / `--custom-generate-function-path` 上）：

```
llm_gateway.rl.slime_train.tasks.{ingest,consolidate,retrieve}_reward.reward.reward_func
llm_gateway.rl.slime_train.tasks.mixed_reward.reward.reward_func
llm_gateway.rl.slime_train.memory_rl.custom_generate.custom_generate
```

支持摄入、演进、消费及混合训练（如 `ingest+consolidate`、`ingest+consolidate+retrieve`）。
混合数据构建入口：`rl/slime_train/memory_rl/build_mixed_data.py`；混合训练脚本：
`rl/slime_train/memory_rl/train_mixed_grpo.sh`。单任务启动脚本见
`rl/slime_train/{ingest,consolidate,retrieve}/scripts/`。

测试：

```bash
PYTHONPATH=$(pwd)/.. pytest tests/rl/ -v
```

12 个测试覆盖 snapshot/restore 闭环、CAS 去重、跨进程 .cbsnap 加载、
retrieve converter 等关键路径。

### `atomic_t2_agent_loop/`（顶层包，被 `t2_agent_loop` task_version 引用）

来自 `memory-ai-agent-workspace` 仓库 `dev-0421-t3` 分支的 T3 任务（仅 agent loop
模式），统一对齐 `BaseContextTask` 接口、命名统一为 atomic + T2（多后端）+ agent
loop。三个任务均继承 `BaseContextTask`，自己持有 `fs/vec/graph` 三后端：

```
llm_gateway/atomic_t2_agent_loop/
├── ingest_task.py        # IngestT2AgentLoopTask    (17 tools, max_turns=5)
├── consolidate_task.py   # ConsolidateT2AgentLoopTask(23 tools, max_turns=50)
└── retrieve_task.py      # RetrieveT2AgentLoopTask  (6  tools, max_turns=5)
```

使用方式与其它 task 完全一致 — 走 `await task.execute(user_id, session_id, **kwargs)`
入口；不再用 `Event` 包装。RL env 启用：

```python
env = MemoryEnv(..., task_version="t2_agent_loop")
```