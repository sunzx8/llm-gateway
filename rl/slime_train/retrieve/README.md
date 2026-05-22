# Retrieve (消费) 阶段 — slime RLVR 训练框架

## 目标

训练当前 `task_version` 绑定的 **检索任务本身**，而不是单独训练 query-rewrite JSON。

- `t2_agent_loop`：rollout 跑完整 retrieve agent loop，模型产生多轮检索 tool calls，最终 `submit` 得到 context。
- `atomic_code_t2`：rollout 跑 atomic retrieve query task：LLM rewrite → FS/Vec/Graph 召回 → format context return。

## 训练逻辑（Agentic RL）

rollout 阶段必须产出完整检索结果（`retrieved_context` 或 `task_result.final_output`）以及 trajectory；
reward 阶段**不再重复执行检索**，只把 rollout 得到的 context 交给冻结模型回答 probe，再与 gold answer 对比。

- **阶段 1（有梯度）**：模型执行 retrieve task loop，产生 trajectory 和 final context。
- **阶段 2（无梯度，冻结模型评判）**：冻结模型基于 final context 回答 `probe_query + alt_queries`，对比 gold answer 打分。

## 目录结构

```
retrieve/
├── README.md                       # 本文件
├── convert_to_slime_format.py      # raw data (rl_data.jsonl) → slime JSONL
├── snapshot_loader.py              # rl_env.snapshot_session 兼容入口
├── scripts/
│   ├── train_retrieve_grpo.sh      # 训练启动脚本（含冻结模型 + SwanLab 监控）
│   └── swanlab_monitor.py          # SwanLab 实时指标监控
├── data/                           # 转换后的训练数据（由 convert_to_slime_format.py 生成）
│   ├── rl_train.jsonl
│   └── rl_val.jsonl
├── outputs/                        # 训练输出 checkpoint
├── logs/                           # 训练日志 + reward 细分指标
└── rollout_dumps/                  # rollout 输出 dump
```

## Reward 入口

Canonical custom reward path:

```text
slime_train.tasks.retrieve_reward.reward.reward_func
```

旧的 `slime_train/retrieve/tasks/retrieve_reward` 目录仅保留兼容用途，新训练/测试脚本应使用统一的 `slime_train.tasks.*` 包。

## 数据目录

共享数据位于 `slime_train/data/rl_data_test_2/`：

```
rl_data_test_2/
├── rl_data.jsonl                   # 原始数据（23 条 step record，81 条 retrieve probe）
├── ingest_snapshots.jsonl          # 快照索引（snapshot_id → trajectory_id, commit_hash, ...）
├── snapshots/{traj_id}/{snap_id}.cbsnap   # 完整快照文件（tar_zst 格式）
├── workdirs/{traj_id}/.git/        # git repo（convert 阶段用 git show 获取 FS 结构）
└── ...
```

### 数据格式

**rl_data.jsonl** 每行一个 step record：
- 外层: `trajectory_id`, `user_id`, `session_id`, `snapshot_id`
- `probes_by_task.retrieve[]` 列表，每条 probe:
  - `probe_query`: 问题文本
  - `alt_queries`: 备选问法（reward 评估时全部评测）
  - `question_type`: `"fill_in_the_blank"` | `"multiple_choice"`
  - `ground_truth`: 答案
  - `options`: 选择题选项
  - `answerable`: 是否可回答（当前数据全部为 true）
  - `probe_type`, `generation_mode`

### 快照格式 (.cbsnap)

`.cbsnap` 使用 `tar_zst` 格式，内含：
- FS 文件的 tar 包（完整的文件系统状态）
- Vec collections（entries + embeddings，numpy float32 + zstd 压缩）
- Graph nodes/edges

通过 `rl_env.serialize.decode_snapshot(encoded, fs, vec, graph)` 一次性恢复三后端状态。

## 工作流

### 1. 数据准备

```bash
python convert_to_slime_format.py \
    --input /path/to/slime_train/data/rl_data_test_2/rl_data.jsonl \
    --output data/rl_train.jsonl \
    --eval_output data/rl_val.jsonl \
    --eval_ratio 0.05
```

转换流程：
1. 遍历 `rl_data.jsonl` 中每个 record 的 `probes_by_task.retrieve`
2. 通过 `git ls-tree` + `git show` 从 workdirs 获取 FS 文件内容（零磁盘写入），用于兼容旧 prompt/debug
3. 输出 slime 格式 JSONL，metadata 中包含：`task=retrieve`、`query=probe_query`、`ground_truth`、`snapshot_id`、`traj_id`

### 2. 快照加载（rollout 阶段）

`SnapshotSession.load()` 的流程：
1. 从 `ingest_snapshots.jsonl` 查到 `snapshot_id → trajectory_id`
2. 创建临时目录，初始化 `MemoryEnv(base_dir=tmp, enable_git=False, task_version=$MEMORY_RL_TASK_VERSION)`
3. 从 `.cbsnap` 加载 `EncodedSnapshot`
4. 调用 `decode_snapshot(encoded, env.fs, env.vec, env.graph)` 恢复完整状态
5. `custom_generate` 调用 `env.step_query(query=metadata.query)`，由当前 task_version 的 retrieve task 产出 context

### 3. 训练

```bash
bash scripts/train_retrieve_grpo.sh
```

脚本自动完成:
1. 启动冻结模型推理引擎（SGLang，端口 30100）
2. 启动 SwanLab 监控
3. 打印 slime 训练命令（供 Docker 内执行）
4. 训练结束后自动清理

### 4. Reward 设计

| 子 Reward | 权重 | 说明 |
|-----------|------|------|
| R_retrieval_hit | 0.90 | 冻结模型根据 rollout 产出的 context 能否正确回答 `probe_query + alt_queries` |
| R_context_payload | 0.10 | rollout payload 是否包含可评分的 `retrieved_context` / `task_result.final_output` 和 trajectory |

**R_retrieval_hit 评分规则**:
- reward 从 rollout response 中读取 `retrieved_context`，不会在 reward 内二次执行检索
- 对 probe_query 和所有 alt_queries 分别让冻结模型回答
- 全部答对 → 1.0（满分）
- 否则 → 答分平均

### 5. SwanLab 监控指标

| 指标 | 来源 |
|------|------|
| `reward/r_retrieval_hit` | reward 函数细分日志 |
| `reward/r_context_payload` | reward 函数细分日志 |
| `reward/r_total_mean/max/min` | reward 函数细分日志 |
| `reward/response_len_mean` | reward 函数细分日志 |
| `reward/truncated_ratio` | reward 函数细分日志 |
| `train/entropy` | slime 训练日志 |
| `train/kl_loss` | slime 训练日志 |
| `train/pg_loss` | slime 训练日志 |
| `rollout/response_len_mean` | slime rollout 日志 |

## 模型 Checkpoint

| 用途 | 格式 | 路径 | 读写 |
|------|------|------|------|
| 冻结模型 + SGLang 推理 | HF (safetensors) | `/data/cloud_disk_4/.../iter_0000109_hf/` | **只读** |
| slime ref model | Megatron torch_dist (.distcp) | `/data/cloud_disk_4/.../iter_0000109/` | **只读** |
| 训练输出 | Megatron torch_dist | `outputs/{RUN_NAME}/checkpoints/` | 写入 |

## 环境变量

```bash
# rollout / reward 函数需要
export SNAPSHOT_DATA_ROOT="/path/to/slime_train/data/rl_data_test_2"
export RETRIEVE_SNAPSHOT_DATA_ROOT="${SNAPSHOT_DATA_ROOT}"
export MEMORY_RL_TASK_VERSION="t2_agent_loop"      # 或 atomic_code_t2
export MEMORY_RL_APPLY_MODE="task_loop"
export MEMORY_RL_TRAIN_TASK_LOOP="1"
export FROZEN_MODEL_URL="http://localhost:30100/v1/chat/completions"
export FROZEN_MODEL_NAME="default"
export FROZEN_MODEL_TIMEOUT="30"
export REWARD_METRICS_LOG="${LOG_DIR}/reward_metrics.jsonl"

# Python 路径
export PYTHONPATH=/root/Megatron-LM:${PROJECT_ROOT}:${WORKSPACE_SRC}
```

## 集群资源

8 节点 × 8 GPU = 64 GPU，训推分离：

| 方案 | 训练 | 推理 | 备注 |
|------|------|------|------|
| A（默认） | 4 节点 (32 GPU) | 4 节点 (32 GPU) | `--plan a` |
| B | 4 节点 (32 GPU) | 3 节点 (24 GPU) | `--plan b`，1 节点做调度 |

## 依赖

- **llm_gateway.rl_env**: `MemoryEnv` / `SnapshotSession` / snapshot serialize/restore
- **llm_gateway task implementations**: `atomic_code_t2` 与 `atomic_t2_agent_loop` 的 retrieve task
- **slime 框架**: Docker 容器内 `/root/slime`
- **Megatron-LM**: Docker 容器内 `/root/Megatron-LM`
- **SGLang**: 冻结模型推理引擎
- **SwanLab**: 训练监控

## 已知问题 & TODO

- [x] `snapshot_loader.py` 已迁移为 `rl_env.snapshot_session.SnapshotSession` 的兼容 wrapper，并统一通过 `decode_snapshot(encoded, fs, vec, graph)` 恢复
- [x] retrieve reward 已切换为基于 rollout context 评分，不再在 reward 阶段重复检索
- [x] 已增加本地单元测试覆盖 reward 解析/评分、转换样本与 SnapshotSession 基础加载链路
