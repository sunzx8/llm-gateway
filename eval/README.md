# 评测脚本使用说明

## 概述

`eval_script.py` 是基于 Context Task（IngestContextTask + RetrieveContextTask）的端到端评测 Runner。它通过调用 LLM Gateway 接口完成记忆摄入和检索问答，并计算评测指标。

## 前置条件

### 1. 安装依赖

```bash
pip3 install -r requirements.txt
```

### 2. 启动 LLM Gateway 服务

评测脚本通过 HTTP 接口调用 Gateway 服务，因此需要先启动服务：

```bash
python3 main.py -c config.yaml --host 127.0.0.1 --port 8000
```

### 3. 准备 Benchmark 数据

目前支持的 benchmark：

- **personamem**：PersonaMem benchmark，评测 LLM 的动态用户画像和个性化回答能力
  - 数据来源：https://huggingface.co/datasets/bowen-upenn/PersonaMem
  - 数据格式：`questions_{split}.csv` + `shared_contexts_{split}.jsonl`
  - 支持的 split：`32k`、`128k`、`1M`

## 运行方式

`eval_script.py` 内置了 CLI 入口，可以直接通过 `python` 命令执行。

### 基本用法

在项目根目录下执行：

```bash
python3 -m eval.eval_script --benchmarks personamem --data-dir data/personamem --split 32k
```

或者直接进入 eval 目录执行：

```bash
cd eval
python3 eval_script.py --benchmarks personamem --data-dir ../data/personamem --split 32k
```

### 快速测试（小规模）

```bash
python3 -m eval.eval_script \
  --benchmarks personamem \
  --data-dir data/personamem \
  --split 32k \
  --subset-size 2 \
  --conv-concurrency 2 \
  --no-judge
```

### 全量评测

```bash
python3 -m eval.eval_script \
  --benchmarks personamem \
  --data-dir data/personamem \
  --split 32k \
  --conv-concurrency 20
```

### 指定 Gateway 地址和模型

```bash
python3 -m eval.eval_script \
  --benchmarks personamem \
  --data-dir data/personamem \
  --gateway-base-url http://10.0.0.1:8000 \
  --gateway-api-key sk-your-key \
  --model gpt-5.4 \
  --ingest-model memory-initialize
```

### 完整命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--benchmarks` | `personamem` | 要评测的 benchmark 名称，多个用逗号分隔 |
| `--data-dir` | `data/personamem` | Benchmark 数据目录路径 |
| `--split` | `32k` | PersonaMem 数据集 split（32k / 128k / 1M） |
| `--subset-size` | 不指定=全量 | 每个 benchmark 取前 N 条 conversation |
| `--conv-concurrency` | `20` | 同一 benchmark 内并行处理的 conversation 数量 |
| `--benchmark-concurrency` | `1` | 同时运行的 benchmark 数量 |
| `--model` | `gpt-5.4` | 用于 retrieve + 回答的 LLM 模型名称 |
| `--ingest-model` | `memory-initialize` | 摄入接口使用的模型名称 |
| `--gateway-base-url` | `http://127.0.0.1:8000` | LLM Gateway 服务地址 |
| `--gateway-api-key` | `sk-my-test-key-123` | Gateway 接口的 API Key |
| `--eval-run-id` | 自动生成时间戳 | 本次评测的时间标识，用作 user_id 后缀 |
| `--judge-model` | `gpt-5.5` | LLM Judge 评分模型 |
| `--judge-base-url` | Azure endpoint | Judge 模型的 API 地址 |
| `--judge-api-key` | - | Judge 模型的 API Key |
| `--no-judge` | `False` | 跳过 LLM Judge 评分 |
| `--log-level` | `INFO` | 日志级别（DEBUG/INFO/WARNING/ERROR） |

### 查看帮助

```bash
python3 -m eval.eval_script --help
```

## 评测流程

```
┌─────────────────────────────────────────────────────────┐
│                   ContextTaskRunner                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  对每个 Conversation:                                     │
│                                                          │
│  1. Ingest Phase (记忆摄入)                               │
│     ├── 调用 POST /llm/v1/chat/completions               │
│     │   model="memory-initialize"                        │
│     │   messages=session_messages                        │
│     └── 触发 consolidate（达到阈值时）                     │
│                                                          │
│  2. Answer Phase (检索问答)                               │
│     ├── 调用 POST /llm/v1/chat/completions               │
│     │   model="{config.model}"                           │
│     │   messages=[{"role":"user","content":question}]    │
│     └── 接口内部自动完成 retrieve + LLM 回答              │
│                                                          │
│  3. Evaluate (评估)                                       │
│     ├── 计算 MCQ accuracy 等指标                          │
│     └── (可选) LLM-as-Judge 评分                          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## 输出结果说明

评测结果 `ContextTaskBenchmarkResult` 包含：

- **metrics**：评测指标（如 `mcq_accuracy`、各 question_type 的准确率）
- **judge_scores**：LLM Judge 评分列表
- **stats**：benchmark 级别的任务统计，包括：
  - 各类型任务（ingest/retrieve/consolidate）的总次数、总耗时、LLM 耗时、token 消耗、工具调用次数等
  - 各类型任务的最大单次耗时、最大 LLM 耗时等
  - 全部任务的汇总统计
- **per_conversation_results**：每个 conversation 的详细结果，包含完整的 `task_results`（含 `task_type`、`input_params`、`result`），方便排查问题

## 注意事项

1. 评测会为每个 user_id 拼接 `eval_run_id` 后缀（默认为当前时间戳），确保不同评测轮次的数据隔离
2. `conv_concurrency` 设置过高可能导致 Gateway 服务压力过大，建议根据服务器性能调整
3. 确保 Gateway 配置中 `memory_config.for_evaluation` 设为 `true`，以避免 on_response 阶段重复 ingest

## Dashboard 查看评测结果

评测完成后，结果会自动保存为 JSON 文件到 `eval/results/` 目录。可以通过内置的 Dashboard 可视化面板查看。

### 启动 Dashboard

在项目根目录下执行：

```bash
python3 -m eval.dashboard.server
```

启动后浏览器访问：**http://localhost:8088**

### 命令行参数

```bash
python3 -m eval.dashboard.server --port 8080 --results-dir eval/results --log-level DEBUG
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | `8088` | Dashboard 服务端口 |
| `--results-dir` | `eval/results` | 评测结果 JSON 文件所在目录 |
| `--log-level` | `INFO` | 日志级别（DEBUG/INFO/WARNING/ERROR） |

### 页面交互说明

Dashboard 采用三级层级交互设计：

```
第一层：Benchmark 概览
  ├── 全局统计（总 benchmark 数、总耗时等）
  └── 各 benchmark 卡片（metrics、耗时、错误统计）
        │
        ▼ 点击某个 benchmark
第二层：Conversation 列表
  ├── 任务统计表格（ingest/retrieve/consolidate 各类指标）
  └── 可搜索、分页的 conversation 列表
        │
        ▼ 点击某个 conversation
第三层：Conversation 详情
  ├── QA 对比面板（预期答案 vs 实际回答）
  └── 任务执行详情（各步骤耗时、token 消耗等）
```

## Full Context 评测脚本（Baseline 对比）

`eval_full_context.py` 直接将完整的对话上下文（所有 session 消息）作为聊天历史传给 LLM 回答问题，不经过 memory 的 ingest/retrieve 流程。用于和 memory 方案进行对比测试。

### 基本用法

```bash
python3 -m eval.eval_full_context \
  --benchmarks personamem \
  --model gpt-5.4 \
  --conv-concurrency 10 \
  --subset-size 5
```

### 快速测试（小规模，跳过 Judge）

```bash
python3 -m eval.eval_full_context \
  --benchmarks personamem \
  --split 32k \
  --subset-size 2 \
  --conv-concurrency 2 \
  --no-judge
```

### 指定 LLM 地址和模型

```bash
python3 -m eval.eval_full_context \
  --benchmarks personamem \
  --model gpt-5.4 \
  --provider openai \
  --base-url https://api.openai.com/v1 \
  --api-key sk-your-key \
  --conv-concurrency 10
```

### 完整命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--benchmarks` | `personamem` | 要评测的 benchmark 名称，多个用逗号分隔 |
| `--data-dir` | `eval/benchmarks/data/personamem` | Benchmark 数据目录路径 |
| `--split` | `32k` | PersonaMem 数据集 split（32k / 128k / 1M） |
| `--subset-size` | 不指定=全量 | 每个 benchmark 取前 N 条 conversation |
| `--conv-concurrency` | `10` | 同一 benchmark 内并行处理的 conversation 数量 |
| `--model` | `gpt-5.4` | 用于回答问题的 LLM 模型名称 |
| `--provider` | `openai` | LLM provider（openai / openai_compat / anthropic） |
| `--base-url` | `None` | LLM API 的 base URL |
| `--api-key` | `None` | LLM API Key |
| `--temperature` | `0.0` | LLM 采样温度 |
| `--max-tokens` | `4096` | LLM 最大生成 token 数 |
| `--eval-run-id` | 自动生成 `fc_` 前缀时间戳 | 本次评测的时间标识 |
| `--judge-model` | `gpt-5.5` | LLM Judge 评分模型 |
| `--judge-base-url` | Azure endpoint | Judge 模型的 API 地址 |
| `--judge-api-key` | - | Judge 模型的 API Key |
| `--no-judge` | `False` | 跳过 LLM Judge 评分 |
| `--log-level` | `INFO` | 日志级别（DEBUG/INFO/WARNING/ERROR） |

### 评测流程

```
┌─────────────────────────────────────────────────────────┐
│                  FullContextRunner                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  对每个 Conversation:                                     │
│                                                          │
│  1. 收集所有 session 消息（保留原始 role）                  │
│                                                          │
│  2. Answer Phase (直接回答)                               │
│     ├── messages = 全部 context 消息                      │
│     │   + [{"role":"user","content": question}]          │
│     ├── system 消息转为 user 消息（兼容性处理）            │
│     └── 直接调用 LLM 生成回答                             │
│                                                          │
│  3. Evaluate (评估)                                       │
│     ├── 计算 MCQ accuracy 等指标                          │
│     └── (可选) LLM-as-Judge 评分                          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Mem0 评测脚本（Baseline 对比）

`eval_mem0.py` 使用 [mem0](https://github.com/mem0ai/mem0) 开源记忆框架作为记忆后端，评测流程与 memory 服务对齐（ingest → search → answer），用于和当前 memory 服务、full context baseline 进行对比测试。

### 前置条件

```bash
pip install mem0ai
```

### 基本用法

```bash
python3 -m eval.eval_mem0 \
  --benchmarks personamem \
  --model gpt-5.4 \
  --mem0-model gpt-5.4 \
  --conv-concurrency 5 \
  --subset-size 5
```

### 快速测试（小规模，跳过 Judge）

```bash
python3 -m eval.eval_mem0 \
  --benchmarks personamem \
  --split 32k \
  --subset-size 2 \
  --conv-concurrency 2 \
  --no-judge
```

### 指定 LLM 和 Mem0 配置

```bash
python3 -m eval.eval_mem0 \
  --benchmarks personamem \
  --model gpt-5.4 \
  --provider openai \
  --base-url https://api.openai.com/v1 \
  --api-key sk-your-key \
  --mem0-model gpt-5.4 \
  --mem0-provider openai \
  --mem0-base-url https://api.openai.com/v1 \
  --mem0-api-key sk-your-key \
  --mem0-embedding-model text-embedding-3-small \
  --mem0-search-limit 20 \
  --conv-concurrency 5
```

### 完整命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| **通用参数** | | |
| `--benchmarks` | `personamem` | 要评测的 benchmark 名称，多个用逗号分隔 |
| `--data-dir` | `eval/benchmarks/data/personamem` | Benchmark 数据目录路径 |
| `--split` | `32k` | PersonaMem 数据集 split（32k / 128k / 1M） |
| `--subset-size` | 不指定=全量 | 每个 benchmark 取前 N 条 conversation |
| `--conv-concurrency` | `5` | 并行处理的 conversation 数量（mem0 建议不要太高） |
| **LLM 配置（回答问题）** | | |
| `--model` | `gpt-5.4` | 用于回答问题的 LLM 模型名称 |
| `--provider` | `openai` | LLM provider（openai / openai_compat / anthropic） |
| `--base-url` | `None` | LLM API 的 base URL |
| `--api-key` | `None` | LLM API Key |
| `--temperature` | `0.0` | LLM 采样温度 |
| `--max-tokens` | `4096` | LLM 最大生成 token 数 |
| **Mem0 配置** | | |
| `--mem0-provider` | `openai` | Mem0 使用的 LLM provider |
| `--mem0-model` | `gpt-5.4` | Mem0 使用的 LLM 模型（用于记忆提取和摘要） |
| `--mem0-base-url` | `None` | Mem0 LLM API 的 base URL |
| `--mem0-api-key` | `None` | Mem0 LLM API Key |
| `--mem0-embedding-model` | `text-embedding-3-small` | Mem0 使用的 embedding 模型 |
| `--mem0-embedding-base-url` | `None` | Mem0 embedding API 的 base URL |
| `--mem0-embedding-api-key` | `None` | Mem0 embedding API Key |
| `--mem0-search-limit` | `20` | mem0.search() 返回的最大记忆条数 |
| **评测标识** | | |
| `--eval-run-id` | 自动生成 `mem0_` 前缀时间戳 | 本次评测的时间标识 |
| **Judge 配置** | | |
| `--judge-model` | `gpt-5.5` | LLM Judge 评分模型 |
| `--judge-base-url` | Azure endpoint | Judge 模型的 API 地址 |
| `--judge-api-key` | - | Judge 模型的 API Key |
| `--no-judge` | `False` | 跳过 LLM Judge 评分 |
| `--log-level` | `INFO` | 日志级别（DEBUG/INFO/WARNING/ERROR） |

### 评测流程

```
┌─────────────────────────────────────────────────────────┐
│                     Mem0Runner                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  对每个 Conversation:                                     │
│                                                          │
│  1. Ingest Phase (记忆摄入)                               │
│     ├── 对每个 session 的消息调用 mem0.add()               │
│     └── mem0 自动提取和存储记忆                            │
│                                                          │
│  2. Search + Answer Phase (检索 + 回答)                   │
│     ├── 调用 mem0.search() 检索相关记忆                   │
│     ├── 将记忆作为上下文 + 问题发送给 LLM                  │
│     └── answer_elapsed = search + llm 完整耗时            │
│                                                          │
│  3. Evaluate (评估)                                       │
│     ├── 计算 MCQ accuracy 等指标                          │
│     └── (可选) LLM-as-Judge 评分                          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Dashboard 三种模式对比

Dashboard 支持同时展示三种评测模式的结果：

| 模式 | 标识 | 脚本 | 说明 |
|------|------|------|------|
| 🧠 Memory | `eval_mode: "memory"` | `eval_script.py` | 当前 memory 服务（ingest + retrieve） |
| 🔤 Full Context | `eval_mode: "full_context"` | `eval_full_context.py` | 直接传入完整上下文 |
| 🔶 Mem0 | `eval_mode: "mem0"` | `eval_mem0.py` | mem0 记忆框架 |

---

### 典型使用流程

```bash
# 1. 启动 Gateway 服务
python3 main.py -c config.yaml --host 127.0.0.1 --port 8000

# 2. 运行 Memory 服务评测
python3 -m eval.eval_script \
  --benchmarks personamem \
  --data-dir data/personamem \
  --split 32k \
  --conv-concurrency 20

# 3. 运行 Full Context 评测（Baseline）
python3 -m eval.eval_full_context \
  --benchmarks personamem \
  --model gpt-5.4 \
  --conv-concurrency 10

# 4. 运行 Mem0 评测（Baseline）
python3 -m eval.eval_mem0 \
  --benchmarks personamem \
  --model gpt-5.4 \
  --mem0-model gpt-5.4 \
  --conv-concurrency 5

# 5. 启动 Dashboard 查看对比结果
python3 -m eval.dashboard.server

# 6. 浏览器打开 http://localhost:8088
```
