# T3 Ingest RLVR

This package trains the T3 ingest policy to emit JSON-serialized tool calls that
are executed by `MemoryEnv.apply_ingest_tool_calls()`.

## Data

Input dataset layout follows `rl_data_test_2/` or legacy `ingest_full_natural_probes_test_1/`:

- `rl_data.jsonl`: pre-ingest snapshot records with `pending_messages` and query probes
- `ingest_snapshots.jsonl`: snapshot index
- `snapshots/{traj_id}/{snapshot_id}.cbsnap`: pre-ingest env snapshots

Probe selection prefers `probes_by_task.ingest`. For legacy data without `task_target`, all `probes` are treated as ingest probes.

Convert data:

```bash
python slime_train/ingest/convert_to_slime_format.py \
  --input ingest_full_natural_probes_test_1/rl_data.jsonl \
  --data-root ingest_full_natural_probes_test_1 \
  --output slime_train/ingest/data/ingest_train.jsonl \
  --eval_output slime_train/ingest/data/ingest_val.jsonl
```

## Reward

Custom reward path:

```text
slime_train.tasks.ingest_reward.reward.reward_func
```

Reward composition:

- `0.90` hidden query-probe QA score after applying generated write ops
- `0.10` JSON/tool-call format score

Probe scoring is always `retrieve then answer`: after applying writes, each probe calls the currently configured query task (`env.step_query`) to get context, then a frozen QA model answers from that context. This means `MEMORY_RL_TASK_VERSION=atomic_code_t2` evaluates atomic retrieve, while `MEMORY_RL_TASK_VERSION=t2_agent_loop` evaluates the retrieve agent loop. Use `MEMORY_RL_RETRIEVE_LLM_API_URL` / `MEMORY_RL_RETRIEVE_LLM_MODEL` to point this probe-time query task at an external frozen model.

Run training wrapper:

```bash
SNAPSHOT_DATA_ROOT=/path/to/rl_data_test_2 \
FROZEN_MODEL_URL=http://<reward-vllm-host>:8000/v1/chat/completions \
FROZEN_MODEL_NAME=<qa-model> \
slime_train/ingest/scripts/train_ingest_grpo.sh --help
```

`FROZEN_MODEL_URL` is optional. If set, reward QA uses that external OpenAI-compatible/vLLM endpoint; the training script does not need to launch a local frozen model.

## Local vLLM Smoke Test Without Ray

```bash
python slime_train/memory_rl/smoke_vllm_rollout_reward.py \
  --task ingest \
  --data slime_train/ingest/data/ingest_train.jsonl \
  --snapshot-data-root /path/to/rl_data_test_2 \
  --api-url http://<vllm-host>:8000/v1/chat/completions \
  --model <policy-model> \
  --frozen-model-url http://<reward-vllm-host>:8000/v1/chat/completions \
  --frozen-model-name <qa-model> \
  --limit 4 \
  --concurrency 2
```
