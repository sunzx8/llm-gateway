# T3 Consolidate / Evolve RLVR

This package trains the T3 memory evolution policy to emit JSON-serialized
consolidation tool calls executed by `MemoryEnv.apply_consolidate_tool_calls()`.

## Data

The converter builds evolution samples from query probes. It first uses explicit
`probes_by_task.consolidate` on each snapshot record (`rl_data_test_2` style).
If a legacy ingest-only dataset has no consolidate probes, it falls back to using
adjacent ingest probes as a best-effort evolution objective.

Convert data:

```bash
python slime_train/consolidate/convert_to_slime_format.py \
  --input ingest_full_natural_probes_test_1/rl_data.jsonl \
  --data-root ingest_full_natural_probes_test_1 \
  --output slime_train/consolidate/data/consolidate_train.jsonl \
  --eval_output slime_train/consolidate/data/consolidate_val.jsonl
```

## Reward

Custom reward path:

```text
slime_train.tasks.consolidate_reward.reward.reward_func
```

Reward is dominated by before/after hidden query-probe behavior:

- `0.50` positive probe correctness-rate delta
- `0.25` positive per-probe QA score delta
- `0.15` after-evolution probe QA score
- `0.05` retrieval context diff score, gated by non-regression
- `0.05` JSON/tool-call format score

Probe scoring is always `retrieve then answer`: before and after evolution, each probe calls the currently configured query task (`env.step_query`) to get context, then a frozen QA model answers from that context. This evaluates the selected query implementation itself (`atomic_code_t2` rewrite+return or `t2_agent_loop` agentic retrieve). Use `MEMORY_RL_RETRIEVE_LLM_API_URL` / `MEMORY_RL_RETRIEVE_LLM_MODEL` for the external frozen model used by probe-time retrieval.

Run training wrapper:

```bash
SNAPSHOT_DATA_ROOT=/path/to/rl_data_test_2 \
FROZEN_MODEL_URL=http://<reward-vllm-host>:8000/v1/chat/completions \
FROZEN_MODEL_NAME=<qa-model> \
slime_train/consolidate/scripts/train_consolidate_grpo.sh --help
```

`FROZEN_MODEL_URL` is optional. If set, reward QA uses that external OpenAI-compatible/vLLM endpoint; the training script does not need to launch a local frozen model.

## Local vLLM Smoke Test Without Ray

```bash
python slime_train/memory_rl/smoke_vllm_rollout_reward.py \
  --task consolidate \
  --data slime_train/consolidate/data/consolidate_train.jsonl \
  --snapshot-data-root /path/to/rl_data_test_2 \
  --api-url http://<vllm-host>:8000/v1/chat/completions \
  --model <policy-model> \
  --frozen-model-url http://<reward-vllm-host>:8000/v1/chat/completions \
  --frozen-model-name <qa-model> \
  --limit 4 \
  --concurrency 2
```
