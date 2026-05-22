#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLIME_TRAIN_ROOT="${SLIME_TRAIN_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LLM_GATEWAY_ROOT="${LLM_GATEWAY_ROOT:-$(cd "${SLIME_TRAIN_ROOT}/../.." && pwd)}"
LLM_GATEWAY_PARENT="${LLM_GATEWAY_PARENT:-$(cd "${LLM_GATEWAY_ROOT}/.." && pwd)}"

INPUT_RL_DATA="${INPUT_RL_DATA:-${LLM_GATEWAY_ROOT}/data/rl_data_test_2/rl_data.jsonl}"
SNAPSHOT_DATA_ROOT="${SNAPSHOT_DATA_ROOT:-$(dirname "${INPUT_RL_DATA}")}"
MIXED_TASKS="${MIXED_TASKS:-ingest+consolidate+retrieve}"
TRAIN_DATA="${TRAIN_DATA:-${SLIME_TRAIN_ROOT}/memory_rl/data/mixed_${MIXED_TASKS//+/_}_train.jsonl}"
EVAL_DATA="${VAL_DATA:-${EVAL_DATA:-${SLIME_TRAIN_ROOT}/memory_rl/data/mixed_${MIXED_TASKS//+/_}_val.jsonl}}"

MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"
SLIME_TRAIN_SCRIPT="${SLIME_TRAIN_SCRIPT:-/root/slime/train_async.py}"
MODEL_CONFIG_ARGS="${MODEL_CONFIG_ARGS:-}"
CUSTOM_RM_PATH="${CUSTOM_RM_PATH:-llm_gateway.rl.slime_train.tasks.mixed_reward.reward.reward_func}"
CUSTOM_GENERATE_FUNCTION_PATH="${CUSTOM_GENERATE_FUNCTION_PATH:-llm_gateway.rl.slime_train.memory_rl.custom_generate.custom_generate}"

MEMORY_RL_MAX_AGENT_TURNS="${MEMORY_RL_MAX_AGENT_TURNS:-4}"
MEMORY_RL_TASK_VERSION="${MEMORY_RL_TASK_VERSION:-atomic_code_t2}"
MEMORY_RL_APPLY_MODE="${MEMORY_RL_APPLY_MODE:-tool_calls}"
MEMORY_RL_TRAIN_TASK_LOOP="${MEMORY_RL_TRAIN_TASK_LOOP:-1}"
FROZEN_MODEL_URL="${FROZEN_MODEL_URL:-}"
FROZEN_MODEL_NAME="${FROZEN_MODEL_NAME:-default}"
FROZEN_MODEL_TIMEOUT="${FROZEN_MODEL_TIMEOUT:-30}"
FROZEN_MODEL_MAX_CONCURRENCY="${FROZEN_MODEL_MAX_CONCURRENCY:-32}"
FROZEN_MODEL_CONN_LIMIT="${FROZEN_MODEL_CONN_LIMIT:-128}"
PROBE_EVAL_MAX_CONCURRENCY="${PROBE_EVAL_MAX_CONCURRENCY:-8}"

mkdir -p "$(dirname "${TRAIN_DATA}")" "$(dirname "${EVAL_DATA}")"
if [ ! -f "${TRAIN_DATA}" ] || [ ! -f "${EVAL_DATA}" ]; then
  echo "[memory-rl] train/val not fully present; building mixed data"
  python3 "${SCRIPT_DIR}/build_mixed_data.py" \
    --input "${INPUT_RL_DATA}" \
    --data-root "${SNAPSHOT_DATA_ROOT}" \
    --tasks "${MIXED_TASKS}" \
    --output "${TRAIN_DATA}" \
    --eval-output "${EVAL_DATA}"
else
  echo "[memory-rl] using pre-split train/val data"
fi
python3 "${SCRIPT_DIR}/summarize_jsonl.py" "${TRAIN_DATA}" --label train
python3 "${SCRIPT_DIR}/summarize_jsonl.py" "${EVAL_DATA}" --label val

export PYTHONPATH="${MEGATRON_ROOT}:${LLM_GATEWAY_PARENT}:${LLM_GATEWAY_ROOT}:${PYTHONPATH:-}"
export SNAPSHOT_DATA_ROOT
export INGEST_SNAPSHOT_DATA_ROOT="${SNAPSHOT_DATA_ROOT}"
export CONSOLIDATE_SNAPSHOT_DATA_ROOT="${SNAPSHOT_DATA_ROOT}"
export RETRIEVE_SNAPSHOT_DATA_ROOT="${SNAPSHOT_DATA_ROOT}"
if [ -n "${FROZEN_MODEL_URL}" ]; then
  export FROZEN_MODEL_URL
fi
export FROZEN_MODEL_NAME FROZEN_MODEL_TIMEOUT FROZEN_MODEL_MAX_CONCURRENCY FROZEN_MODEL_CONN_LIMIT PROBE_EVAL_MAX_CONCURRENCY
export MEMORY_RL_MAX_AGENT_TURNS MEMORY_RL_TASK_VERSION MEMORY_RL_APPLY_MODE MEMORY_RL_TRAIN_TASK_LOOP

exec python3 "${SLIME_TRAIN_SCRIPT}" \
  --custom-rm-path "${CUSTOM_RM_PATH}" \
  --custom-generate-function-path "${CUSTOM_GENERATE_FUNCTION_PATH}" \
  --train-data "${TRAIN_DATA}" \
  --eval-data "${EVAL_DATA}" \
  ${MODEL_CONFIG_ARGS} \
  "$@"
