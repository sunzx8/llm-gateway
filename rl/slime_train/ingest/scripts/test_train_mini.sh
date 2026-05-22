#!/bin/bash
# =============================================================================
# Ingest 摄入 RL 测试训练脚本 — 小规模验证能否跑通梯度更新
#
# 使用方法：在 head 节点容器内执行:
#   bash slime_train/ingest/scripts/test_train_mini.sh
#   bash slime_train/ingest/scripts/test_train_mini.sh --task-version agent_loop
#
# 精简参数（相比完整训练）:
#   - num-rollout: 10（仅跑 10 步）
#   - rollout-batch-size: 8
#   - n-samples-per-prompt: 4
#   - save-interval: 5（每 5 步保存一次）
#   - 使用现有 Ray 集群
# =============================================================================

set -e

# =============================================================================
# 参数解析
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --task-version) MEMORY_RL_TASK_VERSION="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# 路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SLIME_TRAIN_ROOT="${SLIME_TRAIN_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)}"
LLM_GATEWAY_ROOT="${LLM_GATEWAY_ROOT:-$(cd "${SLIME_TRAIN_ROOT}/../.." && pwd)}"
LLM_GATEWAY_PARENT="${LLM_GATEWAY_PARENT:-$(cd "${LLM_GATEWAY_ROOT}/.." && pwd)}"

INPUT_RL_DATA="${INPUT_RL_DATA:-/data/cloud_disk_1/changyuchen/memory_ai_RL/slime_train/data/rl_data_test_2/rl_data.jsonl}"
SNAPSHOT_DATA_ROOT="${SNAPSHOT_DATA_ROOT:-$(dirname "${INPUT_RL_DATA}")}"
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/ingest_train.jsonl}"

# Checkpoint（使用 RL 训练过的模型，非基座）
HF_CHECKPOINT="${HF_CHECKPOINT:-/data/cloud_disk_1/changyuchen/memory_ai_RL/model/iter_0000189_hf}"
TORCH_DIST_CHECKPOINT="${TORCH_DIST_CHECKPOINT:-/data/cloud_disk_1/changyuchen/memory_ai_RL/model/iter_0000189_torch_dist}"
FROZEN_MODEL_PATH="${FROZEN_MODEL_PATH:-/data/cloud_disk_1/changyuchen/memory_ai_RL/model/iter_0000189_hf}"
EXTERNAL_FROZEN_MODEL_URL="${FROZEN_MODEL_URL:-}"
FROZEN_PORT="${FROZEN_PORT:-30100}"
FROZEN_MODEL_URL="${FROZEN_MODEL_URL:-http://${MASTER_ADDR:-192.168.16.48}:${FROZEN_PORT}/v1/chat/completions}"
FROZEN_MODEL_NAME="${FROZEN_MODEL_NAME:-default}"
FROZEN_MODEL_TIMEOUT="${FROZEN_MODEL_TIMEOUT:-30}"
FROZEN_MODEL_MAX_CONCURRENCY="${FROZEN_MODEL_MAX_CONCURRENCY:-32}"
FROZEN_MODEL_CONN_LIMIT="${FROZEN_MODEL_CONN_LIMIT:-128}"
PROBE_EVAL_MAX_CONCURRENCY="${PROBE_EVAL_MAX_CONCURRENCY:-8}"
MEMORY_RL_MAX_AGENT_TURNS="${MEMORY_RL_MAX_AGENT_TURNS:-20}"
MEMORY_RL_TASK_VERSION="${MEMORY_RL_TASK_VERSION:-atomic_code_t2}"
MEMORY_RL_APPLY_MODE="${MEMORY_RL_APPLY_MODE:-tool_calls}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"
SLIME_ROOT="${SLIME_ROOT:-/root/slime}"
RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"

# 模型参数
MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-/data/cloud_disk_1/changyuchen/agent_memory_RLVR/slime_train/scripts/model/qwen3.5-35B-A3B.sh}"
source "${MODEL_CONFIG_SCRIPT}"

# 输出
RUN_NAME="ingest_test_$(date '+%Y%m%d_%H%M%S')"
CKPT_DIR="${PROJECT_ROOT}/outputs/${RUN_NAME}/checkpoints"
DUMP_DIR="${PROJECT_ROOT}/outputs/${RUN_NAME}/rollout_dumps"
LOG_DIR="${PROJECT_ROOT}/outputs/${RUN_NAME}/logs"

mkdir -p "${CKPT_DIR}" "${DUMP_DIR}" "${LOG_DIR}"

# 环境变量
export PYTHONUNBUFFERED=1
export PYTHONPATH="${MEGATRON_ROOT}:${LLM_GATEWAY_PARENT}:${LLM_GATEWAY_ROOT}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR="${MASTER_ADDR:-192.168.16.48}"
export no_proxy="${no_proxy:-127.0.0.1,${MASTER_ADDR}}"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-BVQaRTEEZKWC9p3iF5MMp}"

# Reward 函数需要的环境变量
export INGEST_SNAPSHOT_DATA_ROOT="${SNAPSHOT_DATA_ROOT}"
export SNAPSHOT_DATA_ROOT="${SNAPSHOT_DATA_ROOT}"
export FROZEN_MODEL_URL FROZEN_MODEL_NAME FROZEN_MODEL_TIMEOUT FROZEN_MODEL_MAX_CONCURRENCY FROZEN_MODEL_CONN_LIMIT PROBE_EVAL_MAX_CONCURRENCY
export REWARD_METRICS_LOG="${LOG_DIR}/reward_metrics.jsonl"
export MEMORY_RL_MAX_AGENT_TURNS MEMORY_RL_TASK_VERSION MEMORY_RL_APPLY_MODE

echo "============================================"
echo "Ingest RL 测试训练 (mini)"
echo "============================================"
echo "RUN_NAME:      ${RUN_NAME}"
echo "TRAIN_DATA:    ${TRAIN_DATA}"
echo "HF_CHECKPOINT: ${HF_CHECKPOINT}"
echo "CKPT_DIR:      ${CKPT_DIR}"
echo "============================================"

# Step 0: 生成训练数据
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "=== Step 0: 生成 ingest 训练数据 ==="
    python3 "${PROJECT_ROOT}/convert_to_slime_format.py" \
        --input "${INPUT_RL_DATA}" \
        --data-root "${SNAPSHOT_DATA_ROOT}" \
        --output "${TRAIN_DATA}" \
        --eval_output "${PROJECT_ROOT}/data/ingest_val.jsonl"
    echo "训练数据生成完成"
fi

if [ ! -f "${TRAIN_DATA}" ]; then
    echo "ERROR: 训练数据不存在: ${TRAIN_DATA}"
    exit 1
fi

# Reward QA 模型
echo ""
echo "=== Step 1: 准备 reward QA 模型 ==="

if [ -n "${EXTERNAL_FROZEN_MODEL_URL}" ]; then
    echo "使用外部 reward QA endpoint: ${FROZEN_MODEL_URL}"
elif curl -s "http://localhost:${FROZEN_PORT}/health" > /dev/null 2>&1; then
    echo "冻结模型端口 ${FROZEN_PORT} 已有服务运行，复用"
else
    echo "启动冻结模型 (SGLang, TP=2, port=${FROZEN_PORT})..."
    python3 -m sglang.launch_server \
        --model-path "${FROZEN_MODEL_PATH}" \
        --host 0.0.0.0 \
        --port ${FROZEN_PORT} \
        --tp 2 \
        --mem-fraction-static 0.5 \
        --max-running-requests 16 \
        --trust-remote-code \
        > "${LOG_DIR}/frozen_model.log" 2>&1 &
    FROZEN_PID=$!
    echo "冻结模型 PID=${FROZEN_PID}, 等待就绪..."

    for i in $(seq 1 60); do
        if curl -s "http://localhost:${FROZEN_PORT}/health" > /dev/null 2>&1; then
            echo "冻结模型就绪! (${i}x5s)"
            break
        fi
        if ! kill -0 ${FROZEN_PID} 2>/dev/null; then
            echo "ERROR: 冻结模型进程退出"
            tail -20 "${LOG_DIR}/frozen_model.log"
            exit 1
        fi
        sleep 5
    done

    if ! curl -s "http://localhost:${FROZEN_PORT}/health" > /dev/null 2>&1; then
        echo "ERROR: 冻结模型启动超时"
        kill ${FROZEN_PID} 2>/dev/null
        exit 1
    fi
fi

echo ""
echo "=== Step 2: 启动 SwanLab 监控 ==="
SWANLAB_SCRIPT="${SCRIPT_DIR}/swanlab_monitor.py"
if [ -f "${SWANLAB_SCRIPT}" ]; then
    nohup python3 "${SWANLAB_SCRIPT}" \
        --log-dir "${LOG_DIR}" \
        --task-version "${MEMORY_RL_TASK_VERSION}" \
        --poll-interval 5.0 \
        > "${LOG_DIR}/swanlab_monitor.log" 2>&1 &
    echo "SwanLab 监控已启动 (PID=$!)"
else
    echo "WARN: SwanLab 监控脚本不存在: ${SWANLAB_SCRIPT}"
fi

echo ""
echo "=== Step 3: 启动 GRPO 训练 (mini test) ==="

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_ROOT}:${LLM_GATEWAY_PARENT}:${LLM_GATEWAY_ROOT}\",
    \"PYTHONUNBUFFERED\": \"1\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"NCCL_NVLS_ENABLE\": \"1\",
    \"NCCL_IB_DISABLE\": \"0\",
    \"NCCL_IB_GID_INDEX\": \"3\",
    \"NCCL_IB_HCA\": \"mlx5\",
    \"NCCL_NET_GDR_LEVEL\": \"5\",
    \"NCCL_TIMEOUT_MS\": \"600000\",
    \"NCCL_SOCKET_IFNAME\": \"eth0\",
    \"GLOO_SOCKET_IFNAME\": \"eth0\",
    \"NCCL_DEBUG\": \"WARN\",
    \"TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC\": \"600\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"no_proxy\": \"${no_proxy}\",
    \"INGEST_SNAPSHOT_DATA_ROOT\": \"${SNAPSHOT_DATA_ROOT}\",
    \"SNAPSHOT_DATA_ROOT\": \"${SNAPSHOT_DATA_ROOT}\",
    \"FROZEN_MODEL_URL\": \"${FROZEN_MODEL_URL}\",
    \"FROZEN_MODEL_NAME\": \"${FROZEN_MODEL_NAME}\",
    \"FROZEN_MODEL_TIMEOUT\": \"${FROZEN_MODEL_TIMEOUT}\",
    \"FROZEN_MODEL_MAX_CONCURRENCY\": \"${FROZEN_MODEL_MAX_CONCURRENCY}\",
    \"FROZEN_MODEL_CONN_LIMIT\": \"${FROZEN_MODEL_CONN_LIMIT}\",
    \"PROBE_EVAL_MAX_CONCURRENCY\": \"${PROBE_EVAL_MAX_CONCURRENCY}\",
    \"REWARD_METRICS_LOG\": \"${LOG_DIR}/reward_metrics.jsonl\",
    \"MEMORY_RL_MAX_AGENT_TURNS\": \"${MEMORY_RL_MAX_AGENT_TURNS}\",
    \"MEMORY_RL_TASK_VERSION\": \"${MEMORY_RL_TASK_VERSION}\",
    \"MEMORY_RL_APPLY_MODE\": \"${MEMORY_RL_APPLY_MODE}\"
  }
}"

cd "${SLIME_ROOT}"

ray job submit --address="${RAY_ADDRESS}" \
    --runtime-env-json="$RUNTIME_ENV_JSON" \
    -- python3 "${SLIME_ROOT}/train_async.py" \
    --actor-num-nodes 4 \
    --actor-num-gpus-per-node 8 \
    --rollout-num-gpus 24 \
    --rollout-num-gpus-per-engine 2 \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --ref-load "${TORCH_DIST_CHECKPOINT}" \
    --load "${CKPT_DIR}" \
    --save "${CKPT_DIR}" \
    --save-interval 5 \
    --prompt-data "${TRAIN_DATA}" \
    --input-key prompt \
    --metadata-key metadata \
    --apply-chat-template \
    --custom-rm-path llm_gateway.rl.slime_train.tasks.ingest_reward.reward.reward_func \
    --custom-generate-function-path llm_gateway.rl.slime_train.memory_rl.custom_generate.custom_generate \
    --num-rollout 10 \
    --rollout-batch-size 8 \
    --n-samples-per-prompt 4 \
    --rollout-max-response-len 4096 \
    --rollout-max-prompt-len 8192 \
    --rollout-temperature 1.0 \
    --global-batch-size 16 \
    --balance-data \
    --tensor-model-parallel-size 2 \
    --sequence-parallel \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 8 \
    --expert-tensor-parallel-size 1 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu 4096 \
    --advantage-estimator grpo \
    --use-kl-loss \
    --kl-loss-coef 0.005 \
    --kl-loss-type low_var_kl \
    --entropy-coef 0.0008 \
    --eps-clip 0.2 \
    --eps-clip-high 0.28 \
    --clip-grad 1.0 \
    --optimizer adam \
    --lr 5e-7 \
    --lr-decay-style cosine \
    --lr-warmup-iters 2 \
    --min-lr 1e-7 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    --sglang-mem-fraction-static 0.5 \
    --sglang-disable-custom-all-reduce \
    --sglang-disable-cuda-graph \
    --sglang-watchdog-timeout 1200 \
    --sglang-router-request-timeout-secs 300 \
    --no-check-for-nan-in-loss-and-grad \
    --distributed-timeout-minutes 60 \
    --no-load-optim \
    --no-load-rng \
    --finetune \
    --dump-details "${DUMP_DIR}" \
    2>&1 | tee "${LOG_DIR}/train.log"
