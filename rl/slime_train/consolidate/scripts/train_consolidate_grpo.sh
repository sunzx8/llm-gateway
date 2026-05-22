#!/usr/bin/env bash
# =============================================================================
# Consolidate (演进) 阶段 GRPO 训练脚本
#
# 流程:
#   1. 数据准备（convert_to_slime_format.py）
#   2. 启动冻结模型推理引擎（可选，也可使用外部 endpoint）
#   3. 启动 SwanLab 监控
#   4. 启动 slime GRPO 训练
#   5. 训练结束后关闭推理引擎
#
# 使用:
#   bash train_consolidate_grpo.sh [额外 slime 参数...]
#   FROZEN_MODEL_URL=http://host:8000/v1/chat/completions bash train_consolidate_grpo.sh  # 外部 endpoint
# =============================================================================

set -euo pipefail

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC}  $(date '+%H:%M:%S') $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date '+%H:%M:%S') $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $1"; }
log_step()  { echo -e "${CYAN}[STEP]${NC}  $(date '+%H:%M:%S') $1"; }

# =============================================================================
# 路径配置
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SLIME_TRAIN_ROOT="${SLIME_TRAIN_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)}"
LLM_GATEWAY_ROOT="${LLM_GATEWAY_ROOT:-$(cd "${SLIME_TRAIN_ROOT}/../.." && pwd)}"
LLM_GATEWAY_PARENT="${LLM_GATEWAY_PARENT:-$(cd "${LLM_GATEWAY_ROOT}/.." && pwd)}"

INPUT_RL_DATA="${INPUT_RL_DATA:-/data/home/trevzhang/projects/memory-ai-agent-workspace/ingest_full_natural_probes_test_1/rl_data.jsonl}"
SNAPSHOT_DATA_ROOT="${SNAPSHOT_DATA_ROOT:-$(dirname "${INPUT_RL_DATA}")}"
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/consolidate_train.jsonl}"
EVAL_DATA="${VAL_DATA:-${EVAL_DATA:-${PROJECT_ROOT}/data/consolidate_val.jsonl}}"

MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"
SLIME_TRAIN_SCRIPT="${SLIME_TRAIN_SCRIPT:-/root/slime/train_async.py}"
MODEL_CONFIG_ARGS="${MODEL_CONFIG_ARGS:-}"
CUSTOM_RM_PATH="${CUSTOM_RM_PATH:-llm_gateway.rl.slime_train.tasks.consolidate_reward.reward.reward_func}"
CUSTOM_GENERATE_FUNCTION_PATH="${CUSTOM_GENERATE_FUNCTION_PATH:-llm_gateway.rl.slime_train.memory_rl.custom_generate.custom_generate}"
MEMORY_RL_MAX_AGENT_TURNS="${MEMORY_RL_MAX_AGENT_TURNS:-50}"
MEMORY_RL_TASK_VERSION="${MEMORY_RL_TASK_VERSION:-atomic_code_t2}"
MEMORY_RL_APPLY_MODE="${MEMORY_RL_APPLY_MODE:-tool_calls}"
MEMORY_RL_TRAIN_TASK_LOOP="${MEMORY_RL_TRAIN_TASK_LOOP:-1}"
MEMORY_RL_LLM_API_URL="${MEMORY_RL_LLM_API_URL:-${ROLLOUT_MODEL_URL:-}}"
MEMORY_RL_LLM_MODEL="${MEMORY_RL_LLM_MODEL:-${ROLLOUT_MODEL_NAME:-}}"
MEMORY_RL_TOOL_EXECUTOR_PATH="${MEMORY_RL_TOOL_EXECUTOR_PATH:-}"

# =============================================================================
# 冻结模型配置
# =============================================================================
FROZEN_MODEL_PATH="${FROZEN_MODEL_PATH:-/data/cloud_disk_1/changyuchen/memory_ai_RL/model/iter_0000189_hf}"
FROZEN_MODEL_PORT="${FROZEN_MODEL_PORT:-30100}"
FROZEN_MODEL_URL="${FROZEN_MODEL_URL:-}"
FROZEN_MODEL_NAME="${FROZEN_MODEL_NAME:-default}"
FROZEN_MODEL_TIMEOUT="${FROZEN_MODEL_TIMEOUT:-30}"
FROZEN_MODEL_MAX_CONCURRENCY="${FROZEN_MODEL_MAX_CONCURRENCY:-32}"
FROZEN_MODEL_CONN_LIMIT="${FROZEN_MODEL_CONN_LIMIT:-128}"
PROBE_EVAL_MAX_CONCURRENCY="${PROBE_EVAL_MAX_CONCURRENCY:-8}"
FROZEN_MODEL_TP="${FROZEN_MODEL_TP:-2}"
FROZEN_MODEL_MEM_FRACTION="${FROZEN_MODEL_MEM_FRACTION:-0.85}"

# 唯一标识
RUN_NAME="consolidate_grpo_$(date '+%Y%m%d_%H%M%S')"

# 输出
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${RUN_NAME}"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "${LOG_DIR}" "$(dirname "${TRAIN_DATA}")"

# =============================================================================
# PYTHONPATH
# =============================================================================
export PYTHONPATH="${MEGATRON_ROOT}:${LLM_GATEWAY_PARENT}:${LLM_GATEWAY_ROOT}:${PYTHONPATH:-}"

# =============================================================================
# 环境变量（reward 函数使用）
# =============================================================================
export CONSOLIDATE_SNAPSHOT_DATA_ROOT="${SNAPSHOT_DATA_ROOT}"
export SNAPSHOT_DATA_ROOT="${SNAPSHOT_DATA_ROOT}"
export FROZEN_MODEL_NAME FROZEN_MODEL_TIMEOUT FROZEN_MODEL_MAX_CONCURRENCY FROZEN_MODEL_CONN_LIMIT PROBE_EVAL_MAX_CONCURRENCY
export MEMORY_RL_MAX_AGENT_TURNS MEMORY_RL_TASK_VERSION MEMORY_RL_APPLY_MODE MEMORY_RL_TRAIN_TASK_LOOP MEMORY_RL_LLM_API_URL MEMORY_RL_LLM_MODEL
export REWARD_METRICS_LOG="${LOG_DIR}/reward_metrics.jsonl"
if [ -n "${MEMORY_RL_TOOL_EXECUTOR_PATH}" ]; then
  export MEMORY_RL_TOOL_EXECUTOR_PATH
fi

# =============================================================================
# Step 0: 数据准备
# =============================================================================
log_step "=========================================="
log_step "Consolidate — GRPO 训练"
log_step "=========================================="
log_info "RUN_NAME:           ${RUN_NAME}"
log_info "TRAIN_DATA:         ${TRAIN_DATA}"
log_info "SNAPSHOT_DATA_ROOT: ${SNAPSHOT_DATA_ROOT}"

if [ ! -f "${TRAIN_DATA}" ]; then
  log_step "生成训练数据..."
  python3 "${PROJECT_ROOT}/convert_to_slime_format.py" \
    --input "${INPUT_RL_DATA}" \
    --data-root "${SNAPSHOT_DATA_ROOT}" \
    --output "${TRAIN_DATA}" \
    --eval_output "${EVAL_DATA}"
  log_info "训练数据生成完成"
else
  echo "[memory-rl] using pre-split train/val data"
fi
python3 "${SLIME_TRAIN_ROOT}/memory_rl/summarize_jsonl.py" "${TRAIN_DATA}" --label train
python3 "${SLIME_TRAIN_ROOT}/memory_rl/summarize_jsonl.py" "${EVAL_DATA}" --label val

if [ ! -f "${TRAIN_DATA}" ]; then
  log_error "训练数据不存在: ${TRAIN_DATA}"
  exit 1
fi

# =============================================================================
# Step 1: 启动冻结模型推理引擎
# =============================================================================
FROZEN_PID=""

start_frozen_model() {
    log_step "启动冻结模型推理引擎 (SGLang)..."
    log_info "  模型: ${FROZEN_MODEL_PATH}"
    log_info "  端口: ${FROZEN_MODEL_PORT}"
    log_info "  TP:   ${FROZEN_MODEL_TP}"

    if curl -s "http://localhost:${FROZEN_MODEL_PORT}/health" > /dev/null 2>&1; then
        log_warn "端口 ${FROZEN_MODEL_PORT} 已有服务运行，跳过启动"
        return 0
    fi

    FROZEN_LOG="${LOG_DIR}/frozen_model.log"

    python3 -m sglang.launch_server \
        --model-path "${FROZEN_MODEL_PATH}" \
        --host 0.0.0.0 \
        --port "${FROZEN_MODEL_PORT}" \
        --tp "${FROZEN_MODEL_TP}" \
        --mem-fraction-static "${FROZEN_MODEL_MEM_FRACTION}" \
        --max-running-requests 32 \
        --trust-remote-code \
        > "${FROZEN_LOG}" 2>&1 &
    FROZEN_PID=$!

    log_info "冻结模型进程 PID=${FROZEN_PID}, 日志: ${FROZEN_LOG}"
    log_info "等待推理引擎就绪..."

    local max_wait=300
    local elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if curl -s "http://localhost:${FROZEN_MODEL_PORT}/health" > /dev/null 2>&1; then
            log_info "冻结模型推理引擎就绪！(${elapsed}s)"
            return 0
        fi
        if ! kill -0 "${FROZEN_PID}" 2>/dev/null; then
            log_error "冻结模型进程已退出，查看日志: ${FROZEN_LOG}"
            tail -20 "${FROZEN_LOG}"
            exit 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done

    log_error "冻结模型启动超时 (${max_wait}s)"
    kill "${FROZEN_PID}" 2>/dev/null || true
    exit 1
}

stop_frozen_model() {
    if [ -n "${FROZEN_PID}" ] && kill -0 "${FROZEN_PID}" 2>/dev/null; then
        log_step "关闭冻结模型推理引擎 (PID=${FROZEN_PID})..."
        kill "${FROZEN_PID}" 2>/dev/null || true
        wait "${FROZEN_PID}" 2>/dev/null || true
        log_info "冻结模型已关闭"
    fi
    pkill -f "sglang.*${FROZEN_MODEL_PORT}" 2>/dev/null || true
}

trap stop_frozen_model EXIT

if [ -n "${FROZEN_MODEL_URL}" ]; then
    log_info "使用外部冻结模型: ${FROZEN_MODEL_URL}"
    export FROZEN_MODEL_URL
elif [ -n "${FROZEN_MODEL_PATH}" ]; then
    start_frozen_model
    FROZEN_MODEL_URL="http://localhost:${FROZEN_MODEL_PORT}/v1/chat/completions"
    export FROZEN_MODEL_URL
else
    log_warn "未配置冻结模型 (FROZEN_MODEL_URL 和 FROZEN_MODEL_PATH 均为空)"
    log_warn "reward 将不使用 QA 模型评分"
fi

# =============================================================================
# Step 2: 启动 SwanLab 监控
# =============================================================================
log_step "启动 SwanLab 监控..."
SWANLAB_SCRIPT="${SCRIPT_DIR}/swanlab_monitor.py"
if [ -f "${SWANLAB_SCRIPT}" ]; then
    nohup python3 "${SWANLAB_SCRIPT}" \
        --log-dir "${LOG_DIR}" \
        --task-version "${MEMORY_RL_TASK_VERSION}" \
        --poll-interval 5.0 \
        > "${LOG_DIR}/swanlab_monitor.log" 2>&1 &
    SWANLAB_PID=$!
    echo "${SWANLAB_PID}" > "${LOG_DIR}/swanlab_monitor.pid"
    log_info "SwanLab 监控已启动 (PID=${SWANLAB_PID})"
    log_info "  监控日志: ${LOG_DIR}/swanlab_monitor.log"
    log_info "  reward 指标: ${REWARD_METRICS_LOG}"
else
    log_warn "SwanLab 监控脚本不存在: ${SWANLAB_SCRIPT}"
fi

# =============================================================================
# Step 3: 启动 GRPO 训练
# =============================================================================
log_step "启动 GRPO 训练..."
log_info "custom-rm-path: ${CUSTOM_RM_PATH}"
log_info "custom-generate: ${CUSTOM_GENERATE_FUNCTION_PATH}"

python3 "${SLIME_TRAIN_SCRIPT}" \
  --custom-rm-path "${CUSTOM_RM_PATH}" \
  --custom-generate-function-path "${CUSTOM_GENERATE_FUNCTION_PATH}" \
  --train-data "${TRAIN_DATA}" \
  --eval-data "${EVAL_DATA}" \
  ${MODEL_CONFIG_ARGS} \
  "$@" \
  2>&1 | tee "${LOG_DIR}/train.log"
