#!/bin/bash
# =============================================================================
# Retrieve (消费) 阶段 GRPO 训练脚本 — 原生检索任务训练
#
# 流程:
#   1. 准备 reward QA / frozen 检索模型：可启动本地 SGLang，也可使用外部 OpenAI-compatible endpoint
#   2. rollout 通过 custom_generate 跑真实 retrieve task loop，产出 context/trajectory
#   3. reward 直接基于 rollout context 做 frozen QA 评分，不重复执行检索
#   4. 训练结束后关闭推理引擎
#
# 资源: 8 节点 × 8 GPU = 64 GPU, 训推分离 (32 训 / 32 推)
# 框架: slime + Megatron-LM + SGLang + Ray
#
# 使用:
#   bash train_retrieve_grpo.sh              # 默认配置
#   bash train_retrieve_grpo.sh --kill        # 清理所有残留进程
#   bash train_retrieve_grpo.sh --no-frozen   # 跳过本地冻结模型启动（已手动/外部启动）
#   FROZEN_MODEL_URL=http://host:8000/v1/chat/completions FROZEN_MODEL_NAME=model bash train_retrieve_grpo.sh
# =============================================================================

set -e

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
# llm_gateway 根目录（包含 gateway/, storage/, rl/）
LLM_GATEWAY_ROOT="${LLM_GATEWAY_ROOT:-$(cd "${SLIME_TRAIN_ROOT}/../.." && pwd)}"
LLM_GATEWAY_PARENT="${LLM_GATEWAY_PARENT:-$(cd "${LLM_GATEWAY_ROOT}/.." && pwd)}"

# 训练/验证数据（可传入已切好的 TRAIN_DATA + VAL_DATA/EVAL_DATA）
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/rl_train.jsonl}"
EVAL_DATA="${VAL_DATA:-${EVAL_DATA:-${PROJECT_ROOT}/data/rl_val.jsonl}}"

# 快照数据根目录（reward 函数加载快照时使用）
# 指向统一的数据目录（包含 snapshots/, workdirs/, ingest_snapshots.jsonl 等）
SNAPSHOT_DATA_ROOT="${SNAPSHOT_DATA_ROOT:-/data/cloud_disk_1/changyuchen/memory_ai_RL/slime_train/data/rl_data_test_2}"

# 模型 checkpoint（只读引用，训练过程不会修改这些文件）
# HF 格式（用于 SGLang 推理引擎 + 冻结模型）
HF_CHECKPOINT="${HF_CHECKPOINT:-/data/cloud_disk_1/changyuchen/memory_ai_RL/model/iter_0000189_hf}"
# Megatron torch_dist 格式（用于 slime 训练 ref model，--ref-load 只读加载）
TORCH_DIST_CHECKPOINT="${TORCH_DIST_CHECKPOINT:-/data/cloud_disk_1/changyuchen/memory_ai_RL/model/iter_0000189_torch_dist}"

# Reward QA 模型：可本地启动，也可直接使用外部 OpenAI-compatible/vLLM endpoint。
EXTERNAL_FROZEN_MODEL_URL="${FROZEN_MODEL_URL:-}"
FROZEN_MODEL_PATH="${FROZEN_MODEL_PATH:-${HF_CHECKPOINT}}"
FROZEN_MODEL_PORT="${FROZEN_MODEL_PORT:-30100}"
FROZEN_MODEL_URL="${FROZEN_MODEL_URL:-http://localhost:${FROZEN_MODEL_PORT}/v1/chat/completions}"
FROZEN_MODEL_NAME="${FROZEN_MODEL_NAME:-default}"
FROZEN_MODEL_TIMEOUT="${FROZEN_MODEL_TIMEOUT:-30}"
FROZEN_MODEL_MAX_CONCURRENCY="${FROZEN_MODEL_MAX_CONCURRENCY:-32}"
FROZEN_MODEL_CONN_LIMIT="${FROZEN_MODEL_CONN_LIMIT:-128}"
PROBE_EVAL_MAX_CONCURRENCY="${PROBE_EVAL_MAX_CONCURRENCY:-8}"
PROBE_TASK_MAX_CONCURRENCY="${PROBE_TASK_MAX_CONCURRENCY:-1}"
FROZEN_MODEL_TP="${FROZEN_MODEL_TP:-2}"
FROZEN_MODEL_MEM_FRACTION="${FROZEN_MODEL_MEM_FRACTION:-0.85}"

# Docker/训练框架路径
DOCKER_IMAGE="${DOCKER_IMAGE:-slimerl/slime:latest}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"
SLIME_TRAIN_SCRIPT="${SLIME_TRAIN_SCRIPT:-/root/slime/train_async.py}"

# 唯一标识
RUN_NAME="retrieve_grpo_$(date '+%Y%m%d_%H%M%S')"

# 输出（所有训练产物写入 PROJECT_ROOT 下，绝不修改源 checkpoint）
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${RUN_NAME}"
CKPT_DIR="${OUTPUT_DIR}/checkpoints"
LOG_DIR="${PROJECT_ROOT}/logs/${RUN_NAME}"
DUMP_DETAILS_DIR="${PROJECT_ROOT}/rollout_dumps/${RUN_NAME}"

# =============================================================================
# PYTHONPATH
# =============================================================================
export PYTHONPATH="${MEGATRON_ROOT}:${LLM_GATEWAY_PARENT}:${LLM_GATEWAY_ROOT}"

# =============================================================================
# 集群节点
# =============================================================================
if [ -n "${ALL_IPS_STR:-}" ]; then
    IFS=',' read -r -a ALL_IPS <<< "${ALL_IPS_STR}"
else
    ALL_IPS=(
        192.168.16.48
        192.168.16.49
        192.168.16.38
        192.168.16.31
        192.168.16.52
        192.168.16.37
        192.168.16.40
        192.168.16.34
    )
fi
GPUS_PER_NODE=8

# =============================================================================
# GPU 分配方案（训推分离，异步 GRPO）
#
# 方案 A（默认）: 8 节点全部参与，4 训练 + 4 推理
#   训练: 4 节点 × 8 GPU = 32 GPU
#   推理: 4 节点 × 8 GPU = 32 GPU
#
# 方案 B（如果 head 节点 OOM）: 1 调度 + 4 训练 + 3 推理
#   调度: 1 节点（head，注册 0 GPU，仅 Ray head + 调度）
#   训练: 4 节点 × 8 GPU = 32 GPU
#   推理: 3 节点 × 8 GPU = 24 GPU
#
# 切换方案: --plan b
# =============================================================================
GPU_PLAN="a"

# =============================================================================
# 环境变量（rollout / reward 函数使用）
# =============================================================================
MEMORY_RL_TASK_VERSION="${MEMORY_RL_TASK_VERSION:-t2_agent_loop}"
MEMORY_RL_APPLY_MODE="${MEMORY_RL_APPLY_MODE:-task_loop}"
MEMORY_RL_TRAIN_TASK_LOOP="${MEMORY_RL_TRAIN_TASK_LOOP:-1}"
MEMORY_RL_MAX_AGENT_TURNS="${MEMORY_RL_MAX_AGENT_TURNS:-5}"
CUSTOM_GENERATE_FUNCTION_PATH="${CUSTOM_GENERATE_FUNCTION_PATH:-llm_gateway.rl.slime_train.memory_rl.custom_generate.custom_generate}"
CUSTOM_RM_PATH="${CUSTOM_RM_PATH:-llm_gateway.rl.slime_train.tasks.retrieve_reward.reward.reward_func}"

export SNAPSHOT_DATA_ROOT="${SNAPSHOT_DATA_ROOT}"
export RETRIEVE_SNAPSHOT_DATA_ROOT="${SNAPSHOT_DATA_ROOT}"
export MEMORY_RL_TASK_VERSION MEMORY_RL_APPLY_MODE MEMORY_RL_TRAIN_TASK_LOOP MEMORY_RL_MAX_AGENT_TURNS
export FROZEN_MODEL_URL FROZEN_MODEL_NAME FROZEN_MODEL_TIMEOUT FROZEN_MODEL_MAX_CONCURRENCY FROZEN_MODEL_CONN_LIMIT PROBE_EVAL_MAX_CONCURRENCY PROBE_TASK_MAX_CONCURRENCY
export MEMORY_RL_RETRIEVE_LLM_API_URL="${MEMORY_RL_RETRIEVE_LLM_API_URL:-${FROZEN_MODEL_URL}}"
export MEMORY_RL_RETRIEVE_LLM_MODEL="${MEMORY_RL_RETRIEVE_LLM_MODEL:-${FROZEN_MODEL_NAME}}"

# =============================================================================
# 训练超参
# =============================================================================
NUM_ROLLOUT=400
ROLLOUT_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=8
ROLLOUT_MAX_RESPONSE_LEN=2048
ROLLOUT_TEMPERATURE=1.0
GLOBAL_BATCH_SIZE=32
MAX_TOKENS_PER_GPU=6144
SAVE_INTERVAL=25
LR=5e-7
WEIGHT_DECAY=0.1

# =============================================================================
# 参数解析
# =============================================================================
SKIP_FROZEN=false
DO_KILL=false

if [ -n "${EXTERNAL_FROZEN_MODEL_URL}" ]; then
    SKIP_FROZEN=true
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-frozen)  SKIP_FROZEN=true; shift ;;
        --kill)       DO_KILL=true; shift ;;
        --plan)       GPU_PLAN="$2"; shift 2 ;;
        *)            log_error "未知参数: $1"; exit 1 ;;
    esac
done

# 根据方案设置 GPU 分配参数
if [ "${GPU_PLAN}" = "b" ]; then
    # 方案 B: 1 调度 + 4 训练 + 3 推理
    ACTOR_NUM_NODES=4
    ACTOR_NUM_GPUS_PER_NODE=${GPUS_PER_NODE}
    ROLLOUT_NUM_GPUS=24
    TOTAL_NODES=8
    log_info "GPU 方案 B: 1 调度(0 GPU) + 4 训练(32 GPU) + 3 推理(24 GPU)"
else
    # 方案 A: 4 训练 + 4 推理（head 也参与计算）
    ACTOR_NUM_NODES=4
    ACTOR_NUM_GPUS_PER_NODE=${GPUS_PER_NODE}
    ROLLOUT_NUM_GPUS=32
    TOTAL_NODES=8
    log_info "GPU 方案 A: 4 训练(32 GPU) + 4 推理(32 GPU)"
fi

# =============================================================================
# 清理模式
# =============================================================================
if [ "$DO_KILL" = true ]; then
    log_step "清理所有残留进程..."
    pkill -f "sglang.*${FROZEN_MODEL_PORT}" 2>/dev/null || true
    pkill -f "python.*launch_server.*${FROZEN_MODEL_PORT}" 2>/dev/null || true
    log_info "清理完成"
    exit 0
fi

# =============================================================================
# 预检
# =============================================================================
log_step "=========================================="
log_step "Retrieve Native Task Loop — GRPO 训练"
log_step "=========================================="
log_info "PROJECT_ROOT:       ${PROJECT_ROOT}"
log_info "WORKSPACE_SRC:      ${LLM_GATEWAY_ROOT}"
log_info "TRAIN_DATA:         ${TRAIN_DATA}"
log_info "EVAL_DATA:          ${EVAL_DATA}"
log_info "SNAPSHOT_DATA_ROOT: ${SNAPSHOT_DATA_ROOT}"
log_info "FROZEN_MODEL_PATH:  ${FROZEN_MODEL_PATH}"
log_info "FROZEN_MODEL_PORT:  ${FROZEN_MODEL_PORT}"
log_info "RUN_NAME:           ${RUN_NAME}"
log_info ""
log_info "集群: ${#ALL_IPS[@]} 节点 × ${GPUS_PER_NODE} GPU = $((${#ALL_IPS[@]} * GPUS_PER_NODE)) GPU"
log_info "GPU 方案: ${GPU_PLAN^^}"
log_info "  训练: ${ACTOR_NUM_NODES} 节点 × ${ACTOR_NUM_GPUS_PER_NODE} GPU = $((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE)) GPU"
log_info "  推理: ${ROLLOUT_NUM_GPUS} GPU"
log_info "  TP=2 (per SGLang engine)"

if [ ! -f "${PROJECT_ROOT}/tasks/retrieve_reward/reward.py" ]; then
    log_error "reward 文件不存在: ${PROJECT_ROOT}/tasks/retrieve_reward/reward.py"
    log_error "(参考: ${LLM_GATEWAY_ROOT}/rl/slime_train/tasks/retrieve_reward/reward.py)"
    exit 1
fi

if [ ! -f "${TRAIN_DATA}" ]; then
    log_error "训练数据不存在: ${TRAIN_DATA}"
    log_error "请先运行: python convert_to_slime_format.py"
    exit 1
fi
if [ ! -f "${EVAL_DATA}" ]; then
    log_warn "验证数据不存在: ${EVAL_DATA}（将仍打印训练命令，请确认 Slime 参数是否需要 --eval-data）"
fi
python3 "${SLIME_TRAIN_ROOT}/memory_rl/summarize_jsonl.py" "${TRAIN_DATA}" --label train
python3 "${SLIME_TRAIN_ROOT}/memory_rl/summarize_jsonl.py" "${EVAL_DATA}" --label val

if [ ! -d "${LLM_GATEWAY_ROOT}/rl/rl_env" ]; then
    log_error "llm_gateway/rl/rl_env 不存在: ${LLM_GATEWAY_ROOT}/rl/rl_env"
    exit 1
fi

mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${DUMP_DETAILS_DIR}"

# =============================================================================
# Step 1: 启动冻结模型推理引擎
# =============================================================================
FROZEN_PID=""

start_frozen_model() {
    log_step "启动冻结模型推理引擎 (SGLang)..."
    log_info "  模型: ${FROZEN_MODEL_PATH}"
    log_info "  端口: ${FROZEN_MODEL_PORT}"
    log_info "  TP:   ${FROZEN_MODEL_TP}"

    # 检查端口是否已被占用
    if curl -s "http://localhost:${FROZEN_MODEL_PORT}/health" > /dev/null 2>&1; then
        log_warn "端口 ${FROZEN_MODEL_PORT} 已有服务运行，跳过启动"
        return 0
    fi

    FROZEN_LOG="${LOG_DIR}/frozen_model.log"

    python -m sglang.launch_server \
        --model-path "${FROZEN_MODEL_PATH}" \
        --host 0.0.0.0 \
        --port "${FROZEN_MODEL_PORT}" \
        --tp "${FROZEN_MODEL_TP}" \
        --mem-fraction-static "${FROZEN_MODEL_MEM_FRACTION}" \
        --max-running-requests 32 \
        --trust-remote-code \
        --chat-template qwen3 \
        > "${FROZEN_LOG}" 2>&1 &
    FROZEN_PID=$!

    log_info "冻结模型进程 PID=${FROZEN_PID}, 日志: ${FROZEN_LOG}"
    log_info "等待推理引擎就绪..."

    # 等待服务就绪（最多 5 分钟）
    local max_wait=300
    local elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if curl -s "http://localhost:${FROZEN_MODEL_PORT}/health" > /dev/null 2>&1; then
            log_info "冻结模型推理引擎就绪！(${elapsed}s)"
            return 0
        fi
        # 检查进程是否还活着
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
    # 兜底清理
    pkill -f "sglang.*${FROZEN_MODEL_PORT}" 2>/dev/null || true
}

# 注册退出清理
trap stop_frozen_model EXIT

if [ "$SKIP_FROZEN" = false ]; then
    start_frozen_model
else
    log_warn "跳过冻结模型启动 (--no-frozen)"
    log_warn "请确保 ${FROZEN_MODEL_URL} 已可用"
fi

# =============================================================================
# Step 2: 启动 GRPO 训练
# =============================================================================
log_step "=========================================="
log_step "启动 GRPO 训练"
log_step "=========================================="
log_info "custom-rm-path: ${CUSTOM_RM_PATH}"
log_info "custom-generate-function-path: ${CUSTOM_GENERATE_FUNCTION_PATH}"
log_info "task_version: ${MEMORY_RL_TASK_VERSION}, apply_mode: ${MEMORY_RL_APPLY_MODE}"
log_info "rollout: ${NUM_ROLLOUT} steps, batch=${ROLLOUT_BATCH_SIZE}, samples=${N_SAMPLES_PER_PROMPT}"
log_info "lr: ${LR}, global_batch: ${GLOBAL_BATCH_SIZE}"
log_info ""
log_info "环境变量:"
log_info "  RETRIEVE_SNAPSHOT_DATA_ROOT=${RETRIEVE_SNAPSHOT_DATA_ROOT}"
log_info "  FROZEN_MODEL_URL=${FROZEN_MODEL_URL}"
log_info ""

cat <<EOF

=== 训练启动命令（在 Docker 容器内执行）===

export PYTHONPATH=${MEGATRON_ROOT}:${LLM_GATEWAY_PARENT}:${LLM_GATEWAY_ROOT}
export RETRIEVE_SNAPSHOT_DATA_ROOT=${SNAPSHOT_DATA_ROOT}
export FROZEN_MODEL_URL=${FROZEN_MODEL_URL}
export FROZEN_MODEL_NAME=${FROZEN_MODEL_NAME}
export FROZEN_MODEL_TIMEOUT=${FROZEN_MODEL_TIMEOUT}
export FROZEN_MODEL_MAX_CONCURRENCY=${FROZEN_MODEL_MAX_CONCURRENCY}
export FROZEN_MODEL_CONN_LIMIT=${FROZEN_MODEL_CONN_LIMIT}
export PROBE_EVAL_MAX_CONCURRENCY=${PROBE_EVAL_MAX_CONCURRENCY}
export PROBE_TASK_MAX_CONCURRENCY=${PROBE_TASK_MAX_CONCURRENCY}
export MEMORY_RL_TASK_VERSION=${MEMORY_RL_TASK_VERSION}
export MEMORY_RL_APPLY_MODE=${MEMORY_RL_APPLY_MODE}
export MEMORY_RL_TRAIN_TASK_LOOP=${MEMORY_RL_TRAIN_TASK_LOOP}
export MEMORY_RL_MAX_AGENT_TURNS=${MEMORY_RL_MAX_AGENT_TURNS}
export MEMORY_RL_RETRIEVE_LLM_API_URL=${MEMORY_RL_RETRIEVE_LLM_API_URL}
export MEMORY_RL_RETRIEVE_LLM_MODEL=${MEMORY_RL_RETRIEVE_LLM_MODEL}

python3 ${SLIME_TRAIN_SCRIPT} \\
    --actor-num-nodes ${ACTOR_NUM_NODES} \\
    --actor-num-gpus-per-node ${ACTOR_NUM_GPUS_PER_NODE} \\
    --rollout-num-gpus ${ROLLOUT_NUM_GPUS} \\
    --rollout-num-gpus-per-engine 2 \\
    --hf-checkpoint "${HF_CHECKPOINT}" \\
    --ref-load "${TORCH_DIST_CHECKPOINT}" \\
    --load "${CKPT_DIR}" \\
    --save "${CKPT_DIR}" \\
    --save-interval ${SAVE_INTERVAL} \\
    --prompt-data "${TRAIN_DATA}" \\
    --eval-data "${EVAL_DATA}" \\
    --input-key prompt \\
    --metadata-key metadata \\
    --apply-chat-template \\
    --custom-rm-path "${CUSTOM_RM_PATH}" \\
    --custom-generate-function-path "${CUSTOM_GENERATE_FUNCTION_PATH}" \\
\
    --num-rollout ${NUM_ROLLOUT} \\
    --rollout-batch-size ${ROLLOUT_BATCH_SIZE} \\
    --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT} \\
    --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN} \\
    --rollout-max-prompt-len 4096 \\
    --rollout-temperature ${ROLLOUT_TEMPERATURE} \\
    --global-batch-size ${GLOBAL_BATCH_SIZE} \\
    --use-dynamic-batch-size \\
    --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU} \\
    --advantage-estimator grpo \\
    --use-kl-loss \\
    --kl-loss-coef 0.005 \\
    --kl-loss-type low_var_kl \\
    --entropy-coef 0.0008 \\
    --eps-clip 0.2 \\
    --eps-clip-high 0.28 \\
    --clip-grad 1.0 \\
    --optimizer adam \\
    --lr ${LR} \\
    --lr-decay-style cosine \\
    --lr-warmup-iters 20 \\
    --min-lr 1e-7 \\
    --weight-decay ${WEIGHT_DECAY} \\
    --adam-beta1 0.9 \\
    --adam-beta2 0.98 \\
    --no-check-for-nan-in-loss-and-grad \\
    --distributed-timeout-minutes 60 \\
    --no-load-optim \\
    --no-load-rng \\
    --finetune \\
    --dump-details "${DUMP_DETAILS_DIR}"

EOF

# =============================================================================
# Step 3: 启动 SwanLab 监控
# =============================================================================
log_step "启动 SwanLab 监控..."
SWANLAB_SCRIPT="${SCRIPT_DIR}/swanlab_monitor.py"
if [ -f "${SWANLAB_SCRIPT}" ]; then
    # 设置 reward metrics 日志路径（reward 函数写入）
    export REWARD_METRICS_LOG="${LOG_DIR}/reward_metrics.jsonl"

    nohup python "${SWANLAB_SCRIPT}" \
        --log-dir "${LOG_DIR}" \
        --task-version "${MEMORY_RL_TASK_VERSION:-atomic_code_t2}" \
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

log_info ""
log_info "训练结束后冻结模型将自动关闭 (trap EXIT)"
log_info "手动清理: bash $0 --kill"
