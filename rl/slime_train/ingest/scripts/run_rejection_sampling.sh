#!/usr/bin/env bash
# =============================================================================
# 在 head 节点 (192.168.16.127) 的 slime_rl_ingest 容器内运行拒绝采样过滤脚本。
#
# 用法：
#   bash run_rejection_sampling.sh ingest 10            # ingest 测试 10 条
#   bash run_rejection_sampling.sh ingest 0             # ingest 全量
#   bash run_rejection_sampling.sh retrieve 10
#   bash run_rejection_sampling.sh consolidate 10
#
# 默认 4 端点负载均衡 (103/102/100/87:7777)，并支持断点续传。
# 通过环境变量覆盖：
#   ROLLOUT_URLS="http://h1:7777,http://h2:7777,..."
#   SAMPLE_CONCURRENCY=64
#   N_ROLLOUTS=8
#   RESUME=false               # 默认 true，断点续传
# =============================================================================
set -uo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC}  $(date '+%H:%M:%S') $1"; }
log_step()  { echo -e "${CYAN}[STEP]${NC}  $(date '+%H:%M:%S') $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date '+%H:%M:%S') $1"; }

TASK="${1:?Usage: $0 <ingest|retrieve|consolidate> <limit> [n_rollouts] [sample_concurrency]}"
LIMIT="${2:-10}"
N_ROLLOUTS="${3:-${N_ROLLOUTS:-8}}"
SAMPLE_CONCURRENCY="${4:-${SAMPLE_CONCURRENCY:-25}}"
RESUME="${RESUME:-true}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-3000}"
LLM_TIMEOUT="${LLM_TIMEOUT:-1200}"
FROZEN_MAX_CONCURRENCY="${FROZEN_MAX_CONCURRENCY:-64}"
PROBE_EVAL_CONCURRENCY="${PROBE_EVAL_CONCURRENCY:-16}"

# 默认 head node + 容器
HEAD_NODE="${HEAD_NODE:-192.168.16.127}"
CONTAINER_NAME="${CONTAINER_NAME:-slime_rl_ingest}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes"

# vLLM rollout endpoints（多端点负载均衡）
ROLLOUT_URLS="${ROLLOUT_URLS:-http://192.168.16.103:7777,http://192.168.16.102:7777,http://192.168.16.100:7777,http://192.168.16.87:7777}"
ROLLOUT_MODEL="${ROLLOUT_MODEL:-/data/cloud_disk_1/models/Qwen/Qwen3.6-27B}"

# 数据
# 默认指向 erenpeng/datasets/merged_stage1_e2e/slime_output（最新清洗 + noop_fixed 版本）
# 可通过环境变量 DATA_ROOT 覆盖
DATA_ROOT="${DATA_ROOT:-/data/cloud_disk_1/erenpeng/datasets/merged_stage1_e2e/slime_output}"
SNAPSHOT_DATA_ROOT="${SNAPSHOT_DATA_ROOT:-/data/cloud_disk_1/erenpeng/datasets/merged_stage1_e2e}"
SCRIPT_PATH="/data/cloud_disk_1/erenpeng/llm-gateway/rl/slime_train/ingest/scripts/rejection_sampling_filter.py"

case "${TASK}" in
    ingest)      INPUT="${DATA_ROOT}/ingest_merged_stage1_e2e_train.clean.noop_fixed.jsonl";;
    # retrieve/consolidate 必须用 fix_snapshot_ids.py 修过 snapshot_id 的 .snapfix 版本
    # （原文件的 snapshot_id 指向 step=0 空快照，无记忆数据，跑出来全是 0 分）
    retrieve)    INPUT="${DATA_ROOT}/retrieve_merged_stage1_e2e_train.clean.snapfix.jsonl";;
    consolidate) INPUT="${DATA_ROOT}/consolidate_merged_stage1_e2e_train.clean.snapfix.jsonl";;
    *) log_error "未知 task: ${TASK}"; exit 1 ;;
esac

# 输入文件存在性兜底校验（避免 input 路径错误时浪费时间）
if [ ! -f "${INPUT}" ]; then
    if ! ssh ${SSH_OPTS} root@"${HEAD_NODE}" "docker exec ${CONTAINER_NAME} test -f ${INPUT}" 2>/dev/null; then
        log_error "输入文件不存在: ${INPUT}"
        exit 2
    fi
fi

SUFFIX="$([ ${LIMIT} -gt 0 ] && echo "_test${LIMIT}" || echo "_full")"
OUTPUT_DIR="${DATA_ROOT}/rejection_filtered"
mkdir -p "${OUTPUT_DIR}" 2>/dev/null || ssh ${SSH_OPTS} root@"${HEAD_NODE}" "mkdir -p ${OUTPUT_DIR}"
OUTPUT="${OUTPUT_DIR}/${TASK}${SUFFIX}.jsonl"
LOG_FILE="${OUTPUT_DIR}/${TASK}${SUFFIX}.log"

log_step "==== 拒绝采样: task=${TASK}, limit=${LIMIT}, n_rollouts=${N_ROLLOUTS}, sample_concurrency=${SAMPLE_CONCURRENCY}, resume=${RESUME} ===="
log_info "  Input:        ${INPUT}"
log_info "  Output:       ${OUTPUT}"
log_info "  Log:          ${LOG_FILE}"
log_info "  Rollout URLs: ${ROLLOUT_URLS}"
log_info "  Model:        ${ROLLOUT_MODEL}"
log_info "  Container:    ${HEAD_NODE} :: ${CONTAINER_NAME}"
log_info "  Timeouts:     llm=${LLM_TIMEOUT}s rollout=${ROLLOUT_TIMEOUT}s frozen_conc=${FROZEN_MAX_CONCURRENCY} probe_conc=${PROBE_EVAL_CONCURRENCY}"
log_info "  Total rollout concurrency: $((SAMPLE_CONCURRENCY * N_ROLLOUTS)) ( ${SAMPLE_CONCURRENCY} sample × ${N_ROLLOUTS} rollout )"

# vLLM 健康检查（每个 endpoint）
healthy=0
total=0
IFS=',' read -ra _URLS <<< "${ROLLOUT_URLS}"
for u in "${_URLS[@]}"; do
    u=$(echo "$u" | xargs)
    total=$((total+1))
    if curl -s --connect-timeout 5 --max-time 8 "${u}/v1/models" 2>/dev/null | grep -q '"data"'; then
        log_info "  ✓ healthy: ${u}"
        healthy=$((healthy+1))
    else
        log_warn "  ✗ unreachable: ${u}（将从负载池剔除前请手工检查）"
    fi
done
if [ ${healthy} -eq 0 ]; then
    log_error "所有 vLLM endpoints 不可达，先部署再跑"
    exit 1
fi
log_info "Healthy endpoints: ${healthy}/${total}"

# 仅保留 healthy endpoints
HEALTHY_URLS=""
for u in "${_URLS[@]}"; do
    u=$(echo "$u" | xargs)
    if curl -s --connect-timeout 3 --max-time 5 "${u}/v1/models" 2>/dev/null | grep -q '"data"'; then
        if [ -z "${HEALTHY_URLS}" ]; then
            HEALTHY_URLS="${u}"
        else
            HEALTHY_URLS="${HEALTHY_URLS},${u}"
        fi
    fi
done
log_info "Effective rollout URLs: ${HEALTHY_URLS}"

RESUME_FLAG=""
if [ "${RESUME}" = "true" ]; then
    RESUME_FLAG="--resume"
fi

# tee 默认覆盖；resume 时改为 append，避免丢失之前的日志
TEE_FLAG=""
if [ "${RESUME}" = "true" ]; then
    TEE_FLAG="-a"
fi

# 在容器里运行
CMD="export PYTHONPATH=/root/Megatron-LM:/data/cloud_disk_1/erenpeng:/data/cloud_disk_1/erenpeng/llm-gateway && \
     export PYTHONUNBUFFERED=1 && \
     cd /data/cloud_disk_1/erenpeng/llm-gateway && \
     python3 ${SCRIPT_PATH} \
        --task ${TASK} \
        --input  ${INPUT} \
        --output ${OUTPUT} \
        --limit ${LIMIT} \
        --n-rollouts ${N_ROLLOUTS} \
        --sample-concurrency ${SAMPLE_CONCURRENCY} \
        --rollout-url   '${HEALTHY_URLS}' \
        --rollout-model ${ROLLOUT_MODEL} \
        --snapshot-data-root ${SNAPSHOT_DATA_ROOT} \
        --task-version t2_agent_loop \
        --max-tokens 64000 \
        --llm-timeout ${LLM_TIMEOUT} \
        --max-agent-turns 20 \
        --rollout-timeout ${ROLLOUT_TIMEOUT} \
        --probe-eval-concurrency ${PROBE_EVAL_CONCURRENCY} \
        --frozen-max-concurrency ${FROZEN_MAX_CONCURRENCY} \
        --treat-no-discrimination-as-fail \
        ${RESUME_FLAG} \
        2>&1 | tee ${TEE_FLAG} ${LOG_FILE}"

ssh ${SSH_OPTS} root@"${HEAD_NODE}" "docker exec -i ${CONTAINER_NAME} bash -lc \"${CMD}\""

EXIT=$?
log_info "退出码: ${EXIT}"
exit ${EXIT}
