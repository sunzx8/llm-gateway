#!/usr/bin/env bash
# =============================================================================
# 一键串行跑全量拒绝采样过滤：ingest -> retrieve -> consolidate
# 多端点负载均衡 + 断点续传，目标总并发 ~400
# =============================================================================
set -uo pipefail

GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
log_step()  { echo -e "${CYAN}[STEP]${NC}  $(date '+%F %T') $1"; }
log_info()  { echo -e "${GREEN}[INFO]${NC}  $(date '+%F %T') $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%F %T') $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SH="${SCRIPT_DIR}/run_rejection_sampling.sh"

# 全量配置
LIMIT=0                                   # 0 = 全量
export N_ROLLOUTS="${N_ROLLOUTS:-8}"
export RESUME="${RESUME:-true}"           # 默认开启断点续传
# 各 task 的 sample 并发；总 rollout 并发 = N * sample_concurrency
# 4 endpoints × 50 / endpoint = 200 总并发（用户指定上限）
ING_CONC="${ING_CONC:-25}"                # ingest 25 sample × 8 rollout = 200 并发
RET_CONC="${RET_CONC:-25}"                # retrieve 25 × 8 = 200
CON_CONC="${CON_CONC:-25}"                # consolidate 25 × 8 = 200
# rollout 内部超时（之前 1500 仍不够，部分 rollout 整组都被 1500s 切掉，提到 3000s）
export ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-3000}"
export LLM_TIMEOUT="${LLM_TIMEOUT:-1200}"
export FROZEN_MAX_CONCURRENCY="${FROZEN_MAX_CONCURRENCY:-64}"
export PROBE_EVAL_CONCURRENCY="${PROBE_EVAL_CONCURRENCY:-16}"

OUT_DIR="${OUT_DIR:-/data/cloud_disk_1/erenpeng/datasets/merged_stage1_e2e/slime_output/rejection_filtered}"
mkdir -p "${OUT_DIR}" 2>/dev/null || true

START_TS=$(date +%s)
log_step "============================================================"
log_step " Rejection Sampling — FULL run (ingest → retrieve → consolidate)"
log_step " Output dir: ${OUT_DIR}"
log_step " N_ROLLOUTS=${N_ROLLOUTS}  ING=${ING_CONC} RET=${RET_CONC} CON=${CON_CONC}"
log_step " ROLLOUT_TIMEOUT=${ROLLOUT_TIMEOUT}s  LLM_TIMEOUT=${LLM_TIMEOUT}s  FROZEN_CONC=${FROZEN_MAX_CONCURRENCY}  PROBE_EVAL_CONC=${PROBE_EVAL_CONCURRENCY}"
log_step " RESUME=${RESUME}  ROLLOUT_URLS=${ROLLOUT_URLS:-default 4-endpoint}"
log_step "============================================================"

run_task() {
    local task=$1 conc=$2
    log_step "=== START task=${task} concurrency=${conc} ==="
    local t0=$(date +%s)
    if bash "${RUN_SH}" "${task}" "${LIMIT}" "${N_ROLLOUTS}" "${conc}"; then
        local t1=$(date +%s)
        log_info "=== DONE  task=${task} elapsed=$((t1-t0))s ==="
        return 0
    else
        local rc=$?
        log_error "=== FAIL  task=${task} exit_code=${rc} ==="
        return ${rc}
    fi
}

# 顺序：ingest -> retrieve -> consolidate
run_task ingest      "${ING_CONC}" || log_error "ingest failed but continuing"
run_task retrieve    "${RET_CONC}" || log_error "retrieve failed but continuing"
run_task consolidate "${CON_CONC}" || log_error "consolidate failed but continuing"

END_TS=$(date +%s)
log_step "============================================================"
log_step " ALL DONE  total_elapsed=$((END_TS-START_TS))s"
log_step " Output dir: ${OUT_DIR}"
log_step "============================================================"
ssh -o StrictHostKeyChecking=no -o BatchMode=yes root@192.168.16.127 "docker exec slime_rl_ingest ls -la ${OUT_DIR}" || true
