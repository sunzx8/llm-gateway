#!/usr/bin/env bash
# =============================================================================
# 在内网机器 192.168.16.103 上部署 Qwen3.6-27B vLLM 推理服务（OpenAI 兼容）
#
# 用途：拒绝采样（rejection sampling）所需的 rollout 模型。
#
# 使用：
#   bash deploy_qwen_vllm.sh                # 启动服务
#   bash deploy_qwen_vllm.sh --restart      # 重启容器
#   bash deploy_qwen_vllm.sh --stop         # 停止并删除容器
#   bash deploy_qwen_vllm.sh --status       # 查看容器与端点状态
#   bash deploy_qwen_vllm.sh --logs         # tail 容器日志
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC}  $(date '+%H:%M:%S') $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date '+%H:%M:%S') $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $1"; }
log_step()  { echo -e "${CYAN}[STEP]${NC}  $(date '+%H:%M:%S') $1"; }

# -----------------------------------------------------------------------------
# 部署配置
# -----------------------------------------------------------------------------
DEPLOY_HOST="${DEPLOY_HOST:-192.168.16.103}"
SSH_USER="${SSH_USER:-root}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes"

CONTAINER_NAME="${CONTAINER_NAME:-qwen3.6-27b-fast-inference}"
DOCKER_IMAGE="${DOCKER_IMAGE:-vllm/vllm-openai:nightly}"
# 模型路径（host 侧实际存在的位置）
MODEL_HOST_PATH="${MODEL_HOST_PATH:-/data/cloud_disk_4/dl_models/Qwen/Qwen3.6-27B}"
# 容器内的挂载点（保持与原命令一致的路径风格）
MODEL_CTN_PATH="${MODEL_CTN_PATH:-/data/cloud_disk_1/models/Qwen/Qwen3.6-27B}"

VLLM_PORT="${VLLM_PORT:-7777}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
# 是否启用 Multi-Token-Prediction 投机解码；当前 vllm/vllm-openai:nightly 中
# Qwen3_5MTP 架构尚未注册，开启会启动失败。默认关闭。
ENABLE_MTP="${ENABLE_MTP:-false}"

ACTION="start"
while [[ $# -gt 0 ]]; do
    case $1 in
        --start)   ACTION="start"; shift ;;
        --restart) ACTION="restart"; shift ;;
        --stop)    ACTION="stop"; shift ;;
        --status)  ACTION="status"; shift ;;
        --logs)    ACTION="logs"; shift ;;
        --host)    DEPLOY_HOST="$2"; shift 2 ;;
        --port)    VLLM_PORT="$2"; shift 2 ;;
        *) log_error "未知参数: $1"; exit 1 ;;
    esac
done

ssh_run() { ssh ${SSH_OPTS} "${SSH_USER}@${DEPLOY_HOST}" "$@"; }

check_connectivity() {
    log_step "检查 SSH 连通性: ${SSH_USER}@${DEPLOY_HOST}"
    if ! ssh_run "echo ok" >/dev/null 2>&1; then
        log_error "无法 SSH 到 ${DEPLOY_HOST}，请检查网络或 ssh 密钥"
        exit 1
    fi
    log_info "SSH 连通正常 ✓"
}

stop_container() {
    log_step "停止并删除容器 ${CONTAINER_NAME} ..."
    ssh_run "docker rm -f ${CONTAINER_NAME} 2>/dev/null || true" >/dev/null
    log_info "容器已清理 ✓"
}

start_container() {
    log_step "在 ${DEPLOY_HOST} 上拉起容器 ${CONTAINER_NAME} ..."

    # 校验模型路径
    if ! ssh_run "test -f ${MODEL_HOST_PATH}/config.json"; then
        log_error "模型路径无效: ${MODEL_HOST_PATH}/config.json 不存在"
        exit 1
    fi
    log_info "模型路径校验通过: ${MODEL_HOST_PATH}"

    # 拉镜像（如本地无）
    if ! ssh_run "docker images --format '{{.Repository}}:{{.Tag}}' | grep -q '^${DOCKER_IMAGE}$'"; then
        log_warn "本地无 ${DOCKER_IMAGE}，开始 docker pull ..."
        ssh_run "docker pull ${DOCKER_IMAGE}"
    else
        log_info "本地已有镜像 ${DOCKER_IMAGE}"
    fi

    # 修复 nightly 镜像中 torch 的 _cp_custom_ops.py 误依赖 pytest 的问题：
    # 把 pytest 预装进镜像（commit 一次，不影响后续重启）。
    local FIXED_IMAGE="${DOCKER_IMAGE}-pytestfix"
    if ! ssh_run "docker images --format '{{.Repository}}:{{.Tag}}' | grep -q '^${FIXED_IMAGE}$'"; then
        log_step "首次部署：在 ${DOCKER_IMAGE} 中预装 pytest 并 commit 为 ${FIXED_IMAGE} ..."
        ssh_run "docker rm -f vllm_pytest_fix_tmp 2>/dev/null || true" >/dev/null
        ssh_run "docker run --name vllm_pytest_fix_tmp --entrypoint /bin/bash ${DOCKER_IMAGE} -c 'pip install --no-cache-dir -i https://mirrors.cloud.tencent.com/pypi/simple pytest >/dev/null 2>&1 && python3 -c \"import pytest; print(pytest.__version__)\"'"
        # commit 时恢复原 entrypoint=[\"vllm\",\"serve\"]，否则启动时 args 会被 bash 误解析
        ssh_run "docker commit --change='ENTRYPOINT [\"vllm\",\"serve\"]' --change='CMD []' vllm_pytest_fix_tmp ${FIXED_IMAGE}" >/dev/null
        ssh_run "docker rm -f vllm_pytest_fix_tmp" >/dev/null
        log_info "镜像修复完成: ${FIXED_IMAGE}"
    else
        log_info "已有修复镜像 ${FIXED_IMAGE}，跳过修复"
    fi
    DOCKER_IMAGE="${FIXED_IMAGE}"

    # 端口冲突检测
    if ssh_run "ss -tlnp 2>/dev/null | grep -q ':${VLLM_PORT}\\b'"; then
        log_warn "端口 ${VLLM_PORT} 已被占用，将先停止旧容器"
        stop_container
    fi

    # 启动容器
    # 与用户原始命令保持一致；同时把 host 侧实际模型目录挂载到容器内的同样路径
    local SPEC_FLAG=""
    if [ "${ENABLE_MTP}" = "true" ]; then
        SPEC_FLAG="--speculative-config '{\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":2}'"
    else
        log_info "ENABLE_MTP=false：当前 nightly 镜像未注册 Qwen3_5MTP，跳过 speculative-config"
    fi

    ssh_run "docker run -itd \
        --name ${CONTAINER_NAME} \
        --ipc=host \
        --network host \
        --shm-size 16G \
        --gpus all \
        -v ${MODEL_HOST_PATH}:${MODEL_CTN_PATH} \
        -v /root/.cache/huggingface:/root/.cache/huggingface \
        ${DOCKER_IMAGE} \
        --model ${MODEL_CTN_PATH} \
        --trust-remote-code \
        --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
        --max-model-len ${MAX_MODEL_LEN} \
        --gpu-memory-utilization ${GPU_MEM_UTIL} \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --async-scheduling \
        --language-model-only \
        --limit-mm-per-prompt '{\"image\": 0, \"audio\": 0, \"video\": 0}' \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --enable-auto-tool-choice \
        ${SPEC_FLAG} \
        --default-chat-template-kwargs '{\"enable_thinking\": true}' \
        --max-num-seqs ${MAX_NUM_SEQS} \
        --host 0.0.0.0 \
        --port ${VLLM_PORT}" >/dev/null

    log_info "容器已启动，开始等待健康检查 ..."
    wait_until_healthy
}

wait_until_healthy() {
    local max_wait=900    # 27B + TP=8 加载 + warmup 大约 4-8 分钟
    local elapsed=0
    local interval=10
    local url="http://${DEPLOY_HOST}:${VLLM_PORT}/v1/models"
    while [ $elapsed -lt $max_wait ]; do
        if curl -s --connect-timeout 3 --max-time 5 "${url}" 2>/dev/null | grep -q '"data"'; then
            log_info "vLLM 服务就绪 ✓ (${elapsed}s)"
            log_info "  Endpoint:  http://${DEPLOY_HOST}:${VLLM_PORT}/v1/chat/completions"
            log_info "  Models:    http://${DEPLOY_HOST}:${VLLM_PORT}/v1/models"
            ssh_run "docker logs --tail 5 ${CONTAINER_NAME} 2>&1" | sed 's/^/    /' || true
            return 0
        fi
        # 容器异常退出 fast-fail
        if ! ssh_run "docker ps --format '{{.Names}}' | grep -q '^${CONTAINER_NAME}$'"; then
            log_error "容器 ${CONTAINER_NAME} 已退出，最后日志:"
            ssh_run "docker logs --tail 60 ${CONTAINER_NAME} 2>&1" | sed 's/^/    /' || true
            exit 1
        fi
        printf "\r${GREEN}[INFO]${NC}  $(date '+%H:%M:%S') 等待 vLLM 加载... ${elapsed}s/${max_wait}s"
        sleep ${interval}
        elapsed=$((elapsed + interval))
    done
    echo ""
    log_error "vLLM 启动超时 (${max_wait}s)，最近日志:"
    ssh_run "docker logs --tail 80 ${CONTAINER_NAME} 2>&1" | sed 's/^/    /' || true
    exit 1
}

show_status() {
    log_step "容器状态:"
    ssh_run "docker ps -a --filter name=${CONTAINER_NAME} --format 'table {{.Names}}\\t{{.Status}}\\t{{.Image}}'" || true
    log_step "端点检查 http://${DEPLOY_HOST}:${VLLM_PORT}/v1/models :"
    if curl -s --connect-timeout 3 --max-time 5 "http://${DEPLOY_HOST}:${VLLM_PORT}/v1/models" | head -c 400; then
        echo ""
        log_info "端点可用 ✓"
    else
        log_warn "端点不可达"
    fi
}

show_logs() {
    log_step "tail 容器日志（Ctrl+C 退出）"
    ssh_run "docker logs -f --tail 200 ${CONTAINER_NAME}"
}

case "${ACTION}" in
    start)
        check_connectivity
        if ssh_run "docker ps --format '{{.Names}}' | grep -q '^${CONTAINER_NAME}$'"; then
            log_warn "容器 ${CONTAINER_NAME} 已在运行；使用 --restart 重启或 --status 查看"
            show_status
            exit 0
        fi
        # 残留 stopped 容器
        if ssh_run "docker ps -a --format '{{.Names}}' | grep -q '^${CONTAINER_NAME}$'"; then
            stop_container
        fi
        start_container
        show_status
        ;;
    restart)
        check_connectivity
        stop_container
        start_container
        show_status
        ;;
    stop)
        check_connectivity
        stop_container
        ;;
    status)
        check_connectivity
        show_status
        ;;
    logs)
        check_connectivity
        show_logs
        ;;
    *)
        log_error "未知 action: ${ACTION}"; exit 1 ;;
esac
