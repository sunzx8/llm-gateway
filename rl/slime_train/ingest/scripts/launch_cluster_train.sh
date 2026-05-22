#!/usr/bin/env bash
# =============================================================================
# 一键式集群训练启动脚本
#
# 功能:
#   1. 在所有节点上检查 Docker 镜像是否存在，不存在则从 slime_latest.tar 加载
#   2. 在所有节点上启动 Docker 容器（挂载必要目录）
#   3. 在 head 容器内启动 Ray head 节点
#   4. 在 worker 容器内连接到 Ray head 节点
#   5. 在 head 容器内启动冻结模型推理引擎
#   6. 提交 GRPO 训练任务
#
# 使用:
#   bash launch_cluster_train.sh
#   bash launch_cluster_train.sh --task-version t2_agent_loop
#   bash launch_cluster_train.sh --skip-docker   # 跳过 docker 启动（容器已在运行）
#   bash launch_cluster_train.sh --skip-ray      # 跳过 Ray 集群启动（已有集群）
#   bash launch_cluster_train.sh --dry-run       # 仅打印命令不执行
#   bash /data/cloud_disk_1/erenpeng/llm-gateway/rl/slime_train/ingest/scripts/launch_cluster_train.sh --task-version t2_agent_loop --skip-docker --skip-ray 
# 集群规模:
#   - 16 节点 × 8 GPU = 128 GPU
#   - 训练: actor-num-nodes 可按需调整
#   - 推理: rollout-num-gpus 可按需调整
# =============================================================================

set -euo pipefail

# =============================================================================
# 颜色 & 日志
# =============================================================================
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC}  $(date '+%H:%M:%S') $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date '+%H:%M:%S') $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $1"; }
log_step()  { echo -e "${CYAN}[STEP]${NC}  $(date '+%H:%M:%S') $1"; }

# =============================================================================
# 集群节点配置
# =============================================================================
# 第一个节点作为 head 节点
HEAD_NODE="192.168.16.127"
WORKER_NODES=(
    "192.168.16.109"
    "192.168.16.114"
    "192.168.16.122"
    "192.168.16.112"
    "192.168.16.121"
    "192.168.16.105"
    "192.168.16.129"
    "192.168.16.106"
    "192.168.16.88"
    "192.168.16.83"
    "192.168.16.84"
    "192.168.16.93"
    "192.168.16.80"
    "192.168.16.92"
    "192.168.16.96"
)
ALL_NODES=("${HEAD_NODE}" "${WORKER_NODES[@]}")

export MEMORY_RL_TASK_VERSION=t2_agent_loop
export MEMORY_RL_APPLY_MODE=task_loop
export MEMORY_RL_TRAIN_TASK_LOOP=1
# =============================================================================
# Docker 配置
# =============================================================================
# 使用 rl_snapshot tag 以区分原始 slimerl/slime:latest 镜像（不会删除原镜像）
DOCKER_IMAGE="slimerl/slime:rl_snapshot"
DOCKER_TAR="/data/cloud_disk_1/slime_rl_snapshot.tar"
CONTAINER_NAME="slime_rl_ingest"
# 强制重新 load 镜像（每次不跳过 docker 时都会重新加载 tar）
FORCE_RELOAD_IMAGE=true
RAY_PORT=6379
RAY_DASHBOARD_PORT=8265
RAY_HEAD_ADDRESS="${HEAD_NODE}:${RAY_PORT}"

# 容器内挂载点 (挂载整个 /data 目录，涵盖所有 cloud_disk)
MOUNT_HOST_DATA="/data"
MOUNT_CONTAINER_DATA="/data"

# =============================================================================
# 训练配置
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SLIME_TRAIN_ROOT="${SLIME_TRAIN_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)}"
LLM_GATEWAY_ROOT="${LLM_GATEWAY_ROOT:-$(cd "${SLIME_TRAIN_ROOT}/../.." && pwd)}"
LLM_GATEWAY_PARENT="${LLM_GATEWAY_PARENT:-$(cd "${LLM_GATEWAY_ROOT}/.." && pwd)}"

# 模型路径
HF_CHECKPOINT="${HF_CHECKPOINT:-/data/cloud_disk_1/erenpeng/models/Qwen/Qwen3.6-27B}"
TORCH_DIST_CHECKPOINT="${TORCH_DIST_CHECKPOINT:-/data/cloud_disk_4/megatron_rl_checkpoints/qwen36-27b-combo_v4_claude_sft_ckpt450/iter_0000450}"
# FROZEN_MODEL_PATH="${FROZEN_MODEL_PATH:-/data/cloud_disk_1/changyuchen/memory_ai_RL/model/iter_0000189_hf}"
FROZEN_MODEL_PORT="${FROZEN_MODEL_PORT:-30000}"
FROZEN_MODEL_URL="${FROZEN_MODEL_URL:-http://124.221.221.186:30000/v1/chat/completions}"
FROZEN_MODEL_NAME="${FROZEN_MODEL_NAME:-glm-5.1}"
FROZEN_MODEL_TIMEOUT="${FROZEN_MODEL_TIMEOUT:-60}"
FROZEN_MODEL_MAX_CONCURRENCY="${FROZEN_MODEL_MAX_CONCURRENCY:-32}"
FROZEN_MODEL_CONN_LIMIT="${FROZEN_MODEL_CONN_LIMIT:-128}"
FROZEN_MODEL_MAX_TOKENS="${FROZEN_MODEL_MAX_TOKENS:-128000}"
FROZEN_MODEL_MAX_INPUT_TOKENS="${FROZEN_MODEL_MAX_INPUT_TOKENS:-16384}"
PROBE_EVAL_MAX_CONCURRENCY="${PROBE_EVAL_MAX_CONCURRENCY:-8}"

# 数据路径
# INPUT_RL_DATA="${INPUT_RL_DATA:-/data/cloud_disk_1/erenpeng/datasets/merged_stage1_e2e/train/rl_data.jsonl}"
SNAPSHOT_DATA_ROOT="${SNAPSHOT_DATA_ROOT:-/data/cloud_disk_1/erenpeng/datasets/merged_stage1_e2e}"

TRAIN_VAL_DATA="${TRAIN_VAL_DATA:-/data/cloud_disk_1/erenpeng/datasets/merged_stage1_e2e/slime_output/ingest_merged_stage1_e2e_train.clean.jsonl}"
TRAIN_VAL_SPLIT="${TRAIN_VAL_SPLIT:-0.9}"  # 训练集占比，验证集为 1 - TRAIN_VAL_SPLIT

TRAIN_DATA="${TRAIN_DATA:-}"
EVAL_DATA="${EVAL_DATA:-}"

# RL 训练参数
MEMORY_RL_MAX_AGENT_TURNS="${MEMORY_RL_MAX_AGENT_TURNS:-20}"
MEMORY_RL_TASK_VERSION="${MEMORY_RL_TASK_VERSION:-atomic_code_t2}"
MEMORY_RL_APPLY_MODE="${MEMORY_RL_APPLY_MODE:-tool_calls}"

# 集群训练规模 (可按 GPU 总量调整)
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-8}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-48}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"

MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"
SLIME_ROOT="${SLIME_ROOT:-/root/slime}"
MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-/data/cloud_disk_1/changyuchen/agent_memory_RLVR/slime_train/scripts/model/qwen3.6-27B.sh}"

source "${MODEL_CONFIG_SCRIPT}"


# 在宿主机加载 MODEL_ARGS，以便 heredoc 展开时变量有值
if [ -f "${MODEL_CONFIG_SCRIPT}" ]; then
    source "${MODEL_CONFIG_SCRIPT}"
    log_info "MODEL_CONFIG_SCRIPT loaded: ${MODEL_CONFIG_SCRIPT}"
    log_info "MODEL_ARGS (${#MODEL_ARGS[@]} items): ${MODEL_ARGS[*]}"
else
    log_error "模型配置脚本不存在: ${MODEL_CONFIG_SCRIPT}"
    exit 1
fi

# SSH 配置
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes"
SSH_PARALLEL_LIMIT=8

# =============================================================================
# 命令行参数解析
# =============================================================================
SKIP_DOCKER=false
SKIP_RAY=false
SKIP_DEPS=false
DRY_RUN=false
EXTRA_TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-docker)    SKIP_DOCKER=true; shift ;;
        --skip-ray)       SKIP_RAY=true; shift ;;
        --skip-deps)      SKIP_DEPS=true; shift ;;
        --dry-run)        DRY_RUN=true; shift ;;
        --task-version)   MEMORY_RL_TASK_VERSION="$2"; shift 2 ;;
        --actor-num-nodes) ACTOR_NUM_NODES="$2"; shift 2 ;;
        --rollout-num-gpus) ROLLOUT_NUM_GPUS="$2"; shift 2 ;;
        --num-rollout)    NUM_ROLLOUT="$2"; shift 2 ;;
        --head-node)      HEAD_NODE="$2"; shift 2 ;;
        *)                EXTRA_TRAIN_ARGS+=("$1"); shift ;;
    esac
done

# =============================================================================
# 工具函数
# =============================================================================

# 在远程节点执行命令
run_remote() {
    local host="$1"
    shift
    local cmd="$*"
    if [ "${DRY_RUN}" = true ]; then
        echo "[DRY-RUN] ssh ${host}: ${cmd}"
        return 0
    fi
    ssh ${SSH_OPTS} root@"${host}" "${cmd}"
}

# 在远程节点的容器内执行命令
run_in_container() {
    local host="$1"
    shift
    local cmd="$*"
    if [ "${DRY_RUN}" = true ]; then
        echo "[DRY-RUN] ssh ${host} docker exec ${CONTAINER_NAME}: ${cmd}"
        return 0
    fi
    # 使用 printf %q 正确转义命令中的特殊字符（单引号、双引号等）
    local escaped_cmd
    escaped_cmd=$(printf '%s' "${cmd}" | sed "s/'/'\\\\''/g")
    ssh ${SSH_OPTS} root@"${host}" "docker exec ${CONTAINER_NAME} bash -c '${escaped_cmd}'"
}

# 并行在多个节点执行命令
run_parallel() {
    local cmd="$1"
    shift
    local nodes=("$@")
    local pids=()

    for node in "${nodes[@]}"; do
        run_remote "${node}" "${cmd}" &
        pids+=($!)
        # 限制并发数
        if [ ${#pids[@]} -ge ${SSH_PARALLEL_LIMIT} ]; then
            wait "${pids[0]}"
            pids=("${pids[@]:1}")
        fi
    done

    # 等待所有完成
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            ((failed++))
        fi
    done
    return ${failed}
}

# =============================================================================
# Step 0: 连通性检查
# =============================================================================
check_connectivity() {
    log_step "===== Step 0: 检查节点连通性 (${#ALL_NODES[@]} 节点并行) ====="
    local _conn_pids=()
    local _conn_nodes=()
    local _conn_results_dir=$(mktemp -d)

    for node in "${ALL_NODES[@]}"; do
        (
            if ssh ${SSH_OPTS} root@"${node}" "echo ok" > /dev/null 2>&1; then
                echo "ok" > "${_conn_results_dir}/${node}"
            else
                echo "fail" > "${_conn_results_dir}/${node}"
            fi
        ) &
        _conn_pids+=($!)
        _conn_nodes+=("${node}")
    done

    # 等待并展示进度
    local _conn_done=0
    local _conn_total=${#_conn_pids[@]}
    for pid in "${_conn_pids[@]}"; do
        wait "${pid}" 2>/dev/null || true
        ((_conn_done++)) || true
        printf "\r${GREEN}[INFO]${NC}  $(date '+%H:%M:%S') 连通性检查进度: [${_conn_done}/${_conn_total}]"
    done
    echo ""

    # 收集结果
    local failed_nodes=()
    for node in "${_conn_nodes[@]}"; do
        local result
        result=$(cat "${_conn_results_dir}/${node}" 2>/dev/null || echo "fail")
        if [ "${result}" != "ok" ]; then
            failed_nodes+=("${node}")
            log_error "无法连接节点: ${node}"
        fi
    done
    rm -rf "${_conn_results_dir}"

    if [ ${#failed_nodes[@]} -gt 0 ]; then
        log_error "以下节点不可达: ${failed_nodes[*]}"
        log_error "请检查 SSH 配置或节点状态"
        exit 1
    fi

    log_info "所有 ${#ALL_NODES[@]} 个节点连通性检查通过 ✓"
}

# =============================================================================
# Step 0.5: 清理所有节点 GPU 占用进程
# =============================================================================
clear_gpu_processes() {
    log_step "===== Step 0.5: 清理所有节点 GPU 占用进程 ====="
    local _gpu_pids=()
    local _gpu_results_dir=$(mktemp -d)

    for node in "${ALL_NODES[@]}"; do
        (
            if [ "${DRY_RUN}" = true ]; then
                echo "[DRY-RUN] ssh ${node}: 清理 GPU 进程"
                echo "ok" > "${_gpu_results_dir}/${node}"
            else
                # 杀掉所有占用 GPU 的进程（nvidia-smi 列出的 PID）
                local gpu_pids
                gpu_pids=$(ssh ${SSH_OPTS} root@"${node}" \
                    "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u" 2>/dev/null || true)
                if [ -n "${gpu_pids}" ]; then
                    # 逐个 kill，忽略已不存在的进程
                    ssh ${SSH_OPTS} root@"${node}" \
                        "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u | xargs -r kill -9 2>/dev/null; true"
                    echo "killed" > "${_gpu_results_dir}/${node}"
                else
                    echo "clean" > "${_gpu_results_dir}/${node}"
                fi
            fi
        ) &
        _gpu_pids+=($!)
    done

    # 等待所有节点完成
    local _gpu_done=0
    local _gpu_total=${#_gpu_pids[@]}
    for pid in "${_gpu_pids[@]}"; do
        wait "${pid}" 2>/dev/null || true
        ((_gpu_done++)) || true
        printf "\r${GREEN}[INFO]${NC}  $(date '+%H:%M:%S') GPU 清理进度: [${_gpu_done}/${_gpu_total}]"
    done
    echo ""

    # 打印结果
    local _killed_count=0
    local _clean_count=0
    for node in "${ALL_NODES[@]}"; do
        local result
        result=$(cat "${_gpu_results_dir}/${node}" 2>/dev/null || echo "unknown")
        if [ "${result}" = "killed" ]; then
            log_info "节点 ${node}: GPU 进程已清理"
            ((_killed_count++)) || true
        elif [ "${result}" = "clean" ]; then
            ((_clean_count++)) || true
        fi
    done
    rm -rf "${_gpu_results_dir}"

    if [ ${_killed_count} -gt 0 ]; then
        log_info "已清理 ${_killed_count} 个节点的 GPU 进程，${_clean_count} 个节点无需清理"
        # 等待几秒让 GPU 显存完全释放
        log_info "等待 5 秒让 GPU 显存完全释放..."
        sleep 5
    else
        log_info "所有 ${#ALL_NODES[@]} 个节点 GPU 均无占用，无需清理 ✓"
    fi
}

# =============================================================================
# Step 1: Docker 镜像加载 & 容器启动
# =============================================================================
setup_docker() {
    log_step "===== Step 1: Docker 镜像 & 容器配置 ====="

    # 如果 FORCE_RELOAD_IMAGE=true，则跳过检查，所有节点都强制重新加载
    if [ "${FORCE_RELOAD_IMAGE}" = true ]; then
        log_info "FORCE_RELOAD_IMAGE=true, 所有节点将强制重新加载镜像"
        local nodes_need_load=("${ALL_NODES[@]}")
    else
        # 并行检查哪些节点需要加载镜像
        log_info "检查各节点镜像状态 (${#ALL_NODES[@]} 节点并行)..."
        local _img_pids=()
        local _img_results_dir=$(mktemp -d)

        for node in "${ALL_NODES[@]}"; do
            (
                local has_image
                has_image=$(run_remote "${node}" "docker images --format '{{.Repository}}:{{.Tag}}' | grep -c '${DOCKER_IMAGE}' || true")
                if [ "${has_image}" = "0" ] || [ -z "${has_image}" ]; then
                    echo "need" > "${_img_results_dir}/${node}"
                else
                    echo "ok" > "${_img_results_dir}/${node}"
                fi
            ) &
            _img_pids+=($!)
        done

        local _img_done=0
        local _img_total=${#_img_pids[@]}
        for pid in "${_img_pids[@]}"; do
            wait "${pid}" 2>/dev/null || true
            ((_img_done++)) || true
            printf "\r${GREEN}[INFO]${NC}  $(date '+%H:%M:%S') 镜像检查进度: [${_img_done}/${_img_total}]"
        done
        echo ""

        local nodes_need_load=()
        local _img_ok_count=0
        for node in "${ALL_NODES[@]}"; do
            local result
            result=$(cat "${_img_results_dir}/${node}" 2>/dev/null || echo "need")
            if [ "${result}" = "need" ]; then
                nodes_need_load+=("${node}")
                log_warn "节点 ${node} 缺少镜像 ${DOCKER_IMAGE}"
            else
                ((_img_ok_count++)) || true
            fi
        done
        rm -rf "${_img_results_dir}"

        if [ ${_img_ok_count} -gt 0 ]; then
            log_info "${_img_ok_count} 个节点镜像已存在 ✓"
        fi
    fi

    # 并行加载镜像
    if [ ${#nodes_need_load[@]} -gt 0 ]; then
        log_step "在 ${#nodes_need_load[@]} 个节点上加载镜像 (${DOCKER_TAR})..."
        log_info "镜像文件大小: $(ls -lh ${DOCKER_TAR} | awk '{print $5}')"
        log_info "目标镜像: ${DOCKER_IMAGE} (不影响 slimerl/slime:latest)"
        log_info "并行加载中，请耐心等待..."

        local pids=()
        local _load_nodes=()
        for node in "${nodes_need_load[@]}"; do
            (
                # 加载 tar 镜像（tar 中可能是 slimerl/slime:latest）
                local loaded_img
                loaded_img=$(run_remote "${node}" "docker load -i ${DOCKER_TAR} 2>/dev/null | grep 'Loaded image' | sed 's/Loaded image: //'")
                # 如果加载的镜像名和目标不同，进行 retag
                if [ -n "${loaded_img}" ] && [ "${loaded_img}" != "${DOCKER_IMAGE}" ]; then
                    run_remote "${node}" "docker tag ${loaded_img} ${DOCKER_IMAGE}" 2>/dev/null || true
                fi
            ) &
            pids+=($!)
            _load_nodes+=("${node}")
            # 限制并发，避免共享存储带宽过载
            if [ ${#pids[@]} -ge 4 ]; then
                wait "${pids[0]}" 2>/dev/null || true
                pids=("${pids[@]:1}")
            fi
        done

        local _load_done=0
        local _load_total=${#_load_nodes[@]}
        for pid in "${pids[@]}"; do
            wait "${pid}" 2>/dev/null || log_warn "某节点镜像加载可能失败"
            ((_load_done++)) || true
            printf "\r${GREEN}[INFO]${NC}  $(date '+%H:%M:%S') 镜像加载进度: [${_load_done}/${_load_total}]"
        done
        echo ""

        # 验证加载结果
        for node in "${nodes_need_load[@]}"; do
            local verify
            verify=$(run_remote "${node}" "docker images --format '{{.Repository}}:{{.Tag}}' | grep -c '${DOCKER_IMAGE}' || true")
            if [ "${verify}" = "0" ] || [ -z "${verify}" ]; then
                log_error "节点 ${node} 镜像加载失败!"
                exit 1
            fi
        done
        log_info "所有节点镜像加载完成 ✓"
    else
        log_info "所有节点镜像已就绪，无需加载"
    fi

    # 启动容器（并行，强制销毁同名容器后重建）
    log_step "强制重建 Docker 容器 (${#ALL_NODES[@]} 节点并行)..."

    local _start_pids=()
    local _start_nodes=()

    for node in "${ALL_NODES[@]}"; do
        # 后台并行: 强制删除旧容器 + 启动新容器
        (
            run_remote "${node}" "docker rm -f ${CONTAINER_NAME} 2>/dev/null || true"
            run_remote "${node}" "docker run -d \
                --name ${CONTAINER_NAME} \
                --network host \
                --ipc host \
                --gpus all \
                --privileged \
                --ulimit memlock=-1 \
                --ulimit stack=67108864 \
                -v ${MOUNT_HOST_DATA}:${MOUNT_CONTAINER_DATA} \
                -v /root/.ssh:/root/.ssh:ro \
                -e NVIDIA_VISIBLE_DEVICES=all \
                -e NCCL_IB_DISABLE=0 \
                -e NCCL_IB_GID_INDEX=3 \
                -e NCCL_IB_HCA=mlx5 \
                -e NCCL_NET_GDR_LEVEL=5 \
                -e NCCL_SOCKET_IFNAME=eth0 \
                -e GLOO_SOCKET_IFNAME=eth0 \
                -w /root \
                ${DOCKER_IMAGE} \
                sleep infinity" > /dev/null
        ) &
        _start_pids+=($!)
        _start_nodes+=("${node}")
    done

    # 等待并行启动完成，展示进度
    if [ ${#_start_pids[@]} -gt 0 ]; then
        local _done=0
        local _total=${#_start_pids[@]}
        local _failed_nodes=()

        for i in "${!_start_pids[@]}"; do
            if wait "${_start_pids[$i]}"; then
                ((_done++)) || true
            else
                ((_done++)) || true
                _failed_nodes+=("${_start_nodes[$i]}")
            fi
            printf "\r${GREEN}[INFO]${NC}  $(date '+%H:%M:%S') 容器启动进度: [${_done}/${_total}]"
        done
        echo ""  # 换行

        # 验证所有容器
        if [ ${#_failed_nodes[@]} -gt 0 ]; then
            for node in "${_failed_nodes[@]}"; do
                log_error "节点 ${node} 容器启动失败!"
                run_remote "${node}" "docker logs ${CONTAINER_NAME} 2>&1 | tail -5" || true
            done
            exit 1
        fi

        # 快速批量验证
        local _verify_failed=()
        for node in "${_start_nodes[@]}"; do
            local check
            check=$(run_remote "${node}" "docker ps --format '{{.Names}}' | grep -c '^${CONTAINER_NAME}$' || true")
            if [ "${check}" != "1" ]; then
                _verify_failed+=("${node}")
            fi
        done
        if [ ${#_verify_failed[@]} -gt 0 ]; then
            log_error "以下节点容器验证失败: ${_verify_failed[*]}"
            exit 1
        fi
    fi
    log_info "所有节点容器启动完成 ✓ (${#_start_nodes[@]} 节点)"
}

# =============================================================================
# Step 1.5: 依赖安装 (独立于 Docker 启动，始终执行)
# =============================================================================
install_dependencies() {
    if [ "${SKIP_DEPS}" = true ]; then
        log_info "跳过依赖安装 (--skip-deps)"
        return 0
    fi

    log_step "===== Step 1.5: 检查并安装训练依赖 (${#ALL_NODES[@]} 节点并行) ====="
    local REQUIREMENTS_FILE="/data/cloud_disk_1/erenpeng/llm-gateway/requirements.txt"
    if [ ! -f "${REQUIREMENTS_FILE}" ]; then
        log_warn "requirements.txt 未找到: ${REQUIREMENTS_FILE}，使用备用依赖列表"
        REQUIREMENTS_FILE=""
    fi

    local _dep_pids=()
    for node in "${ALL_NODES[@]}"; do
        (
            # 先安装 requirements.txt 中的所有依赖
            if [ -n "${REQUIREMENTS_FILE}" ]; then
                run_in_container "${node}" "pip install  -i https://mirrors.cloud.tencent.com/pypi/simple -r ${REQUIREMENTS_FILE} -q 2>/dev/null || true"
            fi
            # 额外安装不在 requirements.txt 中但训练需要的包
            # psycopg-binary 提供预编译的 libpq，避免容器中缺少系统 libpq 库的问题
            run_in_container "${node}" "pip install  -i https://mirrors.cloud.tencent.com/pypi/simple zstandard psycopg-binary aiohttp swanlab -q 2>/dev/null || true"
        ) &
        _dep_pids+=($!)
    done

    local _dep_done=0
    local _dep_total=${#_dep_pids[@]}
    for pid in "${_dep_pids[@]}"; do
        wait "${pid}" 2>/dev/null || true
        ((_dep_done++)) || true
        printf "\r${GREEN}[INFO]${NC}  $(date '+%H:%M:%S') 依赖安装进度: [${_dep_done}/${_dep_total}]"
    done
    echo ""
    log_info "依赖安装完成 ✓"
}

# =============================================================================
# Step 2: Ray 集群启动
# =============================================================================
setup_ray_cluster() {
    log_step "===== Step 2: 启动 Ray 集群 ====="

    # 先停止所有节点上可能存在的 Ray 进程
    log_info "清理旧的 Ray 进程..."
    for node in "${ALL_NODES[@]}"; do
        run_in_container "${node}" "ray stop --force 2>/dev/null || true" &
    done
    wait
    sleep 2

    # 启动 head 节点
    log_step "启动 Ray Head 节点: ${HEAD_NODE}"
    run_in_container "${HEAD_NODE}" \
        "ray start --head \
            --port=${RAY_PORT} \
            --dashboard-port=${RAY_DASHBOARD_PORT} \
            --dashboard-host=0.0.0.0 \
            --num-gpus=8 \
            --num-cpus=64 \
            --resources='{\"head\": 1}' \
            --block" &
    RAY_HEAD_PID=$!

    # 等待 head 就绪
    log_info "等待 Ray Head 就绪..."
    local max_wait=60
    local elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if run_in_container "${HEAD_NODE}" "ray status 2>/dev/null | grep -q 'Active'" 2>/dev/null; then
            log_info "Ray Head 就绪! (${elapsed}s)"
            break
        fi
        sleep 3
        elapsed=$((elapsed + 3))
    done

    if [ $elapsed -ge $max_wait ]; then
        log_error "Ray Head 启动超时"
        exit 1
    fi

    # 启动 worker 节点
    log_step "启动 ${#WORKER_NODES[@]} 个 Ray Worker 节点..."
    for node in "${WORKER_NODES[@]}"; do
        log_info "  连接 Worker: ${node} → ${RAY_HEAD_ADDRESS}"
        run_in_container "${node}" \
            "ray start \
                --address=${RAY_HEAD_ADDRESS} \
                --num-gpus=8 \
                --num-cpus=64 \
                --block" &
    done

    # 等待所有 worker 连接
    sleep 10
    log_info "等待所有 Worker 连接..."
    local max_wait_workers=120
    local elapsed=0
    local expected_gpus=$(( ${#ALL_NODES[@]} * 8 ))

    while [ $elapsed -lt $max_wait_workers ]; do
        local gpu_count
        # Ray 2.x 格式: " X.0/Y.0 GPU"，提取总量 Y（分母）表示已注册 GPU 数
        gpu_count=$(run_in_container "${HEAD_NODE}" "ray status 2>/dev/null | grep -oP '[0-9.]+/\K[0-9.]+(?=\s+GPU)'" 2>/dev/null || echo "0")
        # 取整数部分
        gpu_count=${gpu_count%%.*}
        if [ "${gpu_count:-0}" -ge "${expected_gpus}" ]; then
            log_info "所有 Worker 已连接! 总 GPU: ${gpu_count}/${expected_gpus}"
            break
        fi
        log_info "  当前 GPU: ${gpu_count:-0}/${expected_gpus}, 等待中..."
        sleep 5
        elapsed=$((elapsed + 5))
    done

    # 打印 Ray 集群状态
    log_info "Ray 集群状态:"
    run_in_container "${HEAD_NODE}" "ray status" 2>/dev/null || true
    log_info "Ray Dashboard: http://${HEAD_NODE}:${RAY_DASHBOARD_PORT}"
    log_info "Ray 集群启动完成 ✓"
}

# =============================================================================
# Step 3: 启动冻结模型推理引擎
# =============================================================================
start_frozen_model_service() {
    log_step "===== Step 3: 检查冻结模型 API ====="

    log_info "使用外部冻结模型 API (不在集群本地加载)"
    log_info "  URL: ${FROZEN_MODEL_URL}"
    log_info "  模型名称: ${FROZEN_MODEL_NAME}"

    # 检查外部 endpoint 是否可用 (尝试 /health 和 /v1/models 两种健康检查)
    local base_url="${FROZEN_MODEL_URL%/v1/chat/completions}"
    if curl -s --connect-timeout 5 "${base_url}/health" > /dev/null 2>&1 || \
       curl -s --connect-timeout 5 "${base_url}/v1/models" > /dev/null 2>&1; then
        log_info "外部冻结模型 API 可用 ✓"
    else
        log_warn "外部冻结模型 API 暂时无法连通 (${base_url})，训练时将直接使用配置的 URL"
        log_warn "如果训练时 API 仍不可达，reward 计算将会失败"
    fi
    return 0
}

# =============================================================================
# Step 4: 准备训练数据
# =============================================================================
prepare_training_data() {
    log_step "===== Step 4: 准备训练数据 ====="

    # 如果已明确指定 TRAIN_DATA 且文件存在，直接使用
    if [ -n "${TRAIN_DATA}" ] && [ -f "${TRAIN_DATA}" ]; then
        log_info "训练数据已存在: ${TRAIN_DATA}"
        return 0
    fi

    # 从 TRAIN_VAL_DATA 按比例拆分
    if [ -n "${TRAIN_VAL_DATA}" ]; then
        log_info "从 TRAIN_VAL_DATA 按比例 (${TRAIN_VAL_SPLIT}) 拆分训练/验证集..."

        # 确定源文件：支持目录(取目录下所有 .jsonl) 或单个文件
        local SRC_FILES=""
        if [ -d "${TRAIN_VAL_DATA}" ]; then
            SRC_FILES=$(find "${TRAIN_VAL_DATA}" -name "*.clean.jsonl" -type f | sort)
            if [ -z "${SRC_FILES}" ]; then
                SRC_FILES=$(find "${TRAIN_VAL_DATA}" -name "*.jsonl" -type f | sort)
            fi
        elif [ -f "${TRAIN_VAL_DATA}" ]; then
            SRC_FILES="${TRAIN_VAL_DATA}"
        else
            log_error "TRAIN_VAL_DATA 路径不存在: ${TRAIN_VAL_DATA}"
            exit 1
        fi

        # 合并所有源文件到临时文件
        local MERGED_FILE="/tmp/train_val_merged_$$.jsonl"
        cat ${SRC_FILES} > "${MERGED_FILE}"
        local TOTAL_LINES=$(wc -l < "${MERGED_FILE}")
        local TRAIN_LINES=$(python3 -c "import math; print(math.floor(${TOTAL_LINES} * ${TRAIN_VAL_SPLIT}))")
        local VAL_LINES=$((TOTAL_LINES - TRAIN_LINES))

        log_info "  总样本数: ${TOTAL_LINES}, 训练集: ${TRAIN_LINES}, 验证集: ${VAL_LINES}"

        # 打乱并拆分
        local SHUFFLED_FILE="/tmp/train_val_shuffled_$$.jsonl"
        shuf "${MERGED_FILE}" > "${SHUFFLED_FILE}"

        # 设置输出路径
        local OUTPUT_DIR="${SNAPSHOT_DATA_ROOT}/slime_output"
        mkdir -p "${OUTPUT_DIR}"
        TRAIN_DATA="${OUTPUT_DIR}/ingest_train_split.jsonl"
        EVAL_DATA="${OUTPUT_DIR}/ingest_val_split.jsonl"

        head -n ${TRAIN_LINES} "${SHUFFLED_FILE}" > "${TRAIN_DATA}"
        tail -n ${VAL_LINES} "${SHUFFLED_FILE}" > "${EVAL_DATA}"

        # 清理临时文件
        rm -f "${MERGED_FILE}" "${SHUFFLED_FILE}"

        log_info "拆分完成:"
        log_info "  训练集: ${TRAIN_DATA} (${TRAIN_LINES} 条)"
        log_info "  验证集: ${EVAL_DATA} (${VAL_LINES} 条)"
        return 0
    fi

    # 回退: 使用 convert_to_slime_format.py 生成
    log_info "生成训练数据..."
    local OUTPUT_DIR="${SNAPSHOT_DATA_ROOT}/slime_output"
    mkdir -p "${OUTPUT_DIR}"
    TRAIN_DATA="${OUTPUT_DIR}/ingest_merged_stage1_e2e_train.jsonl"
    EVAL_DATA="${OUTPUT_DIR}/ingest_merged_stage1_e2e_val.jsonl"

    run_in_container "${HEAD_NODE}" \
        "cd ${PROJECT_ROOT} && \
         export PYTHONPATH=${MEGATRON_ROOT}:${LLM_GATEWAY_PARENT}:${LLM_GATEWAY_ROOT} && \
         python3 convert_to_slime_format.py \
            --input ${INPUT_RL_DATA} \
            --data-root ${SNAPSHOT_DATA_ROOT} \
            --output ${TRAIN_DATA} \
            --eval_output ${EVAL_DATA}"

    if [ ! -f "${TRAIN_DATA}" ]; then
        log_error "训练数据生成失败: ${TRAIN_DATA}"
        exit 1
    fi
    log_info "训练数据准备完成 ✓"
}

# =============================================================================
# Step 5: 提交训练任务
# =============================================================================
submit_training_job() {
    log_step "===== Step 5: 提交 GRPO 训练任务 ====="

    local RUN_NAME="ingest_grpo_$(date '+%Y%m%d_%H%M%S')"
    local CKPT_DIR="${PROJECT_ROOT}/outputs/${RUN_NAME}/checkpoints"
    local DUMP_DIR="${PROJECT_ROOT}/outputs/${RUN_NAME}/rollout_dumps"
    local LOG_DIR="${PROJECT_ROOT}/outputs/${RUN_NAME}/logs"

    # 创建输出目录
    run_in_container "${HEAD_NODE}" "mkdir -p ${CKPT_DIR} ${DUMP_DIR} ${LOG_DIR}"

    # 仅当 CKPT_DIR 内有已保存的 checkpoint 时才传 --load (首次训练为空目录，跳过)
    local LOAD_ARG=""
    if run_in_container "${HEAD_NODE}" "ls ${CKPT_DIR}/latest_checkpointed_iteration.txt" &>/dev/null; then
        LOAD_ARG="--load ${CKPT_DIR}"
    fi

    log_info "训练配置:"
    log_info "  RUN_NAME:              ${RUN_NAME}"
    log_info "  TASK_VERSION:          ${MEMORY_RL_TASK_VERSION}"
    log_info "  ACTOR_NUM_NODES:       ${ACTOR_NUM_NODES}"
    log_info "  ACTOR_GPUS_PER_NODE:   ${ACTOR_NUM_GPUS_PER_NODE}"
    log_info "  ROLLOUT_NUM_GPUS:      ${ROLLOUT_NUM_GPUS}"
    log_info "  NUM_ROLLOUT:           ${NUM_ROLLOUT}"
    log_info "  GLOBAL_BATCH_SIZE:     ${GLOBAL_BATCH_SIZE}"
    log_info "  FROZEN_MODEL_URL:      ${FROZEN_MODEL_URL}"

    # 提交 Ray Job
    log_step "提交 Ray Job..."
    local RUNTIME_ENV_JSON_FILE="/tmp/runtime_env_${RUN_NAME}.json"
    local TRAIN_SCRIPT="/tmp/run_train_${RUN_NAME}.sh"

    log_info "执行训练命令..."
    if [ "${DRY_RUN}" = true ]; then
        echo ""
        echo "========== DRY-RUN: 训练命令 =========="
        echo "  脚本: ${TRAIN_SCRIPT}"
        echo "  JSON: ${RUNTIME_ENV_JSON_FILE}"
        echo "  ray job submit ... -- python3 train_async.py --actor-num-nodes ${ACTOR_NUM_NODES} ..."
        echo "======================================="
        return 0
    fi

    # 将 runtime env JSON 写入容器内文件 (避免命令行引号问题)
    ssh ${SSH_OPTS} root@"${HEAD_NODE}" "docker exec -i ${CONTAINER_NAME} tee ${RUNTIME_ENV_JSON_FILE} > /dev/null" <<JSON_EOF
{"env_vars":{"PYTHONPATH":"${MEGATRON_ROOT}:${LLM_GATEWAY_PARENT}:${LLM_GATEWAY_ROOT}","PYTHONUNBUFFERED":"1","CUDA_DEVICE_MAX_CONNECTIONS":"1","PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True","NCCL_NVLS_ENABLE":"1","NCCL_IB_DISABLE":"0","NCCL_IB_GID_INDEX":"3","NCCL_IB_HCA":"mlx5","NCCL_NET_GDR_LEVEL":"5","NCCL_TIMEOUT_MS":"600000","NCCL_SOCKET_IFNAME":"eth0","GLOO_SOCKET_IFNAME":"eth0","NCCL_DEBUG":"WARN","TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC":"600","MASTER_ADDR":"${HEAD_NODE}","no_proxy":"127.0.0.1,${HEAD_NODE}","INGEST_SNAPSHOT_DATA_ROOT":"${SNAPSHOT_DATA_ROOT}","SNAPSHOT_DATA_ROOT":"${SNAPSHOT_DATA_ROOT}","FROZEN_MODEL_URL":"${FROZEN_MODEL_URL}","FROZEN_MODEL_NAME":"${FROZEN_MODEL_NAME}","FROZEN_MODEL_TIMEOUT":"${FROZEN_MODEL_TIMEOUT}","FROZEN_MODEL_MAX_CONCURRENCY":"${FROZEN_MODEL_MAX_CONCURRENCY}","FROZEN_MODEL_CONN_LIMIT":"${FROZEN_MODEL_CONN_LIMIT}","FROZEN_MODEL_MAX_TOKENS":"${FROZEN_MODEL_MAX_TOKENS}","FROZEN_MODEL_MAX_INPUT_TOKENS":"${FROZEN_MODEL_MAX_INPUT_TOKENS}","PROBE_EVAL_MAX_CONCURRENCY":"${PROBE_EVAL_MAX_CONCURRENCY}","REWARD_METRICS_LOG":"${LOG_DIR}/reward_metrics.jsonl","MEMORY_RL_MAX_AGENT_TURNS":"${MEMORY_RL_MAX_AGENT_TURNS}","MEMORY_RL_TASK_VERSION":"${MEMORY_RL_TASK_VERSION}","MEMORY_RL_APPLY_MODE":"${MEMORY_RL_APPLY_MODE}","MEMORY_RL_LLM_API_URL":"${FROZEN_MODEL_URL}","MEMORY_RL_LLM_MODEL":"${FROZEN_MODEL_NAME}"}}
JSON_EOF

    # 启动 SwanLab 监控
    log_info "启动 SwanLab 监控..."
    run_in_container "${HEAD_NODE}" \
        "cd ${PROJECT_ROOT}/scripts && \
         export PYTHONPATH=${MEGATRON_ROOT}:${LLM_GATEWAY_PARENT}:${LLM_GATEWAY_ROOT} && \
         export SWANLAB_API_KEY=${SWANLAB_API_KEY:-BVQaRTEEZKWC9p3iF5MMp} && \
         nohup python3 swanlab_monitor.py \
            --log-dir ${LOG_DIR} \
            --task-version ${MEMORY_RL_TASK_VERSION} \
            --poll-interval 5.0 \
            > ${LOG_DIR}/swanlab_monitor.log 2>&1 &" || true

    # 将完整训练命令写入容器内脚本文件 (避免 bash -c 嵌套引号问题)
    ssh ${SSH_OPTS} root@"${HEAD_NODE}" "docker exec -i ${CONTAINER_NAME} tee ${TRAIN_SCRIPT} > /dev/null" << 'TRAIN_SCRIPT_BOUNDARY'
#!/bin/bash
set -e
TRAIN_SCRIPT_BOUNDARY

    # 用非引号 heredoc 写入变量展开的部分
    ssh ${SSH_OPTS} root@"${HEAD_NODE}" "docker exec -i ${CONTAINER_NAME} tee -a ${TRAIN_SCRIPT} > /dev/null" <<TRAIN_SCRIPT_EOF
source ${MODEL_CONFIG_SCRIPT}
export PYTHONPATH=${MEGATRON_ROOT}:${LLM_GATEWAY_PARENT}:${LLM_GATEWAY_ROOT}
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=${HEAD_NODE}
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-BVQaRTEEZKWC9p3iF5MMp}
cd ${SLIME_ROOT}

# 读取 runtime env JSON
RUNTIME_ENV=\$(cat ${RUNTIME_ENV_JSON_FILE})

# 调试: 检查关键路径是否可达
echo "=== DEBUG: 检查路径 ==="
echo "HF_CHECKPOINT: ${HF_CHECKPOINT}"
ls -la ${HF_CHECKPOINT}/config.json 2>&1 || echo "ERROR: config.json not found!"
echo "REF_LOAD: ${TORCH_DIST_CHECKPOINT}"
ls -la ${TORCH_DIST_CHECKPOINT}/ 2>&1 | head -5 || echo "ERROR: ref-load path not found!"
echo "========================"

ray job submit --address=http://127.0.0.1:${RAY_DASHBOARD_PORT} \\
    --runtime-env-json="\${RUNTIME_ENV}" \\
    -- python3 ${SLIME_ROOT}/train_async.py \\
    --actor-num-nodes ${ACTOR_NUM_NODES} \\
    --actor-num-gpus-per-node ${ACTOR_NUM_GPUS_PER_NODE} \\
    --rollout-num-gpus ${ROLLOUT_NUM_GPUS} \\
    --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE} \\
    --hf-checkpoint "${HF_CHECKPOINT}" \\
    ${MODEL_ARGS[*]} \\
    --ref-load ${TORCH_DIST_CHECKPOINT} \\
    ${LOAD_ARG} \\
    --save ${CKPT_DIR} \\
    --save-interval 20 \\
    --prompt-data ${TRAIN_DATA} \\
    --input-key prompt \\
    --metadata-key metadata \\
    --apply-chat-template \\
    --custom-rm-path llm_gateway.rl.slime_train.tasks.ingest_reward.reward.reward_func \\
    --custom-generate-function-path llm_gateway.rl.slime_train.memory_rl.custom_generate.custom_generate \\
    --num-rollout ${NUM_ROLLOUT} \\
    --rollout-batch-size ${ROLLOUT_BATCH_SIZE} \\
    --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT} \\
    --rollout-max-response-len 10000 \\
    --rollout-max-prompt-len 20000 \\
    --rollout-temperature 1.0 \\
    --global-batch-size ${GLOBAL_BATCH_SIZE} \\
    --balance-data \\
    --tensor-model-parallel-size 4 \\
    --sequence-parallel \\
    --pipeline-model-parallel-size 2 \\
    --context-parallel-size 1 \\
    --recompute-granularity full \\
    --recompute-method uniform \\
    --recompute-num-layers 1 \\
    --use-dynamic-batch-size \\
    --max-tokens-per-gpu 30000 \\
    --advantage-estimator grpo \\
    --use-kl-loss \\
    --kl-loss-coef 0.005 \\
    --kl-loss-type low_var_kl \\
    --entropy-coef 0.0008 \\
    --eps-clip 0.2 \\
    --eps-clip-high 0.28 \\
    --clip-grad 1.0 \\
    --optimizer adam \\
    --lr 5e-7 \\
    --lr-decay-style cosine \\
    --lr-warmup-iters 5 \\
    --min-lr 1e-7 \\
    --weight-decay 0.1 \\
    --adam-beta1 0.9 \\
    --adam-beta2 0.98 \\
    --sglang-mem-fraction-static 0.5 \\
    --sglang-disable-custom-all-reduce \\
    --sglang-disable-cuda-graph \\
    --sglang-watchdog-timeout 1200 \\
    --sglang-router-request-timeout-secs 300 \\
    --no-check-for-nan-in-loss-and-grad \\
    --distributed-timeout-minutes 60 \\
    --no-load-optim \\
    --no-load-rng \\
    --finetune \\
    --dump-details ${DUMP_DIR} \\
    ${EXTRA_TRAIN_ARGS[*]:-} 2>&1 | tee ${LOG_DIR}/train.log
TRAIN_SCRIPT_EOF

    run_in_container "${HEAD_NODE}" "chmod +x ${TRAIN_SCRIPT}"

    # 在 head 容器内执行脚本
    ssh ${SSH_OPTS} root@"${HEAD_NODE}" \
        "docker exec ${CONTAINER_NAME} bash ${TRAIN_SCRIPT}" &
    TRAIN_PID=$!

    log_info "训练已在后台提交 (本地 PID=${TRAIN_PID})"
    log_info "查看训练日志:"
    log_info "  ssh root@${HEAD_NODE} docker exec ${CONTAINER_NAME} tail -f ${LOG_DIR}/train.log"
    log_info "Ray Dashboard:"
    log_info "  http://${HEAD_NODE}:${RAY_DASHBOARD_PORT}"

    # 等待训练完成
    wait ${TRAIN_PID}
    local exit_code=$?

    if [ ${exit_code} -eq 0 ]; then
        log_info "🎉 训练完成! Checkpoint 保存在: ${CKPT_DIR}"
    else
        log_error "训练异常退出 (exit code: ${exit_code})"
        log_error "查看日志: ssh root@${HEAD_NODE} docker exec ${CONTAINER_NAME} cat ${LOG_DIR}/train.log"
        exit ${exit_code}
    fi
}

# =============================================================================
# 清理函数
# =============================================================================
cleanup() {
    log_step "===== 清理资源 ====="
    log_info "停止所有 Ray Job..."
    run_in_container "${HEAD_NODE}" "ray job stop --all --address http://127.0.0.1:${RAY_DASHBOARD_PORT}" 2>/dev/null || true
    log_info "清理完成"
}

# 中断信号处理
trap cleanup SIGINT SIGTERM

# =============================================================================
# 主流程
# =============================================================================
main() {
    echo ""
    log_step "=========================================="
    log_step " Slime 一键集群训练启动"
    log_step "=========================================="
    log_info "集群规模: ${#ALL_NODES[@]} 节点 (${HEAD_NODE} + ${#WORKER_NODES[@]} workers)"
    log_info "总 GPU:   $(( ${#ALL_NODES[@]} * 8 )) (${#ALL_NODES[@]} × 8)"
    log_info "训练任务: 一阶段摄入 (Ingest) GRPO"
    log_info "任务版本: ${MEMORY_RL_TASK_VERSION}"
    echo ""

    # Step 0: 连通性检查
    check_connectivity

    # Step 0.5: 清理所有节点 GPU 进程
    clear_gpu_processes

    # Step 1: Docker 配置
    if [ "${SKIP_DOCKER}" = true ]; then
        log_info "跳过 Docker 配置 (--skip-docker)"
    else
        setup_docker
    fi

    # Step 1.5: 依赖安装 (独立于 Docker，始终检查)
    install_dependencies

    # Step 2: Ray 集群
    if [ "${SKIP_RAY}" = true ]; then
        log_info "跳过 Ray 集群启动 (--skip-ray)"
    else
        setup_ray_cluster
    fi

    # Step 3: 冻结模型
    start_frozen_model_service

    # Step 4: 训练数据
    prepare_training_data

    # Step 5: 提交训练
    submit_training_job

    log_step "=========================================="
    log_step " 全部流程完成! 🎉"
    log_step "=========================================="
}

main "$@"
