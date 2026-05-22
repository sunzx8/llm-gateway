#!/usr/bin/env bash
# =============================================================================
# Fixed 16-idle-node slime GRPO launcher for ingest training.
#
# Fixes vs launch_cluster_train.sh:
#   1) Use 16 idle GPUs from /data/cloud_disk_1/shawnxsun/pre_projects/internal_ip.txt lines 33-65.
#   2) Use an isolated container name and Ray ports to avoid colliding with other clusters.
#   3) Sync the same slime image from head to workers to avoid Ray version mismatch.
#   4) Pass MODEL_ARGS into train_async.py.
#   5) Load from TORCH_DIST_CHECKPOINT, not the newly-created output checkpoint dir.
#   6) Use dense-model EP=1 for Qwen3.6-27B.
#
# Usage:
#   bash scripts/launch_cluster_train_fixed_16idle.sh --background
#   bash scripts/launch_cluster_train_fixed_16idle.sh --dry-run
#   bash scripts/launch_cluster_train_fixed_16idle.sh --skip-image-sync --background
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC}  $(date '+%H:%M:%S') $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date '+%H:%M:%S') $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $*"; }
log_step()  { echo -e "${CYAN}[STEP]${NC}  $(date '+%H:%M:%S') $*"; }

# 16 idle nodes selected from internal_ip.txt lines 33-65, skipping nodes with active vLLM/SGLang services.
HEAD_NODE="${HEAD_NODE:-192.168.16.70}"
WORKER_NODES=(
  "192.168.16.103"
  "192.168.16.102"
  "192.168.16.100"
  "192.168.16.87"
  "192.168.16.85"
  "192.168.16.101"
  "192.168.16.98"
  "192.168.16.81"
  "192.168.16.86"
  "192.168.16.82"
  "192.168.16.99"
  "192.168.16.97"
  "192.168.16.91"
  "192.168.16.95"
  "192.168.16.90"
)
ALL_NODES=("${HEAD_NODE}" "${WORKER_NODES[@]}")

DOCKER_IMAGE="${DOCKER_IMAGE:-slimerl/slime:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-slime_rl_ingest_ep16}"
RAY_PORT="${RAY_PORT:-6387}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8275}"
RAY_HEAD_ADDRESS="${HEAD_NODE}:${RAY_PORT}"
# Avoid collision with other host-network Ray clusters whose dashboard agents often bind 52365.
RAY_AGENT_PORT="${RAY_AGENT_PORT:-52385}"
RAY_AGENT_GRPC_PORT="${RAY_AGENT_GRPC_PORT:-52386}"
RAY_RUNTIME_ENV_AGENT_PORT="${RAY_RUNTIME_ENV_AGENT_PORT:-52387}"
MOUNT_HOST_DATA="${MOUNT_HOST_DATA:-/data}"
MOUNT_CONTAINER_DATA="${MOUNT_CONTAINER_DATA:-/data}"
SSH_OPTS="${SSH_OPTS:--o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes -o LogLevel=ERROR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SLIME_TRAIN_ROOT="${SLIME_TRAIN_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)}"
LLM_GATEWAY_ROOT="${LLM_GATEWAY_ROOT:-$(cd "${SLIME_TRAIN_ROOT}/../.." && pwd)}"
LLM_GATEWAY_PARENT="${LLM_GATEWAY_PARENT:-$(cd "${LLM_GATEWAY_ROOT}/.." && pwd)}"

HF_CHECKPOINT="${HF_CHECKPOINT:-/data/cloud_disk_1/erenpeng/models/Qwen/Qwen3.6-27B}"
TORCH_DIST_CHECKPOINT="${TORCH_DIST_CHECKPOINT:-/data/cloud_disk_4/megatron_rl_checkpoints/qwen36-27b-combo_v4_claude_sft_ckpt450/iter_0000450}"
MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-/data/cloud_disk_1/changyuchen/agent_memory_RLVR/slime_train/scripts/model/qwen3.6-27B.sh}"
TRAIN_DATA="${TRAIN_DATA:-/data/cloud_disk_1/erenpeng/datasets/merged_stage1_e2e/slime_output/ingest_merged_stage1_e2e_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-/data/cloud_disk_1/erenpeng/datasets/merged_stage1_e2e/slime_output/ingest_merged_stage1_e2e_val.jsonl}"
# SnapshotSession expects data_root to be the dataset root; metadata/traj paths already include snapshots/...
SNAPSHOT_DATA_ROOT="${SNAPSHOT_DATA_ROOT:-/data/cloud_disk_1/erenpeng/datasets/merged_stage1_e2e}"

FROZEN_MODEL_URL="${FROZEN_MODEL_URL:-http://124.221.221.186:30000/v1/chat/completions}"
FROZEN_MODEL_NAME="${FROZEN_MODEL_NAME:-glm-5.1}"
FROZEN_MODEL_TIMEOUT="${FROZEN_MODEL_TIMEOUT:-60}"
FROZEN_MODEL_MAX_CONCURRENCY="${FROZEN_MODEL_MAX_CONCURRENCY:-32}"
FROZEN_MODEL_CONN_LIMIT="${FROZEN_MODEL_CONN_LIMIT:-128}"
FROZEN_MODEL_MAX_TOKENS="${FROZEN_MODEL_MAX_TOKENS:-128000}"
FROZEN_MODEL_MAX_INPUT_TOKENS="${FROZEN_MODEL_MAX_INPUT_TOKENS:-16384}"
PROBE_EVAL_MAX_CONCURRENCY="${PROBE_EVAL_MAX_CONCURRENCY:-8}"
MEMORY_RL_MAX_AGENT_TURNS="${MEMORY_RL_MAX_AGENT_TURNS:-20}"
MEMORY_RL_TASK_VERSION="${MEMORY_RL_TASK_VERSION:-t2_agent_loop}"
MEMORY_RL_APPLY_MODE="${MEMORY_RL_APPLY_MODE:-task_loop}"
MEMORY_RL_TRAIN_TASK_LOOP="${MEMORY_RL_TRAIN_TASK_LOOP:-1}"

ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-8}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-48}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
NUM_ROLLOUT="${NUM_ROLLOUT:-10}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-20000}"

TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"  # dense model: keep EP=1
ETP_SIZE="${ETP_SIZE:-1}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-28192}"

SLIME_ROOT="${SLIME_ROOT:-/root/slime}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"

DRY_RUN=false
BACKGROUND=false
SKIP_IMAGE_SYNC=false
SKIP_DOCKER=false
SKIP_RAY=false
EXTRA_TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --background) BACKGROUND=true; shift ;;
    --skip-image-sync) SKIP_IMAGE_SYNC=true; shift ;;
    --skip-docker) SKIP_DOCKER=true; shift ;;
    --skip-ray) SKIP_RAY=true; shift ;;
    --num-rollout) NUM_ROLLOUT="$2"; shift 2 ;;
    --actor-num-nodes) ACTOR_NUM_NODES="$2"; shift 2 ;;
    --rollout-num-gpus) ROLLOUT_NUM_GPUS="$2"; shift 2 ;;
    --task-version) MEMORY_RL_TASK_VERSION="$2"; shift 2 ;;
    *) EXTRA_TRAIN_ARGS+=("$1"); shift ;;
  esac
done

run_remote() {
  local host="$1"; shift
  if [ "$DRY_RUN" = true ]; then echo "[DRY-RUN] ssh $host: $*"; return 0; fi
  ssh ${SSH_OPTS} root@"$host" "$@"
}
run_in_container() {
  local host="$1"; shift
  local cmd="$*"
  if [ "$DRY_RUN" = true ]; then echo "[DRY-RUN] docker exec $host/$CONTAINER_NAME: $cmd"; return 0; fi
  ssh ${SSH_OPTS} root@"$host" "docker exec ${CONTAINER_NAME} bash -lc $(printf '%q' "$cmd")"
}

check_paths() {
  [ -f "$MODEL_CONFIG_SCRIPT" ] || { log_error "missing MODEL_CONFIG_SCRIPT: $MODEL_CONFIG_SCRIPT"; exit 1; }
  [ -d "$HF_CHECKPOINT" ] || { log_error "missing HF_CHECKPOINT: $HF_CHECKPOINT"; exit 1; }
  [ -d "$TORCH_DIST_CHECKPOINT" ] || { log_error "missing TORCH_DIST_CHECKPOINT: $TORCH_DIST_CHECKPOINT"; exit 1; }
  [ -f "$TRAIN_DATA" ] || { log_error "missing TRAIN_DATA: $TRAIN_DATA"; exit 1; }
  # shellcheck disable=SC1090
  source "$MODEL_CONFIG_SCRIPT"
  [ ${#MODEL_ARGS[@]} -gt 0 ] || { log_error "MODEL_ARGS is empty after sourcing $MODEL_CONFIG_SCRIPT"; exit 1; }
}

check_nodes_idle() {
  log_step "Check SSH/GPU/service status on ${#ALL_NODES[@]} nodes"
  if [ "$DRY_RUN" = true ]; then
    log_info "dry-run: skip real GPU/service idleness check"
    return 0
  fi
  local bad=0
  for node in "${ALL_NODES[@]}"; do
    if ! run_remote "$node" "echo ok >/dev/null"; then log_error "ssh failed: $node"; bad=1; continue; fi
    if run_remote "$node" "nvidia-smi --query-compute-apps=process_name --format=csv,noheader,nounits 2>/dev/null | egrep -iq 'sglang|vllm|xinference|triton|text-generation'"; then
      log_error "active inference GPU process on $node"; bad=1
    fi
  done
  [ "$bad" = 0 ] || exit 1
}

sync_image() {
  [ "$SKIP_IMAGE_SYNC" = true ] && { log_info "skip image sync"; return; }
  local head_id tar_path
  head_id=$(run_remote "$HEAD_NODE" "docker images --format '{{.ID}}' ${DOCKER_IMAGE} | head -1")
  [ -n "$head_id" ] || { log_error "head has no image $DOCKER_IMAGE"; exit 1; }
  tar_path="/data/cloud_disk_4/docker_images/$(echo "$DOCKER_IMAGE" | tr '/:' '__')_${head_id}.tar"
  log_info "head image id=$head_id tar=$tar_path"
  run_remote "$HEAD_NODE" "mkdir -p /data/cloud_disk_4/docker_images; [ -f $tar_path ] || docker save ${DOCKER_IMAGE} -o $tar_path; du -h $tar_path"
  for node in "${WORKER_NODES[@]}"; do
    (
      cur=$(run_remote "$node" "docker images --format '{{.ID}}' ${DOCKER_IMAGE} | head -1" || true)
      if [ "$cur" = "$head_id" ]; then echo "$node image ok"; else
        echo "$node load image current=${cur:-none}"
        run_remote "$node" "docker load -i $tar_path >/dev/null && docker tag $head_id ${DOCKER_IMAGE}"
      fi
    ) &
  done
  wait
}

start_containers() {
  [ "$SKIP_DOCKER" = true ] && { log_info "skip docker"; return; }
  log_step "Start containers"
  for node in "${ALL_NODES[@]}"; do
    (
      run_remote "$node" "
        DEV_FLAGS=\"\"; for dev in /dev/infiniband/uverbs* /dev/infiniband/rdma_cm; do [ -e \"\$dev\" ] && DEV_FLAGS=\"\$DEV_FLAGS --device=\$dev\"; done
        docker rm -f ${CONTAINER_NAME} 2>/dev/null || true
        docker run -d --name ${CONTAINER_NAME} --network host --ipc=host --gpus all --privileged --ulimit memlock=-1 --ulimit stack=67108864 \$DEV_FLAGS -v ${MOUNT_HOST_DATA}:${MOUNT_CONTAINER_DATA} -v /root/.ssh:/root/.ssh:ro -e NVIDIA_VISIBLE_DEVICES=all -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_HCA=mlx5 -e NCCL_NET_GDR_LEVEL=5 -e NCCL_SOCKET_IFNAME=eth0 -e GLOO_SOCKET_IFNAME=eth0 -w /root ${DOCKER_IMAGE} sleep infinity >/dev/null
      "
    ) &
  done
  wait
}

start_ray() {
  [ "$SKIP_RAY" = true ] && { log_info "skip ray"; return; }
  log_step "Start Ray cluster on ${HEAD_NODE}:${RAY_PORT}"
  for node in "${ALL_NODES[@]}"; do run_in_container "$node" "ray stop --force 2>/dev/null || true; rm -rf /tmp/ray" & done
  wait
  run_in_container "$HEAD_NODE" "ray start --head --node-ip-address=${HEAD_NODE} --port=${RAY_PORT} --dashboard-port=${RAY_DASHBOARD_PORT} --dashboard-host=0.0.0.0 --dashboard-agent-listen-port=${RAY_AGENT_PORT} --dashboard-agent-grpc-port=${RAY_AGENT_GRPC_PORT} --runtime-env-agent-port=${RAY_RUNTIME_ENV_AGENT_PORT} --num-gpus=8 --num-cpus=64 --resources='{\"head\": 1}'" 
  sleep 5
  for node in "${WORKER_NODES[@]}"; do
    run_in_container "$node" "ray start --address=${HEAD_NODE}:${RAY_PORT} --num-gpus=8 --num-cpus=64 --node-ip-address=${node} --dashboard-agent-listen-port=${RAY_AGENT_PORT} --dashboard-agent-grpc-port=${RAY_AGENT_GRPC_PORT} --runtime-env-agent-port=${RAY_RUNTIME_ENV_AGENT_PORT} --disable-usage-stats" &
  done
  wait
  sleep 20
  run_in_container "$HEAD_NODE" "python3 - <<'PY'
import ray
ray.init(address='auto')
ns=[n for n in ray.nodes() if n['Alive']]
print('alive',len(ns),'gpu',sum(n.get('Resources',{}).get('GPU',0) for n in ns))
for ip in sorted(n['NodeManagerAddress'] for n in ns): print(ip)
PY"
}

submit_training() {
  local run_name ckpt_dir dump_dir log_dir runtime_json train_script
  run_name="ingest_grpo_fixed16_$(date '+%Y%m%d_%H%M%S')"
  ckpt_dir="${PROJECT_ROOT}/outputs/${run_name}/checkpoints"
  dump_dir="${PROJECT_ROOT}/outputs/${run_name}/rollout_dumps"
  log_dir="${PROJECT_ROOT}/outputs/${run_name}/logs"
  runtime_json="/tmp/runtime_env_${run_name}.json"
  train_script="/tmp/run_train_${run_name}.sh"
  run_in_container "$HEAD_NODE" "mkdir -p ${ckpt_dir} ${dump_dir} ${log_dir}"
  log_info "RUN_NAME=$run_name"
  log_info "logs=$log_dir/train.log"
  log_info "ckpt=$ckpt_dir"

  ssh ${SSH_OPTS} root@"${HEAD_NODE}" "docker exec -i ${CONTAINER_NAME} tee ${runtime_json} >/dev/null" <<EOF
{"env_vars":{"PYTHONPATH":"${MEGATRON_ROOT}:${LLM_GATEWAY_PARENT}:${LLM_GATEWAY_ROOT}","PYTHONUNBUFFERED":"1","CUDA_DEVICE_MAX_CONNECTIONS":"1","PYTORCH_ALLOC_CONF":"expandable_segments:True","NCCL_NVLS_ENABLE":"1","NCCL_IB_DISABLE":"0","NCCL_IB_GID_INDEX":"3","NCCL_IB_HCA":"mlx5","NCCL_NET_GDR_LEVEL":"5","NCCL_TIMEOUT_MS":"600000","NCCL_SOCKET_IFNAME":"eth0","GLOO_SOCKET_IFNAME":"eth0","NCCL_DEBUG":"WARN","TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC":"600","MASTER_ADDR":"${HEAD_NODE}","no_proxy":"127.0.0.1,${HEAD_NODE}","INGEST_SNAPSHOT_DATA_ROOT":"${SNAPSHOT_DATA_ROOT}","SNAPSHOT_DATA_ROOT":"${SNAPSHOT_DATA_ROOT}","FROZEN_MODEL_URL":"${FROZEN_MODEL_URL}","FROZEN_MODEL_NAME":"${FROZEN_MODEL_NAME}","FROZEN_MODEL_TIMEOUT":"${FROZEN_MODEL_TIMEOUT}","FROZEN_MODEL_MAX_CONCURRENCY":"${FROZEN_MODEL_MAX_CONCURRENCY}","FROZEN_MODEL_CONN_LIMIT":"${FROZEN_MODEL_CONN_LIMIT}","FROZEN_MODEL_MAX_TOKENS":"${FROZEN_MODEL_MAX_TOKENS}","FROZEN_MODEL_MAX_INPUT_TOKENS":"${FROZEN_MODEL_MAX_INPUT_TOKENS}","PROBE_EVAL_MAX_CONCURRENCY":"${PROBE_EVAL_MAX_CONCURRENCY}","REWARD_METRICS_LOG":"${log_dir}/reward_metrics.jsonl","MEMORY_RL_MAX_AGENT_TURNS":"${MEMORY_RL_MAX_AGENT_TURNS}","MEMORY_RL_TASK_VERSION":"${MEMORY_RL_TASK_VERSION}","MEMORY_RL_APPLY_MODE":"${MEMORY_RL_APPLY_MODE}","MEMORY_RL_TRAIN_TASK_LOOP":"${MEMORY_RL_TRAIN_TASK_LOOP}","MEMORY_RL_LLM_API_URL":"${FROZEN_MODEL_URL}","MEMORY_RL_LLM_MODEL":"${FROZEN_MODEL_NAME}"}}
EOF

  # Ensure Python dependencies needed by custom rollout/reward code are present.
  for node in "${ALL_NODES[@]}"; do
    run_in_container "$node" "pip install -q 'psycopg[binary]' psycopg_pool zstandard -i https://pypi.tuna.tsinghua.edu.cn/simple >/dev/null 2>&1 || true" &
  done
  wait

  # Render training script inside head container. MODEL_ARGS is expanded by sourcing config there.
  ssh ${SSH_OPTS} root@"${HEAD_NODE}" "docker exec -i ${CONTAINER_NAME} tee ${train_script} >/dev/null" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source ${MODEL_CONFIG_SCRIPT}
export PYTHONPATH=${MEGATRON_ROOT}:${LLM_GATEWAY_PARENT}:${LLM_GATEWAY_ROOT}
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=${HEAD_NODE}
cd ${SLIME_ROOT}
RUNTIME_ENV=\
\$(cat ${runtime_json})

echo '=== DEBUG paths ==='
ls -la ${HF_CHECKPOINT}/config.json
ls -la ${TORCH_DIST_CHECKPOINT}/ | head -5

echo '=== DEBUG model args ==='
printf '%q ' "\${MODEL_ARGS[@]}"; echo

ray job submit --address=http://127.0.0.1:${RAY_DASHBOARD_PORT} \
  --runtime-env-json="\${RUNTIME_ENV}" \
  -- python3 ${SLIME_ROOT}/train_async.py \
  --actor-num-nodes ${ACTOR_NUM_NODES} \
  --actor-num-gpus-per-node ${ACTOR_NUM_GPUS_PER_NODE} \
  --rollout-num-gpus ${ROLLOUT_NUM_GPUS} \
  --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE} \
  "\${MODEL_ARGS[@]}" \
  --hf-checkpoint ${HF_CHECKPOINT} \
  --ref-load ${TORCH_DIST_CHECKPOINT} \
  --load ${TORCH_DIST_CHECKPOINT} \
  --save ${ckpt_dir} \
  --save-interval 20 \
  --prompt-data ${TRAIN_DATA} \
  --input-key prompt \
  --metadata-key metadata \
  --apply-chat-template \
  --custom-rm-path llm_gateway.rl.slime_train.tasks.ingest_reward.reward.reward_func \
  --custom-generate-function-path llm_gateway.rl.slime_train.memory_rl.custom_generate.custom_generate \
  --num-rollout ${NUM_ROLLOUT} \
  --rollout-batch-size ${ROLLOUT_BATCH_SIZE} \
  --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT} \
  --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN} \
  --rollout-max-prompt-len ${ROLLOUT_MAX_PROMPT_LEN} \
  --rollout-temperature 1.0 \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --balance-data \
  --tensor-model-parallel-size ${TP_SIZE} \
  --sequence-parallel \
  --pipeline-model-parallel-size ${PP_SIZE} \
  --context-parallel-size ${CP_SIZE} \
  --expert-model-parallel-size ${EP_SIZE} \
  --expert-tensor-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU} \
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
  --lr-warmup-iters 5 \
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
  --dump-details ${dump_dir} \
  ${EXTRA_TRAIN_ARGS[*]:-} 2>&1 | tee ${log_dir}/train.log
EOF
  run_in_container "$HEAD_NODE" "chmod +x ${train_script}"
  if [ "$DRY_RUN" = true ]; then
    run_in_container "$HEAD_NODE" "sed -n '1,220p' ${train_script}"
    return
  fi
  if [ "$BACKGROUND" = true ]; then
    run_in_container "$HEAD_NODE" "nohup bash ${train_script} > ${log_dir}/submit.log 2>&1 & echo \\\$! > ${log_dir}/submit.pid"
    log_info "Submitted in background. Logs: ssh root@${HEAD_NODE} docker exec ${CONTAINER_NAME} tail -f ${log_dir}/submit.log"
  else
    run_in_container "$HEAD_NODE" "bash ${train_script}"
  fi
}

main() {
  log_step "Fixed 16-idle-node ingest GRPO launcher"
  log_info "nodes=${#ALL_NODES[@]} head=${HEAD_NODE} ray=${RAY_PORT}/${RAY_DASHBOARD_PORT} container=${CONTAINER_NAME}"
  log_info "actor=${ACTOR_NUM_NODES} rollout_gpus=${ROLLOUT_NUM_GPUS} gbs=${GLOBAL_BATCH_SIZE}"
  check_paths
  check_nodes_idle
  sync_image
  start_containers
  start_ray
  submit_training
}
main "$@"
