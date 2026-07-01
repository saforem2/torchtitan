#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=06:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=10
#PBS -q workq
#PBS -j oe
#
# Production multi-node vLLM-server GRPO on XPU.
#
# The WORKING cross-node recipe, validated by the 2N smoke (job 12469976,
# 2026-07-01): server on the head node + trainer on the OTHER nodes, all from
# the unified venvs/rl-vllm/ venv (torch 2.12 + vllm-xpu + TRL 1.6).
#
# Supersedes aurora2b_sft_arithmetic_8n_vllm.sh, which used the older
# vllm_serve_xpu.sh PYTHONPATH-bridge helper (broke vLLM-XPU platform
# detection: "Device string must not be empty") and spaced --fsdp (broke on
# TRL-1.6's boolean --fsdp; now handled in train_grpo.py _bootstrap_fsdp_env).
#
# Layout (select=10):
#   - node 0 (head): venvs/rl-vllm trl vllm-serve, tile 0, bound 0.0.0.0
#   - nodes 1-9    : 108 GRPO trainer ranks (9 nodes x 12 tiles) hitting the
#                    server over the head node's real hostname; weight-sync
#                    over TCP-KVS XCCL rendezvous at the head node.
#
# Output: outputs/grpo/aurora2b-sft-arithmetic-vllm-xnode/
set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export no_proxy="127.0.0.1,localhost,.hsn.cm.sunspot.alcf.anl.gov,.cm.sunspot.alcf.anl.gov,${no_proxy:-}"
export NO_PROXY="${no_proxy}"
mkdir -p "/tmp/vllm-${USER}"; export TMPDIR="/tmp/vllm-${USER}"

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"

# Cross-tree TCP-KVS XCCL rendezvous (mirror the 2N smoke). Both server and
# trainer must share these + the same CCL_KVS_IP_PORT (the head node).
unset CCL_OP_SYNC CCL_OFI_PROVIDER
unset FI_LOG_LEVEL FI_LOG_PROV FI_LOG_LOCATION
unset FI_CXI_DEFAULT_CQ_SIZE FI_CXI_DEFAULT_TX_SIZE FI_CXI_OFLOW_BUF_COUNT
unset FI_CXI_OFLOW_BUF_SIZE FI_CXI_RDZV_EAGER_SIZE FI_CXI_RDZV_THRESHOLD
unset FI_CXI_REQ_BUF_MAX_CACHED FI_CXI_REQ_BUF_MIN_POSTED FI_CXI_REQ_BUF_SIZE
unset FI_CXI_RX_MATCH_MODE FI_MR_CACHE_MAX_COUNT FI_MR_CACHE_MAX_SIZE
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=tcp

MODEL="${MODEL:-outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf}"
CKPT_DIR="${CKPT_DIR:-outputs/grpo/aurora2b-sft-arithmetic-vllm-xnode}"
LOG_DIR="logs/grpo-aurora2b-sft-arithmetic-vllm-xnode-${PBS_JOBID%%.*}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

HEAD_NODE="$(head -1 "${PBS_NODEFILE}")"
VLLM_PORT=8765
VLLM_URL="http://${HEAD_NODE}:${VLLM_PORT}"
export CCL_KVS_IP_PORT="${HEAD_NODE}_29513"
LOG="${LOG_DIR}/run.log"; SERVE_LOG="${LOG_DIR}/vllm_serve.log"; TRAINER_LOG="${LOG_DIR}/trainer.log"

# Trainer hostfile = all nodes EXCEPT the head (server owns the head node).
TRAINER_HF="${LOG_DIR}/trainer.hostfile"
tail -n +2 "${PBS_NODEFILE}" | sort -u > "${TRAINER_HF}"
NTRAIN_NODES=$(wc -l < "${TRAINER_HF}")
NTRAIN_RANKS=$(( NTRAIN_NODES * 12 ))
echo "HEAD(server)=${HEAD_NODE}  trainer_nodes=${NTRAIN_NODES} ranks=${NTRAIN_RANKS}" | tee "${LOG}"
echo "URL=${VLLM_URL}  KVS=${CCL_KVS_IP_PORT}  MODEL=${MODEL}" | tee -a "${LOG}"

# --- server on head node (local subshell, unified rl-vllm venv, no mpiexec) ---
echo "[$(date +%T)] launching vllm-serve on ${HEAD_NODE}" | tee -a "${LOG}"
(
    export ZE_AFFINITY_MASK=0
    exec "${SUBMIT_DIR}/venvs/rl-vllm/bin/trl" vllm-serve \
        --model "${MODEL}" --tensor_parallel_size 1 \
        --host 0.0.0.0 --port "${VLLM_PORT}" \
        --gpu_memory_utilization 0.5 --enforce_eager
) > "${SERVE_LOG}" 2>&1 &
VLLM_PID=$!
trap 'kill -TERM ${VLLM_PID} 2>/dev/null || true; sleep 2; kill -KILL ${VLLM_PID} 2>/dev/null || true' EXIT INT TERM

echo "[$(date +%T)] waiting for ${VLLM_URL}/health/ (up to 600s)..." | tee -a "${LOG}"
W=0
until curl -sf "${VLLM_URL}/health/" >/dev/null 2>&1; do
    kill -0 "${VLLM_PID}" 2>/dev/null || { echo "FATAL: server died"; tail -40 "${SERVE_LOG}"; exit 1; }
    (( W >= 600 )) && { echo "FATAL: health timeout"; tail -40 "${SERVE_LOG}"; exit 1; }
    sleep 5; W=$((W+5))
done
echo "[$(date +%T)] server healthy after ${W}s" | tee -a "${LOG}"

# --- trainer on all non-head nodes ---
echo "[$(date +%T)] launching ${NTRAIN_RANKS} trainer ranks on ${NTRAIN_NODES} nodes" | tee -a "${LOG}"
"${SUBMIT_DIR}/.venv/bin/ezpz" launch --np "${NTRAIN_RANKS}" -ppn 12 --hostfile "${TRAINER_HF}" \
    "${SUBMIT_DIR}/venvs/rl-vllm/bin/python" \
    -m torchtitan.experiments.ezpz.rl.train_grpo \
    --task arithmetic \
    --model_name_or_path "${MODEL}" \
    --output_dir "${CKPT_DIR}" \
    --per_device_train_batch_size 1 --num_generations 4 \
    --max_completion_length 64 --temperature 0.7 \
    --max_steps 1000 --learning_rate 1e-6 --beta 0.0 --bf16 --fsdp full_shard \
    --logging_steps 1 --save_strategy steps --save_steps 100 --save_total_limit 5 \
    --report_to wandb --resume_from_checkpoint "${CKPT_DIR}" \
    --use_vllm --vllm_mode server --vllm_server_base_url "${VLLM_URL}" \
    --vllm_server_timeout 300 \
    2>&1 | tee -a "${TRAINER_LOG}" "${LOG}"
echo "GRPO_XNODE_VERDICT: trainer exited ${PIPESTATUS[0]}" | tee -a "${LOG}"
