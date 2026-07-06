#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=00:40:00
#PBS -l filesystems=tegu:home
#PBS -l select=3
#PBS -q workq
#PBS -j oe
#
# Multi-TRAINER-node GRPO validation (1 server node + 2 trainer nodes = 24
# trainer ranks). This is the decisive experiment the overnight (2026-07-01)
# session never actually ran.
#
# Root cause found 2026-07-06 from the overnight faulthandler dumps
# (job 12470009): every failed 3N run had trainer CCL_ATL_TRANSPORT UNSET,
# which selects oneCCL's *mpi* default. atl_mpi cannot form the
# trainer-rank0 <-> vLLM-server ProcessGroupXCCL across two separate mpiexec
# worlds -> "atl_mpi_comm.cpp:94 init_transport: comm_create error" -> rank0
# SIGSEGV in ccl_comm::create_comm_id. Ranks 1-23 then block in the main-PG
# all_gather (full_tensor) waiting for the dead rank0 -- which was misread as a
# "desync hang". The working 2N smoke (12469976) had CCL_ATL_TRANSPORT=ofi and
# trained fine. The overnight "clean CXI" runs stripped that ofi export (via
# --no-oneccl-tcp-kvs), dropping the trainer to the mpi default.
#
# This script runs the KNOWN-GOOD transport (ofi + shared TCP-KVS, exactly like
# the 2N smoke and the production xnode recipe) at 2 trainer nodes, WITH the
# committed AVG->SUM FSDP patch (apply_all_xpu_patches auto-applies it). That
# combination -- ofi/KVS ON *and* SUM patch -- was never tested together.
#
# Expected: if the trainer steps across 2 nodes, multi-trainer-node is solved
# and the "desync" conclusion is refuted. If it hits the AVG wall, the SUM
# patch is not taking effect. Either way faulthandler (120s) captures all-rank
# stacks as a safety net (the overnight run PROVED file-target faulthandler
# works: 12470009 wrote full 25KB per-rank stacks).
#
# Output: logs/grpo-3n-validate-<jobid>/
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

# Known-good cross-tree transport: ofi + shared TCP-KVS (mirror the 2N smoke).
# This is what makes the trainer-rank0 <-> server ProcessGroupXCCL form without
# atl_mpi. Do NOT strip these / do NOT pass --no-oneccl-tcp-kvs.
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
CKPT_DIR="${CKPT_DIR:-outputs/grpo/grpo-3n-validate}"
LOG_DIR="logs/grpo-3n-validate-${PBS_JOBID%%.*}"
STACK_DIR="${LOG_DIR}/stacks"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${STACK_DIR}"

HEAD_NODE="$(head -1 "${PBS_NODEFILE}")"
VLLM_PORT=8765
VLLM_URL="http://${HEAD_NODE}:${VLLM_PORT}"
export CCL_KVS_IP_PORT="${HEAD_NODE}_29513"
LOG="${LOG_DIR}/run.log"; SERVE_LOG="${LOG_DIR}/vllm_serve.log"; TRAINER_LOG="${LOG_DIR}/trainer.log"

# All-rank stack safety net (in-process faulthandler-to-file; proven working in
# overnight job 12470009). Fires at 120s if a rank is wedged.
export EZPZ_RL_FAULTHANDLER_SECS=120
export EZPZ_RL_FAULTHANDLER_DIR="${SUBMIT_DIR}/${STACK_DIR}"

# Trainer hostfile = all nodes EXCEPT the head (server owns the head node).
TRAINER_HF="${LOG_DIR}/trainer.hostfile"
tail -n +2 "${PBS_NODEFILE}" | sort -u > "${TRAINER_HF}"
NTRAIN_NODES=$(wc -l < "${TRAINER_HF}")
NTRAIN_RANKS=$(( NTRAIN_NODES * 12 ))
echo "HEAD(server)=${HEAD_NODE}  trainer_nodes=${NTRAIN_NODES} ranks=${NTRAIN_RANKS}" | tee "${LOG}"
echo "TRANSPORT=ofi/tcp KVS=${CCL_KVS_IP_PORT} (SUM patch via apply_all_xpu_patches)" | tee -a "${LOG}"
echo "URL=${VLLM_URL}  MODEL=${MODEL}  faulthandler=${EZPZ_RL_FAULTHANDLER_SECS}s" | tee -a "${LOG}"

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

# --- trainer on all non-head nodes (KVS/ofi ON: no --no-oneccl-tcp-kvs) ---
echo "[$(date +%T)] launching ${NTRAIN_RANKS} trainer ranks on ${NTRAIN_NODES} nodes" | tee -a "${LOG}"
"${SUBMIT_DIR}/.venv/bin/ezpz" launch --np "${NTRAIN_RANKS}" -ppn 12 --hostfile "${TRAINER_HF}" \
    "${SUBMIT_DIR}/venvs/rl-vllm/bin/python" \
    -m torchtitan.experiments.ezpz.rl.train_grpo \
    --task arithmetic \
    --model_name_or_path "${MODEL}" \
    --output_dir "${CKPT_DIR}" \
    --per_device_train_batch_size 1 --num_generations 4 \
    --max_completion_length 64 --temperature 0.7 \
    --max_steps 8 --learning_rate 1e-6 --beta 0.0 --bf16 --fsdp full_shard \
    --logging_steps 1 --save_strategy no \
    --report_to wandb \
    --use_vllm --vllm_mode server --vllm_server_base_url "${VLLM_URL}" \
    --vllm_server_timeout 300 \
    2>&1 | tee -a "${TRAINER_LOG}" "${LOG}"
echo "GRPO_3N_VALIDATE_VERDICT: trainer exited ${PIPESTATUS[0]}" | tee -a "${LOG}"
