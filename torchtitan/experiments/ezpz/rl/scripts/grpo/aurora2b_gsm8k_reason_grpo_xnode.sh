#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=04:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=9
#PBS -q workq
#PBS -j oe
#
# Stage 2 (CoT plan): GRPO-RLVR on the Stage-1 cold-start checkpoint. Teaches
# the model to reason BETTER using the componentized gsm8k_reason reward
# (0.2 format + 0.1 extractable + 0.7 correct). Cross-node vLLM-server recipe,
# mirroring the validated aurora2b_sft_arithmetic_vllm_xnode.sh, with the two
# CoT-specific changes: (1) the vLLM server runs fp32 (agpt-2b emits gibberish
# in bf16 through vLLM -- confirmed by the Stage 0 eval), and (2)
# max_completion_length is 700, not 64, so reasoning traces are not truncated.
#
# Layout (select=9): node 0 = vLLM server (fp32, tile 0); nodes 1-8 = 96 GRPO
# trainer ranks. Output: outputs/grpo/agpt2b-gsm8k-reason-cot/
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

# Cross-tree TCP-KVS XCCL rendezvous (mirror the validated xnode recipe).
unset CCL_OP_SYNC CCL_OFI_PROVIDER
unset FI_LOG_LEVEL FI_LOG_PROV FI_LOG_LOCATION
unset FI_CXI_DEFAULT_CQ_SIZE FI_CXI_DEFAULT_TX_SIZE FI_CXI_OFLOW_BUF_COUNT
unset FI_CXI_OFLOW_BUF_SIZE FI_CXI_RDZV_EAGER_SIZE FI_CXI_RDZV_THRESHOLD
unset FI_CXI_REQ_BUF_MAX_CACHED FI_CXI_REQ_BUF_MIN_POSTED FI_CXI_REQ_BUF_SIZE
unset FI_CXI_RX_MATCH_MODE FI_MR_CACHE_MAX_COUNT FI_MR_CACHE_MAX_SIZE
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=tcp
# Bound oneCCL Level-Zero IPC-handle caches -- they grow ~1 handle/collective/step
# and are NOT freed on XPU, crossing the ~94%% HBM baseline into OOM at a fixed step
# count (~13) regardless of per-step size. Same signature+fix as the 32N SFT OOM
# (job 12470088). Without these, both prior GRPO runs died at step 13.
export CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD=${CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD:-8000}
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=${CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD:-8000}

# Stage-1 cold-start checkpoint (95.5% envelope emission). Has the gemma
# chat_template (consolidation copied it from the staged dir).
MODEL="${MODEL:-outputs/sft/agpt2b-gsm8k-r1cot-8n/checkpoint-16-hf}"
CKPT_DIR="${CKPT_DIR:-outputs/grpo/agpt2b-gsm8k-reason-cot}"
LOG_DIR="logs/grpo-agpt2b-gsm8k-reason-cot-${PBS_JOBID%%.*}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

if [[ ! -d "${MODEL}" ]]; then echo "FATAL: MODEL missing: ${MODEL}"; exit 1; fi

HEAD_NODE="$(head -1 "${PBS_NODEFILE}")"
VLLM_PORT=8765
VLLM_URL="http://${HEAD_NODE}:${VLLM_PORT}"
export CCL_KVS_IP_PORT="${HEAD_NODE}_29513"
LOG="${LOG_DIR}/run.log"; SERVE_LOG="${LOG_DIR}/vllm_serve.log"; TRAINER_LOG="${LOG_DIR}/trainer.log"

TRAINER_HF="${LOG_DIR}/trainer.hostfile"
tail -n +2 "${PBS_NODEFILE}" | sort -u > "${TRAINER_HF}"
NTRAIN_NODES=$(wc -l < "${TRAINER_HF}")
NTRAIN_RANKS=$(( NTRAIN_NODES * 12 ))
echo "HEAD(server)=${HEAD_NODE} trainer_nodes=${NTRAIN_NODES} ranks=${NTRAIN_RANKS}" | tee "${LOG}"
echo "URL=${VLLM_URL} KVS=${CCL_KVS_IP_PORT} MODEL=${MODEL}" | tee -a "${LOG}"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/tasks/gsm8k_reason.py 2>&1 | tee -a "${LOG}"

# --- server on head node (fp32 -- MANDATORY for agpt-2b) ---
echo "[$(date +%T)] launching vllm-serve (fp32) on ${HEAD_NODE}" | tee -a "${LOG}"
(
    export ZE_AFFINITY_MASK=0
    exec "${SUBMIT_DIR}/venvs/rl-vllm/bin/trl" vllm-serve \
        --model "${MODEL}" --tensor_parallel_size 1 \
        --dtype float32 \
        --host 0.0.0.0 --port "${VLLM_PORT}" \
        --gpu_memory_utilization 0.5 --enforce_eager \
        --max_model_len 2048
) > "${SERVE_LOG}" 2>&1 &
VLLM_PID=$!
trap 'kill -TERM ${VLLM_PID} 2>/dev/null || true; sleep 2; kill -KILL ${VLLM_PID} 2>/dev/null || true' EXIT INT TERM

echo "[$(date +%T)] waiting for ${VLLM_URL}/health/ (up to 900s, fp32 load is slower)..." | tee -a "${LOG}"
W=0
until curl -sf "${VLLM_URL}/health/" >/dev/null 2>&1; do
    kill -0 "${VLLM_PID}" 2>/dev/null || { echo "FATAL: server died"; tail -40 "${SERVE_LOG}"; exit 1; }
    (( W >= 900 )) && { echo "FATAL: health timeout"; tail -40 "${SERVE_LOG}"; exit 1; }
    sleep 5; W=$((W+5))
done
echo "[$(date +%T)] server healthy after ${W}s" | tee -a "${LOG}"

# --- trainer on all non-head nodes ---
echo "[$(date +%T)] launching ${NTRAIN_RANKS} trainer ranks on ${NTRAIN_NODES} nodes" | tee -a "${LOG}"
"${SUBMIT_DIR}/.venv/bin/ezpz" launch --np "${NTRAIN_RANKS}" -ppn 12 --hostfile "${TRAINER_HF}" \
    "${SUBMIT_DIR}/venvs/rl-vllm/bin/python" \
    -m torchtitan.experiments.ezpz.rl.train_grpo \
    --task gsm8k_reason \
    --model_name_or_path "${MODEL}" \
    --output_dir "${CKPT_DIR}" \
    --per_device_train_batch_size 1 --num_generations 8 \
    --max_completion_length 700 --temperature 0.7 \
    --max_steps 400 --learning_rate 1e-6 --beta 0.0 --bf16 --fsdp full_shard \
    --gradient_checkpointing \
    --logging_steps 1 --save_strategy steps --save_steps 50 --save_total_limit 5 \
    --report_to wandb --resume_from_checkpoint "${CKPT_DIR}" \
    --use_vllm --vllm_mode server --vllm_server_base_url "${VLLM_URL}" \
    --vllm_server_timeout 600 \
    2>&1 | tee -a "${TRAINER_LOG}" "${LOG}"
echo "GRPO_COT_VERDICT: trainer exited ${PIPESTATUS[0]}" | tee -a "${LOG}"
