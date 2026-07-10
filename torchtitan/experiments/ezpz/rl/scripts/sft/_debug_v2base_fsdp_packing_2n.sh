#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=00:30:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe
#
# 2N DEBUG repro for the v2-base SFT GPU illegal-access crash.
#
# Context: the 32N SFT on the v2 2B base (agpt2b_v2_256n_tulu_mix_32n) crashes
# with `Segmentation fault from GPU ... type: 0 (NotPresent) ... Write` at an
# early step (step 211 on resume-from-200; step 14 fresh), on a variable rank,
# across DIFFERENT physical nodes -> NOT a bad node. Ruled out: bad node, empty
# rows (mix filtered, still crashes), out-of-range tokens (max 235737 < 256128),
# model weights (1-tile eager fwd+bwd works, loss 9.3). The crash needs the
# distributed / FSDP / packing path.
#
# This job = the SMALLEST config that includes that machinery: 2 nodes, FSDP
# full_shard, packing=True. ZE_SERIALIZE=2 forces synchronous L0 kernel launches
# so the faulting kernel throws a Python-visible stack at the offending op
# instead of an async device abort with no stack. max_steps 30 (crash is <=14).
#
# NOTE: ZE_SERIALIZE is slow (synchronous) -- fine for a 30-step debug, do NOT
# use for production.
#
# Output: logs/sft-debug-v2base-2n-${PBS_JOBID%%.*}/
set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD="${CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD:-8000}"
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD="${CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD:-8000}"

# DEBUG: synchronous L0 launches so a GPU illegal access throws at the op with a
# real stack, not an async abort. Also enable py + device-side fault detail.
export ZE_SERIALIZE=2
export PYTORCH_ENABLE_XPU_FALLBACK=0
export TORCH_SHOW_CPP_STACKTRACES=1

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"
source .venv/bin/activate

BASE_MODEL="${SUBMIT_DIR}/AuroraGPT-2B-v2-256n-step92859-hf"
CKPT_DIR="outputs/sft/_debug-v2base-2n"
LOG_DIR="logs/sft-debug-v2base-2n-${PBS_JOBID%%.*}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

echo "=== 2N DEBUG: v2-base SFT FSDP+packing repro (ZE_SERIALIZE=2) ===" \
    | tee "${LOG_DIR}/run.log"
echo "BASE_MODEL=${BASE_MODEL}" | tee -a "${LOG_DIR}/run.log"

# 24 ranks (2 nodes x 12), FSDP full_shard, packing on (default), 30 steps.
# No checkpointing; report_to none (fast). Same mix + tokenization as prod.
ezpz launch --np 24 -ppn 12 \
    python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --sft_dataset 'tulu-3-sft-mixture:0.65,metamathqa:0.15,ultrachat-200k:0.20' \
    --model_name_or_path "${BASE_MODEL}" \
    --output_dir "${CKPT_DIR}" \
    --max_steps 30 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --max_length 1024 \
    --bf16 --fsdp full_shard \
    --logging_steps 1 \
    --save_strategy no \
    --report_to none \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true

echo "=== DONE: log in ${LOG_DIR}/ ===" | tee -a "${LOG_DIR}/run.log"
