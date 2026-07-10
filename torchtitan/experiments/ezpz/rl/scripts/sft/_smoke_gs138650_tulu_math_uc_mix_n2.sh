#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=00:45:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe
#
# 2N Sunspot smoke validating the BIG tulu_math_uc_mix (registered recipe:
# tulu-3 0.65 + OpenMathInstruct-2 0.15 + ultrachat 0.20, ~93.1M rows, ~54B
# tokens) on the global_step138650 base (vocab 256000, the checkpoint the
# already-completed 729-step SFT used -- trained clean at 32N, so it sidesteps
# the v2-base oneCCL-scale crash).
#
# Purpose: confirm the pre-built mix cache (hash 44608c8c15bd9714, 286GB,
# built by _prebuild_openmath_mix_1n.sh via the vectorized interleave) loads in
# <5s and that the recipe tokenizes + steps end-to-end. This is the payoff of
# the vectorized-interleave work: the OpenMathInstruct-2 path used to take
# ~90min to build + blew the XPU oneCCL barrier at scale; now it is a <5s
# cache load.
#
# Uses the REGISTERED name `tulu_math_uc_mix` (NOT a metamathqa mix-spec) so it
# resolves to the big cached mix. --max_train_samples 50000 truncates AFTER the
# <5s cache load so tokenize+pack stays fast for a smoke; drop it for prod.
#
# Output: outputs/sft/gs138650-tulu-math-uc-mix-smoke-n2/
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

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"
source .venv/bin/activate

# Base: global_step138650 (vocab 256000). Absolute path avoids the cwd-relative
# resolution + silent Qwen-0.6B fallback in _resolve_model().
BASE_MODEL="${HOME}/global_step138650"

CKPT_DIR="outputs/sft/gs138650-tulu-math-uc-mix-smoke-n2"
LOG_DIR="logs/sft-gs138650-tulu-math-uc-mix-smoke-n2-${PBS_JOBID%%.*}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

echo "=== 2N SFT smoke: gs138650 + tulu_math_uc_mix (BIG cached mix), 10 steps ===" \
    | tee "${LOG_DIR}/run.log"
echo "BASE_MODEL=${BASE_MODEL}" | tee -a "${LOG_DIR}/run.log"
echo "expecting mix-cache HIT on hash 44608c8c15bd9714 (<5s load)" \
    | tee -a "${LOG_DIR}/run.log"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/datasets_sft.py \
    torchtitan/experiments/ezpz/rl/train_sft.py 2>&1 | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

# Registered `tulu_math_uc_mix` -> the big cached mix (14M-row OpenMathInstruct-2
# variant). --max_train_samples keeps the smoke fast after the cache load.
ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --sft_dataset tulu_math_uc_mix \
    --model_name_or_path "${BASE_MODEL}" \
    --output_dir "${CKPT_DIR}" \
    --max_steps 10 \
    --max_train_samples 50000 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_length 1024 \
    --bf16 --fsdp full_shard \
    --logging_steps 1 \
    --save_strategy no \
    --report_to wandb \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== DONE: log in ${LOG_DIR}/ ===" | tee -a "${LOG_DIR}/run.log"
