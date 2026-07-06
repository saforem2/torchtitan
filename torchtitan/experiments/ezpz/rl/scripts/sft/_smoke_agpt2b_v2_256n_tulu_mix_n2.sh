#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=00:45:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe
#
# 2N Sunspot smoke for the tulu_math_uc_mix SFT recipe on the COMPLETED
# v2 2B production base (step-92,859 = 4.674T tokens, 256N chain), as
# opposed to the older AuroraGPT-2B-sophiag-gs138650 base the prior
# production SFT used. Goal: verify the recipe builds + trains end-to-end
# on the new base before burning a 32N production slot.
#
# The v2 base was DCP->HF converted on Aurora (step-92859) and staged to
# the Sunspot repo root as AuroraGPT-2B-v2-256n-step92859-hf (2026-07-06).
# Pre-flight confirmed: model_type=llama, vocab=256128, 12 layers.
#
# Walltime 45min covers the metamathqa Map() + tokenize/pack; 10 training
# steps land in ~30s once the mix is materialized.
#
# Output: outputs/sft/agpt-2b-v2-256n-tulu-mix-smoke-n2/
#         logs/sft-agpt-2b-v2-256n-tulu-mix-smoke-n2-${PBS_JOBID%%.*}/

set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"
source .venv/bin/activate

# NEW base: absolute path to the converted v2 256N step-92,859 checkpoint.
# Absolute (not bare/relative) because the script does `cd "${SUBMIT_DIR}"`
# and _resolve_model() resolves relative names against repo root -- an
# absolute path avoids cwd ambiguity and the silent Qwen-0.6B fallback that
# _resolve_model() takes if from_pretrained throws.
BASE_MODEL="${SUBMIT_DIR}/AuroraGPT-2B-v2-256n-step92859-hf"

# Fresh, unique CKPT_DIR: distinct from the gs138650 run's dir so nothing
# auto-resumes an old checkpoint over the new base.
CKPT_DIR="outputs/sft/agpt-2b-v2-256n-tulu-mix-smoke-n2"
LOG_DIR="logs/sft-agpt-2b-v2-256n-tulu-mix-smoke-n2-${PBS_JOBID%%.*}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

echo "=== 2N SFT smoke: AuroraGPT-2B v2 256N step-92859 + tulu_math_uc_mix, 10 steps ===" \
    | tee "${LOG_DIR}/run.log"
echo "BASE_MODEL=${BASE_MODEL}" | tee -a "${LOG_DIR}/run.log"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/datasets_sft.py \
    torchtitan/experiments/ezpz/rl/train_sft.py 2>&1 | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

# Use the mix-spec syntax (not the registered tulu_math_uc_mix name) to swap
# OpenMathInstruct-2 -> metamathqa: 14M rows -> 395k rows, eliminating the
# rank-0 interleave-setup bottleneck that blows the XPU oneCCL barrier
# (crashed job 12468398). Same GSM8K+MATH distribution coverage.
#
# --max_train_samples 50000 truncates after the build so tokenize+pack fits
# in <2 min. Smoke covers all the real shapes (multi-turn tulu, single-turn
# math, multi-turn ultrachat) without the full-mix wall-clock cost.
ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --sft_dataset 'tulu-3-sft-mixture:0.65,metamathqa:0.15,ultrachat-200k:0.20' \
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
