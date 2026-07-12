#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=00:45:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe
#
# 2N SCALE-vs-DATA diagnostic for the 384-rank GPU page fault.
#
# The 32N big-mix SFT (len1024, bsz2/gas8) dies at step 1-2 with a GPU
# "Segmentation fault ... NotPresent Write" -> rank signal 6, reproducibly on
# rank 221 across DIFFERENT nodes (jobs 12470336/338/339). Same launch as the
# COMPLETED 729-step SFT; only the DATASET differs (big OpenMathInstruct-2
# tulu_math_uc_mix vs small metamathqa). OOV token id RULED OUT (full scan:
# global max 255998 < base vocab 256000).
#
# This runs the SAME pretokenized len1024 dataset at 2N (24 ranks) for a few
# steps to split the hypothesis:
#   - crashes at 2N with the same GPU fault  -> PURE DATA bug (a pathological
#     packed sequence), independent of scale.
#   - steps cleanly at 2N                    -> 384-RANK SCALE fault (oneCCL/
#     collective GPU fault, the project_sft_v2_base_oom_badnode class).
#
# No --auto-retry (stuck_pre_training false-positive on TRL). --max_steps 20.
set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD="${CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD:-8000}"
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD="${CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD:-8000}"
export HF_HUB_OFFLINE=1
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"
source .venv/bin/activate

BASE_MODEL="${HOME}/global_step138650"
PRETOK_DIR="${HOME}/.cache/ezpz_sft_mixes/tokenized/tulu_math_uc_mix-gs138650-len1024"
CKPT_DIR="outputs/sft/_diag_bigmix_2n_len1024"
LOG_DIR="logs/diag-bigmix-2n-len1024-${PBS_JOBID%%.*}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

echo "=== 2N DIAG: big len1024 pretokenized mix, 20 steps, split scale-vs-data ===" \
    | tee "${LOG_DIR}/run.log"
echo "PRETOK_DIR=${PRETOK_DIR}" | tee -a "${LOG_DIR}/run.log"

# 2N x 12 = 24 ranks. Same per-device shape as the 32N run (bsz2/gas8, len1024).
ezpz launch --np 24 -ppn 12 --timeout "${IDLE_TIMEOUT:-1800}" \
    python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --pretokenized_dataset "${PRETOK_DIR}" \
    --model_name_or_path "${BASE_MODEL}" \
    --output_dir "${CKPT_DIR}" \
    --max_steps 20 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_length 1024 \
    --bf16 --fsdp full_shard \
    --logging_steps 1 \
    --save_strategy no \
    --report_to none \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== DONE: if 20 steps clean -> SCALE fault; if GPU segfault -> DATA bug ===" \
    | tee -a "${LOG_DIR}/run.log"
