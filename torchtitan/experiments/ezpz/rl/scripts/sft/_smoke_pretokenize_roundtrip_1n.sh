#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=00:45:00
#PBS -l filesystems=tegu:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe
#
# 1N smoke validating the NEW pre-tokenize round-trip (train_sft.py
# --pretokenize_to / --pretokenized_dataset) end-to-end on a SMALL slice before
# committing the multi-hour full pre-tokenize of the 93M-row mix.
#
# Two phases in one job:
#   1. --pretokenize_to on a 50k-row slice (--max_train_samples 50000 truncates
#      BEFORE prep, so this is fast): build mix -> TRL tokenize+pack -> save
#      trainer.train_dataset (with input_ids) to a scratch dir -> exit.
#   2. --pretokenized_dataset on that scratch dir: load_from_disk -> hand to
#      SFTTrainer (must SKIP prep: no "Tokenizing train dataset" line) -> train
#      5 steps to confirm the packed shapes are trainable.
#
# PASS = phase 1 writes input_ids, phase 2 logs "will skip tokenize+pack" and
# takes 5 steps with finite loss and NO "Tokenizing train dataset" progress bar.
#
# Output scratch: outputs/sft/_smoke_pretok_roundtrip/ (dataset + ckpts)
set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export HF_HUB_OFFLINE=1
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"
source .venv/bin/activate

BASE_MODEL="${HOME}/global_step138650"
PRETOK_DIR="${SUBMIT_DIR}/outputs/sft/_smoke_pretok_roundtrip/dataset"
CKPT_DIR="outputs/sft/_smoke_pretok_roundtrip/ckpts"
LOG_DIR="logs/smoke-pretok-roundtrip-${PBS_JOBID%%.*}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"
backup "${PRETOK_DIR}" 2>/dev/null || true   # clean prior smoke dataset

echo "=== PHASE 1: pre-tokenize 50k-row slice -> ${PRETOK_DIR} ===" \
    | tee "${LOG_DIR}/run.log"
ezpz launch --np 1 python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --sft_dataset tulu_math_uc_mix \
    --model_name_or_path "${BASE_MODEL}" \
    --pretokenize_to "${PRETOK_DIR}" \
    --max_train_samples 50000 \
    --max_length 2048 \
    --dataset_num_proc 32 \
    --report_to none \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== PHASE 2: train 5 steps FROM pre-tokenized dir (must skip prep) ===" \
    | tee -a "${LOG_DIR}/run.log"
ezpz launch --np 12 -ppn 12 python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --pretokenized_dataset "${PRETOK_DIR}" \
    --model_name_or_path "${BASE_MODEL}" \
    --output_dir "${CKPT_DIR}" \
    --max_steps 5 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_length 2048 \
    --bf16 --fsdp full_shard \
    --logging_steps 1 \
    --save_strategy no \
    --report_to none \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== CHECK: phase-2 log should NOT contain 'Tokenizing train dataset' ===" \
    | tee -a "${LOG_DIR}/run.log"
if grep -q "Tokenizing train dataset" "${LOG_DIR}/run.log"; then
    echo "RESULT: prep ran in phase 2 -- but check WHICH phase (phase 1 tokenizes by design)" \
        | tee -a "${LOG_DIR}/run.log"
fi
echo "=== DONE: log in ${LOG_DIR}/ ===" | tee -a "${LOG_DIR}/run.log"
