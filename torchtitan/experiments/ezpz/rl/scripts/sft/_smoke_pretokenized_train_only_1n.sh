#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=00:20:00
#PBS -l filesystems=tegu:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe
#
# 1N phase-2-only smoke: train 5 steps from an EXISTING pre-tokenized dataset
# (produced by _smoke_pretokenize_roundtrip_1n.sh phase 1). Confirms the fix for
# the assistant_only_loss "dataset is not conversational" guard (commit
# eac2b135c) -- the first round-trip smoke (12470282) passed phase 1 but phase 2
# hit that guard. Reuses the on-disk dataset so this is fast (no re-tokenize).
#
# PASS = no "not conversational" ValueError, log shows "setting
# assistant_only_loss=False", NO "Tokenizing train dataset" bar, 5 steps land
# with finite loss.
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
CKPT_DIR="outputs/sft/_smoke_pretok_train_only/ckpts"
LOG_DIR="logs/smoke-pretok-train-only-${PBS_JOBID%%.*}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

if [[ ! -d "${PRETOK_DIR}" ]]; then
    echo "FATAL: no pre-tokenized dataset at ${PRETOK_DIR}; run the roundtrip smoke first" \
        | tee "${LOG_DIR}/run.log"
    exit 1
fi

echo "=== phase-2-only: train 5 steps FROM ${PRETOK_DIR} (assistant_only_loss guard fix) ===" \
    | tee "${LOG_DIR}/run.log"
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
echo "=== DONE: log in ${LOG_DIR}/ ===" | tee -a "${LOG_DIR}/run.log"
