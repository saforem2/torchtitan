#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=01:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=8
#PBS -q workq
#PBS -j oe
#
# Stage 1 (CoT plan): cold-start CoT-SFT smoke. Continue-SFT the instruction-
# tuned agpt-2b (checkpoint-900-hf) on gsm8k-r1cot -- gsm8k rationales wrapped
# in the <think>/<answer> envelope -- to teach the model to EMIT reasoning
# traces. 7473 examples, 2 epochs, 8N: a short smoke (~10-20 min train).
#
# Gate (measured by scripts/eval/eval_cot_gsm8k.py on a saved checkpoint):
#   format hit-rate ~0 -> >90%, CoT accuracy no worse than the 0.15 baseline.
#
# Usage: qsub torchtitan/experiments/ezpz/rl/scripts/sft/aurora2b_gsm8k_r1cot_sft.sh
# no set -e (module load returns nonzero under Lmod); pipefail only.
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
python3 -c "import trl; print('trl', trl.__version__)" || { echo "FATAL: trl missing"; exit 1; }

# Base = the shipped full-mix SFT deliverable (instruction-tuned; has seen math
# rationales but does not emit a delimited <think> block). Continue-SFT from it.
MODEL_PATH="${MODEL_PATH:-outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900-hf}"
CKPT_DIR="${CKPT_DIR:-outputs/sft/agpt2b-gsm8k-r1cot-8n}"
MAX_LENGTH="${MAX_LENGTH:-2048}"      # MUST be >1024: the <answer> tail would truncate
NUM_EPOCHS="${NUM_EPOCHS:-2}"
LR="${LR:-2e-5}"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "FATAL: MODEL_PATH missing: ${MODEL_PATH}"; exit 1
fi

LOG_DIR="logs/sft-agpt2b-gsm8k-r1cot-8n-${PBS_JOBID%%.*}"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

echo "=== Stage 1 cold-start CoT-SFT: agpt-2b ckpt-900 + gsm8k-r1cot ===" \
    | tee "${LOG_DIR}/run.log"
echo "base=${MODEL_PATH} epochs=${NUM_EPOCHS} lr=${LR} max_len=${MAX_LENGTH}" \
    | tee -a "${LOG_DIR}/run.log"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/datasets_sft.py \
    2>&1 | tee -a "${LOG_DIR}/run.log"

ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --sft_dataset gsm8k-r1cot \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir "${CKPT_DIR}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --learning_rate "${LR}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_length "${MAX_LENGTH}" \
    --bf16 --fsdp full_shard \
    --gradient_checkpointing \
    --logging_steps 10 \
    --save_strategy epoch \
    --save_total_limit 4 \
    --report_to wandb \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true

echo "=== DONE: log ${LOG_DIR}/, ckpts ${CKPT_DIR}/ ===" | tee -a "${LOG_DIR}/run.log"
