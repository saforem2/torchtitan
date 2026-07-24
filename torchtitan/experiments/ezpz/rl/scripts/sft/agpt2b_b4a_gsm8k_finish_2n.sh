#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=01:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe
#
# B4a: short gsm8k-r1cot FINISHING stage on the B3 base (restores B2's winning
# in-distribution short-CoT final stage). Saves 1 ckpt/epoch (3 epochs) for
# per-epoch eval -- B4a's base is more capable than B2's so 3 epochs may
# overfit 7473 examples; per-epoch eval finds the sweet spot.
#
# save_strategy=epoch verified: a prior 2N gsm8k-r1cot run on this same
# train_sft.py/EzpzSFTConfig path (outputs/sft/agpt2b-gsm8k-r1cot-2n/) produced
# real per-epoch checkpoints (checkpoint-31/62/93 -- 3 epochs at 31 steps/
# epoch), and transformers' IntervalStrategy accepts "epoch" directly. No
# ezpz override forces "steps" (EzpzSFTConfig only sets a "no" default).
#
# Usage: qsub torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b4a_gsm8k_finish_2n.sh
set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export HF_DATASETS_OFFLINE=0   # gsm8k is small + may need download; proxy on
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job

cd "${SUBMIT_DIR}"
source .venv/bin/activate
python3 -c "import trl; print('trl', trl.__version__)" || { echo "FATAL: trl missing"; exit 1; }

BASE_MODEL="${BASE_MODEL:-$SUBMIT_DIR/outputs/evals/cot/b3-3605-hf-diag}"
CKPT_DIR="outputs/sft/agpt2b-b4a-gsm8k-finish"
LOG_DIR="logs/sft-b4a-gsm8k-finish-${PBS_JOBID%%.*}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

[[ -f "${BASE_MODEL}/model.safetensors" ]] || { echo "FATAL: B3 base missing ${BASE_MODEL}"; exit 1; }

echo "=== B4a: gsm8k-r1cot finishing stage on B3 base, 3 epochs, per-epoch ckpts ===" \
    | tee "${LOG_DIR}/run.log"
echo "BASE_MODEL=${BASE_MODEL}" | tee -a "${LOG_DIR}/run.log"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/train_sft.py \
    torchtitan/experiments/ezpz/rl/datasets_sft.py 2>&1 | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

ezpz launch --np 24 -ppn 12 --timeout "${IDLE_TIMEOUT:-1800}" \
    python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --sft_dataset gsm8k-r1cot \
    --model_name_or_path "${BASE_MODEL}" \
    --output_dir "${CKPT_DIR}" \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_length 2048 \
    --bf16 --fsdp full_shard \
    --logging_steps 5 \
    --save_strategy epoch \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== DONE B4a: ckpts in ${CKPT_DIR} (one per epoch), log in ${LOG_DIR}/ ===" \
    | tee -a "${LOG_DIR}/run.log"
