#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=02:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe
#
# B2v2: stronger stage-2 cold-start SFT extending B2's proven two-stage lineage.
#
# Lineage:  global_step138650 -> ckpt-900 (tulu_math_uc_mix, 8N) -> THIS
#   B2 was ckpt-900 -> gsm8k-r1cot (7.4k ex, 93 steps, 3ep) = 0.205 GSM8K-CoT
#   B2v2 is ckpt-900 -> distill_cot_mix (OpenThoughts-114k-decontam + gsm8k-r1cot)
#   Broad frontier CoT traces replace the narrow GSM8K-only signal.
#   Same envelope (<think>/<answer>\boxed{}), GSM8K-test-decontaminated.
#
# Dataset: distill_cot_mix = 50% OpenThoughts-114k-decontam + 50% gsm8k-r1cot
#   Uses datasets_sft.py mix-spec: "OpenThoughts-114k-decontam:0.5,gsm8k-r1cot:0.5"
#   Alternative: pure distill_cot_mix (OpenThoughts + OpenR1) if we want
#   even broader CoT without GSM8K anchoring.
#
# Why this lever: POST-TRAINING-2B.md next-step #1 says "stronger/longer
# cold-start SFT -- more epochs, better CoT data, or STaR". B3/B4 proved
# single-stage rebuilds cannot beat B2's two-stage structure. B2v2 keeps
# the two-stage structure and swaps in richer CoT data.
#
# Smoke first: 1 epoch validates loss descends + dataset loads.
# Full: 3 epochs for apple-to-apple vs B2's 3ep, then eval vs B2's 0.205.
#
# Usage:
#   qsub torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh       # 3ep (default)
#   qsub -v NUM_EPOCHS=1 torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh  # smoke

set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export HF_DATASETS_OFFLINE=0
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

VENV="${VENV:-/lus/tegu/projects/datascience/foremans/venvs/rl-monarch-torch214-locality-20261001}"
UTILS="${HOME}/stage/utils.sh"
if [[ -f "${UTILS}" ]]; then
    source "${UTILS}" && ezpz_get_job_env
else
    source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_get_job_env
fi
# Force correct venv via PATH — ezpz_get_job_env clobbers VIRTUAL_ENV to
# stale xpu-torch214 (no trl). VIRTUAL_ENV alone isn't enough; need PATH.
export VIRTUAL_ENV="${VENV}"
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${VENV}/lib/python3.12/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
# Prepend venv's bundled UR/SYCL so it wins over stale system 2026.0 on x1921 nodes
export LD_LIBRARY_PATH="${VENV}/lib:${LD_LIBRARY_PATH:-}"

python3 -c "import trl; print('trl', trl.__version__)" || { echo "FATAL: trl missing"; exit 1; }
python3 -c "import torch; print('torch', torch.__version__, 'xpu', torch.xpu.is_available())" 2>&1 | head -2

# Base = ckpt-900 sharded output (re-consolidate if -hf missing, else use sharded directly)
# train_sft.py accepts either HF dir or sharded DCP via --model_name_or_path
BASE_MODEL="${BASE_MODEL:-$SUBMIT_DIR/outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900-hf}"
# Fallback: if checkpoint-900 doesn't exist, try the alternate path
if [[ ! -d "${BASE_MODEL}" ]]; then
    BASE_MODEL="$SUBMIT_DIR/outputs/sft/agpt2b-gsm8k-r1cot-2n"
    echo "WARN: checkpoint-900 not found, falling back to ${BASE_MODEL}" >&2
fi

NUM_EPOCHS="${NUM_EPOCHS:-3}"
CKPT_DIR="outputs/sft/agpt2b-b2v2-distill-cot-2n-ep${NUM_EPOCHS}"
LOG_DIR="logs/sft-b2v2-distill-cot-${PBS_JOBID%%.*}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

# Validate base exists (check for either HF safetensors or sharded FSDP)
if [[ ! -d "${BASE_MODEL}" ]]; then
    echo "FATAL: BASE_MODEL not found: ${BASE_MODEL}" >&2
    echo "Available SFT outputs:" >&2
    ls -d outputs/sft/*/ 2>&1 | head -10 >&2
    exit 1
fi

# Dataset: gsm8k-r1cot alone, more epochs — proven B2 path, just longer.
# distill_cot_mix (OpenThoughts) hit HF cache race on multi-rank; shelve it
# until we fix the streaming+cache coherency or pretokenize.
SFT_DATASET="${SFT_DATASET:-gsm8k-r1cot}"

echo "=== B2v2: distill CoT SFT on ckpt-900, ${NUM_EPOCHS}ep, dataset=${SFT_DATASET} ===" | tee "${LOG_DIR}/run.log"
echo "BASE_MODEL=${BASE_MODEL}" | tee -a "${LOG_DIR}/run.log"
echo "SFT_DATASET=${SFT_DATASET}" | tee -a "${LOG_DIR}/run.log"
echo "CKPT_DIR=${CKPT_DIR}" | tee -a "${LOG_DIR}/run.log"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/train_sft.py \
    torchtitan/experiments/ezpz/rl/datasets_sft.py 2>&1 | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

ezpz launch --np 24 -ppn 12 --timeout "${IDLE_TIMEOUT:-1800}" \
    python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --sft_dataset "${SFT_DATASET}" \
    --model_name_or_path "${BASE_MODEL}" \
    --output_dir "${CKPT_DIR}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_length 2048 \
    --dataset_num_proc 8 \
    --bf16 --fsdp full_shard \
    --logging_steps 5 \
    --save_strategy epoch \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== DONE B2v2: ckpts in ${CKPT_DIR}, log in ${LOG_DIR}/ ===" | tee -a "${LOG_DIR}/run.log"
echo "Next: consolidate + eval GSM8K-CoT vs B2's 0.205" | tee -a "${LOG_DIR}/run.log"
echo "  ./torchtitan/experiments/ezpz/scripts/eval/consolidate_and_eval_cot.sh ${CKPT_DIR}/checkpoint-<N>" | tee -a "${LOG_DIR}/run.log"
