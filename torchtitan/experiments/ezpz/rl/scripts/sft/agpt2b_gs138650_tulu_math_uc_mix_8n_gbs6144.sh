#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=12:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=8
#PBS -q workq
#PBS -j oe
#
# 8N Sunspot SFT continuing the global_step138650 base (vocab 256000) on the BIG
# tulu_math_uc_mix (~93.1M rows -> 53.3M packed seqs at len1024, ~54B tokens),
# 1 epoch at production-equivalent GBS=6144.
#
# WHY 8N (not 32N): the 32N run GPU-page-faults at step 1-2
# ("Segmentation fault from GPU ... NotPresent Write" -> rank signal 6). Full
# diagnosis in docs/experiments/agpt/sunspot/2026-07-10-sft-2b-gs138650-big-mix-32n.md:
# NOT a bad node (reproduces on rank 221 across nodes), NOT an OOV token id
# (full 53M-row scan: max 255998 < vocab 256000) -- it is a 384-rank SCALE
# fault (the project_sft_v2_base_oom_badnode class, base-independent). A scale
# bisect (jobs 12470343/346/347/348/349) found the onset between 8N and 12N:
#   2N/4N/8N (24/48/96 ranks) CLEAN ; 12N/16N/32N (144/192/384) GPU segfault.
# So 8N = 96 ranks is the largest proven-safe scale. Same len1024 pretokenized
# dataset + bsz2/gas config as the (crashing) 32N run -- only the scale differs.
#
# Sizing (GBS held at 6144, so LR calibration + step count match the 32N plan):
#   NGPUS = 8 nodes x 12 ranks = 96
#   GBS = NGPUS x per_device_train_batch_size x gradient_accumulation_steps
#       = 96 x 2 x 32 = 6144
#   tokens/step = GBS x max_length = 6144 x 1024 = 6.29M
#   per-device micro-batch = 2 x 1024 = 2048 tokens (proven-fitting at 1024)
#
# 1 epoch over ~54B tokens at 6.29M tokens/step is ~8.6k steps. At 8N the
# per-step wall is ~4x the 32N rate, so this needs several 12h windows: chained
# afterany continuations + --resume_from_checkpoint carry it across.
#
# Output: outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/

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
python3 -c "import trl; print('trl', trl.__version__)" || { echo "FATAL: trl missing"; exit 1; }

BASE_MODEL="${HOME}/global_step138650"
CKPT_DIR="outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144"
LOG_DIR="logs/sft-agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144-${PBS_JOBID%%.*}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

echo "=== 8N SFT: gs138650 + tulu_math_uc_mix (BIG cached mix), 1 epoch, GBS=6144, seq=1024 ===" \
    | tee "${LOG_DIR}/run.log"
echo "BASE_MODEL=${BASE_MODEL}" | tee -a "${LOG_DIR}/run.log"
echo "Sizing: 96 ranks x bsz=2 x gas=32 = GBS=6144, 6.29M tokens/step (8N = largest safe scale)" \
    | tee -a "${LOG_DIR}/run.log"
echo "loading PRE-TOKENIZED dataset (TRL skips tokenize+pack; <60s to step 1)" \
    | tee -a "${LOG_DIR}/run.log"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/train_sft.py \
    torchtitan/experiments/ezpz/rl/datasets_sft.py 2>&1 | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

# PRE-TOKENIZED len1024 dataset (input_ids + assistant_masks + seq_lengths).
# train_sft.py auto-sets packing=False + assistant_only_loss=False on this path.
PRETOK_DIR="${HOME}/.cache/ezpz_sft_mixes/tokenized/tulu_math_uc_mix-gs138650-len1024"
if [[ ! -d "${PRETOK_DIR}" ]]; then
    echo "FATAL: pre-tokenized dataset missing at ${PRETOK_DIR}" \
        | tee -a "${LOG_DIR}/run.log"
    echo "Run _pretokenize_tulu_math_uc_mix_1n.sh first." \
        | tee -a "${LOG_DIR}/run.log"
    exit 1
fi

# NO --auto-retry (stuck_pre_training false-positive on TRL's {'loss':...} output
# -- ezpz auto-retry only recognizes its own step=N marker). Fault tolerance via
# the afterany chain + --resume_from_checkpoint + save_steps=50; a fresh PBS
# allocation per chain link also sidesteps any single bad node.
ezpz launch --np 96 -ppn 12 --timeout "${IDLE_TIMEOUT:-1800}" \
    python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --pretokenized_dataset "${PRETOK_DIR}" \
    --model_name_or_path "${BASE_MODEL}" \
    --output_dir "${CKPT_DIR}" \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --max_length 1024 \
    --bf16 --fsdp full_shard \
    --logging_steps 10 \
    --save_strategy steps --save_steps 50 \
    --report_to wandb \
    --resume_from_checkpoint "${CKPT_DIR}" \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== DONE: log in ${LOG_DIR}/, ckpts in ${CKPT_DIR}/ ===" \
    | tee -a "${LOG_DIR}/run.log"
