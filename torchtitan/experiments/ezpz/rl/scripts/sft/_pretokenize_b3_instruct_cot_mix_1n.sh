#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=24:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe
#
# 1N offline DOWNLOAD + PRE-TOKENIZE of b3_instruct_cot_mix @ 8192 for the B3
# cold-start SFT (docs/live/chains/sft/agpt/2b-mds/b3-instruct-cot-mix/).
#
# WHY: SFTTrainer re-tokenizes+packs the WHOLE train dataset at job start, so
# a multi-million-row mix takes hours -- long enough that the auto-retry
# idle-watchdog SIGTERMs the training job before the first step (the same
# failure mode hit by tulu_math_uc_mix, job 12470281). Pre-tokenizing moves
# that cost to a dedicated one-time job: run TRL's tokenize+pack ONCE here via
# --pretokenize_to, save the processed dataset (with input_ids) to disk. The
# 8N training job then loads it with --pretokenized_dataset; TRL sees
# input_ids and SKIPS prep, so training starts in <60s. train_sft.py saves
# trainer.train_dataset directly, so the offline result is bit-identical to
# in-process prep (same chat template, packing, assistant masks, max_length).
#
# This mix pulls 5 HF datasets that are NOT all pre-cached locally (unlike
# tulu_math_uc_mix's 3 sources), so proxy is ON here and the raw downloads +
# the interleaved-mix cache both land under the shared /tegu/datasets staging
# area (datasets group, separate quota) rather than ~/.cache, so the 8N job
# and any teammate can reuse them without re-downloading.
#
# num_proc: dataset_num_proc controls BOTH the tokenize map AND the packing
# map. The tulu_math_uc_mix pretokenize (this script's template) found 96/64
# SIGBUS the packing map at scale (~26TB aggregate vmem); 32 is the safe
# EzpzSFTConfig default -- still parallel, far lower peak memory. Plenty of
# 12h walltime headroom. Runs on ONE rank (rank 0 does the save); no
# distributed collective, no oneCCL barrier.
#
# Output: pre-tokenized dataset at
#   /tegu/datasets/datasets/agpt2b-b3-instruct-cot-mix-len8192/
# consumed by Tasks 4/5 via --pretokenized_dataset. (A different base
# tokenizer or max_length needs its own pre-tokenize.)
set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
# Shared dataset staging area (datasets group, separate quota) -- the raw HF
# downloads for this mix are NOT all pre-cached locally, so this run needs
# network access (proxy on, no HF_HUB_OFFLINE).
export HF_HOME=/tegu/datasets/datasets/hf
export HF_DATASETS_CACHE=/tegu/datasets/datasets/hf/datasets
# Materialized-mix cache also under the shared area so the 8N job's mix build
# (if ever needed) finds the same interleave result.
export EZPZ_SFT_MIX_CACHE_DIR=/tegu/datasets/datasets/ezpz_sft_mixes

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"
source .venv/bin/activate
python3 -c "import trl; print('trl', trl.__version__)" || { echo "FATAL: trl missing"; exit 1; }

# num_proc override knob: walk down (e.g. NPROC=16) if the 8192 packing map
# SIGBUSes -- see the tulu pretokenize memory lesson in the header comment.
NPROC="${NPROC:-32}"
BASE_MODEL="${HOME}/global_step138650"
OUT_DIR="/tegu/datasets/datasets/agpt2b-b3-instruct-cot-mix-len8192"
LOG_DIR="logs/pretokenize-b3-instruct-cot-mix-${PBS_JOBID%%.*}"
mkdir -p "${LOG_DIR}"

echo "=== 1N DOWNLOAD+PRETOKENIZE: b3_instruct_cot_mix -> ${OUT_DIR} (len=8192, base=gs138650) ===" \
    | tee -a "${LOG_DIR}/run.log"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/train_sft.py \
    torchtitan/experiments/ezpz/rl/datasets_sft.py 2>&1 | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

# Single-rank launch: pre-tokenize is a data job, no model training / no
# collectives. `ezpz launch --np 1` keeps the ezpz env plumbing but runs one
# rank.
ezpz launch --np 1 python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --sft_dataset b3_instruct_cot_mix \
    --model_name_or_path "${BASE_MODEL}" \
    --pretokenize_to "${OUT_DIR}" \
    --max_length 8192 \
    --dataset_num_proc "${NPROC:-32}" \
    --output_dir "${LOG_DIR}/_pretok_scratch" \
    --report_to none \
    2>&1 | tee -a "${LOG_DIR}/run.log"

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== DONE: pretokenized dataset at ${OUT_DIR}; log in ${LOG_DIR}/ ===" \
    | tee -a "${LOG_DIR}/run.log"
du -sh "${OUT_DIR}" 2>/dev/null | tee -a "${LOG_DIR}/run.log"
