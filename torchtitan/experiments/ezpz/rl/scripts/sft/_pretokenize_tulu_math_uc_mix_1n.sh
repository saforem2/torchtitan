#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=12:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe
#
# 1N offline PRE-TOKENIZE of the big tulu_math_uc_mix for the gs138650 SFT.
#
# WHY: SFTTrainer re-tokenizes+packs the WHOLE train dataset at job start
# (~2.5k examples/s), so the 93.1M-row tulu_math_uc_mix takes ~8.5h before the
# first training step. In the 32N job (12470281) that blew past the auto-retry
# idle-watchdog, which SIGTERM'd the job at ~19 min (rc=143) -- the oneCCL
# allreduce errors in that log were teardown noise, NOT the v2-base scale crash.
# The 2N smoke never hit this because --max_train_samples 50000 truncated BEFORE
# tokenize.
#
# FIX (mirrors the interleave pre-build): run TRL's tokenize+pack ONCE here via
# --pretokenize_to, save the processed dataset (with input_ids) to disk. The
# 32N training job then loads it with --pretokenized_dataset; TRL sees input_ids
# and SKIPS prep, so training starts in <60s. train_sft.py saves
# trainer.train_dataset directly, so the offline result is bit-identical to
# in-process prep (same chat template, packing, assistant masks, max_length).
#
# num_proc: default dataset_num_proc=32 in EzpzSFTConfig; a Sunspot SPR node has
# 104 cores, so 96 procs ~3x the per-node tokenize throughput. Even at 96 procs
# 93M rows is a few hours -- hence the 12h walltime. Runs on ONE rank (rank 0
# does the save); no distributed collective, no oneCCL barrier, no watchdog.
#
# Output: pre-tokenized dataset at
#   ~/.cache/ezpz_sft_mixes/tokenized/tulu_math_uc_mix-gs138650-len2048/
# (keyed by mix + tokenizer(base) + max_length; a different base tokenizer or
# max_length needs its own pre-tokenize.)
set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export HF_HUB_OFFLINE=1   # mix is on-disk cached; base tokenizer is local
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"
source .venv/bin/activate
python3 -c "import trl; print('trl', trl.__version__)" || { echo "FATAL: trl missing"; exit 1; }

BASE_MODEL="${HOME}/global_step138650"
OUT_DIR="${HOME}/.cache/ezpz_sft_mixes/tokenized/tulu_math_uc_mix-gs138650-len2048"
LOG_DIR="logs/pretokenize-tulu-math-uc-mix-${PBS_JOBID%%.*}"
mkdir -p "${LOG_DIR}"

echo "=== 1N PRE-TOKENIZE: tulu_math_uc_mix -> ${OUT_DIR} (len=2048, base=gs138650) ===" \
    | tee "${LOG_DIR}/run.log"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/train_sft.py \
    torchtitan/experiments/ezpz/rl/datasets_sft.py 2>&1 | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

# Single-rank launch: pre-tokenize is a data job, no model training / no
# collectives. `ezpz launch --np 1` keeps the ezpz env plumbing but runs one
# rank. max_length 2048 + dataset_num_proc 96 MUST match the training job's
# --max_length (2048) and the base tokenizer, or TRL's skip-prep produces the
# wrong shapes / a silent mismatch.
ezpz launch --np 1 python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --sft_dataset tulu_math_uc_mix \
    --model_name_or_path "${BASE_MODEL}" \
    --pretokenize_to "${OUT_DIR}" \
    --max_length 2048 \
    --dataset_num_proc 96 \
    --report_to none \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== DONE: pre-tokenized dataset at ${OUT_DIR}; log in ${LOG_DIR}/ ===" \
    | tee -a "${LOG_DIR}/run.log"
