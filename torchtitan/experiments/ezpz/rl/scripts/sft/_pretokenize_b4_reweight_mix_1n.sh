#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=12:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe
#
# 1N offline DOWNLOAD + PRE-TOKENIZE of b4_reweight_mix @ 4096 for the B4b
# reweighted single-stage SFT (docs/production/sft/agpt/2b-mds/
# b4-finish-and-reweight/).
#
# WHY: same rationale as _pretokenize_b3_instruct_cot_mix_1n.sh -- SFTTrainer
# re-tokenizes+packs the WHOLE train dataset at job start, so pre-tokenizing
# here moves that cost to a dedicated one-time job via --pretokenize_to. The
# 8N training job then loads it with --pretokenized_dataset; TRL sees
# input_ids and SKIPS prep, so training starts in <60s.
#
# b4_reweight_mix fixes the B3 dilution regression (cot_accuracy 0.05 vs B2's
# 0.205): gsm8k-r1cot restored to 0.40 (was 0.15 in B3), OpenR1-Math-220k cut
# to 0.15 and LENGTH-FILTERED (OPENR1_MAX_THINK_CHARS drops run-on <think>
# traces above ~1200 chars -- the traced cause of B3's verbose-wrong-answer
# style), tulu-3-sft-mixture 0.30, ultrachat-200k 0.15. OpenMathInstruct-2 is
# DROPPED entirely (added math breadth the 2B couldn't convert to accuracy).
# See torchtitan/experiments/ezpz/docs/production/sft/agpt/2b-mds/
# b4-finish-and-reweight/design.md for the full diagnosis.
#
# CRITICAL: OPENR1_MAX_THINK_CHARS must be exported here so the filter is
# ACTIVE while this job builds+caches the mix -- the whole point of B4b is
# training on the filtered traces. Without it the mix cache would silently
# fall back to the class default (also 1200, see datasets_sft.py), but we set
# it explicitly so the value used is visible in this script and in the run
# log, not implicit.
#
# max_length 4096 (vs B3's 8192): after the OpenR1 length filter there are no
# long traces left worth preserving, and gsm8k-r1cot / tulu / ultrachat are
# all short-to-medium. Halves the attention cost of B3's 8192 packing.
#
# This mix pulls the same style of HF datasets as b3_instruct_cot_mix (not
# all pre-cached locally), so proxy is ON and both the raw downloads and the
# interleaved-mix cache land under the shared /tegu/datasets staging area
# (datasets group, separate quota) rather than ~/.cache.
#
# num_proc: dataset_num_proc controls BOTH the tokenize map AND the packing
# map. 32 matches the safe EzpzSFTConfig default used for b3 (96/64 SIGBUS'd
# the tulu_math_uc_mix packing map at scale -- see that pretokenize's header).
# Plenty of 12h walltime headroom; this mix is far smaller than b3's 13.3M
# rows (OpenMath dropped + OpenR1 filtered), so expect ~1-2h actual runtime.
# Runs on ONE rank (rank 0 does the save); no distributed collective, no
# oneCCL barrier.
#
# Output: pre-tokenized dataset at
#   /tegu/datasets/datasets/agpt2b-b4-reweight-len4096/
# consumed by agpt2b_b4b_reweight_8n.sh via --pretokenized_dataset.
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
# MUST be set so the OpenR1 length filter is active while THIS job builds and
# caches the mix -- see header. Default matches the class default (1200) but
# is set explicitly here so it is visible in the run log.
export OPENR1_MAX_THINK_CHARS="${OPENR1_MAX_THINK_CHARS:-1200}"

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"
source .venv/bin/activate
python3 -c "import trl; print('trl', trl.__version__)" || { echo "FATAL: trl missing"; exit 1; }

# num_proc override knob: walk down (e.g. NPROC=16) if the 4096 packing map
# SIGBUSes -- see the tulu pretokenize memory lesson in the header comment.
NPROC="${NPROC:-32}"
BASE_MODEL="${HOME}/global_step138650"
OUT_DIR="/tegu/datasets/datasets/agpt2b-b4-reweight-len4096"
LOG_DIR="logs/pretokenize-b4-reweight-mix-${PBS_JOBID%%.*}"
mkdir -p "${LOG_DIR}"

echo "=== 1N DOWNLOAD+PRETOKENIZE: b4_reweight_mix -> ${OUT_DIR} (len=4096, base=gs138650) ===" \
    | tee -a "${LOG_DIR}/run.log"
echo "OPENR1_MAX_THINK_CHARS=${OPENR1_MAX_THINK_CHARS}" | tee -a "${LOG_DIR}/run.log"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/train_sft.py \
    torchtitan/experiments/ezpz/rl/datasets_sft.py 2>&1 | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

# Single-rank launch: pre-tokenize is a data job, no model training / no
# collectives. `ezpz launch --np 1` keeps the ezpz env plumbing but runs one
# rank.
ezpz launch --np 1 python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --sft_dataset b4_reweight_mix \
    --model_name_or_path "${BASE_MODEL}" \
    --pretokenize_to "${OUT_DIR}" \
    --max_length 4096 \
    --dataset_num_proc "${NPROC:-32}" \
    --output_dir "${LOG_DIR}/_pretok_scratch" \
    --report_to none \
    2>&1 | tee -a "${LOG_DIR}/run.log"

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== DONE: pretokenized dataset at ${OUT_DIR}; log in ${LOG_DIR}/ ===" \
    | tee -a "${LOG_DIR}/run.log"
du -sh "${OUT_DIR}" 2>/dev/null | tee -a "${LOG_DIR}/run.log"
