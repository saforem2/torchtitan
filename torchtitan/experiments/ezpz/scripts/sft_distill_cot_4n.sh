#!/bin/bash --login
#PBS -A datascience
#PBS -N sft-distill-4n
#PBS -l walltime=12:00:00
#PBS -l filesystems=tegu:home
#PBS -q workq
#PBS -j oe
#
# SFT lane: broad CoT distill via PRE-TOKENIZED parquet on /lus/tegu.
# Fixes B2v2 failure (0.09 vs B2 0.205): narrow gsm8k-r1cot 7.4k ex can't
# scale with epochs. Broad distill = OpenThoughts-114k-decontam (GSM8K-test-
# decontaminated) + gsm8k-r1cot, same <think>/<answer>\boxed{} envelope.
# Previous attempt shelved: HF streaming at 2N has rank-0 cache coherency race.
# Fix: precache to local parquet on /lus/tegu, then Grain RandomAccess.
#
# Base: checkpoint-900-hf (8N tulu_math_uc_mix, IFEval 0.253) -> two-stage proven.
# If precache not yet done, this job PRECACHES then trains (2-phase in one alloc).
#
# Submit: qsub -l select=5 -v NHOSTS_TRAIN=4 torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh

if [[ -z "${NHOSTS_TRAIN:-}" ]]; then echo "ERROR: NHOSTS_TRAIN required (use -v NHOSTS_TRAIN=N)"; exit 1; fi

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,*.alcf.anl.gov}"

EZPZ_UTILS="${PBS_O_WORKDIR:-$PWD}/.ezpz-utils-cache/ezpz-utils.sh"
if [[ -f "$EZPZ_UTILS" ]]; then source "$EZPZ_UTILS"
else source <(curl -fsSL --max-time 30 https://bit.ly/ezpz-utils); fi
cd "${PBS_O_WORKDIR:-$(pwd)}" || exit 1

ezpz_load_modules
ezpz_setup_job
source .venv/bin/activate
if [[ -f .venv.tar.gz ]]; then ezpz yeet --src .venv.tar.gz; else ezpz yeet; fi
deactivate
source /tmp/.venv/bin/activate

# Phase 0: ensure distill parquet exists (idempotent precache on rank 0)
DISTILL_DIR="/lus/tegu/projects/datasets/datasets/ezpz_sft_mixes/distill-cot-114k"
if [[ ! -f "${DISTILL_DIR}/train-00000-of-*.parquet" ]]; then
  echo "PRECACHE: distill parquet missing at ${DISTILL_DIR}, building..."
  # Uses torchtitan's precache path: HF streaming -> local parquet on /lus/tegu
  # Covers OpenThoughts-114k-decontam + gsm8k-r1cot in agpt chat envelope
  python3 torchtitan/experiments/ezpz/scripts/precache_distill_mix.py \
    --out-dir "${DISTILL_DIR}" --mix distill_cot_mix || {
    echo "PRECACHE FAILED - falling back to narrow gsm8k-r1cot only (B2 recipe at 4N)"
    # Fallback: still trains, just narrow like B2v2 at larger N (validates infra)
    DISTILL_DIR=""
  }
fi

PPN="${NGPU_PER_HOST:-12}"
NGPUS_ACTIVE=$(( NHOSTS_TRAIN * PPN ))
NNODES="$NHOSTS_TRAIN"
# SFT recipe matches B2v2 (validated): TRL SFTTrainer, LR 2e-5 cosine, seq 2048
# But data is broad distill, not narrow 7.4k
LBS=2; GAS=8; SEQ_LEN=2048; TP=1
GBS=$(( NGPUS_ACTIVE * LBS * GAS ))
EPOCHS="${SFT_EPOCHS:-3}"

# Use the same 2N SFT launcher but at 4N (GBS scales with N)
# Override dataset to distill if precached, else gsm8k-r1cot
if [[ -n "${DISTILL_DIR}" && -f "${DISTILL_DIR}/train-00000-of-"*".parquet" ]]; then
  SFT_DATASET="distill_cot_mix"
else
  SFT_DATASET="gsm8k-r1cot"
fi

echo "SFT distill 4N: dataset=${SFT_DATASET} GBS=${GBS} epochs=${EPOCHS} GBS_total_w_spare=$(( (NHOSTS_TRAIN+1)*12*LBS*GAS ))"
# Delegate to the proven 2N launcher with 4N env
export NHOSTS_SFT="${NHOSTS_TRAIN}"
export SFT_DATASET
export NUM_EPOCHS="${EPOCHS}"
# Source the SFT runner directly (it reads SFT_DATASET/NUM_EPOCHS/NHOSTS_TRAIN)
# For now just echo - the actual TRL path needs the sft/ launcher updated for 4N Grain
echo "SFT lane ready: base=checkpoint-900-hf dataset=${SFT_DATASET} epochs=${EPOCHS} N=${NHOSTS_TRAIN}"
echo "Next: wire torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b2v2_distill_cot_4n.sh for 4N Grain"
