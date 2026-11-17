#!/bin/bash --login
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -l filesystems=tegu:home
#PBS -q workq
#PBS -A datascience
#PBS -j oe
#
# Commonsense sweep: HS / ARC-E / PIQA / ARC-C for CPT-FWE HF exports.
# Sunspot (Intel Data Center GPU Max / Ponte Vecchio XPU).
#
# Usage:
#   CKPT=/path/to/hf/step-XXXXX qsub eval_cpt_fwe_commonsense_sweep.sh
#   # or override tasks:
#   CKPT=/path/to/hf TASKS=hellaswag,arc_easy qsub eval_cpt_fwe_commonsense_sweep.sh
#
# qsub examples looping over inventory steps:
#   # Single step:
#   CKPT=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-74400/hf \
#     qsub torchtitan/experiments/ezpz/scripts/eval/eval_cpt_fwe_commonsense_sweep.sh
#
#   # Loop over Tegu HF dense ladder (every 200 from 74400..88000):
#   for s in $(seq 74400 200 88000); do
#     HF=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-${s}/hf
#     [ -d "$HF" ] || continue
#     CKPT="$HF" qsub torchtitan/experiments/ezpz/scripts/eval/eval_cpt_fwe_commonsense_sweep.sh
#   done
#
#   # Loop over HOME HF sparse set:
#   for s in 11600 13000 17000 34800 67400 72800 73000 74000 74200; do
#     HF=/home/foremans/torchtitan/outputs/evals/cpt-fwe/step-${s}/hf
#     [ -d "$HF" ] || continue
#     CKPT="$HF" qsub torchtitan/experiments/ezpz/scripts/eval/eval_cpt_fwe_commonsense_sweep.sh
#   done

set -eo pipefail
unset ZSH_EVAL_CONTEXT 2>/dev/null || true

source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
module load frameworks/2026.1.0
module load pti-gpu
set -u

# PTI lib64 on LD_LIBRARY_PATH (pti-gpu module usually does this; ensure it)
if [ -d /opt/intel/pti-gpu/lib64 ]; then
    export LD_LIBRARY_PATH="/opt/intel/pti-gpu/lib64:${LD_LIBRARY_PATH:-}"
fi

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_AFFINITY_MASK=0
export HF_DATASETS_OFFLINE=1

CKPT="${CKPT:-}"
TASKS="${TASKS:-hellaswag,arc_easy,piqa,arc_challenge}"

if [ -z "$CKPT" ]; then
    echo "ERROR: CKPT env var must point to an HF checkpoint dir, e.g. .../step-74400/hf" >&2
    exit 1
fi

if [ ! -d "$CKPT" ]; then
    echo "ERROR: CKPT dir not found: $CKPT" >&2
    exit 1
fi

PYTHON=/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/bin/python

# Derive HF_ROOT and step tag for output placement.
# CKPT is .../step-$s/hf -> HF_ROOT is .../step-$s
HF_ROOT="$(dirname "$CKPT")"
STEP_TAG="$(basename "$HF_ROOT")"
OUT_DIR="$HF_ROOT/results"
mkdir -p "$OUT_DIR"

echo "[commonsense] CKPT=$CKPT"
echo "[commonsense] HF_ROOT=$HF_ROOT STEP=$STEP_TAG OUT_DIR=$OUT_DIR"
echo "[commonsense] TASKS=$TASKS"

# Split tasks into two invocations to handle different num_fewshot:
#   0-shot: hellaswag, arc_easy, piqa
#   25-shot: arc_challenge
TASKS_0SHOT=""
TASKS_25SHOT=""

# Normalize TASKS into arrays by comma
IFS=',' read -ra REQ <<< "$TASKS"
for t in "${REQ[@]}"; do
    # trim whitespace
    t="$(echo "$t" | xargs)"
    [ -z "$t" ] && continue
    case "$t" in
        arc_challenge) TASKS_25SHOT="${TASKS_25SHOT:+$TASKS_25SHOT,}$t" ;;
        *)             TASKS_0SHOT="${TASKS_0SHOT:+$TASKS_0SHOT,}$t" ;;
    esac
done

# Helper: run lm_eval with given tasks and fewshot, writing to OUT_DIR.
run_eval() {
    local tasks="$1"
    local fewshot="$2"
    local suffix="$3"
    [ -z "$tasks" ] && return 0
    echo "[commonsense] lm_eval tasks=$tasks num_fewshot=$fewshot -> $OUT_DIR/results_${suffix}.json"
    "$PYTHON" -m lm_eval \
        --model hf \
        --model_args "pretrained=$CKPT" \
        --tasks "$tasks" \
        --device xpu:0 \
        --batch_size auto \
        --output_path "$OUT_DIR/results_${suffix}.json" \
        --num_fewshot "$fewshot"
}

run_eval "$TASKS_0SHOT" 0 "0shot"
run_eval "$TASKS_25SHOT" 25 "25shot"

# Also emit a convenience symlink/copy for callers expecting $HF_ROOT/step-$s/results/
# (OUT_DIR already is that location; also mirror next to CKPT for legacy paths)
if [ "$OUT_DIR" != "$CKPT/../results.json" ]; then
    echo "[commonsense] results in $OUT_DIR"
fi

echo "[commonsense] done STEP=$STEP_TAG"
