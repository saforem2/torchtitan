#!/bin/bash --login
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -l filesystems=tegu:home
#PBS -q workq
#PBS -A datascience
#PBS -j oe
#
# MMLU (MCF + CF + max) for CPT-FWE on Sunspot.
# Sunspot variant of eval_cpt_fwe_mmlu_aurora.sh: same lm_eval harness,
# same TASKS, but Sunspot PBS header + Tegu HF paths. Use when Aurora
# capacity is unavailable or to avoid Aurora->Tegu HF rsync.
#
# Usage:
#   CKPT=/path/to/hf qsub eval_cpt_fwe_mmlu_sunspot.sh
#   # dense ladder on Tegu (79 HFs):
#   for s in 1000 5000 10000 25000 50000 75000 100000 111200 125000 136800; do
#     HF=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-${s}/hf
#     [ -d "$HF" ] || continue
#     qsub -v CKPT="$HF" torchtitan/experiments/ezpz/scripts/eval/eval_cpt_fwe_mmlu_sunspot.sh
#   done
#
# Walltime 06:00 covers 57 subtasks x 5-shot on XPU; bump to 12:00 if needed.
# Override batch_size if XPU OOR: EVAL_BATCH_SIZE=4 qsub -v CKPT=...

set -eo pipefail
unset ZSH_EVAL_CONTEXT 2>/dev/null || true

source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
module load frameworks/2026.1.0
module load pti-gpu
set -u

if [ -d /opt/intel/pti-gpu/lib64 ]; then
    export LD_LIBRARY_PATH="/opt/intel/pti-gpu/lib64:${LD_LIBRARY_PATH:-}"
fi
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_AFFINITY_MASK=0
export HF_DATASETS_OFFLINE=1

CKPT="${CKPT:-}"
TASKS="${TASKS:-mmlu,mmlu_continuation}"

if [ -z "$CKPT" ]; then
    echo "ERROR: CKPT env var must point to an HF dir" >&2
    exit 1
fi
if [ ! -d "$CKPT" ]; then
    echo "ERROR: CKPT not found: $CKPT" >&2
    exit 1
fi

HF_ROOT="$(dirname "$CKPT")"
STEP_TAG="$(basename "$HF_ROOT")"
OUT_DIR="$HF_ROOT/results"
mkdir -p "$OUT_DIR"

echo "[mmlu] CKPT=$CKPT STEP=$STEP_TAG OUT=$OUT_DIR TASKS=$TASKS"

BS="${EVAL_BATCH_SIZE:-8}"
python -m lm_eval \
    --model hf \
    --model_args "pretrained=$CKPT" \
    --tasks "$TASKS" \
    --device xpu:0 \
    --batch_size "$BS" \
    --output_path "$OUT_DIR/results_mmlu.json" \
    --num_fewshot 5

echo "[mmlu] done STEP=$STEP_TAG -> $OUT_DIR/results_mmlu.json"
