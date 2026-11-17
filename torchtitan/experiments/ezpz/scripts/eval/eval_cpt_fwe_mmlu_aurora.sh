#!/bin/bash --login
#PBS -l select=1
#PBS -l walltime=48:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -A AuroraGPT
#PBS -j oe
#
# MMLU (MCF + CF + max) for CPT-FWE on Aurora.
# NOTE: workq with walltime 48:00:00 is also valid on Aurora; capacity is
# preferred for long MMLU sweeps. Swap -q capacity -> -q workq if needed.
#
# Band / scoring notes:
#   - Random band for MMLU 4-way MC is ~0.2295-0.2689; always-answer-D
#     control is 0.2689 on this harness snapshot. Scores near this band are
#     at chance; use delta vs band, not raw acc, to judge signal.
#   - Group key: mmlu (lm-eval aggregates 57 subtasks under group "mmlu").
#   - Report max(CF, MCF): mmlu = CF (continuation / loglikelihood over
#     answer strings), mmlu_continuation is alias; mmlu MCF variant scores
#     letter-choice scoring. Take max per subtask then micro-average, and
#     also report group-level max(CF,MCF) headline.
#   - Letter-prior fit on base/MDS checkpoints: p~(0.37, 0.32, 0.20, 0.11)
#     for (A,B,C,D). A constant-D predictor at 0.2689 beats most 2B
#     checkpoints; this is the letter prior, not knowledge.
#
# Usage:
#   CKPT=/path/to/hf qsub eval_cpt_fwe_mmlu_aurora.sh
#
# qsub loop examples:
#   for s in $(seq 74400 200 88000); do
#     HF=/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz/outputs/evals/cpt-fwe/step-${s}/hf
#     [ -d "$HF" ] || HF=/home/foremans/torchtitan/outputs/evals/cpt-fwe/step-${s}/hf
#     [ -d "$HF" ] || continue
#     CKPT="$HF" qsub torchtitan/experiments/ezpz/scripts/eval/eval_cpt_fwe_mmlu_aurora.sh
#   done
#
#   # HOME sparse HF (11600/13000/.../74200) mirrored on Aurora:
#   for s in 11600 13000 17000 34800 67400 72800 73000 74000 74200; do
#     HF=/home/foremans/torchtitan/outputs/evals/cpt-fwe/step-${s}/hf
#     [ -d "$HF" ] || continue
#     CKPT="$HF" qsub torchtitan/experiments/ezpz/scripts/eval/eval_cpt_fwe_mmlu_aurora.sh
#   done

set -euo pipefail

module load frameworks/2026.1.0
module load pti-gpu

if [ -d /opt/intel/pti-gpu/lib64 ]; then
    export LD_LIBRARY_PATH="/opt/intel/pti-gpu/lib64:${LD_LIBRARY_PATH:-}"
fi
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_AFFINITY_MASK=0
export HF_DATASETS_OFFLINE=1

# Aurora checkout path (canonical); fall back to PBS_O_WORKDIR if present.
AURORA_REPO="/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz"
if [ -d "$AURORA_REPO/venvs/aurora/tt-lm-eval/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$AURORA_REPO/venvs/aurora/tt-lm-eval/bin/activate"
elif [ -n "${PBS_O_WORKDIR:-}" ] && [ -f "$PBS_O_WORKDIR/venvs/aurora/tt-lm-eval/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$PBS_O_WORKDIR/venvs/aurora/tt-lm-eval/bin/activate"
fi

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

# Full 14k MMLU, 5-shot, no --limit. lm-eval uses --num_fewshot 5 for both
# mmlu (MCF-style) and mmlu_continuation (CF-style); headline is max(CF,MCF).
# Device xpu:0; handle both xpu:0 and legacy device parse in lm_eval.
python -m lm_eval \
    --model hf \
    --model_args "pretrained=$CKPT" \
    --tasks "$TASKS" \
    --device xpu:0 \
    --batch_size auto \
    --output_path "$OUT_DIR/results_mmlu.json" \
    --num_fewshot 5

echo "[mmlu] done STEP=$STEP_TAG -> $OUT_DIR/results_mmlu.json"
