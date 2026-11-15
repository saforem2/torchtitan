#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=00:40:00
#PBS -l select=1
#PBS -l filesystems=tegu:home
#PBS -q workq
#PBS -j oe
#
# Phase 1 baseline lock — re-score both deliverables on the current harness
# (same 200-problem GSM8K split) per POST-TRAINING-PLAN.md Phase 1 and
# POST-TRAINING-2B.md one-line status (ckpt-93 0.205 CoT current best).
#
#   ckpt-900-hf : general baseline (IFEval 0.253, base-LM intact)
#   ckpt-93 B2  : CoT specialist  (GSM8K-CoT 0.205, format 0.985, two-stage)
#
# Each gets a float32 GSM8K-CoT eval via the same eval_cot_gsm8k.py / vLLM
# fp32 path used by consolidate_and_eval_cot.sh. Gate is informational:
# B2 expected ~0.205, ckpt-900 lower on CoT but higher on IFEval — record
# both, do not assert strict equality. Harness variance noted (B2 0.165 on
# preserved ep3-ckpt93-b2 vs old 0.205).
#
# Usage:
#   qsub torchtitan/experiments/ezpz/scripts/eval/eval_baseline_rescore.sh
#   # override HF locations when staging elsewhere:
#   qsub -v CKPT_900_HF=/path/to/checkpoint-900-hf,CKPT_93_HF=/path/to/checkpoint-93-hf torchtitan/experiments/ezpz/scripts/eval/eval_baseline_rescore.sh
#
set -o pipefail

SUBMIT_DIR=${PBS_O_WORKDIR:-$(pwd)}
cd "$SUBMIT_DIR" || exit 1

# shellcheck disable=SC1090,SC1091
source <(curl -fsSL https://ezpz.cool/utils.sh) && ezpz_setup /home/foremans/torchtitan/.venv

REPO=/home/foremans/torchtitan
# fall back to SUBMIT_DIR when REPO path not present (local dev checkout)
[[ -d "$REPO/torchtitan/experiments/ezpz/scripts/eval" ]] || REPO="$SUBMIT_DIR"

# Resolve HF dirs — prefer env override, then canonical Sunspot paths, then
# legacy/worktree-relative globs. Both checkpoints are already HF exports
# (no DCP consolidate needed here, unlike consolidate_and_eval_cot.sh).
CKPT_900_HF="${CKPT_900_HF:-}"
if [[ -z "$CKPT_900_HF" ]]; then
  for c in \
    "$REPO/outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900-hf" \
    "$REPO/outputs/sft/agpt2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900-hf" \
    "$SUBMIT_DIR/outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900-hf" \
    "$SUBMIT_DIR/outputs/sft/agpt2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900-hf"; do
    [[ -d "$c" ]] && { CKPT_900_HF="$c"; break; }
  done
fi

CKPT_93_HF="${CKPT_93_HF:-}"
if [[ -z "$CKPT_93_HF" ]]; then
  for c in \
    "$REPO/outputs/sft/agpt2b-gsm8k-r1cot-2n/checkpoint-93-hf" \
    "$REPO/outputs/sft/agpt2b-gsm8k-r1cot-2n/checkpoint-93" \
    "$REPO/outputs/checkpoints/agpt-2b-gsm8k-r1cot-2n/step-93" \
    "$SUBMIT_DIR/outputs/sft/agpt2b-gsm8k-r1cot-2n/checkpoint-93-hf" \
    "$SUBMIT_DIR/outputs/checkpoints/agpt-2b-gsm8k-r1cot-2n/step-93"; do
    [[ -d "$c" ]] && { CKPT_93_HF="$c"; break; }
  done
fi

LIMIT="${LIMIT:-200}"
DTYPE="${DTYPE:-float32}"
EVAL_SCRIPT="$REPO/torchtitan/experiments/ezpz/scripts/eval/eval_cot_gsm8k.py"
[[ -f "$EVAL_SCRIPT" ]] || EVAL_SCRIPT="$SUBMIT_DIR/torchtitan/experiments/ezpz/scripts/eval/eval_cot_gsm8k.py"
# Prefer activated venv python after ezpz_setup; fall back to repo venv.
PY="${PY:-python}"
if ! command -v "$PY" >/dev/null 2>&1; then
  PY="$REPO/.venv/bin/python"
fi

mkdir -p "$REPO/outputs/evals/cot" "$SUBMIT_DIR/logs" logs 2>/dev/null || true
OUT_900="${OUT_900:-$REPO/outputs/evals/cot/gsm8k-cot-ckpt-900-baseline.jsonl}"
OUT_93="${OUT_93:-$REPO/outputs/evals/cot/gsm8k-cot-ckpt-93-b2-baseline.jsonl}"
[[ -d "$(dirname "$OUT_900")" ]] || OUT_900="$SUBMIT_DIR/outputs/evals/cot/gsm8k-cot-ckpt-900-baseline.jsonl"
[[ -d "$(dirname "$OUT_93")" ]] || OUT_93="$SUBMIT_DIR/outputs/evals/cot/gsm8k-cot-ckpt-93-b2-baseline.jsonl"
mkdir -p "$(dirname "$OUT_900")" "$(dirname "$OUT_93")"

echo "BASELINE-RESCORE START $(date +%T) host=$(hostname) limit=$LIMIT dtype=$DTYPE"
echo "  ckpt-900-hf (general, IFEval 0.253) : ${CKPT_900_HF:-<not found>}"
echo "  ckpt-93 B2  (CoT 0.205 specialist)   : ${CKPT_93_HF:-<not found>}"
echo "  eval: $EVAL_SCRIPT"
echo "  python: $PY ($($PY --version 2>&1 || echo 'version probe failed'))"

# Gate: B2 expected ~0.205, ckpt-900 lower on CoT but higher on IFEval.
# Record both JSON summaries — do not assert strict equality (harness
# variance: B2 0.165 on preserved ep3-ckpt93-b2 vs old 0.205).

rc_overall=0

run_one() {
  local label="$1" ckpt="$2" out="$3"
  echo ""
  echo "=== COT-EVAL $label ckpt=$ckpt limit=$LIMIT dtype=$DTYPE ==="
  if [[ -z "$ckpt" || ! -d "$ckpt" ]]; then
    echo "[SKIP] $label: HF dir not found: ${ckpt:-<empty>} — set CKPT_900_HF / CKPT_93_HF to override"
    return 0
  fi
  if [[ ! -f "$EVAL_SCRIPT" ]]; then
    echo "[ERROR] $label: eval script missing: $EVAL_SCRIPT"
    return 2
  fi
  # CoT eval is single-tile fp32 (bf16 is gibberish on agpt-2b per skill).
  # Uses --dtype float32 --limit 200 --model <HF path> pattern from
  # consolidate_and_eval_cot.sh (task spec aliases --limit as --shots 200).
  "$PY" -u "$EVAL_SCRIPT" --model "$ckpt" --limit "$LIMIT" --dtype "$DTYPE" --out "$out"
  local rc=$?
  echo "COT-EVAL $label EXIT rc=$rc out=$out"
  if [[ -f "$out" ]]; then
    echo "  wrote per-example -> $out ($(wc -l < "$out" 2>/dev/null || echo '?') rows)"
  fi
  return $rc
}

run_one "ckpt-900-hf" "$CKPT_900_HF" "$OUT_900" || rc_overall=$?
run_one "ckpt-93-b2" "$CKPT_93_HF" "$OUT_93" || rc_overall=$?

echo ""
echo "BASELINE-RESCORE DONE $(date +%T) rc=$rc_overall"
echo "  ckpt-900-hf (IFEval 0.253 general) -> $OUT_900"
echo "  ckpt-93 B2  (CoT 0.205)            -> $OUT_93"
echo "  Gate: B2 ~0.205, ckpt-900 lower on CoT / higher on IFEval — informational, no strict assert."
exit $rc_overall
