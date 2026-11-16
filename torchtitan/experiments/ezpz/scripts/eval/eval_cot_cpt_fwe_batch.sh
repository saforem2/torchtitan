#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=00:40:00
#PBS -l select=1
#PBS -l filesystems=tegu:home
#PBS -q workq
#PBS -j oe
#
# Batch GSM8K-CoT eval for CPT-FWE frontier HFs (Sunspot).
# Iterates existing HF dirs in priority order (74000 primary) and
# delegates each to the canonical single-tile vLLM fp32 harness.
# The harness itself module-loads frameworks/2026.1.0 + pti-gpu;
# this batch wrapper stays machine-agnostic via ezpz_setup.
# For distributed evals, wrap the delegate with `ezpz launch`.
#
#   qsub torchtitan/experiments/ezpz/scripts/eval/eval_cot_cpt_fwe_batch.sh

SUBMIT_DIR=${PBS_O_WORKDIR:-$(pwd)}
cd "$SUBMIT_DIR" || exit 1

# shellcheck disable=SC1090,SC1091
source <(curl -fsSL https://ezpz.cool/utils.sh) && ezpz_setup /home/foremans/torchtitan/.venv

# Ensure log dir exists
mkdir -p logs

# CoT eval is single-tile fp32 (bf16 is gibberish on agpt-2b per skill).
# submit_cot_eval.sh reads CKPT/LIMIT/DTYPE as env vars, not argv.
for step in 74000 74200 67400 73000 72800; do
  CKPT=/home/foremans/torchtitan/outputs/evals/cpt-fwe/step-$step/hf
  [[ -d $CKPT ]] || continue
  echo "=== COT batch step=$step ckpt=$CKPT ==="
  CKPT=$CKPT LIMIT=200 DTYPE=float32 bash torchtitan/experiments/ezpz/scripts/eval/submit_cot_eval.sh 2>&1 | tee -a logs/cot-batch-$step.log
done

echo "COT-BATCH DONE $(date +%T)"
