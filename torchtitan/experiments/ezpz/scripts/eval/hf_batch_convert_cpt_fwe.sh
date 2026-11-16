#!/bin/bash --login
#PBS -N hf-batch-cpt-fwe
#PBS -q workq
#PBS -l walltime=02:00:00
#PBS -l select=1
#PBS -l filesystems=tegu:home
#PBS -A datascience
#PBS -j oe
#
# Frontier: 136800/190700 (12483869 R 4N). HF 67400 READY, 72800 blocked
# (no tokenizer), 13000 empty, 74400-136800 DCP exists but HF missing.
#
# Fixups:
#   72800 - copy config.json + tokenizer.* from 67400 HF (already converted)
#   13000 - reconvert DCP -> HF (empty HF dir, included in loop)
# Batch:  74400 -> 136800 every 200 via seq
#
# Staging lock pattern from hf_export.py (staged_export) is already handled
# inside convert_to_hf.py -- no manual .lock/.staging needed here.
#
# Usage:
#   qsub torchtitan/experiments/ezpz/scripts/eval/hf_batch_convert_cpt_fwe.sh
#

set -euo pipefail

SUBMIT_DIR=${PBS_O_WORKDIR:-$(pwd)}
cd $SUBMIT_DIR

REPO=/home/foremans/torchtitan
# Live DCP stays on /home (15T chain at outputs/outputs/checkpoints -- moving mid-train breaks resume)
DCP_ROOT=$REPO/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64
# New HF outputs prefer Tegu project space per user request; PBS filesystems=tegu:home covers both.
# Fall back to HOME if Tegu project path not present (local dev).
TEGU_ROOT=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan
if [[ -d "$TEGU_ROOT" ]]; then HF_ROOT=$TEGU_ROOT/outputs/evals/cpt-fwe; else HF_ROOT=$REPO/outputs/evals/cpt-fwe; fi
HF_CONFIG=$REPO/torchtitan/experiments/ezpz/eval/configs/agpt_2b_mds_config.json
TOK_SRC=$REPO/assets/hf/gemma-7b

echo "HF-BATCH-CPT-FWE START $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) repo=$REPO"
echo "  DCP_ROOT=$DCP_ROOT"
echo "  HF_ROOT=$HF_ROOT"
echo "  flavor=2b-mds dtype=bfloat16"

# ---------------------------------------------------------------------------
# Fixups first inside the job
# ---------------------------------------------------------------------------
# 72800 missing config.json+tokenizer.* -> copy from 67400 HF + TOK_SRC
if [[ -d "$HF_ROOT/step-72800/hf" ]]; then
  cp $HF_ROOT/step-67400/hf/config.json $HF_ROOT/step-72800/hf/ && cp $TOK_SRC/tokenizer.* $HF_ROOT/step-72800/hf/ || echo "[WARN] 72800 fixup copy failed (check src exists)"
fi

# 13000 empty -> reconvert (included in loop below; no separate handling needed)

# ---------------------------------------------------------------------------
# Batch convert loop (spec one-liner, expanded for readability but kept verbatim)
# ---------------------------------------------------------------------------
for s in 13000 $(seq 74400 200 136800); do DCP=$DCP_ROOT/step-$s HF=$HF_ROOT/step-$s/hf; [[ -f $HF/ezpz_export.json ]] && continue; $REPO/.venv/bin/python -m torchtitan.experiments.ezpz.eval.convert_to_hf "$DCP" "$HF" --model_name experiments.ezpz.agpt --model_flavor 2b-mds --export_dtype bfloat16 || exit 1; cp "$HF_CONFIG" "$HF/config.json"; cp "$TOK_SRC"/tokenizer.* "$HF/"; ls -lh "$HF"/model-*.safetensors; done

echo CONVERTED $(ls $HF_ROOT/step-*/hf/ezpz_export.json 2>/dev/tty | wc -l) HF
