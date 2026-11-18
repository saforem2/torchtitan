#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=00:40:00
#PBS -l select=1
#PBS -l filesystems=tegu:home
#PBS -q workq
#PBS -j oe
#
# GSM8K-CoT SFT probe harness (Phase 2D).
#
# Mirrors the execution contract of submit_cot_eval.sh (frameworks/2026.1.0
# + pti-gpu + PTI LD + ZE env) and provides an SFT-specific wrapper with
# documented 4-tuple gates, SE table, and qsub loop templates.
#
# Harness contract source: torchtitan/experiments/ezpz/scripts/eval/submit_cot_eval.sh
#   - PBS: select=1 walltime=00:40:00 filesystems=tegu:home q=workq A=datascience
#   - modules: frameworks/2026.1.0 + pti-gpu
#   - LD_LIBRARY_PATH: pti-gpu lib64 prepended
#   - ZE env: ZE_FLAT_DEVICE_HIERARCHY=FLAT, ONEAPI_DEVICE_SELECTOR=level_zero:gpu
#   - runtime: $PY $EVAL_SCRIPT --model "$CKPT" --limit 200 --dtype float32 --out "$OUT"
#             where PY=/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/bin/python
#                   EVAL_SCRIPT=torchtitan/experiments/ezpz/scripts/eval/eval_cot_gsm8k.py
#   - verify contract: diff -u <(grep -E '^(#PBS|module load|export (LD_|ZE_|ONEAPI))' submit_cot_eval.sh) \
#                              <(grep -E '^(#PBS|module load|export (LD_|ZE_|ONEAPI))' eval_cot_sft_probe.sh)
#
# Usage:
#   qsub -v CKPT=<hf-dir> torchtitan/experiments/ezpz/scripts/eval/eval_cot_sft_probe.sh
#   qsub -v CKPT=<hf-dir>,OUT=/custom/out.jsonl torchtitan/experiments/ezpz/scripts/eval/eval_cot_sft_probe.sh
#   CKPT=/path/to/hf OUT=/tmp/x.jsonl bash torchtitan/experiments/ezpz/scripts/eval/eval_cot_sft_probe.sh  # interactive
#
# ============================================================================
# 4-tuple measured per run (parsed from $OUT JSONL tail / eval summary)
# ============================================================================
#   cot_accuracy   — GSM8K-CoT accuracy (extracted \boxed{} vs gold).
#     B2 baseline: 0.205 (41/200), 95% CI [0.149, 0.260] (binomial, N=200).
#     Gate: informational; SFT expected to lift, not regress below baseline lower bound.
#
#   format_hit_rate — fraction emitting valid contracted format
#     (<think> ... </think> then <answer>\boxed{...}</answer>).
#     B2: 0.985. Gate: >= 0.90 post-SFT. Pre-SFT 0/200 is EXPECTED (no format
#     contract in base/CPT checkpoints) — do NOT interpret as failure.
#
#   mean_gen_len   — mean generation length in tokens (or chars, per eval script).
#     B2: 282. Gate: <= 550 (catches degenerate runaway / unclosed reasoning).
#
#   n_unclosed     — count of generations with unclosed <think> / malformed tail.
#     B2: ~0. Gate: <= 10.
#
# Gate rule: 0/200 fmt == 0 pre-SFT is expected (base/CPT never trained to emit
# <think>/<answer>\boxed{}). Only flag format failures post-SFT (after
# sft_distill_cot / B2v2-style runs).
#
# ============================================================================
# N=200 power / SE table (binomial, normal approx; exact Clopper-Pearson similar)
# ============================================================================
#   p     SE=sqrt(p(1-p)/N)   95% half-width (1.96*SE)   detectable lift vs p
#   0.20  0.028               +/-0.055                  +0.06 (=2.1 sigma) detectable
#   0.25  0.031               +/-0.060                  +0.06 = 1.9 sigma (borderline)
#   0.10  0.021               +/-0.042                  +0.06 = 2.8 sigma
#   0.30  0.032               +/-0.063                  +0.06 = 1.9 sigma
#
# At B2 p=0.20, SE=0.028, half-width +/-0.055. A +0.06 lift (e.g. 0.20->0.26)
# is ~2 sigma and detectable at N=200 (p~0.04 one-sided). Smaller lifts need
# larger N (N=500 -> SE 0.018, half-width 0.035). LIMIT is fixed at 200 for
# walltime=00:40:00; do not raise without bumping walltime.
#
# ============================================================================
# qsub loop over SFT HF dirs (Tegu)
# ============================================================================
#   for ckpt in \
#     /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/sft/agpt2b-sft-checkpoint-900-hf \
#     /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/sft/agpt2b-sft-checkpoint-93-hf \
#     /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/sft/agpt2b-b2v2-distill-cot-2n-ep3/final \
#     /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/sft/agpt2b-b2v2-distill-cot-2n-ep6/final; do
#     qsub -v CKPT="$ckpt" torchtitan/experiments/ezpz/scripts/eval/eval_cot_sft_probe.sh
#   done
#
# ============================================================================
# CPT ladder post-SFT probe template
# ============================================================================
# After CPT-FWE P0-P3 HF exports exist, SFT each CPT base then probe:
#
#   # 1. SFT on a CPT base (via agpt2b_b2v2_distill_cot_2n.sh pattern):
#   #    CKPT_CPT=/lus/tegu/.../outputs/evals/cpt-fwe/step-<N>/hf  # HF converted via convert_to_hf
#   #    qsub -v BASE_HF="$CKPT_CPT",OUT_SFT=/lus/tegu/.../outputs/sft/cpt-step-<N>-sft-hf \
#   #      torchtitan/experiments/ezpz/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh
#
#   # 2. Probe the resulting SFT HF:
#   #    qsub -v CKPT=/lus/tegu/.../outputs/sft/cpt-step-<N>-sft-hf \
#   #      torchtitan/experiments/ezpz/scripts/eval/eval_cot_sft_probe.sh
#
#   # Batch over ladder:
#   #   for n in 11600 13000 17000 34800 67400 72800 73000 74000 74200 74400 ... 190734; do
#   #     base=/lus/tegu/.../outputs/evals/cpt-fwe/step-${n}/hf
#   #     [[ -d $base ]] || continue
#   #     # SFT step (agpt2b_b2v2_distill_cot_2n.sh) then probe (this script) as above
#   #   done
#

REPO=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan
cd "${PBS_O_WORKDIR:-$REPO}" || exit 9

source /usr/share/lmod/lmod/init/bash 2>/dev/null
module load frameworks/2026.1.0 pti-gpu >/dev/null 2>&1
export LD_LIBRARY_PATH=/opt/aurora/26.181.0/spack/unified/1.1.1/install/linux-x86_64/pti-gpu-0.17.0-62kxpbr/lib64:$LD_LIBRARY_PATH
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
export ZE_AFFINITY_MASK=0
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PATH="/opt/pbs/bin:$PATH"

CKPT="${CKPT:-$1}"
# .../step-<N>/hf -> step-<N>; any other HF dir (e.g. SFT checkpoint-900-hf)
# names itself, so two SFT checkpoints under outputs/sft/ get distinct outputs.
if [[ "$(basename "$CKPT")" == hf ]]; then
  STEP_TAG="$(basename "$(dirname "$CKPT")")"
else
  STEP_TAG="$(basename "$CKPT")"
fi
[[ -n "$STEP_TAG" ]] || STEP_TAG=baseline
OUT="${OUT:-$REPO/outputs/evals/cot/cpt-fwe-${STEP_TAG}-gsm8k-cot.jsonl}"
LIMIT=200
DTYPE="${DTYPE:-float32}"
PY="${PY:-/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/bin/python}"
EVAL_SCRIPT="$REPO/torchtitan/experiments/ezpz/scripts/eval/eval_cot_gsm8k.py"
[[ -f "$EVAL_SCRIPT" ]] || EVAL_SCRIPT="$HOME/torchtitan/experiments/ezpz/scripts/eval/eval_cot_gsm8k.py"
mkdir -p "$(dirname "$OUT")"

if [[ -z "$CKPT" ]]; then
  echo "ERROR: CKPT (HF dir) not set. Use: qsub -v CKPT=<hf-dir> $0" >&2
  exit 2
fi

echo "COT-SFT-PROBE START $(date +%T) host=$(hostname) ckpt=$CKPT limit=$LIMIT dtype=$DTYPE out=$OUT"
"$PY" -u "$EVAL_SCRIPT" --model "$CKPT" --limit "$LIMIT" --dtype "$DTYPE" --out "$OUT"
rc=$?
echo "COT-SFT-PROBE EXIT rc=$rc $(date +%T) out=$OUT"
exit $rc
