#!/bin/bash
#PBS -A datascience
#PBS -l walltime=00:40:00
#PBS -l select=1
#PBS -l filesystems=tegu:home
#PBS -q workq
#PBS -j oe
#
# GSM8K chain-of-thought eval for agpt-2b — single tile, vLLM fp32.
#   qsub -v CKPT=<hf-dir>,LIMIT=200 torchtitan/experiments/ezpz/scripts/eval/submit_cot_eval.sh

REPO=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan
cd "${PBS_O_WORKDIR:-$REPO}" || exit 9

source /usr/share/lmod/lmod/init/bash 2>/dev/null
module load frameworks/2026.1.0 oneapi/release/2025.3.1 pti-gpu >/dev/null 2>&1
export LD_LIBRARY_PATH="/opt/aurora/26.181.0/spack/unified/1.1.1/install/linux-x86_64/pti-gpu-0.17.0-62kxpbr/lib64:/opt/aurora/26.181.0/oneapi/mkl/latest/lib:${LD_LIBRARY_PATH}"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=0
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PATH="/opt/pbs/bin:$PATH"

CKPT="${CKPT:-$HOME/rl-repro/run/agpt2b-ckpt900}"
LIMIT="${LIMIT:-200}"
DTYPE="${DTYPE:-float32}"
OUT="${OUT:-$REPO/outputs/evals/cot/gsm8k-cot-baseline.jsonl}"
PY="${PY:-/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/bin/python}"
EVAL_SCRIPT="$REPO/torchtitan/experiments/ezpz/scripts/eval/eval_cot_gsm8k.py"
[[ -f "$EVAL_SCRIPT" ]] || EVAL_SCRIPT="$HOME/torchtitan/experiments/ezpz/scripts/eval/eval_cot_gsm8k.py"
mkdir -p "$(dirname "$OUT")"

echo "COT-EVAL START $(date +%T) host=$(hostname) ckpt=$CKPT limit=$LIMIT dtype=$DTYPE"
"$PY" -u "$EVAL_SCRIPT" --model "$CKPT" --limit "$LIMIT" --dtype "$DTYPE" --out "$OUT"
echo "COT-EVAL EXIT rc=$? $(date +%T)"
