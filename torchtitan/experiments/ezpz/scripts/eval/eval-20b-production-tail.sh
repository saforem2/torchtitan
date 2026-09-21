#!/bin/bash --login
# Submit one missing 20B production-tail eval with explicit, conventional shots.
# Usage: qsub -v EVAL_CHAIN=256n|512n-constlr eval-20b-production-tail.sh
#PBS -A AuroraGPT
#PBS -l walltime=08:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -N eval20b-tail
#PBS -j oe

set -o pipefail
W="${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"
cd "$W" || exit 1
export V2_REPO="$W"
export MODEL_FLAVOR=20b_real
export SHOTS_SPEC='0:hellaswag,arc_easy,winogrande,piqa,openbookqa,boolq;25:arc_challenge'

case "${EVAL_CHAIN:?set EVAL_CHAIN=256n or 512n-constlr}" in
  256n)
    export STEPS=16000
    export CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144
    export LABEL=256n
    export CKPT_REPO=/flare/AuroraGPT/foremans/runs/agpt-20b-n256/torchtitan-ezpz
    ;;
  512n-constlr)
    export STEPS=10900
    export CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288-constlr-from9000
    export LABEL=512n-constlr
    export CKPT_REPO="$W"
    ;;
  *)
    echo "unknown EVAL_CHAIN=${EVAL_CHAIN}" >&2
    exit 2
    ;;
esac

exec bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh
