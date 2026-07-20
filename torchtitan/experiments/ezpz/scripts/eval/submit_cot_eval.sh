#!/bin/bash
#PBS -A datascience
#PBS -l walltime=00:40:00
#PBS -l select=1
#PBS -l filesystems=tegu:home
#PBS -q workq
#PBS -j oe
#
# Stage 0 baseline: generation-based GSM8K chain-of-thought eval for agpt-2b.
# Single tile, vLLM (fp32 -- agpt-2b needs it), rl-vllm venv.
#
#   qsub -v CKPT=<hf-dir>,LIMIT=200 torchtitan/experiments/ezpz/scripts/eval/submit_cot_eval.sh
#
# no set -euo: module load returns nonzero under Lmod (CLAUDE.md rule)

REPO=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan
cd "${PBS_O_WORKDIR:-$REPO}" || exit 9

source /usr/share/lmod/lmod/init/bash 2>/dev/null
module load oneapi/release/2025.3.1 hdf5 pti-gpu >/dev/null 2>&1
export CCL_ROOT=/opt/aurora/26.26.0/oneapi/ccl/latest
export LD_LIBRARY_PATH=/opt/cray/libfabric/2.3.1/lib64:/opt/aurora/26.26.0/oneapi/2025.3/opt/mpi/libfabric/lib:$CCL_ROOT/lib:/opt/aurora/26.26.0/oneapi/2025.3/lib:$LD_LIBRARY_PATH
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export ZE_AFFINITY_MASK=0            # single tile is plenty for the eval
export ZES_ENABLE_SYSMAN=1
export TORCHINDUCTOR_MAX_AUTOTUNE=0 VLLM_ENABLE_V1_MULTIPROCESSING=1
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1   # gsm8k is cached
export PATH="/opt/pbs/bin:$PATH"

CKPT="${CKPT:-$HOME/rl-repro/run/agpt2b-ckpt900}"   # staged ckpt-900 w/ gemma template
LIMIT="${LIMIT:-200}"
DTYPE="${DTYPE:-float32}"
OUT="${OUT:-$REPO/outputs/evals/cot/gsm8k-cot-baseline.jsonl}"
PY=$REPO/venvs/rl-vllm/bin/python
mkdir -p "$(dirname "$OUT")"

echo "COT-EVAL START $(date +%T) host=$(hostname) ckpt=$CKPT limit=$LIMIT dtype=$DTYPE"
"$PY" -u torchtitan/experiments/ezpz/scripts/eval/eval_cot_gsm8k.py \
    --model "$CKPT" \
    --limit "$LIMIT" \
    --dtype "$DTYPE" \
    --out "$OUT"
echo "COT-EVAL EXIT rc=$? $(date +%T)"
