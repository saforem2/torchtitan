#!/bin/bash --login
#PBS -N cot-cpt67k
#PBS -A datascience
#PBS -q workq
#PBS -l walltime=00:40:00
#PBS -l select=1
#PBS -l filesystems=tegu:home
#PBS -j oe
#
# Sunspot GSM8K-CoT smoke on CPT 67.4k HF — vLLM fp32, 200 problems.
#   qsub torchtitan/experiments/ezpz/scripts/eval/eval_cot_sunspot_cpt67k.sh
#   qsub -v CKPT=/path/to/other/hf,LIMIT=200 torchtitan/experiments/ezpz/scripts/eval/eval_cot_sunspot_cpt67k.sh

REPO=/home/foremans/torchtitan
[[ -d "$REPO/outputs" ]] || REPO=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan
cd "${PBS_O_WORKDIR:-$REPO}" || exit 9

source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
module load oneapi/release/2025.3.1 hdf5 pti-gpu frameworks/2026.1.0 2>/dev/null || true
export LD_LIBRARY_PATH="/opt/aurora/26.181.0/spack/unified/1.1.1/install/linux-x86_64/pti-gpu-0.17.0-62kxpbr/lib64:${LD_LIBRARY_PATH:-}"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
export ZE_AFFINITY_MASK=0
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
export PATH="/opt/pbs/bin:${PATH}"
CKPT="${CKPT:-outputs/evals/cpt-fwe/step-67400/hf}"
LIMIT="${LIMIT:-200}"
DTYPE="${DTYPE:-float32}"
# Delegate to the canonical CoT eval — keeps vLLM fp32 path in one place.
CKPT="$CKPT" LIMIT="$LIMIT" DTYPE="$DTYPE" bash torchtitan/experiments/ezpz/scripts/eval/submit_cot_eval.sh
