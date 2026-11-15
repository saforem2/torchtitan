#!/bin/bash --login
#PBS -N mmlu-cpt67k
#PBS -A datascience
#PBS -q capacity
#PBS -l walltime=02:00:00
#PBS -l select=1
#PBS -l filesystems=home:flare
#PBS -j oe
#
# Aurora MMLU re-score on CPT-FWE 67.4k HF (MCF gate — constant-D 0.2689 band).
#   qsub torchtitan/experiments/ezpz/scripts/eval/eval_mmlu_aurora_cpt67k.sh
#
# Uses convert_and_eval.sh — skips convert if HF exists, runs lm-eval via vLLM.
# SHOTS_SPEC enforces mmlu 5-shot + arc_challenge 25-shot (flat NUM_FEWSHOT=0 would 0-shot mmlu).

set -euo pipefail
REPO=/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan
[[ -d "$REPO" ]] || REPO=/home/foremans/torchtitan
cd "${PBS_O_WORKDIR:-$REPO}" || exit 9

source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
module load oneapi/release/2025.3.1 hdf5 pti-gpu frameworks/2025.3.1 2>/dev/null || true
export LD_LIBRARY_PATH="/opt/aurora/26.181.0/spack/unified/1.1.1/install/linux-x86_64/pti-gpu-0.17.0-62kxpbr/lib64:${LD_LIBRARY_PATH:-}"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
export PATH="/opt/pbs/bin:${PATH}"

STEP=67400
CKPT_NAME=agpt-2b-cpt-fineweb-edu-n4-gbs64
TASKS="hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq,mmlu"
SHOTS_SPEC="5:mmlu;25:arc_challenge"

echo "MMLU-EVAL START $(date) step=$STEP ckpt=$CKPT_NAME tasks=$TASKS shots=$SHOTS_SPEC"
TASKS="$TASKS" SHOTS_SPEC="$SHOTS_SPEC" bash torchtitan/experiments/ezpz/scripts/eval/convert_and_eval.sh \
  --model 2b --step "$STEP" --ckpt-name "$CKPT_NAME"
echo "MMLU-EVAL EXIT rc=$? $(date)"
