#!/bin/bash
#PBS -A datascience
#PBS -l walltime=00:50:00
#PBS -l select=1
#PBS -l filesystems=tegu:home
#PBS -q workq
#PBS -j oe
#
# Merge a LoRA-DCP GRPO checkpoint into its base, then run the 200-problem GSM8K
# CoT eval on the merged HF model. One job: merge -> eval -> print metrics.
#
#   qsub -v DCP=<...>/checkpoint/step-100,BASE=<base-hf>,OUT=<merged-hf-dir>,LIMIT=200 \
#       torchtitan/experiments/ezpz/scripts/eval/merge_and_eval_cot_lora.sh
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
export ZE_AFFINITY_MASK=0
export ZES_ENABLE_SYSMAN=1
export TORCHINDUCTOR_MAX_AUTOTUNE=0 VLLM_ENABLE_V1_MULTIPROCESSING=1
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PATH="/opt/pbs/bin:$PATH"

PY_MERGE=$REPO/venvs/rl-grpo-lora/bin/python   # has editable torchtitan (2b-rl registry)
PY_EVAL=$REPO/venvs/rl-vllm/bin/python          # proven vLLM eval path
DCP="${DCP:?set DCP=<...>/checkpoint/step-N}"
BASE="${BASE:-$REPO/outputs/sft/agpt2b-gsm8k-r1cot-2n/checkpoint-93-hf}"
OUT="${OUT:-$REPO/outputs/evals/cot/$(basename $(dirname $(dirname "$DCP")))_$(basename "$DCP")_merged_hf}"
LIMIT="${LIMIT:-200}"

echo "MERGE+EVAL START $(date +%T) dcp=$DCP base=$BASE out=$OUT"
if [ ! -f "$OUT/model.safetensors" ] && [ ! -f "$OUT/model-00001-of-00001.safetensors" ]; then
    PYTHONPATH=$REPO "$PY_MERGE" torchtitan/experiments/ezpz/scripts/eval/merge_lora_dcp_to_hf.py \
        --dcp "$DCP" --base-hf "$BASE" --out "$OUT" \
        --lora-rank 8 --lora-alpha 16 --model-flavor 2b-rl 2>&1 | tail -8
else
    echo "[merge] $OUT exists, skip"
fi
"$PY_EVAL" torchtitan/experiments/ezpz/scripts/eval/fix_ckpt_eos.py "$OUT" 2>&1 | tail -1
echo "COT-EVAL $(date +%T) limit=$LIMIT"
"$PY_EVAL" torchtitan/experiments/ezpz/scripts/eval/eval_cot_gsm8k.py \
    --model "$OUT" --limit "$LIMIT" --dtype float32 --out /tmp/cot_eval_$(basename "$DCP").jsonl > /tmp/eval_out_$(basename "$DCP").txt 2>&1
EVAL_RC=$?
tail -20 /tmp/eval_out_$(basename "$DCP").txt
echo "MERGE+EVAL EXIT rc=$EVAL_RC $(date +%T)"
