#!/bin/bash
#PBS -A datascience
#PBS -l walltime=00:40:00
#PBS -l select=1
#PBS -l filesystems=tegu:home
#PBS -q workq
#PBS -j oe
#
# Consolidate an FSDP-sharded CoT-SFT checkpoint to HF format, then run the
# Stage 0 CoT eval on it (and, for reference, re-confirm the baseline once).
#   qsub -v CKPT=outputs/sft/agpt2b-gsm8k-r1cot-8n/checkpoint-16 consolidate_and_eval_cot.sh
# no set -euo (module load returns nonzero under Lmod)
set -o pipefail

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

CKPT="${CKPT:-$REPO/outputs/sft/agpt2b-gsm8k-r1cot-8n/checkpoint-16}"
# BASE supplies config.json; TOK_SRC supplies the tokenizer WITH the gemma
# chat_template (the raw checkpoint-900-hf has no template, which breaks
# apply_chat_template at eval time -- the staged dir has it injected).
BASE="${BASE:-$REPO/outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900-hf}"
TOK_SRC="${TOK_SRC:-$HOME/rl-repro/run/agpt2b-ckpt900}"
HF="${CKPT}-hf"
LIMIT="${LIMIT:-200}"
VENV_MAIN=$REPO/.venv/bin            # accelerate lives here
PY=$REPO/venvs/rl-vllm/bin/python    # vLLM eval

echo "CONSOLIDATE+EVAL START $(date +%T) ckpt=$CKPT"

# --- 1. consolidate FSDP shards -> HF ---
if [[ -f "${HF}/model.safetensors" || -f "${HF}/model-00001-of-00001.safetensors" ]]; then
    echo "[consolidate] ${HF} exists, skip"
elif [[ -d "${CKPT}/pytorch_model_fsdp_0" ]]; then
    echo "[consolidate] ${CKPT} -> ${HF}"
    source "$REPO/.venv/bin/activate"
    accelerate merge-weights "${CKPT}/pytorch_model_fsdp_0" "${HF}" 2>&1 | tail -3
    # config from BASE; tokenizer (with chat_template) from TOK_SRC.
    [[ -f "${BASE}/config.json" && ! -f "${HF}/config.json" ]] && cp "${BASE}/config.json" "${HF}/"
    for f in tokenizer.json tokenizer.model tokenizer_config.json special_tokens_map.json; do
        [[ -f "${TOK_SRC}/${f}" ]] && cp "${TOK_SRC}/${f}" "${HF}/"
    done
    # Fix the consolidated eos_token_id (stock Llama eos=2 is wrong for gemma and
    # not what the model emits; it stops on <end_of_turn>=107). Without this
    # GRPO/vLLM never terminate. See fix_ckpt_eos.py.
    "${VENV_MAIN}/python" torchtitan/experiments/ezpz/scripts/eval/fix_ckpt_eos.py "${HF}"
    deactivate 2>/dev/null || true
else
    echo "FATAL: no pytorch_model_fsdp_0 in ${CKPT}"; exit 2
fi
ls -la "${HF}" | grep -E "safetensors|config.json|tokenizer_config" | head

# --- 2. CoT eval on the consolidated checkpoint ---
OUT="$REPO/outputs/evals/cot/gsm8k-cot-$(basename "$CKPT").jsonl"
mkdir -p "$(dirname "$OUT")"
echo "COT-EVAL (sft ckpt) $(date +%T)"
"$PY" -u torchtitan/experiments/ezpz/scripts/eval/eval_cot_gsm8k.py \
    --model "${HF}" --limit "${LIMIT}" --dtype float32 --out "$OUT"
echo "CONSOLIDATE+EVAL EXIT rc=$? $(date +%T)"
