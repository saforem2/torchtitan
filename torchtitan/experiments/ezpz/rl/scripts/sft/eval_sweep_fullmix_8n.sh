#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=02:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe
#
# Base-LM lm-eval SWEEP over checkpoints of the 8N full-mix SFT
# (gs138650 x tulu_math_uc_mix_full), vs the shared gs138650 baseline.
#
# For each checkpoint-N in SWEEP_CKPTS:
#   1. consolidate FSDP shards -> checkpoint-N-hf (if not already done), via
#      `accelerate merge-weights` (same as consolidate_sft_ckpt.sh).
#   2. run the 7 multiple-choice base-LM tasks (0-shot), one per XPU tile,
#      reusing the exact env + transformers dtype-shim from
#      eval_sft_vs_baseline_parallel.sh.
# The baseline (gs138650) is evaluated ONCE (its scores don't change).
#
# This builds an eval-metric-vs-step trajectory to see whether the ~54B-token
# big-mix SFT improves base-LM benchmarks as it trains.
#
# Config (env overrides):
#   SWEEP_CKPTS  space-separated checkpoint step numbers (default "300 600 900")
#   EVAL_BASELINE  1 to also eval the gs138650 baseline (default 1)
#
# Runs on ONE node (12 tiles): 7 tasks fit on 7 tiles; models run
# back-to-back on the same tiles (fair per-tile comparison -- cross-node
# variance is too large, learned in eval_sft_vs_baseline_parallel.sh).
#
# Output: outputs/evals/fullmix-8n-sweep-<jobid>/<label>/<task>/
set -o pipefail

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

CKPT_ROOT="outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144"
BASELINE_MODEL="${HOME}/global_step138650"
SWEEP_CKPTS="${SWEEP_CKPTS:-300 600 900}"
EVAL_BASELINE="${EVAL_BASELINE:-1}"
TASKS=(hellaswag arc_easy arc_challenge winogrande piqa openbookqa boolq)

OUT_BASE="outputs/evals/fullmix-8n-sweep-${PBS_JOBID%%.*}"
LOG_DIR="logs/eval-fullmix-8n-sweep-${PBS_JOBID%%.*}"
mkdir -p "${OUT_BASE}" "${LOG_DIR}"
RUNLOG="${LOG_DIR}/run.log"
echo "=== full-mix 8N SFT eval sweep: ckpts [${SWEEP_CKPTS}] vs gs138650 baseline ===" | tee "${RUNLOG}"

HOST=$(hostname)

# --- 1. consolidate each sweep checkpoint to -hf (module needs oneapi only) ---
module load oneapi/release/2025.3.1 hdf5 pti-gpu 2>&1 | tail -1
source .venv/bin/activate 2>/dev/null || true
for n in ${SWEEP_CKPTS}; do
    ck="${CKPT_ROOT}/checkpoint-${n}"
    hf="${ck}-hf"
    if [[ -f "${hf}/model.safetensors" || -f "${hf}/model-00001-of-00001.safetensors" ]]; then
        echo "[consolidate] ${hf} exists, skip" | tee -a "${RUNLOG}"; continue
    fi
    if [[ ! -d "${ck}/pytorch_model_fsdp_0" ]]; then
        echo "[consolidate] SKIP ckpt-${n}: no pytorch_model_fsdp_0" | tee -a "${RUNLOG}"; continue
    fi
    echo "[consolidate] ${ck} -> ${hf}" | tee -a "${RUNLOG}"
    accelerate merge-weights "${ck}/pytorch_model_fsdp_0" "${hf}" 2>&1 | tail -3 | tee -a "${RUNLOG}"
    # copy config + tokenizer from the base so from_pretrained works
    for f in config.json tokenizer.json tokenizer.model tokenizer_config.json special_tokens_map.json; do
        [[ -f "${BASELINE_MODEL}/${f}" && ! -f "${hf}/${f}" ]] && cp "${BASELINE_MODEL}/${f}" "${hf}/"
    done
done

# --- 2. per-tile lm-eval launcher (reuses the proven env + dtype shim) ---
launch_one() {
    local model_path="$1" label="$2" task="$3" tile="$4"
    local out_dir="${OUT_BASE}/${label}/${task}"
    mkdir -p "${out_dir}"
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${HOST}" bash -s <<EOF >"${out_dir}/eval.log" 2>&1 &
set -o pipefail
cd ${SUBMIT_DIR}
source /etc/profile.d/lmod.sh 2>/dev/null || source /etc/profile 2>/dev/null
module load oneapi/release/2025.3.1 hdf5 pti-gpu frameworks/2025.3.1 2>&1 | tail -1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR="level_zero:${tile}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
source venvs/sunspot/torchtitan-aurora_frameworks-2025.3.1/bin/activate
echo "=== ${label}:${task} tile=${tile} on \$(hostname) ==="
python3 - <<'PYEOF'
import transformers
_real = transformers.AutoModelForCausalLM.from_pretrained
def _shim(*a, **k):
    if 'dtype' in k and 'torch_dtype' not in k:
        k['torch_dtype'] = k.pop('dtype')
    return _real(*a, **k)
transformers.AutoModelForCausalLM.from_pretrained = _shim
import sys
from lm_eval.__main__ import cli_evaluate
sys.argv = ['lm_eval','--model','hf','--model_args','pretrained=${model_path}',
    '--tasks','${task}','--batch_size','4','--num_fewshot','0',
    '--device','xpu:0','--output_path','${out_dir}/']
cli_evaluate()
PYEOF
echo "=== ${label}:${task} done ==="
EOF
}

run_model() {
    local model_path="$1" label="$2"
    echo ">>> evaluating ${label} (${model_path})" | tee -a "${RUNLOG}"
    local i=0
    for task in "${TASKS[@]}"; do
        launch_one "${model_path}" "${label}" "${task}" "${i}"
        i=$((i + 1))
    done
    wait
    echo ">>> ${label} done" | tee -a "${RUNLOG}"
}

# --- 3. baseline once, then each sweep checkpoint ---
[[ "${EVAL_BASELINE}" == "1" ]] && run_model "${BASELINE_MODEL}" "baseline-gs138650"
for n in ${SWEEP_CKPTS}; do
    hf="${CKPT_ROOT}/checkpoint-${n}-hf"
    [[ -d "${hf}" ]] && run_model "${hf}" "sft-step${n}" || echo "SKIP sft-step${n}: no ${hf}" | tee -a "${RUNLOG}"
done

echo "=== DONE: results under ${OUT_BASE}/ ; log ${RUNLOG} ===" | tee -a "${RUNLOG}"
