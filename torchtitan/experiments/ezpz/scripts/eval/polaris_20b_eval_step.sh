#!/bin/bash --login
#PBS -N p20b-eval-step
#PBS -A AuroraGPT
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -j oe
#
# Convert + lm-eval a SINGLE Polaris 20B (dolma) checkpoint. One debug job
# per checkpoint -- debug schedules in minutes (vs preemptable's 135-deep
# queue for 10 nodes). convert+eval for one step fits in ~40min < 1h.
#
# Submit per step:
#   qsub -v STEP=2400 torchtitan/experiments/ezpz/scripts/eval/polaris_20b_eval_step.sh
#
# Polaris facts encoded (see polaris_20b_eval_sweep.sh header for the full
# rationale): convert on the compute node under the TRAINING venv; eval under
# the CONDA-BASE eval venv (venvs/lm-eval) so vllm<->torch stay ABI-matched;
# TP=4 (4 GPU/node); numpy stays conda's 2.2.6. Idempotent: skips if results
# already exist.

set -x
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export no_proxy=localhost,127.0.0.1,*.alcf.anl.gov,*.anl.gov
# Do NOT set HF_HUB_OFFLINE=1: lm-eval fetches task DATASETS (ai2_arc, piqa,
# etc.) from the Hub, and only some are cached. The model is loaded from a
# LOCAL path so it needs no Hub access. HF_HUB_ENABLE_HF_TRANSFER=0 avoids the
# hf_transfer download bug (per the Aurora eval guidance). Proxy (set above)
# gives the compute node Hub access.
export HF_HUB_ENABLE_HF_TRANSFER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# 20B bf16 (~40GB) across 4x A100-40GB is tight; give the allocator room to
# avoid fragmentation OOM (seen on some nodes at step-1000). vllm args below
# also drop max_model_len to 2048 (ample for these MC tasks) to shrink KV cache.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/eagle/AuroraGPT/foremans/projects/saforem2/torchtitan
cd "$REPO" || exit 1
export PYTHONPATH="$REPO"

STEP="${STEP:?must pass -v STEP=N}"
TASKS="${TASKS:-hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq}"
TP="${TP:-4}"
# This 20B chain trained with the _real (cos_sin RoPE) flavor. The converter
# MUST build a matching cos_sin model or it applies the complex-only Q/K permute
# and exports gibberish (see agpt/state_dict_adapter.py). Use the _real flavor.
MODEL_FLAVOR="${MODEL_FLAVOR:-20b_real}"

CKPT_NAME=agpt-20b-sophiag-dolma-n128-gbs1024
DCP="$REPO/outputs/checkpoints/$CKPT_NAME/step-${STEP}"
HF="$REPO/outputs/evals/agpt-20b-dolma-n128/step-${STEP}/hf"
RES="$REPO/outputs/evals/agpt-20b-dolma-n128/step-${STEP}/results"
TOK="$REPO/assets/hf/gemma-7b"
HF_CONFIG="$REPO/torchtitan/experiments/ezpz/eval/configs/agpt_20b_config.json"

if [ ! -f "$DCP/.metadata" ]; then
    echo "ABORT step-${STEP}: no complete DCP checkpoint (.metadata missing)"
    exit 1
fi
if compgen -G "$RES/**/results_*.json" > /dev/null 2>&1; then
    echo "DONE step-${STEP}: results already exist, nothing to do"
    exit 0
fi

# ---- Convert (training venv, compute-node RAM) ----
if [ ! -f "$HF/model-00001-of-00001.safetensors" ]; then
    echo "[convert] step-${STEP} DCP -> HF"
    mkdir -p "$HF"
    "$REPO/.venv/bin/python" "$REPO/torchtitan/experiments/ezpz/eval/convert_to_hf.py" \
        "$DCP" "$HF" \
        --model_name experiments.ezpz.agpt --model_flavor "$MODEL_FLAVOR" --export_dtype bfloat16 || {
            echo "CONVERT FAILED step-${STEP}"; exit 1; }
    cp "$HF_CONFIG" "$HF/config.json"
    for f in tokenizer.json tokenizer.model tokenizer_config.json special_tokens_map.json; do
        cp "$TOK/$f" "$HF/"
    done
fi

# ---- Eval (eval venv: conda torch 2.8.0 <-> vllm 0.11, TP=4) ----
echo "[eval] step-${STEP} lm-eval ${TASKS}"
mkdir -p "$RES"
source "$REPO/venvs/lm-eval/bin/activate"
lm_eval --model vllm \
    --model_args "pretrained=$HF,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.90,max_model_len=2048,enforce_eager=True" \
    --tasks "$TASKS" --batch_size auto --num_fewshot 0 --output_path "$RES"
echo "[eval] step-${STEP} DONE rc=$?"
