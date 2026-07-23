#!/bin/bash --login
#PBS -N polaris-20b-eval-sweep
#PBS -A AuroraGPT
#PBS -q preemptable
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:eagle
#PBS -j oe
#
# Convert + lm-eval a sampled sweep of Polaris 20B (dolma) checkpoints.
#
# Polaris-specific facts this encodes (learned the hard way, 2026-07-22):
#   1. convert_to_hf.py must run on a COMPUTE node -- it loads the whole 20B
#      into CPU RAM and OOM-kills (exit 137) on a login node. Compute nodes
#      have ~503GB, plenty.
#   2. Conversion runs under the TRAINING venv (.venv, torch 2.12.1+cu129).
#   3. lm-eval runs under a SEPARATE eval venv (venvs/lm-eval) built on the
#      CONDA-BASE python so vllm 0.11 <-> torch 2.8.0 stay ABI-matched. Forcing
#      the training venv's torch 2.12.1 gives `undefined symbol` in vllm's
#      _C.abi3.so. numpy must stay conda's 2.2.6 (numba needs <=2.2), not 2.5.
#   4. Polaris = 4 GPU/node, so TP=4 for the 20B (not Aurora's 12).
#
# Idempotent: skips any step whose results.json already exists, so a preempt
# just resumes. Submit:
#   qsub torchtitan/experiments/ezpz/scripts/eval/polaris_20b_eval_sweep.sh
# Override the step list / tasks via -v:
#   qsub -v STEPS="100 1000 2400",TASKS=hellaswag,arc_easy ... this_script

set -x
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export no_proxy=localhost,127.0.0.1,*.alcf.anl.gov,*.anl.gov
# Do NOT set HF_HUB_OFFLINE=1: lm-eval fetches task datasets from the Hub
# (only some are cached); the model loads from a local path. See the per-step
# script for the full rationale.
export HF_HUB_ENABLE_HF_TRANSFER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

REPO=/eagle/AuroraGPT/foremans/projects/saforem2/torchtitan
cd "$REPO" || exit 1
export PYTHONPATH="$REPO"

CKPT_NAME=agpt-20b-sophiag-dolma-n128-gbs1024
CKPT_BASE="$REPO/outputs/checkpoints/$CKPT_NAME"
EVAL_BASE="$REPO/outputs/evals/agpt-20b-dolma-n128"
TOK="$REPO/assets/hf/gemma-7b"
HF_CONFIG="$REPO/torchtitan/experiments/ezpz/eval/configs/agpt_20b_config.json"

STEPS="${STEPS:-100 500 1000 1500 2000 2400}"
TASKS="${TASKS:-hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq}"
TP="${TP:-4}"
# _real (cos_sin RoPE) flavor -- MUST match how this chain trained, else the
# converter applies the complex-only Q/K permute and exports gibberish. See
# agpt/state_dict_adapter.py.
MODEL_FLAVOR="${MODEL_FLAVOR:-20b_real}"

for STEP in $STEPS; do
    DCP="$CKPT_BASE/step-${STEP}"
    HF="$EVAL_BASE/step-${STEP}/hf"
    RES="$EVAL_BASE/step-${STEP}/results"

    echo "=================================================="
    echo "STEP ${STEP}  ($(date))"
    echo "=================================================="

    if [ ! -f "$DCP/.metadata" ]; then
        echo "SKIP step-${STEP}: no complete DCP checkpoint (.metadata missing)"
        continue
    fi
    if compgen -G "$RES/**/results_*.json" > /dev/null 2>&1 || compgen -G "$RES/results*.json" > /dev/null 2>&1; then
        echo "SKIP step-${STEP}: results already exist"
        continue
    fi

    # ---- Convert (training venv, compute-node RAM) ----
    if [ ! -f "$HF/model-00001-of-00001.safetensors" ]; then
        echo "[convert] step-${STEP} DCP -> HF"
        mkdir -p "$HF"
        "$REPO/.venv/bin/python" "$REPO/torchtitan/experiments/ezpz/eval/convert_to_hf.py" \
            "$DCP" "$HF" \
            --model_name experiments.ezpz.agpt --model_flavor "$MODEL_FLAVOR" --export_dtype bfloat16
        crc=$?
        if [ $crc -ne 0 ]; then echo "CONVERT FAILED step-${STEP} rc=$crc"; continue; fi
        cp "$HF_CONFIG" "$HF/config.json"
        for f in tokenizer.json tokenizer.model tokenizer_config.json special_tokens_map.json; do
            cp "$TOK/$f" "$HF/"
        done
    else
        echo "[convert] step-${STEP} HF already present, skipping conversion"
    fi

    # ---- Eval (eval venv: conda torch 2.8.0 <-> vllm 0.11, TP=4) ----
    echo "[eval] step-${STEP} lm-eval ${TASKS}"
    mkdir -p "$RES"
    source "$REPO/venvs/lm-eval/bin/activate"
    lm_eval --model vllm \
        --model_args "pretrained=$HF,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=4096" \
        --tasks "$TASKS" --batch_size auto --num_fewshot 0 --output_path "$RES"
    echo "[eval] step-${STEP} rc=$?"
    deactivate 2>/dev/null || true
done

echo "=================================================="
echo "SWEEP COMPLETE ($(date))"
echo "=================================================="
