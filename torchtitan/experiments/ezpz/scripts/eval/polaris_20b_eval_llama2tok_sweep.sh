#!/bin/bash --login
#PBS -N p20b-l2sweep
#PBS -A AuroraGPT
#PBS -q preemptable
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:eagle
#PBS -j oe
#
# Corrected Polaris 20B eval sweep: re-eval existing faithful HF exports with the
# LLAMA2 tokenizer (the run's true tokenizer). See
# docs/reference/known-bugs/polaris-20b-tokenizer-mismatch.md. The DCP->HF weight
# export was always correct; only the eval tokenizer was wrong (gemma vs Llama2),
# so this reuses each step-N/hf/ export and only overrides tokenizer= -- no
# re-conversion. gmu=0.80 for the 256128-vocab logit headroom (0.90 OOMs).
#
# Idempotent: skips a step whose results-llama2tok/ already has results.
# Submit:  qsub torchtitan/experiments/ezpz/scripts/eval/polaris_20b_eval_llama2tok_sweep.sh
# Override: qsub -v STEPS="100 500 1000 1500 2000 2400" ... this_script
set -x
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export no_proxy=localhost,127.0.0.1,*.alcf.anl.gov,*.anl.gov
export HF_HUB_ENABLE_HF_TRANSFER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO=/eagle/AuroraGPT/foremans/projects/saforem2/torchtitan
cd "$REPO" || exit 1
export PYTHONPATH="$REPO"
EVAL_BASE="$REPO/outputs/evals/agpt-20b-dolma-n128"
TOK="$REPO/assets/hf/llama2-dolma-tokenizer"
STEPS="${STEPS:-100 500 1000 1500 2000 2400}"
TASKS="${TASKS:-hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq}"
TP="${TP:-4}"
source "$REPO/venvs/lm-eval/bin/activate"
for STEP in $STEPS; do
    HF="$EVAL_BASE/step-${STEP}/hf"
    RES="$EVAL_BASE/step-${STEP}/results-llama2tok"
    echo "==================== step-${STEP} ($(date)) ===================="
    if [ ! -e "$HF/model-00001-of-00001.safetensors" ]; then
        echo "SKIP step-${STEP}: no HF export"; continue
    fi
    if compgen -G "$RES/**/results_*.json" > /dev/null 2>&1; then
        echo "SKIP step-${STEP}: results-llama2tok already exist"; continue
    fi
    mkdir -p "$RES"
    lm_eval --model vllm \
        --model_args "pretrained=$HF,tokenizer=$TOK,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.80,max_model_len=2048,enforce_eager=True" \
        --tasks "$TASKS" --batch_size auto --num_fewshot 0 --output_path "$RES"
    echo "[eval] step-${STEP} rc=$?"
done
echo "==================== SWEEP COMPLETE ($(date)) ===================="
