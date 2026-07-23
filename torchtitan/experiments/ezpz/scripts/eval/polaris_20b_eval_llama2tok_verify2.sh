#!/bin/bash --login
#PBS -N p20b-l2verify2
#PBS -A AuroraGPT
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -j oe
#
# Fallback of polaris_20b_eval_llama2tok_verify.sh that avoids the symlinked
# hf-llama2/ dir (vLLM safetensors resolution can choke on symlinks). Points
# pretrained= at the ORIGINAL hf/ export (real safetensors + gemma-vocab
# config), and overrides tokenizer= to the Llama2 dir. For loglikelihood MC
# tasks the tokenizer override is what determines id-space; config bos/eos
# only affect free generation. See known-bugs/polaris-20b-tokenizer-mismatch.md.
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
STEP="${STEP:-2400}"
TASKS="${TASKS:-hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq}"
TP="${TP:-4}"
HF="$REPO/outputs/evals/agpt-20b-dolma-n128/step-${STEP}/hf"
TOK="$REPO/assets/hf/llama2-dolma-tokenizer"
RES="$REPO/outputs/evals/agpt-20b-dolma-n128/step-${STEP}/results-llama2tok"
mkdir -p "$RES"
source "$REPO/venvs/lm-eval/bin/activate"
echo "[eval] step-${STEP} lm-eval (Llama2 tokenizer override) ${TASKS}"
lm_eval --model vllm     --model_args "pretrained=$HF,tokenizer=$TOK,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.80,max_model_len=2048,enforce_eager=True"     --tasks "$TASKS" --batch_size auto --num_fewshot 0 --output_path "$RES"
echo "[eval] step-${STEP} DONE rc=$?"
