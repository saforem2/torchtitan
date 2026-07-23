#!/bin/bash --login
#PBS -N p20b-eval-l2verify
#PBS -A AuroraGPT
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -j oe
#
# VERIFY the tokenizer-mismatch root cause for the Polaris 20B dolma run.
#
# Root cause (2026-07-23): the Polaris data-list points at
# /eagle/datasets/dolma/data_v1.7_Llama2Tokenizer -- i.e. the dolma training
# data is LLAMA2-tokenized (vocab 32000, uint16 bins, all ids < 32000). But the
# model config declares vocab_size=256128 and the eval used the GEMMA-7B
# tokenizer. The model is coherent in Llama2 id-space (loss ~2.05 is real);
# decoding its outputs with gemma produces fluent-subword salad. Proof: raw
# training ids decode to clean English under Llama2 and to salad under gemma.
#
# This job re-evals step-2400 with the CORRECT (Llama2) tokenizer. It reuses the
# existing faithful HF weight export via the sibling hf-llama2/ dir (safetensors
# symlinked, Llama2 tokenizer files, config.json with bos=1/eos=2). If the MC
# tasks now score ABOVE CHANCE, the model is healthy and the mismatch was the
# whole bug. All eval tasks here are loglikelihood (no free generation), so the
# untrained embedding rows >=32000 never appear in gold continuations.
#
# Submit:  qsub torchtitan/experiments/ezpz/scripts/eval/polaris_20b_eval_llama2tok_verify.sh

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

HF="$REPO/outputs/evals/agpt-20b-dolma-n128/step-${STEP}/hf-llama2"
RES="$REPO/outputs/evals/agpt-20b-dolma-n128/step-${STEP}/results-llama2tok"

if [ ! -e "$HF/model-00001-of-00001.safetensors" ]; then
    echo "ABORT: $HF weights missing (build hf-llama2 dir first)"; exit 1
fi
mkdir -p "$RES"
source "$REPO/venvs/lm-eval/bin/activate"

# Quick generation sanity FIRST: this must be fluent English now.
python - <<PY
from vllm import LLM, SamplingParams
llm = LLM(model="$HF", tensor_parallel_size=, dtype="bfloat16",
          gpu_memory_utilization=0.90, max_model_len=2048, enforce_eager=True)
sp = SamplingParams(max_tokens=40, temperature=0.0)
for p in ["The capital of France is", "Once upon a time, there was a", "2 + 2 ="]:
    o = llm.generate([p], sp)[0].outputs[0].text
    print("PROMPT:", repr(p)); print("GEN:   ", repr(o)); print("-"*40)
PY

echo "[eval] step-${STEP} lm-eval (Llama2 tokenizer) ${TASKS}"
lm_eval --model vllm     --model_args "pretrained=$HF,tokenizer=$HF,tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.90,max_model_len=2048,enforce_eager=True"     --tasks "$TASKS" --batch_size auto --num_fewshot 0 --output_path "$RES"
echo "[eval] step-${STEP} DONE rc=$?"
