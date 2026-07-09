#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=12:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=36
#PBS -q workq
#PBS -j oe
#
# 32N Sunspot SFT for the COMPLETED v2 2B production base
# (step-92,859 = 4.674T tokens, 256N chain) on tulu_math_uc_mix, 3 epochs
# at production-equivalent GBS=6144. This is the first production SFT on the
# completed v2 base; the prior production SFT
# (aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf) used the older
# AuroraGPT-2B-sophiag-gs138650 base.
#
# The v2 base was DCP->HF converted on Aurora (step-92859) and staged to the
# Sunspot repo root as AuroraGPT-2B-v2-256n-step92859-hf (2026-07-06).
# Validated by the 2N smoke (job 12470086): 10 steps, loss 2.08->1.7,
# mean_token_accuracy ~0.60, 0 Qwen fallback, 0 barrier crash.
#
# Sizing:
#   NGPUS = 32 nodes x 12 ranks = 384
#   GBS = NGPUS x per_device_train_batch_size x gradient_accumulation_steps
#       = 384 x 2 x 8 = 6144
#   tokens/step = GBS x max_length = 6144 x 1024 = 6.29M
#
# 3 epochs at GBS=6144 over the materialized metamathqa-swap mix is ~729
# steps (matches the gs138650 run's step count at the same mix + GBS).
#
# Allocation: select=36 (32 train + 4 spare) for --auto-retry. Past 32N SFT
# runs hit ccl::v1::exception -> SIGABRT on a worker rank within ~10-15 min;
# auto-retry's bad-node detection swaps in a spare and continues from the
# latest checkpoint. CAVEAT: ezpz#163 (auto-retry _drain can die on non-UTF-8
# stdout, then the watchdog SIGTERMs healthy training after 30 min idle) is
# still open -- watch for that signature.
#
# Output: outputs/sft/agpt-2b-v2-256n-tulu-mix-32n-gbs6144/

set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
# Raise oneCCL's Level-Zero IPC-handle caches. With the defaults, the GET/OPEN
# handle caches grow unbounded over a long run and eventually exhaust device
# memory in a collective memcpy -- job 12470088 OOM'd at step ~248/729 with
# `ze error at zeCommandListAppendMemoryCopy, ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`
# and oneCCL's own hint "Increase the cache size CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD".
# Same knob moe_ab_check.sh already uses (OPEN variant) for its A2A segfault.
export CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD="${CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD:-8000}"
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD="${CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD:-8000}"
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"
source .venv/bin/activate
python3 -c "import trl; print('trl', trl.__version__)" || { echo "FATAL: trl missing"; exit 1; }

# NEW base: absolute path to the converted v2 256N step-92,859 checkpoint.
# Absolute (not bare/relative) because the script does `cd "${SUBMIT_DIR}"`
# and _resolve_model() resolves relative names against repo root; an absolute
# path avoids cwd ambiguity and the silent Qwen-0.6B fallback that
# _resolve_model() takes if from_pretrained throws.
BASE_MODEL="${SUBMIT_DIR}/AuroraGPT-2B-v2-256n-step92859-hf"

# Fresh, unique CKPT_DIR distinct from the gs138650 run's dir. --resume_from_
# checkpoint below points here: empty on first launch (train_sft.py coerces to
# fresh start), then enables auto-retry resume from the latest checkpoint-N.
CKPT_DIR="outputs/sft/agpt-2b-v2-256n-tulu-mix-32n-gbs6144"
LOG_DIR="logs/sft-agpt-2b-v2-256n-tulu-mix-32n-gbs6144-${PBS_JOBID%%.*}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

echo "=== 32N SFT: AuroraGPT-2B v2 256N step-92859 + tulu_math_uc_mix, 3 epochs, GBS=6144 ===" \
    | tee "${LOG_DIR}/run.log"
echo "BASE_MODEL=${BASE_MODEL}" | tee -a "${LOG_DIR}/run.log"
echo "Allocation: select=36 (32 train + 4 spare for --auto-retry)" \
    | tee -a "${LOG_DIR}/run.log"
echo "Sizing: 384 ranks x bsz=2 x gas=8 = GBS=6144, 6.29M tokens/step" \
    | tee -a "${LOG_DIR}/run.log"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/train_sft.py \
    torchtitan/experiments/ezpz/rl/datasets_sft.py 2>&1 | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

# Mix-spec syntax (not the registered tulu_math_uc_mix name) swaps
# OpenMathInstruct-2 (14M rows, ~20 min interleave -> blows the XPU oneCCL
# barrier, crashed job 12468398) for metamathqa (395k, ~10s build). Same
# GSM8K+MATH distribution coverage.
# --spare-nodes auto is REQUIRED for --auto-retry to actually swap a bad node:
# the job requests select=36 (32 active + 4 spare); --np 384 pins the trainer
# to 32 nodes' ranks, and --spare-nodes auto tells ezpz the remaining 4 are
# swappable spares. WITHOUT this flag, --auto-retry has no spare to swap in, so
# a bad-node GPU fault (job 12470254 rank 61 / node x1922c3s3b0n0, and again
# 12470258 which PBS re-handed the same node) just dead-ends at
# `FAILOVER STOP: stuck_pre_training`. The SFT venv is on the shared tegu FS
# (source .venv/bin/activate above), NOT a per-node /tmp yeet, so a swapped-in
# spare already sees it -- no `ezpz yeet` needed (unlike the pretraining
# autoretry scripts).
ezpz launch --np 384 -ppn 12 --auto-retry --spare-nodes auto \
    --max-failover-retries 3 --timeout "${IDLE_TIMEOUT:-1800}" \
    python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --sft_dataset 'tulu-3-sft-mixture:0.65,metamathqa:0.15,ultrachat-200k:0.20' \
    --model_name_or_path "${BASE_MODEL}" \
    --output_dir "${CKPT_DIR}" \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_length 1024 \
    --bf16 --fsdp full_shard \
    --logging_steps 10 \
    --save_strategy steps --save_steps 100 \
    --save_total_limit 8 \
    --report_to wandb \
    --resume_from_checkpoint "${CKPT_DIR}" \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true

# --resume_from_checkpoint <dir> is HF Trainer's auto-resume hook: if <dir>
# has a checkpoint-N subdir, training picks up from the latest; if not,
# training starts from scratch. Critical for ezpz auto-retry: a bad-node
# crash triggers a fresh `python3 -m train_sft` relaunch (not an in-process
# restart), so without this flag every retry restarts from step 0.

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== DONE: log in ${LOG_DIR}/, ckpts in ${CKPT_DIR}/ ===" \
    | tee -a "${LOG_DIR}/run.log"
