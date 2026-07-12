#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=12:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=36
#PBS -q workq
#PBS -j oe
#
# 32N Sunspot SFT continuing the global_step138650 base (vocab 256000) on the
# BIG tulu_math_uc_mix (tulu-3 0.65 + OpenMathInstruct-2 0.15 + ultrachat 0.20,
# ~93.1M rows, ~54B tokens), 1 epoch at production-equivalent GBS=6144.
#
# WHY this base (not the v2 256N step-92859 checkpoint): global_step138650
# (vocab 256000) is the checkpoint the already-completed 729-step SFT
# (checkpoint-729-hf) trained on, and it ran CLEAN at 32N. The v2 base
# (vocab 256128) deterministically hit a scale-only oneCCL collective GPU fault
# (libccl.so backtrace, 384 ranks only; 1-tile + 2N are fine) across jobs
# 12470254/258/262. Building on gs138650 sidesteps that crash.
#
# WHY the big mix (not the metamathqa mix-spec): the OpenMathInstruct-2 mix used
# to take ~90 min to interleave-build and blew the XPU oneCCL barrier at 384
# ranks. The vectorized interleave (commit b5798f532) + pre-built on-disk cache
# (hash 44608c8c15bd9714, 286GB, job 12470278) turns that into a <5s
# Dataset.load_from_disk with NO Hub calls -- so this path has zero HF 429
# exposure (unlike streaming --dataloader.dataset paths, which have no
# rank-0 barrier and re-probe the Hub per rank). 2N smoke (job 12470280)
# confirmed: cache loaded 93M rows in 49.1s, 10 steps, 0 crash.
#
# Sizing (GBS held at 6144):
#   NGPUS = 32 nodes x 12 ranks = 384
#   GBS = NGPUS x per_device_train_batch_size x gradient_accumulation_steps
#       = 384 x 2 x 8 = 6144
#   tokens/step = GBS x max_length = 6144 x 1024 = 6.29M
#   per-device micro-batch = 2 x 1024 = 2048 tokens
#
# seq_len (max_length) = 1024. seq_len 2048 OOM'd the XPU tile at step 0
# (job 12470328, UR_RESULT_ERROR_OUT_OF_RESOURCES in compute_loss): packing
# makes every sequence a dense full-length block, so 2048 ~doubled activation
# memory vs the completed SFT's 1024, and bsz=1 + AC (via fsdp_config) were
# already maxed. 1024 is the proven-fitting length -- the 729-step SFT ran at
# 1024 with bsz=2/gas=8, so we use that known-good corner here.
#
# 1 epoch over ~54B tokens at 6.29M tokens/step is ~8.6k steps -- longer than
# one 12h walltime window. Checkpoints (every 200 steps) + --resume_from_
# checkpoint + a chained afterany continuation carry it across windows.
#
# Allocation: select=36 (32 train + 4 spare) for --auto-retry / --spare-nodes.
#
# Output: outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-32n-gbs6144/

set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
# Raise oneCCL's Level-Zero IPC-handle caches -- the GET/OPEN handle caches grow
# unbounded over a long run and eventually exhaust device memory in a collective
# memcpy (job 12470088 OOM'd at step ~248 with ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY
# and oneCCL's own "Increase CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD" hint).
export CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD="${CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD:-8000}"
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD="${CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD:-8000}"
# The mix is a pre-built on-disk cache (load_from_disk); no Hub downloads are
# needed. OFFLINE avoids any stray per-rank dataset_info probe -> 429 storm.
export HF_HUB_OFFLINE=1
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"
source .venv/bin/activate
python3 -c "import trl; print('trl', trl.__version__)" || { echo "FATAL: trl missing"; exit 1; }

# Base: global_step138650 (vocab 256000). Absolute path avoids the cwd-relative
# resolution + silent Qwen-0.6B fallback in _resolve_model().
BASE_MODEL="${HOME}/global_step138650"

CKPT_DIR="outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-32n-gbs6144"
LOG_DIR="logs/sft-agpt-2b-gs138650-tulu-math-uc-mix-32n-gbs6144-${PBS_JOBID%%.*}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

echo "=== 32N SFT: gs138650 + tulu_math_uc_mix (BIG cached mix), 1 epoch, GBS=6144, seq=1024 ===" \
    | tee "${LOG_DIR}/run.log"
echo "BASE_MODEL=${BASE_MODEL}" | tee -a "${LOG_DIR}/run.log"
echo "Allocation: select=36 (32 train + 4 spare for --auto-retry)" \
    | tee -a "${LOG_DIR}/run.log"
echo "Sizing: 384 ranks x bsz=2 x gas=8 = GBS=6144, 6.29M tokens/step" \
    | tee -a "${LOG_DIR}/run.log"
echo "loading PRE-TOKENIZED dataset (TRL skips tokenize+pack; <60s to step 1)" \
    | tee -a "${LOG_DIR}/run.log"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/train_sft.py \
    torchtitan/experiments/ezpz/rl/datasets_sft.py 2>&1 | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

# PRE-TOKENIZED dataset (produced by _pretokenize_tulu_math_uc_mix_1n.sh):
# already tokenized+packed with input_ids, so TRL SKIPS the ~8.5h runtime
# tokenize+pack that killed job 12470281 (SFTTrainer re-tokenizes the whole 93M-
# row mix at job start; the auto-retry idle-watchdog SIGTERM'd it before step 1).
# With input_ids present, TRL detects the dataset as processed and training
# starts in <60s. MUST match the pre-tokenize job's --max_length (1024) + base.
PRETOK_DIR="${HOME}/.cache/ezpz_sft_mixes/tokenized/tulu_math_uc_mix-gs138650-len1024"
if [[ ! -d "${PRETOK_DIR}" ]]; then
    echo "FATAL: pre-tokenized dataset missing at ${PRETOK_DIR}" \
        | tee -a "${LOG_DIR}/run.log"
    echo "Run _pretokenize_tulu_math_uc_mix_1n.sh first." \
        | tee -a "${LOG_DIR}/run.log"
    exit 1
fi

# NO --auto-retry for the SFT path. ezpz auto-retry detects training progress
# via `_PROGRESS_MARKER_RX = re.compile(r"\bstep=\d+")` -- the `step=N` marker
# that ezpz's OWN training loop prints. TRL's HF Trainer never emits `step=`; it
# prints `{'loss': ..., 'epoch': ...}` dicts. So auto-retry is structurally
# BLIND to SFT progress and ALWAYS declares `FAILOVER STOP: stuck_pre_training`
# after 2 attempts with "zero step= markers" -- even though training is stepping
# fine (job 12470336: stepped loss 1.334, hit one intermittent SIGABRT after
# step 1, auto-retry then killed the whole run as "stuck pre-training"). That is
# a false positive we cannot fix from the ezpz experiment tree (it lives in the
# pip-installed ezpz lib).
#
# Fault tolerance instead comes from JOB-LEVEL chaining: the `afterany`
# continuation + --resume_from_checkpoint. A SIGABRT ends this job; the queued
# continuation relaunches and resumes from the latest checkpoint. To make an
# early SIGABRT cheap (no checkpoint exists until the first save), save_steps=50
# so a resume loses at most ~50 steps. select=36 gives PBS spare capacity for
# the relaunch to land on healthy nodes.
# No --save_total_limit: keep ALL checkpoints (golden rule -- never keep-latest-k;
# intermediate ckpts are wanted for per-token eval).
ezpz launch --np 384 -ppn 12 --timeout "${IDLE_TIMEOUT:-1800}" \
    python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --pretokenized_dataset "${PRETOK_DIR}" \
    --model_name_or_path "${BASE_MODEL}" \
    --output_dir "${CKPT_DIR}" \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_length 1024 \
    --bf16 --fsdp full_shard \
    --logging_steps 10 \
    --save_strategy steps --save_steps 50 \
    --report_to wandb \
    --resume_from_checkpoint "${CKPT_DIR}" \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true

# --resume_from_checkpoint <dir> is HF Trainer's auto-resume hook: empty dir ->
# fresh start; has checkpoint-N -> resume latest. Critical for the chained
# afterany continuation to pick up across a SIGABRT or a walltime window.

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== DONE: log in ${LOG_DIR}/, ckpts in ${CKPT_DIR}/ ===" \
    | tee -a "${LOG_DIR}/run.log"
