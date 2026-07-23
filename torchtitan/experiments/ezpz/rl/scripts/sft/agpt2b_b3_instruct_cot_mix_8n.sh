#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=24:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=8
#PBS -q workq
#PBS -j oe
#
# 8N Sunspot SFT: full B3 cold-start mix run (docs/production/sft/agpt/2b-mds/
# b3-instruct-cot-mix/), continuing the global_step138650 base (vocab 256000)
# on the pretokenized agpt2b-b3-instruct-cot-mix-len8192 dataset, 1 epoch.
#
# CONFIRMED FACTS FROM THE 2N SMOKE (job 12471596, rc=0):
#   - per_device_train_batch_size=1 @ max_length=8192 FITS on XPU (no OOM /
#     OUT_OF_RESOURCES over 20 logged steps) -- this is the fit ceiling, do
#     NOT increase micro-batch beyond 1 @ 8192.
#   - ~14s/step measured at 2N (24 ranks, GBS=192 seqs).
#   - Dataset: 2,767,878 packed sequences at
#     /tegu/datasets/datasets/agpt2b-b3-instruct-cot-mix-len8192 (arrow,
#     input_ids + assistant_masks + seq_lengths already on disk).
#   - Pretokenized path auto-sets packing=False + assistant_only_loss=False
#     in train_sft.py (assistant_masks present) -- correct, left as-is.
#
# WHY 8N (mirrors agpt2b_gs138650_tulu_math_uc_mix_8n_gbs6144.sh, not the 32N
# variant): that script's 32N tulu_math_uc_mix run hit a scale-only oneCCL
# collective GPU fault at 384 ranks (12/16/32N), while 2N/4N/8N (24/48/96
# ranks) ran clean -- see that script's header for the full bisect. 8N is the
# largest scale proven safe on this base/mix family, so this run starts there
# directly rather than re-deriving the bisect for the B3 mix.
#
# Sizing (micro-batch held at the smoke-proven fit; GAS chosen so tokens/step
# matches the tulu-math 8N run's 6.29M tokens/step, since B3 sequences are 8x
# longer than tulu-math's 1024 so GBS in *sequences* must drop by 8x to hold
# tokens/step roughly constant):
#   NGPUS = 8 nodes x 12 ranks = 96
#   GBS = NGPUS x per_device_train_batch_size x gradient_accumulation_steps
#       = 96 x 1 x 8 = 768 sequences/step
#   tokens/step = GBS x max_length = 768 x 8192 = 6,291,456 (~6.29M, matches
#       the tulu-math 8N run's 96 x 2 x 32 x 1024 = 6.29M tokens/step exactly)
#   per-device micro-batch = 1 x 8192 = 8192 tokens (proven-fitting, smoke)
#
# 2,767,878 packed seqs / 768 seqs/step = ~3604 steps for 1 epoch. At the
# smoke's measured ~14s/step (same per-rank micro-batch/GAS, only node count
# differs from the smoke), 1 epoch is ~3604 x 14s ~= 14h -- inside this job's
# 24h walltime window, but --resume_from_checkpoint + save_steps=50 make this
# safely chainable with `qsub -W depend=afterany:<jobid>` if actual per-step
# wall time at 8N (vs the 2N smoke) runs long enough to miss the window.
#
# NO --auto-retry: same rationale as the tulu-math 8N/32N launchers --
# ezpz auto-retry's stuck_pre_training heuristic false-positives on TRL's
# `{'loss': ...}` log-dict output (it only recognizes ezpz's own step=N
# marker). Fault tolerance here is the afterany chain + --resume_from_checkpoint
# + frequent checkpoints (save_steps=50), plus a fresh PBS allocation per
# chain link to sidestep any single bad node.
#
# Output: outputs/sft/agpt2b-b3-instruct-cot-mix-8n/

set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
# Raise oneCCL's Level-Zero IPC-handle caches -- unbounded growth over a long
# run has previously exhausted device memory in a collective memcpy (see the
# tulu-math 32N launcher header, job 12470088).
export CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD="${CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD:-8000}"
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD="${CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD:-8000}"
# Offline: the pretokenized dataset already has input_ids on disk, and the
# base model is a local HF dir -- nothing here should touch the network.
export HF_HUB_OFFLINE=1
export HF_HOME=/tegu/datasets/datasets/hf
export HF_DATASETS_CACHE=/tegu/datasets/datasets/hf/datasets
export EZPZ_SFT_MIX_CACHE_DIR=/tegu/datasets/datasets/ezpz_sft_mixes
# ezpz#163 UnicodeDecodeError backstop -- see agpt2b_gs138650_tulu_math_uc_mix_8n_gbs6144.sh
# header for the full diagnosis (killed jobs 12470352 + 12470380 mid-run).
export PYTHONIOENCODING="ascii:replace"
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"
source .venv/bin/activate
python3 -c "import trl; print('trl', trl.__version__)" || { echo "FATAL: trl missing"; exit 1; }

BASE_MODEL="${HOME}/global_step138650"
PRETOK_DIR="/tegu/datasets/datasets/agpt2b-b3-instruct-cot-mix-len8192"
CKPT_DIR="outputs/sft/agpt2b-b3-instruct-cot-mix-8n"
LOG_DIR="logs/sft-agpt2b-b3-instruct-cot-mix-8n-${PBS_JOBID%%.*}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

if [[ ! -d "${PRETOK_DIR}" ]]; then
    echo "FATAL: pre-tokenized dataset missing at ${PRETOK_DIR}" \
        | tee -a "${LOG_DIR}/run.log"
    echo "Run _pretokenize_b3_instruct_cot_mix_1n.sh first." \
        | tee -a "${LOG_DIR}/run.log"
    exit 1
fi

echo "=== 8N SFT: gs138650 + B3 instruct-cot mix, 1 epoch, GBS=768 seqs, seq=8192 ===" \
    | tee "${LOG_DIR}/run.log"
echo "BASE_MODEL=${BASE_MODEL}" | tee -a "${LOG_DIR}/run.log"
echo "PRETOK_DIR=${PRETOK_DIR}" | tee -a "${LOG_DIR}/run.log"
echo "Sizing: 96 ranks x bsz=1 x gas=8 = GBS=768 seqs, 6.29M tokens/step; ~3604 steps/epoch" \
    | tee -a "${LOG_DIR}/run.log"
echo "loading PRE-TOKENIZED dataset (TRL skips tokenize+pack; fast to step 1)" \
    | tee -a "${LOG_DIR}/run.log"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/train_sft.py \
    torchtitan/experiments/ezpz/rl/datasets_sft.py 2>&1 | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

# NO --auto-retry (see header). Fault tolerance via the afterany chain +
# --resume_from_checkpoint + save_steps=50 -- a hang just times out and the
# next chain link resumes from the last checkpoint.
ezpz launch --np 96 -ppn 12 --timeout "${IDLE_TIMEOUT:-1800}" \
    python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --pretokenized_dataset "${PRETOK_DIR}" \
    --model_name_or_path "${BASE_MODEL}" \
    --output_dir "${CKPT_DIR}" \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_length 8192 \
    --bf16 --fsdp full_shard \
    --logging_steps 10 \
    --save_strategy steps --save_steps 50 \
    --report_to wandb \
    --resume_from_checkpoint "${CKPT_DIR}" \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== DONE: log in ${LOG_DIR}/, ckpts in ${CKPT_DIR}/ ===" \
    | tee -a "${LOG_DIR}/run.log"
