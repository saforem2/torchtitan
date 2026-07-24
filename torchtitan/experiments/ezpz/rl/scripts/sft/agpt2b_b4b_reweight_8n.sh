#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=24:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=8
#PBS -q workq
#PBS -j oe
#
# 8N Sunspot SFT: B4b reweighted single-stage mix (docs/production/sft/agpt/
# 2b-mds/b4-finish-and-reweight/), continuing the global_step138650 base
# (vocab 256000) on the pretokenized agpt2b-b4-reweight-len4096 dataset, 1
# epoch.
#
# WHY THIS RUN EXISTS: B3 (single combined SFT from gs138650 on a broad
# instruct+CoT mix at gsm8k-r1cot weight 0.15) REGRESSED the GSM8K CoT metric
# (cot_accuracy 0.05 vs B2's 0.205) -- traced to long-form OpenR1 R1 traces
# diluting the short-CoT finishing signal and teaching a verbose, often-wrong
# style (mean gen_len 601 vs B2's 282, 26/200 generations never closed
# </answer>). B4b (this launcher, "Path B") tests whether a single
# well-balanced SFT beats B3 by (1) restoring gsm8k-r1cot to 0.40, (2)
# length-filtering OpenR1 traces to <=1200 chars so only short, useful
# reasoning survives, (3) dropping OpenMathInstruct-2 entirely, (4) halving
# max_length to 4096 since no long traces remain worth preserving. See
# design.md for the full diagnosis; B4a ("Path A", a separate launcher) tests
# the alternative hypothesis of a 2nd finishing stage on the B3 checkpoint.
#
# b4_reweight_mix dataset info (see datasets_sft.py):
#   0.40 gsm8k-r1cot, 0.15 OpenR1-Math-220k (length-filtered), 0.30
#   tulu-3-sft-mixture, 0.15 ultrachat-200k. Pretokenized via
#   _pretokenize_b4_reweight_mix_1n.sh @ max_length=4096 to
#   /tegu/datasets/datasets/agpt2b-b4-reweight-len4096/ -- run that FIRST.
#   Pretokenized path auto-sets packing=False + assistant_only_loss=False in
#   train_sft.py (assistant_masks present on disk) -- correct, left as-is.
#
# Sizing (mirrors agpt2b_b3_instruct_cot_mix_8n.sh's node count/topology;
# max_length is halved here so the same per-device micro-batch has headroom,
# but we hold micro-batch/GAS at the B3-proven values rather than speculatively
# doubling GBS -- confirm fit with a smoke before scaling further):
#   NGPUS = 8 nodes x 12 ranks = 96
#   per-device micro-batch = 1 x 4096 = 4096 tokens (B3 proved 1 x 8192 fits;
#       4096 is strictly smaller, so this is expected to fit with margin)
#   GBS = NGPUS x per_device_train_batch_size x gradient_accumulation_steps
#       = 96 x 1 x 8 = 768 sequences/step
#   tokens/step = GBS x max_length = 768 x 4096 = 3,145,728 (~3.15M)
#
# The b4_reweight_mix row count is expected to be well below B3's 13.3M
# (OpenMathInstruct-2 dropped entirely + OpenR1 length-filtered), so exact
# steps/epoch depends on the pretokenize job's reported row count -- check
# logs/pretokenize-b4-reweight-mix-*/run.log before estimating a walltime
# budget for this launcher.
#
# NO --auto-retry: same rationale as agpt2b_b3_instruct_cot_mix_8n.sh --
# ezpz auto-retry's stuck_pre_training heuristic false-positives on TRL's
# `{'loss': ...}` log-dict output (it only recognizes ezpz's own step=N
# marker). Fault tolerance here is the afterany chain + --resume_from_checkpoint
# + frequent checkpoints (save_steps=50), plus a fresh PBS allocation per
# chain link to sidestep any single bad node.
#
# Output: outputs/sft/agpt2b-b4b-reweight-8n/

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
PRETOK_DIR="/tegu/datasets/datasets/agpt2b-b4-reweight-len4096"
CKPT_DIR="outputs/sft/agpt2b-b4b-reweight-8n"
LOG_DIR="logs/sft-agpt2b-b4b-reweight-8n-${PBS_JOBID%%.*}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

if [[ ! -d "${PRETOK_DIR}" ]]; then
    echo "FATAL: pre-tokenized dataset missing at ${PRETOK_DIR}" \
        | tee -a "${LOG_DIR}/run.log"
    echo "Run _pretokenize_b4_reweight_mix_1n.sh first." \
        | tee -a "${LOG_DIR}/run.log"
    exit 1
fi

echo "=== 8N SFT: gs138650 + B4b reweighted mix, 1 epoch, GBS=768 seqs, seq=4096 ===" \
    | tee "${LOG_DIR}/run.log"
echo "BASE_MODEL=${BASE_MODEL}" | tee -a "${LOG_DIR}/run.log"
echo "PRETOK_DIR=${PRETOK_DIR}" | tee -a "${LOG_DIR}/run.log"
echo "Sizing: 96 ranks x bsz=1 x gas=8 = GBS=768 seqs, 3.15M tokens/step" \
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
    --max_length 4096 \
    --bf16 --fsdp full_shard \
    --logging_steps 10 \
    --save_strategy steps --save_steps 50 \
    --report_to wandb \
    --resume_from_checkpoint "${CKPT_DIR}" \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== DONE: log in ${LOG_DIR}/, ckpts in ${CKPT_DIR}/ ===" \
    | tee -a "${LOG_DIR}/run.log"
