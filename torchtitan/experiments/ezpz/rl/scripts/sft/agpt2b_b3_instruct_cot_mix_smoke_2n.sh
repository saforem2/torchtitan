#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=01:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe
#
# 2N Sunspot smoke: B3 cold-start SFT mix (agpt2b-b3-instruct-cot-mix-len8192)
# @ max_length=8192, micro-batch=1. Proves the batch-fit gate before the 8N
# run (Task 5): does per_device_train_batch_size=1 @ 8192 tokens FIT on XPU
# without flash-attn (OOM / OUT_OF_RESOURCES is the primary risk -- 8192 is
# 8x the proven-fitting 1024 token budget from the tulu_math_uc_mix 8N run),
# and does loss actually descend over a handful of logged steps.
#
# Sizing:
#   NGPUS = 2 nodes x 12 ranks = 24
#   micro-batch = 1 x 8192 = 8192 tokens/device (8x the proven 2x1024=2048)
#   GBS = 24 x 1 x 8 = 192 sequences/step = 1.57M tokens/step
#
# This is a SMOKE, not a training run: hard-capped to 20 steps via --max_steps
# (independent of dataset size -- --max_train_samples truncates the dataset
# via `dataset.select` but says nothing about how many optimizer steps run
# against it, so on a 2.77M-row pretokenized dataset it would not bound wall
# time the way we need here). --max_steps is a real EzpzSFTConfig/TRL field
# (`max_steps: int = -1` in EzpzSFTConfig, TRL SFTConfig convention: -1 means
# "use num_train_epochs", >0 hard-stops the Trainer after N optimizer steps)
# and is the safest cap because it bounds wall time regardless of dataset
# size. A ckpt also saves at step 10 (--save_steps 10) so a walltime-kill
# still leaves something to inspect.
#
# WHY 2N before 8N: same "prove the config at small scale before committing a
# multi-hour multi-node allocation" pattern as the tulu_math_uc_mix work --
# see project_sft_v2_base_oom_badnode / project_sft_bigmix_tokenize_bottleneck
# for why skipping this step has been costly before (OOMs and stalls only
# show up under the real batch/seqlen config, and the tokenize/pack cost
# alone can burn an entire allocation window before step 1).
#
# Output: outputs/sft/agpt2b-b3-instruct-cot-mix-smoke-2n/
set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
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
CKPT_DIR="outputs/sft/agpt2b-b3-instruct-cot-mix-smoke-2n"
LOG_DIR="logs/sft-agpt2b-b3-instruct-cot-mix-smoke-2n-${PBS_JOBID%%.*}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

if [[ ! -d "${PRETOK_DIR}" ]]; then
    echo "FATAL: pre-tokenized dataset missing at ${PRETOK_DIR}" \
        | tee -a "${LOG_DIR}/run.log"
    echo "Run _pretokenize_b3_instruct_cot_mix_1n.sh first." \
        | tee -a "${LOG_DIR}/run.log"
    exit 1
fi

echo "=== 2N SMOKE: gs138650 + B3 instruct-cot mix, max_length=8192, micro-batch=1 (batch-fit gate) ===" \
    | tee "${LOG_DIR}/run.log"
echo "BASE_MODEL=${BASE_MODEL}" | tee -a "${LOG_DIR}/run.log"
echo "PRETOK_DIR=${PRETOK_DIR}" | tee -a "${LOG_DIR}/run.log"
echo "Sizing: 24 ranks x bsz=1 x gas=8 = GBS=192 seqs, 1.57M tokens/step; capped at --max_steps 20" \
    | tee -a "${LOG_DIR}/run.log"
echo "loading PRE-TOKENIZED dataset (TRL skips tokenize+pack; fast to step 1)" \
    | tee -a "${LOG_DIR}/run.log"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/train_sft.py \
    torchtitan/experiments/ezpz/rl/datasets_sft.py 2>&1 | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

# NO --auto-retry (stuck_pre_training false-positive on TRL's {'loss':...}
# output -- same rationale as the 8N tulu_math_uc_mix launcher). This is a
# bounded 20-step smoke inside a 1h walltime window; a hang just times out.
ezpz launch --np 24 -ppn 12 --timeout "${IDLE_TIMEOUT:-1800}" \
    python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --pretokenized_dataset "${PRETOK_DIR}" \
    --model_name_or_path "${BASE_MODEL}" \
    --output_dir "${CKPT_DIR}" \
    --max_steps 20 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_length 8192 \
    --bf16 --fsdp full_shard \
    --logging_steps 1 \
    --save_strategy steps --save_steps 10 \
    --report_to none \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== DONE: log in ${LOG_DIR}/, ckpts in ${CKPT_DIR}/ ===" \
    | tee -a "${LOG_DIR}/run.log"
