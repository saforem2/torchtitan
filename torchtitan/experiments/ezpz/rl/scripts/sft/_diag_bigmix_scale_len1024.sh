#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=00:45:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe
#
# SCALE-THRESHOLD bisect for the 384-rank GPU page fault.
#
# The big-mix len1024 SFT (bsz2/gas8) trains CLEAN at 2N (24 ranks, job
# 12470343) but GPU-page-faults at 32N (384 ranks) at step 1-2:
# "Segmentation fault from GPU ... NotPresent Write" -> rank signal 6. OOV and
# bad-node ruled out -> it is a rank-scale collective/GPU fault. This script
# runs the SAME dataset+config at whatever node count `select` is set to (pass
# `qsub -l select=N`), computing --np from the live allocation, to bisect the
# largest N that still trains. 20 steps is enough (the fault hits at step 1-2).
#
# Bracket: 2N works, 32N fails. Bisect 4/8/16N.
set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD="${CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD:-8000}"
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD="${CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD:-8000}"
export HF_HUB_OFFLINE=1
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"
source .venv/bin/activate

# Compute NP from the live allocation: 12 ranks per node x number of hosts.
NHOSTS=$(sort -u "${PBS_NODEFILE}" | wc -l)
NP=$(( NHOSTS * 12 ))

BASE_MODEL="${HOME}/global_step138650"
PRETOK_DIR="${HOME}/.cache/ezpz_sft_mixes/tokenized/tulu_math_uc_mix-gs138650-len1024"
CKPT_DIR="outputs/sft/_diag_bigmix_scale_len1024"
LOG_DIR="logs/diag-bigmix-scale-${NHOSTS}n-${PBS_JOBID%%.*}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

echo "=== SCALE DIAG: big len1024 mix, ${NHOSTS}N = ${NP} ranks, 20 steps ===" \
    | tee "${LOG_DIR}/run.log"
echo "PRETOK_DIR=${PRETOK_DIR}" | tee -a "${LOG_DIR}/run.log"

ezpz launch --np "${NP}" -ppn 12 --timeout "${IDLE_TIMEOUT:-1800}" \
    python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --pretokenized_dataset "${PRETOK_DIR}" \
    --model_name_or_path "${BASE_MODEL}" \
    --output_dir "${CKPT_DIR}" \
    --max_steps 20 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_length 1024 \
    --bf16 --fsdp full_shard \
    --logging_steps 1 \
    --save_strategy no \
    --report_to none \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== DONE ${NHOSTS}N: 20 steps clean = OK at this scale; GPU segfault = fault at this scale ===" \
    | tee -a "${LOG_DIR}/run.log"
