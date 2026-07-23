#!/bin/bash --login
#PBS -A datascience
#PBS -N test-agpt-maxautotune
#PBS -l walltime=01:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe
#
# Runtime test of the ezpz AGPT_COMPILE_MODE=max-autotune override
# (agpt/parallelize.py _apply_compile_with_mode). Proves torch.compile
# mode=max-autotune threads through torchtitan's per-TransformerBlock compile
# and engages Triton GEMM autotune on XPU. Uses agpt_2b_real (complex-RoPE,
# compile-lowerable, NON-flex) to isolate GEMM autotune from the FlexAttention
# backward-autotune OOM. SUCCESS = log shows "(backend=inductor,
# mode=max-autotune) [ezpz AGPT_COMPILE_MODE]" + "SingleProcess AUTOTUNE
# benchmarking ... triton_mm_*". Short (6 steps), no checkpoint.
# no set -euo (venv activate has unbound vars)
set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu >/dev/null 2>&1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
# THE knob under test:
export AGPT_COMPILE_MODE="${AGPT_COMPILE_MODE:-max-autotune}"

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"
source .venv/bin/activate

CONFIG="${CONFIG:-agpt_2b_real}"
STEPS="${STEPS:-6}"
DFL="torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/olmo-mix-1124.txt"
LOG_DIR="logs/test-maxautotune-${PBS_JOBID%%.*}"; mkdir -p "${LOG_DIR}"

echo "=== max-autotune test: config=${CONFIG} AGPT_COMPILE_MODE=${AGPT_COMPILE_MODE} steps=${STEPS} ===" \
    | tee -a "${LOG_DIR}/run.log"

ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module=ezpz.agpt \
    --config="${CONFIG}" \
    --compile.enable \
    --checkpoint.no-enable \
    --dataloader.dataset=blendcorpus \
    --dataloader.dataset-path="${DFL}" \
    --training.steps="${STEPS}" \
    2>&1 | tee -a "${LOG_DIR}/run.log"

echo "=== VERDICT (grep for the override + Triton autotune) ===" | tee -a "${LOG_DIR}/run.log"
grep -E "ezpz AGPT_COMPILE_MODE|SingleProcess AUTOTUNE|triton_mm_" "${LOG_DIR}/run.log" | head | tee -a "${LOG_DIR}/run.log"
echo "=== DONE: log in ${LOG_DIR}/run.log ===" | tee -a "${LOG_DIR}/run.log"
