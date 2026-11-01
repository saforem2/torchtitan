#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N ezpz-sync-smoke-polaris
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:eagle
#PBS -l select=2
#PBS -q debug
#PBS -j oe

# Polaris smoke: 4B + 10B olmo3 on A100/CUDA, 2 nodes, 3 steps each.
#
# Validates that the sync branch's attention/activation/compile changes didn't
# break the production CUDA path. Uses --debug.deterministic + fixed seed so
# losses are reproducible across runs.
#
# Unlike the XPU smokes, this does NOT set ZE_FLAT/CCL/ONEAPI — those are
# Intel XPU vars. Polaris is A100/CUDA (NCCL). ezpz_load_modules is
# machine-aware and selects the right stack.
#
# Usage:
#   qsub torchtitan/experiments/ezpz/scripts/sync_smoke_polaris.sh

set -euo pipefail

export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"

STEPS="${SMOKE_STEPS:-3}"
SEED="${SEED:-42}"
VENV="${SMOKE_VENV:-.venv}"

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

# Staged utils (for compute nodes without internet)
EZPZ_UTILS="${SUBMIT_DIR}/.ezpz-utils-cache/ezpz-utils.sh"
if [[ -f "${EZPZ_UTILS}" ]]; then
    # shellcheck disable=SC1090
    source "${EZPZ_UTILS}"
else
    source <(curl -fsSL --max-time 30 https://bit.ly/ezpz-utils)
fi

ezpz_load_modules
ezpz_setup_job

if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "FATAL: no venv at ${VENV}/bin/activate" >&2
    exit 1
fi
source "${VENV}/bin/activate"

echo "===== ezpz sync smoke (polaris, CUDA) ====="
date
echo "node0:  $(hostname)"
echo "tree:   ${SUBMIT_DIR}"
echo "commit: $(git -C "${SUBMIT_DIR}" log --oneline -1 2>/dev/null)"
echo "venv:   ${VENV}"
echo "steps=${STEPS} seed=${SEED}"
python -c "import torch; print('torch:  ', torch.__version__)" 2>&1 | tail -1
python -c "import torch; print('cuda:   ', torch.cuda.is_available(), torch.version.cuda if hasattr(torch.version, 'cuda') else 'n/a')" 2>&1 | tail -1
echo ""

# --- Phase 1: import probe ---
echo "--- import probe ---"
python -c "
from torchtitan.experiments.ezpz.agpt.config_registry import agpt_5b_olmo3_polaris, agpt_10b_olmo3_polaris
from torchtitan.experiments.ezpz.agpt.config_registry import agpt_5b_olmo3_polaris_smoke, agpt_10b_olmo3_polaris_smoke
print('IMPORT_OK')
" 2>&1

if ! python -c "
from torchtitan.experiments.ezpz.agpt.config_registry import agpt_5b_olmo3_polaris, agpt_10b_olmo3_polaris
from torchtitan.experiments.ezpz.agpt.config_registry import agpt_5b_olmo3_polaris_smoke, agpt_10b_olmo3_polaris_smoke
print('IMPORT_OK')
" 2>&1 | grep -q "IMPORT_OK"; then
    echo "VERDICT: import_failed"
    exit 1
fi

# --- Phase 2: per-config train ---
# Use ezpz launch (handles NCCL on Polaris the same way it handles oneCCL on XPU)
OVERALL_OK=1
for MODEL in "agpt_5b_olmo3_polaris_smoke" "agpt_10b_olmo3_polaris_smoke"; do
    echo ""
    echo "--- ${MODEL}: ${STEPS} deterministic steps ---"
    ezpz launch python -m torchtitan.experiments.ezpz.train \
        --module ezpz.agpt \
        --config "${MODEL}" \
        --training.steps="${STEPS}" \
        --debug.seed="${SEED}" \
        --debug.deterministic \
        --checkpoint.no-enable \
        --metrics.enable-wandb \
        2>&1 | tee "polaris-smoke-${MODEL}.log"
    rc=${PIPESTATUS[0]}
    echo "${MODEL} rc=${rc}"
    [[ ${rc} -ne 0 ]] && OVERALL_OK=0
done

echo ""
if [[ ${OVERALL_OK} -eq 1 ]]; then
    echo "VERDICT: ok"
else
    echo "VERDICT: failed"
fi
