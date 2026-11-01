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
VENV="${SMOKE_VENV:-.venv-pytorch216}"

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
export MPICH_GPU_SUPPORT_ENABLED=0
export LD_LIBRARY_PATH="${HOME}/.local/mpi-compat:${LD_LIBRARY_PATH}"

if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "FATAL: no venv at ${VENV}/bin/activate" >&2
    exit 1
fi
source "${VENV}/bin/activate"

# Broadcast venv to compute nodes (fixes uv Python libpython.so on PALS)
# Same as submit_agpt_olmo3_polaris.sh -- required for uv-managed Python on Polaris
if [ -f "${VENV}.tar.gz" ]; then
    echo "yeet: broadcasting ${VENV}.tar.gz"
    ezpz yeet --src "${VENV}.tar.gz" 2>&1 | tail -3
else
    echo "yeet: no ${VENV}.tar.gz, falling back to rsync"
    ezpz yeet 2>&1 | tail -3
fi
# Switch to broadcast venv on compute (ezpz yeet puts it at /tmp/<venv-name>)
# Keep login venv active for now -- compute will use /tmp version

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
# On Polaris, ezpz -> mpi4py pulls Cray MPICH; test model path directly.
echo "--- import probe ---"
python -c "
import torch
print(f'torch: {torch.__version__} cuda={torch.cuda.is_available()}')
from torchtitan.models.llama3 import Llama3Model
print('Llama3Model OK')
from torchtitan.components.data import GrainDataLoader
print('Grain OK')
print('IMPORT_OK')
" 2>&1

if ! python -c "
from torchtitan.models.llama3 import Llama3Model
from torchtitan.components.data import GrainDataLoader
print('IMPORT_OK')
" 2>&1 | grep -q "IMPORT_OK"; then
    echo "VERDICT: import_failed"
    exit 1
fi
# --- Phase 2: per-config train ---
# Restore native mpi4py if the venv has the portable wheel (login-built).
# On compute, Cray MPICH is available at $MPICH_DIR but portable mpi4py
# was built against system MPI and won't find libmpi.so.
if python -c "import mpi4py" 2>&1 | grep -q "libmpi.*cannot open"; then
    echo "Fixing mpi4py: restoring native Cray MPICH build..."
    _OLD_MPI4PY="/eagle/AuroraGPT/foremans/projects/saforem2/torchtitan/.venv"
    if [ -d "${_OLD_MPI4PY}/lib/python3.12/site-packages/mpi4py" ]; then
        _VENV_SITE="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
        rm -rf "${_VENV_SITE}/mpi4py" "${_VENV_SITE}/mpi4py-4.1.2.dist-info" 2>/dev/null
        cp -r "${_OLD_MPI4PY}/lib/python3.12/site-packages/mpi4py" "${_VENV_SITE}/"
        cp -r "${_OLD_MPI4PY}/lib/python3.12/site-packages/mpi4py-4.1.2.dist-info" "${_VENV_SITE}/"
        echo "Restored native mpi4py from ${_OLD_MPI4PY}"
    fi
fi

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
