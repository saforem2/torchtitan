#!/bin/bash --login
#PBS -A datascience
#PBS -N ezpz-sync-smoke-sirius
#PBS -l walltime=00:30:00
#PBS -l filesystems=home
#PBS -l select=2
#PBS -q workq
#PBS -j oe

# Sirius smoke: debugmodel_local on c4_test (Grain, bundled) -- no tegu, no
# BlendCorpus, no network. Validates the model/attention/optimizer path on
# Sirius' A100/CUDA stack (same as Polaris). Uses --debug.deterministic so
# losses are reproducible; c4_test is 2000 rows of real text, so the loss
# trajectory is meaningful (unlike synthetic data).

set -euo pipefail

export TORCH_CPP_LOG_LEVEL=ERROR
export PALS_RPC_TIMEOUT="${PALS_RPC_TIMEOUT:-600}" 
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"

STEPS="${SMOKE_STEPS:-3}"
SEED="${SEED:-42}"
VENV="${SMOKE_VENV:-.venv-pytorch216}"

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

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
export LD_LIBRARY_PATH="/opt/cray/pe/mpich/9.1.0/ofi/nvidia/23.3/lib:${HOME}/.local/mpi-compat:${LD_LIBRARY_PATH}"

if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "FATAL: no venv at ${VENV}/bin/activate" >&2
    exit 1
fi
source "${VENV}/bin/activate"

if [ -f "${VENV}.tar.gz" ]; then
    echo "yeet: broadcasting ${VENV}.tar.gz"
    if ! ezpz yeet --src "${VENV}.tar.gz" 2>&1 | tee yeet.log | tail -5; then
        echo "yeet: FAILED (non-fatal, continuing with login venv) -- see yeet.log"
        cat yeet.log 2>&1 | tail -20
    fi
else
    echo "yeet: no tarball, using login venv directly (2N, no broadcast needed)"
fi
# Validate that the venv is usable either way (login venv or broadcast /tmp copy)
python -c "import ezpz; print('ezpz import OK after yeet stage')" 2>&1 | tail -2

echo "===== ezpz sync smoke (sirius, CUDA, c4_test) ====="
date
echo "node0:  $(hostname)"
echo "tree:   ${SUBMIT_DIR}"
echo "commit: $(git -C "${SUBMIT_DIR}" log --oneline -1 2>/dev/null)"
echo "venv:   ${VENV}"
echo "steps=${STEPS} seed=${SEED}"
python -c "import torch; print('torch:  ', torch.__version__)" 2>&1 | tail -1
python -c "import torch; print('cuda:   ', torch.cuda.is_available(), torch.version.cuda if hasattr(torch.version, 'cuda') else 'n/a')" 2>&1 | tail -1
echo ""

echo "--- import probe ---"
python -c "
import torch
print(f'torch: {torch.__version__} cuda={torch.cuda.is_available()}')
from torchtitan.models.llama3 import Llama3Model
print('Llama3Model OK')
from torchtitan.components.data import GrainDataLoader
print('Grain OK')
from torchtitan.experiments.ezpz.agpt.config_registry import agpt_debugmodel_local
cfg = agpt_debugmodel_local()
print(f'agpt_debugmodel_local OK: {cfg.dataloader.dataset}')
print('IMPORT_OK')
" 2>&1

if ! python -c "
from torchtitan.experiments.ezpz.agpt.config_registry import agpt_debugmodel_local
cfg = agpt_debugmodel_local()
print('IMPORT_OK')
" 2>&1 | grep -q "IMPORT_OK"; then
    echo "VERDICT: import_failed"
    exit 1
fi

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

echo "--- agpt_debugmodel_local: ${STEPS} deterministic steps on c4_test ---"
ezpz launch python -m torchtitan.experiments.ezpz.train \
    --module ezpz.agpt \
    --config agpt_debugmodel_local \
    --training.steps="${STEPS}" \
    --debug.seed="${SEED}" \
    --debug.deterministic \
    --checkpoint.no-enable \
    --metrics.enable-wandb \
    2>&1 | tee "sirius-smoke-agpt_debugmodel_local.log"
rc=${PIPESTATUS[0]}
echo "agpt_debugmodel_local rc=${rc}"

echo ""
if [[ ${rc} -eq 0 ]]; then
    echo "VERDICT: ok"
else
    echo "VERDICT: failed"
    exit 1
fi
