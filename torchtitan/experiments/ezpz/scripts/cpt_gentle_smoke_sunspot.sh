#!/bin/bash --login
#PBS -A datascience
#PBS -N cpt-gentle-smoke
#PBS -l walltime=00:45:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe
# Gentle-LR CPT 10-step smoke on Sunspot XPU 2N (16 ranks (8/node, 11008%16==0), 12/tile FLAT).
# Forks 256N olmo-mix step-92859 at 2e-6 constant (warmup 20). House pattern.
# Data: blendcorpus/books override (HF delegate removed in 80th sync).
set -o pipefail
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export TORCH_CPP_LOG_LEVEL=ERROR
export PALS_RPC_TIMEOUT="${PALS_RPC_TIMEOUT:-600}"
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export ftp_proxy="${ftp_proxy:-http://proxy.alcf.anl.gov:3128}"
export OLMO_ANNEAL_BASE="/lus/tegu/projects/datascience/fsimini/data_sandbox/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144/step-92859"
VENV="/lus/tegu/projects/datascience/foremans/venvs/xpu-torch216-sync-5a7a622f16"
VENV="$(readlink -f "${VENV}")"
SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source "${SUBMIT_DIR}/torchtitan/experiments/ezpz/scripts/load_pinned_ezpz_utils.sh"
load_pinned_ezpz_utils || exit $?
ezpz_setup_job
ezpz_load_modules
cd "${SUBMIT_DIR}" || exit 2
if [[ "${SMOKE_VENV_LD_FIRST:-1}" == "1" && -d "${VENV}/lib" ]]; then
    _ld_first="${SUBMIT_DIR}/.cache/xpu-ld-first/$(basename "${VENV}")"
    if [[ ! -e "${_ld_first}/.complete" ]]; then
        mkdir -p "${_ld_first}"
        for _lib in "${VENV}"/lib/libur_* "${VENV}"/lib/libsycl.so* "${VENV}"/lib/libumf* "${VENV}"/lib/libtcm*; do [[ -e "${_lib}" ]] && ln -sf "${_lib}" "${_ld_first}/"; done
        touch "${_ld_first}/.complete"
    fi
    export LD_LIBRARY_PATH="${_ld_first}:${LD_LIBRARY_PATH:-}"
fi
export VIRTUAL_ENV="${VENV}"
export PATH="${VENV}/bin:/opt/pbs/bin:${PATH}"
hash -r
[[ "$(command -v python)" == "${VENV}/bin/python" ]] || exit 93
JOBID_SHORT="${PBS_JOBID%%.*}"
LOG_DIR="logs/cpt-gentle-smoke-${JOBID_SHORT:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/run.log"
echo "===== cpt-gentle-smoke (2N Sunspot, 16 ranks (8/node, 11008%16==0)) =====" | tee "${LOG}"
date | tee -a "${LOG}"
echo "node0:  $(hostname)" | tee -a "${LOG}"
echo "venv:   ${VENV}" | tee -a "${LOG}"
echo "tree:   ${SUBMIT_DIR}" | tee -a "${LOG}"
echo "commit: $(git -C "${SUBMIT_DIR}" log --oneline -1 2>/dev/null)" | tee -a "${LOG}"
echo "base:   ${OLMO_ANNEAL_BASE}" | tee -a "${LOG}"
python3 -c "import torch;print('torch:  ',torch.__version__)" 2>&1 | tail -1 | tee -a "${LOG}"
echo "" | tee -a "${LOG}"
echo "--- import probe ---" | tee -a "${LOG}"
mpiexec -n 1 --ppn 1 "${VENV}/bin/python" -c "
from torchtitan.experiments.ezpz.agpt.config_registry import _agpt_2b_olmo_anneal_base
cfg=_agpt_2b_olmo_anneal_base()
print('IMPORT_OK', cfg.checkpointer.initial_load_path, cfg.training.steps)
" 2>&1 | tee -a "${LOG}"
if ! grep -q "IMPORT_OK" "${LOG}"; then echo "VERDICT: import_failed" | tee -a "${LOG}"; exit 1; fi
echo "--- 10-step gentle-LR CPT smoke (fork step-92859 @ 2e-6, blendcorpus/books) ---" | tee -a "${LOG}"
# agpt_2b_olmo_anneal_flat uses HF delegate (open-web-math) removed in 80th sync.
# Override dataloader to blendcorpus/books on the CLI (same family, local).
"${VENV}/bin/python" -c 'from ezpz.cli import main; main()' launch \
    --nproc 16 --nproc_per_node 8 -- \
    "${VENV}/bin/python" -m torchtitan.experiments.ezpz.train \
    --module ezpz.agpt --config agpt_2b_olmo_anneal_flat \
    --training.steps=10 \
    --dataloader.dataset=blendcorpus \
    --dataloader.dataset-path=torchtitan/experiments/ezpz/data-lists/sunspot/books.txt \
    --checkpointer.interval=1000000 --compile.no-enable --training.disable-accelerator-graphs --validator.no-enable \
    2>&1 | tee -a "${LOG_DIR}/train.log" | tee -a "${LOG}"
rc=${PIPESTATUS[0]}
echo "train rc=${rc}" | tee -a "${LOG}"
echo "--- losses ---" | tee -a "${LOG}"
grep -E "step:.*loss:|loss:" "${LOG_DIR}/train.log" 2>/dev/null | grep -v "W1103\|WARNING" | tail -20 | tee -a "${LOG}"
if [[ ${rc} -eq 0 ]]; then echo "VERDICT: ok" | tee -a "${LOG}"; else echo "VERDICT: failed rc=${rc}" | tee -a "${LOG}"; exit 1; fi
