#!/bin/bash --login
#PBS -A datascience
#PBS -N cpt-weighted-smoke
#PBS -l walltime=00:45:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe
# Weighted CPT smoke: agpt_2b_mds_mix_owm_edu_7525 (max_num_documents fix).
# Mirrors sync_smoke_sunspot.sh house pattern (ZE_FLAT, LD-first, venv).
# Must qsub FROM checkout so PBS_O_WORKDIR correct (12481940 held H bug).
set -o pipefail
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export TORCH_CPP_LOG_LEVEL=ERROR
export PALS_RPC_TIMEOUT="${PALS_RPC_TIMEOUT:-600}"
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export ftp_proxy="${ftp_proxy:-http://proxy.alcf.anl.gov:3128}"
VENV="/lus/tegu/projects/datascience/foremans/venvs/xpu-torch216-sync-5a7a622f16"
VENV="$(readlink -f "${VENV}")"
SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
export MDS_ANNEAL_BASE=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/checkpoints/agpt-2b-mds-gs138650/step-0
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
LOG_DIR="logs/cpt-weighted-smoke-${JOBID_SHORT:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/run.log"
echo "===== cpt-weighted-smoke agpt_2b_mds_mix_owm_edu_7525 (2N Sunspot, 24 ranks HSDP) =====" | tee "${LOG}"
date | tee -a "${LOG}"
echo "node0:  $(hostname)" | tee -a "${LOG}"
echo "venv:   ${VENV}" | tee -a "${LOG}"
echo "tree:   ${SUBMIT_DIR}" | tee -a "${LOG}"
echo "commit: $(git -C "${SUBMIT_DIR}" log --oneline -1 2>/dev/null)" | tee -a "${LOG}"
python3 -c "import torch;print('torch:  ',torch.__version__)" 2>&1 | tail -1 | tee -a "${LOG}"
echo "" | tee -a "${LOG}"
echo "--- import probe (max_num_documents) ---" | tee -a "${LOG}"
mpiexec -n 1 --ppn 1 "${VENV}/bin/python" -c "
from torchtitan.experiments.ezpz.agpt.config_registry import agpt_2b_mds_mix_owm_edu_7525
cfg = agpt_2b_mds_mix_owm_edu_7525()
print('CONFIG_OK', cfg.dataloader.__class__.__name__, cfg.checkpointer.folder)
assert hasattr(cfg.dataloader, 'max_num_documents'), 'missing max_num_documents'
print('IMPORT_OK', cfg.dataloader.dataset.__class__.__name__)
" 2>&1 | tee -a "${LOG}"
if ! grep -q "IMPORT_OK" "${LOG}"; then echo "VERDICT: import_failed" | tee -a "${LOG}"; exit 1; fi
echo "--- 3-step weighted CPT smoke (owm 75% + finemath4plus 25%) ---" | tee -a "${LOG}"
"${VENV}/bin/python" -c 'from ezpz.cli import main; main()' launch \
    --nproc 24 --nproc_per_node 12 -- \
    "${VENV}/bin/python" -m torchtitan.experiments.ezpz.train \
    --module ezpz.agpt --config agpt_2b_mds_mix_owm_edu_7525 \
    --training.steps=3 --debug.seed=42 --debug.deterministic \
    --parallelism.data-parallel-shard-degree=1 --parallelism.data-parallel-replicate-degree=-1 \
    --checkpoint.no-enable --compile.no-enable --validator.no-enable \
    activation-checkpoint:full \
    2>&1 | tee -a "${LOG_DIR}/train.log" | tee -a "${LOG}"
rc=${PIPESTATUS[0]}
echo "train rc=${rc}" | tee -a "${LOG}"
echo "--- losses ---" | tee -a "${LOG}"
grep -E "step:.*loss:|loss:" "${LOG_DIR}/train.log" 2>/dev/null | grep -v "W1103\|WARNING" | tail -20 | tee -a "${LOG}"
if [[ ${rc} -eq 0 ]]; then echo "VERDICT: ok" | tee -a "${LOG}"; else echo "VERDICT: failed rc=${rc}" | tee -a "${LOG}"; exit 1; fi
