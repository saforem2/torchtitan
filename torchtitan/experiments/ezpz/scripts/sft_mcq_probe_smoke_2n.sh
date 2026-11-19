#!/bin/bash --login
#PBS -A datascience
#PBS -N sft-mcq-probe-smoke
#PBS -l walltime=00:45:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe
# 2N SFT MCQ template-learning smoke on Sunspot XPU (24 ranks, 12/tile FLAT).
# 20-step Grain-streaming probe: 40/30/20/10 CoT/tulu/MCQ/math on CPT-136800 HF
# (3.7G, 9/9 files, job 12484888). House pattern per cpt_gentle_smoke_sunspot.sh
# and _smoke_agpt2b_v2_256n_tulu_mix_n2.sh.
# Grain tokenizes on-the-fly via EZPZTokenizer (~4.5k tps/tile) — NO pre-tokenize.
set -o pipefail
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export TORCH_CPP_LOG_LEVEL=ERROR
export PALS_RPC_TIMEOUT="${PALS_RPC_TIMEOUT:-600}"
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export ftp_proxy="${ftp_proxy:-http://proxy.alcf.anl.gov:3128}"

# Base: MDS stage-3 step-154391 (HF + DCP under foremans/artifacts on Tegu),
# the config defaults. Override via env to probe another base.
export MDS154391_SFT_BASE="${MDS154391_SFT_BASE:-/lus/tegu/projects/datascience/foremans/artifacts/agpt-2b-mds-stage3-mix-step154391-hf}"
export MDS154391_SFT_DCP="${MDS154391_SFT_DCP:-/lus/tegu/projects/datascience/foremans/artifacts/agpt-2b-mds-stage3-mix-step154391-dcp/step-0}"

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
LOG_DIR="logs/sft-mcq-probe-smoke-${JOBID_SHORT:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/run.log"
echo "===== sft-mcq-probe-smoke (2N Sunspot, 24 ranks, 20 steps) =====" | tee "${LOG}"
date | tee -a "${LOG}"
echo "node0:    $(hostname)" | tee -a "${LOG}"
echo "venv:     ${VENV}" | tee -a "${LOG}"
echo "tree:     ${SUBMIT_DIR}" | tee -a "${LOG}"
echo "commit:   $(git -C "${SUBMIT_DIR}" log --oneline -1 2>/dev/null)" | tee -a "${LOG}"
echo "base HF:  ${MDS154391_SFT_BASE}" | tee -a "${LOG}"
echo "base DCP: ${MDS154391_SFT_DCP}" | tee -a "${LOG}"
python3 -c "import torch;print('torch:  ',torch.__version__)" 2>&1 | tail -1 | tee -a "${LOG}"
echo "" | tee -a "${LOG}"
echo "--- base HF check ---" | tee -a "${LOG}"
ls -lh "${MDS154391_SFT_BASE}/model-00001-of-00001.safetensors" 2>&1 | tee -a "${LOG}" || echo "WARN: HF safetensors missing at ${MDS154391_SFT_BASE}" | tee -a "${LOG}"
ls -lh "${MDS154391_SFT_DCP}/.metadata" 2>&1 | tee -a "${LOG}" || echo "WARN: DCP .metadata missing at ${MDS154391_SFT_DCP}" | tee -a "${LOG}"
echo "--- import probe ---" | tee -a "${LOG}"
mpiexec -n 1 --ppn 1 "${VENV}/bin/python" -c "
from torchtitan.experiments.ezpz.agpt.config_registry import agpt_2b_mds_sft_mcq_template_probe_streaming
cfg=agpt_2b_mds_sft_mcq_template_probe_streaming()
print('IMPORT_OK', cfg.training.steps, cfg.checkpointer.folder, cfg.dataloader.dataset.dataset.datasets[0].weight)
" 2>&1 | tee -a "${LOG}"
if ! grep -q "IMPORT_OK" "${LOG}"; then echo "VERDICT: import_failed" | tee -a "${LOG}"; exit 1; fi
echo "--- 20-step SFT MCQ template probe (Grain 40/30/20/10, no pre-tokenize) ---" | tee -a "${LOG}"
"${VENV}/bin/python" -c 'from ezpz.cli import main; main()' launch \
    --nproc 24 --nproc_per_node 12 -- \
    "${VENV}/bin/python" -m torchtitan.experiments.ezpz.train \
    --module ezpz.agpt --config agpt_2b_mds_sft_mcq_template_probe_streaming \
    --training.steps=20 \
    --checkpoint.no-enable --compile.no-enable \
    2>&1 | tee -a "${LOG_DIR}/train.log" | tee -a "${LOG}"
rc=${PIPESTATUS[0]}
echo "train rc=${rc}" | tee -a "${LOG}"
echo "--- losses ---" | tee -a "${LOG}"
grep -E "step:.*loss:|loss:" "${LOG_DIR}/train.log" 2>/dev/null | grep -v "W1103\|WARNING" | tail -20 | tee -a "${LOG}"
if [[ ${rc} -eq 0 ]]; then echo "VERDICT: ok" | tee -a "${LOG}"; else echo "VERDICT: failed rc=${rc}" | tee -a "${LOG}"; exit 1; fi
