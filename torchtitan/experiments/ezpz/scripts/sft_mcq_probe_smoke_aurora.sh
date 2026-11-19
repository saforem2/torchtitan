#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N sft-mcq-probe-aurora
#PBS -l walltime=00:45:00
#PBS -l filesystems=home:flare
#PBS -l select=2
#PBS -q debug-scaling
#PBS -j oe
# 2N SFT MCQ template-learning smoke on Aurora XPU (24 ranks, 12/tile FLAT).
# Aurora variant of sft_mcq_probe_smoke_2n.sh — same VENV/PBS adaptation as
# grain_cpt_anneal_smoke_aurora.sh (A=AuroraGPT, flare:home, debug-scaling,
# VENV /lus/flare/.../xpu-torch216-sync-5a7a622f16, LD-first SYCL/UR shim).
# 20-step Grain-streaming probe: 40/30/20/10 CoT/tulu/MCQ/math.
# Grain tokenizes on-the-fly via EZPZTokenizer (~4.5k tps/tile) — NO pre-tokenize.
#
# BASE AVAILABILITY (2026-11-18): This smoke requires an HF safetensors base
# plus a DCP counterpart (MDS154391_SFT_BASE / MDS154391_SFT_DCP). On Sunspot
# the canonical base is step-136800 HF 3.7G 9/9 (job 12484888) at
#   /lus/tegu/projects/datasets/datasets/outputs/evals/cpt-fwe/step-136800/hf
#   /home/foremans/torchtitan/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64/step-136800
#   /flare/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/public/sophiag/hf/global_step138650
# which ships pytorch_model.bin (MDS Megatron format, no safetensors) and has
# no converted DCP at outputs/checkpoints/agpt-2b-mds-gs138650/step-0 on flare.
# Flare also has torchtitan-ezpz cooldown converts at
#   torchtitan-ezpz/outputs/evals/agpt-2b-mds-cooldown/cooldown-{3,4,5}/step-{52650,72500,92400}/hf
# (3.7G safetensors, proper 256k vocab) but their step-*/ parents contain only
# hf/ + results/ — no .metadata DCP — and they are on an older cooldown lineage
# (not the agpt-2b-cpt-fineweb-edu-n4-gbs64 / step-136800 SFT chain), so they
# also fail config_registry.py:712-714. See ledger 13-aurora-smokes.md SFT Why(b).
# Until a safetensors HF + DCP pair is staged on flare, this script validates only the
# import/config path when a base IS supplied (via env), and otherwise exits
# with a documented Sunspot-only verdict. See docs/run-ledger/13-aurora-smokes.md SFT section.
set -o pipefail
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export TORCH_CPP_LOG_LEVEL=ERROR
export PALS_RPC_TIMEOUT="${PALS_RPC_TIMEOUT:-600}"
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export ftp_proxy="${ftp_proxy:-http://proxy.alcf.anl.gov:3128}"

# Base — override via env to point at an Aurora-local HF/DCP when staged.
# Default intentionally empty on Aurora so the base-check below fires with a
# clear message; callers who have staged a base set these to that path.
# Example once converted: export MDS154391_SFT_BASE=/lus/flare/.../step-136800/hf
export MDS154391_SFT_BASE="${MDS154391_SFT_BASE:-}"
export MDS154391_SFT_DCP="${MDS154391_SFT_DCP:-}"

VENV="/lus/flare/projects/AuroraGPT/foremans/venvs/xpu-torch216-sync-5a7a622f16"
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
LOG_DIR="logs/sft-mcq-probe-smoke-aurora-${JOBID_SHORT:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/run.log"
echo "===== sft-mcq-probe-smoke-aurora (2N Aurora PVC, 24 ranks, 20 steps) =====" | tee "${LOG}"
date | tee -a "${LOG}"
echo "node0:    $(hostname)" | tee -a "${LOG}"
echo "venv:     ${VENV}" | tee -a "${LOG}"
echo "tree:     ${SUBMIT_DIR}" | tee -a "${LOG}"
echo "commit:   $(git -C "${SUBMIT_DIR}" log --oneline -1 2>/dev/null)" | tee -a "${LOG}"
echo "base HF:  ${MDS154391_SFT_BASE:-<unset>}" | tee -a "${LOG}"
echo "base DCP: ${MDS154391_SFT_DCP:-<unset>}" | tee -a "${LOG}"
python3 -c "import torch;print('torch:  ',torch.__version__)" 2>&1 | tail -1 | tee -a "${LOG}"
echo "" | tee -a "${LOG}"

# --- base HF check: document Aurora unavailability ---
echo "--- base HF check ---" | tee -a "${LOG}"
_have_base=1
if [[ -z "${MDS154391_SFT_BASE}" ]]; then
    echo "INFO: MDS154391_SFT_BASE unset — no Aurora-local HF staged." | tee -a "${LOG}"
    _have_base=0
elif [[ ! -f "${MDS154391_SFT_BASE}/model-00001-of-00001.safetensors" ]]; then
    if [[ -f "${MDS154391_SFT_BASE}/pytorch_model.bin" ]]; then
        echo "INFO: Base at ${MDS154391_SFT_BASE} is MDS pytorch_model.bin, not safetensors — not usable by Grain SFT (needs model-00001-of-00001.safetensors)." | tee -a "${LOG}"
    else
        echo "INFO: HF safetensors missing at ${MDS154391_SFT_BASE}" | tee -a "${LOG}"
        ls -lh "${MDS154391_SFT_BASE}/" 2>&1 | head -20 | tee -a "${LOG}"
    fi
    _have_base=0
else
    ls -lh "${MDS154391_SFT_BASE}/model-00001-of-00001.safetensors" 2>&1 | tee -a "${LOG}"
fi
if [[ -n "${MDS154391_SFT_DCP}" ]]; then
    ls -lh "${MDS154391_SFT_DCP}/.metadata" 2>&1 | tee -a "${LOG}" || echo "WARN: DCP .metadata missing at ${MDS154391_SFT_DCP}" | tee -a "${LOG}"
fi
# Known Aurora candidates (informational).
echo "--- Aurora candidate probes ---" | tee -a "${LOG}"
for _cand in \
    "/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan/AuroraGPT-2B-sophiag-gs138650" \
    "/flare/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/public/sophiag/hf/global_step138650" \
    "/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan/outputs/checkpoints/agpt-2b-mds-gs138650/step-0" \
; do
    if [[ -e "${_cand}" ]]; then ls -lh "${_cand}" 2>&1 | head -5 | tee -a "${LOG}"; else echo "  absent: ${_cand}" | tee -a "${LOG}"; fi
done

if [[ ${_have_base} -eq 0 ]]; then
    cat <<'MSG' | tee -a "${LOG}"
--- Aurora SFT base unavailable (expected) ---
This smoke is Sunspot-only until an Aurora-local HF+DCP pair is staged.
The script is syntactically valid and PBS-correct (A=AuroraGPT, flare:home,
debug-scaling, xpu-torch216-sync LD-first) — it intentionally stops here
with VERDICT: skipped_no_base rather than failing at import.
Next Steps: run on Sunspot post-maintenance, or stage a base on flare
(see ledger 13-aurora-smokes.md SFT section for staging recipe), then
re-submit with MDS154391_SFT_BASE/DCP set.
MSG
    echo "VERDICT: skipped_no_base" | tee -a "${LOG}"
    exit 0
fi

echo "--- import probe ---" | tee -a "${LOG}"
mpiexec -n 1 --ppn 1 "${VENV}/bin/python" -c "
from torchtitan.experiments.ezpz.agpt.config_registry import agpt_2b_mds_sft_mcq_template_probe_streaming
cfg=agpt_2b_mds_sft_mcq_template_probe_streaming()
print('IMPORT_OK', cfg.training.steps, cfg.checkpointer.folder, cfg.dataloader.dataset.dataset.datasets[0].weight)
" 2>&1 | tee -a "${LOG}"
if ! grep -q "IMPORT_OK" "${LOG}"; then echo "VERDICT: import_failed" | tee -a "${LOG}"; exit 1; fi
# The config's production batch is 6144 sequences/step: 128 microbatches per
# rank at 24 ranks, ~20 min/step. One 2048-token microbatch per rank instead.
echo "--- 20-step SFT MCQ template probe (Grain 40/30/20/10, no pre-tokenize) ---" | tee -a "${LOG}"
"${VENV}/bin/python" -c 'from ezpz.cli import main; main()' launch \
    --nproc 24 --nproc_per_node 12 -- \
    "${VENV}/bin/python" -m torchtitan.experiments.ezpz.train \
    --module ezpz.agpt --config agpt_2b_mds_sft_mcq_template_probe_streaming \
    --training.steps=20 \
    --training.num-tokens-per-train-step=49152 \
    --checkpoint.no-enable --compile.no-enable \
    2>&1 | tee -a "${LOG_DIR}/train.log" | tee -a "${LOG}"
rc=${PIPESTATUS[0]}
echo "train rc=${rc}" | tee -a "${LOG}"
echo "--- losses ---" | tee -a "${LOG}"
grep -E "step:.*loss:|loss:" "${LOG_DIR}/train.log" 2>/dev/null | grep -v "W1103\|WARNING" | tail -20 | tee -a "${LOG}"
if [[ ${rc} -eq 0 ]]; then echo "VERDICT: ok" | tee -a "${LOG}"; else echo "VERDICT: failed rc=${rc}" | tee -a "${LOG}"; exit 1; fi
