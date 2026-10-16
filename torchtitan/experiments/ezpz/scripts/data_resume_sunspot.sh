#!/bin/bash --login
#PBS -A datascience
#PBS -N ezpz-data-resume-sunspot
#PBS -l walltime=01:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe

# Save-then-restart check for BlendCorpus data position, Sunspot XPU, 2 nodes.
#
# Verifies on a merged tree that
#   1. every save stores dataloader.consumed_samples == step * global batch,
#   2. a restart restores it (resumed loss == uninterrupted loss at the same
#      step, deterministic seed),
#   3. a checkpoint whose counter is 0 resumes at --dataloader.resume-consumed-samples
#      and at BLENDCORPUS_RESUME_CONSUMED_SAMPLES, and replays from 0 without either,
#   4. a nonzero stored counter wins over a conflicting override.
#
# Arms (agpt_debugmodel, dataset=blendcorpus, 24 ranks, 512 tokens/rank, GBS=24):
#   A  ref      6 steps, no checkpoint
#   A2 ref_ckpt 6 steps, save every 2 (uninterrupted reference for C/E/F/H)
#   A3 repeat   A2 again (run-to-run reproducibility)
#   B  save     4 steps, save every 2
#   C  restart  from B step-4, 6 steps
#   E  flag     counter-0 copy of B step-4, flag =96
#   F  env      counter-0 copy of B step-4, env var =96
#   G  none     counter-0 copy of B step-4, no override (negative control)
#   H  mismatch nonzero copy of B step-4, flag =48 (must be ignored)
#
# Env knobs: SMOKE_VENV, SEED (default 42), OUT (default outputs/data-resume-<jobid>),
# DATA_CACHE (blendcorpus index cache, default <OUT>/blendcorpus-cache).

set -o pipefail

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export TORCH_CPP_LOG_LEVEL=ERROR
# 24 ranks cold-importing torch+torchao+torchtitan from Lustre can exceed the
# 120 s PALS start-RPC default ("start RPC timeout ... after 120s", rc=127,
# job 12481007) before any rank reaches PMI init.
export PALS_RPC_TIMEOUT="${PALS_RPC_TIMEOUT:-600}"
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"

SEED="${SEED:-42}"
# Resolve the venv path. `/flare` is a symlink to `/lus/flare/projects`;
# canonicalizing keeps log lines unambiguous. NOTE: an earlier version of this
# comment claimed the unresolved prefix CAUSED an "argument matching returned
# error" from mpiexec. That was wrong -- 8831657 used the resolved path and hit
# the identical failure. The real cause of that failure is still unknown; see
# docs/experiments/sync84-xpu-smoke.md.
VENV="${SMOKE_VENV:-/lus/tegu/projects/datascience/foremans/venvs/xpu-torch216-sync-5a7a622f16}"
VENV="$(readlink -f "${VENV}")"

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source "${SUBMIT_DIR}/torchtitan/experiments/ezpz/scripts/load_pinned_ezpz_utils.sh"
load_pinned_ezpz_utils || exit $?
ezpz_setup_job
ezpz_load_modules
cd "${SUBMIT_DIR}" || exit 2

# A venv whose torch wheel is newer than the loaded oneAPI module needs its own
# Unified Runtime / SYCL libraries ahead of the module's on LD_LIBRARY_PATH.
# Measured on torch 2.16.0.dev20261015+xpu against oneapi/release/2026.1.0
# (jobs 12481003, 12481004): `import torch` dies with
#   libsycl.so.9: undefined symbol: urGraphGetIdExp, version LIBUR_LOADER_0.12
# because the module's older libur_loader is found first. Putting the whole
# venv lib/ first fixes it but also shadows the system MPI/libfabric, so link
# only the UR/SYCL/UMF/TCM set into a shared-filesystem directory and prepend
# that. Harmless for venvs that already match the module (same files).
if [[ "${SMOKE_VENV_LD_FIRST:-1}" == "1" && -d "${VENV}/lib" ]]; then
    _ld_first="${SUBMIT_DIR}/.cache/xpu-ld-first/$(basename "${VENV}")"
    if [[ ! -e "${_ld_first}/.complete" ]]; then
        mkdir -p "${_ld_first}"
        for _lib in "${VENV}"/lib/libur_* "${VENV}"/lib/libsycl.so* \
                    "${VENV}"/lib/libumf* "${VENV}"/lib/libtcm*; do
            [[ -e "${_lib}" ]] && ln -sf "${_lib}" "${_ld_first}/"
        done
        touch "${_ld_first}/.complete"
    fi
    export LD_LIBRARY_PATH="${_ld_first}:${LD_LIBRARY_PATH:-}"
fi

# This copied venv is intentionally bound without sourcing bin/activate: its
# activation script and console entry points retain paths from the source tree.
export VIRTUAL_ENV="${VENV}"
export PATH="${VENV}/bin:/opt/pbs/bin:${PATH}"
hash -r
[[ "$(command -v python)" == "${VENV}/bin/python" ]] || exit 93

JOBID_SHORT="${PBS_JOBID%%.*}"
LOG_DIR="logs/ezpz-data-resume-sunspot-${JOBID_SHORT:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/run.log"
OUT="${OUT:-$(pwd)/outputs/data-resume-${JOBID_SHORT}}"
SEED="${SEED:-42}"
mkdir -p "${OUT}"
# Private blendcorpus index cache: never share with a production chain.
CACHE="${DATA_CACHE:-${OUT}/blendcorpus-cache}"
DATALIST="$(pwd)/torchtitan/experiments/ezpz/data-lists/sunspot/books.txt"
PY="${VENV}/bin/python"
TOOL="torchtitan/experiments/ezpz/scripts/dcp_dataloader_counter.py"
GBS=24   # 24 ranks x 1 sequence (512 tokens) x GAS 1
RESUME_AT=$((4 * GBS))

{
echo "===== ezpz data-resume (sunspot) ====="
date
echo "node0:  $(hostname)"
echo "venv:   ${VENV}"
echo "tree:   ${SUBMIT_DIR}"
echo "commit: $(git -C "${SUBMIT_DIR}" log --oneline -1 2>/dev/null)"
echo "out:    ${OUT}"
"${PY}" -c "import torch;print('torch:  ',torch.__version__)" 2>&1 | tail -1
} | tee "${LOG}"

# run_arm LABEL STEPS CKPT_FOLDER [extra cli ...]  (env passes through)
run_arm() {
    local label="$1" steps="$2" folder="$3"
    shift 3
    echo "" | tee -a "${LOG}"
    echo "--- arm ${label}: steps=${steps} ckpt=${folder:-none} extra=$* ---" | tee -a "${LOG}"
    local ckpt=(--checkpoint.no-enable)
    [[ -n "${folder}" ]] && ckpt=(--checkpoint.enable --checkpoint.interval=2 "--checkpoint.folder=${folder}")
    "${PY}" -c 'from ezpz.cli import main; main()' launch \
        --nproc 24 --nproc_per_node 12 -- \
        "${PY}" -m torchtitan.experiments.ezpz.train \
        --module=ezpz.agpt --config=agpt_debugmodel \
        --training.steps="${steps}" \
        --training.max-context-length=512 \
        --training.num-tokens-per-microbatch-per-dp-rank=512 \
        --dataloader.dataset=blendcorpus \
        "--dataloader.dataset-path=${DATALIST}" \
        "--dataloader.data-cache-path=${CACHE}" \
        --debug.seed="${SEED}" --debug.deterministic \
        --metrics.enable-wandb \
        "${ckpt[@]}" "$@" \
        2>&1 | tee "${LOG_DIR}/${label}.log" | tail -n +1 >> "${LOG}"
    local rc=${PIPESTATUS[0]}
    echo "arm ${label} rc=${rc}" | tee -a "${LOG}"
    echo "  losses: $(grep -hoE 'step: +[0-9]+ +loss: +[0-9.]+' "${LOG_DIR}/${label}.log" | tr -s ' ' | tr '\n' ';')" | tee -a "${LOG}"
    return "${rc}"
}

# Copy a step dir with a modified counter into a fresh checkpoint folder.
copy_step() { # SRC_STEP_DIR DST_FOLDER zero|keep
    mkdir -p "$2"
    if [[ "$3" == "zero" ]]; then
        "${PY}" "${TOOL}" zero "$1" "$2/$(basename "$1")"
    else
        cp -r "$1" "$2/"
    fi
}

run_arm A_ref 6 ""                    || true
run_arm A2_ref_ckpt 6 "${OUT}/A2"     || true
run_arm A3_ref_repeat 6 "${OUT}/A3"   || true
run_arm B_save 4 "${OUT}/B"           || true

echo "" | tee -a "${LOG}"
echo "--- counters after B (expect step-2 = $((2*GBS)), step-4 = $((4*GBS))) ---" | tee -a "${LOG}"
ls -1d "${OUT}"/B/step-* 2>&1 | tee -a "${LOG}"
"${PY}" "${TOOL}" show "${OUT}"/B/step-* 2>&1 | tee -a "${LOG}"

copy_step "${OUT}/B/step-4" "${OUT}/E" zero 2>&1 | tee -a "${LOG}"
copy_step "${OUT}/B/step-4" "${OUT}/F" zero 2>&1 | tee -a "${LOG}"
copy_step "${OUT}/B/step-4" "${OUT}/G" zero 2>&1 | tee -a "${LOG}"
copy_step "${OUT}/B/step-4" "${OUT}/H" keep 2>&1 | tee -a "${LOG}"

run_arm C_restart 6 "${OUT}/B"        || true
run_arm E_flag 6 "${OUT}/E" "--dataloader.resume-consumed-samples=${RESUME_AT}" || true
BLENDCORPUS_RESUME_CONSUMED_SAMPLES="${RESUME_AT}" run_arm F_env 6 "${OUT}/F" || true
run_arm G_none 6 "${OUT}/G"           || true
run_arm H_mismatch 6 "${OUT}/H" "--dataloader.resume-consumed-samples=$((2 * GBS))" || true

echo "" | tee -a "${LOG}"
echo "--- counters after restarts (expect step-6 = $((6*GBS)) in B, E, F, H; G replays from 0) ---" | tee -a "${LOG}"
"${PY}" "${TOOL}" show "${OUT}"/B/step-6 "${OUT}"/E/step-6 "${OUT}"/F/step-6 "${OUT}"/G/step-6 "${OUT}"/H/step-6 2>&1 | tee -a "${LOG}"

echo "" | tee -a "${LOG}"
echo "--- resume log lines ---" | tee -a "${LOG}"
grep -H "resuming the data stream\|Ignoring --dataloader.resume\|Finished loading the checkpoint\|Loading the checkpoint" "${LOG_DIR}"/*.log 2>/dev/null | cut -c1-260 | sort -u | tee -a "${LOG}"
echo "DONE" | tee -a "${LOG}"
