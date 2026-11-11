#!/bin/bash --login
# Post-sync / pre-merge train smoke for ezpz on any ALCF PBS system.
#
# Called by the per-machine wrappers (sync_smoke_{sunspot,aurora,polaris,
# sirius}.sh), which only carry the #PBS headers. Can also be run directly
# inside an allocation:  bash torchtitan/experiments/ezpz/scripts/sync_smoke_alcf.sh
#
# Phases:
#   1. import probe on one rank (fail fast)
#   2. seeded deterministic train steps per config, each under a timeout
#   3. per-config losses (ANSI stripped) and one machine-greppable VERDICT
#
# VERDICT values:
#   ok                       every config passed
#   ok_with_known_failures   only configs matching a KNOWN_FAILURES signature
#                            failed (listed); still a pass for merge purposes
#   failed (label=rc ...)    anything else; rc 124 means the config timed out
#   import_failed / dirty_source / setup_failed
#
# Env knobs (all optional):
#   SMOKE_MACHINE    sunspot|aurora|polaris|sirius (default: from hostname)
#   SMOKE_VENV       venv root (default: per machine, below)
#   SMOKE_HF_ASSETS  tokenizer dir passed as --hf-assets-path (default: per machine)
#   SMOKE_CONFIGS    newline-separated "module:config[:extra cli ...]"
#   SMOKE_STEPS      steps per config (default 6)
#   SMOKE_CONFIG_TIMEOUT  seconds per config before it is killed (default 900)
#   SMOKE_WANDB      1 to log to W&B (default 0)
#   SMOKE_RESUME     1 (default) to run the exact checkpoint-resume check
#   SMOKE_RESUME_AT  checkpoint step (default 3); SMOKE_RESUME_TOL (default 1e-3)
#   SMOKE_RESUME_EXTRA  extra flags for both resume runs ("{NPROC}" substituted)
#   SEED             debug seed (default 42)

set -o pipefail

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
cd "${SUBMIT_DIR}" || { echo "VERDICT: setup_failed (cd ${SUBMIT_DIR})"; exit 2; }

# --- machine detection and per-machine defaults ---
_fqdn="$(hostname -f 2>/dev/null || hostname)"
MACHINE="${SMOKE_MACHINE:-}"
if [[ -z "${MACHINE}" ]]; then
    case "${_fqdn}" in
        *sunspot*) MACHINE=sunspot ;;
        *aurora*) MACHINE=aurora ;;
        *polaris*) MACHINE=polaris ;;
        *sirius*) MACHINE=sirius ;;
        *) echo "VERDICT: setup_failed (unknown machine ${_fqdn}; set SMOKE_MACHINE)"; exit 2 ;;
    esac
fi

case "${MACHINE}" in
    sunspot)
        ACCEL=xpu; PPN=12
        DEF_VENV=/lus/tegu/projects/datascience/foremans/venvs/xpu-torch216-sync-5a7a622f16
        DEF_ASSETS=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/assets/hf/gemma-7b ;;
    aurora)
        ACCEL=xpu; PPN=12
        DEF_VENV=/flare/AuroraGPT/foremans/venvs/xpu-torch216-sync-5a7a622f16
        DEF_ASSETS=/flare/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz/assets/hf/gemma-7b ;;
    polaris)
        ACCEL=cuda; PPN=4
        DEF_VENV=/eagle/AuroraGPT/foremans/venvs/cuda-torch215-cu132
        DEF_ASSETS=/eagle/AuroraGPT/foremans/projects/saforem2/torchtitan/assets/hf/gemma-7b ;;
    sirius)
        ACCEL=cuda; PPN=4
        DEF_VENV=/home/foremans/torchtitan/.venv
        # Sirius mounts tegu but not Sunspot's project directories.
        DEF_ASSETS=/home/foremans/assets/hf/gemma-7b
        # The default utils pin predates ezpz_load_modules_sirius (it was a
        # no-op), so no MPI module loads and mpi4py cannot find libmpi.
        # saforem2/ezpz 4a2aba37 delegates Sirius to the Polaris loader.
        export EZPZ_UTILS_PIN_COMMIT="${EZPZ_UTILS_PIN_COMMIT:-4a2aba376eb855004baeaebc5f953cc2a95b1850}"
        export EZPZ_UTILS_PIN_SHA256="${EZPZ_UTILS_PIN_SHA256:-12c2e056d45c8a4ce15f5097f83eda86caa08353ac645cde0133c9087a3cc8cd}" ;;
    *) echo "VERDICT: setup_failed (unsupported SMOKE_MACHINE=${MACHINE})"; exit 2 ;;
esac

VENV="$(readlink -f "${SMOKE_VENV:-${DEF_VENV}}")"
HF_ASSETS="${SMOKE_HF_ASSETS:-${DEF_ASSETS}}"
STEPS="${SMOKE_STEPS:-6}"
SEED="${SEED:-42}"
CONFIG_TIMEOUT="${SMOKE_CONFIG_TIMEOUT:-900}"

# max_context_length must EQUAL num-tokens-per-microbatch-per-dp-rank: the
# blendcorpus fold emits fixed-length rows and the SDPA wrapper divides by it.
_SEQ="--training.max-context-length=512 --training.num-tokens-per-microbatch-per-dp-rank=512"
DEFAULT_CONFIGS=(
    "ezpz.agpt:agpt_debugmodel:${_SEQ}"
    # TP=2 is the arm that catches the local_map/SPMD contract breaks
    # (#4533 class), which only fire under TP>1. It currently fails on ezpz
    # with the KNOWN_FAILURES signature below; keep it so a change in how it
    # fails (or a fix) is visible.
    "ezpz.agpt:agpt_debugmodel:${_SEQ} --parallelism.tensor-parallel-degree=2"
    "ezpz.moe:moe_debugmodel:${_SEQ}"
)
# label-suffix|regex. Open bug: at TP>1 the attention local_map does not
# redistribute, so the SDPA wrapper receives sequence-sharded tokens
# (256 of 512 at TP=2). See docs/guides/known-bugs/moe-tp2-wo-placement.md.
KNOWN_FAILURES=(
    "tensor-parallel-degree=2|token count [0-9]+ is not a multiple of max_context_length"
)

if [[ -n "${SMOKE_CONFIGS:-}" ]]; then
    CONFIGS=()
    while IFS= read -r _line; do
        [[ -n "$_line" ]] && CONFIGS+=("$_line")
    done <<< "${SMOKE_CONFIGS}"
else
    CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

[[ -x "${VENV}/bin/python" ]] || { echo "VERDICT: setup_failed (no venv at ${VENV})"; exit 2; }
[[ -s "${HF_ASSETS}/tokenizer.json" || -s "${HF_ASSETS}/tokenizer.model" ]] || {
    echo "VERDICT: setup_failed (no tokenizer in ${HF_ASSETS}; set SMOKE_HF_ASSETS)"
    exit 2
}

# --- environment ---
export TORCH_CPP_LOG_LEVEL=ERROR
export PALS_RPC_TIMEOUT="${PALS_RPC_TIMEOUT:-600}"
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
if [[ "${ACCEL}" == xpu ]]; then
    export ZE_FLAT_DEVICE_HIERARCHY=FLAT
    export CCL_PROCESS_LAUNCHER=pmix
    export CCL_OP_SYNC=1
fi

source "${SUBMIT_DIR}/torchtitan/experiments/ezpz/scripts/load_pinned_ezpz_utils.sh"
load_pinned_ezpz_utils || { echo "VERDICT: setup_failed (ezpz utils)"; exit 2; }
ezpz_setup_job || { echo "VERDICT: setup_failed (ezpz_setup_job)"; exit 2; }
ezpz_load_modules || { echo "VERDICT: setup_failed (ezpz_load_modules)"; exit 2; }
cd "${SUBMIT_DIR}" || exit 2

if [[ "${ACCEL}" == cuda ]]; then
    # After the module loader, which sets MPICH_GPU_SUPPORT_ENABLED=1: the
    # venv's mpi4py is not linked against Cray GTL, and MPI_Init aborts with
    # "GPU_SUPPORT_ENABLED is requested, but GTL library is not linked"
    # (Sirius job 33436). Same order as submit_agpt_olmo3_polaris.sh, which
    # also puts its libmpi.so.12 shim first.
    export MPICH_GPU_SUPPORT_ENABLED=0
    [[ -d "${HOME}/.local/mpi-compat" ]] \
        && export LD_LIBRARY_PATH="${HOME}/.local/mpi-compat:${LD_LIBRARY_PATH:-}"
fi

if [[ "${ACCEL}" == xpu && -d "${VENV}/lib" ]]; then
    # A torch wheel newer than the loaded oneAPI module needs its own Unified
    # Runtime / SYCL libraries first, or `import torch` dies with
    #   libsycl.so.9: undefined symbol: urGraphGetIdExp
    # (torch 2.16.0.dev20261015+xpu vs oneapi/release/2026.1.0, measured on
    # Sunspot jobs 12481003/12481004 and an Aurora login node 2026-11-11).
    # Link only the UR/SYCL/UMF/TCM set; the whole venv lib/ would also shadow
    # the system MPI/libfabric.
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

# Bind the venv without bin/activate: copied venvs keep stale paths in it.
export VIRTUAL_ENV="${VENV}"
export PATH="${VENV}/bin:/opt/pbs/bin:${PATH}"
hash -r
[[ "$(command -v python)" == "${VENV}/bin/python" ]] || {
    echo "VERDICT: setup_failed (python is $(command -v python), not ${VENV}/bin/python)"
    exit 2
}

NHOSTS="$(sort -u "${PBS_NODEFILE:-/dev/null}" 2>/dev/null | wc -l)"
(( NHOSTS > 0 )) || NHOSTS=1
NPROC=$((NHOSTS * PPN))

JOBID_SHORT="${PBS_JOBID%%.*}"
LOG_DIR="logs/ezpz-sync-smoke-${MACHINE}-${JOBID_SHORT:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/run.log"
strip_ansi() { sed 's/\x1b\[[0-9;]*m//g' "$@"; }

{
    echo "===== ezpz sync smoke (${MACHINE}) ====="
    date
    echo "node0:  $(hostname)"
    echo "venv:   ${VENV}"
    echo "assets: ${HF_ASSETS}"
    echo "tree:   ${SUBMIT_DIR}"
    echo "commit: $(git -C "${SUBMIT_DIR}" log --oneline -1 2>/dev/null)"
    echo "ranks:  ${NPROC} (${NHOSTS} x ${PPN})  steps=${STEPS} seed=${SEED} timeout=${CONFIG_TIMEOUT}s"
} | tee "${LOG}"
[[ -z "$(git -C "${SUBMIT_DIR}" status --porcelain --untracked-files=no 2>/dev/null)" ]] \
    || { echo "VERDICT: dirty_source" | tee -a "${LOG}"; exit 1; }
python -c "import torch;print('torch:  ',torch.__version__)" 2>&1 | tail -1 | tee -a "${LOG}"

# --- Phase 1: import probe ---
# ezpz imports mpi4py, which calls MPI_Init_thread at import; run under the
# launcher so PMIx exists.
echo "--- import probe ---" | tee -a "${LOG}"
mpiexec -n 1 --ppn 1 "${VENV}/bin/python" -c "
from torchtitan.experiments.ezpz.agpt.parallelize import parallelize_llama
from torchtitan.experiments.ezpz.agpt.config_registry import agpt_debugmodel
from torchtitan.experiments.ezpz.moe.parallelize import parallelize_moe
from torchtitan.experiments.ezpz.moe.config_registry import moe_debugmodel
from torchtitan.experiments.ezpz.trainer import FaultTolerantTrainer
print('IMPORT_OK')
" 2>&1 | tee -a "${LOG}"
grep -q "IMPORT_OK" "${LOG}" || { echo "VERDICT: import_failed" | tee -a "${LOG}"; exit 1; }

# --- Phase 2: per-config train smoke ---
WANDB=(--metrics.no-enable-wandb)
[[ "${SMOKE_WANDB:-0}" == 1 ]] && WANDB=(--metrics.enable-wandb)
declare -a FAILED=() KNOWN=()
idx=0
for spec in "${CONFIGS[@]}"; do
    idx=$((idx + 1))
    module="${spec%%:*}"
    rest="${spec#*:}"
    config="${rest%%:*}"
    extra="${rest#*:}"
    [[ "${extra}" == "${rest}" ]] && extra=""
    label="${idx}-${config}"
    clog="${LOG_DIR}/${label}.log"

    echo "" | tee -a "${LOG}"
    echo "--- [${idx}] ${module} / ${config} ${extra:+(${extra})} ---" | tee -a "${LOG}"
    # extra is an intentional word-split argument list.
    # shellcheck disable=SC2086
    timeout --kill-after=60 "${CONFIG_TIMEOUT}" \
        "${VENV}/bin/python" -c 'from ezpz.cli import main; main()' launch \
        --nproc "${NPROC}" --nproc_per_node "${PPN}" -- \
        "${VENV}/bin/python" -m torchtitan.experiments.ezpz.train \
        --module="${module}" \
        --config="${config}" \
        --hf-assets-path="${HF_ASSETS}" \
        --training.steps="${STEPS}" \
        --debug.seed="${SEED}" \
        --debug.deterministic \
        --checkpoint.no-enable \
        "${WANDB[@]}" \
        ${extra} \
        > "${clog}" 2>&1
    rc=$?
    steps_done=$(strip_ansi "${clog}" | grep -cE 'step: +[0-9]+ +loss')
    status="rc=${rc}"
    (( rc == 124 )) && status="rc=124 (timed out after ${steps_done} steps)"
    if (( rc != 0 )); then
        known=""
        for kf in "${KNOWN_FAILURES[@]}"; do
            # grep without -q: with pipefail, -q exits early, sed gets SIGPIPE
            # and the pipeline reports failure even on a match.
            if [[ "${extra}" == *"${kf%%|*}"* ]] \
                && strip_ansi "${clog}" | grep -E "${kf#*|}" > /dev/null; then
                known="${kf#*|}"
            fi
        done
        if [[ -n "${known}" ]]; then
            KNOWN+=("${label}")
            status="${status} KNOWN_FAILURE"
        else
            FAILED+=("${label}=${rc}")
        fi
    fi
    echo "${label} ${status}" | tee -a "${LOG}"
    strip_ansi "${clog}" | grep -oE 'step: +[0-9]+ +loss: +[0-9.]+' \
        | awk '{printf "%s:%s ", $2, $4} END {print ""}' | tee -a "${LOG}"
done

# --- Phase 3: checkpoint resume (exact) ---
# Run config 1 for STEPS steps with a full checkpoint at RESUME_AT, copy that
# checkpoint into a fresh dump folder, resume to STEPS, and require steps
# RESUME_AT+1..STEPS to match the uninterrupted run. Both runs have the same
# training.steps, so LR schedule and data order are identical.
# SMOKE_RESUME_EXTRA adds flags to both runs ("{NPROC}" is substituted), e.g.
# native DDP + ZeRO-1:
#   --parallelism.data-parallel-shard-degree=1
#   --parallelism.data-parallel-replicate-degree={NPROC}
#   --parallelism.enable-data-parallel-native-ddp
#   --parallelism.native-ddp-zero1 --parallelism.native-ddp-zero1-allgather-dtype=bfloat16
#   --training.dtype=float32 --training.mixed-precision-param=bfloat16
#   --training.mixed-precision-reduce=float32
if [[ "${SMOKE_RESUME:-1}" == 1 ]]; then
    RESUME_AT="${SMOKE_RESUME_AT:-3}"
    RESUME_TOL="${SMOKE_RESUME_TOL:-1e-3}"
    spec="${CONFIGS[0]}"
    module="${spec%%:*}"; rest="${spec#*:}"; config="${rest%%:*}"; extra="${rest#*:}"
    [[ "${extra}" == "${rest}" ]] && extra=""
    rextra="${SMOKE_RESUME_EXTRA:-}"
    rextra="${rextra//\{NPROC\}/${NPROC}}"
    echo "" | tee -a "${LOG}"
    echo "--- resume: ${config}, checkpoint at ${RESUME_AT}, ${STEPS} steps ${rextra:+(${rextra})} ---" | tee -a "${LOG}"
    resume_run() {  # resume_run <name>
        # shellcheck disable=SC2086
        timeout --kill-after=60 "${CONFIG_TIMEOUT}" \
            "${VENV}/bin/python" -c 'from ezpz.cli import main; main()' launch \
            --nproc "${NPROC}" --nproc_per_node "${PPN}" -- \
            "${VENV}/bin/python" -m torchtitan.experiments.ezpz.train \
            --module="${module}" --config="${config}" \
            --hf-assets-path="${HF_ASSETS}" --training.steps="${STEPS}" \
            --debug.seed="${SEED}" --debug.deterministic "${WANDB[@]}" \
            --checkpoint.enable --checkpoint.interval="${RESUME_AT}" \
            --checkpoint.keep-latest-k=0 \
            --dump-folder="${LOG_DIR}/resume-$1" \
            ${extra} ${rextra} \
            > "${LOG_DIR}/resume-$1.log" 2>&1
    }
    losses() { strip_ansi "$1" | grep -oE 'step: +[0-9]+ +loss: +[0-9.]+' | awk '{print $2, $4}'; }
    resume_run cont; rc_cont=$?
    rc_r2=99
    ck="${LOG_DIR}/resume-cont/checkpoint/step-${RESUME_AT}"
    if (( rc_cont == 0 )) && [[ -d "${ck}" ]]; then
        mkdir -p "${LOG_DIR}/resume-r2/checkpoint"
        cp -r "${ck}" "${LOG_DIR}/resume-r2/checkpoint/"
        resume_run r2; rc_r2=$?
    fi
    loaded=$(strip_ansi "${LOG_DIR}/resume-r2.log" 2>/dev/null | grep -c "Loading the checkpoint.*step-${RESUME_AT}")
    verdict=$(join <(losses "${LOG_DIR}/resume-cont.log") <(losses "${LOG_DIR}/resume-r2.log" 2>/dev/null) \
        | awk -v k="${RESUME_AT}" -v n="${STEPS}" -v tol="${RESUME_TOL}" '
            $1 > k { d = $2 - $3; if (d < 0) d = -d; if (d > m) m = d; c++
                     printf "step %d cont=%s resumed=%s\n", $1, $2, $3 }
            END { want = n - k; printf "max_abs_diff=%g compared=%d/%d %s\n", m, c, want,
                  (c == want && m <= tol) ? "RESUME_OK" : "RESUME_MISMATCH" }')
    echo "${verdict}" | tee -a "${LOG}"
    echo "resume rc: cont=${rc_cont} r2=${rc_r2} loaded_step_${RESUME_AT}=${loaded}" | tee -a "${LOG}"
    if (( rc_cont != 0 || rc_r2 != 0 || loaded == 0 )) || [[ "${verdict}" != *RESUME_OK* ]]; then
        FAILED+=("resume=${rc_cont}/${rc_r2}")
    fi
fi

echo "" | tee -a "${LOG}"
if (( ${#FAILED[@]} > 0 )); then
    echo "VERDICT: failed (${FAILED[*]})" | tee -a "${LOG}"
    exit 1
elif (( ${#KNOWN[@]} > 0 )); then
    echo "VERDICT: ok_with_known_failures (${KNOWN[*]})" | tee -a "${LOG}"
else
    echo "VERDICT: ok" | tee -a "${LOG}"
fi
