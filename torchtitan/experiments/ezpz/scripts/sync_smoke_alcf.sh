#!/bin/bash --login
# Post-sync / pre-merge train smoke for ezpz on any ALCF PBS system (and
# locally, for a fast import-only check).
#
# Called by the per-machine wrappers (sync_smoke_{sunspot,aurora,polaris,
# sirius}.sh), which only carry the #PBS headers. Can also be run directly
# inside an allocation:  bash torchtitan/experiments/ezpz/scripts/sync_smoke_alcf.sh
#
# Setup uses the canonical ezpz entry point:
#
#   source <(curl -fsSL https://ezpz.cool/utils.sh) && ezpz_setup /path/to/venv
#   ezpz launch ...
#
# `ezpz_setup <venv>` does: ezpz_setup_job + ezpz_load_modules + activate.
# It owns all the machine-specific state that used to live in this file
# (MPICH_GPU_SUPPORT_ENABLED, PALS_TRANSFER, libmpi.so.12 shim,
# xpu-ld-first, ZE_FLAT/CCL env, etc.) so we do not reimplement it.
#
# Phases:
#   1. import probe on one rank (fail fast, under the launcher so PMIx exists)
#   2. seeded deterministic train steps per config, each under a timeout
#   3. checkpoint-resume exactness (reuse config 1, compare post-resume losses)
#   4. one machine-greppable VERDICT
#
# VERDICT values:
#   ok                       every config passed
#   ok_with_known_failures   only configs matching a KNOWN_FAILURES signature
#                            failed (listed); still a pass for merge purposes
#   failed (label=rc ...)    anything else; rc 124 means the config timed out
#   import_failed / dirty_source / setup_failed
#
# Env knobs (all optional):
#   SMOKE_VENV       venv root (default: .venv in this checkout)
#   SMOKE_HF_ASSETS  tokenizer dir passed as --hf-assets-path (default: vendored)
#   SMOKE_CONFIGS    newline-separated "module:config[:extra cli ...]"
#   SMOKE_STEPS      steps per config (default 6)
#   SMOKE_CONFIG_TIMEOUT  seconds per config before it is killed (default 900)
#   SMOKE_WANDB      1 to log to W&B (default 0)
#   SMOKE_RESUME     1 (default) to run the exact checkpoint-resume check
#   SMOKE_RESUME_AT  checkpoint step (default 3); SMOKE_RESUME_TOL (default 1e-3)
#   SMOKE_RESUME_EXTRA  extra flags for both resume runs ("{NPROC}" substituted)
#   SEED             debug seed (default 42)
#   SMOKE_CUDA_GRAPHS  0 to run eager on CUDA (default 1: upstream accelerator graph capture)

set -o pipefail

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
cd "${SUBMIT_DIR}" || { echo "VERDICT: setup_failed (cd ${SUBMIT_DIR})"; exit 2; }

# --- defaults (repo-relative; work everywhere, including macOS) ---
DEF_VENV="${SMOKE_VENV:-${SUBMIT_DIR}/.venv}"
DEF_ASSETS="${SMOKE_HF_ASSETS:-${SUBMIT_DIR}/tests/assets/tokenizer}"

VENV="$(readlink -f "${DEF_VENV}" 2>/dev/null || printf '%s' "${DEF_VENV}")"
# HF assets may legitimately not exist (blendcorpus path); don't hard-fail
# here — let the caller skip --hf-assets-path or fail loud inside the config.
HF_ASSETS="${DEF_ASSETS}"
STEPS="${SMOKE_STEPS:-6}"
SEED="${SEED:-42}"
CONFIG_TIMEOUT="${SMOKE_CONFIG_TIMEOUT:-900}"

# max_context_length must EQUAL num-tokens-per-microbatch-per-dp-rank: the
# blendcorpus fold emits fixed-length rows and the SDPA wrapper divides by it.
_SEQ="--training.max-context-length=512 --training.num-tokens-per-microbatch-per-dp-rank=512"
DEFAULT_CONFIGS=(
    "ezpz.agpt:agpt_debugmodel:${_SEQ}"
    # TP=2 is the arm that catches the local_map/SPMD contract breaks
    # (#4533 class), which only fire under TP>1.
    "ezpz.agpt:agpt_debugmodel:${_SEQ} --parallelism.tensor-parallel-degree=2"
    "ezpz.moe:moe_debugmodel:${_SEQ}"
)
# label-suffix|regex, e.g. "tensor-parallel-degree=2|some error text".
# Empty: the TP=2 "token count 256 is not a multiple of max_context_length"
# failure was autograd multithreading running AC recompute without the SPMD
# mesh; train.py now disables it (Sunspot 12483861: agpt TP=2 trains).
KNOWN_FAILURES=()

if [[ -n "${SMOKE_CONFIGS:-}" ]]; then
    CONFIGS=()
    while IFS= read -r _line; do
        [[ -n "$_line" ]] && CONFIGS+=("$_line")
    done <<< "${SMOKE_CONFIGS}"
else
    CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

# --- environment via the canonical ezpz path ---
# ezpz_setup <venv> does ezpz_setup_job + ezpz_load_modules + activate.
# It sets HOSTFILE/PBS_NODEFILE, DIST_LAUNCH, NGPUS/NGPU_PER_HOST, the XPU
# env (ZE_FLAT/CCL_OP_SYNC/ONEAPI_DEVICE_SELECTOR), the CUDA shims
# (MPICH_GPU_SUPPORT_ENABLED/PALS_TRANSFER/mpi-compat), and the venv itself.
# Nothing below reimplements that; we just consume what it exported.

# Allow callers to pin an older utils revision via the existing contract.
if [[ -n "${EZPZ_UTILS_PIN_COMMIT:-}" || -n "${EZPZ_UTILS_PIN_SHA256:-}" ]]; then
    source "${SUBMIT_DIR}/torchtitan/experiments/ezpz/scripts/load_pinned_ezpz_utils.sh"
    load_pinned_ezpz_utils || { echo "VERDICT: setup_failed (ezpz utils)"; exit 2; }
    # Pinned flow still goes through the same canonical entry point.
    VENV_PINNED="${VENV}"
    # shellcheck disable=SC2154  # EZPZ_UTILS_* come from the pin env above
    ezpz_setup "${VENV_PINNED}" || { echo "VERDICT: setup_failed (ezpz_setup ${VENV_PINNED})"; exit 2; }
else
    source <(curl -fsSL https://ezpz.cool/utils.sh) && ezpz_setup "${VENV}" \
        || { echo "VERDICT: setup_failed (ezpz_setup ${VENV})"; exit 2; }
fi

# Back in the checkout (ezpz_setup may have cd'd).
cd "${SUBMIT_DIR}" || exit 2

ACCEL=""; [[ -n "${ZE_FLAT_DEVICE_HIERARCHY:-}${CCL_PROCESS_LAUNCHER:-}" ]] && ACCEL=xpu || ACCEL=cuda

# --- library-path workarounds, applied AFTER ezpz_setup's module loads ---
# ezpz_setup does not provide these; without them every production smoke on
# 2026-11-17 died in the import probe (Sunspot 12484699 and Aurora 9098335:
# libsycl urGraphGetIdExp; Polaris 7857660 and Sirius 33741: "cannot load MPI
# library"). They must come after the module loads, which re-prepend the
# system oneAPI/MPI paths. TODO: drop once ezpz_setup owns them
# (https://github.com/saforem2/ezpz/issues/308).
if [[ "${ACCEL}" == cuda ]]; then
    # The venv's mpi4py is not linked against Cray GTL: MPI_Init aborts with
    # "GPU_SUPPORT_ENABLED is requested, but GTL library is not linked"
    # (Sirius 33436) unless this is off.
    export MPICH_GPU_SUPPORT_ENABLED=0
    # PALS copies the interpreter to /var/run/palsd on multi-host launches; a
    # uv-managed python then misses $ORIGIN/../lib/libpython3.12.so.1.0
    # (rank exit 127, Polaris 7839193).
    export PALS_TRANSFER=0
    [[ -d "${HOME}/.local/mpi-compat" ]] \
        && export LD_LIBRARY_PATH="${HOME}/.local/mpi-compat:${LD_LIBRARY_PATH:-}"
    # pip's mpi4py 4.x (MPI-ABI build) dlopens the generic libmpi.so.12, which
    # Cray MPICH does not ship. Point it at Cray's libmpi.so.
    if [[ -n "${CRAY_MPICH_DIR:-}" && -e "${CRAY_MPICH_DIR}/lib/libmpi.so" ]]; then
        _mpi_abi="${SUBMIT_DIR}/.cache/mpi-abi"
        mkdir -p "${_mpi_abi}"
        ln -sfn "${CRAY_MPICH_DIR}/lib/libmpi.so" "${_mpi_abi}/libmpi.so.12"
        export LD_LIBRARY_PATH="${_mpi_abi}:${LD_LIBRARY_PATH:-}"
    fi
elif [[ -d "${VENV}/lib" ]]; then
    # A torch wheel newer than the loaded oneAPI module needs its own Unified
    # Runtime / SYCL libs first, or `import torch` dies with
    #   libsycl.so.9: undefined symbol: urGraphGetIdExp
    # Link only the UR/SYCL/UMF/TCM set: the whole venv lib/ would also shadow
    # the system MPI/libfabric and break PMIx.
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

# Keep the venv's compile-cache policy: node-local, not on Lustre.
export TORCHINDUCTOR_CACHE_DIR="/tmp/${USER}-smoke-inductor"
export TRITON_CACHE_DIR="/tmp/${USER}-smoke-triton"

# --- launch args from what ezpz_setup exported ---
# Prefer what the scheduler/utils already know; fall back to detection for
# local runs with no PBS_NODEFILE.
if [[ -n "${NGPUS:-}" && -n "${NGPU_PER_HOST:-}" ]]; then
    NPROC="${NGPUS}"
    PPN="${NGPU_PER_HOST}"
elif [[ -n "${PBS_NODEFILE:-}" ]]; then
    # Classic fallback (older utils or non-ezpz launchers).
    NHOSTS="$(sort -u "${PBS_NODEFILE}" 2>/dev/null | wc -l)"
    (( NHOSTS > 0 )) || NHOSTS=1
    # Detect accel from what ezpz_setup loaded (has CCL/ZE_FLAT on XPU).
    if [[ -n "${ZE_FLAT_DEVICE_HIERARCHY:-}" || -n "${CCL_PROCESS_LAUNCHER:-}" ]]; then
        PPN=12
    else
        PPN=4
    fi
    NPROC=$((NHOSTS * PPN))
else
    NPROC=1; PPN=1
fi

# Derive a human label for logs (keep SMOKE_MACHINE if caller set it).
MACHINE="${SMOKE_MACHINE:-$(hostname -s 2>/dev/null || hostname)}"

# Build the optional hf-assets flag once (omit when the dir doesn't exist).
HF_FLAG=()
if [[ -s "${HF_ASSETS}/tokenizer.json" || -s "${HF_ASSETS}/tokenizer.model" ]]; then
    HF_FLAG=(--hf-assets-path="${HF_ASSETS}")
fi

JOBID_SHORT="${PBS_JOBID%%.*}"
[[ -z "${JOBID_SHORT}" ]] && JOBID_SHORT="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="logs/ezpz-sync-smoke-${MACHINE}-${JOBID_SHORT}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/run.log"
strip_ansi() { sed 's/\x1b\[[0-9;]*m//g' "$@"; }

{
    echo "===== ezpz sync smoke (${MACHINE}) ====="
    date
    echo "node0:  $(hostname)"
    echo "venv:   ${VIRTUAL_ENV:-${VENV}} ($(python -c 'import sys;print(sys.executable)' 2>/dev/null || echo n/a))"
    echo "assets: ${HF_ASSETS:-<none>}"
    echo "tree:   ${SUBMIT_DIR}"
    echo "commit: $(git -C "${SUBMIT_DIR}" log --oneline -1 2>/dev/null)"
    echo "ranks:  ${NPROC} (ppn=${PPN})  steps=${STEPS} seed=${SEED} timeout=${CONFIG_TIMEOUT}s  accel=${ACCEL}"
    echo "launch: ${DIST_LAUNCH:-ezpz launch --nproc ${NPROC} --nproc_per_node ${PPN}}"
} | tee "${LOG}"
[[ -z "$(git -C "${SUBMIT_DIR}" status --porcelain --untracked-files=no 2>/dev/null)" ]] \
    || { echo "VERDICT: dirty_source" | tee -a "${LOG}"; exit 1; }
python -c "import torch;print('torch:  ',torch.__version__)" 2>&1 | tail -1 | tee -a "${LOG}"

# Build the launcher command once. When ezpz_setup ran inside PBS it set
# DIST_LAUNCH; outside PBS (local import probe) we fall back to a bare
# single-rank mpiexec if available, else plain python.
_run_dist() {  # _run_dist <timeout> <logfile> -- <py args...>
    local _timeout="$1" _log="$2"; shift 2
    # shellcheck disable=SC2086  # DIST_LAUNCH is an intentional word-split cmd
    if [[ -n "${DIST_LAUNCH:-}" ]]; then
        # shellcheck disable=SC2086
        timeout --kill-after=60 "${_timeout}" \
            python -c 'from ezpz.cli import main; main()' launch \
            --nproc "${NPROC}" --nproc_per_node "${PPN}" -- \
            python -m torchtitan.experiments.ezpz.train "$@" > "${_log}" 2>&1
    elif command -v mpiexec >/dev/null 2>&1; then
        timeout --kill-after=60 "${_timeout}" \
            mpiexec -n "${NPROC}" --ppn "${PPN}" \
            python -m torchtitan.experiments.ezpz.train "$@" > "${_log}" 2>&1
    else
        timeout --kill-after=60 "${_timeout}" \
            python -m torchtitan.experiments.ezpz.train "$@" > "${_log}" 2>&1
    fi
}

# --- Phase 1: import probe ---
# ezpz imports mpi4py, which calls MPI_Init_thread at import; run under the
# launcher so PMIx exists when we're inside PBS.
echo "--- import probe ---" | tee -a "${LOG}"
if [[ -n "${DIST_LAUNCH:-}" ]] && command -v mpiexec >/dev/null 2>&1; then
    mpiexec -n 1 --ppn 1 python -c "
from torchtitan.experiments.ezpz.agpt.parallelize import parallelize_llama
from torchtitan.experiments.ezpz.agpt.config_registry import agpt_debugmodel
from torchtitan.experiments.ezpz.moe.parallelize import parallelize_moe
from torchtitan.experiments.ezpz.moe.config_registry import moe_debugmodel
from torchtitan.experiments.ezpz.trainer import FaultTolerantTrainer
print('IMPORT_OK')
" 2>&1 | tee -a "${LOG}"
else
    python -c "
from torchtitan.experiments.ezpz.agpt.parallelize import parallelize_llama
from torchtitan.experiments.ezpz.agpt.config_registry import agpt_debugmodel
from torchtitan.experiments.ezpz.moe.parallelize import parallelize_moe
from torchtitan.experiments.ezpz.moe.config_registry import moe_debugmodel
from torchtitan.experiments.ezpz.trainer import FaultTolerantTrainer
print('IMPORT_OK')
" 2>&1 | tee -a "${LOG}"
fi
grep -q "IMPORT_OK" "${LOG}" || { echo "VERDICT: import_failed" | tee -a "${LOG}"; exit 1; }

# --- Phase 2: per-config train smoke ---
WANDB=(--metrics.no-enable-wandb)
[[ "${SMOKE_WANDB:-0}" == 1 ]] && WANDB=(--metrics.enable-wandb)
# Upstream captures fwd/bwd in an accelerator graph by default (training.
# disable_accelerator_graphs=False, backend-agnostic since #5353). The ezpz
# trainer drops to eager itself for native DDP, host-syncing MoE expert
# backends and XPU (docs/guides/known-bugs/cuda-graphs-moe-native-ddp.md), so
# smoke keeps upstream's default; SMOKE_CUDA_GRAPHS=0 forces eager on CUDA.
GRAPHS=()
[[ "${ACCEL}" == cuda && "${SMOKE_CUDA_GRAPHS:-1}" == 0 ]] \
    && GRAPHS=(--training.disable-accelerator-graphs)
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
    _run_dist "${CONFIG_TIMEOUT}" "${clog}" \
        --module="${module}" \
        --config="${config}" \
        "${HF_FLAG[@]}" \
        --training.steps="${STEPS}" \
        --debug.seed="${SEED}" \
        --debug.deterministic \
        --checkpoint.no-enable \
        "${WANDB[@]}" \
        "${GRAPHS[@]}" \
        ${extra}
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
        _run_dist "${CONFIG_TIMEOUT}" "${LOG_DIR}/resume-$1.log" \
            --module="${module}" --config="${config}" \
            "${HF_FLAG[@]}" --training.steps="${STEPS}" \
            --debug.seed="${SEED}" --debug.deterministic "${WANDB[@]}" "${GRAPHS[@]}" \
            --checkpoint.enable --checkpoint.interval="${RESUME_AT}" \
            --checkpoint.keep-latest-k=0 \
            --dump-folder="${LOG_DIR}/resume-$1" \
            ${extra} ${rextra}
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
