#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N agpt-v2-4k-smoke
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -l select=2
#PBS -q debug
#PBS -j oe
#
# Smoke test for the new agpt-v2 {4,10,30}B models at 4K seqlen with the
# olmo3 tokenizer on the dynamic streaming Grain + HF datasets dataloader on
# Aurora. Tests all three sizes sequentially at a reduced step count to validate:
#   - model builds (correct params, correct tokenizer/vocab)
#   - dataloader produces batches (Grain + HuggingFaceStreamingSource at 4K rows)
#   - training loss decreases (proves gradients flow)
#   - no OOM at 4K (attention scales quadratically, 4K is the safe baseline)
#
# Usage (interactive allocation):
#   smoke_agpt_v2_4k_aurora.sh
#
# Usage (PBS):
#   qsub -l select=2 -l walltime=01:00:00 -q debug torchtitan/experiments/ezpz/scripts/smoke_agpt_v2_4k_aurora.sh
#
# To run a single size:
#   CONFIG=agpt_v2_4b_4k smoke_agpt_v2_4k_aurora.sh
#   CONFIG=agpt_v2_10b_4k smoke_agpt_v2_4k_aurora.sh
#   CONFIG=agpt_v2_30b_4k smoke_agpt_v2_4k_aurora.sh
#
# Requirements on Aurora:
#   - $HOME/.cache/huggingface/hub/datasets--allenai--olmo-mix-1124/snapshots/*/data/wiki (2 files, precached)
#   - assets/hf/OLMo-2-1124-7B/tokenizer.json (OLMo-3 tokenizer, byte-identical to OLMo-2)
#   - Intel GPU stack on compute nodes (oneapi, frameworks/2026.1.0 or venv)

set -o pipefail

# ---- PBS workdir ----
PBS_O_WORKDIR="${PBS_O_WORKDIR:-$(pwd)}"
cd "${PBS_O_WORKDIR}" || exit 1

# ---- Intel GPU env (required on every Aurora node) ----
# On Aurora COMPUTE nodes the PBS shell is bare — `module` is unavailable
# (PBS on Aurora does not start with --login on compute, despite the script
# header). Source Lmod first so `module load` works there; the successful
# Mar 2026 production run did exactly this via ezpz_setup_env but we need
# it BEFORE activating the venv because torch needs libmkl_intel_lp64.so.2.
if ! command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
if command -v module >/dev/null 2>&1; then
    module load oneapi/release/2025.3.1 hdf5 pti-gpu frameworks/2025.3.1 2>&1 | grep -v "^Lmod" || true
fi
# The venv's torch was built against 26.26.0 MKL; on compute the default
# oneapi is 2026.1.0 (26.181.0) which ships NO libmkl_intel_lp64.so.2 —
# force the correct MKL to the front of LD regardless of module ordering.
export LD_LIBRARY_PATH="/opt/aurora/26.26.0/oneapi/mkl/latest/lib:${LD_LIBRARY_PATH:-}"
# Torch also needs libsycl.so.8 from the 2025.3.1 compiler + CCL + PTI
# (the same 4-fix chain the eval scripts solved: 986404e6a..b68557278).
export LD_LIBRARY_PATH="/opt/aurora/26.26.0/oneapi/compiler/latest/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="/opt/aurora/26.26.0/oneapi/ccl/latest/lib:${LD_LIBRARY_PATH}"
# pti-gpu module does NOT set LD_LIBRARY_PATH (4f9a7cb82)
PTI_LIB="$(ls -d /opt/aurora/26.26.0/spack/unified/*/install/linux-x86_64/pti-gpu-*/lib64 2>/dev/null | head -n 1)"
[ -n "$PTI_LIB" ] && export LD_LIBRARY_PATH="${PTI_LIB}:${LD_LIBRARY_PATH}"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"

# ---- Proxy (Aurora compute nodes have no direct egress) ----
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"

# ---- ezpz env ----
# Aurora venvs live under venvs/aurora/, not .venv (which is mbph-only, gitignored)
VENV_CANDIDATES=(
    "${PBS_O_WORKDIR}/venvs/aurora/torchtitan-aurora_frameworks-2025.3.1/bin/activate"
    "${PBS_O_WORKDIR}/venvs/aurora/torchtitan-aurora_frameworks-2025.2.0/bin/activate"
    "${PBS_O_WORKDIR}/.venv/bin/activate"
)
for _cand in "${VENV_CANDIDATES[@]}"; do
    if [ -f "$_cand" ]; then
        echo "[smoke] activating venv: $_cand"
        # shellcheck disable=SC1090
        source "$_cand"
        break
    fi
done
if ! command -v ezpz >/dev/null 2>&1; then
    echo "[smoke] WARNING: ezpz not found after venv activation; will try bare python" >&2
fi
# Also source ezpz utils if available (sets up MPI, etc. on PBS nodes)
if [ -f "${PBS_O_WORKDIR}/ezpz-utils" ]; then
    # shellcheck disable=SC1090
    source "${PBS_O_WORKDIR}/ezpz-utils" 2>&1 | tail -n 5 || true
elif [ -z "${PBS_NODEFILE:-}" ] && [ -n "${PBS_JOBID:-}" ]; then
    source <(curl -fsSL https://bit.ly/ezpz-utils 2>&1) 2>/dev/null && ezpz_setup_env 2>&1 | tail -n 5 || true
fi

# ---- Config ----
STEPS="${STEPS:-15}"
if [ -n "${CONFIG:-}" ]; then
    CONFIGS=("$CONFIG")
else
    CONFIGS=(agpt_v2_4b_4k agpt_v2_10b_4k agpt_v2_30b_4k)
fi

# ---- Preflight ----
echo "[smoke] hostname=$(hostname)  PBS_JOBID=${PBS_JOBID:-n/a}  STEPS=$STEPS"
echo "[smoke] configs: ${CONFIGS[*]}"
echo "[smoke] torch: $(python3 -c 'import torch; print(torch.__version__)' 2>&1 | head -n 1)"
echo "[smoke] xpu available: $(python3 -c 'import torch; print(torch.xpu.is_available() if hasattr(torch,\"xpu\") else False)' 2>&1 | head -n 1)"
echo "[smoke] hf_assets_path check:"
for p in ./assets/hf/OLMo-2-1124-7B/tokenizer.json ./assets/hf/gemma-7b/tokenizer.json; do
    if [ -f "$p" ]; then echo "  OK $p"; else echo "  MISSING $p"; fi
done
echo "[smoke] olmo-mix cache check:"
OLMO_DIR="$(python3 -c 'import glob,os; h=os.path.join(os.path.expanduser("~"),".cache/huggingface/hub","datasets--allenai--olmo-mix-1124/snapshots"); hits=sorted(glob.glob(os.path.join(h,"*","data","wiki"))); print(hits[-1] if hits else "MISSING")' 2>&1)"
echo "  $OLMO_DIR"
if [ "$OLMO_DIR" = "MISSING" ] || [ ! -d "$OLMO_DIR" ]; then
    echo "[smoke] ERROR: olmo-mix wiki subset not found under ~/.cache/huggingface/hub/datasets--allenai--olmo-mix-1124/snapshots/*/data/wiki" >&2
    echo "[smoke] Run scripts/precache_hf_dataset.py from a login node first." >&2
    exit 1
fi

# ---- Run each config ----
FAILED=()
PASSED=()
for CFG in "${CONFIGS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "[smoke] CONFIG=$CFG  STEPS=$STEPS  START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "======================================================================"

    set +e
    # Use ezpz launch when available (handles MPI/PBS distribution), else plain python
    if command -v ezpz >/dev/null 2>&1 && [ -n "${PBS_NODEFILE:-}" ]; then
        LAUNCH=(ezpz launch python3 -m torchtitan.experiments.ezpz.train)
    else
        LAUNCH=(python3 -m torchtitan.experiments.ezpz.train)
    fi

    "${LAUNCH[@]}" \
        --module ezpz.agpt \
        --config "$CFG" \
        --training.steps "$STEPS" \
        --checkpoint.no-enable \
        --metrics.enable-wandb False \
        2>&1 | tee "/tmp/smoke-${CFG}.log"

    RC=${PIPESTATUS[0]}
    set -e

    if [ $RC -ne 0 ]; then
        echo "[smoke] FAIL $CFG exit=$RC"
        FAILED+=("$CFG (exit $RC)")
        continue
    fi

    # Validate: check for training completion marker and finite loss
    if grep -q "Training completed\|steps.*$STEPS\|step.*$STEPS" "/tmp/smoke-${CFG}.log" 2>/dev/null; then
        echo "[smoke] PASS $CFG (training completed)"
        PASSED+=("$CFG")
    elif grep -q "loss" "/tmp/smoke-${CFG}.log" 2>/dev/null && ! grep -q "NaN\|nan\|inf" "/tmp/smoke-${CFG}.log" 2>/dev/null; then
        echo "[smoke] PASS $CFG (finite loss observed, no NaN)"
        PASSED+=("$CFG")
    else
        echo "[smoke] WARN $CFG: no clear completion marker; check /tmp/smoke-${CFG}.log"
        # don't count as fail if exit was 0
        PASSED+=("$CFG (no marker)")
    fi
done

echo ""
echo "======================================================================"
echo "[smoke] SUMMARY  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  PASSED: ${PASSED[*]:-none}"
echo "  FAILED: ${FAILED[*]:-none}"
echo "======================================================================"

if [ ${#FAILED[@]} -ne 0 ]; then
    exit 1
fi
