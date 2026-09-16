#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N ezpz-sync-smoke-aurora
#PBS -l walltime=01:00:00
#PBS -l filesystems=flare:home
#PBS -l select=2
#PBS -q debug-scaling
#PBS -j oe

# Aurora twin of sync_smoke.sh: post-upstream-sync smoke on XPU, 2 nodes.
#
# Same two phases as the sunspot original (import probe, then seeded
# deterministic train steps with a machine-greppable VERDICT), with three
# deliberate differences:
#
#   1. Queue/filesystem are Aurora's.
#
#      QUEUE CHOICE IS NOT FREE -- it is coupled to the venv. `next-eval` runs
#      bkc_definition compute_aurora_test_* while every other queue pins
#      compute_aurora_prod_*. The venv below has
#        home = /opt/aurora/26.26.0/spack/.../python-3.12.12-nvje3vk/bin
#      i.e. a base interpreter from the PROD image, so on next-eval its python
#      does not exist and every import dies with a misleading
#        ModuleNotFoundError: No module named 'importlib.metadata'
#      (measured: jobs 8829080 and 8829104, both ~10 s, both trees identical).
#      Hence debug-scaling, which runs the prod bkc. To use next-eval instead,
#      either build a venv against the test image or `module load
#      frameworks/2026.1.0` and let it supply the interpreter.
#      Check `pyvenv.cfg`'s home line before changing this queue.
#   2. No `ezpz yeet-env`. The sunspot script broadcasts a .venv.tar.gz to
#      /tmp on every node; a freshly cloned sync tree has no tarball. This
#      activates the existing project venv directly. Slower to start at scale,
#      irrelevant at 2 nodes, and it removes a moving part from a run whose
#      whole job is to isolate the merge.
#   3. TP=2 arm is kept. That is the arm that catches the #4533 class of
#      break: the local_map/SPMD contract matches by positional-arg NAME and
#      only fires under TP>1, so a TP=1-only smoke passes over exactly the
#      bug this sync was most likely to introduce.
#
# Usage:
#   qsub torchtitan/experiments/ezpz/scripts/sync_smoke_aurora.sh
#
# Env knobs:
#   SMOKE_VENV     path to the venv to activate (default: the torchtitan-ezpz
#                  XPU venv, torch 2.13.0.dev20260520+xpu)
#   SMOKE_CONFIGS  newline-separated "module:config[:extra cli ...]" specs
#   SMOKE_STEPS    steps per config (default 3)
#   SEED           debug seed (default 42)

set -o pipefail

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"

STEPS="${SMOKE_STEPS:-3}"
SEED="${SEED:-42}"
# Resolve the venv path. `/flare` is a symlink to `/lus/flare/projects`, and an
# UNRESOLVED prefix breaks `ezpz launch`: it emits an mpiexec line with no
# program and hydra rejects it with "argument matching returned error" in 0.1s.
# Measured on 8831640 -- same node, same command, two venvs differing only in
# prefix: /flare/... fails, /lus/flare/projects/... runs. Cost job 8831612.
VENV="${SMOKE_VENV:-/flare/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz/.venv}"
VENV="$(readlink -f "${VENV}")"

# max_context_length must EQUAL num-tokens-per-microbatch-per-dp-rank: the
# blendcorpus fold emits fixed-length rows and the SDPA wrapper divides by it.
# There is no training.seq_len / local_batch_size -- those spellings are
# silently rejected by tyro.
DEFAULT_CONFIGS=(
    "ezpz.agpt:agpt_debugmodel:--training.max-context-length=512 --training.num-tokens-per-microbatch-per-dp-rank=512"
    "ezpz.agpt:agpt_debugmodel:--training.max-context-length=512 --training.num-tokens-per-microbatch-per-dp-rank=512 --parallelism.tensor-parallel-degree=2"
    "ezpz.moe:moe_debugmodel:--training.max-context-length=512 --training.num-tokens-per-microbatch-per-dp-rank=512"
)
if [[ -n "${SMOKE_CONFIGS:-}" ]]; then
    CONFIGS=()
    while IFS= read -r _line; do
        [[ -n "$_line" ]] && CONFIGS+=("$_line")
    done <<< "${SMOKE_CONFIGS}"
else
    CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"

source "${VENV}/bin/activate"

JOBID_SHORT="${PBS_JOBID%%.*}"
LOG_DIR="logs/ezpz-sync-smoke-aurora-${JOBID_SHORT:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/run.log"

echo "===== ezpz sync smoke (aurora, sync 84) =====" | tee "${LOG}"
date | tee -a "${LOG}"
echo "node0:  $(hostname)" | tee -a "${LOG}"
echo "venv:   ${VENV}" | tee -a "${LOG}"
echo "tree:   ${SUBMIT_DIR}" | tee -a "${LOG}"
echo "commit: $(git -C "${SUBMIT_DIR}" log --oneline -1 2>/dev/null)" | tee -a "${LOG}"
echo "steps=${STEPS} seed=${SEED}" | tee -a "${LOG}"
python3 -c "import torch;print('torch:  ',torch.__version__)" 2>&1 | tail -1 | tee -a "${LOG}"
echo "" | tee -a "${LOG}"

# --- Phase 1: import probe (fail fast) ---
echo "--- import probe ---" | tee -a "${LOG}"
python3 -c "
from torchtitan.experiments.ezpz.agpt.parallelize import parallelize_llama
from torchtitan.experiments.ezpz.agpt.config_registry import agpt_debugmodel
from torchtitan.experiments.ezpz.moe.parallelize import parallelize_moe
from torchtitan.experiments.ezpz.moe.config_registry import moe_debugmodel
from torchtitan.experiments.ezpz.moe.activation_checkpoint import MoeSelectiveAC
from torchtitan.experiments.ezpz.trainer import FaultTolerantTrainer
print('IMPORT_OK')
" 2>&1 | tee -a "${LOG}"

if ! grep -q "IMPORT_OK" "${LOG}"; then
    echo "VERDICT: import_failed" | tee -a "${LOG}"
    exit 1
fi

# --- Phase 2: per-config train smoke ---
declare -a RESULTS=()
OVERALL_OK=1
idx=0
for spec in "${CONFIGS[@]}"; do
    idx=$((idx + 1))
    module="${spec%%:*}"
    rest="${spec#*:}"
    config="${rest%%:*}"
    extra="${rest#*:}"
    [[ "${extra}" == "${rest}" ]] && extra=""

    label="${idx}-${config}"
    echo "" | tee -a "${LOG}"
    echo "--- [${idx}] ${module} / ${config}: ${STEPS} deterministic steps ${extra:+(${extra})} ---" | tee -a "${LOG}"
    # shellcheck disable=SC2086 -- extra is an intentional word-split arg list
    ezpz launch python3 -m torchtitan.experiments.ezpz.train \
        --module="${module}" \
        --config="${config}" \
        --training.steps="${STEPS}" \
        --debug.seed="${SEED}" \
        --debug.deterministic \
        --metrics.enable-wandb \
        --checkpoint.no-enable \
        ${extra} \
        2>&1 | tee -a "${LOG_DIR}/${label}.log" | tee -a "${LOG}"
    rc=${PIPESTATUS[0]}
    RESULTS+=("${label}=${rc}")
    echo "${label} rc=${rc}" | tee -a "${LOG}"
    [[ ${rc} -ne 0 ]] && OVERALL_OK=0
done

echo "" | tee -a "${LOG}"
echo "--- per-step losses (for eyeball vs the pre-merge baseline) ---" | tee -a "${LOG}"
grep -hoE "step: +[0-9]+ +loss: +[0-9.]+" "${LOG_DIR}"/*.log 2>/dev/null | head -20 | tee -a "${LOG}"

echo "" | tee -a "${LOG}"
if [[ ${OVERALL_OK} -eq 1 ]]; then
    echo "VERDICT: ok" | tee -a "${LOG}"
else
    echo "VERDICT: failed (${RESULTS[*]})" | tee -a "${LOG}"
    exit 1
fi
