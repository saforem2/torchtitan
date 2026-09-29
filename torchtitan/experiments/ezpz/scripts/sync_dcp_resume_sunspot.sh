#!/bin/bash --login
#PBS -A datascience
#PBS -N ezpz-sync-dcp
#PBS -l walltime=00:30:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe

# Exact-head DCP behavioral gate: uninterrupted control, full-state save, then
# a fresh process that loads and resumes. Success requires the resumed step-3/4
# loss and gradient norm to match the uninterrupted control exactly.
set -o pipefail

D="${SYNC_DCP_WORKTREE:?set SYNC_DCP_WORKTREE}"
V="${SYNC_DCP_VENV:?set SYNC_DCP_VENV}"
EXPECTED_SHA="${EXPECTED_SHA:?set EXPECTED_SHA}"
JOB="${PBS_JOBID%%.*}"
ROOT="$D/outputs/sync-dcp-$JOB"
CKPT="$ROOT/checkpoints"
TOKENIZER="$D/assets/hf/gemma-7b"
mkdir -p "$ROOT"
cd "$D" || exit 2

source <(curl -fsSL https://ezpz.cool/utils.sh)
ezpz_setup_job
ezpz_load_modules
export VIRTUAL_ENV="$V"
export PATH="$V/bin:/opt/pbs/bin:$PATH"
hash -r
[[ "$(command -v python)" == "$V/bin/python" ]] || exit 93
[[ "$(git rev-parse HEAD)" == "$EXPECTED_SHA" ]] || exit 95
[[ -s "$TOKENIZER/tokenizer.json" ]] || exit 97

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export CCL_ATL_SYNC_COLL=1
export CCL_SYCL_KERNEL_SYNC=0
export CCL_KVS_MODE=pmi
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"

run_phase() {
    phase=$1
    steps=$2
    shift 2
    log="$ROOT/$phase.log"
    "$V/bin/python" -c 'from ezpz.cli import main; main()' launch \
        --nproc 24 --nproc_per_node 12 --timeout 600 -- \
        "$V/bin/python" -m torchtitan.experiments.ezpz.train \
        --module ezpz.agpt \
        --config agpt_debugmodel \
        --hf-assets-path "$TOKENIZER" \
        --job.dump-folder "$ROOT/$phase" \
        --training.steps "$steps" \
        --training.max-context-length 512 \
        --training.num-tokens-per-microbatch-per-dp-rank 512 \
        --debug.seed 20260929 \
        --debug.deterministic \
        --compile.no-enable \
        --metrics.no-enable-wandb \
        "$@" \
        2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    printf '%s rc=%s\n' "$phase" "$rc" | tee -a "$log"
    return "$rc"
}

run_phase control 4 --checkpoint.no-enable || exit $?
run_phase save 4 \
    --checkpoint.enable \
    --checkpoint.folder "$CKPT" \
    --checkpoint.interval 2 \
    --checkpoint.keep-latest-k 0 \
    --checkpoint.no-last-save-model-only \
    --checkpoint.async-mode disabled || exit $?

test -s "$CKPT/step-2/.metadata" || {
    echo "DCP_SYNC_VERDICT: missing step-2 metadata"
    exit 31
}

run_phase resume 4 \
    --checkpoint.enable \
    --checkpoint.folder "$CKPT" \
    --checkpoint.load-step 2 \
    --checkpoint.interval 4 \
    --checkpoint.keep-latest-k 0 \
    --checkpoint.no-last-save-model-only \
    --checkpoint.async-mode disabled || exit $?

"$V/bin/python" - "$ROOT/control.log" "$ROOT/resume.log" <<'PY'
import math
import re
import sys
from pathlib import Path

pattern = re.compile(
    r"step:\s+(?P<step>[0-9]+)\s+loss:\s+(?P<loss>[0-9.eE+-]+)"
    r"\s+grad_norm:\s+(?P<grad>[0-9.eE+-]+)"
)
ansi_escape = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

def metrics(path: str) -> dict[int, tuple[float, float]]:
    found = {}
    text = ansi_escape.sub("", Path(path).read_text(errors="replace"))
    for match in pattern.finditer(text):
        found[int(match["step"])] = (float(match["loss"]), float(match["grad"]))
    return found

control = metrics(sys.argv[1])
resumed = metrics(sys.argv[2])
for step in (3, 4):
    if step not in control or step not in resumed:
        raise SystemExit(f"DCP_SYNC_VERDICT: missing metrics at step {step}")
    for name, left, right in zip(("loss", "grad_norm"), control[step], resumed[step]):
        if not (math.isfinite(left) and math.isfinite(right) and left == right):
            raise SystemExit(
                f"DCP_SYNC_VERDICT: mismatch step={step} {name} control={left} resume={right}"
            )
print("DCP_SYNC_VERDICT: ok steps=3,4 exact=true")
PY
