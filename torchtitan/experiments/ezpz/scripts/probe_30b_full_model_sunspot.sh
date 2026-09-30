#!/bin/bash --login
#PBS -A datascience
#PBS -N tt30b-model-canary
#PBS -l walltime=01:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=4
#PBS -q workq
#PBS -j oe
set -o pipefail

D="${PROBE_WORKTREE:?set PROBE_WORKTREE}"
V="${PROBE_VENV:?set PROBE_VENV}"
EXPECTED_SHA="${EXPECTED_SHA:?set EXPECTED_SHA}"
TOKENIZER="${TOKENIZER:?set TOKENIZER}"
JOB="${PBS_JOBID%%.*}"
MODEL_CONFIG="${CANARY_MODEL_CONFIG:-agpt_30b_olmo2tok_smoke}"
NPROC="${CANARY_NPROC:-48}"
DP_REPLICATE="${CANARY_DP_REPLICATE:-3}"
DP_SHARD="${CANARY_DP_SHARD:-16}"
TOKENS_PER_STEP="${CANARY_TOKENS_PER_STEP:-196608}"
OUT="$D/outputs/30b-full-model-canary-$JOB"
LOG="$OUT/run.log"
mkdir -p "$OUT"
cd "$D" || exit 2

export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-$http_proxy}"
source "$D/torchtitan/experiments/ezpz/scripts/load_pinned_ezpz_utils.sh"
load_pinned_ezpz_utils || exit $?
export PATH="/opt/pbs/bin:$PATH"
ezpz_setup_job || exit $?
ezpz_load_modules || exit $?
export VIRTUAL_ENV="$V"
export PATH="$V/bin:$PATH"
hash -r
[[ "$(command -v python)" == "$V/bin/python" ]] || exit 93
[[ "$(git rev-parse HEAD)" == "$EXPECTED_SHA" ]] || exit 95
[[ -s "$TOKENIZER/tokenizer.json" ]] || exit 97

export PYTHONPATH="$D${PYTHONPATH:+:$PYTHONPATH}"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export CCL_ATL_SYNC_COLL=1
export CCL_SYCL_KERNEL_SYNC=0
export CCL_KVS_MODE=pmi
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR


run_model() {
    "$V/bin/python" -m torchtitan.experiments.ezpz.train \
    --module ezpz.agpt \
    --config "$MODEL_CONFIG" \
    --hf-assets-path "$TOKENIZER" \
    --job.dump-folder "$OUT" \
    --optimizer adamw \
    --optimizer.lr 1e-6 \
    --training.steps 3 \
    --training.max-context-length 4096 \
    --training.num-tokens-per-microbatch-per-dp-rank 4096 \
    --training.num-tokens-per-train-step "$TOKENS_PER_STEP" \
    --parallelism.tensor-parallel-degree 1 \
    --parallelism.data-parallel-replicate-degree "$DP_REPLICATE" \
    --parallelism.data-parallel-shard-degree "$DP_SHARD" \
    --metrics.no-enable-wandb \
    --checkpoint.no-enable \
    activation-checkpoint:full
}
export V D TOKENIZER OUT MODEL_CONFIG NPROC DP_REPLICATE DP_SHARD TOKENS_PER_STEP
export -f run_model

if ((NPROC % 12 != 0)); then
    CPU_BIND_SUNSPOT="list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
    WORLD_SIZE="$NPROC" timeout 1800 mpiexec --envall --line-buffer --np="$NPROC" --ppn=12 \
        --hostfile="$PBS_NODEFILE" --cpu-bind="$CPU_BIND_SUNSPOT" \
        bash -c 'run_model' 2>&1 | tee "$LOG"
else
    "$V/bin/python" -c 'from ezpz.cli import main; main()' launch \
        --nproc "$NPROC" --nproc_per_node 12 --timeout 1800 -- \
        bash -c 'run_model' 2>&1 | tee "$LOG"
fi
rc=${PIPESTATUS[0]}
if [[ "$rc" -ne 0 ]]; then
    printf 'FULL_MODEL_CANARY_FAILED job=%s rc=%s\n' "$JOB" "$rc" | tee "$OUT/FAILED"
    exit "$rc"
fi

"$V/bin/python" - "$LOG" "$OUT/VALIDATED" "$JOB" "$EXPECTED_SHA" <<'PY'
import math
import re
import sys
from pathlib import Path

log_path, verdict_path, job, commit = sys.argv[1:]
text = Path(log_path).read_text(errors="replace")
ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
text = ansi.sub("", text)
metric = re.compile(
    r"step:\s+(?P<step>[0-9]+)\s+loss:\s+(?P<loss>[0-9.eE+-]+)"
    r"\s+grad_norm:\s+(?P<grad>[0-9.eE+-]+)"
)
rows = {int(m["step"]): (float(m["loss"]), float(m["grad"])) for m in metric.finditer(text)}
for step in (1, 2, 3):
    if step not in rows:
        raise SystemExit(f"FULL_MODEL_CANARY_INVALID missing metrics step={step}")
    if not all(math.isfinite(value) and value > 0 for value in rows[step]):
        raise SystemExit(f"FULL_MODEL_CANARY_INVALID nonfinite/zero metrics step={step} values={rows[step]}")
if any(token in text for token in ("Traceback", "SIGSEGV", "died from signal", "OUT_OF_RESOURCES")):
    raise SystemExit("FULL_MODEL_CANARY_INVALID failure signature present")
verdict = (
    f"FULL_MODEL_CANARY_PASS job={job} commit={commit} steps=3 "
    f"loss3={rows[3][0]} grad3={rows[3][1]}"
)
Path(verdict_path).write_text(verdict + "\n")
print(verdict)
PY
