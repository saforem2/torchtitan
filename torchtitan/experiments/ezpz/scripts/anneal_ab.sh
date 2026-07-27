#!/bin/bash
# Eval-gated anneal A/B (v2) -- the CORRECT redo of the flawed first pass.
#
# Forks a converged base and continues on open-web-math with two LR schedules:
#   ARM A (flat): constant LR 2e-6            (agpt_2b_{mds,olmo}_anneal_flat)
#   ARM B (wsd):  LR 2e-6 -> 0, decay-to-0    (agpt_2b_{mds,olmo}_anneal_wsd)
# The arms are IDENTICAL except the schedule, so the A/B isolates generalization.
#
# WHAT THE FIRST (FLAWED) A/B GOT WRONG, and how this fixes it:
#   * v1 passed --checkpoint.load-only  -> _should_save() always False
#       -> ZERO checkpoints saved -> nothing to eval. FIX: do NOT pass load-only.
#       The config forks via initial_load_path + initial_load_model_only=True
#       (loads the base) and saves via checkpoint.enable=True + interval.
#   * v1 passed --checkpoint.interval=100000 -> no saves. FIX: interval=100.
#   * v1 omitted --training.local-batch-size=2 -> LBS defaulted to 1 -> token
#       budget HALVED. FIX: pass LBS=2 (the config comment assumes GBS=6144=LBS2).
#   * v1 omitted --debug.seed -> token order not pinned across arms. FIX: seed=42
#       so flat + wsd stream byte-identical data (differ ONLY in schedule).
#   * v1 ran 300 steps (~0.1B tok) and compared final TRAIN loss (invalid: wsd
#       ends at LR~0 so its train loss is structurally lower). FIX: 1600 steps
#       (~10B tok cheap-decisive) + decide on held-out val loss of saved ckpts.
#
# Validated recipe corners: HSDP (dp_shard=12 / dp_replicate=NHOSTS, +2.7% MFU),
# activation-checkpoint selective, CHECKPOINT_ASYNC_MODE=disabled (XPU async
# broken), compile on. Run INSIDE a PBS allocation (like tying_1b.sh); it uses
# `ezpz launch` to fan out across the allocated nodes.
#
# Usage (inside an interactive PBS alloc, N>=... nodes; N=32 for the ~10B budget):
#   ANNEAL_BASES="mds olmo" ANNEAL_STEPS=1600 bash anneal_ab.sh
# Override ANNEAL_BASES / ANNEAL_STEPS / ANNEAL_LBS to change scope.
set -o pipefail

REPO="${REPO:-/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan}"
LOGDIR="${LOGDIR:-$REPO/outputs/logs/anneal-ab}"
mkdir -p "$LOGDIR"

export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128 no_proxy=localhost,127.0.0.1,*.alcf.anl.gov
export ZE_FLAT_DEVICE_HIERARCHY=FLAT PYTORCH_XPU_ALLOC_CONF=expandable_segments:True CCL_LOG_LEVEL=ERROR
export CHECKPOINT_ASYNC_MODE=disabled
# Base DCPs (absolute; the config fails loud at build time if these are wrong).
export MDS_ANNEAL_BASE="${MDS_ANNEAL_BASE:-$REPO/outputs/checkpoints/agpt-2b-mds-gs138650/step-0}"
export OLMO_ANNEAL_BASE="${OLMO_ANNEAL_BASE:-$REPO/outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144/step-92859}"

cd "$REPO" || exit 9
curl -fsSL https://bit.ly/ezpz-utils -o /tmp/ezu.sh 2>/dev/null
source /tmp/ezu.sh >/dev/null 2>&1
ezpz_setup .venv >/dev/null 2>&1
echo "torch=$(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null)"

# Pre-warm the open-web-math HF hub cache on the HEAD node BEFORE launching the
# distributed job. The in-training rank-0 prefetch (datasets.py
# _rank0_prefetch_then_barrier) hits the hub live; a transient HfHubHTTPError
# there (rate-limit / proxy blip) kills rank 0, and all other ranks then die at
# the following dist.barrier (job 12471848 died exactly this way at 384 ranks).
# A serial, retried warm-up here makes the in-training fetch a warm-cache hit.
export HF_HUB_ENABLE_HF_TRANSFER=0
_ds="${_MDS_ANNEAL_DATASET:-open-web-math/open-web-math}"
for attempt in 1 2 3 4 5; do
  if python3 -c "
from datasets import load_dataset
ds = load_dataset('open-web-math/open-web-math', split='train', streaming=True)
next(iter(ds))
print('owm cache warm')
" 2>/dev/null; then
    echo "prewarm: open-web-math cache warm (attempt $attempt)"; break
  fi
  echo "prewarm: attempt $attempt failed (transient hub error); retrying in 30s"
  sleep 30
done

# NHOSTS from the PBS nodefile; HSDP shards intra-node (12 tiles) and replicates
# across nodes. dp_replicate * dp_shard = world size, so dp_replicate = NHOSTS.
NHOSTS="${NHOSTS:-$(wc -l < "${PBS_NODEFILE:-/dev/null}" 2>/dev/null)}"
NHOSTS="${NHOSTS:-2}"
ANNEAL_STEPS="${ANNEAL_STEPS:-1600}"
ANNEAL_LBS="${ANNEAL_LBS:-2}"
ANNEAL_BASES="${ANNEAL_BASES:-mds olmo}"
echo "NHOSTS=$NHOSTS  steps=$ANNEAL_STEPS  lbs=$ANNEAL_LBS  bases='$ANNEAL_BASES'"

summ() {  # tag logfile
  local t=$1 f=$2
  local loads=$(grep -ac "Finished loading the checkpoint" "$f")
  local first=$(grep -a "step: " "$f" 2>/dev/null | sed -E "s/\x1b\[[0-9;]*m//g" | grep -oE "step: *[0-9]+  loss: *[0-9.]+" | head -1)
  local last=$(grep -a "step: " "$f" 2>/dev/null | sed -E "s/\x1b\[[0-9;]*m//g" | grep -oE "step: *[0-9]+  loss: *[0-9.]+" | tail -1)
  # Real save-log markers (checkpoint.py): "Saving the checkpoint.",
  # "Saving a model only checkpoint ...", "Saving a full checkpoint at last
  # step ...". The v1 flaw this experiment fixes was ZERO saves, so this
  # counter is the guardrail -- it must match strings the code actually emits.
  local saved=$(grep -acE "Saving the checkpoint|Saving a (model only|full) checkpoint" "$f")
  local fail=$(grep -acE "OutOfMemory|OUT_OF_RESOURCES|died from signal|Traceback" "$f")
  echo "[$t] base_loaded=$loads saves=$saved | first=$first last=$last | fail=$fail"
}

run() {  # tag config
  local tag="$1" config="$2"
  local log="$LOGDIR/${tag:-arm}.log"
  echo ""; echo "=== $tag ($config), $ANNEAL_STEPS steps, HSDP shard12/rep$NHOSTS -> $log ==="
  # NOTE: NO --checkpoint.load-only (it disables ALL saving; that was the v1
  # flaw). The config forks the base via initial_load_path +
  # initial_load_model_only and saves via its own checkpoint.enable/interval;
  # we only OVERRIDE interval down to 100. AC is left at the config's own
  # setting (the base builder sets it; identical across a base's flat+wsd arms,
  # so it never confounds the A/B). compile.enable is already True in-config.
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module=ezpz.agpt --config="$config" \
    --training.steps="$ANNEAL_STEPS" --training.local-batch-size="$ANNEAL_LBS" \
    --debug.seed=42 --metrics.log-freq=10 \
    --parallelism.tensor-parallel-degree=1 \
    --parallelism.data-parallel-shard-degree=12 \
    --parallelism.data-parallel-replicate-degree="$NHOSTS" \
    --checkpoint.interval=100 --checkpoint.async-mode=disabled \
    --dataloader.num-workers=0 \
    > "$log" 2>&1
  summ "$tag" "$log"
}

for base in $ANNEAL_BASES; do
  run "${base}-flat" "agpt_2b_${base}_anneal_flat"
  run "${base}-wsd"  "agpt_2b_${base}_anneal_wsd"
done
echo "ANNEAL_AB_DONE"
