#!/bin/bash
# Data-mix A/B (stage-2 mid-training): the anneal showed flat>wsd => DATA is the
# lever. These arms fork the SAME MDS base at the SAME winning constant LR 2e-6
# and differ ONLY in the training data mix. Wave 1 = single-corpus arms:
#   owm  = agpt_2b_mds_mix_owm  (open-web-math 100%, == anneal flat winner, control)
#   edu  = agpt_2b_mds_mix_edu  (fineweb_edu_local 100%, diversity extreme)
# Reuses the anneal recipe verbatim (HSDP shard12/rep=NHOSTS, LBS2, seed42,
# ckpt interval 100, async disabled, offline HF cache). Run INSIDE a PBS alloc.
#
# Usage (inside a 32N interactive alloc):
#   MIX_ARMS="owm edu" MIX_STEPS=1600 bash mix_ab.sh
set -o pipefail

REPO="${REPO:-/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan}"
LOGDIR="${LOGDIR:-$REPO/outputs/logs/mix-ab}"
mkdir -p "$LOGDIR"

export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128 no_proxy=localhost,127.0.0.1,*.alcf.anl.gov
export ZE_FLAT_DEVICE_HIERARCHY=FLAT PYTORCH_XPU_ALLOC_CONF=expandable_segments:True CCL_LOG_LEVEL=ERROR
export CHECKPOINT_ASYNC_MODE=disabled
# HF served offline from the shared precached cache (no 429 at 384 ranks -- see
# anneal_ab.sh header). owm is precached local parquet; fineweb_edu_local is a
# registered LOCAL parquet dir (no hub at all).
export HF_HOME="${HF_HOME:-/lus/tegu/projects/datasets/hf_cache}"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0
export MDS_ANNEAL_BASE="${MDS_ANNEAL_BASE:-$REPO/outputs/checkpoints/agpt-2b-mds-gs138650/step-0}"

cd "$REPO" || exit 9
curl -fsSL https://bit.ly/ezpz-utils -o /tmp/ezu.sh 2>/dev/null
source /tmp/ezu.sh >/dev/null 2>&1
ezpz_setup .venv >/dev/null 2>&1
echo "torch=$(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null)"

NHOSTS="${NHOSTS:-$(wc -l < "${PBS_NODEFILE:-/dev/null}" 2>/dev/null)}"
NHOSTS="${NHOSTS:-2}"
MIX_STEPS="${MIX_STEPS:-1600}"
MIX_LBS="${MIX_LBS:-2}"
MIX_ARMS="${MIX_ARMS:-owm edu}"
echo "NHOSTS=$NHOSTS steps=$MIX_STEPS lbs=$MIX_LBS arms='$MIX_ARMS'"

# Preflight: verify each arm's data source resolves BEFORE burning the alloc.
# owm must be precached local parquet; edu is a local dir (register_local_dataset).
preflight_ok=1
for arm in $MIX_ARMS; do
  case "$arm" in
    owm)
      python3 -c "
import torchtitan.experiments.ezpz.datasets as d
p=d.resolve_precached_parquet_dir('open-web-math/open-web-math'); assert p, 'owm not precached'
from datasets import load_dataset; next(iter(load_dataset('parquet',data_dir=p,split='train',streaming=True)))
print('owm OK')" 2>/dev/null || { echo "PREFLIGHT FAIL: owm"; preflight_ok=0; } ;;
    edu)
      python3 -c "
from datasets import load_dataset
next(iter(load_dataset('parquet',data_dir='/lus/tegu/projects/datasets/datasets/fineweb-edu-100BT/sample/100BT/',split='train',streaming=True)))
print('edu OK')" 2>/dev/null || { echo "PREFLIGHT FAIL: edu"; preflight_ok=0; } ;;
  esac
done
[ "$preflight_ok" = 1 ] || { echo "ERROR: a data source did not resolve; aborting before alloc burn." >&2; exit 7; }
echo "all mix data sources resolved"

summ() {
  local t=$1 f=$2
  local loads=$(grep -ac "Finished loading the checkpoint" "$f")
  local first=$(grep -a "step: " "$f" 2>/dev/null | sed -E "s/\x1b\[[0-9;]*m//g" | grep -oE "step: *[0-9]+  loss: *[0-9.]+" | head -1)
  local last=$(grep -a "step: " "$f" 2>/dev/null | sed -E "s/\x1b\[[0-9;]*m//g" | grep -oE "step: *[0-9]+  loss: *[0-9.]+" | tail -1)
  local saved=$(grep -acE "Saving the checkpoint|Saving a (model only|full) checkpoint" "$f")
  local fail=$(grep -acE "OutOfMemory|OUT_OF_RESOURCES|died from signal|Traceback" "$f")
  echo "[$t] base_loaded=$loads saves=$saved | first=$first last=$last | fail=$fail"
}

run() {
  local tag="$1" config="$2"
  local log="$LOGDIR/${tag:-arm}.log"
  # MIX_FOLDER_SUFFIX isolates a run's checkpoints from other runs of the SAME
  # config (e.g. a 2N smoke vs a 32N prod both use agpt_2b_mds_mix_edu whose
  # config folder is checkpoints/agpt-2b-mds-mix-<arm>). Without a suffix the
  # prod run AUTO-RESUMES the smoke's checkpoint, which has the WRONG dataloader
  # rank count (24 vs 384) -> "Missing key ... dataloader.dp_rank_80" crash at
  # load (job 12471928 died exactly so). Set MIX_FOLDER_SUFFIX=-smoke for smokes.
  local folder_override=""
  if [ -n "${MIX_FOLDER_SUFFIX:-}" ]; then
    folder_override="--checkpoint.folder=checkpoints/agpt-2b-mds-mix-${arm}${MIX_FOLDER_SUFFIX}"
  fi
  echo ""; echo "=== $tag ($config), $MIX_STEPS steps, HSDP shard12/rep$NHOSTS -> $log ${folder_override} ==="
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module=ezpz.agpt --config="$config" \
    --training.steps="$MIX_STEPS" --training.local-batch-size="$MIX_LBS" \
    --debug.seed=42 --metrics.log-freq=10 \
    --parallelism.tensor-parallel-degree=1 \
    --parallelism.data-parallel-shard-degree=12 \
    --parallelism.data-parallel-replicate-degree="$NHOSTS" \
    --checkpoint.interval=100 --checkpoint.async-mode=disabled \
    --dataloader.num-workers=0 \
    ${folder_override} \
    > "$log" 2>&1
  summ "$tag" "$log"
}

for arm in $MIX_ARMS; do
  run "mix-$arm" "agpt_2b_mds_mix_$arm"
done
echo "MIX_AB_DONE"
