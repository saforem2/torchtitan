#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=00:50:00
#PBS -l filesystems=home:flare
#PBS -N lm-probe
#PBS -j oe
set -o pipefail
cd /flare/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz || exit 1
source .venv/bin/activate || exit 1
export EZPZ_LM_PROBE=1 ZE_FLAT_DEVICE_HIERARCHY=FLAT
NPROC=${NPROC:-4}
COMMON="--training.steps=5 --training.max-context-length=512 \
--training.num-tokens-per-microbatch-per-dp-rank=512 \
--parallelism.tensor-parallel-degree=2 --checkpoint.no-enable \
--debug.seed=42 --debug.deterministic"
run () {
  local tag=$1 mod=$2 cfg=$3; shift 3
  local out; out=$(mktemp -d "/tmp/lm-$tag-XXXX")
  echo "##### $tag #####"
  timeout 700 python3 -m ezpz.launch --nproc "$NPROC" --nproc_per_node "$NPROC" \
    -- python3 -m torchtitan.experiments.ezpz.train --module=$mod --config=$cfg \
       $COMMON --dump-folder="$out" "$@" 2>&1 \
    | grep -aE "LM-PROBE|step: |out_src_shardings|Error|Exception|assert" | tail -30
  echo
}
run agpt ezpz.agpt agpt_debugmodel --compile.no-enable
run moe-tp1 ezpz.moe moe_debugmodel --compile.no-enable --parallelism.tensor-parallel-degree=1
run moe  ezpz.moe  moe_debugmodel --compile.no-enable
