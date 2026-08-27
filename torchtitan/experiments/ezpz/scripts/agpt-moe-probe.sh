#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=00:55:00
#PBS -l filesystems=home:flare
#PBS -N agpt-det-probe
#PBS -j oe

# Does agpt reach the 3D unflatten at TP=2, and with what shape?
#
# History of this probe, all harness bugs rather than findings:
#   8785479  hardcoded --nproc 12 without ZE_FLAT_DEVICE_HIERARCHY=FLAT, so
#            torch saw 6 composite devices and ezpz refused. A node is 6 cards
#            x 2 tiles = 12 XPUs; FLAT is what makes torch see 12.
#   8785489  no --debug.deterministic, so agpt hit the compile+AC+TP AOT
#            assertion that sync_smoke's arm never reaches.
#   8785507  every arm shares dump_folder=./outputs -> checkpoint/, so one
#            model loaded the other's checkpoint:
#              Size mismatch ... saved [512, 2048] vs current [256, 256]
#
# Arms: agpt+det, agpt+det+nocompile (is compile the AOT trigger?), moe+det.
set -o pipefail

cd /flare/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz || exit 1
source .venv/bin/activate || exit 1
export EZPZ_MLA_PLACEMENT_PROBE=1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

NPROC=${NPROC:-4}
echo "xpu devices visible: $(python3 -c 'import torch; print(torch.xpu.device_count())' 2>/dev/null)"
echo "launching $NPROC ranks (TP=2)"
echo

COMMON="--training.steps=1 --training.max-context-length=512 \
--training.num-tokens-per-microbatch-per-dp-rank=512 \
--parallelism.tensor-parallel-degree=2 --checkpoint.no-enable \
--debug.seed=42 --debug.deterministic"

run () {   # run <tag> <module> <config> [extra...]
    local tag="$1" mod="$2" cfg="$3"; shift 3
    local out; out=$(mktemp -d "/tmp/probe-${tag}-XXXX")
    echo "##### ${tag} #####"
    timeout 700 python3 -m ezpz.launch --nproc "$NPROC" --nproc_per_node "$NPROC" \
      -- python3 -m torchtitan.experiments.ezpz.train \
         --module="$mod" --config="$cfg" $COMMON \
         --dump-folder="$out" "$@" 2>&1 \
      | grep -aE 'AGPT-PROBE|MLA-PROBE|step: |vc_check|out_src_shardings|^[A-Za-z]*Error' \
      | head -14
    echo "  rc=$?"
    echo
}

run "agpt-det"           ezpz.agpt agpt_debugmodel
run "agpt-det-nocompile" ezpz.agpt agpt_debugmodel --compile.no-enable
run "moe-det"            ezpz.moe  moe_debugmodel
