#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -q validation
#PBS -l select=2
#PBS -l walltime=00:55:00
#PBS -l filesystems=home:flare
#PBS -N rc-matrix
#PBS -j oe

# What is NOT yet tested on the Aurora frameworks/2026.1.0 RC.
#
# Everything green so far (job 8784615) is agpt_debugmodel at TP=1. This adds
# the corners that today's work actually touched:
#
#   agpt TP=2   -- known to need --compile.no-enable on torch 2.13 (the
#                  DeviceMesh AOT assertion). Untested on the RC, where the
#                  compile regression may or may not still be there.
#   agpt TP=2 compiled -- the RC was supposed to FIX that assertion (RC4 on
#                  Sunspot did). Never checked on Aurora.
#   moe  TP=1   -- moe has never run on the RC at all.
#   moe  TP=2   -- the corner fixed in 6e4e1996f, only ever verified on the
#                  production .venv stack, never on the RC.
set -o pipefail

REPO=/flare/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz
cd $REPO || exit 1

module load frameworks/2026.1.0
FW=/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0
# libglog.so.0 ships in the module tree but the modulefile does not export
# it; without this every import torch dies. docs/guides/known-bugs/fw-rc-libglog-not-on-loader-path.md
export LD_LIBRARY_PATH="$FW/lib:$LD_LIBRARY_PATH"
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env
python3 -c 'import torch; print("torch", torch.__version__, "| xpu", torch.xpu.is_available(), "|", torch.xpu.device_count(), "devices")'
echo

NPROC=${NPROC:-4}
BASE="--training.steps=5 --training.max-context-length=512 \
--training.num-tokens-per-microbatch-per-dp-rank=512 \
--checkpoint.no-enable --debug.seed=42 --debug.deterministic"

run () {   # run <tag> <module> <config> [extra...]
    local tag="$1" mod="$2" cfg="$3"; shift 3
    local out; out=$(mktemp -d "/tmp/rcm-$tag-XXXX")
    echo "##### $tag #####"
    timeout 600 python3 -m ezpz.launch --nproc "$NPROC" --nproc_per_node "$NPROC" \
      -- python3 -m torchtitan.experiments.ezpz.train \
         --module="$mod" --config="$cfg" $BASE --dump-folder="$out" "$@" 2>&1 \
      | grep -aE 'step: |Error|Exception|assert|out_src_shardings' | tail -12
    echo
}

run "agpt-tp1"          ezpz.agpt agpt_debugmodel --compile.no-enable
run "agpt-tp2-nocomp"   ezpz.agpt agpt_debugmodel --compile.no-enable --parallelism.tensor-parallel-degree=2
run "agpt-tp2-COMPILED" ezpz.agpt agpt_debugmodel --parallelism.tensor-parallel-degree=2
run "moe-tp1"           ezpz.moe  moe_debugmodel  --compile.no-enable
run "moe-tp2"           ezpz.moe  moe_debugmodel  --compile.no-enable --parallelism.tensor-parallel-degree=2
