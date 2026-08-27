#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare
#PBS -N moe-tp2-probe
#PBS -j oe

# Measure the DTensor placement at every step between SDPA and wo at TP=2.
# Four theories about this bug were wrong; this prints the answer instead.
set -o pipefail
cd /flare/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz || exit 1
export EZPZ_MLA_PLACEMENT_PROBE=1

CFG='ezpz.moe:moe_debugmodel:--training.max-context-length=512 --training.num-tokens-per-microbatch-per-dp-rank=512 --parallelism.tensor-parallel-degree=2'
SMOKE_CONFIGS="$CFG" SMOKE_STEPS=1 \
  bash torchtitan/experiments/ezpz/scripts/sync_smoke.sh
