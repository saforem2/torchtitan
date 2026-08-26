#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=00:40:00
#PBS -l filesystems=home:flare
#PBS -N agpt-vs-moe-tp2
#PBS -j oe

# agpt and moe carry a BYTE-IDENTICAL unflatten, yet agpt TP=2 passes and moe
# TP=2 dies. Run both at TP=2 with the same probe and diff the output.
set -o pipefail
cd /flare/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz || exit 1
export EZPZ_MLA_PLACEMENT_PROBE=1

FLAGS='--training.max-context-length=512 --training.num-tokens-per-microbatch-per-dp-rank=512 --parallelism.tensor-parallel-degree=2'

echo "##### AGPT TP=2 #####"
SMOKE_CONFIGS="ezpz.agpt:agpt_debugmodel:$FLAGS" SMOKE_STEPS=1 \
  bash torchtitan/experiments/ezpz/scripts/sync_smoke.sh 2>&1 \
  | grep -aE 'AGPT-PROBE|MLA-PROBE|VERDICT|rc=|Error' | head -20

echo
echo "##### MOE TP=2 #####"
SMOKE_CONFIGS="ezpz.moe:moe_debugmodel:$FLAGS" SMOKE_STEPS=1 \
  bash torchtitan/experiments/ezpz/scripts/sync_smoke.sh 2>&1 \
  | grep -aE 'AGPT-PROBE|MLA-PROBE|VERDICT|rc=|Error' | head -20
