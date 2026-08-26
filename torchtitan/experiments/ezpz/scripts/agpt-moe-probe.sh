#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=00:55:00
#PBS -l filesystems=home:flare
#PBS -N agpt-moe-probe
#PBS -j oe

# Measure the agpt side of the MoE TP>1 question.
#
# agpt and moe carry a byte-identical unflatten; moe TP=2 dies and agpt TP=2
# passes. moe is measured (job 8784667); agpt is not -- job 8784895 tried and
# was killed by walltime because sync_smoke.sh yeets a venv to /tmp first.
# On ONE node that is pure overhead: the repo .venv is already on flare.
# This launches ezpz directly instead.
set -o pipefail

cd /flare/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz || exit 1
source .venv/bin/activate || exit 1
export EZPZ_MLA_PLACEMENT_PROBE=1

# FLAT, like every other script here. A Max 1550 node is 6 cards x 2 tiles =
# 12 XPUs; ZE_FLAT_DEVICE_HIERARCHY decides whether torch sees 12 flat devices
# or 6 composite ones. Without it torch.xpu.device_count() reports 6 and
# ezpz rejects --nproc 12 with "ngpus must be > 0 and <= 6" -- which I first
# misread as a debug-queue limit (job 8785479). It is not; it is this export.
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

NPROC=${NPROC:-4}          # TP=2 needs 2; 4 gives dp=2 x tp=2
echo "xpu devices visible: $(python3 -c 'import torch; print(torch.xpu.device_count())' 2>/dev/null)"
echo "launching $NPROC ranks (TP=2)"

echo "##### AGPT TP=2 #####"
timeout 900 python3 -m ezpz.launch --nproc "$NPROC" --nproc_per_node "$NPROC" \
  -- python3 -m torchtitan.experiments.ezpz.train \
     --module=ezpz.agpt --config=agpt_debugmodel $COMMON 2>&1 \
  | grep -aE 'AGPT-PROBE|MLA-PROBE|step: |Error|out_src_shardings' | head -20
echo "  AGPT_RC=$?"

echo
echo "##### MOE TP=2 #####"
timeout 900 python3 -m ezpz.launch --nproc "$NPROC" --nproc_per_node "$NPROC" \
  -- python3 -m torchtitan.experiments.ezpz.train \
     --module=ezpz.moe --config=moe_debugmodel $COMMON 2>&1 \
  | grep -aE 'AGPT-PROBE|MLA-PROBE|step: |Error|out_src_shardings' | head -20
echo "  MOE_RC=$?"
