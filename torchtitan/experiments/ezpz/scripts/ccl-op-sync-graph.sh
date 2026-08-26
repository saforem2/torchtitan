#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=00:40:00
#PBS -l filesystems=home:flare
#PBS -N ccl-op-sync-graph
#PBS -j oe

# Does CCL_OP_SYNC=1 let an XPU graph capture a oneCCL collective?
#
# The open blocker in xpu-graphs-block-oneccl-collectives.md is:
#   "wait method cannot be used for an event associated with a command graph"
# CCL_OP_SYNC=1 forces synchronous collectives, so there should be no async
# event for the graph to choke on. Filed as an Intel ask; never tested.
#
# Note the RL code already unsets CCL_OP_SYNC in five places, so it is a known
# lever here -- just used in the other direction.
set -o pipefail

cd /flare/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz || exit 1
source .venv/bin/activate || exit 1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

for setting in "unset" "0" "1"; do
    echo "================ CCL_OP_SYNC=$setting ================"
    if [[ "$setting" == "unset" ]]; then
        unset CCL_OP_SYNC
    else
        export CCL_OP_SYNC="$setting"
    fi
    timeout 400 python3 -m ezpz.launch --nproc 2 --nproc_per_node 2 \
        -- python3 -m torchtitan.experiments.ezpz.tests.probe_xpu_graph_collective \
        2>&1 | grep -aE "OUTSIDE capture|INSIDE capture|CAPTURES|FAIL|Error|wait method" | head -8
    echo "  rc=$?"
    echo
done
