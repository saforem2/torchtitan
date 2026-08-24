#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N smoke-30b-compile
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -l select=64
#PBS -q debug-scaling
#PBS -j oe
#
# Can the 30B COMPILE at the LR-finder's shape? Answer this on 64 debug nodes
# before spending a 2088-node allocation on the assumption.
#
# THE DOUBT. docs/live/dashboard.md says "torch.compile OOM at 512N -- 2B
# OOMs on GPU, 80B OOMs on CPU. Use --compile.no-enable for 512+ node jobs."
# I initially took that at face value and hardcoded --compile.no-enable into
# the 4x512N umbrella, which cost ~27% throughput (exp05: 340 vs 466 tps) and,
# worse, would have calibrated an LR on an uncompiled model for a compiled
# production run.
#
# THE NOTE IS ALREADY CONTRADICTED. Umbrella 8764675's trainer logs contain
# ZERO `compile.no-enable` occurrences for trainer-0 (2b-n512), trainer-1
# (20b-n512) and trainer-3 (2b-n512) -- the very 2B the note says OOMs at 512N
# has been training compiled at 512N continuously. So the prior is that compile
# is fine and the note is stale.
#
# WHAT THIS STILL DOES NOT SETTLE, and why the smoke is worth an hour: the 30B
# has never been compiled ABOVE 64N (exp06's ladder tops out there), and it has
# never been compiled at LBS=1 -- the live chain runs LBS=5, and LBS=1 is what
# a pinned GBS=6144 forces at 512N. Compile memory is dominated by the graph
# and the per-rank activation working set, so LBS=1 should be the EASIEST case,
# but "should" is what this job replaces.
#
# 64 nodes is the largest debug-scaling allows (nodect<=256, walltime<=1h) that
# still leaves the 1h budget comfortable, and it is the top of exp06's measured
# ladder so the result is directly comparable to 25.54% MFU there.
#
# THIS IS NOT A FULL-SCALE PROOF. It cannot rule out a failure that only appears
# at 6144 ranks. What it rules out is the far more likely case -- that the 30B
# graph simply cannot compile at LBS=1 -- for one hour instead of 2088 nodes.
#
# PASS: reaches step 3+ compiled, no OOM, memory well under ~90%.
# FAIL: OOM or a compile error -> set LRF_NO_COMPILE=1 on the umbrella.

# PBS scripts must NOT use `set -euo pipefail` per CLAUDE.md.
set -o pipefail

MAIN=/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz
cd "${PBS_O_WORKDIR:-$MAIN}" || exit 1

if ! command -v module >/dev/null 2>&1 || [[ -z "${MODULEPATH:-}" ]]; then
    [[ -r /etc/bash.bashrc.local ]] && source /etc/bash.bashrc.local
fi
module load oneapi/release/2025.3.1 hdf5 pti-gpu
export PATH="/opt/pbs/bin:${PATH}"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export ftp_proxy="${ftp_proxy:-http://proxy.alcf.anl.gov:3128}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,*.alcf.anl.gov,*.aurora.alcf.anl.gov}"

set +u
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
set -u

source .venv/bin/activate
ezpz yeet --src .venv.tar.gz || exit 1
deactivate
source /tmp/.venv/bin/activate

NNODES="${NHOSTS:-64}"
DP=$(( NNODES * 12 ))
# LBS=1 mirrors what a pinned GBS=6144 forces at 512N. GBS here is just dp*1 --
# the point is the COMPILE at LBS=1, not the batch, and using a GBS that does
# not divide would only add gradient accumulation noise to a memory question.
GBS=$DP

echo "=========================================================="
echo "[smoke] 30B COMPILED at ${NNODES}N, LBS=1, seq=4096, full AC"
echo "[smoke]   dp=$DP  GBS=$GBS"
echo "[smoke]   PASS = reaches step 3+, no OOM"
echo "=========================================================="

ezpz launch \
    --nproc "$DP" \
    --nproc_per_node 12 \
    --auto-retry \
    --spare-nodes auto \
    --timeout 1800 \
    -- \
    python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.agpt \
    --config agpt_30b \
    --job.dump-folder "outputs/_smoke/30b-compile-${PBS_JOBID%%.*}" \
    --optimizer adamw \
    --optimizer.lr 1e-5 \
    --training.steps 5 \
    --training.local_batch_size 1 \
    --training.global_batch_size "$GBS" \
    --training.seq_len 4096 \
    --metrics.log_freq 1 \
    --checkpoint.no-enable \
    --validator.no-enable \
    --dataloader.dataset blendcorpus \
    --dataloader.dataset_path torchtitan/experiments/ezpz/data-lists/aurora/olmo-mix-1124.txt \
    activation-checkpoint:full
rc=$?

echo ""
echo "=========================================================="
echo "[smoke] launch rc=$rc  -- but READ THE STEP LINES, not this."
echo "  step 3+ with a memory %% well under 90  -> compile is fine at LBS=1;"
echo "     leave the umbrella compiled (its default)."
echo "  OOM / compile error                     -> submit the umbrella with"
echo "     LRF_NO_COMPILE=1 and note it here."
echo "=========================================================="
