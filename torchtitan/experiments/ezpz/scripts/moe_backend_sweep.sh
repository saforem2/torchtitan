#!/bin/bash
# Layer 2: 100-step training perf sweep across the 3 expert compute backends.
# Run on a compute node, NOT the login node. Uses the torch 2.13 venv recipe.
# Invoke via `ssh ... 'bash -lc /path/to/this'` so the login env (mpiexec,
# qstat, modules) is available.
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
cd /lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz
source <(curl -fsSL https://bit.ly/ezpz-utils)
ezpz_setup_job >/dev/null 2>&1
ezpz_setup_xpu >/dev/null 2>&1
source .venv/bin/activate

LOG_DIR=logs/moe-backend-bench
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d-%H%M%S)

CONFIGS=(
    moe_10b_2b_sdpa                    # grouped_mm (default)
    moe_10b_2b_sdpa_for_loop           # for_loop
    moe_10b_2b_sdpa_batched_mm_padded  # batched_mm_padded
)

for cfg in "${CONFIGS[@]}"; do
    LOG="$LOG_DIR/${cfg}-${TS}.log"
    echo "=== $cfg @ $(date '+%H:%M:%S') ==="
    echo "    log: $LOG"
    ezpz launch python3 -m torchtitan.experiments.ezpz.train \
        --module ezpz.moe --config "$cfg" \
        --training.steps 100 \
        --metrics.log_freq 10 \
        --checkpoint.no-enable \
        2>&1 | tee "$LOG"
    echo "=== $cfg done @ $(date '+%H:%M:%S') ==="
    echo
done

echo "=== ALL DONE @ $(date '+%H:%M:%S') ==="
echo "Logs in $LOG_DIR/*${TS}.log"
