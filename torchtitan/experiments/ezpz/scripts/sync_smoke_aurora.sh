#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N ezpz-sync-smoke-aurora
#PBS -l walltime=00:45:00
#PBS -l filesystems=flare:home
#PBS -l select=2
#PBS -q debug-scaling
#PBS -j oe
# Thin PBS wrapper — all logic lives in sync_smoke_alcf.sh. Queue is pinned
# to debug-scaling because the venv was built against the prod oneAPI image,
# not next-eval (see AGENTS.md).
#   qsub torchtitan/experiments/ezpz/scripts/sync_smoke_aurora.sh
exec bash "${PBS_O_WORKDIR:-$(pwd)}/torchtitan/experiments/ezpz/scripts/sync_smoke_alcf.sh"
