#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N ezpz-sync-smoke-polaris
#PBS -l walltime=00:45:00
#PBS -l filesystems=home:eagle
#PBS -l select=2
#PBS -q debug
#PBS -j oe
# Thin PBS wrapper — all logic lives in sync_smoke_alcf.sh.
#   qsub torchtitan/experiments/ezpz/scripts/sync_smoke_polaris.sh
exec bash "${PBS_O_WORKDIR:-$(pwd)}/torchtitan/experiments/ezpz/scripts/sync_smoke_alcf.sh"
