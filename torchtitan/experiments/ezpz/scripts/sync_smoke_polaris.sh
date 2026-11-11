#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N ezpz-sync-smoke-polaris
#PBS -l walltime=00:45:00
#PBS -l filesystems=home:eagle
#PBS -l select=2
#PBS -q debug
#PBS -j oe

# Post-sync / pre-merge smoke on polaris, 2 nodes. All logic lives in
# sync_smoke_alcf.sh; this file only carries the PBS headers.
#
#
# Usage (from the tree to test):
#   qsub torchtitan/experiments/ezpz/scripts/sync_smoke_polaris.sh
#   qsub -v SMOKE_VENV=/path/to/venv,SMOKE_STEPS=10 torchtitan/experiments/ezpz/scripts/sync_smoke_polaris.sh
# Knobs and VERDICT values: see sync_smoke_alcf.sh.

export SMOKE_MACHINE=polaris
exec bash "${PBS_O_WORKDIR:-$(pwd)}/torchtitan/experiments/ezpz/scripts/sync_smoke_alcf.sh"
