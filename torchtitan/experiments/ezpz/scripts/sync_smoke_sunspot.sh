#!/bin/bash --login
#PBS -A datascience
#PBS -N ezpz-sync-smoke-sunspot
#PBS -l walltime=00:45:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe

# Post-sync / pre-merge smoke on sunspot, 2 nodes. All logic lives in
# sync_smoke_alcf.sh; this file only carries the PBS headers.
#
#
# Usage (from the tree to test):
#   qsub torchtitan/experiments/ezpz/scripts/sync_smoke_sunspot.sh
#   qsub -v SMOKE_VENV=/path/to/venv,SMOKE_STEPS=10 torchtitan/experiments/ezpz/scripts/sync_smoke_sunspot.sh
# Knobs and VERDICT values: see sync_smoke_alcf.sh.

export SMOKE_MACHINE=sunspot
exec bash "${PBS_O_WORKDIR:-$(pwd)}/torchtitan/experiments/ezpz/scripts/sync_smoke_alcf.sh"
