#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N ezpz-sync-smoke-aurora
#PBS -l walltime=00:45:00
#PBS -l filesystems=flare:home
#PBS -l select=2
#PBS -q debug-scaling
#PBS -j oe

# Post-sync / pre-merge smoke on aurora, 2 nodes. All logic lives in
# sync_smoke_alcf.sh; this file only carries the PBS headers.
#
# QUEUE CHOICE IS COUPLED TO THE VENV. next-eval runs the compute_aurora_test_*
# image; the other queues run compute_aurora_prod_*. A venv built on the prod
# image has no interpreter on next-eval, and every import dies with a
# misleading ModuleNotFoundError: No module named 'importlib.metadata'
# (jobs 8829080, 8829104). Check pyvenv.cfg's home line before changing it.
#
# Usage (from the tree to test):
#   qsub torchtitan/experiments/ezpz/scripts/sync_smoke_aurora.sh
#   qsub -v SMOKE_VENV=/path/to/venv,SMOKE_STEPS=10 torchtitan/experiments/ezpz/scripts/sync_smoke_aurora.sh
# Knobs and VERDICT values: see sync_smoke_alcf.sh.

export SMOKE_MACHINE=aurora
exec bash "${PBS_O_WORKDIR:-$(pwd)}/torchtitan/experiments/ezpz/scripts/sync_smoke_alcf.sh"
