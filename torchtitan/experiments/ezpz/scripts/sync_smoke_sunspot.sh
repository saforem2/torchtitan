#!/bin/bash --login
#PBS -A datascience
#PBS -N ezpz-sync-smoke-sunspot
#PBS -l walltime=00:45:00
#PBS -l filesystems=tegu:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe
# Thin PBS wrapper — all logic lives in sync_smoke_alcf.sh. Override the
# checkout venv with qsub -v SMOKE_VENV=/path/to/venv if needed.
#   qsub torchtitan/experiments/ezpz/scripts/sync_smoke_sunspot.sh
#   qsub -v SMOKE_VENV=/path/to/venv,SMOKE_STEPS=10 torchtitan/experiments/ezpz/scripts/sync_smoke_sunspot.sh
exec bash "${PBS_O_WORKDIR:-$(pwd)}/torchtitan/experiments/ezpz/scripts/sync_smoke_alcf.sh"
