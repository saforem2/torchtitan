#!/bin/bash --login
#PBS -A datascience
#PBS -N ezpz-sync-smoke-sirius
#PBS -l walltime=00:45:00
#PBS -l filesystems=home:tegu
#PBS -l select=2
#PBS -q workq
#PBS -j oe
# Thin PBS wrapper — all logic lives in sync_smoke_alcf.sh.
#   qsub torchtitan/experiments/ezpz/scripts/sync_smoke_sirius.sh
exec bash "${PBS_O_WORKDIR:-$(pwd)}/torchtitan/experiments/ezpz/scripts/sync_smoke_alcf.sh"
