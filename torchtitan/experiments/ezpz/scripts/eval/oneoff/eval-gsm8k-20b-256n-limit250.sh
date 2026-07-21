#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -j oe
cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"
STEPS="4200" CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144 \
  REPO=/flare/AuroraGPT/foremans/runs/agpt-20b-n256/torchtitan-ezpz LABEL=256n \
  SHOTS_SPEC="5:gsm8k" LIMIT=250 \
  bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh
