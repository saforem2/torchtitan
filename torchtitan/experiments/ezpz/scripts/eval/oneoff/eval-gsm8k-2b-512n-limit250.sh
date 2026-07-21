#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=04:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -j oe
cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"
STEPS="35000" CKPT_NAME=agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288 LABEL=512n \
  SHOTS_SPEC="5:gsm8k" LIMIT=250 \
  bash torchtitan/experiments/ezpz/scripts/eval/eval-2b-v2.sh
