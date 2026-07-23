#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -j oe
# Backfill the 2B-512 chain with the MODERN eval suite. Cost split: MMLU+ARC-C
# full ladder (loglikelihood), GSM8K latest-only (generative). 2B generation
# is faster than 20B but still the dominant cost, so keep it latest-only.
cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"
STEPS="5000 10000 15000 20000 25000 30000 35000 37000 39000 39600" \
CKPT_NAME=agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288 \
LABEL=512n \
SHOTS_SPEC="5:mmlu;25:arc_challenge" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-2b-v2.sh
STEPS="35000 39600" \
CKPT_NAME=agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288 \
LABEL=512n \
SHOTS_SPEC="5:gsm8k" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-2b-v2.sh
