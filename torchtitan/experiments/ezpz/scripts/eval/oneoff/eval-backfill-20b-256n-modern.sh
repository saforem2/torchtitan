#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -j oe
# Backfill the 20B-256 chain with the MODERN eval suite. 256N chain ckpts
# live in the agpt-20b-n256 clone -> REPO override. Cost split: MMLU+ARC-C
# full ladder (loglikelihood), GSM8K latest-only (generative, ~3h/ckpt).
cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"
STEPS="500 1000 2000 3000 4000 4200" \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144 \
REPO=/flare/AuroraGPT/foremans/runs/agpt-20b-n256/torchtitan-ezpz \
LABEL=256n \
SHOTS_SPEC="5:mmlu;25:arc_challenge" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh
STEPS="4200" \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144 \
REPO=/flare/AuroraGPT/foremans/runs/agpt-20b-n256/torchtitan-ezpz \
LABEL=256n \
SHOTS_SPEC="5:gsm8k" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh
