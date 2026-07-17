#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -j oe
# Backfill the 20B-512 chain with the MODERN eval suite (2026-07 landscape
# review). Split by cost: MMLU (5-shot) + ARC-Challenge (25-shot) are
# loglikelihood tasks -> run across the FULL ladder. GSM8K is generative
# (~3h/ckpt on 20B, generate_until) -> run on the LATEST ckpt only (the one
# we would cite). skip-guard is content-aware, so re-running merges tasks in.
cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"
# fast loglikelihood tasks across the full ladder
STEPS="1000 2000 3000 4000 5000 6000" \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288 \
LABEL=512n \
SHOTS_SPEC="5:mmlu;25:arc_challenge" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh
# GSM8K (generative, slow) on the latest ckpt only
STEPS="6000" \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288 \
LABEL=512n \
SHOTS_SPEC="5:gsm8k" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh
