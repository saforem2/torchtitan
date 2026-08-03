#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=48:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -j oe
# Backfill the MODERN eval block (MMLU-57 loglikelihood + gsm8k generative) on
# the recent 20B-512 tail. Commonsense-7 is complete to step-6500; the modern
# block only reached 6100 in the last pass (8714837 finished Exit 0 but its
# modern phase covered 5500-6100 only). This fills 6200-6500. capacity queue at
# 48h so the slow MMLU pass cannot be walltime-killed. Content-aware skip-guard
# merges into existing results.json; HF conversions from the commonsense pass are
# reused. 512N ckpts live in the default agpt-20b-v2 repo (no REPO override).
cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"
# MMLU-5 + ARC-C-25 (loglikelihood) on the steps missing the modern block
STEPS="6200 6300 6400 6500" \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288 \
LABEL=512n \
SHOTS_SPEC="5:mmlu;25:arc_challenge" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh
# gsm8k (generative, slow ~3h/ckpt) on the latest ckpt only
STEPS="6500" \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288 \
LABEL=512n \
SHOTS_SPEC="5:gsm8k" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh
