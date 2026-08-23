#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=48:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -j oe
# Backfill the MODERN eval block (MMLU-57 loglikelihood + gsm8k generative) on
# the recent 20B-256 tail. The commonsense-7 ladder is already complete to the
# live tip (step-6800); only the modern block is missing on 5900-6800 because
# job 8714836 died Exit -29 (walltime) mid-MMLU under the old 12h cap. This runs
# in the capacity queue at 48h so the slow MMLU loglikelihood pass (56k+ requests
# per step) cannot be walltime-killed again. Content-aware skip-guard merges into
# the existing per-step results.json, so already-present tasks are skipped and the
# HF conversions cached by the commonsense pass are reused (no re-convert).
# 256N chain ckpts live in the relocated agpt-20b-n256 clone -> REPO override.
cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"
N256_REPO=/flare/AuroraGPT/foremans/runs/agpt-20b-n256/torchtitan-ezpz
# MMLU-5 + ARC-C-25 (loglikelihood) across the tail that has commonsense but no modern
STEPS="5900 6100 6400 6500 6600 6700 6800" \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144 \
REPO=$N256_REPO \
LABEL=256n \
SHOTS_SPEC="5:mmlu;25:arc_challenge" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh
# gsm8k (generative, slow ~3h/ckpt) on the latest ckpt only
STEPS="6800" \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144 \
REPO=$N256_REPO \
LABEL=256n \
SHOTS_SPEC="5:gsm8k" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh
