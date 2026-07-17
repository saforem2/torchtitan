#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -j oe

# Backfill the 20B-512 chain with the MODERN eval suite (2026-07 landscape
# review): MMLU (5-shot), GSM8K (5-shot), ARC-Challenge (25-shot) -- the
# base-model benchmarks peers (SmolLM3, Llama-3.2, OLMo-2) report but we didn't.
# SHOTS_SPEC drives per-task few-shot (a single lm-eval call can't mix shots).
# skip-if-results.json-exists makes this restart-safe.

cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"

STEPS="1000 2000 3000 4000 5000 6000" \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288 \
LABEL=512n \
SHOTS_SPEC="5:mmlu,gsm8k;25:arc_challenge" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh
