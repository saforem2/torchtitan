#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=48:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -j oe
# Eval the 20B-512 tail that the 2026-08-05 umbrella (8714502) produced.
# Prior coverage stops at step-6500; the umbrella advanced the chain to
# step-7100, leaving steps 6550-7100 unevaluated. Runs the full ladder:
# commonsense-7 at 0-shot, then the modern block (MMLU-5 + ARC-C-25), then
# gsm8k on the tip only (generative, ~3h/ckpt -- too slow to run per step).
#
# Companion to eval-20b-256n-tail-6900-7800.sh; see that script for the
# ARC-Challenge decline this is meant to characterize. 512N ckpts live in
# the default agpt-20b-v2 repo, so no REPO override is needed.
#
# 48h capacity queue: the MMLU-57 loglikelihood pass (56k+ requests/step at
# ~3.6s/it) was walltime-killed twice under the old 12h cap.
cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"
TAIL_STEPS="6550 6600 6700 6800 6900 7000 7100"

# commonsense-7 at 0-shot (fast, loglikelihood)
STEPS="$TAIL_STEPS" \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288 \
LABEL=512n \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh

# modern block: MMLU-5 + ARC-C-25 (loglikelihood, slow)
STEPS="$TAIL_STEPS" \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288 \
LABEL=512n \
SHOTS_SPEC="5:mmlu;25:arc_challenge" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh

# gsm8k (generative, ~3h/ckpt) on the tip only
STEPS="7100" \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288 \
LABEL=512n \
SHOTS_SPEC="5:gsm8k" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh
