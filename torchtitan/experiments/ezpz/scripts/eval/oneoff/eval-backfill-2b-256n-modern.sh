#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -j oe
# Backfill the 2B-256 chain (the COMPLETED 2B model, step-92859 = 100% of the
# 4.67T target) with the MODERN eval suite. This chain only ever got the old
# commonsense suite + arc_challenge -- no MMLU, no GSM8K -- so it is not
# comparable to the 20B chains or to peers (OLMo-2 etc.). MMLU (5-shot) +
# ARC-C (25-shot) are loglikelihood -> sampled ladder every ~5k steps + final.
# GSM8K is generative (slow) -> latest ckpt only (the one we would cite).
# Ckpts live in the agpt-2b-v2 clone (V2_REPO default). skip-guard is
# content-aware, so re-running merges the new tasks into existing results.json.
# NOTE: this chain's early ckpts were rotated away -- the earliest surviving
# step is 35600, so the modern ladder starts there (steps verified on disk).
cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"
STEPS="35600 45100 54600 64100 73600 83100 92600 92859" \
CKPT_NAME=agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144 \
LABEL=256n \
SHOTS_SPEC="5:mmlu;25:arc_challenge" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-2b-v2.sh
STEPS="92859" \
CKPT_NAME=agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144 \
LABEL=256n \
SHOTS_SPEC="5:gsm8k" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-2b-v2.sh
