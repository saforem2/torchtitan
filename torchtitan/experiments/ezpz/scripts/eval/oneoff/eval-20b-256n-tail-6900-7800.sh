#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=48:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -j oe
# Eval the 20B-256 tail that the 2026-08-05 umbrella (8714502) produced.
# Prior coverage stops at step-6800; the umbrella advanced the chain to
# step-7800, leaving steps 6900-7800 unevaluated. Runs the full ladder:
# commonsense-7 at 0-shot, then the modern block (MMLU-5 + ARC-C-25), then
# gsm8k on the tip only (generative, ~3h/ckpt -- too slow to run per step).
#
# Motivation: ARC-Challenge fell 0.4138 (step-4000) -> 0.2875 (step-6500),
# approaching the 0.25 chance floor, while train loss kept improving. The
# tokenizer-mismatch explanation was ruled out (gemma-7b assets match the
# gemma-tokenized olmo-mix-1124 training data; vocab 256128 vs 256000 is
# standard 128-alignment padding). These steps show whether the decline
# continues or bottoms out.
#
# 48h capacity queue: the MMLU-57 loglikelihood pass (56k+ requests/step at
# ~3.6s/it) was walltime-killed twice under the old 12h cap.
# 256N chain ckpts live in the relocated agpt-20b-n256 clone -> REPO override.
cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"
N256_REPO=/flare/AuroraGPT/foremans/runs/agpt-20b-n256/torchtitan-ezpz
TAIL_STEPS="6900 7000 7100 7200 7300 7400 7500 7600 7700 7800"

# commonsense-7 at 0-shot (fast, loglikelihood)
STEPS="$TAIL_STEPS" \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144 \
REPO=$N256_REPO \
LABEL=256n \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh

# modern block: MMLU-5 + ARC-C-25 (loglikelihood, slow)
STEPS="$TAIL_STEPS" \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144 \
REPO=$N256_REPO \
LABEL=256n \
SHOTS_SPEC="5:mmlu;25:arc_challenge" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh

# gsm8k (generative, ~3h/ckpt) on the tip only
STEPS="7800" \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144 \
REPO=$N256_REPO \
LABEL=256n \
SHOTS_SPEC="5:gsm8k" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh
