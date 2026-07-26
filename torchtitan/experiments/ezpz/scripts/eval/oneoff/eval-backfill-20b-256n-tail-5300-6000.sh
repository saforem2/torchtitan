#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -j oe
# Backfill the 20B-256 chain TAIL (steps 5300-6000). The ladder was evaluated
# through 5200 (200-step cadence), then nothing until the frontier at 6000 --
# this fills the whole tail with the commonsense-7 + modern suite. Content-aware
# skip-guard merges. capacity queue, 1 node.
cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"
# commonsense-7 (0-shot) across the tail
STEPS="5300 5400 5500 5600 5700 5800 5900 6000" \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144 \
LABEL=256n \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh
# modern loglikelihood (mmlu-5, arc_challenge-25) across the tail
STEPS="5300 5400 5500 5600 5700 5800 5900 6000" \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144 \
LABEL=256n \
SHOTS_SPEC="5:mmlu;25:arc_challenge" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh
# gsm8k (generative, slow) on the latest ckpt only
STEPS="6000" \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144 \
LABEL=256n \
SHOTS_SPEC="5:gsm8k" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh
