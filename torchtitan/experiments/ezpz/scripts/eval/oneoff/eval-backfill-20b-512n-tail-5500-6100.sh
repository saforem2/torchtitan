#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -j oe
# Backfill the 20B-512 chain TAIL (steps 5500-6100), which the modern-suite
# backfill left partial: the ladder had 5400 then jumped to 6000/6010 (modern
# subset only). This fills (a) the missing steps 5500-5900 + 6100 and (b) the
# commonsense-7 tasks on the tail. Content-aware skip-guard merges, so already-
# present tasks are skipped. capacity queue, 1 node.
cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"
# commonsense-7 (0-shot) across the full tail incl 6000/6010 (fills their gap)
STEPS="5500 5600 5700 5800 5900 6000 6100 6200 6300 6400 6500" \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288 \
LABEL=512n \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh
# modern loglikelihood (mmlu-5, arc_challenge-25) on the new steps
STEPS="5500 5600 5700 5800 5900 6100" \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288 \
LABEL=512n \
SHOTS_SPEC="5:mmlu;25:arc_challenge" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh
# gsm8k (generative, slow) on the latest ckpt only
STEPS="6100" \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288 \
LABEL=512n \
SHOTS_SPEC="5:gsm8k" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh
