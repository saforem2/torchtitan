#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -j oe
# GSM8K (5-shot) with --limit 250 on the LATEST 20b-512 ckpt. GSM8K is
# generative (~26s/sample on 20B); full 1319 = ~9.5h blew the 12h walltime
# when chained after the MMLU ladder. 250 samples (~1.7h) gives a fast
# base-model estimate (base gsm8k ~0 this early anyway). Merges into the
# existing results.json (content-aware skip + merge).
cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"
STEPS="6000" CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288 LABEL=512n \
  SHOTS_SPEC="5:gsm8k" LIMIT=250 \
  bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh
