#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=12:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe
#
# 1N offline DOWNLOAD + PRE-TOKENIZE of distill_cot_mix @ 8192 for the
# reasoning-distillation cold-start SFT (docs/production/sft/agpt/2b-mds/
# distill-cot-mix/).
#
# WHY: same rationale as _pretokenize_b4_reweight_mix_1n.sh -- SFTTrainer
# re-tokenizes+packs the WHOLE train dataset at job start, so pre-tokenizing
# here moves that cost to a dedicated one-time job via --pretokenize_to. The
# training job then loads it with --pretokenized_dataset; TRL sees input_ids
# and SKIPS prep, so training starts in <60s.
#
# distill_cot_mix replaces the team's narrow math-only CoT cold-start with
# breadth from PRE-DISTILLED frontier reasoning traces:
#   0.50 OpenThoughts-114k-decontam  (math + science + code R1 traces)
#   0.50 OpenR1-Math-220k-decontam   (math-focused R1 traces)
# Both are re-wrapped into the <think>/<answer>\boxed{} envelope (matching
# gsm8k-r1cot + scripts/eval/eval_cot_gsm8k.py) and GSM8K-test-DECONTAMINATED
# before interleave. See datasets_sft.py::_build_distill_cot_mix.
#
# CRITICAL -- DECONTAMINATION. The distillation corpora are built from the same
# public math pools GSM8K was drawn from, so a copied GSM8K TEST question would
# silently inflate the GSM8K CoT eval and any "gain" would be a mirage. The
# registered components (OpenThoughts-114k-decontam, OpenR1-Math-220k-decontam)
# run decontam_traces.GSM8KDecontaminator (13-gram + normalized-exact question
# match) BEFORE the interleave. The decontam is baked into each component build,
# so it is captured in the materialized-mix recipe hash -- a cache HIT is
# guaranteed decontaminated. The per-source drop counts are logged (search the
# run.log for "[decontam]"); confirm they are non-trivial before trusting eval.
#
# max_length 8192: OpenThoughts frontier traces are longer than OpenR1's;
# DISTILL_MAX_THINK_CHARS (default 16000 chars) already trims run-ons at format
# time. If the packed size or the B4 verbose-dilution style re-appears, tighten
# DISTILL_MAX_THINK_CHARS (e.g. 4000) here and re-run -- it is part of the
# recipe via the component builds, so a new value forces a fresh mix cache.
#
# Only OpenThoughts-114k needs a fresh download (~3.4G); OpenR1-Math-220k is
# already cached under /tegu/datasets/datasets/hf/hub. Proxy is ON and both the
# raw downloads and the interleaved-mix cache land under the shared
# /tegu/datasets staging area (datasets group, separate quota), not ~/.cache.
#
# num_proc: dataset_num_proc controls the tokenize AND packing maps; 32 is the
# safe EzpzSFTConfig default (96/64 SIGBUS'd the tulu packing map at scale).
# NOTE: the decontam .filter()/.map() inside the component builds also honor a
# high num_proc; if a subprocess dies during decontam ("One of the subprocesses
# has abruptly died"), walk NPROC down (e.g. NPROC=16). Runs on ONE rank; no
# distributed collective, no oneCCL barrier.
#
# Output: pre-tokenized dataset at
#   /tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192/
# consumed by the training job via --pretokenized_dataset.
set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
# Shared dataset staging area (datasets group, separate quota) -- OpenThoughts
# is not pre-cached, so this run needs network (proxy on, no HF_HUB_OFFLINE).
export HF_HOME=/tegu/datasets/datasets/hf
export HF_DATASETS_CACHE=/tegu/datasets/datasets/hf/datasets
export HF_HUB_ENABLE_HF_TRANSFER=0
# Materialized-mix cache also under the shared area so the training job's mix
# build (if ever needed) finds the same decontaminated interleave result.
export EZPZ_SFT_MIX_CACHE_DIR=/tegu/datasets/datasets/ezpz_sft_mixes
# OpenThoughts <think> length cap (chars). Set explicitly so the value used is
# visible in the run log; part of the recipe (component build) so a change here
# forces a fresh mix cache.
export DISTILL_MAX_THINK_CHARS="${DISTILL_MAX_THINK_CHARS:-16000}"

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"
source .venv/bin/activate
python3 -c "import trl; print('trl', trl.__version__)" || { echo "FATAL: trl missing"; exit 1; }

# num_proc override knob: walk down (e.g. NPROC=16) if the packing map OR the
# decontam map SIGBUSes / a subprocess dies -- see header.
NPROC="${NPROC:-32}"
BASE_MODEL="${HOME}/global_step138650"
OUT_DIR="/tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192"
LOG_DIR="logs/pretokenize-distill-cot-mix-${PBS_JOBID%%.*}"
mkdir -p "${LOG_DIR}"

echo "=== 1N DOWNLOAD+PRETOKENIZE: distill_cot_mix -> ${OUT_DIR} (len=8192, base=gs138650) ===" \
    | tee -a "${LOG_DIR}/run.log"
echo "DISTILL_MAX_THINK_CHARS=${DISTILL_MAX_THINK_CHARS}" | tee -a "${LOG_DIR}/run.log"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/train_sft.py \
    torchtitan/experiments/ezpz/rl/datasets_sft.py \
    torchtitan/experiments/ezpz/rl/decontam_traces.py 2>&1 | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

# Single-rank launch: pre-tokenize is a data job, no model training / no
# collectives. `ezpz launch --np 1` keeps the ezpz env plumbing but runs one
# rank.
ezpz launch --np 1 python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --sft_dataset distill_cot_mix \
    --model_name_or_path "${BASE_MODEL}" \
    --pretokenize_to "${OUT_DIR}" \
    --max_length 8192 \
    --dataset_num_proc "${NPROC:-32}" \
    --output_dir "${LOG_DIR}/_pretok_scratch" \
    --report_to none \
    2>&1 | tee -a "${LOG_DIR}/run.log"

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== decontam drop summary (from run.log) ===" | tee -a "${LOG_DIR}/run.log"
grep -F "[decontam]" "${LOG_DIR}/run.log" | tee -a "${LOG_DIR}/run.log"
echo "=== DONE: pretokenized dataset at ${OUT_DIR}; log in ${LOG_DIR}/ ===" \
    | tee -a "${LOG_DIR}/run.log"
du -sh "${OUT_DIR}" 2>/dev/null | tee -a "${LOG_DIR}/run.log"
