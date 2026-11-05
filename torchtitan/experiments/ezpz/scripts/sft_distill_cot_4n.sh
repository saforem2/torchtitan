#!/bin/bash --login
#PBS -A datascience
#PBS -N sft-distill-pretok
#PBS -l walltime=12:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=5
#PBS -q workq
#PBS -j oe
#
# SFT distill CoT -- PRE-TOKENIZED -- from ckpt-900-hf (tulu_math_uc_mix 8N GBS6144)
# on a broad reasoning-distillation mix.
#
# Base: outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900-hf
#   (Sunspot, POST-TRAINING-2B.md + SFT README.md: ckpt-900 is the general-instruct
#    deliverable, IFEval 0.253, pre-collapse; -hf export re-consolidated if missing)
#   Fallback: checkpoint-900/ (sharded DCP, 96 .distcp shards) -- train_sft.py
#   accepts HF or DCP, but we prefer HF for TRL's AutoTokenizer path.
#
# Mix: distill_cot_mix = 50% OpenThoughts-114k-decontam + 50% gsm8k-r1cot
#   (datasets_sft.py:register_sft_dataset(distill_cot_mix), SFTDataset
#    _build_distill_cot_mix -> _materialized_mix_load_or_build with GSM8K-test
#    decontam baked into each component's build => recipe hash encodes decontam,
#    a cache hit is guaranteed decontaminated, per-file [decontam] drop counts)
#   Alt spec: OpenThoughts-114k-decontam:0.5,gsm8k-r1cot:0.5 via mix-spec parser.
#
# Why pre-tokenized: B2v2's distill_cot_mix hit the HF streaming cache race at 2N
#   (rank 0 writes new arrow cache, others look for old shards; also 429 storms)
#   and TRL re-tokenizes the whole mix at job start (93M-row mix ~8.5h, longer than
#   walltime, idle-watchdog SIGTERM job 12470281). Pre-tokenizing moves that cost
#   to a dedicated one-time job (see STAGE 1) and the training job loads input_ids
#   via --pretokenized_dataset, where TRL skips prep because input_ids is present
#   (train_sft.py: if use_pretokenized: load_from_disk + skip_prepare_dataset).
#   This also avoids the HF streaming cache entirely (no HF cache on compute ranks).
#
# 2-stage wire (precache pattern from precache_hf_dataset.py + pretokenize):
#   STAGE 1 -- precache + pre-tokenize (1N, login or compute):
#     qsub torchtitan/experiments/ezpz/rl/scripts/sft/_pretokenize_distill_cot_mix_1n.sh
#     -> writes /tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192 (shared /lus/tegu)
#     That job itself runs: HF_HOME=/tegu/datasets/datasets/hf snapshot_download +
#     load_dataset(streaming=True) verify + datasets_sft mix-cache build +
#     train_sft.py --pretokenize_to (TRL tokenize+pack ONCE, save_to_disk num_proc=16).
#     Verify stage 1 before stage 2:
#       ls /tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192/dataset_info.json
#       du -sh /tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192
#       grep -F "[decontam]" logs/pretokenize-distill-cot-mix-*/run.log
#   STAGE 2 -- train (this script, 4N or 8N):
#     If PRETOK_DIR exists we train immediately (Grain-style: shared HuggingFace
#     RandomAccessSource is subsumed by TRL's load_from_disk; both read local
#     parquet/arrow, zero Hub calls, HF_DATASETS_OFFLINE=1).
#     If PRETOK_DIR is still building, this script waits (sleep loop) rather than
#     re-triggering the HF download -- avoids the rank-0 cache race.
#
# GBS: per-device BS 2, GAS 8, seq 8192, packing=True, max_length 8192
#   GBS_train = NGPU * 2 * GAS / (TP*PP*CP). At 4N: 48*2*8=768; at 8N: 96*2*8=1536.
#   Keep GAS modest so loss scale stays near B2's 384 (seq 2048 vs 8192 so tokens/step ~same).
#
# Packing: 4N active (select=5 with spare) + peer 4N+4N + GRPO 2N + test.sh 2N = 16N active.
#   For a standalone 8N run (workq max 128N, this lane budgets ~16N active across 4 jobs
#   so it plays nice in parallel): qsub -l select=9 -v NHOSTS_TRAIN=8 this_script.sh
#   Header default select=5 is the 4N pack value; override at submit for 8N.
#
# Submit (pack default, 4N active):
#   qsub -l select=5 -v NHOSTS_TRAIN=4 torchtitan/experiments/ezpz/scripts/sft_distill_cot_pretok_8n.sh
# Submit (standalone, 8N active):
#   qsub -l select=9 -v NHOSTS_TRAIN=8 torchtitan/experiments/ezpz/scripts/sft_distill_cot_pretok_8n.sh
# Smoke (1 epoch):
#   qsub -l select=5 -v NHOSTS_TRAIN=4,NUM_EPOCHS=1 torchtitan/experiments/ezpz/scripts/sft_distill_cot_pretok_8n.sh
set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu >/dev/null 2>&1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,*.alcf.anl.gov}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

# Worktree-current check (preflight-clean): pinned worktree + EXPECTED_COMMIT gate
# matches agpt2b_multihost_torchstore_validate.pbs contract. If EXPECTED_COMMIT
# not supplied (normal local submit), just verify not behind upstream.
SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
cd "${SUBMIT_DIR}" || exit 1
if [[ -n "${EXPECTED_COMMIT:-}" ]]; then
  [[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || { echo "FATAL: commit $(git rev-parse HEAD) != EXPECTED_COMMIT ${EXPECTED_COMMIT}" >&2; exit 15; }
  git diff --quiet && git diff --cached --quiet || { echo "FATAL: tracked worktree changes present" >&2; exit 15; }
else
  # Best-effort staleness check (informational, not fatal for preflight)
  if git rev-parse --git-dir >/dev/null 2>&1; then
    h=$(git rev-parse --short HEAD 2>/dev/null); u=$(git rev-parse --short '@{u}' 2>/dev/null || echo "$h")
    [[ "$h" == "$u" ]] || echo "WARN: worktree $h != upstream $u -- fetch/checkout first" >&2
  fi
fi

# Env bootstrap (Monarch / TRL)
VENV="${VENV:-/lus/tegu/projects/datascience/foremans/venvs/rl-monarch-torch214-locality-20261001}"
export VIRTUAL_ENV="${VENV}"
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${VENV}/lib:${LD_LIBRARY_PATH:-}"

# Shared dataset staging area -- same as _pretokenize_distill_cot_mix_1n.sh.
# Raw HF repos + materialized-mix cache live under /tegu/datasets (datasets
# group, separate quota) so every rank sees the same build without re-download.
export HF_HOME=/tegu/datasets/datasets/hf
export HF_DATASETS_CACHE=/tegu/datasets/datasets/hf/datasets
export EZPZ_SFT_MIX_CACHE_DIR=/tegu/datasets/datasets/ezpz_sft_mixes

python3 -c "import trl; print('trl', trl.__version__)" || { echo "FATAL: trl missing in ${VENV}" >&2; exit 1; }
python3 -c "import torch; print('torch', torch.__version__, 'xpu', torch.xpu.is_available())" 2>&1 | head -2

# ---- Base checkpoint: ckpt-900-hf (preferred) -> ckpt-900 sharded -> legacy agpt2b-gsm8k path ----
# Source of truth: torchtitan/experiments/ezpz/docs/production/sft/README.md
# "checkpoint-900-hf is the deliverable" (IFEval 0.253, base-LM intact). Sharded
# form is outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900/
# (96 .distcp shards, intact). Consolidate if -hf missing (see SFT docs for cmd).
BASE_MODEL="${BASE_MODEL:-${SUBMIT_DIR}/outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900-hf}"
if [[ ! -d "${BASE_MODEL}" ]]; then
  SHARDED="${SUBMIT_DIR}/outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900"
  if [[ -d "${SHARDED}" ]]; then
    echo "WARN: ${BASE_MODEL} not found; sharded ${SHARDED} exists -- using sharded (train_sft accepts DCP) or re-consolidate to -hf first" >&2
    BASE_MODEL="${SHARDED}"
  else
    ALT="${SUBMIT_DIR}/outputs/sft/agpt2b-gsm8k-r1cot-2n"
    if [[ -d "${ALT}" ]]; then
      echo "WARN: ckpt-900 not found at all; falling back to legacy ${ALT} (B2 path)" >&2
      BASE_MODEL="${ALT}"
    fi
  fi
fi
if [[ ! -d "${BASE_MODEL}" ]]; then
  echo "FATAL: BASE_MODEL not found: ${BASE_MODEL}" >&2
  echo "Available SFT outputs (shared, on tegu):" >&2
  find "${SUBMIT_DIR}/outputs/sft" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -20 >&2
  echo "Re-consolidate checkpoint-900 if only sharded exists (see docs/production/sft/README.md)" >&2
  exit 1
fi

# ---- Pre-tokenized dataset (avoids HF streaming cache race) ----
# Built by _pretokenize_distill_cot_mix_1n.sh via --pretokenize_to; consumed here via --pretokenized_dataset.
# If not ready, document the 2-stage path and optionally launch stage 1 then wait -- do NOT try to
# rebuild the mix from scratch on N ranks (that re-triggers the multi-rank cache race + 8h tokenize).
PRETOK_DIR="${PRETOK_DIR:-/tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192}"
PRETOK_FALLBACK="${SUBMIT_DIR}/outputs/sft-pretok/distill-cot-mix-len8192"
mkdir -p "$(dirname "${PRETOK_FALLBACK}")"

_wait_for_pretok() {
  local dir="$1" tries="${PRETOK_WAIT_TRIES:-6}" sleep_s="${PRETOK_WAIT_SLEEP:-300}"
  for ((i=1;i<=tries;i++)); do
    [[ -f "${dir}/dataset_info.json" ]] && return 0
    echo "[pretok] ${dir} not ready (try $i/$tries), sleeping ${sleep_s}s -- stage 1 may still be running" >&2
    sleep "${sleep_s}"
  done
  return 1
}

USE_PRETOK=""
if [[ -f "${PRETOK_DIR}/dataset_info.json" ]]; then
  USE_PRETOK="${PRETOK_DIR}"
  echo "PRETOK hit: ${PRETOK_DIR} ($(du -sh "${PRETOK_DIR}" 2>/dev/null | cut -f1))" | tee -a /dev/stderr
elif [[ -f "${PRETOK_FALLBACK}/dataset_info.json" ]]; then
  USE_PRETOK="${PRETOK_FALLBACK}"
  echo "PRETOK hit (fallback): ${PRETOK_FALLBACK}" >&2
else
  echo "PRETOK miss: ${PRETOK_DIR} not present" >&2
  if [[ "${PRETOK_AUTO_BUILD:-0}" == "1" ]]; then
    echo "[pretok] PRETOK_AUTO_BUILD=1 -> launching 1N precache+pre-tokenize inline, then training" >&2
    # Stage 1 inline: runs on 1 rank (no collectives, no barrier), proxy ON so
    # OpenThoughts-114k can be snapshot_downloaded, decontam + mix-cache + TRL
    # tokenize+pack run ONCE here so stage 2 starts in <60s.
    # This is the "2-stage script where stage 1 is precache + stage 2 is train"
    # requested when precache is not feasible ahead of time.
    LOG_DIR_STAGE1="logs/pretokenize-distill-cot-mix-${PBS_JOBID%%.*}-stage1"
    mkdir -p "${LOG_DIR_STAGE1}"
    echo "=== STAGE 1: 1N precache+pre-tokenize -> ${PRETOK_DIR} ===" | tee "${LOG_DIR_STAGE1}/run.log"
    git log -1 --oneline -- torchtitan/experiments/ezpz/rl/train_sft.py torchtitan/experiments/ezpz/rl/datasets_sft.py torchtitan/experiments/ezpz/rl/decontam_traces.py 2>&1 | tee -a "${LOG_DIR_STAGE1}/run.log"
    # Pick a base model path that exists on this host for the tokenizer (pre-tokenize needs it)
    PRETOK_BASE="${BASE_MODEL}"
    NPROC="${NPROC:-32}"
    ezpz launch --np 1 --timeout 1800 -- python3 -m torchtitan.experiments.ezpz.rl.train_sft \
      --sft_dataset distill_cot_mix \
      --model_name_or_path "${PRETOK_BASE}" \
      --pretokenize_to "${PRETOK_DIR}" \
      --max_length 8192 \
      --dataset_num_proc "${NPROC}" \
      --output_dir "${LOG_DIR_STAGE1}/_pretok_scratch" \
      --report_to none \
      2>&1 | tee -a "${LOG_DIR_STAGE1}/run.log"
    if _wait_for_pretok "${PRETOK_DIR}"; then
      USE_PRETOK="${PRETOK_DIR}"
    else
      echo "FATAL: stage 1 finished but ${PRETOK_DIR}/dataset_info.json still missing" >&2
      exit 1
    fi
    echo "PRETOK not ready and PRETOK_AUTO_BUILD=0 (default)" >&2
    echo "  Either:" >&2
    echo "    1) Run stage 1 first: qsub torchtitan/experiments/ezpz/rl/scripts/sft/_pretokenize_distill_cot_mix_1n.sh" >&2
    echo "       -> builds ${PRETOK_DIR} on shared /tegu (proxy ON, decontam, mix-cache, tokenize+pack)" >&2
    echo "       -> then re-submit this script (stage 2) once dataset_info.json appears" >&2
    echo "    2) Or submit with PRETOK_AUTO_BUILD=1 to do both in one allocation:" >&2
    echo "       qsub -v PRETOK_AUTO_BUILD=1,NHOSTS_TRAIN=4 -l select=5 $0" >&2
    echo "  (PRETOK_AUTO_BUILD serializes precache on 1 rank first, then trains on N ranks; slower walltime but no extra queue hop)" >&2
    # Wait a short window in case a sibling stage-1 just finished (avoids a needless fail)
    if ! _wait_for_pretok "${PRETOK_DIR}"; then
      echo "FATAL: no pretokenized dataset; follow the 2-stage docs above" >&2
      exit 1
    fi
    USE_PRETOK="${PRETOK_DIR}"
  fi
fi

# ---- Node / batch sizing (pack-aware) ----
if [[ -z "${NHOSTS_TRAIN:-}" ]]; then
  # Derive from PBS select when not explicitly set (header select=5 => 4N active + 1 spare)
  PBS_SELECT="${PBS_SELECT:-${PBS_NUM_NODES:-}}"
  if [[ -n "${PBS_SELECT}" && "${PBS_SELECT}" -ge 2 ]]; then
    NHOSTS_TRAIN=$(( PBS_SELECT - 1 ))
  else
    NHOSTS_TRAIN=4
    echo "WARN: NHOSTS_TRAIN not set and PBS select unavailable; defaulting to 4 (pack: 4+4+4+2=14 active + test 2N =16N)" >&2
  fi
fi
PPN="${NGPU_PER_HOST:-12}"
NGPUS_ACTIVE=$(( NHOSTS_TRAIN * PPN ))
NNODES="${NHOSTS_TRAIN}"

NUM_EPOCHS="${NUM_EPOCHS:-3}"
CKPT_DIR="${SUBMIT_DIR}/outputs/sft/agpt2b-distill-cot-pretok-${NNODES}n-ep${NUM_EPOCHS}"
LOG_DIR="${SUBMIT_DIR}/logs/sft-distill-pretok-${NHOSTS_TRAIN}n-${PBS_JOBID%%.*}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

GBS=$(( NGPUS_ACTIVE * 2 * 8 ))  # LBS=2 GAS=8 TP=1 => GBS = NGPU*LBS*GAS

echo "=== SFT distill CoT PRETOK: base=${BASE_MODEL} pretokenized=${USE_PRETOK} N=${NHOSTS_TRAIN} (NGPU=${NGPUS_ACTIVE}) GBS=${GBS} epochs=${NUM_EPOCHS} ===" | tee "${LOG_DIR}/run.log"
echo "PBS select=${PBS_SELECT:-?} NHOSTS_TRAIN=${NHOSTS_TRAIN} PPN=${PPN} NGPUS_ACTIVE=${NGPUS_ACTIVE} GBS=${GBS}" | tee -a "${LOG_DIR}/run.log"
echo "PACK: this ${NHOSTS_TRAIN}N active + peers (4N+4N) + RL_GRPO 2N + test.sh 2N = $(( NHOSTS_TRAIN + 4 + 4 + 2 + 2 ))N active (budget 16N, workq max 128N)" | tee -a "${LOG_DIR}/run.log"
echo "BASE_MODEL=${BASE_MODEL}" | tee -a "${LOG_DIR}/run.log"
echo "PRETOK_DIR=${USE_PRETOK}" | tee -a "${LOG_DIR}/run.log"
echo "CKPT_DIR=${CKPT_DIR}" | tee -a "${LOG_DIR}/run.log"
echo "VENV (LOCALITY-20261001)=${VENV}" | tee -a "${LOG_DIR}/run.log"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/train_sft.py torchtitan/experiments/ezpz/rl/datasets_sft.py torchtitan/experiments/ezpz/rl/decontam_traces.py 2>&1 | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

# Stage 2: TRAIN from pre-tokenized dataset (no HF streaming, no rank-0 cache race).
# dataset_num_proc 8 per B2v2 fix (torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh:
#   32 -> 8 after OOM in map subprocesses; 8 is stable + keeps packing bounded).
# HF_DATASETS_OFFLINE=1 because all reads are local arrow (no Hub calls).
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

ezpz launch --nproc "${NGPUS_ACTIVE}" --nproc_per_node "${PPN}" --auto-retry --spare-nodes auto --timeout 1800 -- \
  python3 -m torchtitan.experiments.ezpz.rl.train_sft \
  --model_name_or_path "${BASE_MODEL}" \
  --pretokenized_dataset "${USE_PRETOK}" \
  --output_dir "${CKPT_DIR}" \
  --num_train_epochs "${NUM_EPOCHS}" \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_length 8192 \
  --dataset_num_proc 8 \
  --bf16 --fsdp full_shard \
  --logging_steps 5 \
  --save_strategy epoch \
  2>&1 | tee -a "${LOG_DIR}/run.log" || true

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== DONE SFT distill-pretok: ckpts in ${CKPT_DIR}, log in ${LOG_DIR}/ ===" | tee -a "${LOG_DIR}/run.log"
echo "Pack total active ~16N: this ${NHOSTS_TRAIN}N + cpt_fineweb 4N + cpt_olmo2575 4N + rl_grpo 2N + test 2N" | tee -a "${LOG_DIR}/run.log"
echo "Next: consolidate + eval GSM8K-CoT vs B2 0.205 (and vs B2v2 0.09/0.105 refutation)" | tee -a "${LOG_DIR}/run.log"
echo "  ./torchtitan/experiments/ezpz/scripts/eval/consolidate_and_eval_cot.sh ${CKPT_DIR}/final" | tee -a "${LOG_DIR}/run.log"
