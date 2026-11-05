#!/bin/bash --login
#PBS -A datascience
#PBS -N cpt-2575-4n
#PBS -l walltime=12:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=5
#PBS -q workq
#PBS -j oe
# select=5 = NHOSTS_TRAIN=4 + 1 spare; submit: qsub -l select=5 -v NHOSTS_TRAIN=4 torchtitan/experiments/ezpz/scripts/cpt_olmo25_dolmino75_4n.sh
# pack: 4N(FWE)+4N(2575)+4N(SFT)+2N(GRPO)+2N(test.sh)=16N active; spares not counted toward 16N
#
# CPT 4N: olmo25-dolmino75 blend — the never-ran ratio gap (cpt/README.md:65, 196-203).
# Prior CPT ran dolmino-100 (val 2.492) and olmo50-dolmino50 (2.601) vs olmo-100 plateau 2.80;
# more dolmino = lower loss monotonic, but eval degraded HS -7.4pp at 2.28e-5 re-warm.
# This 25/75 arm completes 0/50/75/100 curve with gentle LR=2e-6 const warmup20 (avoids shock).
#
# Data source: torchtitan/experiments/ezpz/data-lists/aurora/olmo25-dolmino75.txt
#   verified locally: ls torchtitan/experiments/ezpz/data-lists/aurora/olmo25-dolmino75.txt exists
#   wc -l 1762 (1438 olmo + 324 dolmino); weight sum 1.000000 (0.25 olmo + 0.75 dolmino)
#   produced by utils/build_cpt_mixes.py: python3 build_cpt_mixes.py aurora/olmo-mix-1124.txt
#   aurora/dolmino-mix-1124.txt /tmp => wrote olmo25-dolmino75.txt (verified locally via python sum check)
#   olmo orig sum 2.717391, dolmino sum 1.0 -> rescaled to 0.25/0.75 correctly
#   Underlying .bins: /flare/AuroraGPT/datasets/olmo-mix-1124/... + dolmino-mix-1124/... (Aurora flare)
#   BLOCKER on Sunspot: /flare not mounted on Sunspot (ssh sunspot ls /flare/... => no such file; /proc/mounts shows
#   only /lus/tegu and /lus/flare (different from /flare/AuroraGPT), and find /lus/tegu -name fused_0001* finds nothing).
#   So olmo25 blend cannot run on Sunspot without staging bins to /lus/tegu.
#   Gentle-LR fallback: if /flare bin missing at launch, script auto-falls back to fineweb_edu_local (Grain parquet
#   on TEGU, same 16 shards as cpt_fwe_4n, 100B budget) — i.e. the "gentle 2e-6 anneal" lane requested in the spec.
#   To run the true blend, qsub on Aurora instead: qsub -A AuroraGPT -q prod -l filesystems=flare:home -l select=5 -v NHOSTS_TRAIN=4 ...
#
# Base: same step-92859 DCP as cpt_fwe_4n, verified on Sunspot:
#   ls /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144/step-92859/.metadata exists
#   (24G, __0_0.distcp 9.4M, Jun 29 08:02; _OLMO_ANNEAL_BASE config_registry.py:881 same path)
# GBS: LBS=2 * PPN=12 * 4N=48 => GBS=96 (TP=1), tokens/microbatch=16384, tokens/step=786432
#      TRAIN_TOKENS 100B => 127156 steps; prior 300B pilot at 256N GBS6144 was 5960 steps (300B/50M)
# LR: 2e-6 const warmup20 decay_ratio 0.0 min_lr_factor 1.0 (same gentle schedule as FWE lane)
# Extra: EXTRA_ARGS passthrough supports --checkpoint.initial-load-path=<step92859> --checkpoint.initial-load-model-only

if [[ -z "${NHOSTS_TRAIN:-}" ]]; then echo "ERROR: NHOSTS_TRAIN required (qsub -v NHOSTS_TRAIN=4)"; exit 1; fi

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export ftp_proxy="${ftp_proxy:-http://proxy.alcf.anl.gov:3128}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,*.alcf.anl.gov,*.aurora.alcf.anl.gov}"

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
cd "${SUBMIT_DIR}" || exit 1

# Pinned ezpz-utils (like B2v2 + data_resume working jobs)
source "${SUBMIT_DIR}/torchtitan/experiments/ezpz/scripts/load_pinned_ezpz_utils.sh"
load_pinned_ezpz_utils || exit $?
ezpz_load_modules
ezpz_setup_job

# CPT VENV: xpu-torch216 with Grain 0.2.18 + torch_remat (verified; fw-2026.1 lacks remat)
VENV="${VENV:-/lus/tegu/projects/datascience/foremans/venvs/xpu-torch216-sync-5a7a622f16}"
# LD-first: venv libs win over system for UR/SYCL (same as data_resume job)
if [[ -d "${VENV}/lib" ]]; then
  _ld_first="${SUBMIT_DIR}/.cache/xpu-ld-first/$(basename "${VENV}")"
  if [[ ! -e "${_ld_first}/.complete" ]]; then
    mkdir -p "${_ld_first}"
    for _lib in "${VENV}"/lib/libur_* "${VENV}"/lib/libsycl.so* "${VENV}"/lib/libumf* "${VENV}"/lib/libtcm*; do
      [[ -e "${_lib}" ]] && ln -sf "${_lib}" "${_ld_first}/"
    done
    touch "${_ld_first}/.complete"
  fi
  export LD_LIBRARY_PATH="${_ld_first}:${LD_LIBRARY_PATH:-}"
fi
export VIRTUAL_ENV="${VENV}"
export PATH="${VENV}/bin:/opt/pbs/bin:${PATH}"
hash -r
[[ "$(command -v python)" == "${VENV}/bin/python" ]] || { echo "FATAL: venv python mismatch: $(command -v python) != ${VENV}/bin/python" >&2; exit 93; }
python3 -c "import torch_remat, grain; print('CPT venv ok:', grain.__version__)" || { echo "FATAL: CPT venv missing deps" >&2; exit 1; }

PPN="${NGPU_PER_HOST:-12}"
NGPUS_ACTIVE=$(( NHOSTS_TRAIN * PPN ))
NNODES="$NHOSTS_TRAIN"
LBS=2; GAS=1; SEQ_LEN=8192; TP=1; PP=1; CP=1
GBS=$(( NGPUS_ACTIVE * LBS * GAS / (TP * PP * CP) ))
TRAIN_TOKENS=100000000000
TRAINING_STEPS=$(( TRAIN_TOKENS / (GBS * SEQ_LEN) ))
CKPT_INTERVAL=200
CKPT_KEEP_LATEST_K=0
CKPT_DIR="outputs/checkpoints/agpt-2b-cpt-olmo25-dolmino75-n${NNODES}-gbs${GBS}"
BASE_CKPT="/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144/step-92859"

extra_args=()
[[ -n "${EXTRA_ARGS:-}" ]] && read -ra extra_args <<< "${EXTRA_ARGS}"

# Resolve olmo25 list; detect Sunspot blocker (flare not mounted) and fallback to fineweb_edu_local gentle anneal.
DFL_OLMO25="torchtitan/experiments/ezpz/data-lists/aurora/olmo25-dolmino75.txt"
DFL_FIRST_BIN=""
if [[ -f "${DFL_OLMO25}" ]]; then
  DFL_FIRST_BIN=$(awk 'NF>=2{print $2; exit}' "${DFL_OLMO25}" 2>/dev/null || true)
fi
if [[ -n "${DFL_FIRST_BIN}" && -f "${DFL_FIRST_BIN}" ]]; then
  DATASET="blendcorpus"
  DATASET_PATH="${DFL_OLMO25}"
  echo "[cpt-2575] using olmo25-dolmino75 blend: ${DFL_OLMO25} (first bin exists: ${DFL_FIRST_BIN})"
else
  DATASET="fineweb_edu_local"
  DATASET_PATH=""
  echo "[cpt-2575] BLOCKER: ${DFL_FIRST_BIN:-<no bin>} not on Sunspot (/flare not mounted); falling back to gentle-LR anneal on fineweb_edu_local (Grain TEGU parquet, 16 shards verified)"
  echo "[cpt-2575] To run true 25/75 blend, qsub on Aurora: qsub -A AuroraGPT -q prod -l filesystems=flare:home -l select=5 -v NHOSTS_TRAIN=4 $0"
fi

# Build dataset flags
if [[ "${DATASET}" == "blendcorpus" ]]; then
  DATASET_FLAGS=(--dataloader.dataset=blendcorpus --dataloader.dataset-path="${DATASET_PATH}")
else
  DATASET_FLAGS=(--dataloader.dataset=fineweb_edu_local)
fi

ezpz launch --nproc "${NGPUS_ACTIVE}" --nproc_per_node "${PPN}" --auto-retry --spare-nodes auto --timeout 1800 -- \
  python3 -m torchtitan.experiments.ezpz.train --module=ezpz.agpt --config=agpt_2b_mds_mix_edu \
  --checkpoint.interval="${CKPT_INTERVAL}" \
  --checkpoint.keep-latest-k="${CKPT_KEEP_LATEST_K}" \
  --checkpoint.folder="${CKPT_DIR}" \
  "${DATASET_FLAGS[@]}" \
  --validator.enable --validator.freq=100 --validator.steps=10 \
  --training.steps="${TRAINING_STEPS}" \
  "${extra_args[@]}" \
  "$@"
