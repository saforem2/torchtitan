#!/bin/bash --login
#PBS -A datascience
#PBS -N cpt-fwe-4n
#PBS -l walltime=12:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=4
#PBS -q workq
#PBS -j oe
# PPN=8 proven at 12483025 (F0): 2N=16 ranks, hidden_dim 11008 % 32 == 0 but
# Use 4N x PPN=8 = 32r (walltime 12:00:00, workq up to 23h — was 02:00 backfill
# churn, now 12h = ~24k steps/hop vs 4k; chain watcher remains as safety net).
# Submit: qsub -l select=4 -l walltime=12:00:00 -v NHOSTS_TRAIN=4,NGPU_PER_HOST=8 torchtitan/experiments/ezpz/scripts/cpt_fineweb_edu_4n.sh
# pack: 4N(FWE)+4N(mix)+4N(SFT)+2N(GRPO)+2N(test) = 16N active on workq (max 128N)
#
# CPT 4N: MDS gs138650 step-0 (2B-mds, 11008) -> 60/40 owm/FineWeb-Edu blend on Sunspot.
# Base: MDS_ANNEAL_BASE=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/checkpoints/agpt-2b-mds-gs138650/step-0
#       verified: .metadata exists (TEGU); Sunspot ~/torchtitan has NO outputs/ so the default relative path is wrong
# Data: /lus/tegu/projects/datasets/datasets/fineweb-edu-100BT/sample/100BT 140 parquet, 16 shards (registry _fineweb_edu_shards)
# GBS: LBS=2 * PPN=8 * 4N=32r => GBS=64 (TP=1), tokens/step=64*8192=524288; 100B => 190734 steps (chain via CKPT_DIR resume)
# LR: 2e-6 const warmup20 (decay_ratio 0.0) -- anneal-proven flat schedule
# Extra: EXTRA_ARGS="--training.steps=NNN --checkpoint.interval=NNN ..." for overrides

export MDS_ANNEAL_BASE="${MDS_ANNEAL_BASE:-/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/checkpoints/agpt-2b-mds-gs138650/step-0}"
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
export PYTHONPATH="${SUBMIT_DIR}:${PYTHONPATH:-}"
# Fix checkpoint double-nesting: trainer dumps to outputs/ already, so ensure
# resume finds existing ckpts at outputs/outputs/checkpoints/ (from prior runs).
if [[ -d "${SUBMIT_DIR}/outputs/outputs/checkpoints" && ! -e "${SUBMIT_DIR}/outputs/checkpoints" ]]; then
  ln -s outputs/checkpoints "${SUBMIT_DIR}/outputs/checkpoints" 2>/dev/null || true
  echo "Linked outputs/checkpoints -> outputs/outputs/checkpoints for resume" >&2
fi

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

PPN=8
NGPUS_ACTIVE=$(( NHOSTS_TRAIN * PPN ))
NNODES="$NHOSTS_TRAIN"
LBS=2; GAS=1; SEQ_LEN=8192; TP=1; PP=1; CP=1
GBS=$(( NGPUS_ACTIVE * LBS * GAS / (TP * PP * CP) ))
TRAIN_TOKENS=100000000000
TRAINING_STEPS=$(( TRAIN_TOKENS / (GBS * SEQ_LEN) ))
CKPT_INTERVAL=200
CKPT_KEEP_LATEST_K=0
CKPT_DIR="outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n${NNODES}-gbs${GBS}"

# EXTRA_ARGS passthrough for CPT fork flags
extra_args=()
[[ -n "${EXTRA_ARGS:-}" ]] && read -ra extra_args <<< "${EXTRA_ARGS}"

"${VENV}/bin/python" -c 'from ezpz.cli import main; main()' launch \
  --nproc "${NGPUS_ACTIVE}" --nproc_per_node "${PPN}" -- \
  "${VENV}/bin/python" -m torchtitan.experiments.ezpz.train --module=ezpz.agpt --config=agpt_2b_mds_mix_owm_edu_6040 \
  --checkpoint.interval="${CKPT_INTERVAL}" \
  --checkpoint.keep-latest-k="${CKPT_KEEP_LATEST_K}" \
  --checkpoint.folder="${CKPT_DIR}" \
  --validator.enable --validator.freq=100 --validator.steps=10 \
  --training.steps="${TRAINING_STEPS}" \
  "${extra_args[@]}" \
  "$@"
