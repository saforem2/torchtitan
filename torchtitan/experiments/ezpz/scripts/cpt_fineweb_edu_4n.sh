#!/bin/bash --login
#PBS -A datascience
#PBS -N cpt-fwe-4n
#PBS -l walltime=12:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=5
#PBS -q workq
#PBS -j oe
# select=5 = NHOSTS_TRAIN=4 + 1 spare; submit: qsub -l select=5 -v NHOSTS_TRAIN=4 torchtitan/experiments/ezpz/scripts/cpt_fineweb_edu_4n.sh
# pack: 4N(FWE)+4N(2575)+4N(SFT)+2N(GRPO)+2N(test.sh)=16N active; spares are PBS overprovision not counted
#
# CPT 4N from plateaued 2B-256N base step-92859 on FineWeb-Edu 100BT via Grain.
# Base: /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144/step-92859
#       verified on Sunspot: ls .../step-92859/.metadata exists (9.4M __0_0.distcp, 24G total, 2026-06-29)
#       _OLMO_ANNEAL_BASE in config_registry.py:881 points to outputs/checkpoints/.../step-92859 (same DCP)
# Data: /lus/tegu/projects/datasets/datasets/fineweb-edu-100BT/sample/100BT 140 parquet (2.1G each), 16 shards
#       verified: ssh sunspot ls .../100BT/*.parquet | wc -l ==140; sorted 000_00000..001_00005 are first 16
#       _fineweb_edu_shards() config_registry.py:1268-1288 sorts glob and slices 16; _FINEWEB_EDU_DIR same path
#       datasets.py:468 registers fineweb_edu_local -> that TEGU dir; Grain HuggingFaceRandomAccessSource parquet
#       Each shard ~1.4B tokens; 16 shards ~22B tokens > 10-100B budget so no repeat (registry comment lines 1262-1267)
# GBS: LBS=2 * PPN=12 * NHOSTS_TRAIN=4 => NGPUS_ACTIVE=48 => GBS=48*2*1/(1*1*1)=96; TP=1 so GBS=96 microbatch
#      tokens/microbatch/rank = LBS*SEQ_LEN=16384; tokens/step=GBS*8192=786432
#      TRAIN_TOKENS 100B => TRAINING_STEPS=127156 (100e9/786432); 12h walltime ~chained, resume from CKPT_DIR
# LR: 2e-6 const warmup20 (decay_ratio 0.0 min_lr_factor 1.0) avoids 2.28e-5 re-warm shock that tanked CPT HS -7.4pp
#     (cpt/README.md:68-88 gentle retry hypothesis)
# Extra: EXTRA_ARGS="--checkpoint.initial-load-path=<step92859> --checkpoint.initial-load-model-only" supported

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
# 100B CPT screening; prior pilot 300B at 256N GBS6144 was 5960 steps — 4N GBS96 scales steps by 64x
TRAIN_TOKENS=100000000000
TRAINING_STEPS=$(( TRAIN_TOKENS / (GBS * SEQ_LEN) ))
CKPT_INTERVAL=200
CKPT_KEEP_LATEST_K=0
CKPT_DIR="outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n${NNODES}-gbs${GBS}"
BASE_CKPT="/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144/step-92859"

# EXTRA_ARGS passthrough for CPT fork flags (qsub -v EXTRA_ARGS="--checkpoint.initial-load-path=... ...")
extra_args=()
[[ -n "${EXTRA_ARGS:-}" ]] && read -ra extra_args <<< "${EXTRA_ARGS}"

ezpz launch --nproc "${NGPUS_ACTIVE}" --nproc_per_node "${PPN}" --auto-retry --spare-nodes auto --timeout 1800 -- \
  python3 -m torchtitan.experiments.ezpz.train --module=ezpz.agpt --config=agpt_2b_real \
  --checkpoint.enable --checkpoint.folder="${CKPT_DIR}" --checkpoint.interval="${CKPT_INTERVAL}" \
  --checkpoint.keep-latest-k="${CKPT_KEEP_LATEST_K}" --checkpoint.no-last-save-model-only \
  --checkpoint.async-mode=disabled \
  --dataloader.dataset=fineweb_edu_local \
  --validator.enable --validator.freq=100 --validator.steps=10 \
  --optimizer=sophiag --optimizer.lr=2e-6 \
  --training.local-batch-size="${LBS}" --training.global-batch-size="${GBS}" \
  --training.seq-len="${SEQ_LEN}" --training.steps="${TRAINING_STEPS}" \
  --checkpoint.initial-load-path="${BASE_CKPT}" \
  --checkpoint.initial-load-model-only \
  --optim.lr-scheduler.warmup-steps=20 --optim.lr-scheduler.decay-ratio=0.0 --optim.lr-scheduler.min-lr-factor=1.0 \
  "${extra_args[@]}" \
  "$@"
