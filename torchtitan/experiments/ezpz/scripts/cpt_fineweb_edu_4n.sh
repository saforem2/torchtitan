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

EZPZ_UTILS="${PBS_O_WORKDIR:-$PWD}/.ezpz-utils-cache/ezpz-utils.sh"
if [[ -f "$EZPZ_UTILS" ]]; then source "$EZPZ_UTILS"
else source <(curl -fsSL --max-time 30 https://bit.ly/ezpz-utils); fi
cd "${PBS_O_WORKDIR:-$(pwd)}"

ezpz_load_modules
ezpz_setup_job
source .venv/bin/activate
if [[ -f .venv.tar.gz ]]; then ezpz yeet --src .venv.tar.gz; else ezpz yeet; fi
deactivate
source /tmp/.venv/bin/activate

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
  --lr-scheduler.warmup-steps=20 --lr-scheduler.decay-ratio=0.0 --lr-scheduler.min-lr-factor=1.0 \
  "${extra_args[@]}" \
  "$@"
