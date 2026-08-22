#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N agpt-30b-lrf-umbrella
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:flare
#PBS -l select=2088
#PBS -q prod
#PBS -j oe
#
# 30B LR finder, FOUR OPTIMIZERS IN PARALLEL -- one 512-node seat each.
#
#   seat 0  adamw       seat 1  torchmuon
#   seat 2  mano        seat 3  sophiag
#
# WHY THIS EXISTS. scripts/submit_lr_finder_30b_aurora.sh (job 8774595) sweeps
# the same model SERIALLY at 256N: three optimizers, one after another, inside
# one allocation. This trades nodes for wall-clock -- four seats finish in the
# time one takes, and adds Muon as a fourth arm. It borrows the nodefile
# slicing, spare carving and concurrent-launch pattern from
# submit_agpt_multi_autoretry.sh but does NOT touch that file: the production
# umbrella has a 2098-node job queued against it, and an edit there risks the
# production path for an experiment.
#
# SIZING: 522 nodes/seat = 512 ACTIVE + 10 SPARE, 4 x 522 = 2088.
#   - 512 ACTIVE is what pins the batch: dp = 512*12 = 6144 = GBS exactly, so
#     LBS=1/GAS=1 is the ONLY admissible shape (6144 = 2^11*3; see below).
#   - 10 spares/seat is submit_agpt_multi_autoretry.sh's own SPARES default,
#     so `ezpz launch --auto-retry --spare-nodes 10` is proven at this shape.
#   - 2088 >= 2000 clears the `large` queue floor (verified `qstat -Qf large`:
#     resources_min.nodect = 2000, resources_max.walltime = 24:00:00).
#
# GBS=6144 IS THE POINT, not a default. The optimal LR is batch-size dependent
# and a number found at another batch does not transfer -- the 80B's
# small-batch finder said 1.1e-5 while its real production ceiling was ~14x
# lower. Sweeping at the batch production will actually use is the exercise.
#
# LBS=1 IS FORCED, and it is the worst LBS for throughput. exp05 measured the
# batch-size lever at 30B: LBS=1 -> 360 tps, LBS=3 -> 466 tps compiled (+29%).
# At 512N there is no choice -- dp already equals GBS. This is the cost of
# parallelising across seats rather than sweeping serially at 256N/LBS=2, and
# it is accepted deliberately.
#
# torchmuon, NOT muon. The custom Newton-Schulz implementation was DROPPED from
# the 2B production-batch sweep: "muon dropped -- it crashes every job via a
# oneCCL collective abort; use torchmuon if needed"
# (docs/experiments/lr-finder/agpt/2b/README.md). That abort is a collective
# fault, not an LR cliff, so a `muon` seat would burn 522 nodes producing no
# curve. torch.optim.Muon is also ~35% faster (CLAUDE.md).
#
# AND MUON IS EXPECTED TO WORK AT THIS SIZE. Its 80B failure was bf16 overflow
# in Newton-Schulz at dim=9216. Both muon.py and mano.py gate on
# `max(p.shape) > 10000`, so at 30B (dim 6144, ffn 16384, vocab 256128) only
# the attention matrices route to Newton-Schulz -- max dim 6144, identical to
# the 20B, where the finder doc records a real minimum at 1.7e-4 and states
# "works fine for Muon". The 16384-wide FFN and the embedding fall back to
# AdamW at both sizes. So this seat is genuinely uncertain, not expected to
# fail -- which is what makes it worth a seat.
#
# CONFIG: agpt_30b (gemma 256k vocab, 28.1B), NOT agpt_30b_olmo2tok. That
# config sets hf_assets_path=./assets/hf/OLMo-2-1124-7B and AURORA DOES NOT
# HAVE IT -- `ls assets/hf/` returns only DeepSeek-v3.2, gemma-7b,
# llama-2-7b-hf, and a find over /flare plus the HF cache found no OLMo-2 model
# repo. The 30B campaign ran on Sunspot where those assets live. gemma is also
# the tokenizer every 2B/20B production chain uses, so an LR calibrated on it
# transfers to an Aurora 30B built the same way. The cost: vocab 256,128 vs
# 100,352 moves embedding+head from 1.23B to 3.15B params, so this optimum is
# for THIS geometry.
#
# COMPILE IS OFF. CLAUDE.md: "torch.compile OOM at 512N: 2B OOMs on GPU, 80B
# OOMs on CPU. Use --compile.no-enable for 512N jobs." That is recorded for 2B
# and 80B and NOT established for 30B -- but the failure mode is an OOM at
# launch, which would cost all four seats at once, and compile also costs
# 7-15 min per seat against only ~150 finder steps. Uncompiled is ~27% slower
# (exp05: 340 vs 466 tps) and that is the cheaper risk. All four seats share
# the setting, so the comparison BETWEEN optimizers stays fair either way.
#
# READ THE SUGGESTED LR PER SEAT, NOT THE EXIT CODE. A seat that NaNs early
# still exits 0 through the finder; a seat whose curve never turns over has
# found nothing. Both are reported in the summary at the end.

# PBS scripts must NOT use `set -euo pipefail` per CLAUDE.md -- venv activate
# trips unbound vars and -u also kills lmod.
set -o pipefail

MAIN=/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz
cd "${PBS_O_WORKDIR:-$MAIN}" || exit 1

JOBID="${PBS_JOBID%%.*}"
LOGDIR="${MAIN}/logs/lrf-umbrella-${JOBID}"
mkdir -p "$LOGDIR"

log() { echo "[lrf-umbrella $(date +%H:%M:%S)] $*"; }
die() { echo "[lrf-umbrella ERROR] $*" >&2; exit 1; }

# ---- Seats -------------------------------------------------------------------
# optimizer|master_port. Everything else is shared: same model, same batch, same
# LR window -- that is what makes the four curves comparable.
SEATS=(
    "adamw|29500"
    "torchmuon|29600"
    "mano|29700"
    "sophiag|29800"
)

NNODES_ACTIVE="${LRF_NNODES:-512}"
SPARES="${LRF_SPARES:-10}"
PER_SEAT=$(( NNODES_ACTIVE + SPARES ))

# ---- Shared sweep parameters -------------------------------------------------
CONFIG="${LRF_CONFIG:-agpt_30b}"
GBS="${LRF_GBS:-6144}"
LBS="${LRF_LBS:-1}"
SEQ_LEN="${LRF_SEQ_LEN:-4096}"
STEPS="${LRF_STEPS:-1000}"
FRACTION="${LRF_FRACTION:-0.15}"
INIT_LR="${LRF_INIT_LR:-1e-8}"
MAX_LR="${LRF_MAX_LR:-1e-3}"
IDLE_TIMEOUT="${LRF_IDLE_TIMEOUT:-1800}"
DFL_NAME="${LRF_DFL_NAME:-olmo-mix-1124}"
DFL="torchtitan/experiments/ezpz/data-lists/aurora/${DFL_NAME}.txt"

# The finder writes its CSV/plot/npz under <dump>/lr_finder/ezpz.agpt/<flavor>/
# <optimizer>/ -- keyed by model+optimizer, so four DIFFERENT optimizers cannot
# collide with each other. They CAN collide with an earlier run of the same
# pair, so scope the dump folder by jobid.
DUMP_BASE="outputs/lrf_umbrella_${JOBID}"

PROBES=$(python3 -c "print(max(1,int($STEPS*$FRACTION)))")

# ---- Preflight ---------------------------------------------------------------
[[ -n "${PBS_NODEFILE:-}" && -f "$PBS_NODEFILE" ]] \
    || die "PBS_NODEFILE not set or missing (got '${PBS_NODEFILE:-}')"
TOTAL_AVAIL=$(wc -l < "$PBS_NODEFILE")
NEED=$(( ${#SEATS[@]} * PER_SEAT ))
(( TOTAL_AVAIL >= NEED )) \
    || die "need $NEED nodes (${#SEATS[@]} x $PER_SEAT), allocation has $TOTAL_AVAIL"

# GBS must divide evenly by dp*LBS or the trainer rejects it at startup
# (trainer.py: global_batch_size % (local_batch_size * batch_degree) == 0).
# Catch it HERE rather than four times in parallel, six minutes into a 2088-node
# allocation. 6144 = 2^11 * 3, so LBS must be a power of two -- notably the live
# 30B chain's LBS=5 is IMPOSSIBLE under a pinned GBS=6144.
DP=$(( NNODES_ACTIVE * 12 ))
if (( GBS % (LBS * DP) != 0 )); then
    die "GBS=$GBS is not divisible by LBS($LBS) * dp($DP) -- the trainer will reject this"
fi
GAS=$(( GBS / (LBS * DP) ))
[[ -f "$DFL" ]] || die "data list missing: $DFL"

# ---- Slice the nodefile ------------------------------------------------------
declare -a S_OPT S_PORT S_SLICE
offset=1
for idx in "${!SEATS[@]}"; do
    IFS='|' read -r opt port <<< "${SEATS[$idx]}"
    slice="$LOGDIR/seat-${idx}-${opt}.hostfile"
    sed -n "${offset},$((offset + PER_SEAT - 1))p" "$PBS_NODEFILE" > "$slice"
    got=$(wc -l < "$slice")
    (( got == PER_SEAT )) || die "seat $idx: sliced $got nodes, expected $PER_SEAT"
    offset=$(( offset + PER_SEAT ))
    S_OPT[$idx]="$opt"; S_PORT[$idx]="$port"; S_SLICE[$idx]="$slice"
done

# No node may appear in two slices -- overlapping seats would have two 6144-rank
# jobs fighting for the same GPUs, which fails in confusing ways rather than
# cleanly.
dupes=$(cat "$LOGDIR"/seat-*.hostfile | sort | uniq -d)
[[ -z "$dupes" ]] || die "slices overlap:"$'\n'"$dupes"
log "slices are disjoint (${#SEATS[@]} x $PER_SEAT nodes)"

# ---- Environment -------------------------------------------------------------
if ! command -v module >/dev/null 2>&1 || [[ -z "${MODULEPATH:-}" ]]; then
    [[ -r /etc/bash.bashrc.local ]] && source /etc/bash.bashrc.local
fi
module load oneapi/release/2025.3.1 hdf5 pti-gpu
export PATH="/opt/pbs/bin:${PATH}"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export ftp_proxy="${ftp_proxy:-http://proxy.alcf.anl.gov:3128}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,*.alcf.anl.gov,*.aurora.alcf.anl.gov}"

set +u
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
set -u

# Stage the venv to EVERY node in the allocation once, before any seat starts.
# Doing it per-seat would run four broadcasts concurrently against the same
# filesystem; ezpz yeet is single-job-per-alloc by design.
source .venv/bin/activate
if [[ -f .venv.tar.gz ]]; then
    log "yeet: tarball broadcast to $TOTAL_AVAIL nodes"
    ezpz yeet --src .venv.tar.gz || die "yeet failed"
else
    die ".venv.tar.gz missing -- build it with 'ezpz tar-env' first (per-file rsync at 2088 nodes is not viable)"
fi
deactivate
source /tmp/.venv/bin/activate

# ---- Banner ------------------------------------------------------------------
echo "============================================================"
echo "  30B LR-finder umbrella   jobid=$JOBID"
echo "  seats     : ${#SEATS[@]} x ${PER_SEAT} nodes (${NNODES_ACTIVE} active + ${SPARES} spare)"
echo "  config    : $CONFIG"
echo "  batch     : GBS=$GBS  LBS=$LBS  GAS=$GAS  seq=$SEQ_LEN  (dp=$DP)"
echo "  tokens/stp: $(( GBS * SEQ_LEN ))"
echo "  LR window : $INIT_LR -> $MAX_LR over $PROBES probes"
echo "  data      : $DFL_NAME"
for idx in "${!SEATS[@]}"; do
    echo "    seat $idx: ${S_OPT[$idx]}  port=${S_PORT[$idx]}"
done
echo "============================================================"

# ---- Launch all seats concurrently -------------------------------------------
declare -a PIDS
for idx in "${!SEATS[@]}"; do
    opt="${S_OPT[$idx]}"
    console="$LOGDIR/seat-${idx}-${opt}.console.log"
    (
        export MASTER_PORT="${S_PORT[$idx]}"; unset MASTER_ADDR
        ezpz launch \
            --nproc "$(( NNODES_ACTIVE * 12 ))" \
            --nproc_per_node 12 \
            --hostfile "${S_SLICE[$idx]}" \
            --auto-retry \
            --spare-nodes "$SPARES" \
            --timeout "$IDLE_TIMEOUT" \
            -- \
            python3 -m torchtitan.experiments.ezpz.train \
            --module ezpz.agpt \
            --config "$CONFIG" \
            --job.dump-folder "${DUMP_BASE}/${opt}" \
            --optimizer "$opt" \
            --training.steps "$STEPS" \
            --training.local_batch_size "$LBS" \
            --training.global_batch_size "$GBS" \
            --training.seq_len "$SEQ_LEN" \
            --metrics.log_freq 1 \
            --checkpoint.no-enable \
            --validator.no-enable \
            --compile.no-enable \
            --dataloader.dataset blendcorpus \
            --dataloader.dataset_path "$DFL" \
            --dataloader.data-cache-path "${DUMP_BASE}/${opt}/.cache/${DFL_NAME}/index-cache" \
            --lr_finder.enable \
            --lr_finder.init_lr "$INIT_LR" \
            --lr_finder.max_lr "$MAX_LR" \
            --lr_finder.fraction "$FRACTION" \
            activation-checkpoint:full
    ) > "$console" 2>&1 &
    PIDS[$idx]=$!
    log "launched seat $idx ($opt) pid=${PIDS[$idx]} -> $console"
    # Stagger so four 6144-rank launches do not thundering-herd pals RPC.
    sleep "${LAUNCH_STAGGER:-60}"
done

log "all ${#SEATS[@]} seats launched; waiting..."
declare -a RC
for idx in "${!SEATS[@]}"; do
    wait "${PIDS[$idx]}"; RC[$idx]=$?
    log "seat $idx (${S_OPT[$idx]}) finished rc=${RC[$idx]}"
done

# ---- Summary -- read the CURVE, not the exit code ----------------------------
echo ""
echo "============================================================"
echo "  30B LR-finder umbrella results (jobid=$JOBID)"
echo "============================================================"
for idx in "${!SEATS[@]}"; do
    opt="${S_OPT[$idx]}"
    console="$LOGDIR/seat-${idx}-${opt}.console.log"
    printf "  %-10s rc=%-4s " "$opt" "${RC[$idx]}"
    sug=$(grep -hoE "suggested_lr[^ ]*[ =:]+[0-9.eE+-]+|Suggested LR[^0-9]*[0-9.eE+-]+" "$console" 2>/dev/null | tail -1)
    nan=$(grep -ciE "nan|inf" "$console" 2>/dev/null)
    if [[ -n "$sug" ]]; then echo "$sug   (nan/inf mentions: $nan)"
    else echo "NO SUGGESTED LR IN LOG -- read $console"; fi
done
echo ""
echo "  A seat can exit 0 and still have found NOTHING: if its loss never"
echo "  turns over across the window the curve has no minimum, and if it NaNs"
echo "  in the first probes the minimum is an artifact. Check both before"
echo "  quoting a number."
echo "  CSV/plot/npz: ${DUMP_BASE}/<optimizer>/lr_finder/ezpz.agpt/"
echo "============================================================"
