#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N smoke-multi-autoretry
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug-scaling
#PBS -j oe
#
# Small multi-chain umbrella smoke on NATIVE `ezpz launch --auto-retry`.
#
# Purpose: reproduce, at tiny scale, the co-allocated multi-trainer pattern
# of submit_agpt_multi_aurora_venv_failover.sh, but launch each trainer with
# `ezpz launch --auto-retry` (ezpz >= 0.17.1) instead of the legacy
# scripts/failover_lib.sh blind-swap path. The prod umbrella 8568429 lost 3/4
# chains to the legacy failover exhausting retries on recoverable bad-node
# events (rank signal 11 / node-unreachable); this smoke exercises whether the
# native auto-retry scraper behaves better under the same co-allocation.
#
# Design notes:
#   - ONE venv broadcast for the whole allocation (all trainers share the 2B
#     venv), so there is NO concurrent-yeet collision on /tmp/.venv (the
#     single-job-per-alloc hazard that corrupts N simultaneous yeets).
#   - `ezpz launch --auto-retry` does NOT yeet; it takes --hostfile + --nproc
#     + --spare-nodes and runs the command against the already-staged
#     /tmp/.venv. So we split PBS_NODEFILE into per-trainer slices (active +
#     per-trainer spares) and fire N concurrent launches.
#   - Each trainer gets a distinct CKPT_DIR (throwaway SMOKE dir), master port,
#     and failover log dir. Tiny step count.
#
# Submit (debug-scaling, <=256 nodes, 1h):
#   qsub -l select=<NT*(NN+SP)> -v NUM_TRAINERS=2,TRAINER_NNODES=2,TRAINER_SPARES=1 \
#     torchtitan/experiments/ezpz/scripts/smoke_multi_autoretry.sh
#   e.g. 2 trainers x (2 active + 1 spare) = select=6.

set -o pipefail

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"

NUM_TRAINERS="${NUM_TRAINERS:-2}"
TRAINER_NNODES="${TRAINER_NNODES:-2}"     # active nodes per trainer
TRAINER_SPARES="${TRAINER_SPARES:-1}"     # spare nodes per trainer (for --auto-retry)
TRAINER_STEPS="${TRAINER_STEPS:-20}"
MODEL="${MODEL:-2b}"
CONFIG_SUFFIX="${CONFIG_SUFFIX-_real}"
PPN="${NGPU_PER_HOST:-12}"
IDLE_TIMEOUT="${IDLE_TIMEOUT:-600}"

cd "${PBS_O_WORKDIR:-$(pwd)}"
REPO="$(pwd)"
JOBID="${PBS_JOBID%%.*}"; [[ -z "$JOBID" ]] && JOBID="nojob"
LOGDIR="$REPO/logs/smoke-multi-autoretry-${JOBID}"
mkdir -p "$LOGDIR"
log() { echo "[smoke-multi $(date +%H:%M:%S)] $*"; }

# ---- ezpz utils (for ezpz_load_modules / ezpz_setup_job / yeet) ----
EZPZ_UTILS="${PBS_O_WORKDIR:-$PWD}/.ezpz-utils-cache/ezpz-utils.sh"
if [[ -f "$EZPZ_UTILS" ]]; then source "$EZPZ_UTILS"; else source <(curl -fsSL --max-time 30 https://bit.ly/ezpz-utils); fi

# ---- Validate allocation ----
[[ -n "${PBS_NODEFILE:-}" && -f "$PBS_NODEFILE" ]] || { echo "no PBS_NODEFILE"; exit 2; }
TOTAL=$(wc -l < "$PBS_NODEFILE")
PER_TRAINER=$(( TRAINER_NNODES + TRAINER_SPARES ))
NEED=$(( NUM_TRAINERS * PER_TRAINER ))
log "allocation: $TOTAL nodes; need $NEED ($NUM_TRAINERS x $PER_TRAINER)"
(( TOTAL >= NEED )) || { echo "not enough nodes: have $TOTAL need $NEED"; exit 2; }

# ---- ONE venv broadcast for the whole allocation (no concurrent-yeet race) ----
ezpz_load_modules
ezpz_setup_job
source .venv/bin/activate
if [[ -f .venv.tar.gz ]]; then ezpz yeet --src .venv.tar.gz; else ezpz yeet; fi
deactivate
source /tmp/.venv/bin/activate
log "venv broadcast complete -> /tmp/.venv on all $TOTAL nodes"

# ---- Split PBS_NODEFILE into per-trainer slices ----
split_dir="$LOGDIR/slices"; mkdir -p "$split_dir"
mapfile -t ALL_NODES < "$PBS_NODEFILE"
declare -a SLICE
for (( t=0; t<NUM_TRAINERS; t++ )); do
    start=$(( t * PER_TRAINER ))
    SLICE[$t]="$split_dir/trainer-${t}.hostfile"
    printf '%s\n' "${ALL_NODES[@]:$start:$PER_TRAINER}" > "${SLICE[$t]}"
    log "trainer $t slice: $(wc -l < "${SLICE[$t]}") nodes -> ${SLICE[$t]}"
done

# ---- Launch one trainer via native ezpz launch --auto-retry ----
declare -a PIDS
launch_trainer() {
    local t="$1"
    local console="$LOGDIR/trainer-${t}.console.log"
    local nproc=$(( TRAINER_NNODES * PPN ))
    local port=$(( 29500 + t ))
    local ckpt="checkpoints/agpt-${MODEL}-SMOKE-multiautoretry-${JOBID}-t${t}"
    (
        export MASTER_PORT="$port"; unset MASTER_ADDR
        # --auto-retry: split the slice into active+spare internally from --nproc.
        # Venv already staged (/tmp/.venv), so launch does NOT re-yeet.
        ezpz launch \
            --nproc "$nproc" \
            --nproc_per_node "$PPN" \
            --hostfile "${SLICE[$t]}" \
            --auto-retry \
            --spare-nodes "$TRAINER_SPARES" \
            --timeout "$IDLE_TIMEOUT" \
            --max-failover-retries 3 \
            -- \
            python3 -m torchtitan.experiments.ezpz.train \
            --module=ezpz.agpt \
            --config="agpt_${MODEL}${CONFIG_SUFFIX:-}" \
            --checkpoint.enable \
            --checkpoint.folder="$ckpt" \
            --checkpoint.interval=999999 \
            --checkpoint.keep-latest-k=0 \
            --checkpoint.async-mode=disabled \
            --dataloader.dataset=blendcorpus \
            --dataloader.dataset-path="torchtitan/experiments/ezpz/data-lists/aurora/olmo-mix-1124.txt" \
            --dataloader.data-cache-path="${ckpt}/.cache/olmo-mix-1124/index-cache" \
            --validator.no-enable \
            --optimizer=sophiag \
            --optimizer.lr=2.28e-5 \
            --training.local-batch-size=2 \
            --training.seq-len=8192 \
            --training.steps="$TRAINER_STEPS"
    ) > "$console" 2>&1 &
    PIDS[$t]=$!
    log "launched trainer $t (nproc=$nproc, spares=$TRAINER_SPARES) pid=${PIDS[$t]} -> $console"
    sleep 15   # small stagger so N concurrent launches don't thundering-herd pals
}

for (( t=0; t<NUM_TRAINERS; t++ )); do launch_trainer "$t"; done
log "all $NUM_TRAINERS trainers launched; waiting..."

# ---- Wait + per-trainer rc ----
declare -a RC
fails=0
for (( t=0; t<NUM_TRAINERS; t++ )); do
    wait "${PIDS[$t]}"; RC[$t]=$?
    (( RC[$t] == 0 )) || fails=$(( fails + 1 ))
    log "trainer $t finished rc=${RC[$t]}"
done

echo "============================================================"
echo "  smoke-multi-autoretry summary (jobid=$JOBID)"
echo "============================================================"
for (( t=0; t<NUM_TRAINERS; t++ )); do
    status="OK"; (( RC[$t] == 0 )) || status="FAIL(${RC[$t]})"
    printf '  trainer %d: %-10s %s\n' "$t" "$status" "$LOGDIR/trainer-${t}.console.log"
done
echo "  failed: $fails / $NUM_TRAINERS"
echo "============================================================"
exit "$fails"
