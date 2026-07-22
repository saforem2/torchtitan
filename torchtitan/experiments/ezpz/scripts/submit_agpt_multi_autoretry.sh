#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N agpt-multi-autoretry
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
# select= overridden via qsub -l select=N. Default production layout:
#   1536 train (512+512+256+256) + per-trainer SPARES.
#   SPARES=10 -> select = 1536 + 4*10 = 1576.
#PBS -q prod
#PBS -j oe

# ============================================================================
# Umbrella job: advance ALL 4 canonical AGPT chains in ONE PBS allocation,
# each launched via NATIVE `ezpz launch --auto-retry` (ezpz >= 0.17.1).
# ============================================================================
#
# Why this exists
# ---------------
# The 512N canonical chains (2B + 20B) starve for weeks in the `small` queue:
# a 522-node scatter request rarely wins a slot. Packing all four canonical
# chains (2B-512, 20B-512, 2B-256, 20B-256) into ONE ~1536-train-node job
# routes to the far-less-contended `medium` queue (1025-1999 band) and
# advances every chain in a single dispatch.
#
# Why native auto-retry (this file) vs the legacy failover_lib.sh umbrella
# -----------------------------------------------------------------------
# The legacy `submit_agpt_multi_aurora_venv_failover.sh` drives each chain
# through the bash `failover_lib.sh` wrapper, whose bad-node scraper CANNOT
# parse a hostname out of a `died from signal 11` line, so it blind-swaps
# innocent spares and exhausts retries on recoverable events. That umbrella
# (8568429) lost 3 of 4 co-allocated chains that way. `ezpz launch
# --auto-retry` has a better scraper (owns split + scrape + swap + retry),
# validated at small scale by smoke_multi_autoretry.sh (8639375, 0/2 failed).
# This file promotes that smoke to production: real clones, real ckpt dirs,
# per-model venvs.
#
# How it works
# ------------
# THIN orchestrator; reimplements no training/failover logic. It:
#   1. Pre-stages each model's venv tarball to a per-model node-local dir
#      (/tmp/.venv-<model>), deduped by RESOLVED tarball so clones that share
#      one tarball (20b-n256 symlinks 20b-v2's) collapse to one broadcast.
#      `ezpz launch --auto-retry` only READS the venv (never re-yeets), so
#      concurrent trainers on disjoint slices never collide on it.
#   2. Slices the flat $PBS_NODEFILE into 4 DISJOINT hostfiles (active+spares
#      per chain).
#   3. Fires 4 concurrent `ezpz launch --auto-retry` -- each cd'd to its OWN
#      pinned clone (torchtitan imports from cwd; 20b clones are rolled back
#      pre-#3623 for DCP-resume compat, so the cwd is load-bearing), with a
#      distinct MASTER_PORT and CKPT_DIR. One chain crashing does NOT abort
#      the others.
#   4. wait()s on all 4 and reports per-trainer exit codes.
#
# The training command below is a faithful inline copy of the flag set in
# scripts/submit_agpt_{2b,20b}_autoretry.sh (same config, optimizer, LR, GBS
# arithmetic, validator, checkpoint policy) -- inlined rather than calling the
# clones' child scripts so a bug in a pinned clone's script can't break the
# umbrella, and so there is no palsd cross-kill between co-allocated trainers.
#
# Usage
# -----
#   # Production (real 1576-node submit, 1536 train + 4*10 spare):
#   qsub -q prod -A AuroraGPT -l select=1576 -l walltime=12:00:00 \
#     -l filesystems=home:flare -N agpt-multi-autoretry -j oe \
#     torchtitan/experiments/ezpz/scripts/submit_agpt_multi_autoretry.sh
#
#   # Dry-run on a login node (no allocation, zero cost) -- slice + print plan:
#   DRY_RUN=1 PBS_JOBID=9999999 PBS_O_WORKDIR=$PWD \
#     PBS_NODEFILE=/path/to/fixture-1576-hosts.txt \
#     bash torchtitan/experiments/ezpz/scripts/submit_agpt_multi_autoretry.sh
#
#   # Tiny real concurrency (debug-scaling, <=256 nodes) -- 4x 2B on throwaway
#   # ckpt dirs, tiny steps:
#   qsub -q debug-scaling -A AuroraGPT -l select=8 -l walltime=01:00:00 \
#     -l filesystems=home:flare -N agpt-multi-autoretry-smoke -j oe \
#     -v MULTI_PROFILE=tiny \
#     torchtitan/experiments/ezpz/scripts/submit_agpt_multi_autoretry.sh
#
# Env knobs
# ---------
#   MULTI_PROFILE   prod (default) | tiny -- tiny shrinks every chain to
#                   MULTI_TINY_NNODES nodes (all 2B), runs MULTI_TINY_STEPS
#                   steps, and writes THROWAWAY ckpt dirs (never a canonical
#                   chain).
#   SPARES          spare nodes per chain for --auto-retry (default 10).
#   DRY_RUN         1 -> slice + print resolved plan, then exit (no launch).
#   LAUNCH_STAGGER  seconds between background launches (default 20) so the
#                   concurrent launches don't thundering-herd pals.
#   MAX_FAILOVER_RETRIES  passed to each `ezpz launch` (default: unbounded).
#   IDLE_TIMEOUT    per-launch idle timeout, seconds (default 1800).
#   CKPT_INTERVAL   checkpoint every N steps (default 100).
#   MULTI_TINY_NNODES (default 2), MULTI_TINY_SPARES (default 1),
#   MULTI_TINY_STEPS (default 20) -- tiny-profile sizing.
#
# NOTE: deliberately NO `set -euo pipefail`. venv activation trips unbound
# vars, and the umbrella must survive a single child failing without aborting
# the rest. Only `pipefail` is safe here.
# ============================================================================

set -o pipefail

RUNS="/flare/AuroraGPT/foremans/runs"

PROFILE="${MULTI_PROFILE:-prod}"
SPARES="${SPARES:-10}"
DRY_RUN="${DRY_RUN:-0}"
LAUNCH_STAGGER="${LAUNCH_STAGGER:-20}"
TINY_NNODES="${MULTI_TINY_NNODES:-2}"
TINY_SPARES="${MULTI_TINY_SPARES:-1}"
TINY_STEPS="${MULTI_TINY_STEPS:-20}"
IDLE_TIMEOUT="${IDLE_TIMEOUT:-1800}"
CKPT_INTERVAL="${CKPT_INTERVAL:-100}"
PPN="${NGPU_PER_HOST:-12}"
JOBID="${PBS_JOBID%%.*}"
[[ -z "$JOBID" ]] && JOBID="nojob"

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"

log() { echo "[multi $(date +%H:%M:%S)] $*"; }
die() { echo "[multi ERROR] $*" >&2; exit 1; }

# ---- Per-trainer config table -------------------------------------------------
# Format (5 required + 6 optional trailing override fields; empty override =>
# fall back to the global default so a plain 5-field row is unchanged behavior):
#   model|nnodes|master_port|workdir(clone; cwd == torchtitan source)|
#     ckpt_dir(relative; job.dump_folder=./outputs PREPENDS outputs/)|
#     dfl_name|lr|initial_load_path|decay_ratio|min_lr_factor|train_tokens
# GBS is computed per-slice by the launch as NGPUS_ACTIVE*LBS(2)*GAS(1):
#   512*12*2 = 12288 ; 256*12*2 = 6144. The GBS in each ckpt dir name below
# MUST match, or the chain fresh-starts instead of resuming.
# The overrides let ONE trainer run a different recipe (e.g. the 2B stage-2
# mid-training fork below) while the others keep their olmo-mix chains.
# Bail a trainer that produces N consecutive non-finite (NaN/inf) losses
# instead of burning the whole allocation. A dolmino-mix CPT trainer
# NaN'd at step 3801 in job 8663177 and ran ~370 NaN steps unbounded
# because this was unset (default 0 = off). 5 matches the 80B script.
# Default OFF: the pinned pre-#3623 production clones' train.py has no
# nan_abort_consecutive field, so passing the flag crashes every rank
# ("Unrecognized options: --nan-abort-consecutive", rc=143 -- killed umbrella
# 8680578). Set NAN_ABORT_CONSECUTIVE=5 explicitly only on a HEAD-based clone
# that has the field. Empty => flag is omitted entirely (see nan_abort_args).
NAN_ABORT_CONSECUTIVE="${NAN_ABORT_CONSECUTIVE:-}"
nan_abort_args=()
if [[ -n "${NAN_ABORT_CONSECUTIVE}" ]]; then
    nan_abort_args=(--nan-abort-consecutive="${NAN_ABORT_CONSECUTIVE}")
fi

TRAINERS=(
    "2b|512|29500|$RUNS/agpt-2b-v2/torchtitan-ezpz|checkpoints/agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288"
    "20b|512|29600|$RUNS/agpt-20b-v2/torchtitan-ezpz|checkpoints/agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288"
    # DROPPED 2026-07-18: this 50/50 dolmino CPT trainer NaN-diverged at
    # step 3801 in job 8663177 (single-step overflow on a dolmino batch)
    # and NaN-wrote to step 6600. Config preserved for a fixed retry
    # (gentler LR / data audit) but removed from the production umbrella.
#    "2b|256|29700|$RUNS/agpt-2b-v2/torchtitan-ezpz|checkpoints/agpt-2b-stage2-olmo50dolmino50-const2e6-n256-gbs6144|olmo50-dolmino50|2e-6|$RUNS/agpt-2b-v2/torchtitan-ezpz/outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144/step-92859|0.0|1.0|2391000000000"
    "20b|256|29800|$RUNS/agpt-20b-n256/torchtitan-ezpz|checkpoints/agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144"
)

# Shared training defaults (match submit_agpt_{2b,20b}_autoretry.sh).
SEQ_LEN="${SEQ_LEN:-8192}"
LBS="${LBS:-2}"
GAS="${GAS:-1}"
TP="${TP:-1}"; PP="${PP:-1}"; CP="${CP:-1}"
OPTIMIZER="${OPTIMIZER:-sophiag}"
LR="${LR:-2.28e-5}"
DFL_NAME="${DFL_NAME:-olmo-mix-1124}"
TRAIN_TOKENS="${TRAIN_TOKENS:-4673780159710}"
CONFIG_SUFFIX="${CONFIG_SUFFIX-_real}"
VALIDATOR_FREQ="${VALIDATOR_FREQ:-100}"
VALIDATOR_STEPS="${VALIDATOR_STEPS:-10}"

# ---- Validate allocation ------------------------------------------------------
[[ -n "${PBS_NODEFILE:-}" && -f "$PBS_NODEFILE" ]] \
    || die "PBS_NODEFILE not set or missing (got '${PBS_NODEFILE:-}')"
TOTAL_AVAIL=$(wc -l < "$PBS_NODEFILE")

need=0
for row in "${TRAINERS[@]}"; do
    IFS='|' read -r _m nnodes _p _w _c <<< "$row"
    if [[ "$PROFILE" == "tiny" ]]; then
        need=$(( need + TINY_NNODES + TINY_SPARES ))
    else
        need=$(( need + nnodes + SPARES ))
    fi
done
log "profile=$PROFILE  spares/chain=$([[ $PROFILE == tiny ]] && echo "$TINY_SPARES" || echo "$SPARES")"
log "nodes: need=$need  available=$TOTAL_AVAIL  (jobid=$JOBID)"
if (( TOTAL_AVAIL < need )); then
    if (( DRY_RUN == 1 )); then
        log "WARNING (dry-run): allocation $TOTAL_AVAIL < need $need; slices will be short"
    else
        die "allocation too small: have $TOTAL_AVAIL nodes, need $need"
    fi
fi

# ---- Slice dir (ABSOLUTE so children resolve it after cd-ing to their clone) --
MULTI_LOG_DIR="${MULTI_LOG_DIR:-${PBS_O_WORKDIR:-$PWD}/logs/multi-autoretry-${JOBID}}"
mkdir -p "$MULTI_LOG_DIR" || die "cannot create $MULTI_LOG_DIR"
case "$MULTI_LOG_DIR" in
    /*) : ;;
    *)  MULTI_LOG_DIR="$(cd "$MULTI_LOG_DIR" && pwd)" ;;
esac
log "slice + console logs under: $MULTI_LOG_DIR"

# ---- Slice the nodefile + build each trainer's resolved plan ------------------
declare -a T_MODEL T_NNODES T_SPARES T_PORT T_WORKDIR T_CKPT T_SLICE T_VENVDST T_VENVSRC
declare -a T_DFL T_LR T_INITLOAD T_DECAY T_MINLR T_TOKENS
offset=1
for idx in "${!TRAINERS[@]}"; do
    IFS='|' read -r model nnodes port workdir ckpt o_dfl o_lr o_init o_decay o_minlr o_tokens <<< "${TRAINERS[$idx]}"

    # In tiny profile every trainer runs the 2B model on TINY_NNODES nodes and
    # writes a THROWAWAY ckpt dir keyed by jobid+idx -- NEVER a canonical chain.
    child_model="$model"
    if [[ "$PROFILE" == "tiny" ]]; then
        child_model="2b"; model="2b"
        nnodes="$TINY_NNODES"; spares="$TINY_SPARES"
        workdir="$RUNS/agpt-2b-v2/torchtitan-ezpz"
        ckpt="checkpoints/multi-autoretry-smoke-${JOBID}/t${idx}-n${nnodes}"
    else
        spares="$SPARES"
    fi

    slice_count=$(( nnodes + spares ))
    slice_file="$MULTI_LOG_DIR/trainer-${idx}.hostfile"
    sed -n "${offset},$((offset + slice_count - 1))p" "$PBS_NODEFILE" > "$slice_file"
    got=$(wc -l < "$slice_file")
    if (( got != slice_count && DRY_RUN != 1 )); then
        die "trainer $idx: sliced $got nodes, expected $slice_count (offset=$offset)"
    fi
    offset=$(( offset + slice_count ))

    T_MODEL[$idx]="$model"
    T_NNODES[$idx]="$nnodes"
    T_SPARES[$idx]="$spares"
    T_PORT[$idx]="$port"
    T_WORKDIR[$idx]="$workdir"
    T_CKPT[$idx]="$ckpt"
    T_SLICE[$idx]="$slice_file"
    # Optional per-trainer recipe overrides (empty -> global fallback in launch).
    T_DFL[$idx]="$o_dfl"
    T_LR[$idx]="$o_lr"
    T_INITLOAD[$idx]="$o_init"
    T_DECAY[$idx]="$o_decay"
    T_MINLR[$idx]="$o_minlr"
    T_TOKENS[$idx]="$o_tokens"
    # Per-model venv dst -- distinct names keep the shared mom node's 2b/20b
    # copies from clobbering each other; on disjoint compute slices the same
    # dst name resolves to physically distinct node-local dirs.
    T_VENVDST[$idx]="/tmp/.venv-${child_model}"
    _vsrc="$workdir/.venv.tar.gz"
    [[ -e "$_vsrc" ]] && _vsrc="$(readlink -f "$_vsrc")"
    T_VENVSRC[$idx]="$_vsrc"
done

# ---- Disjointness assertion (no node may appear in two slices) ---------------
dupes=$(cat "$MULTI_LOG_DIR"/trainer-*.hostfile 2>/dev/null | sort | uniq -d)
if [[ -n "$dupes" ]]; then
    die "slices overlap -- the same node(s) appear in multiple trainers:"$'\n'"$dupes"
fi
log "slices are disjoint (no shared nodes)"

# ---- Existence checks ---------------------------------------------------------
for idx in "${!TRAINERS[@]}"; do
    [[ -d "${T_WORKDIR[$idx]}" ]] || die "trainer $idx workdir missing: ${T_WORKDIR[$idx]}"
    [[ -f "${T_VENVSRC[$idx]}" ]] || die "trainer $idx venv tarball missing: ${T_VENVSRC[$idx]}"
done

# ---- Resolved-plan banner -----------------------------------------------------
echo "============================================================"
echo "  agpt multi-chain umbrella (native auto-retry)  jobid=$JOBID"
echo "  profile=$PROFILE  total_avail=$TOTAL_AVAIL  need=$need"
echo "============================================================"
for idx in "${!TRAINERS[@]}"; do
    printf '  trainer %d: %-4s n=%-4s spare=%-3s port=%s\n' \
        "$idx" "${T_MODEL[$idx]}" "${T_NNODES[$idx]}" "${T_SPARES[$idx]}" "${T_PORT[$idx]}"
    printf '             clone : %s\n' "${T_WORKDIR[$idx]}"
    printf '             ckpt  : %s\n' "${T_CKPT[$idx]}"
    printf '             slice : %s (%s nodes)\n' "${T_SLICE[$idx]}" "$(wc -l < "${T_SLICE[$idx]}")"
    printf '             venv  : %s <- %s\n' "${T_VENVDST[$idx]}" "${T_VENVSRC[$idx]}"
done
echo "============================================================"

if (( DRY_RUN == 1 )); then
    log "DRY_RUN=1 -> plan printed, no launch. exiting 0."
    exit 0
fi

# ---- ezpz utils (for ezpz_load_modules / ezpz_setup_job / yeet-env) ----------
EZPZ_UTILS="${PBS_O_WORKDIR:-$PWD}/.ezpz-utils-cache/ezpz-utils.sh"
if [[ -f "$EZPZ_UTILS" ]]; then
    source "$EZPZ_UTILS"
else
    source <(curl -fsSL --max-time 30 https://bit.ly/ezpz-utils)
fi

# ---- Pre-stage each model's venv (dedup by resolved src|dst) ------------------
# Activate trainer 0's clone venv so the `ezpz` CLI is on PATH for yeet-env.
if [[ -f "${T_WORKDIR[0]}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${T_WORKDIR[0]}/.venv/bin/activate" || die "cannot activate ${T_WORKDIR[0]}/.venv for ezpz CLI"
fi
ezpz_load_modules

declare -A _staged=()
for idx in "${!TRAINERS[@]}"; do
    src="${T_VENVSRC[$idx]}"; dst="${T_VENVDST[$idx]}"
    key="${src}|${dst}"
    [[ -n "${_staged[$key]:-}" ]] && continue
    _staged[$key]=1
    # Union hostfile: every slice sharing this (src,dst) contributes its nodes.
    stage_hf="$MULTI_LOG_DIR/prestage-$(basename "$dst").hostfile"
    : > "$stage_hf"
    for j in "${!TRAINERS[@]}"; do
        if [[ "${T_VENVSRC[$j]}|${T_VENVDST[$j]}" == "$key" ]]; then
            cat "${T_SLICE[$j]}" >> "$stage_hf"
        fi
    done
    sort -u "$stage_hf" -o "$stage_hf"
    log "prestage: $src -> $dst on $(wc -l < "$stage_hf") nodes"
    ezpz yeet-env --src "$src" --hostfile "$stage_hf" --dst "$dst"
    rc=$?
    (( rc == 0 )) || die "prestage failed (rc=$rc) for $src -> $dst"
    log "prestage OK: $dst"
done
[[ -n "${VIRTUAL_ENV:-}" ]] && deactivate 2>/dev/null

# ---- Launch one trainer via native `ezpz launch --auto-retry` ----------------
declare -a PIDS
launch_trainer() {
    local idx="$1"
    local model="${T_MODEL[$idx]}"
    local nnodes="${T_NNODES[$idx]}"
    local spares="${T_SPARES[$idx]}"
    local port="${T_PORT[$idx]}"
    local workdir="${T_WORKDIR[$idx]}"
    local ckpt="${T_CKPT[$idx]}"
    local slice="${T_SLICE[$idx]}"
    local venvdst="${T_VENVDST[$idx]}"
    local console="$MULTI_LOG_DIR/trainer-${idx}-${model}-n${nnodes}.console.log"

    local nproc=$(( nnodes * PPN ))
    local gbs=$(( nproc * LBS * GAS / (TP * PP * CP) ))
    # Per-trainer recipe overrides (fall back to the global default when empty).
    local dfl_name="${T_DFL[$idx]:-$DFL_NAME}"
    local lr="${T_LR[$idx]:-$LR}"
    local tok="${T_TOKENS[$idx]:-$TRAIN_TOKENS}"
    local training_steps
    if [[ "$PROFILE" == "tiny" ]]; then
        training_steps="$TINY_STEPS"
    else
        training_steps=$(( tok / (gbs * SEQ_LEN) ))
    fi

    local dfl="torchtitan/experiments/ezpz/data-lists/aurora/${dfl_name}.txt"
    local data_cache="${ckpt}/.cache/${dfl_name}/index-cache"

    # Optional fork (initial-load) + constant/decayed-LR schedule overrides.
    local xtra=()
    [[ -n "${T_INITLOAD[$idx]}" ]] && xtra+=(--checkpoint.initial-load-path="${T_INITLOAD[$idx]}")
    if [[ -n "${T_DECAY[$idx]}" ]]; then
        xtra+=(--lr-scheduler.decay-ratio="${T_DECAY[$idx]}"
               --lr-scheduler.min-lr-factor="${T_MINLR[$idx]:-1.0}"
               --lr-scheduler.warmup-steps=20)
    fi

    local val_flags
    if [[ "$PROFILE" == "tiny" ]]; then
        val_flags="--validator.no-enable"
    else
        val_flags="--validator.enable --validator.freq=${VALIDATOR_FREQ} --validator.steps=${VALIDATOR_STEPS} --validator.dataloader.dataset-path=${dfl} --validator.dataloader.data-cache-path=${data_cache}"
    fi

    local mfr=()
    [[ -n "${MAX_FAILOVER_RETRIES:-}" ]] && mfr=(--max-failover-retries "${MAX_FAILOVER_RETRIES}")

    (
        cd "$workdir" || { echo "cd failed: $workdir"; exit 97; }
        # Import torchtitan from THIS clone's (pinned) source tree; run the
        # driver from the node-local per-model venv.
        # shellcheck disable=SC1090
        source "$venvdst/bin/activate" || { echo "activate failed: $venvdst"; exit 98; }
        export MASTER_PORT="$port"; unset MASTER_ADDR
        # shellcheck disable=SC2086
        ezpz launch \
            --nproc "$nproc" \
            --nproc_per_node "$PPN" \
            --hostfile "$slice" \
            --auto-retry \
            --spare-nodes "$spares" \
            --timeout "$IDLE_TIMEOUT" \
            "${mfr[@]}" \
            -- \
            python3 -m torchtitan.experiments.ezpz.train \
            --module=ezpz.agpt \
            --config="agpt_${model}${CONFIG_SUFFIX:-}" \
            --checkpoint.enable \
            --checkpoint.folder="$ckpt" \
            --checkpoint.interval="$CKPT_INTERVAL" \
            --checkpoint.keep-latest-k=0 \
            "${nan_abort_args[@]}" \
            --checkpoint.no-last-save-model-only \
            --checkpoint.async-mode=disabled \
            --dataloader.dataset=blendcorpus \
            --dataloader.dataset-path="$dfl" \
            --dataloader.data-cache-path="$data_cache" \
            $val_flags \
            "${xtra[@]}" \
            --optimizer="$OPTIMIZER" \
            --optimizer.lr="$lr" \
            --training.local-batch-size="$LBS" \
            --training.global-batch-size="$gbs" \
            --training.seq-len="$SEQ_LEN" \
            --training.steps="$training_steps"
    ) > "$console" 2>&1 &
    PIDS[$idx]=$!
    log "launched trainer $idx ($model n=$nnodes nproc=$nproc gbs=$gbs steps=$training_steps spares=$spares) pid=${PIDS[$idx]} -> $console"
    sleep "$LAUNCH_STAGGER"
}

for idx in "${!TRAINERS[@]}"; do launch_trainer "$idx"; done
log "all ${#TRAINERS[@]} trainers launched; waiting..."

# ---- Wait + per-trainer rc ---------------------------------------------------
declare -a RC
fails=0
for idx in "${!TRAINERS[@]}"; do
    wait "${PIDS[$idx]}"; RC[$idx]=$?
    (( RC[$idx] == 0 )) || fails=$(( fails + 1 ))
    log "trainer $idx finished rc=${RC[$idx]}"
done

echo "============================================================"
echo "  multi-autoretry summary (jobid=$JOBID)"
echo "============================================================"
for idx in "${!TRAINERS[@]}"; do
    status="OK"; (( RC[$idx] == 0 )) || status="FAIL(${RC[$idx]})"
    printf '  trainer %d: %-4s n=%-4s %-10s %s\n' \
        "$idx" "${T_MODEL[$idx]}" "${T_NNODES[$idx]}" "$status" \
        "$MULTI_LOG_DIR/trainer-${idx}-${T_MODEL[$idx]}-n${T_NNODES[$idx]}.console.log"
done
echo "  failed: $fails / ${#TRAINERS[@]}"
echo "============================================================"
exit "$fails"
