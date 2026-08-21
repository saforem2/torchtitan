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
#   MULTI_ONLY      comma-separated trainer indices to run (default: all).
#                   Keeps each seat's REAL ckpt/config/dataset/RoPE, unlike
#                   MULTI_PROFILE=tiny which rewrites them to throwaways.
#   MULTI_NNODES_OVERRIDE  force every selected seat to N nodes (smoke sizing).
#   MULTI_STEPS_OVERRIDE   force every selected seat to N training steps. Only
#                   useful with MULTI_ONLY: the prod step count is derived from
#                   train_tokens and is ~6M, so a seed-validation smoke would
#                   otherwise run until walltime instead of exiting on its own.
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
    # t0 -- 2B STAGE 2 (continued pre-training on dolmino-mix-1124).
    # The stage-1 512N chain COMPLETED 2026-08-13 at step 46,429 = 4.6737T
    # tokens, so its old slot here would have had no budget left. Replaced
    # with stage 2, seeded from that finished endpoint.
    #
    # Recipe mirrors MDS train_aGPT_2B_sophiag_stage2.sh:
    #   DATA_FILE_LIST=dolmino-mix-1124 (fused)   LR=2.17e-5
    #   LR_DECAY_STYLE=constant -> decay_ratio 0.0, min_lr_factor 1.0
    #   OPT=sophiag
    #
    # TRAIN_TOKENS here is the STAGE-2 INCREMENT (2.390375382006T), NOT MDS's
    # cumulative 7064155541716. Fixed 2026-08-16 after job 8756070: stage 2
    # writes to a NEW ckpt dir, so its step counter starts at 0 and the
    # umbrella's steps = tok/(gbs*seq_len) turns a cumulative figure into
    # 70,176 steps of pure dolmino (7.06T) -- 2.96x the intended 2.39T.
    # Harmless to the LR shape (constant LR: decay_ratio 0, min_lr_factor 1),
    # so it only moved the stopping point, but it was still wrong.
    #   2390375382006 / (12288 * 8192) = 23,742 steps  <- intended
    #
    # Motivation: the 2026-08-14 eval of the finished stage-1 endpoint showed
    # the last 500B tokens moved NO metric and MMLU ended at chance (0.2511).
    # More olmo-mix is demonstrably not the lever; a different mix might be.
    #
    # NOT the 2026-07-18 config that NaN'd (job 8663177, step 3801): that was
    # olmo50-dolmino50 at LR 2e-6 -- a TENTH of this LR -- so its LR was not
    # the cause; it was a single-step overflow on a dolmino batch. This run
    # matches MDS (pure dolmino, 2.17e-5), which survived the full stage, and
    # relies on NAN_ABORT_CONSECUTIVE to bail early instead of NaN-writing to
    # step 6600 the way that attempt did.
    "2b|512|29500|$RUNS/agpt-2b-v2/torchtitan-ezpz|checkpoints/agpt-2b-stage2-dolmino-n512-gbs12288|dolmino-mix-1124|2.17e-5|$RUNS/agpt-2b-v2/torchtitan-ezpz/outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288/step-46429|0.0|1.0|2390375382006|_real"
    # 20B-512 CONSTANT-LR FORK from step-9000.
    #
    # This row used to stop after the ckpt dir. An EMPTY decay_ratio field does
    # NOT inherit the constant-LR default -- launch_one only passes
    # --lr-scheduler.decay-ratio when T_DECAY is non-empty, so an omitted field
    # silently falls through to torchtitan's OWN default of 0.8. The chain
    # therefore ran warmup-stable-decay when constant LR was intended.
    #
    # decay_ratio=0.8 means decay occupies the LAST 80% of training, i.e. it
    # STARTS at 20%: round(46429*0.8)=37143 decay steps, so stable ends at
    # 46429+1-200-37143 = step 9087. The chain reached 10,200 before this was
    # caught, so ~1,200 steps ran on a decaying LR (2.28e-5 -> 2.22389e-5).
    #
    # Fork from step-9000 (last checkpoint before onset; 6144 shards, valid
    # .metadata, 244G) into its own dir so the canonical chain is untouched,
    # exactly as the 2B constlr forks do. Now passes 0.0/1.0 EXPLICITLY.
    "20b|512|29600|$RUNS/agpt-20b-v2/torchtitan-ezpz|checkpoints/agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288-constlr-from9000|olmo-mix-1124|2.28e-5|$RUNS/agpt-20b-v2/torchtitan-ezpz/outputs/checkpoints/agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288/step-9000|0.0|1.0|4673780159710|_real"
    # DROPPED 2026-07-18: this 50/50 dolmino CPT trainer NaN-diverged at
    # step 3801 in job 8663177 (single-step overflow on a dolmino batch)
    # and NaN-wrote to step 6600. Config preserved for a fixed retry
    # (gentler LR / data audit) but removed from the production umbrella.
#    "2b|256|29700|$RUNS/agpt-2b-v2/torchtitan-ezpz|checkpoints/agpt-2b-stage2-olmo50dolmino50-const2e6-n256-gbs6144|olmo50-dolmino50|2e-6|$RUNS/agpt-2b-v2/torchtitan-ezpz/outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144/step-92859|0.0|1.0|2391000000000"
    # 20B-256: same missing-decay_ratio bug, but NO FORK NEEDED. Its onset is
    # 92859+1-200-round(92859*0.8) = step 18373 and the chain is at ~11,100, so
    # its LR is still flat at 2.28e-5. Passing 0.0/1.0 now means it simply never
    # decays -- no restart from an earlier checkpoint, nothing discarded.
    # Fixing this before step 18373 is what avoids a second fork.
    "20b|256|29800|$RUNS/agpt-20b-n256/torchtitan-ezpz|checkpoints/agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144|olmo-mix-1124|2.28e-5||0.0|1.0|4673780159710|_real"
    # 2B-512 constant-LR fork (from base step-9200, right before LR decay).
    # Own prod dir already holds the full step-9200 (model+optim); plain
    # resume-from-latest, decay_ratio=0.0 => constant LR (no decay phase).
    #
    # RoPE=_real, CORRECTED 2026-08-21 (was `complex`, which would have
    # corrupted this live 122-checkpoint chain). The `complex` value came from
    # resolving the flavor against the PARENT chain at the SEED step -- the
    # wrong rule for a fork that plain-resumes.
    #
    # THE RULE: for a fork with an EMPTY field 8, resolve RoPE against the
    # flavor that wrote the FORK'S OWN latest checkpoint, not the parent's at
    # the branch point. The seed's flavor governs only the very first launch
    # into an empty dir; after that the fork has its own history.
    #
    # Evidence: every t3 run launched --config=agpt_2b_real (verified across
    # umbrellas 8756957 and 8764675), and the fork's step-21300 was written
    # 2026-08-17, well after the 2026-06-25 cos_sin switch (5ffb850a1).
    # The fork DID pay the transition once: at its first resume (job 8744247)
    # step 9201 logged loss 6.51705 / grad_norm 18.4066 -- the complex-weights-
    # under-cos_sin signature -- then re-converged to 2.93 by step 9300. It has
    # been a cos_sin chain ever since. (That spike is also the true origin of
    # the "step-9200 resumes at ~6.5" note: step-9200 is NOT corrupt -- it is
    # fp32 with cos 0.999 to its parent. The 6.5 was the flavor transition.)
    "2b|512|29700|$RUNS/agpt-2b-constlr-from9200/torchtitan-ezpz|checkpoints/agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288-constlr-from9200|olmo-mix-1124|2.28e-5||0.0|1.0|4673780159710|_real"
    # RETIRED 2026-08-21 -- 2B-256 constant-LR fork "from step-9500".
    # Its seed was never a trained checkpoint. The file at
    # .../n256-gbs6144-constlr-from9500/step-9500 is near-random-init weights
    # mislabeled step-9500. Verified three independent ways:
    #   - dtype: bfloat16 throughout (13G). EVERY healthy 2B ckpt is float32
    #     only (24G, exactly 2x). No key rename can change a dtype.
    #   - stats: layers.0.attention_norm.weight is exactly 1.0 with std 0.0
    #     (RMSNorm gains never updated; parent is 0.921 +/- 0.0177);
    #     tok_embeddings std 0.99999 (unit normal); absmax exactly 2.0000 on
    #     every projection (truncation artifact).
    #   - cosine to parent ~ ZERO: wq +0.000663, wk -0.000178, wv +0.001649,
    #     tok_embeddings -0.000011, head +0.003091. All 6 off-diagonal
    #     wq/wk/wv pairings were also tested -- nothing above 0.9, so it is
    #     not a permuted/mis-mapped tensor either.
    # Its own train_state self-reports ntokens_seen=155,648,000 where step 9500
    # at gbs 6144 x 8192 should be ~478e9 -- 325x too few. The checkpoint
    # contradicts its own label. It resumed at loss 5.97 (not 12.45 = ln(vocab)
    # only because output.weight retains a weak unigram prior) and went flat:
    # that run was pretraining from scratch at a fine-tuning LR.
    # No genuine 256N step-9500 exists anywhere on the filesystem -- the
    # surviving agpt-2b-v2 n256 chain starts at step 35600 -- so the seat could
    # not be repaired, only redefined. Seven attempts, zero real steps.
#    "2b|256|29900|$RUNS/agpt-2b-constlr-from9200/torchtitan-ezpz|checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144-constlr-from9500|olmo-mix-1124|2.28e-5||0.0|1.0|4673780159710|complex"
    # 2B-256 STAGE-2 DOLMINO -- the 256N twin of t0, added 2026-08-21 in the
    # retired seat's place.
    #
    # Seeds (weights-only) from the COMPLETED 256N stage-1 chain at step-92859.
    # That chain finished: it exited cleanly 2026-07-03 with "Training starts
    # at step 92860 / Training completed", having hit its token budget
    # (92859 * 6144 * 8192 = 4.6737T vs the 4.673780159710T target). The seed
    # is intact: 3072 shards (256 nodes x 12 ranks), .metadata 108,980,217 B,
    # 24G, fp32. Its 16 zero-byte tail-rank shards are normal -- step-92800 has
    # 16 too, and the known-good 512N seed has 39 of 6144 in like proportion.
    #
    # AN EARLIER ATTEMPT AT THIS SEAT NaN'd, AND IT WAS NOT THE DATA OR THE LR.
    # Job 8663177 (2026-07-18, dropped from the umbrella) launched this same
    # seed under --config=agpt_2b_real -- cos_sin against COMPLEX-trained
    # weights. It resumed at loss 7.23840 / grad_norm 46.4271 where the
    # correct-flavor 512N twin resumed at 2.60254 / 0.2505. The load itself
    # succeeded, which is the documented flavor-mismatch signature. It spent
    # 3800 steps re-learning the scrambled Q/K pairing back to 2.63 and NaN'd
    # at 3801 on a model driven through a large recovery excursion. The
    # umbrella comment blaming the 50/50 mix or the 2e-6 LR was wrong on both:
    # the 512N seat runs PURE dolmino at a 10x HIGHER LR and is clean past
    # step 7700. Field 12 = complex closes that failure mode.
    #
    # rope=complex VERIFIED, not assumed: rope_flavor_for_step.py --chain
    # 2b_v2_256 --step 92859 -> `2b`, and all 22 runs of that chain (2026-05-01
    # through 2026-06-28) report `2b`. It never crossed the cos_sin switch.
    # Contrast 2b_v2_512 @46429 -> `2b_real`, which is why t0 carries _real.
    # (Unlike t3 above, the seed's flavor IS the right resolution here: this
    # dir is empty, so the first launch loads the seed.)
    #
    # Token budget is DELIBERATELY the same 2390375382006 as t0, not half.
    # Field 11 is tokens and steps = tok/(gbs*seq_len), so the same value
    # yields 47,492 steps at gbs 6144 vs 23,746 at 12288 -- identical token
    # exposure (291,790,848 samples both ways), which is what makes the 256N
    # and 512N arms comparable.
    #
    # Port 30000: 29700 (used by the dead commented row above) now belongs to
    # t3. Do not revive the old row verbatim -- it would collide.
    "2b|256|30000|$RUNS/agpt-2b-v2/torchtitan-ezpz|checkpoints/agpt-2b-stage2-dolmino-n256-gbs6144|dolmino-mix-1124|2.17e-5|$RUNS/agpt-2b-v2/torchtitan-ezpz/outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144/step-92859|0.0|1.0|2390375382006|complex"
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

# ---- Optional: run a SUBSET of the seats, optionally shrunk -------------------
#
# MULTI_ONLY="4"    -> run only trainer index 4
# MULTI_ONLY="1,4"  -> run trainers 1 and 4
# MULTI_NNODES_OVERRIDE=4 -> force every selected seat to 4 nodes
#
# This exists so a single seat can be smoke-tested THROUGH THIS LAUNCHER rather
# than by hand-reconstructing its `ezpz launch` invocation. Three attempts at
# the latter (2026-08-20) burned three allocations on: a missing venv
# activation, `--optimizer.name` when the build wants `--optimizer=`, and
# finally "Optimizer SophiaG not added" because `ezpz launch` needs
# --nproc/--nproc_per_node/--hostfile and a `--` separator and must run from
# the NODE-LOCAL venv. Reconstructing the shape is the bug; reuse the launcher.
#
# Unlike MULTI_PROFILE=tiny this keeps each seat's REAL ckpt dir, config,
# dataset and RoPE flavor -- tiny rewrites those to throwaway values, which is
# right for a plumbing smoke and wrong for "does THIS seed actually resume".
# The filter runs BEFORE the node-budget loop so every downstream calculation
# (need, slicing, offsets) sees only the selected seats.
if [[ -n "${MULTI_ONLY:-}" ]]; then
    _sel=()
    IFS=',' read -ra _want <<< "$MULTI_ONLY"
    for _i in "${_want[@]}"; do
        _i="${_i// /}"
        [[ "$_i" =~ ^[0-9]+$ ]] || die "MULTI_ONLY: '$_i' is not an index"
        (( _i < ${#TRAINERS[@]} )) || die "MULTI_ONLY: index $_i >= ${#TRAINERS[@]} trainers"
        _sel+=( "${TRAINERS[$_i]}" )
        log "MULTI_ONLY: selected trainer $_i"
    done
    TRAINERS=( "${_sel[@]}" )
fi
if [[ -n "${MULTI_NNODES_OVERRIDE:-}" ]]; then
    _shrunk=()
    for _row in "${TRAINERS[@]}"; do
        IFS='|' read -r _m _n _rest <<< "$_row"
        # rebuild with field 2 replaced, preserving all other fields verbatim
        _shrunk+=( "$(awk -v n="$MULTI_NNODES_OVERRIDE" 'BEGIN{FS=OFS="|"}{$2=n;print}' <<< "$_row")" )
    done
    TRAINERS=( "${_shrunk[@]}" )
    log "MULTI_NNODES_OVERRIDE: every selected seat forced to $MULTI_NNODES_OVERRIDE nodes"
fi

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
declare -a T_DFL T_LR T_INITLOAD T_DECAY T_MINLR T_TOKENS T_ROPE
offset=1
for idx in "${!TRAINERS[@]}"; do
    IFS='|' read -r model nnodes port workdir ckpt o_dfl o_lr o_init o_decay o_minlr o_tokens o_rope <<< "${TRAINERS[$idx]}"

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
    # 12th field: RoPE config suffix. REQUIRED -- see the guard in launch_one.
    T_ROPE[$idx]="$o_rope"
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
    elif [[ -n "${MULTI_STEPS_OVERRIDE:-}" ]]; then
        # A prod seat's step count comes from train_tokens (~6M steps). A smoke
        # that only needs to answer "does this seed resume to a sane loss" must
        # be able to stop on its own rather than at walltime.
        training_steps="$MULTI_STEPS_OVERRIDE"
    else
        training_steps=$(( tok / (gbs * SEQ_LEN) ))
    fi

    local dfl="torchtitan/experiments/ezpz/data-lists/aurora/${dfl_name}.txt"
    local data_cache="${ckpt}/.cache/${dfl_name}/index-cache"

    # Optional fork (initial-load) + constant/decayed-LR schedule overrides.
    local xtra=()
    [[ -n "${T_INITLOAD[$idx]}" ]] && xtra+=(--checkpoint.initial-load-path="${T_INITLOAD[$idx]}")
    # An OMITTED decay_ratio is a bug, not a default. Leaving the field empty
    # skips the flag entirely, and the run then inherits torchtitan's own
    # decay_ratio=0.8 -- the OPPOSITE of the constant-LR schedule every
    # production chain here wants. That silently put both 20B chains on
    # warmup-stable-decay; 20b_v2_512 ran ~1,200 steps of unintended decay
    # before it was caught, and needed a fork from step-9000 to undo.
    #
    # So refuse to launch a seat whose decay_ratio was not stated. Being
    # explicit costs one field; being wrong costs a fork.
    # RoPE FLAVOR IS PER-SEAT AND REQUIRED. There is no safe global default.
    #
    # CONFIG_SUFFIX used to be one global applied to all five seats. But each
    # chain crossed the 2026-06-25 cos_sin switch at a DIFFERENT step, and
    # 2b_v2_256 never crossed it at all -- so a single value is wrong for at
    # least one seat by construction. Audited 2026-08-20 via
    # scripts/eval/rope_flavor_for_step.py at each seat's actual resume/seed
    # step:
    #
    #   t0 stage2_dolmino  <- 2b_v2_512  @46429  trained _real   OK
    #   t1 20b-512 constlr <- 20b_v2_512 @9000   trained _real   OK
    #   t2 20b-256         <- 20b_v2_256 @10369  trained _real   OK
    #   t3 2b-512 constlr  <- 2b_v2_512  @21307  trained COMPLEX  was WRONG
    #   t4 2b-256 constlr  <- 2b_v2_256  @9500   trained COMPLEX  was WRONG
    #
    # Complex and cos_sin rotate DIFFERENT Q/K channel pairings, so loading
    # complex-trained weights under a cos_sin config silently corrupts the
    # model: it loads fine and only shows up as bad loss. t3 is a 512N
    # PRODUCTION seat resuming in place, so it would have written corrupted
    # checkpoints into a live chain the moment its std::bad_alloc cleared.
    #
    # Empty string is a MEANINGFUL value here (complex), so the guard tests for
    # the literal sentinel "-" rather than emptiness -- the same trap that let
    # an omitted decay_ratio fall through to torchtitan's 0.8 default.
    if [[ -z "${T_ROPE[$idx]+set}" ]]; then
        echo "FATAL: trainer $idx ($model|$nnodes) has no rope field (12th)." >&2
        echo "  Resolve it per seat:" >&2
        echo "    rope_flavor_for_step.py --chain <parent> --step <resume/seed>" >&2
        echo "  Then set field 12 to '_real' (cos_sin) or 'complex' (empty suffix)." >&2
        exit 95
    fi
    case "${T_ROPE[$idx]}" in
        _real)   rope_suffix="_real" ;;
        complex) rope_suffix="" ;;
        *) echo "FATAL: trainer $idx rope field is '${T_ROPE[$idx]}'; want _real|complex" >&2
           exit 95 ;;
    esac

    if [[ -z "${T_DECAY[$idx]}" ]]; then
        echo "FATAL: trainer $idx ($model|$nnodes) has no decay_ratio field." >&2
        echo "  An empty field does NOT mean 'constant LR' -- it means the flag" >&2
        echo "  is never passed and torchtitan defaults to 0.8 (warmup-stable-" >&2
        echo "  decay). State it: append |<dfl>|<lr>|<init>|0.0|1.0|<tokens>" >&2
        echo "  to the row (0.0/1.0 == constant LR)." >&2
        exit 96
    fi
    xtra+=(--lr-scheduler.decay-ratio="${T_DECAY[$idx]}"
           --lr-scheduler.min-lr-factor="${T_MINLR[$idx]:-1.0}"
           --lr-scheduler.warmup-steps=20)

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
            --config="agpt_${model}${rope_suffix}" \
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
