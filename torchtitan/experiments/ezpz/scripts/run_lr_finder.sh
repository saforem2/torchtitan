#!/bin/bash --login
# LR Finder sweep for agpt models across optimizers.
# Runs LR finder for each (model, optimizer) combination.
#
# Usage (inside a PBS allocation):
#   bash torchtitan/experiments/ezpz/scripts/run_lr_finder.sh
#
# Environment variables:
#   LRF_MODELS      — space-separated model flavors (default: "2b 20b")
#   LRF_OPTIMIZERS  — space-separated optimizers (default: "adamw muon sophiag")
#   LRF_STEPS       — total training steps for fraction calc (default: 1000)
#   LRF_FRACTION    — fraction of steps to sweep (default: 0.1)
#   LRF_INIT_LR     — starting LR (default: 1e-6)
#   LRF_MAX_LR      — max LR (default: 1.0)
#   LRF_TIMEOUT     — per-run timeout in seconds (default: 1800)
#   LRF_DFL_NAME    — data list basename (default "books"). MUST match the
#                     config's tokenizer: books.txt is Llama2-tokenized on
#                     Aurora, so gemma configs need olmo-mix-1124.
#   LRF_SEQ_LEN     — sequence length (default 8192). GBS counts SEQUENCES, so
#                     this scales tokens/step and wall clock. Use the length the
#                     target model was measured at (30B: 4096).
#   LRF_CONFIG      — exact config flavor, overriding "agpt_<model>". Needed
#                     whenever the flavor is not just the size (e.g.
#                     agpt_30b_olmo2tok).
#   LRF_AC          — activation-checkpoint mode (e.g. "full"). Appends the
#                     tyro positional subcommand. Required at 30B/80B.
#   LRF_NO_COMPILE  — set to 1 to pass --compile.no-enable.
#   LRF_LBS         — local batch size per device (default: 1). At production
#                     N + LBS to measure the optimal LR at the actual GBS the
#                     production chain will run at (e.g. LBS=2 for 2b 256N
#                     gives GBS=6144 matching submit_agpt_2b_aurora_venv.sh).

set -o pipefail

# ---------------------------------------------------------------------------
# Environment setup — production torch 2.13 .venv + ezpz yeet-env (matches
# scripts/submit_agpt_2b_aurora_venv.sh so the LR-finder runs in the SAME
# stack as the production chain it's calibrating).
#
# When this script is invoked via `qsub -- /bin/bash -c 'bash <this>'`,
# the inner bash is NOT a login shell, so module/MODULEPATH/lmod aren't
# initialized. Source Aurora's Cray PE init (which defines `module` AND
# populates MODULEPATH from /etc/cray-pe.d/cray-pe-configuration.sh).
# `/etc/bash.bashrc.local` is what `#!/bin/bash --login` gets via
# /etc/bash.bashrc -> /etc/bash.bashrc.local.
# ---------------------------------------------------------------------------
if ! command -v module >/dev/null 2>&1 || [[ -z "${MODULEPATH:-}" ]]; then
    if [[ -r /etc/bash.bashrc.local ]]; then
        source /etc/bash.bashrc.local
    elif [[ -r /usr/share/lmod/lmod/init/bash ]]; then
        source /usr/share/lmod/lmod/init/bash
    fi
fi
module load oneapi/release/2025.3.1 hdf5 pti-gpu
# /opt/pbs/bin must be on PATH so `sh.qstat` works inside `ezpz launch`
# (ezpz.pbs.get_pbs_jobid_of_active_job calls `from sh import qstat`).
# bash --login on login node has it via /etc/profile; this re-export
# ensures it propagates through mpiexec --envall to compute nodes too.
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

# Must cd into the repo BEFORE ezpz_setup_job (and before sourcing .venv).
# Reasons:
#   1. PBS spawns scripts in $HOME, so `source .venv/bin/activate` (relative)
#      would pick up $HOME/.venv if present, with incompatible torch.
#   2. ezpz_setup_job's `WORKING_DIR=$(pwd)` runs at script start. Calling
#      ezpz_setup_job before this cd captures WORKING_DIR=$HOME, then its
#      "WORKING_DIR doesn't match PBS_O_WORKDIR" branch OVERWRITES
#      PBS_O_WORKDIR with $HOME — defeating any subsequent cd.
cd "${PBS_O_WORKDIR:-$(pwd)}"

set +u
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
set -u

source .venv/bin/activate
if [[ -f .venv.tar.gz ]]; then
    log_message INFO "lr-finder: yeet-env via tarball (.venv.tar.gz)"
    ezpz yeet-env --src .venv.tar.gz
else
    log_message INFO "lr-finder: yeet-env via per-file rsync (.venv.tar.gz not present)"
    ezpz yeet-env
fi
deactivate
source /tmp/.venv/bin/activate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LRF_MODELS="${LRF_MODELS:-2b 20b}"
LRF_OPTIMIZERS="${LRF_OPTIMIZERS:-adamw muon sophiag}"
LRF_STEPS="${LRF_STEPS:-1000}"
LRF_FRACTION="${LRF_FRACTION:-0.1}"
LRF_INIT_LR="${LRF_INIT_LR:-1e-6}"
LRF_MAX_LR="${LRF_MAX_LR:-1.0}"
LRF_TIMEOUT="${LRF_TIMEOUT:-1800}"
LRF_LBS="${LRF_LBS:-1}"
# Sequence length. Was hardcoded to 8192 in the launch below, which is wrong
# for any model whose measurements are at another length: every 30B datapoint
# (exp05 tuning, exp06 scaling, exp07 tokenizer/LBS, exp08's convergence run)
# is at seq=4096, and GBS counts SEQUENCES -- so 8192 would double tokens/step
# and the wall clock for no calibration benefit.
LRF_SEQ_LEN="${LRF_SEQ_LEN:-8192}"
# Config flavor override. The default composes "agpt_${model}", which is right
# for 2b/20b/80b but picks the WRONG 30B: agpt_30b is the gemma-256k-vocab
# 28.1B variant, while every 30B measurement and the converged chain use
# agpt_30b_olmo2tok (26.2B, olmo2 vocab). Set this to name the flavor exactly.
LRF_CONFIG="${LRF_CONFIG:-}"
# Activation checkpointing. `full` appends the tyro positional subcommand
# activation-checkpoint:full, which MUST be the last argv token. At 30B this
# is load-bearing: exp05 found `none` OOMs and `selective` errors at this size.
LRF_AC="${LRF_AC:-}"
# Set to 1 to pass --compile.no-enable. The 80B path forces this (its compile
# is broken); 30B trains compiled and should stay compiled.
LRF_NO_COMPILE="${LRF_NO_COMPILE:-}"
# Target global batch for the sweep. The optimal LR is batch-size
# dependent, so to calibrate a production run you must sweep at THAT run's
# GBS. Empty (default) = whatever world_size*LBS/TP gives. Set e.g.
# LRF_GBS=6144 to add --training.global-batch-size and let the trainer
# derive GAS to hit it (GBS = dp_degree * LBS * GAS). For 80B at TP=4 the
# small-batch default (192) is ~32x below the 6144 production target and
# does NOT generalize -- always set LRF_GBS for an 80B production calibration.
LRF_GBS="${LRF_GBS:-}"
# Shared blendcorpus index-cache dir (see cache_args below). Prewarm it at
# the SAME GBS + LRF_STEPS so the hash matches and every optimizer loads
# warm. Empty = default .cache/blendcorpus (cold-build race on optimizer 1).
LRF_DATA_CACHE_PATH="${LRF_DATA_CACHE_PATH:-}"

read -ra MODELS <<< "${LRF_MODELS}"
read -ra OPTIMIZERS <<< "${LRF_OPTIMIZERS}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
NUM_NODES="${NHOSTS:-${SLURM_NNODES:-1}}"
# Include the PBS jobid so concurrent finder jobs (e.g. a GBS-trend sweep
# submitted in the same second) get DISTINCT per-run dirs. date alone is
# second-resolution and collides on simultaneous submits.
_JOBTAG="${PBS_JOBID%%.*}"
OUTDIR="outputs/lr_finder/${TIMESTAMP}${_JOBTAG:+_${_JOBTAG}}"
mkdir -p "${OUTDIR}"
# The trainer writes the CSV/plot/npz under <dump_folder>/lr_finder/
# ezpz.agpt/<flavor>/<optimizer>/ -- a path keyed by model+optimizer, NOT
# by GBS. Concurrent same-(model,optimizer) jobs at different GBS therefore
# clobber each other's CSV. Set LRF_DUMP_FOLDER per job (default ./outputs)
# so a trend sweep isolates each GBS's outputs.
LRF_DUMP_FOLDER="${LRF_DUMP_FOLDER:-outputs}"

# Data list. The default is books.txt for historical reasons, but on Aurora
# that points at dolma/data_v1.7_Llama2Tokenizer -- LLAMA-2 token ids. Sweeping
# a GEMMA-vocab config (every agpt_* except the *_llama3tok / *_olmo2tok
# variants) against it calibrates the LR on the wrong vocabulary, and the
# resulting number does not transfer -- which is the exact failure this whole
# production-batch exercise exists to avoid. It fails SILENTLY: ids below the
# embedding size index fine and the loss curve looks plausible.
#
# Set LRF_DFL_NAME to match the config's tokenizer:
#   gemma configs (agpt_2b/20b/30b/80b)  -> olmo-mix-1124 (data_fused_gemma_eod)
#   *_llama3tok / *_olmo2tok             -> a list tokenized to match
LRF_DFL_NAME="${LRF_DFL_NAME:-books}"
DATASET_PATH="torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/${LRF_DFL_NAME}.txt"
if [[ ! -f "$DATASET_PATH" ]]; then
    echo "lr-finder FATAL: data list not found: $DATASET_PATH" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Kill stale processes
# ---------------------------------------------------------------------------
echo "--- Cleaning up stale processes and cache ---"
pkill -u "${USER}" -f "torchtitan.experiments.ezpz.train" 2>/dev/null && sleep 2 || true
rm -rf .cache/blendcorpus/*.npy 2>/dev/null || true
echo ""

# ---------------------------------------------------------------------------
# Job metadata
# ---------------------------------------------------------------------------
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
RUN_DATE="$(date -Iseconds)"
MACHINE_NAME="$(ezpz_get_machine_name 2>/dev/null || hostname -s)"
JOB_ID="${PBS_JOBID:-${SLURM_JOB_ID:-${COBALT_JOBID:-local}}}"
DEVICES_PER_NODE=$(( NGPUS / NUM_NODES ))
FINDER_STEPS=$(python3 -c "print(max(1, int(${LRF_STEPS} * ${LRF_FRACTION})))")

echo "============================================================"
echo " LR Finder Sweep — ${TIMESTAMP}"
echo " devices=${NGPUS}  nodes=${NUM_NODES}"
echo " models: ${LRF_MODELS}"
echo " optimizers: ${LRF_OPTIMIZERS}"
echo " LR range: ${LRF_INIT_LR} → ${LRF_MAX_LR}"
echo " finder steps: ${FINDER_STEPS} (${LRF_FRACTION} × ${LRF_STEPS})"
echo "============================================================"
echo ""

declare -a R_MODEL R_OPT R_STATUS R_WALL R_MIN_LOSS R_LR_AT_MIN
RUN_IDX=0

for model in "${MODELS[@]}"; do
    config="${LRF_CONFIG:-agpt_${model}}"
    # Per-model parallelism + stability flags.
    #
    # For 80B the validated stable corner is TP=4 / LBS=1 / compile OFF /
    # AC=full / pure FSDP (see docs/production/agpt/80b/README.md). TP=2
    # here would put dp_degree at 62*12/2 = 372 -- past the ~186 NaN
    # ceiling -- so every 80B sweep would NaN from the dp-degree trigger
    # rather than from the LR being swept, making the curve meaningless.
    # TP=4 keeps dp_degree=186 (safe) so the sweep actually measures the
    # LR cliff. Override the TP via LRF_TP if probing a different corner.
    tp_args=()
    ac_subcommand=()
    # Generic AC / compile knobs, applied to ANY model. These run BEFORE the
    # 80B block so that block can still override them wholesale.
    if [[ -n "${LRF_AC}" ]]; then
        ac_subcommand=("activation-checkpoint:${LRF_AC}")
    fi
    if [[ -n "${LRF_NO_COMPILE}" ]]; then
        tp_args+=(--compile.no-enable)
    fi
    if [[ "${model}" == "80b" || "${model}" == "80B" ]]; then
        tp_args=(
            --parallelism.tensor_parallel_degree "${LRF_TP:-4}"
            --parallelism.expert_parallel_degree 1
            --parallelism.data_parallel_replicate_degree 1
            --parallelism.data_parallel_shard_degree -1
            --compile.no-enable
        )
        # activation-checkpoint is a positional tyro subcommand and must
        # be the LAST argv token (after every --flag and "$@").
        ac_subcommand=("activation-checkpoint:full")
    fi

    # Optional target global batch. The optimal LR is batch-size
    # dependent, so calibrating a production run requires sweeping at that
    # run's GBS (the trainer derives the needed GAS from
    # GBS = dp_degree * LBS * GAS). Empty = use world_size*LBS/TP.
    gbs_args=()
    if [[ -n "${LRF_GBS}" ]]; then
        gbs_args=(--training.global_batch_size "${LRF_GBS}")
    fi

    # Optional shared index-cache dir. The blendcorpus index cold-builds
    # on the FIRST optimizer of a sweep and (at TP>1) races the
    # build-then-load; the finder has no auto-retry, so a cold first
    # optimizer crashes (this is what killed adamw/muon in the GBS=192
    # run). Point all jobs at one LRF_DATA_CACHE_PATH that a small prewarm
    # built, so every optimizer loads warm. Empty = default .cache/blendcorpus.
    cache_args=()
    if [[ -n "${LRF_DATA_CACHE_PATH}" ]]; then
        cache_args=(--dataloader.data-cache-path "${LRF_DATA_CACHE_PATH}")
    fi

    for opt in "${OPTIMIZERS[@]}"; do
        label="${model}_${opt}"
        logfile="${OUTDIR}/${label}.log"

        echo "--- [${label}] LR finder (config=${config} optimizer=${opt}) ---"
        echo "    started @ $(date +%Y%m%d-%H%M%S)"
        echo "    logfile: ${logfile}"
        echo ""

        R_MODEL[$RUN_IDX]="${model}"
        R_OPT[$RUN_IDX]="${opt}"

        start_seconds=$SECONDS

        # Use `ezpz launch --auto-retry` (the same path the production
        # submit scripts use, e.g. submit_agpt_2b_autoretry.sh) rather than
        # a bare launch. This gives the finder the retry-on-crash loop for
        # free, which self-heals the blendcorpus cold-cache build-then-load
        # race: at TP>1 the first attempt cold-builds the index on rank 0
        # while other ranks race to np.load it mid-write (EOFError / "mmap
        # length is greater than file size" / DistNetworkError); auto-retry
        # re-runs, and attempt 2 loads the now-written index warm (exactly
        # how the GBS=5952 production run recovered). `--spare-nodes auto`
        # carves spares from any nodes beyond what the sweep uses; the venv
        # is already yeeted to the whole nodefile above so a swapped spare
        # is a filesystem no-op. (Prewarming single-rank still avoids the
        # race up front; this is the in-band safety net + bad-node failover.)
        # Proper fix = a rank-0-build barrier in blendcorpus, tracked
        # separately.
        lrf_mfr=()
        [[ -n "${LRF_MAX_FAILOVER_RETRIES:-}" ]] && \
            lrf_mfr=(--max-failover-retries "${LRF_MAX_FAILOVER_RETRIES}")
        timeout "${LRF_TIMEOUT}" \
            stdbuf -oL -eL \
            env NGPU="${NGPUS}" PYTHONUNBUFFERED=1 \
            ezpz launch \
            --nproc "${NGPUS}" \
            --nproc_per_node "${NGPU_PER_HOST:-12}" \
            --auto-retry \
            --spare-nodes auto \
            --timeout "${LRF_IDLE_TIMEOUT:-1800}" \
            "${lrf_mfr[@]}" \
            -- \
            python3 -m torchtitan.experiments.ezpz.train \
            --module ezpz.agpt \
            --config "${config}" \
            --job.dump-folder "${LRF_DUMP_FOLDER}" \
            --optimizer "${opt}" \
            --training.steps "${LRF_STEPS}" \
            --training.local_batch_size "${LRF_LBS}" \
            "${gbs_args[@]}" \
            --training.seq_len "${LRF_SEQ_LEN}" \
            --metrics.log_freq 1 \
            --checkpoint.no-enable \
            --dataloader.dataset blendcorpus \
            --dataloader.dataset_path "${DATASET_PATH}" \
            "${cache_args[@]}" \
            --lr_finder.enable \
            --lr_finder.init_lr "${LRF_INIT_LR}" \
            --lr_finder.max_lr "${LRF_MAX_LR}" \
            --lr_finder.fraction "${LRF_FRACTION}" \
            "${tp_args[@]}" \
            "$@" \
            "${ac_subcommand[@]}" \
            >"${logfile}" 2>&1 || true
        exit_code=$?

        # Kill any leftover processes
        pkill -u "${USER}" -f "torchtitan.experiments.ezpz.train" 2>/dev/null || true
        sleep 2

        elapsed=$(( SECONDS - start_seconds ))
        R_WALL[$RUN_IDX]="${elapsed}"

        # Determine status
        if ((exit_code == 124)); then
            R_STATUS[$RUN_IDX]="TIMEOUT"
        elif grep -q 'OUT_OF_RESOURCES\|out of memory\|OOM' "${logfile}"; then
            R_STATUS[$RUN_IDX]="OOM"
        elif grep -q 'LR Finder complete' "${logfile}"; then
            R_STATUS[$RUN_IDX]="OK"
        elif grep -q 'Traceback\|Error\|Exception' "${logfile}"; then
            R_STATUS[$RUN_IDX]="CRASH"
        else
            R_STATUS[$RUN_IDX]="UNKNOWN"
        fi

        # Parse min loss and LR from finder output
        min_line="$(grep 'Min loss @' "${logfile}" 2>/dev/null | tail -1 || true)"
        if [[ -n "${min_line}" ]]; then
            R_LR_AT_MIN[$RUN_IDX]="$(echo "${min_line}" | sed -n 's/.*lr=\([0-9.eE+-]*\).*/\1/p')"
            R_MIN_LOSS[$RUN_IDX]="$(echo "${min_line}" | sed -n 's/.*loss=\([0-9.]*\).*/\1/p')"
        else
            # Parse from CSV if available
            csv_dir="$(grep 'saved CSV to' "${logfile}" 2>/dev/null | sed -n 's/.*saved CSV to \(.*\)/\1/p' | head -1 || true)"
            if [[ -n "${csv_dir}" && -f "${csv_dir}" ]]; then
                # Find min loss row
                min_row="$(tail -n +2 "${csv_dir}" | sort -t, -k2 -n | head -1 || true)"
                R_LR_AT_MIN[$RUN_IDX]="$(echo "${min_row}" | cut -d, -f1)"
                R_MIN_LOSS[$RUN_IDX]="$(echo "${min_row}" | cut -d, -f2)"
            else
                R_LR_AT_MIN[$RUN_IDX]="N/A"
                R_MIN_LOSS[$RUN_IDX]="N/A"
            fi
        fi

        R_LR_AT_MIN[$RUN_IDX]="${R_LR_AT_MIN[$RUN_IDX]:-N/A}"
        R_MIN_LOSS[$RUN_IDX]="${R_MIN_LOSS[$RUN_IDX]:-N/A}"

        echo "    status=${R_STATUS[$RUN_IDX]}  wall=${elapsed}s  min_loss=${R_MIN_LOSS[$RUN_IDX]}  lr@min=${R_LR_AT_MIN[$RUN_IDX]}"
        echo ""

        RUN_IDX=$(( RUN_IDX + 1 ))
    done
done

NUM_RUNS="${RUN_IDX}"

# ---------------------------------------------------------------------------
# Generate report
# ---------------------------------------------------------------------------
REPORT="${OUTDIR}/report.md"

{
    echo "# LR Finder Report"
    echo ""

    _meta_keys=("Date" "Commit" "Machine" "Job ID" "Nodes" "Devices" "Finder Steps" "LR Range")
    _meta_vals=("${RUN_DATE}" "${GIT_COMMIT}" "${MACHINE_NAME}" "${JOB_ID}" "${NUM_NODES}" "${NGPUS}" "${FINDER_STEPS}" "${LRF_INIT_LR} → ${LRF_MAX_LR}")
    _vw=5
    for _v in "${_meta_vals[@]}"; do
        (( ${#_v} > _vw )) && _vw=${#_v}
    done

    printf "| %-12s | %-${_vw}s |\n" "Field" "Value"
    printf "|-%s-|-%s-|\n" "$(printf '%0.s-' $(seq 1 12))" "$(printf '%0.s-' $(seq 1 "${_vw}"))"
    for ((_j = 0; _j < ${#_meta_keys[@]}; _j++)); do
        printf "| %-12s | %-${_vw}s |\n" "${_meta_keys[$_j]}" "${_meta_vals[$_j]}"
    done
    echo ""

    echo "## Results"
    echo ""
    printf "| %-6s | %-10s | %10s | %12s | %10s | %-8s |\n" \
        "Model" "Optimizer" "Min Loss" "LR @ Min" "Wall (s)" "Status"
    _sep() { printf '%0.s-' $(seq 1 "$1"); }
    printf "|-%s-|-%s-|-%s-|-%s-|-%s-|-%s-|\n" \
        "$(_sep 6)" "$(_sep 10)" "$(_sep 10)" "$(_sep 12)" "$(_sep 10)" "$(_sep 8)"

    for ((i = 0; i < NUM_RUNS; i++)); do
        printf "| %-6s | %-10s | %10s | %12s | %10s | %-8s |\n" \
            "${R_MODEL[$i]}" \
            "${R_OPT[$i]}" \
            "${R_MIN_LOSS[$i]}" \
            "${R_LR_AT_MIN[$i]}" \
            "${R_WALL[$i]}" \
            "${R_STATUS[$i]}"
    done

    echo ""
    echo "Logs: \`${OUTDIR}/\`"
    echo ""
    echo "LR finder curves saved to \`outputs/lr_finder/ezpz/\`"
} >"${REPORT}"

echo "============================================================"
cat "${REPORT}"
echo "============================================================"
echo ""
echo "Report saved to: ${REPORT}"
echo "Logs saved to:   ${OUTDIR}/"
