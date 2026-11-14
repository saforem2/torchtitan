#!/bin/bash --login
#PBS -A datascience
#PBS -N perf-probe-sirius
#PBS -l select=8
#PBS -l walltime=02:00:00
#PBS -l filesystems=home:tegu
#PBS -q workq
#PBS -j oe
# Perf/MFU probes on Sirius — short 100-step timing runs, NOT LR finder.
# Model: agpt_5b_olmo2tok_smoke at GBS=128 seq_len 4096, 8N 32 GPUs.
# Each probe varies ONE knob via a distinct ezpz launch block logged via tee,
# with a distinct --checkpoint.folder, and reports tps/mfu/tflops from the
# observability/metrics lines in the log.
#
# Dispatch: PERF_PROBE=baseline|compile|selective|none|lbs2|all
#   qsub -v PERF_PROBE=compile torchtitan/experiments/ezpz/scripts/submit_perf_probe_sirius.sh
#   qsub -v PERF_PROBE=all     torchtitan/experiments/ezpz/scripts/submit_perf_probe_sirius.sh
# Or inside an allocation: PERF_PROBE=all bash torchtitan/experiments/ezpz/scripts/submit_perf_probe_sirius.sh
set -o pipefail
cd "${PBS_O_WORKDIR:-$(pwd)}" || exit 1

# Shared-NFS mode (no yeet): mirror submit_lr_finder_sirius.sh setup exactly.
# shellcheck source=/dev/null
if [[ -f "${PBS_O_WORKDIR:-.}/.ezpz-utils-cache/ezpz-utils.sh" ]]; then
    source "${PBS_O_WORKDIR:-.}/.ezpz-utils-cache/ezpz-utils.sh"
else
    source <(curl -fsSL --max-time 30 https://bit.ly/ezpz-utils)
fi
ezpz_load_modules
ezpz_setup_job
export MPICH_GPU_SUPPORT_ENABLED=0
export EZPZ_MPI_LABEL=1
export LD_LIBRARY_PATH="${HOME}/.local/mpi-compat:${LD_LIBRARY_PATH}"

VENV_DIR="${HOME}/torchtitan/.venv"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
log_message INFO "Sirius perf probe (shared-NFS): ${VENV_DIR}/bin/python — PERF_PROBE=${PERF_PROBE:-all}"

PERF_PROBE="${PERF_PROBE:-all}"
MODEL_CFG="agpt_5b_olmo2tok_smoke"
GBS=128
SEQ_LEN=4096
STEPS=100
T_TOKENS_PER_STEP=$(( GBS * SEQ_LEN ))
# Default AC for this model: full (agpt(..., activation_checkpoint_mode="full")).
# AC is varied per-probe via the positional activation-checkpoint:{none,full,selective}
# token that ConfigManager._migrate_legacy_args maps onto activation_remat.
# See torchtitan/experiments/ezpz/config_manager.py:_LEGACY_ACTIVATION_CHECKPOINT.
# AC selective's closest successor is DefaultActivationRematConfig (torch_remat).
# Compile uses --compile.enable / --compile.no-enable (torchtitan/experiments/ezpz/compile.py).

mkdir -p outputs

# 1. baseline: compile off + AC full + LBS1 — reference, expect ~tps 419 / mfu 4.3%
probe_baseline() {
  local LBS=1
  echo "=========================================================="
  echo " probe: baseline  ckpt: checkpoints/perf-baseline  (compile off, AC full, LBS1)"
  echo "=========================================================="
  ezpz launch \
    --nproc $(( 8 * 4 )) \
    --nproc_per_node 4 \
    --auto-retry \
    --spare-nodes auto \
    --timeout 1800 \
    --no-transfer \
    -- \
    "${VENV_DIR}/bin/python" -m torchtitan.experiments.ezpz.train \
    --module ezpz.agpt --config "${MODEL_CFG}" \
    --training.steps "${STEPS}" \
    --training.num-tokens-per-microbatch-per-dp-rank "$(( LBS * SEQ_LEN ))" \
    --training.num-tokens-per-train-step "${T_TOKENS_PER_STEP}" \
    --training.max-context-length "${SEQ_LEN}" \
    --metrics.log_freq 1 \
    --checkpoint.enable --checkpoint.folder "checkpoints/perf-baseline" \
    --checkpoint.interval 50 --checkpoint.no-last-save-model-only \
    --checkpoint.async-mode disabled \
    --compile.no-enable \
    activation-checkpoint:full \
    2>&1 | tee "outputs/perf_baseline_${PBS_JOBID%%.*}.log"
  echo "--- [baseline] done (exit ${PIPESTATUS[0]}) ---"
}

# 2. compile on: same AC/LBS as baseline, torch.compile enabled via --compile.enable
probe_compile() {
  local LBS=1
  echo "=========================================================="
  echo " probe: compile  ckpt: checkpoints/perf-compile  (compile on, AC full, LBS1)"
  echo "=========================================================="
  ezpz launch \
    --nproc $(( 8 * 4 )) \
    --nproc_per_node 4 \
    --auto-retry \
    --spare-nodes auto \
    --timeout 1800 \
    --no-transfer \
    -- \
    "${VENV_DIR}/bin/python" -m torchtitan.experiments.ezpz.train \
    --module ezpz.agpt --config "${MODEL_CFG}" \
    --training.steps "${STEPS}" \
    --training.num-tokens-per-microbatch-per-dp-rank "$(( LBS * SEQ_LEN ))" \
    --training.num-tokens-per-train-step "${T_TOKENS_PER_STEP}" \
    --training.max-context-length "${SEQ_LEN}" \
    --metrics.log_freq 1 \
    --checkpoint.enable --checkpoint.folder "checkpoints/perf-compile" \
    --checkpoint.interval 50 --checkpoint.no-last-save-model-only \
    --checkpoint.async-mode disabled \
    --compile.enable \
    activation-checkpoint:full \
    2>&1 | tee "outputs/perf_compile_${PBS_JOBID%%.*}.log"
  echo "--- [compile] done (exit ${PIPESTATUS[0]}) ---"
}

# 3. AC selective: activation_checkpoint_mode="selective" -> DefaultActivationRematConfig
#    via the activation-checkpoint:selective token. This is the torch_remat
#    successor of selective-op AC (see config_manager _LEGACY_ACTIVATION_CHECKPOINT).
probe_selective() {
  local LBS=1
  echo "=========================================================="
  echo " probe: selective  ckpt: checkpoints/perf-selective  (compile off, AC selective, LBS1)"
  echo "=========================================================="
  ezpz launch \
    --nproc $(( 8 * 4 )) \
    --nproc_per_node 4 \
    --auto-retry \
    --spare-nodes auto \
    --timeout 1800 \
    --no-transfer \
    -- \
    "${VENV_DIR}/bin/python" -m torchtitan.experiments.ezpz.train \
    --module ezpz.agpt --config "${MODEL_CFG}" \
    --training.steps "${STEPS}" \
    --training.num-tokens-per-microbatch-per-dp-rank "$(( LBS * SEQ_LEN ))" \
    --training.num-tokens-per-train-step "${T_TOKENS_PER_STEP}" \
    --training.max-context-length "${SEQ_LEN}" \
    --metrics.log_freq 1 \
    --checkpoint.enable --checkpoint.folder "checkpoints/perf-selective" \
    --checkpoint.interval 50 --checkpoint.no-last-save-model-only \
    --checkpoint.async-mode disabled \
    --compile.no-enable \
    activation-checkpoint:selective \
    2>&1 | tee "outputs/perf_selective_${PBS_JOBID%%.*}.log"
  echo "--- [selective] done (exit ${PIPESTATUS[0]}) ---"
}

# 4. AC none: activation_checkpoint_mode="none" -> activation_remat=None
probe_none() {
  local LBS=1
  echo "=========================================================="
  echo " probe: none  ckpt: checkpoints/perf-none  (compile off, AC none, LBS1)"
  echo "=========================================================="
  ezpz launch \
    --nproc $(( 8 * 4 )) \
    --nproc_per_node 4 \
    --auto-retry \
    --spare-nodes auto \
    --timeout 1800 \
    --no-transfer \
    -- \
    "${VENV_DIR}/bin/python" -m torchtitan.experiments.ezpz.train \
    --module ezpz.agpt --config "${MODEL_CFG}" \
    --training.steps "${STEPS}" \
    --training.num-tokens-per-microbatch-per-dp-rank "$(( LBS * SEQ_LEN ))" \
    --training.num-tokens-per-train-step "${T_TOKENS_PER_STEP}" \
    --training.max-context-length "${SEQ_LEN}" \
    --metrics.log_freq 1 \
    --checkpoint.enable --checkpoint.folder "checkpoints/perf-none" \
    --checkpoint.interval 50 --checkpoint.no-last-save-model-only \
    --checkpoint.async-mode disabled \
    --compile.no-enable \
    activation-checkpoint:none \
    2>&1 | tee "outputs/perf_none_${PBS_JOBID%%.*}.log"
  echo "--- [none] done (exit ${PIPESTATUS[0]}) ---"
}

# 5. LBS2: microbatch 2*4096 = 8192 tokens, keep GBS 128 (524288 tokens) so GAS halves.
#    GAS = GBS/LBS = 64 (vs 128 at LBS1), otherwise identical to baseline.
probe_lbs2() {
  local LBS=2
  echo "=========================================================="
  echo " probe: lbs2  ckpt: checkpoints/perf-lbs2  (compile off, AC full, LBS2 -> GAS 64)"
  echo "=========================================================="
  ezpz launch \
    --nproc $(( 8 * 4 )) \
    --nproc_per_node 4 \
    --auto-retry \
    --spare-nodes auto \
    --timeout 1800 \
    --no-transfer \
    -- \
    "${VENV_DIR}/bin/python" -m torchtitan.experiments.ezpz.train \
    --module ezpz.agpt --config "${MODEL_CFG}" \
    --training.steps "${STEPS}" \
    --training.num-tokens-per-microbatch-per-dp-rank "$(( LBS * SEQ_LEN ))" \
    --training.num-tokens-per-train-step "${T_TOKENS_PER_STEP}" \
    --training.max-context-length "${SEQ_LEN}" \
    --metrics.log_freq 1 \
    --checkpoint.enable --checkpoint.folder "checkpoints/perf-lbs2" \
    --checkpoint.interval 50 --checkpoint.no-last-save-model-only \
    --checkpoint.async-mode disabled \
    --compile.no-enable \
    activation-checkpoint:full \
    2>&1 | tee "outputs/perf_lbs2_${PBS_JOBID%%.*}.log"
  echo "--- [lbs2] done (exit ${PIPESTATUS[0]}) ---"
}

case "${PERF_PROBE}" in
  baseline)  probe_baseline ;;
  compile)   probe_compile ;;
  selective) probe_selective ;;
  none)      probe_none ;;
  lbs2)      probe_lbs2 ;;
  all)
    probe_baseline
    probe_compile
    probe_selective
    probe_none
    probe_lbs2
    echo "=== all perf probes done — tps/mfu/tflops summary (from metrics lines) ==="
    # MetricsProcessor logs: "tps: N  tflops: X  mfu: Y%"; mirror the LR finder's summary.
    grep -h "tps:" outputs/perf_*_"${PBS_JOBID%%.*}".log 2>/dev/null | tail -30 || true
    ;;
  *)
    echo "Unknown PERF_PROBE=${PERF_PROBE}; expected one of baseline|compile|selective|none|lbs2|all" >&2
    exit 2
    ;;
esac
