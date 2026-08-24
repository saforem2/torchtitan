#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N fp32-norms-throughput
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -l select=2
#PBS -q debug-scaling
#PBS -j oe

# Throughput/memory side of the master-weight-dtype ablation.
#
# The correctness question (do norms train?) is answered by the single-rank
# job (submit_fp32_norms_ablation.sh). This job measures the COST of arm B's
# required regrouping at real FSDP degree: promoting norms to fp32 forces each
# norm into its OWN fully_shard group (FSDP2 asserts a uniform orig_dtype per
# group), so the norms no longer ride along in the block's flat parameter and
# get their own all-gather/reshard traffic.
#
# Runs the standard agpt debugmodel training loop (not the ablation driver)
# across all ranks at each dtype setting, and reports steps/sec + memory from
# the normal metrics line.
#
# See docs/records/proposals/30b-exp/exp02-fp32-norms-ablation.md.

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export ftp_proxy="${ftp_proxy:-http://proxy.alcf.anl.gov:3128}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,*.alcf.anl.gov,*.aurora.alcf.anl.gov}"

cd "${PBS_O_WORKDIR:-$(pwd)}"

# Activate the REPO venv (torch 2.13 + xpu) FIRST and keep it active for the
# whole job. ezpz_setup_job / `ezpz` otherwise resolve against ~/.venv, whose
# torch is built for a different oneAPI (libsycl.so.8: undefined symbol
# urEnqueueCooperativeKernelLaunchE) -- that killed job 8757164.
source .venv/bin/activate

source <(curl -fsSL --max-time 30 https://bit.ly/ezpz-utils)
ezpz_setup_job

# Broadcast the venv to node-local /tmp, then run from there.
if [[ -f .venv.tar.gz ]]; then
    ezpz yeet-env --src .venv.tar.gz
else
    ezpz yeet-env
fi
deactivate
source /tmp/.venv/bin/activate

STEPS="${STEPS:-100}"
OUTDIR="${OUTDIR:-outputs/fp32-norms-ablation/${PBS_JOBID%%.*}-throughput}"
mkdir -p "${OUTDIR}"

run_arm() {
    local arm="$1" dtype="$2" fp32norms="$3"
    log_message INFO "--- throughput arm ${arm} (dtype=${dtype} EZPZ_FP32_NORMS=${fp32norms}) ---"
    EZPZ_FP32_NORMS="${fp32norms}" ezpz launch python3 -m torchtitan.experiments.ezpz.train \
        --module=ezpz.agpt \
        --config=agpt_debugmodel \
        --compile.no-enable \
        --checkpoint.no-enable \
        --validator.no-enable \
        --metrics.no-enable-wandb \
        --metrics.log-freq=10 \
        --training.dtype="${dtype}" \
        --training.steps="${STEPS}" \
        --training.seq-len=2048 \
        --training.local-batch-size=2 \
        --debug.seed=42 \
        2>&1 | tee "${OUTDIR}/arm${arm}.log"
}

run_arm A bfloat16 0
run_arm B bfloat16 1
run_arm C float32  0

log_message INFO "=== steady-state throughput (mean of last 5 logged steps) ==="
for arm in A B C; do
    python3 - "$arm" "${OUTDIR}/arm${arm}.log" <<'PY'
import re, sys
arm, path = sys.argv[1], sys.argv[2]
txt = open(path, errors="ignore").read()
clean = re.sub(r"\x1b\[[0-9;]*m", "", txt)
tps = [float(x) for x in re.findall(r"tps:\s*([0-9.]+)", clean)]
mem = [float(x) for x in re.findall(r"memory:\s*([0-9.]+)GiB", clean)]
mfu = [float(x) for x in re.findall(r"mfu:\s*([0-9.]+)%", clean)]
def tail(v, n=5):
    v = v[-n:]
    return sum(v)/len(v) if v else float("nan")
print(f"arm {arm}: tps={tail(tps):.1f}  mfu={tail(mfu):.2f}%  mem={max(mem) if mem else float('nan'):.2f}GiB")
PY
done | tee "${OUTDIR}/throughput.txt"
