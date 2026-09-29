#!/bin/bash --login
#PBS -A datascience
#PBS -N sync-agpt2b-weight-sync
#PBS -l walltime=00:45:00
#PBS -l filesystems=tegu:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe
set -o pipefail

WT="${SYNC_RL_WORKTREE:?set SYNC_RL_WORKTREE}"
V="${SYNC_RL_VENV:-/lus/tegu/projects/datascience/foremans/venvs/rl-monarch-torch214}"
CKPT="${SYNC_RL_CHECKPOINT:-/lus/tegu/projects/datascience/foremans/reproductions/agpt2b-mds154391-alphabet-sft100/final}"
EXPECTED_SHA="${EXPECTED_SHA:?set EXPECTED_SHA}"
EXPECTED_MODEL_SHA="${EXPECTED_MODEL_SHA:-cd8c185a84c0ef274d04b0be91ba72046b02e7533d9f0507f4948b146e0a5c6e}"
JOB="${PBS_JOBID%%.*}"
OUT="/lus/tegu/projects/datascience/foremans/reproductions/agpt2b-sync-weight-sync-${JOB}"
LOG="$OUT/controller.log"

cd "$WT" || exit 11
source <(curl -fsSL https://ezpz.cool/utils.sh)
ezpz_setup_job
ezpz_load_modules
export VIRTUAL_ENV="$V"
export PATH="$V/bin:/opt/pbs/bin:$PATH"
hash -r
[[ "$(command -v python)" == "$V/bin/python" ]] || exit 13
[[ "$ZE_FLAT_DEVICE_HIERARCHY" == FLAT ]] || exit 14
[[ "$(git rev-parse HEAD)" == "$EXPECTED_SHA" ]] || exit 15
[[ "$(sha256sum "$CKPT/model.safetensors" | cut -d' ' -f1)" == "$EXPECTED_MODEL_SHA" ]] || exit 16

# Preserve the validated same-host Monarch/TorchStore actor contract. This uses
# automatic transport selection; it does not claim a network/XCCL transfer.
unset CCL_OP_SYNC CCL_OFI_PROVIDER TORCHTITAN_TORCHSTORE_TRANSPORT
unset TORCHSTORE_GLOO_ENABLED TORCHSTORE_XCCL_ENABLED
unset FI_CXI_DEFAULT_CQ_SIZE FI_CXI_DEFAULT_TX_SIZE FI_CXI_OFLOW_BUF_COUNT
unset FI_CXI_OFLOW_BUF_SIZE FI_CXI_RDZV_EAGER_SIZE FI_CXI_RDZV_THRESHOLD
unset FI_CXI_REQ_BUF_MAX_CACHED FI_CXI_REQ_BUF_MIN_POSTED FI_CXI_REQ_BUF_SIZE
unset FI_CXI_RX_MATCH_MODE FI_MR_CACHE_MAX_COUNT FI_MR_CACHE_MAX_SIZE
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=tcp
export CCL_KVS_IP_PORT="127.0.0.1_$((29500 + JOB % 1000))"
export PYTHONPATH="$WT${PYTHONPATH:+:$PYTHONPATH}"
export TORCHINDUCTOR_MAX_AUTOTUNE=0 VLLM_ENABLE_V1_MULTIPROCESSING=1
export WANDB_MODE=disabled HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT"

"$V/bin/python" - <<'PY' || exit 17
import importlib.metadata as metadata
from torchtitan.experiments.ezpz.rl.alphabet_sort_agpt.config_registry import (
    rl_grpo_lora_agpt_2b,
)
config = rl_grpo_lora_agpt_2b()
assert config.async_loop.num_training_steps == 3
for package in ("torch", "torchmonarch", "torchstore", "vllm"):
    print(package, metadata.version(package))
PY

printf 'RL_SYNC_START job=%s commit=%s out=%s\n' "$JOB" "$EXPECTED_SHA" "$OUT" | tee "$LOG"
"$V/bin/python" -u -m torchtitan.experiments.ezpz.rl.train_upstream \
    --module torchtitan.experiments.ezpz.rl.alphabet_sort_agpt \
    --config rl_grpo_lora_agpt_2b \
    --hf_assets_path="$CKPT" \
    --dump_folder="$OUT" \
    --async-loop.num-training-steps=3 \
    --async-loop.num-prompts-per-train-step=4 \
    --async-loop.num-samples-per-prompt=4 \
    --async-loop.target-offpolicy-steps=0 \
    --async-loop.validation.num-samples=8 \
    --async-loop.training-sample-builder.no-drop-zero-std-reward-groups \
    --generator.sampling.max-tokens=128 \
    --generator.parallelism.data-parallel-degree=1 \
    --generator.parallelism.tensor-parallel-degree=1 \
    --trainer.parallelism.data-parallel-shard-degree=1 \
    --trainer.parallelism.tensor-parallel-degree=1 \
    --trainer.checkpointer.interval=1 \
    --metrics.no-enable-wandb \
    2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
test "$rc" -eq 0 || {
    printf 'RL_SYNC_VERDICT: failed rc=%s\n' "$rc" | tee -a "$LOG"
    exit "$rc"
}

"$V/bin/python" - "$OUT" <<'PY' | tee -a "$LOG"
import glob
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = [json.loads(line) for line in (root / "rollout_samples.jsonl").read_text().splitlines() if line.strip()]
versions = {
    turn["max_policy_version"]
    for row in rows
    for turn in row.get("turns", [])
}
assert {0, 1}.issubset(versions), versions
assert all(row.get("status") == "completed" for row in rows), "incomplete rollout"
responses = [
    turn.get("completion_message", {}).get("content", "")
    for row in rows
    for turn in row.get("turns", [])
]
assert responses and all("<end_of_turn>" in response for response in responses)
assert (root / "checkpoint/step-3/.metadata").stat().st_size > 0

metric_lines = [
    line for line in (root / "controller.log").read_text(errors="replace").splitlines()
    if "Train | Step:" in line
]
assert len(metric_lines) == 3, len(metric_lines)
grad_norms = []
losses = []
for line in metric_lines:
    grad_norms.append(float(line.split("trainer/grad_norm/mean:", 1)[1].split()[0]))
    losses.append(float(line.split("loss/mean:", 1)[1].split()[0]))
assert all(value == value for value in grad_norms + losses)
assert any(value > 0 for value in grad_norms), grad_norms
assert any(value != 0 for value in losses), losses

events = []
for path in glob.glob(str(root / "structured_logs/*.jsonl")):
    for line in Path(path).read_text(errors="replace").splitlines():
        try:
            events.append(json.loads(line).get("log_type_name"))
        except json.JSONDecodeError:
            pass
assert events.count("push_model_state_dict_end") >= 2, events.count("push_model_state_dict_end")
assert events.count("pull_model_state_dict_end") >= 2, events.count("pull_model_state_dict_end")
assert "optimizer_step_end" in events
print(
    "RL_SYNC_VERDICT: ok "
    f"rows={len(rows)} versions={sorted(versions)} "
    f"grad_norms={grad_norms} losses={losses} "
    f"pushes={events.count('push_model_state_dict_end')} "
    f"pulls={events.count('pull_model_state_dict_end')}"
)
PY
