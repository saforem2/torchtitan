#!/bin/bash --login
#PBS -A datascience
#PBS -N rl-grpo-b2-2n
#PBS -l walltime=06:00:00
#PBS -l filesystems=tegu:home
#PBS -l select=3
#PBS -q workq
#PBS -j oe
#
# 2N Monarch GRPO on B2 checkpoint-93 (0.205 GSM8K-CoT CoT/drift-proof base).
#
# Base: outputs/sft/agpt2b-gsm8k-r1cot-2n/checkpoint-93-hf
#   (POST-TRAINING-2B.md deliverable: CoT/math reasoning, B2 = ckpt-93,
#    0.205 GSM8K-CoT format 0.985, gen_len 282. Strong cold-start, format
#    held 0.985 and 1.000 under prior GRPO, so safe to raise LR.)
#   SFT README.md confirms checkpoint-93 is present at outputs/sft/agpt2b-gsm8k-r1cot-2n/
#   (unlike ckpt-900-hf which needed re-consolidate; ckpt-93 never went stale).
#   Validate at submit: ls .../checkpoint-93* + chat_template + model file.
#
# Hyperparams (POST-TRAINING-2B.md Next steps #2, gated 100-step on B2 0.205->0.215 noise):
#   group_size 8 (POST-TRAINING next-step says 8-16 for denser gradient; at ~20%
#     solve rate group_size 4 left most groups all-wrong so advantage=0; 8 doubles
#     the MIXED-group rate, B2-100 already probed 4)
#   LR 5e-6 (next-step lever: raise into 5e-6--1e-5 now that format is bulletproof
#     at 1.0; prior 2e-6 was deliberately UNATTENDED-safe; we step to the lower
#     end of the recommended band)
#   Both quoted from docs/production/POST-TRAINING-2B.md (the spec) and
#   docs/production/sft/agpt/2b-mds/README.md (SFT deliverables).
#
# Launcher: MonarchXpuScout-validated (MDS154391) multihost TorchStore pattern
#   from torchtitan/experiments/ezpz/rl/scripts/grpo/agpt2b_multihost_torchstore_validate.pbs
#   (job-validated: 2N host-mesh, FileStore, TORCHSTORE_TRANSPORT auto, repair
#    hostname env -- NOT raw TRL server which can't do 2-host vLLM weight sync).
#   Key pieces copied verbatim: FileStore + host_mesh_from_store + CCL_KVS_IP_PORT,
#   TORCHSTORE_TRANSPORT toggle, CCL=none/ofi/tcp, rank via PMI/PALS, controller-
#   only wandb, multihost_train_upstream.py entry.
#
# Why 6h walltime: B2 smoke used 20 steps / ~2h; this lane targets 100 steps
#   (b2long: 100 steps, ckpt_interval 25) which fits in 6h on PVC (vLLM float32
#   generation is ~30% slower but correct -- bf16 flips argmax per POST-TRAINING
#   XPU landmine). 3 spare-friendly walltime (the SFT lane takes 12h).
#
# Pack: 2N active (select=3 = 2+1 spare) + SFT 4N + CPT 4N + CPT 4N + test.sh 2N = 16N active.
#   4-script budget: 4+4+4+2=14N + test.sh 2N = 16N active (spare nodes are PBS
#   over-provision, not counted toward the 16N workq budget; workq max is 128N).
#
# Submit:
#   qsub torchtitan/experiments/ezpz/scripts/rl_grpo_b2_2n.sh
#   qsub -v EXPECTED_COMMIT=<sha> torchtitan/experiments/ezpz/scripts/rl_grpo_b2_2n.sh  # pinned worktree
# Override N (not recommended -- GRPO validated at 2N):
#   qsub -l select=3 -v EXPECTED_COMMIT=<sha> torchtitan/experiments/ezpz/scripts/rl_grpo_b2_2n.sh
set -euo pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu >/dev/null 2>&1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=tcp
export TORCHINDUCTOR_MAX_AUTOTUNE=0
export VLLM_ENABLE_V1_MULTIPROCESSING=1

# Worktree-current check (same gate as the validated PBS)
WT="${PBS_O_WORKDIR:-$(pwd)}"
cd "${WT}" || exit 11
if [[ -n "${EXPECTED_COMMIT:-}" ]]; then
  actual_commit=$(git rev-parse HEAD) || exit 15
  [[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || { echo "FATAL: commit mismatch expected=${EXPECTED_COMMIT} actual=${actual_commit}" >&2; exit 15; }
  git diff --quiet && git diff --cached --quiet || { echo "FATAL: tracked worktree changes present" >&2; exit 15; }
else
  h=$(git rev-parse --short HEAD 2>/dev/null); u=$(git rev-parse --short '@{u}' 2>/dev/null || echo "$h")
  [[ "$h" == "$u" ]] || echo "WARN: worktree $h != upstream $u -- fetch/checkout first" >&2
fi

# Venv: Monarch+XPU+TRL+torch 2.14 (same family as validated .pbs; locality
# variant carries the same Monarch stack as the validated rl-monarch-torch214)
V="${V:-/lus/tegu/projects/datascience/foremans/venvs/rl-monarch-torch214-locality-20261001}"
# Fallback to the non-locality sibling if the locality venv was GC'd
if [[ ! -x "${V}/bin/python" ]]; then V="/lus/tegu/projects/datascience/foremans/venvs/rl-monarch-torch214"; fi
[[ -x "${V}/bin/python" ]] || { echo "FATAL: venv not found: ${V}" >&2; exit 1; }

# Offline: model + datasets + tokenizer are all on shared FS; no Hub calls from ranks
export WANDB_MODE=disabled HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING="ascii:replace"

# Monarch transport gate -- copied verbatim from the validated PBS so preflight
# + runtime agree. TORCHSTORE_TRANSPORT=auto is the validated default (Sunspot
# PVC picks the right transport via torchstore's own probe); auto_no_shm is the
# Sunspot repair_hostname fallback.
export TORCHSTORE_TRANSPORT="${TORCHSTORE_TRANSPORT:-auto}"
case "${TORCHSTORE_TRANSPORT}" in
  auto)         unset TORCHTITAN_TORCHSTORE_TRANSPORT TORCHSTORE_SHARED_MEMORY_ENABLED ;;
  auto_no_shm)  unset TORCHTITAN_TORCHSTORE_TRANSPORT; export TORCHSTORE_SHARED_MEMORY_ENABLED=0 ;;
  gloo|xccl|monarch_rdma) export TORCHTITAN_TORCHSTORE_TRANSPORT="${TORCHSTORE_TRANSPORT}" ;;
  *) echo "FATAL: unsupported TORCHSTORE_TRANSPORT=${TORCHSTORE_TRANSPORT}" >&2; exit 15 ;;
esac
unset TORCHSTORE_GLOO_ENABLED TORCHSTORE_XCCL_ENABLED
unset CCL_OP_SYNC CCL_OFI_PROVIDER

# CCL rendezvous (validated pattern): host 0 addr + job-scoped port
head_node=$(head -1 "${PBS_NODEFILE}")
export CCL_KVS_IP_PORT="${head_node}_$((29500 + ${PBS_JOBID%%.*} % 1000))"

# B2 base: checkpoint-93-hf (preferred) -- SFT deliverable per POST-TRAINING-2B.md
# If HF export is missing, accept the sharded ckpt-93 dir (multihost entry can
# load either via initial_load_in_hf; hf_assets_path is HF-only, but the
# controller's checkpointer handles sharded fallback).
DEFAULT_CKPT_HF="${WT}/outputs/sft/agpt2b-gsm8k-r1cot-2n/checkpoint-93-hf"
DEFAULT_CKPT_SHARDED="${WT}/outputs/sft/agpt2b-gsm8k-r1cot-2n/checkpoint-93"
if [[ -n "${CKPT:-}" ]]; then
  : # user-supplied CKPT wins
else
  if [[ -s "${DEFAULT_CKPT_HF}/model.safetensors" ]]; then CKPT="${DEFAULT_CKPT_HF}"
  elif [[ -s "${DEFAULT_CKPT_HF}/pytorch_model.bin" ]]; then CKPT="${DEFAULT_CKPT_HF}"
  elif [[ -d "${DEFAULT_CKPT_SHARDED}" ]]; then CKPT="${DEFAULT_CKPT_SHARDED}"
  else CKPT="${DEFAULT_CKPT_HF}"
  fi
fi
# If only sharded DCP exists (checkpoint-93 is FSDP, not HF), consolidate to -hf.
# This is a one-time merge via accelerate (needs the SFT base tokenizer at ckpt-900-hf or agpt model).
# SFT ckpt-93's tokenizer is the same gemma tokenizer as the base; we derive SRC from the stage-1 base.
if [[ ! -s "${CKPT}/config.json" ]] && [[ -d "${CKPT}/pytorch_model_fsdp_0" ]]; then
  echo "CKPT ${CKPT} is sharded DCP (no config.json), consolidating to ${DEFAULT_CKPT_HF} ..." >&2
  # Source HF config/tokenizer: prefer ckpt-900-hf (same model/tokenizer), fallback to any SFT final
  SRC_HF="${WT}/outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900-hf"
  if [[ ! -f "${SRC_HF}/config.json" ]]; then
    SRC_HF="${WT}/outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900"
    # If still sharded, try the B2v2 SFT final which has config (it's TRL, writes HF)
    if [[ ! -f "${SRC_HF}/config.json" ]]; then
      SRC_HF=$(find "${WT}/outputs/sft/agpt2b-b2v2-distill-cot-2n-ep3" -name "config.json" -printf "%h\n" 2>/dev/null | head -1)
    fi
  fi
  if [[ -n "${SRC_HF}" && -f "${SRC_HF}/config.json" ]]; then
    mkdir -p "${DEFAULT_CKPT_HF}"
    "${V}/bin/python" -m accelerate.commands.merge_weights --help >/dev/null 2>&1 || echo "WARN: accelerate merge-weights not available, trying alt" >&2
    "${V}/bin/accelerate" merge-weights "${CKPT}/pytorch_model_fsdp_0" "${DEFAULT_CKPT_HF}" 2>&1 | tail -5
    cp "${SRC_HF}/config.json" "${DEFAULT_CKPT_HF}/" 2>/dev/null || true
    cp "${SRC_HF}"/tokenizer* "${DEFAULT_CKPT_HF}/" 2>/dev/null || true
    cp "${SRC_HF}/special_tokens_map.json" "${DEFAULT_CKPT_HF}/" 2>/dev/null || true
    "${V}/bin/python" "${WT}/torchtitan/experiments/ezpz/rl/scripts/repair_agpt_hf_artifact.py" "${DEFAULT_CKPT_HF}" 2>&1 | tail -3
    echo "Consolidated ${CKPT} -> ${DEFAULT_CKPT_HF}" >&2
    ls -lh "${DEFAULT_CKPT_HF}/model.safetensors" 2>&1 | head -1
    CKPT="${DEFAULT_CKPT_HF}"
  else
    echo "WARN: cannot consolidate ${CKPT}: no SRC_HF with config.json found (tried ${SRC_HF})" >&2
  fi
fi

OUT="${OUT:-${WT}/outputs/rl/grpo-agpt2b-b2-gs${GROUP_SIZE:-8}-lr${LR:-5e-6}-${PBS_JOBID%%.*}}"

if [[ ! -s "${CKPT}/config.json" ]] && [[ ! -s "${CKPT}/model.safetensors" ]] && [[ ! -s "${CKPT}/pytorch_model.bin" ]]; then
  echo "FATAL: model missing at CKPT=${CKPT} (expected config.json or model.safetensors)" >&2
  find "${WT}/outputs/sft/agpt2b-gsm8k-r1cot-2n" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -20 >&2
  exit 16
fi

STORE="${WT}/.multihost-rl-${PBS_JOBID%%.*}.store"
rm -f "${STORE}"
PORT=$((35000 + ${PBS_JOBID%%.*} % 20000))

# LD-first for UR/SYCL (rl-monarch venv, same as SFT B2v2)
if [[ -d "${V}/lib" ]]; then
  _ld_first="${WT}/.cache/xpu-ld-first/$(basename "${V}")"
  if [[ ! -e "${_ld_first}/.complete" ]]; then
    mkdir -p "${_ld_first}"
    for _lib in "${V}"/lib/libur_* "${V}"/lib/libsycl.so* "${V}"/lib/libumf* "${V}"/lib/libtcm*; do
      [[ -e "${_lib}" ]] && ln -sf "${_lib}" "${_ld_first}/"
    done
    touch "${_ld_first}/.complete"
  fi
  export LD_LIBRARY_PATH="${_ld_first}:${LD_LIBRARY_PATH:-}"
fi
export PATH="${V}/bin:/opt/pbs/bin:${PATH}"
export PYTHONPATH="${WT}${PYTHONPATH:+:${PYTHONPATH}}"

"${V}/bin/python" -m py_compile torchtitan/experiments/ezpz/rl/scripts/grpo/multihost_train_upstream.py

# Preflight: config contract (same assertions as validated PBS, plus our group_size/LR)
GROUP_SIZE="${GROUP_SIZE:-8}"
LR="${LR:-5e-6}"
NUM_STEPS="${NUM_STEPS:-100}"
NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-8}"
CKPT_INTERVAL="${CKPT_INTERVAL:-25}"
"${V}/bin/python" - <<PY
import os
from torchtitan.experiments.ezpz.rl.reason_agpt.config_registry import (
    rl_grpo_lora_agpt_2b_gsm8k_b2smoke,
)
c = rl_grpo_lora_agpt_2b_gsm8k_b2smoke()
assert c.async_loop.num_samples_per_prompt == 4, c.async_loop.num_samples_per_prompt
assert c.generator.model_dtype == "float32", c.generator.model_dtype
assert tuple(c.renderer.extra_stop_token_ids) == (1, 107), c.renderer.extra_stop_token_ids
requested = os.environ.get("TORCHTITAN_TORCHSTORE_TRANSPORT", "auto")
assert requested in {"auto", "gloo", "xccl", "monarch_rdma"}, requested
if os.environ.get("TORCHSTORE_TRANSPORT") == "auto_no_shm":
    assert os.environ["TORCHSTORE_SHARED_MEMORY_ENABLED"] == "0"
# Our overrides vs smoke (smoke: gs=4 lr=2e-6 steps=20): this lane is 8-16 / 5e-6
# as called out in POST-TRAINING-2B.md Next steps.
group_size = int("${GROUP_SIZE}")
assert 8 <= group_size <= 16, f"GROUP_SIZE {group_size} not in 8-16 band"
lr_s = "${LR}"
lr = float(lr_s)
assert 4.9e-6 <= lr <= 1.1e-5, f"LR {lr} not in 5e-6--1e-5 band"
print(
    f"MULTIHOST_RL_PREFLIGHT_OK transport={requested} "
    f"shared_memory={os.environ.get('TORCHSTORE_SHARED_MEMORY_ENABLED', 'default')} "
    f"group_size={group_size} lr={lr} smoke_gs=4 smoke_lr=2e-6"
)
PY

echo "MULTIHOST_RL_START commit=$(git rev-parse HEAD) transport=${TORCHSTORE_TRANSPORT} trainer_host=0 generator_host=1 out=${OUT} group_size=${GROUP_SIZE} lr=${LR} steps=${NUM_STEPS} ckpt=${CKPT} $(date -Is)"
set +e
"${V}/bin/ezpz" launch -n 2 -ppn 1 --cpu-bind none --auto-retry --spare-nodes auto --timeout 1800 -- \
  "${V}/bin/python" -u \
  torchtitan/experiments/ezpz/rl/scripts/grpo/multihost_train_upstream.py \
  --multihost-store "${STORE}" --multihost-port "${PORT}" --multihost-world-size 2 \
  --module torchtitan.experiments.ezpz.rl.reason_agpt \
  --config rl_grpo_lora_agpt_2b_gsm8k_b2smoke \
  --hf_assets_path="${CKPT}" \
  --dump_folder="${OUT}" \
  --async-loop.num-training-steps="${NUM_STEPS}" \
  --async-loop.num-prompts-per-train-step="${NUM_PROMPTS_PER_STEP}" \
  --async-loop.num-samples-per-prompt="${GROUP_SIZE}" \
  --async-loop.target-offpolicy-steps=0 \
  --async-loop.validation.num-samples=20 \
  --async-loop.training-sample-builder.no-drop-zero-std-reward-groups \
  --generator.sampling.max-tokens=700 \
  --generator.parallelism.data-parallel-degree=1 \
  --generator.parallelism.tensor-parallel-degree=1 \
  --trainer.parallelism.data-parallel-shard-degree=1 \
  --trainer.parallelism.tensor-parallel-degree=1 \
  --trainer.checkpointer.interval="${CKPT_INTERVAL}" \
  --trainer.optim.optimizer.lr="${LR}" \
  --metrics.no-enable-wandb
rc=$?
set -e
rm -f "${STORE}"
echo "MULTIHOST_RL_DONE rc=${rc} $(date -Is) out=${OUT}"
# GRPO success = mean reward rises AND format_hit_rate stays >=0.9 (per the
# b2smoke docstring's hard tripwire: with no KL anchor the B2 prior holding the
# envelope is the sole restoring force). Check ${OUT}/metrics or W&B if enabled.
[[ "${rc}" -eq 0 ]] || exit "${rc}"
