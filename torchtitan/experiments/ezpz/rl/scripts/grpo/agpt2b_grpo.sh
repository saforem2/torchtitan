#!/bin/bash
# GRPO+LoRA training for AuroraGPT-2B (SFT checkpoint-900) on Intel XPU, run from
# OUR repo (no ~/rl-repro fork): the ezpz overlay config drives the upstream RL
# engine via the ezpz train_upstream bridge.
#
# Prereqs: venvs/rl-grpo-lora built (scripts/build_rl_grpo_lora_venv.sh) and a
# staged ckpt dir with the gemma chat_template (scripts/grpo_lora_agpt2b_patches/
# stage_agpt2b.sh). Run on a compute node (2 tiles).
#
# no set -e: module load returns nonzero under Lmod (CLAUDE.md rule)
LOCK=/tmp/foremans/agpt2b_grpo.lock
if [ -e "$LOCK" ]; then echo "ALREADY RUNNING (lock $LOCK) -- abort"; exit 3; fi
mkdir -p /tmp/foremans; touch "$LOCK"; trap 'rm -f "$LOCK"' EXIT

REPO=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan
cd "$REPO" || exit 9
source /usr/share/lmod/lmod/init/bash 2>/dev/null
module load oneapi/release/2025.3.1 hdf5 pti-gpu >/dev/null 2>&1
export CCL_ROOT=/opt/aurora/26.26.0/oneapi/ccl/latest
export LD_LIBRARY_PATH=/opt/cray/libfabric/2.3.1/lib64:/opt/aurora/26.26.0/oneapi/2025.3/opt/mpi/libfabric/lib:$CCL_ROOT/lib:/opt/aurora/26.26.0/oneapi/2025.3/lib:$LD_LIBRARY_PATH
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu" ZE_AFFINITY_MASK=0,1
export ZES_ENABLE_SYSMAN=1
export FI_PROVIDER=tcp CCL_ATL_TRANSPORT=ofi CCL_ATL_OFI_PROVIDER=tcp
export TORCHINDUCTOR_MAX_AUTOTUNE=0 VLLM_ENABLE_V1_MULTIPROCESSING=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PATH="$HOME/.cargo/bin:/opt/pbs/bin:$PATH"

# Staged ckpt-900 dir (symlinked weights + gemma-templated tokenizer). Override
# via $CKPT. Config bakes in fp32 gen/trainer, linear reward, few-shot env.
CKPT="${CKPT:-$HOME/rl-repro/run/agpt2b-ckpt900}"
CONFIG="${CONFIG:-rl_grpo_lora_agpt_2b_easy}"   # or rl_grpo_lora_agpt_2b (stock task)
DUMP="${DUMP:-$REPO/outputs/rl_lora_agpt2b_grpo}"
PY=$REPO/venvs/rl-grpo-lora/bin/python

cd "$REPO"   # run from OUR repo root: our torchtitan.experiments.ezpz wins over the fork editable install
echo "AGPT2B-GRPO START $(date +%T) host=$(hostname) config=$CONFIG ckpt=$CKPT"
timeout "${WALLTIME_SEC:-14400}" "$PY" -u -m torchtitan.experiments.ezpz.rl.train_upstream \
    --module torchtitan.experiments.ezpz.rl.alphabet_sort_agpt \
    --config "$CONFIG" \
    --hf_assets_path="$CKPT" \
    --dump_folder="$DUMP" \
    --async-loop.training-sample-builder.no-drop-zero-std-reward-groups \
    --trainer.parallelism.data-parallel-shard-degree=1 \
    --generator.parallelism.data-parallel-degree=1 \
    --generator.parallelism.tensor-parallel-degree=1 \
    --generator.gpu-memory-limit=0.70
echo "AGPT2B-GRPO EXIT rc=$? $(date +%T)"
