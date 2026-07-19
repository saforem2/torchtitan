# no set -e: module load returns nonzero under Lmod (CLAUDE.md rule)
LOCK=/tmp/foremans/s4_agpt2b_fp32.lock
if [ -e "$LOCK" ]; then echo "ALREADY RUNNING (lock $LOCK) -- abort"; exit 3; fi
mkdir -p /tmp/foremans; touch "$LOCK"; trap 'rm -f "$LOCK"' EXIT
cd /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan
source /usr/share/lmod/lmod/init/bash 2>/dev/null
module load oneapi/release/2025.3.1 hdf5 pti-gpu >/dev/null 2>&1
export CCL_ROOT=/opt/aurora/26.26.0/oneapi/ccl/latest
export LD_LIBRARY_PATH=/opt/cray/libfabric/2.3.1/lib64:/opt/aurora/26.26.0/oneapi/2025.3/opt/mpi/libfabric/lib:$CCL_ROOT/lib:/opt/aurora/26.26.0/oneapi/2025.3/lib:$LD_LIBRARY_PATH
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu" ZE_AFFINITY_MASK=0,1
export ZES_ENABLE_SYSMAN=1
export FI_PROVIDER=tcp CCL_ATL_TRANSPORT=ofi CCL_ATL_OFI_PROVIDER=tcp
export TORCHINDUCTOR_MAX_AUTOTUNE=0 VLLM_ENABLE_V1_MULTIPROCESSING=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PATH="$HOME/.cargo/bin:/opt/pbs/bin:$PATH"
cd "$HOME/rl-repro/run"
CKPT=/home/foremans/rl-repro/run/agpt2b-ckpt900   # staged: symlinked weights + templated tokenizer
REPO=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan
PY=$REPO/venvs/rl-grpo-lora/bin/python
echo "AGPT2B-FP32 START $(date +%T) host=$(hostname) ckpt=$CKPT"
timeout 3300 $PY -u -m torchtitan.experiments.rl.train \
    --module alphabet_sort --config rl_grpo_lora_agpt_2b \
    --hf_assets_path="$CKPT" \
    --async-loop.num-training-steps=3 \
    --async-loop.num-groups-per-train-step=4 \
    --async-loop.training-sample-builder.no-drop-zero-std-reward-groups \
    --dump_folder="$REPO/outputs/rl_lora_agpt2b_fp32" \
    --trainer.parallelism.data-parallel-shard-degree=1 \
    --generator.parallelism.data-parallel-degree=1 \
    --generator.parallelism.tensor-parallel-degree=1 \
    --generator.gpu-memory-limit=0.70 \
    --generator.model-dtype=float32 --trainer.training.dtype=float32 --generator.sampling.max-tokens=700
echo "AGPT2B-FP32 EXIT rc=$? $(date +%T)"
