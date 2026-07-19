#!/bin/bash --login
# Build venvs/rl-grpo-lora/ -- py3.12 venv for the UPSTREAM GRPO+LoRA-on-XPU
# path (Monarch actors + TorchStore + vLLM + fork-torchtitan), reproducing
# songhappy/torchtitan@rl `torchtitan/experiments/rl/GRPO_LORA_XPU.md`.
#
# DIFFERENCE from build_rl_vllm_venv.sh: that one uses RELEASED wheels for the
# TRL vllm-serve path. This one builds monarch/torchstore/vllm/torchtitan-fork
# EDITABLE FROM SOURCE at specific PR branches (the upstream Monarch+TorchStore
# RDMA-weight-store path we abandoned) -- so it needs a real C++20 toolchain
# (gcc-14 on Sunspot; recipe used gcc-13.3 on Borealis) + --no-build-isolation.
#
# Preserves the two CRITICAL dances from build_rl_vllm_venv.sh:
#   1. uninstall impi-rt/oneccl (in-venv libccl shadows the system oneCCL that
#      knows Sunspot's USM allocator -> XCCL "invalid usm pointer" otherwise).
#   2. undo xgrammar's vanilla-`triton` clobber of triton-xpu.
#
# Recipe pins (GRPO_LORA_XPU.md, 2026-07-07): torch 2.12.0+xpu, triton-xpu 3.7.1,
# vllm-xpu-kernels 0.1.10 (NOT 0.1.9.1), transformers 5.9.0, datasets 4.7.0.
# Branches: torchstore #171 xpu-upstream, monarch #4307 xpu-upstream, vllm main,
# torchtitan fork #3890 xpu-upstream.
#
# Build on the UAN. Run: bash rl/scripts/build_rl_grpo_lora_venv.sh
set -e

REPO="${REPO:-$(pwd)}"
VENV="${REPO}/venvs/rl-grpo-lora"
SRC="${RL_REPRO_SRC:-${HOME}/rl-repro}"          # external clones (NOT in repo)
UV="${HOME}/.local/bin/uv"

# --- C++20 toolchain for the from-source builds (gcc-14 on Sunspot) -----------
# Force gcc-14 UNCONDITIONALLY: the login profile exports CC=icx/CXX=icpx, and a
# ${CC:-...} default would respect that. The recipe built from source with gcc
# (13.3); use gcc-14 (not oneAPI icpx) to avoid clang-vs-gcc build/ABI surprises
# in the vLLM/Monarch C++ extensions. oneAPI is still loaded (below) for the
# SYCL/xpu RUNTIME, just not as the C/C++ compiler.
export CC=/usr/bin/gcc-14
export CXX=/usr/bin/g++-14
export PATH="${HOME}/.cargo/bin:${PATH}"  # cargo for monarch Rust build
echo "=== toolchain: $($CC --version | head -1) ==="
"$CXX" --version | head -1

# oneAPI (SYCL/xpu runtime) for the vllm/torch xpu compiles
module load oneapi/release/2025.3.1 hdf5 pti-gpu 2>&1 | tail -1 || true

mkdir -p "${SRC}"

echo ""
echo "=== Step 1: py3.12 venv ==="
"$UV" venv -p 3.12 "${VENV}" --link-mode=copy --clear

echo ""
echo "=== Step 2: torch 2.12+xpu + triton-xpu 3.7.1 (resolve deps for triton-xpu) ==="
VIRTUAL_ENV="${VENV}" "$UV" pip install --no-cache --link-mode=copy \
    --index-url https://download.pytorch.org/whl/xpu \
    "torch==2.12.0+xpu" "torchaudio==2.11.0+xpu" "torchvision==0.27.0+xpu"
VIRTUAL_ENV="${VENV}" "$UV" pip install --no-cache --no-deps --link-mode=copy \
    --index-url https://download.pytorch.org/whl/xpu "triton-xpu==3.7.1"

echo ""
echo "=== Step 2b: build-backend deps for --no-build-isolation ==="
# With --no-build-isolation, each from-source package build needs its build
# backend present in the venv. torchstore is plain setuptools (ok); monarch
# is setuptools-rust; vllm needs cmake/ninja. Pre-install them here.
VIRTUAL_ENV="${VENV}" "$UV" pip install --no-cache --link-mode=copy \
    setuptools "setuptools-rust>=1.9" wheel "numpy<2.5" cmake ninja pybind11 \
    protoc-wheel-0 setuptools_scm
# monarch Rust crate tracing-perfetto-sdk-schema needs protoc at build time
# (not on Sunspot bare shell). protoc-wheel-0 ships a modern one in the venv.
export PROTOC="${VENV}/bin/protoc"

echo ""
echo "=== Step 3: clone/checkout the 4 PR branches into ${SRC} ==="
clone_co () {  # $1=url  $2=dir  $3=ref
    if [ ! -d "${SRC}/$2/.git" ]; then
        git clone "$1" "${SRC}/$2"
    fi
    git -C "${SRC}/$2" fetch --all --tags -q
    git -C "${SRC}/$2" checkout "$3"
    git -C "${SRC}/$2" submodule update --init --recursive -q || true
    echo "  $2 @ $(git -C "${SRC}/$2" rev-parse --short HEAD) ($3)"
}
clone_co https://github.com/songhappy/torchstore.git torchstore xpu-upstream
clone_co https://github.com/songhappy/monarch.git monarch xpu-upstream
clone_co https://github.com/vllm-project/vllm.git         vllm         main
clone_co https://github.com/songhappy/torchtitan.git      torchtitan-fork rl

echo ""
echo "=== Step 4: editable from-source installs (--no-deps --no-build-isolation) ==="
# torchstore
VIRTUAL_ENV="${VENV}" "$UV" pip install --no-cache --link-mode=copy \
    -e "${SRC}/torchstore" --no-deps --no-build-isolation
VIRTUAL_ENV="${VENV}" "$UV" pip install --no-cache --link-mode=copy pygtrie portpicker
# monarch (Rust + setuptools-rust; project at repo ROOT on this fork, not
# python/. Needs cargo on PATH -- it is a from-source Rust build.)
export PATH="${HOME}/.cargo/bin:${PATH}"
VIRTUAL_ENV="${VENV}" "$UV" pip install --no-cache --link-mode=copy \
    -e "${SRC}/monarch" --no-deps --no-build-isolation
# vllm (long compile)
VIRTUAL_ENV="${VENV}" "$UV" pip install --no-cache --link-mode=copy \
    -e "${SRC}/vllm" --no-deps --no-build-isolation
# vllm-xpu-kernels 0.1.10 (must match torch 2.12+xpu)
VIRTUAL_ENV="${VENV}" "$UV" pip install --no-cache --no-deps --link-mode=copy \
    "vllm-xpu-kernels @ https://github.com/vllm-project/vllm-xpu-kernels/releases/download/v0.1.10/vllm_xpu_kernels-0.1.10-cp38-abi3-manylinux_2_28_x86_64.whl"
# torchtitan fork
VIRTUAL_ENV="${VENV}" "$UV" pip install --no-cache --link-mode=copy \
    -e "${SRC}/torchtitan-fork" --no-deps --no-build-isolation

echo ""
echo "=== Step 5: pure-Python deps (safe to resolve; do not touch torch/triton) ==="
VIRTUAL_ENV="${VENV}" "$UV" pip install --no-cache --link-mode=copy \
    pyzmq pyarrow requests "numpy<2.5" pyre-extensions "typing-extensions>=4.12" \
    cloudpickle msgspec fastapi 'uvicorn[standard]' openai pydantic \
    prometheus-client tiktoken sentencepiece protobuf psutil py-cpuinfo cbor2 \
    gguf einops blake3 partial-json-parser outlines outlines-core lm-format-enforcer \
    diskcache mistral-common ray starlette anyio h11 httptools websockets watchfiles \
    python-multipart openai-harmony pybase64 cachetools uvloop xgrammar llguidance \
    opentelemetry-api opentelemetry-sdk tabulate depyf astor \
    torchdata tyro spmd-types tensorboard wandb pillow \
    "renderers @ git+https://github.com/PrimeIntellect-ai/renderers.git@main"

echo ""
echo "=== Step 6: HF/training deps at recipe pins ==="
VIRTUAL_ENV="${VENV}" "$UV" pip install --no-cache --link-mode=copy \
    "transformers==5.9.0" "datasets==4.7.0" tokenizers safetensors \
    --constraint <(echo "torch==2.12.0+xpu")

echo ""
echo "=== Step 7: undo xgrammar's vanilla-'triton' clobber of triton-xpu ==="
VIRTUAL_ENV="${VENV}" "$UV" pip uninstall triton 2>/dev/null || true
VIRTUAL_ENV="${VENV}" "$UV" pip install --reinstall --no-cache --no-deps --link-mode=copy \
    --index-url https://download.pytorch.org/whl/xpu "triton-xpu==3.7.1"

echo ""
echo "=== Step 8: uninstall impi-rt + in-venv oneCCL (CRITICAL for USM/XCCL) ==="
VIRTUAL_ENV="${VENV}" "$UV" pip uninstall impi-rt oneccl oneccl-devel 2>/dev/null || true
echo "  ldd check -- libccl should resolve to SYSTEM oneapi, not the venv:"
ldd "${VENV}/lib/python3.12/site-packages/torch/lib/libtorch_xpu.so" 2>/dev/null | grep -i ccl || true

echo ""
echo "=== Step 9: verify imports (build gate) ==="
"${VENV}/bin/python" -c "
import torch; print('torch', torch.__version__, 'xpu', torch.xpu.is_available(), torch.xpu.device_count())
import triton; print('triton', triton.__version__)
from triton._C.libtriton import intel; print('intel symbols OK')
import vllm; print('vllm', vllm.__version__)
import vllm_xpu_kernels; print('vllm_xpu_kernels OK')
import monarch; from monarch.actor import Actor, endpoint, this_host; print('monarch OK')
import torchstore; from torchstore.transport import TransportType; print('torchstore OK')
import torchtitan.experiments.rl as rl; print('fork torchtitan.experiments.rl OK')
print('ALL IMPORTS OK')
"
echo ""
echo "=== DONE: venvs/rl-grpo-lora built. Clones in ${SRC} ==="
