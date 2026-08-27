#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -q validation
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -N rc-full
#PBS -j oe

# End-to-end on Aurora's frameworks/2026.1.0: setup -> deps -> collectives ->
# torchtitan gate -> real distributed training.
#
# EVERYTHING runs on the validation node: the overlay venv's interpreter
# symlinks into /opt/aurora/26.181.0, absent on a login node (uv there fails
# and still exits 0).
#
# Prior: 8784472 RC python lacks our stack | 8784477 hand venv missing grain
#        8784483 uvi/ezpz absent in batch shell
#        8784495 deps ok, but wandb from the RC base has no .api -> ezpz
#                verify_wandb() dies in cleanup, killing smoke AND training
set -o pipefail

REPO=/flare/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz
cd $REPO || exit 1

module load frameworks/2026.1.0
FW=/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0
export LD_LIBRARY_PATH="$FW/lib:$LD_LIBRARY_PATH"
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

echo "=== [1] ezpz_setup_env ==="
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env
python3 -c 'import torch; print("  torch :", torch.__version__, "| xpu:", torch.xpu.is_available())'

echo
echo "=== [2] deps ==="
UV=/home/foremans/.local/bin/uv
# wandb: the RC base ships something importable WITHOUT .api, which ezpz
# verify_wandb() calls. Install a real one INTO the overlay so it shadows it.
# --no-deps is a TORCH guard (a resolver may swap the XPU build for CUDA).
# It also skips legitimate pure-python deps, which is why grain imported as
# MISSING in 8784502: etils and xarray were never pulled in. Name them.
# grain's deps read from .venv (where grain WORKS) via
# importlib.metadata.requires. Chasing them one failed job at a time cost
# three runs: etils, then xarray, then portpicker.
# Two tiers, because neither blanket policy works:
#
#   --no-deps on everything  -> silently drops pure-python deps. Cost four
#                               jobs (etils, xarray, portpicker, typeguard).
#   resolver on everything   -> pulled a generic PyPI torch-2.13.0 into the
#                               overlay, shadowing the RC's XPU build, even
#                               with -P pins. Caught by the guard in [3]
#                               (job 8784535); -P does NOT prevent this.
#
# So: resolve deps for the leaf packages that cannot drag torch in, and use
# --no-deps for the ones that declare it.
RESOLVE=(tensorboard tyro grain 'etils[epath,epy]' xarray absl-py
         array-record cloudpickle portpicker 'protobuf>=5.28.3' typeguard)
NODEPS=('spmd_types==0.2.1' wandb
        'git+https://github.com/saforem2/ezpz'
        'git+https://github.com/saforem2/blendcorpus@feat/remove-deepspeed')

for p in "${RESOLVE[@]}"; do
    printf '  %-52s ' "$p"
    if $UV pip install --no-cache --link-mode=copy "$p" >/dev/null 2>&1; then
        echo ok
    elif python3 -c "import ${p%%[<>=\[]*}" >/dev/null 2>&1; then
        echo "ok (already satisfied)"
    else
        echo FAILED
    fi
done
for p in "${NODEPS[@]}"; do
    printf '  %-52s ' "$p (no-deps)"
    $UV pip install --no-deps --no-cache --link-mode=copy "$p" >/dev/null 2>&1 \
      && echo ok || echo FAILED
done

# Belt and braces: if anything DID land a torch in the overlay it shadows the
# RC build. Move it aside rather than let the guard just abort the job.
SP=$(python3 -c 'import site; print(site.getsitepackages()[0])')
# triton too: the PyPI torch pull brings a generic triton-3.7.1 that
# shadows the RC's Intel-enabled build. Evicting torch alone leaves it,
# and config_registry then dies on "No module named triton.backends.intel"
# (job 8784572).
for d in torch torchgen functorch triton; do
    if [[ -e "$SP/$d" ]]; then
        mv "$SP/$d" "$SP/$d.evicted-$(date +%H%M%S)"
        echo "  EVICTED overlay $d (was shadowing the RC torch)"
    fi
done
for d in "$SP"/torch-*.dist-info "$SP"/triton-*.dist-info; do
    [[ -e "$d" ]] && mv "$d" "$d.evicted" && echo "  EVICTED $(basename $d)"
done

# torchtitan is NEVER pip-installed; it is imported from the repo tree. A
# PYTHONPATH export does not reach ranks, so drop a .pth into the venv --
# 8784502 failed the gate with "No module named torchtitan" for want of this.
SP=$(python3 -c 'import site; print(site.getsitepackages()[0])')
echo "$REPO" > "$SP/_torchtitan_repo.pth"
echo "  .pth -> $SP/_torchtitan_repo.pth"

echo
echo "=== [3] verify IN THE VENV python (not the login one) ==="
# HARD GATE: dropping --no-deps means a resolver COULD pull a CUDA torch over
# the XPU build. That is the one unrecoverable outcome here, so stop the job
# rather than train on a silently-wrong stack.
python3 - <<'GUARD'
import sys, torch
v = torch.__version__
ok = ("xpu" in v or "gitcf30153" in v) and torch.xpu.is_available()
print("  torch guard:", v, "| xpu:", torch.xpu.is_available(), "|",
      "OK" if ok else "CLOBBERED")
sys.exit(0 if ok else 1)
GUARD
if [[ $? -ne 0 ]]; then
    echo "  ABORT: torch was replaced by the resolver -- refusing to continue."
    exit 1
fi
python3 -c 'import torch; print("  torch :", torch.__version__, "| xpu:", torch.xpu.is_available())'
for m in ezpz spmd_types grain etils xarray blendcorpus tyro tensorboard wandb torchtitan; do
    printf '  %-12s ' $m
    python3 -c "import $m" 2>/dev/null && echo ok || echo MISSING
done
python3 -c 'import wandb; print("  wandb :", wandb.__version__, "| has .api:", hasattr(wandb, "api"))'

echo
echo "=== [4] ezpz smoke: launch + collectives, 2N ==="
python3 -m ezpz.launch python3 -m ezpz.examples.test 2>&1 | tail -10
echo "  SMOKE_RC=$?"

echo
echo "=== [5] torchtitan gate: config_registry ==="
( cd /tmp && unset PYTHONPATH && python3 -c \
   'import importlib; importlib.import_module("torchtitan.experiments.ezpz.agpt.config_registry"); print("  CONFIG_REGISTRY OK")' ) 2>&1 \
  | grep -aE 'CONFIG_REGISTRY OK|^[A-Za-z]*Error' | tail -2

echo
echo "=== [6] REAL distributed training, 2N x 12, 10 steps ==="
python3 -m ezpz.launch python3 -m torchtitan.experiments.ezpz.train \
    --module=ezpz.agpt --config=agpt_debugmodel \
    --training.steps=10 \
    --training.max-context-length=512 \
    --training.num-tokens-per-microbatch-per-dp-rank=512 \
    --checkpoint.no-enable 2>&1 | tail -22
echo "  TRAIN_RC=$?"
