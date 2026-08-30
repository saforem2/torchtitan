# TorchTitan on Aurora -- quickstart (validation queue + `frameworks/2026.1.0`)

> Last validated: **2026-08-28**, job `8789506` (2 nodes, 5 corners, all pass)

No tarball copy and no `relocate-venv.sh`: the RC module provides torch, and
`ezpz_setup_env` layers a project venv on top of it for everything else. For
the shared-tarball path see
[aurora-quickstart-tarball.md](aurora-quickstart-tarball.md).

For RC re-validation status across the wider matrix (what has and has not been
re-measured on this stack), see
[frameworks-rc-validation.md](frameworks-rc-validation.md). This page is just
the recipe.

## 1. Submit an interactive job

The validation queue is small and exists for exactly this kind of check --
keep jobs short and do not park on it.

```bash
qsub -q validation -A <project> \
  -l walltime=00:60:00,filesystems=flare:home \
  -l select=2 -I
```

## 2. Clone TorchTitan

```bash
git clone https://github.com/saforem2/torchtitan --branch ezpz
cd torchtitan
```

## 3. Load the RC and set up the environment

Each of these four exports cost a failed job to find. None is optional.

```bash
module load frameworks/2026.1.0

# libglog.so.0 ships here and is NOT on the default loader path.
# Without this every `import torch` dies -- see known-bugs/fw-rc-libglog-not-on-loader-path.md
FW=/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0
export LD_LIBRARY_PATH="$FW/lib:$LD_LIBRARY_PATH"

# compute nodes have no direct egress; ezpz-utils and HF assets need the proxy
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

# 6 cards x 2 tiles. FLAT exposes all 12 tiles as individual XPUs;
# COMPOSITE shows 6 and every --nproc 12 launch fails ngpus validation.
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env
```

`ezpz_setup_env` detects the active RC conda env and creates/activates a
project venv layered on it, at
`venvs/aurora/torchtitan-ezpz-aurora_frameworks-2026.1.0/`. torch comes from
the module underneath; the venv holds everything in step 4. On a fresh clone
that venv starts empty, which is why step 4 exists.

Confirm before going further:

```bash
python3 -c 'import torch; print("torch", torch.__version__, "| xpu", torch.xpu.is_available(), "|", torch.xpu.device_count(), "devices")'
# torch 2.13.0a0+gitcf30153 | xpu True | 12 devices
```

If `device_count()` reports 6 rather than 12, `ZE_FLAT_DEVICE_HIERARCHY` did
not take.

## 4. Dependencies

The module ships torch; these are the extras the training stack needs.

```bash
uvi "git+https://github.com/saforem2/blendcorpus@feat/remove-deepspeed"
uvi "git+https://github.com/saforem2/ezpz"
```

> **Never `pip install torch`, or anything that resolves it.** A resolver run
> will pull a PyPI CUDA build over the XPU one while every line still prints
> `ok` -- only an explicit `torch.__version__` assert catches it. Re-run the
> check from step 3 after any install.

Sanity-check the launcher end to end:

```bash
ezpz launch python3 -m ezpz.examples.test
```

## 5. Get the tokenizer

```bash
python3 scripts/download_hf_assets.py --repo_id google/gemma-7b --assets tokenizer
```

## 6. Launch training

AuroraGPT-2B:

```bash
MODEL=2b
DFL=torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt

ezpz launch python3 -m torchtitan.experiments.ezpz.train \
   --module ezpz.agpt \
   --config "ezpz_agpt_${MODEL}" \
   --training.dataset_path "${DFL}" \
   --debug.print_config
```

### Quick smoke instead

Five steps at a small context, to prove the stack before committing to a real
run:

```bash
ezpz launch --nproc 4 --nproc_per_node 4 \
  -- python3 -m torchtitan.experiments.ezpz.train \
     --module=ezpz.agpt --config=agpt_debugmodel \
     --training.steps=5 \
     --training.max-context-length=512 \
     --training.num-tokens-per-microbatch-per-dp-rank=512 \
     --checkpoint.no-enable --debug.seed=42 --debug.deterministic
```

Expect monotonic descent, roughly `10.78 -> 10.32` over five steps.

## What has been validated on this stack

Job `8789506`, 2 nodes x 4 ranks, 5 steps each, via
[`scripts/rc-matrix.sh`](../../scripts/rc-matrix.sh):

| arm | step 1 -> step 5 |
|---|---|
| agpt TP=1 | 10.78117 -> 10.32300 |
| agpt TP=2, eager | 10.88838 -> 10.43820 |
| agpt TP=2, **compiled** | 10.88835 -> 10.43784 |
| moe TP=1 | 12.93609 -> 11.32376 |
| moe TP=2 | 12.94930 -> 11.49335 |

All monotonic with finite grad norms. Two differences from the June `.venv`
stack are worth knowing before you plan around them:

**Compiled agpt at TP=2 works here.** On the production `.venv` that corner
dies in `tensors_saved_with_vc_check` with a `DeviceMesh` in
saved-for-backward. It does not fire on the RC, and compiled agrees with eager
to `3.6e-4` over five steps -- a real pass, not a silently different code
path. Before retiring the `compile=OFF` workaround generally, confirm at 80B:
that is where the assertion was originally characterised.

**MoE at TP>1 needs commit `6e4e1996f`** (the SDPA wrapper re-flatten). Both
moe arms above reproduce the production-stack numbers bit-identically, so the
fix is not version-dependent -- but an older checkout still fails in `wo` with
`output DTensor has placements (Shard(dim=0),), but out_src_shardings expects
(Partial(sum),)`.
