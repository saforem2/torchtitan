# Frameworks RC (oneAPI 2026.1.0) -- validation status

> [!IMPORTANT]
> **The RC is the target stack**, not a workaround -- it is the build heading to
> Aurora. Every finding in our docs was measured on the June `.venv`
> (torch `2.13.0.dev20260519+xpu` / oneAPI 2025.3.1), so each is **unverified
> on the RC until re-run**. This page tracks that re-validation.

**Stack:** `frameworks/2026.1.0` (oneAPI 2026.1.0, oneCCL `26.181.0`) +
torch `2.13.0a0+gitcf30153` (RC4 wheelforge conda env).

```bash
module use /opt/aurora/26.181.0/modulefiles
module load frameworks/2026.1.0          # oneAPI + oneCCL. Provides NO torch.
source venvs/fw-2026.1-rc2/bin/activate  # RC4 conda env + torchtitan deps
unset PYTHONPATH
```

Gate on `import torchtitan.experiments.ezpz.agpt.config_registry`, never on
`import agpt` -- `config/manager.py` masks every missing dependency behind one
generic error, so the latter is a false pass.

## Aurora deployment (2026-08-25): a DIFFERENT layout, and one blocker

The recipe above is the **Sunspot** one. `frameworks/2026.1.0` reached Aurora
in the maintenance that ended 2026-08-25, and it is deployed differently:

| | Sunspot | Aurora validation nodes |
|---|---|---|
| torch | none -- module provides oneAPI only, venv supplies torch | **bundled**: the module's own `python3` (3.12.12) has torch `2.13.0a0+gitcf30153` |
| venv | `venvs/fw-2026.1-rc2` | does not exist; no `grain`-bearing or RC venv on this host |
| visible from login node | yes | **no** -- `/opt/aurora/26.181.0` exists only in the validation-node image |

That last row matters operationally: `module avail frameworks` on an Aurora
login node shows **only** `2025.3.1`, and a `find` for anything under
`26.181.0` returns nothing. The module is real, but you cannot see it -- or
test it -- without landing on a validation node.

Queue access is by ACL (`acl_user_enable = True`); `foremans` was added
2026-08-25. Three nodes, one of them offline for image testing:
`x4413c2s2b0n0`, `x4003c0s0b0n0`, `x4508c2s0b0n0` (offline, "Image testing --
bsallen [2026-08-18]").

Image delta vs a login node: SLES **15-SP7** (login: 15-SP4), level-zero
`1.6.33578.77-1146` (login: `.42-1146`), opencl `25.18.33578.77` (login:
`.42`), oneAPI tree `26.181.0` (login: `25.190.0` + `26.26.0`).

### BLOCKER: `import torch` fails until you export the module's own `lib/`

Every torch import dies immediately after `module load frameworks/2026.1.0`:

```
torchcomms/__init__.py:42  ctypes.CDLL(libtorchcomms_path, mode=RTLD_LOCAL)
OSError: libglog.so.0: cannot open shared object file: No such file or directory
```

`torchcomms` is imported from `torch/__init__.py`, so this gates everything --
not an optional component.

The library is not missing. It ships **inside the module's own tree** and the
modulefile does not put that directory on `LD_LIBRARY_PATH`:

```
/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/lib/libglog.so.0     (0.4.0)
```

`ldd libtorchcomms.so` reports `libglog.so.0 => not found` and
`libgflags.so.2.2 => not found` (the torch libs also show not-found there, but
resolve at import time through torch's own loader).

Version skew is worth noting: 2025.3.1 ships `libglog.so.2` (0.7.1);
2026.1.0 ships `libglog.so.0` (0.4.0) -- a downgrade, not an omission.

**Workaround**, MEASURED to fix it (job `8781129`):

```bash
FW=/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0
module load frameworks/2026.1.0
export LD_LIBRARY_PATH="$FW/lib:$LD_LIBRARY_PATH"
```

A `module load` must be at TOP LEVEL, never piped. `module load ... | tail -3`
runs the load in a subshell and it silently evaporates -- python stays
`/usr/bin/python3` 3.6.15 and every import fails with a misleading
`No module named 'torch'`. Cost one allocation (job `8781054`) to relearn.

### What passes on Aurora with the workaround (job `8781129`, 1N)

| check | result |
|---|---|
| torch | `2.13.0a0+gitcf30153`, `xpu.is_available()` True, 12 devices, `Intel(R) Data Center GPU Max 1550` |
| `import torchcomms` | OK |
| bf16 matmul (512x512) | OK, mean 0.1505 |
| SDPA fwd+bwd bf16 | OK, `|grad_q|` 51.5 |
| **compiled** SDPA fwd+bwd | **OK**, `|grad_q|` 52.5 |
| `dist.is_xccl_available()` | **True** (job `8781210`) |

The compiled SDPA backward is the surface that failed on the Sunspot fw-RC with
`assert_size_stride` (see `known-bugs/fw-rc-compile-sdpa-backward-tp4.md`).
Same build hash, so this is RC4 and consistent with RC4 having fixed it --
but note this is **1 rank**, not TP=4, so it does not yet retire that entry.

`oneccl_bindings_for_pytorch` is absent and that is CORRECT, not a gap: it is
the IPEX-era external shim, and torch 2.13 carries XCCL natively.

### Not yet established

- **XCCL collectives.** `is_xccl_available()` is True but no all_reduce has
  completed. Two attempts failed in the harness, not the backend: PALS exports
  `PMI_RANK`/`PMI_SIZE` rather than the `RANK`/`WORLD_SIZE` that `env://`
  rendezvous wants (job `8781210`), then `EADDRINUSE` on a hardcoded port
  (`8781295`). Use `ezpz launch` rather than hand-rolled rendezvous.
- **Anything above 1 rank**, including the TP=4 question above.
- Beware: `val-xccl.sh` set `pipefail` without `set -e`, so the job reported
  `Exit_status = 0` while `mpiexec` failed. Do not read a 0 from these probes
  as a pass.

## Results (job `12473146`, Sunspot, 2-4N, 10 steps each)

| case | verdict | final loss | note |
|---|---|---:|---|
| 2B eager | **PASS** | 11.06 | baseline |
| 2B compile TP=1 | **PASS** | 8.94 | |
| 2B compile **TP=4** | **PASS** | 8.74 | the fw-RC `assert_size_stride` regression is **fixed at 4N** (previously only claimed at 2N) |
| 2B HSDP (shard=12, replicate=2) | **PASS** | 11.73 | |
| MoE EP=1 (`moe_10b_2b_sdpa`) | **PASS** | 9.29 | first confirmation **above 2N**; this config SIGABRT'd on the old stack |
| 80B TP=4 @ 4N | **PASS** | 11.26 | below the NaN regime |

## Fixed by the RC

- **Reducing collectives.** `reduce_scatter_tensor` and `all_reduce` SIGSEGV on
  oneAPI 2025.3.1 (12 ranks, one node, 1 KiB buffer) and run clean on the RC.
  This blocked *all* multi-rank training. Full diagnosis:
  [sunspot-reduce-scatter-segv-20260814](known-bugs/sunspot-reduce-scatter-segv-20260814.md).
- **`torch.compile` at TP=4** -- see the table above.
- **MoE EP=1 above 2N.**

## NOT fixed by the RC -- still open

The RC is a collectives/compiler fix. It does **not** touch the numerics walls:

- **80B bf16 NaN (Wall 1) -- CONFIRMED STILL PRESENT ON THE RC.** Job
  `12473142`, dp=192, bf16: `grad_norm=nan` at **step 30**, loss NaN from 31.
  The nan-abort guard fired at 35 and reclaimed the walltime.

  ```
  step: 28  loss: 11.77  grad_norm: 8.4614
  step: 29  loss: 11.69  grad_norm: 8.6428
  step: 30  loss: 12.12  grad_norm:     nan
  ```

  Step 30 sits squarely in the historical ladder (14 / 17 / 19 / 38), so the
  wall is **stack-independent** -- which strengthens the root cause (a deep
  bf16 residual overflow) rather than weakening it.

  One difference from the recorded signature: grad_norm was **rising** into the
  failure (8.06 -> 8.29 -> 8.46 -> 8.64 -> nan), not dead-flat then instant-NaN.
  Worth noting for diagnosis. LR was only ~1.5e-7 at step 30 (warmup 200), so
  this is not an LR-ceiling effect.
- **The dp ceiling above 264 is unmapped.** Only dp-axis NaN on record is
  dp=372, and that run was TP=2 (confounded). Production wants dp=768-1536.
  Sunspot caps at dp=276 at TP=4, so this needs Aurora.
- **QK-norm TP=4 backward** (`tensor does not have a device`) -- not retested.
- **Pipeline parallelism** -- a torch version floor, unrelated; not retested.

## fp32 activations at dp=192 -- what it revealed (job `12473149`)

Running the SAME config with `--training.mixed-precision-param=float32` at the
dp where bf16 dies exposes how different the two numerically are:

| step | bf16 grad_norm | fp32 grad_norm | bf16 loss | fp32 loss |
|---:|---:|---:|---:|---:|
| 5 | 8.33 | **17,418** | 15.46 | 12.84 |
| 6 | 8.01 | **80,210** | 12.88 | 12.79 |
| 7 | 7.86 | **91,467** | 12.90 | 12.75 |
| 11 | 8.05 | **80,065** | 12.76 | **12.39** |

**bf16 is not mis-reporting a large gradient -- it is computing a different,
smaller one.** bf16 has fp32's full exponent range (max ~3.4e38), so 18,000 is
trivially representable; the 65504 ceiling is *fp16*. What bf16 loses is its
**7-bit mantissa**, which destroys the small-magnitude contributions whose
accumulation across an 84-layer residual stream produces these spikes. This is
exactly the mantissa-not-exponent mechanism the root-cause report describes.

Two consequences:

- **"grad_norm ~= 8" in every historical 80B bf16 log is a real value for a
  degraded gradient**, not a masked reading of the true one. Treat those logs as
  measuring a different optimization problem than the one we intend to solve.
- **Gradient clipping is not broken.** `max_norm=1.0` scales by `1/total_norm`
  using the norm computed *in that run*; the bf16 run genuinely has norm ~8. The
  gradients are wrong, not the clipping. (An earlier draft of this page claimed
  clipping was misbehaving by ~2000x -- that was wrong and is corrected here.)
- **fp32 learns faster**, which is independent corroboration: at step 11 fp32 is
  at loss 12.39 vs bf16 12.76, and accelerating. If bf16 merely mis-reported a
  correct gradient, both would descend identically.

### It CLEARS the wall -- 120/120 steps, zero NaN (COMPLETE)

**Final: `rc=0, steps=120, nan_lines=0`, loss 12.95 -> 8.098.** The run
completed its full budget at production dp with no NaN, through full LR.

bf16 died at step 30 at this dp. fp32 walked straight through it:

```
step: 28  loss: 10.55757  grad_norm: 15.6207
step: 29  loss: 10.39867  grad_norm: 11.2650
step: 30  loss: 10.28401  grad_norm:  7.8810   <- bf16 NaN'd here
step: 31  loss: 10.17815  grad_norm:  9.0314
step: 32  loss: 10.08726  grad_norm:  9.7592
```

**And it does so at a HIGHER learning rate.** The fp32 run uses warmup=40, so at
step 30 it is at LR 7.5e-7; the bf16 run used warmup=200 and failed at only
1.5e-7 -- 5x lower. The comparison is therefore conservative: fp32 survives a
strictly harder condition than the one that killed bf16. (The two runs are not
LR-matched, so this is not a controlled A/B on LR; it is a one-sided result --
fp32 clears a bar bf16 could not, with margin to spare.)

Note also that the grad_norm spikes decay as training settles: 17K-91K over
steps 2-12, then 13-180 by step 20, then single digits by step 30. The early
spikes are a startup transient, not a persistent regime.

Cost, measured at dp=192: memory 36.5 GiB (57%) vs bf16 20.5 GiB (32%);
throughput **16 tps vs 54 -- a 3.4x slowdown**, matching the documented 3-5x.

## Throughput re-baselined on the RC -- FASTER at every scale (job `12473156`)

Config matched to the original 2B baseline exactly so the comparison is valid:
FSDP-only (TP=1), compile=ON, AC=full, seq_len=8192, LBS=2, GBS = N x 12 x 2.
Reported value is the **median of the last 10 steps** (the first few are
compile/warmup, and a single tail step is noisy).

| nodes | GBS | RC tps/gpu | old tps/gpu | delta | RC MFU | old MFU |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 48 | **7,800** | 7,142 | **+9.2%** | 29.26% | 27.6% |
| 4 | 96 | **7,673** | 7,068 | **+8.6%** | 28.79% | 27.3% |
| 16 | 384 | **7,386** | 6,995 | **+5.6%** | 27.71% | 27.0% |
| 64 | 1,536 | **7,003** | 6,702 | **+4.5%** | 26.28% | 25.9% |

**No regression -- the RC is 4.5-9.2% faster**, with the gain largest at small N
and narrowing as scale grows (consistent with communication taking a larger
share of the step at 64N, which the RC does not change). Scaling efficiency is
essentially unchanged: 64N/2N is 89.8% on the RC vs 93.8% before, so the RC
improves per-GPU compute more than it improves the collective path.

Median and last-step agree to within 0.1% at every point (7005/7003, 7387/7386,
7671/7673, 7808/7800), so these are stable measurements rather than tail noise.

### Still stale

The **80B** numbers and the **20B/MoE** scaling tables have not been re-measured
on the RC -- only the 2B ladder above. The dp=192/264 "clean run" evidence is
also pre-RC, though the 80B NaN behaviour has now been re-confirmed directly
(bf16 NaNs at step 30; fp32-acts runs 120/120 clean).

## Aurora overlay venv on `frameworks/2026.1.0` (moved from the ezpz README, 2026-11-14)

> [!NOTE]
> Written 2026-08-26 against the pre-sync tree. Since then: TP > 1 needs
> `spmd_types==0.3.1` (not `0.2.1` below), XPU needs torch >= 2.14 (this module's
> `2.13.0a0` is below it), and the module is visible from login nodes.

> [!IMPORTANT]
> **Validation-queue only, as of 2026-08-26.** `/opt/aurora/26.181.0` exists
> only in the validation-node image -- `module avail frameworks` on a login
> node shows just `2025.3.1`, and you cannot build or test against the RC from
> there. When the RC goes live cluster-wide this section becomes the normal
> path; until then every command below must run **on a validation node**.

The RC bundles its own python 3.12.12 with torch `2.13.0a0+gitcf30153`
(xpu available, 12 devices, `dist.is_xccl_available()` True). What it does
**not** bundle is our stack -- `ezpz`, `spmd_types`, `grain`, `blendcorpus`
and friends are all absent, so a bare `module load` gets you a working torch
and nothing that can train.

#### Setup

```bash
# 1. Load the module at TOP LEVEL. Never pipe it (see traps below).
module load frameworks/2026.1.0

# 2. libglog.so.0 lives inside the module tree but is not on the path.
#    Every `import torch` dies without this. Full writeup:
#    docs/guides/known-bugs/fw-rc-libglog-not-on-loader-path.md
#    Without this every `import torch` dies in torchcomms.
FW=/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0
export LD_LIBRARY_PATH="$FW/lib:$LD_LIBRARY_PATH"

# 3. Compute nodes need the proxy for any install.
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

# 4. Build the overlay venv on the RC python.
source <(curl -fsSL https://ezpz.cool/utils.sh) && ezpz_setup_env
#    -> venvs/aurora/torchtitan-ezpz-aurora_frameworks-2026.1.0

# 5. Dependencies. -P pins the torch family out of the resolver's reach;
#    the XPU build is on no index and would be replaced by a CUDA wheel.
uv pip install --no-cache --link-mode=copy \
    -P torch -P pytorch-triton-xpu -P torchvision -P torchaudio \
    tensorboard tyro grain 'etils[epath,epy]' xarray absl-py \
    array-record cloudpickle portpicker 'protobuf>=5.28.3' \
    'spmd_types==0.2.1' wandb \
    'git+https://github.com/saforem2/ezpz' \
    'git+https://github.com/saforem2/blendcorpus@feat/remove-deepspeed'

# 6. torchtitan is NEVER pip-installed -- it is imported from the repo tree,
#    and a PYTHONPATH export does not reach ranks. Drop a .pth instead.
SP=$(python3 -c 'import site; print(site.getsitepackages()[0])')
echo "$PWD" > "$SP/_torchtitan_repo.pth"
```

Then verify, **with the venv python** (not the login one):

```bash
python3 -c "import torch; print(torch.__version__, torch.xpu.is_available())"
python3 -c "import importlib; importlib.import_module(
    'torchtitan.experiments.ezpz.agpt.config_registry')"
```

A ready-made script that does all of the above plus a 2N collectives smoke and
a real training run is
[`scripts/val-rc-full.sh`](../../scripts/val-rc-full.sh).

#### Why each dependency is here

| Package | Needed by |
|---|---|
| `grain` + `etils[epath,epy]`, `xarray`, `absl-py`, `array-record`, `cloudpickle`, `portpicker`, `protobuf` | core `torchtitan/components/data/loader.py` imports `grain.python`; the rest are grain's own deps |
| `tyro` (pulls `typeguard`) | config parsing -- `config_manager.parse_args()` |
| `tensorboard` | `torchtitan/components/metrics.py` |
| `spmd_types` | sharding specs (`spmd.S(n)`, `spmd.R`) |
| `blendcorpus` @ `feat/remove-deepspeed` | the production dataloader; that branch because the RC has no deepspeed |
| `wandb` | the RC base ships a `wandb` that is importable but has **no `.api`**, which `ezpz.verify_wandb()` calls. Installing a real one into the overlay shadows it |
| `ezpz` from **git** | PyPI `ezpz` is an unrelated package (v0.1.2) |

#### Traps, all of which report success

Every one of these cost a job. They share a failure mode: the command exits 0
or prints nothing alarming, and you only find out at the next stage.

1. **`module load ... | tail -3`** runs the load in a subshell, which silently
   evaporates. Python stays 3.6.15 and every import fails with a misleading
   `No module named 'torch'`. Load at top level.
2. **`uvi` and `ezpz` are interactive-shell only.** `uvi` is an atuin alias
   (`uv pip install --no-cache --link-mode=copy`); neither is defined in a PBS
   batch shell, where you get `command not found` and rc 127. Use `uv pip`
   directly.
3. **`uv pip` from a login node** against the overlay venv fails with
   `No virtual environment found` -- the venv's interpreter symlinks into
   `/opt/aurora/26.181.0`, which is not there -- and **still exits 0**. Install
   on the node.
4. **Neither blanket install policy works.** `--no-deps` on everything skips
   legitimate pure-python deps -- chasing them one failed job at a time cost
   four runs (`etils`, `xarray`, `portpicker`, `typeguard`). But letting the
   resolver run pulled a generic PyPI **`torch-2.13.0`** into the overlay,
   shadowing the RC's XPU build, **even with `-P torch -P pytorch-triton-xpu`
   pins**: all 14 installs printed `ok` and only an explicit
   `torch.__version__` assertion caught it (job `8784535`). Use the resolver
   for leaf packages, `--no-deps` for anything that declares torch, and assert
   on the torch version afterwards.
5. **Evicting a stray torch is not enough -- take `triton` with it.** The same
   pull brings a generic `triton-3.7.1` that shadows the RC's Intel-enabled
   build; removing torch alone leaves `config_registry` dying on
   `No module named 'triton.backends.intel'` (job `8784572`).
6. **`import torchtitan.experiments.ezpz.agpt` is not a readiness gate** -- it
   succeeds while `config_registry` is still broken, because
   `config/manager.py` catches every exception and re-raises one generic
   `Cannot import config_registry`. Gate on `config_registry` itself, with
   `PYTHONPATH` unset from a neutral cwd, which is what ranks see.
7. **Rank errors are not on stdout.** A launch that prints only
   `Execution finished with 143` usually has the real traceback in
   `logs/<module>/<timestamp>-rank0.jsonl`.

#### Status

| Check | Result |
|---|---|
| torch / XPU / XCCL on the RC | **PASS** -- `2.13.0a0+gitcf30153`, 12 devices, xccl True |
| compiled SDPA fwd+bwd | **PASS** (job `8781129`, 1N) |
| all deps import in the overlay | **PASS** (job `8784615`) |
| `config_registry` | **PASS** (job `8784615`) |
| 2N collectives (`ezpz.examples.test`) | **PASS** -- rc=0, 24 ranks, 268s (job `8784615`) |
| **2N x 12 distributed training** | **PASS** -- 10/10 steps, loss 10.865 -> 9.059 (job `8784615`) |
| W&B from a real run | **PASS** -- `torchtitan.ezpz.train/runs/70fumwgc` |
| **agpt TP=2, eager** | **PASS** -- 10.88838 -> 10.43820 (job `8789506`) |
| **agpt TP=2, COMPILED** | **PASS** -- the torch-2.13 `DeviceMesh` AOT assertion does NOT fire on the RC |
| **moe TP=1** | **PASS** -- 12.93609 -> 11.32376 |
| **moe TP=2** | **PASS** -- 12.94930 -> 11.49335, with the `6e4e1996f` wrapper fix |

Coverage matrix run: [`scripts/rc-matrix.sh`](../../scripts/rc-matrix.sh), 2N x 4
ranks, 5 steps each, all five arms monotonic with finite grad norms.

Two results worth calling out:

**Compiled agpt at TP=2 works on the RC.** On the production `.venv` stack
(torch `2.13.0.dev20260519+xpu`) that corner dies in
`tensors_saved_with_vc_check` with a `DeviceMesh` in saved-for-backward -- the
assertion documented for the 80B family, which also fires at debugmodel scale.
It does not fire here. Compiled and eager agree to `3.6e-4` over five steps
(10.88838 vs 10.88835 at step 1), so this is a real pass and not a silently
different code path. **If the RC ships, the `compile=OFF` workaround can be
retired** -- but confirm at 80B before acting on it, since that is where the
assertion was originally characterised.

**The moe TP>1 fix holds on the RC.** Both moe arms reproduce the
production-stack losses (job `8787243`) to the five decimals stdout prints,
across a different torch build and oneCCL. That is the reported precision, not
a bitwise check. The fix is a reshape, so version-independence is what you
would expect.

Full trajectory, `agpt_debugmodel`, 2 nodes x 12 ranks:

```
step:  1  loss: 10.86492  grad_norm: 0.5543
step:  5  loss: 10.50758  grad_norm: 0.7077
step: 10  loss:  9.05897  grad_norm: 1.0462
```

Monotonic descent, finite grad norms throughout. MFU is ~0.05% because
`agpt_debugmodel` at `max_context_length=512` is a plumbing check, not a
performance measurement -- do not read a throughput number off this run.

Note the 2B / MoE / 80B results in
[`docs/guides/frameworks-rc-validation.md`](../guides/frameworks-rc-validation.md)
are **Sunspot** (`1247xxxx`), a different machine. The rows above are the
first training on Aurora's RC.

> [!TIP]
> To suppress the `UserWarning: Torchinductor` error seen when using
> `--compile.enable` on Aurora:
> ```bash
> export SYCL_DISABLE_FSYCL_SYCLHPP_WARNING=1
> ```
