# Trying a newer XPU torch against `spmd_types` (2026-08-20)

**Outcome: CONFIRMED.** A newer XPU torch fixes `spmd_types`. Same venv, same
config, same command -- only torch differs:

| torch | backend | steps | result |
|---|---|---|---|
| **2.14.0.dev20260722+xpu** | `spmd_types` | **3/3** | **PASS** |
| 2.14.0.dev20260722+xpu | `partial_dtensor` | 3/3 | PASS |
| 2.13.0a0+gitcf30153 (shipped) | `spmd_types` | 0/3 | `params-not-DTensors` |

Job 12473463. The version hypothesis is proven: `spmd_types` needs the FSDP
consumer added by pytorch `da19cbd78` (2026-06-23), which our build lacks.

## How to reproduce the working env

`venvs/rc-plus-nightly` is a byte copy of `venvs/fw-2026.1-rc2` with the
nightly installed INTO it. Three things had to be right:

1. **Clone the working venv, do not build one.** The working venv has
   `include-system-site-packages = true` and inherits torch from the RC4
   conda env; a from-scratch venv loses the launcher wiring and dies with
   mpiexec `error parsing parameters`. Cloning keeps ezpz 0.24.3 and every
   dep intact.
2. **Repoint the console-script shebangs** (45 of them) at the clone's
   python, or every launched command silently runs the old venv.
3. **`pip install --ignore-installed`**, since pip otherwise sees the
   inherited conda torch as already satisfying the requirement and no-ops.

## Picking the nightly: three constraints, not one

Not every nightly works. The usable window is narrow:

| nightly | imports | consumer | inductor | |
|---|---|---|---|---|
| dev20260701 | yes | yes | broken | no |
| dev20260708 | yes | yes | broken | no |
| dev20260715 | yes | yes | broken | no |
| **dev20260722** | **yes** | **yes** | **ok** | **USABLE** |
| dev20260729+ | -- | yes | -- | needs `libpti_view.so.1` |
| dev20260820 | -- | yes | -- | needs `libpti_view.so.1` |

- **Too old** (before 2026-06-23): no FSDP consumer, same failure as ours.
- **Too new** (2026-07-29 onward): links `libpti_view.so.1`, and this system
  has only `.so.0` (checked everywhere under `/opt/aurora`). `libtorch_cpu.so`
  hard-links it, so torch will not import at all.
- **Mid-July**: an inductor regression -- `torch._inductor` imports
  `tensorssa_reduction` from one of its own modules that lacks it.

The old "no newer torch on Aurora" blocker (nightlies need `libsycl.so.9`,
oneAPI 2025.3.1 had only `.so.8`) is genuinely stale -- the RC4 stack is
oneAPI 2026.1.0 and `/opt/aurora/26.181.0/.../libsycl.so.9` exists.

## What is now known good

**The old "no newer torch on Aurora" blocker is gone.** That finding was
against oneAPI 2025.3.1, which shipped only `libsycl.so.8` while the
nightlies need `.so.9`. The RC4 stack is oneAPI 2026.1.0 and
`/opt/aurora/26.181.0/oneapi/compiler/latest/lib/libsycl.so.9` exists.

**A current XPU nightly installs and sees the hardware:**

```
torch      : 2.15.0.dev20260820+xpu
xpu avail  : True
xpu count  : 12
```

Built into `venvs/xpu-nightly-test` from the RC venv's python 3.12.12, with
no `--system-site-packages` so it resolves its own torch.

**It contains the FSDP consumer our build lacks:**

| symbol | our 2.13 | nightly 2.15 | pytorch main |
|---|---:|---:|---:|
| `_is_spmd_types_available` | 0 | 2 | 2 |
| `_resolve_spmd_types_for_storage` | 0 | 2 | 2 |
| `get_local_type` | 0 | 1 | 1 |
| `_fsdp_param.py` lines | 1095 | 1329 | 1373 |

## Why the FIRST attempts were inconclusive (kept: the traps are reusable)

Two separate scaffolding failures, neither about torch:

1. **Hand-rolled FSDP test ran unannotated.** I called `fully_shard` on a bare
   `nn.Linear`, then on one I tried to annotate with
   `spmd.MeshAxisName...` -- which is not the API (`AttributeError: module
   'spmd_types' has no attribute 'MeshAxisName'`). The consumer branch needs
   a truthy `spmd.get_local_type(param)`, so BOTH torches raised for a reason
   unrelated to their version. Those two runs prove nothing.

2. **The nightly venv cannot launch the real trainer.** `ezpz launch` and a
   direct `mpiexec` both fail with `error parsing parameters` from this venv,
   while the same invocation works from `fw-2026.1-rc2`. Something in the
   nightly venv's launcher/MPI wiring differs; I did not chase it.

A fourth trap, found later: my nightly-selection probe ran on the LOGIN
node, where `libsycl.so.9` is not on the library path, so `import torch`
failed and every candidate scored "inductor broken". All five verdicts were
false. Nightly selection has to run inside the job.

## What the next attempt should do

Skip the synthetic test. The trainer produces correctly annotated params by
construction, so the only thing needed is to make the nightly venv launch:

- diff the working venv's `mpiexec`/PALS environment against the nightly
  venv's (the failure is in argument parsing, so likely a wrapper or an env
  var, not torch)
- or install the nightly INTO a copy of the working venv rather than building
  one from scratch, so the launcher wiring is inherited intact

Deps that had to be added by hand to the fresh venv, for reference: `click`,
`omegaconf`, plus the full ezpz dependency closure. Installing ezpz
non-editable with a `--constraint torch==<pinned>` file resolves them all and
provably does not swap torch (verified before/after each install).

## Guardrail that held

Every install used a hard torch constraint, and torch was verified after each
one. It stayed `2.15.0.dev20260820+xpu` throughout -- pip never substituted a
CUDA build, which is the standing risk with any `pip install` near torch on
this stack.
