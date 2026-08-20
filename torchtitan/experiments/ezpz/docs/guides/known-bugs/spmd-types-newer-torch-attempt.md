# Trying a newer XPU torch against `spmd_types` (2026-08-20)

**Outcome: INCONCLUSIVE.** The nightly installs and runs, but I did not get a
clean A/B. Recording it so the next attempt starts from the working parts
instead of rediscovering them.

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

## Why it is inconclusive

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

So the honest status: **the version hypothesis is untested, not disproven.**

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
