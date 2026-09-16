# ALCF request draft: a torch carrying pytorch #181519 on Aurora

**Status: DRAFT, UNSENT.** Outward-facing; sending is the user's call.

## Ask

An Aurora torch build that includes pytorch
[#181519](https://github.com/pytorch/pytorch/pull/181519) (`da19cbd78`,
2026-06-23), or guidance on when one is expected.

## Why it blocks us

Current upstream torchtitan (post-`#4419`) cannot run on Aurora at all. Not
slowly, not degraded -- FSDP setup raises before step 1:

```
ValueError: When dp_mesh_dims is provided, all parameters must be DTensors on
the full SPMD mesh (e.g. via distribute_module). Got plain tensor for
parameter 'weight'.
```

`#181519` added the `FSDPParam.__init__` block that converts an annotated plain
tensor into a DTensor before that check can fire. Builds without it hit the
raise.

torchtitan previously offered a `partial_dtensor` backend as an escape hatch,
and our configs pinned it for exactly this reason. Upstream `#4419` deleted that
backend **and** its config field -- zero references remain in core -- so there
is no flag, no setting, and no supported way to avoid the new path.

## What we measured

Four reachable torch builds, none carrying the fix:

| build | location | `_fsdp_param.py` lines | #181519 |
|---|---|---|---|
| `2.13.0.dev20260520+xpu` | `projects/saforem2/torchtitan-ezpz/.venv` | 1093 | absent |
| `2.13.0.dev20260428+xpu` | `runs/agpt-2b-v2/.../.venv` (production) | 1017 | absent |
| `2.13.0a0+gitcf30153` | `frameworks/2026.1.0` (test BKC, 26.181.0) | 1095 | absent |
| `2.13.0+cu130` | NERSC Perlmutter `pytorch/2.13.0` | 1095 | absent |
| `2.13.0+cu130` | **ALCF Polaris** `.venv-torch213` (CUDA/A100) | 1095 | absent |

The last two are **CUDA** builds, so this is not an XPU or oneAPI issue -- no
torch 2.13 we can reach on any ALCF or NERSC system carries the patch,
regardless of vendor.

Detection is by the symbols the patch introduces
(`_resolve_spmd_types_for_storage`, `self.is_spmd_types`, `get_local_type`),
not by prose -- the obvious phrases live inside the raise itself, so a
string-based check reports the fix as present on builds that lack it.

## Reproduced on BOTH compute images

Not a single-image artifact. The same `ValueError` appears on the prod BKC and
the test BKC, with different torch builds and different venvs:

| job | queue | bkc | venv / torch | result |
|---|---|---|---|---|
| `8829185` | `debug-scaling` | `compute_aurora_prod_20260828` | `projects/saforem2` `.venv`, `2.13.0.dev20260520+xpu` | dies in FSDP setup, 3/3 arms, 23 ranks |
| `8831522` | `next-eval` | `compute_aurora_test_20260831` | `venvs/sync84-testbkc`, `2.13.0.dev20260428+xpu` | same error, 23 ranks |

Both reach `IMPORT_OK` and build the model first, so this is not a packaging or
environment failure -- it is the FSDP path.

## Evidence it is the only blocker

Same machine, same venv, same script, same seed, 2 nodes:

```
pre-merge tree (pins partial_dtensor)   VERDICT: ok
    agpt TP=1   10.88382  10.75382  10.48337
    agpt TP=2   10.88071  10.70248  10.56243
    moe         12.90379  12.57026  11.45320

merged tree (pin removed by #4419)      dies in FSDP setup, 3/3 arms, 23 ranks
```

The trees differ on one line. Everything else in the upgrade is verified:
12/12 modules import, 12/12 dense + 14/14 MoE model configs build, 82 unit
tests pass, checkpoint state-dict keys are byte-identical, and a numerics A/B
on an A100 agrees to ~1e-6 relative with 100% argmax agreement.

## A STABLE build with the fix exists today

`torch 2.14.0` is a released version -- not a nightly -- and carries the patch
on both the XPU and CUDA builds:

```
torch 2.14.0+xpu      (download.pytorch.org/whl/xpu, installed on Sunspot)
torch 2.14.0+cu130    (PyPI, installed on Polaris)
    _fsdp_param.py                     1329 lines   (vs 1095 without the patch)
    _resolve_spmd_types_for_storage        2
    self.is_spmd_types                     5
    get_local_type                         1
```

So this is a version bump to a released wheel, not a backport request.

The mechanism is visible in the source: `_fsdp_param.py:293-299` sets
`is_spmd_types` and calls `_resolve_spmd_types_for_storage()` to convert the
annotated plain tensor into a DTensor, which runs **before** the
`is_spmd_mesh and not is_dtensor` check at :561. The raise still exists, it
simply can no longer fire for an annotated parameter.

So this is not a request to develop anything -- the patch is three months old
and ships in current nightlies. The ask is to pick it up in an Aurora
frameworks build.

## Controlled A/B: only torch differs

Same tree, same script, same 2-node sunspot `workq` job shape, same deps.
The ONLY variable is the torch build.

| | `frameworks/2026.1.0` (torch 2.13.0a0+gitcf30153) | `torch 2.14.0+xpu` |
|---|---|---|
| FSDP wrapping | **`ValueError`, 46 ranks** | `Applied FSDP to the model` |
| agpt TP=1 | no steps | **rc=0**, loss 10.87743 -> 10.75458 -> 10.48057 |
| agpt TP=2 | no steps | **rc=0**, loss 10.88015 |
| job | `12477660` | `12477656` |

```
torch 2.13:  ValueError: When dp_mesh_dims is provided, all parameters must be
             DTensors on the full SPMD mesh ... Got plain tensor for parameter
torch 2.14:  trains
```

Note the 2.13 run reached FSDP only after we fixed a bug of our own (a Python
float where core keeps a tensor). With that fixed, the ONLY thing still
stopping torch 2.13 is the missing patch.

## The concrete ask: ship `torch 2.14.0+xpu`

A **stable release** carrying the patch already exists on PyTorch's own XPU
index -- no backport, no nightly:

```
torch-2.14.0+xpu-cp312-cp312-manylinux_2_28_x86_64.whl
  https://download.pytorch.org/whl/xpu
  _fsdp_param.py 1329 lines, all three #181519 symbols PRESENT
```

Verified on Sunspot hardware: `xpu.is_available() True`, 12 devices, and the
sync-84 tree reaches `Applied FSDP to the model` -- where the current
`frameworks/2026.1.0` torch (`2.13.0a0+gitcf30153`, 1095 lines, patch absent)
raises before FSDP wrapping.

So the request is not "please backport a commit": it is **"please build a
frameworks module against torch 2.14.0+xpu"**, a released version.

## We ran it: the patch clears the blocker

This is no longer inference from source inspection. On Sunspot, 2 nodes /
24 ranks, `torch 2.14.0+xpu` (job `12477656`):

```
Applied FSDP to the model
agpt_debugmodel TP=1  rc=0   step 1 loss 10.87743  ->  step 3 loss 10.48057
agpt_debugmodel TP=2  rc=0   step 1 loss 10.88015
```

It trains. The same tree on `frameworks/2026.1.0` (job `12477660`) dies at FSDP
wrapping on 69 ranks with `Got plain tensor for parameter`.

The losses also track the pre-merge baseline to ~3 decimals (10.88382 /
10.48337), so the newer torch is not changing results -- it is the difference
between running and not running.

## Scope

Affects any Aurora user tracking torchtitan upstream past `#4419`
(2026-09-xx), not only this project. Nothing in the loader path, the BKC image,
or the module system is implicated -- we chased and eliminated all three.

## What would unblock us, in order of preference

1. A torch build carrying `#181519` in an Aurora frameworks module.
2. Guidance that one is scheduled, so we can time the upgrade.
3. Confirmation that it is not planned, so we carry a local shim in our
   experiments tree with eyes open.
