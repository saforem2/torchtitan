# DRAFT upstream issue: `spmd_types` yields plain tensors when every Shard axis is size 1

> [!CAUTION]
> **DO NOT FILE. This draft is wrong.** Kept only as a record of the
> investigation.
>
> Resolved 2026-08-20: the plain tensor is not a torchtitan defect. Handing
> FSDP a plain-but-ANNOTATED tensor is what upstream's design expects --
> pytorch `da19cbd78` (#181519, 2026-06-23) added the code in
> `FSDPParam.__init__` that converts it. **Our torch build does not contain
> that commit** (`_resolve_spmd_types_for_storage`, `get_local_type`, and
> `_is_spmd_types_available` all appear 0 times in our `_fsdp_param.py`;
> pytorch main has all three).
>
> So upstream CI is green because their torch has the consumer, and every
> "suggested fix" below -- wrapping in `spmd_distribute_tensor`, making the
> `resolve_fsdp_mesh` guard per-parameter -- would be patching the wrong
> layer. `spmd_types` needs a newer torch, not a torchtitan change.
>
> See [spmd-types-plain-tensor.md](../guides/known-bugs/spmd-types-plain-tensor.md).

Original draft follows, unedited.

---

## Title

`spmd_types`: `spmd_distribute_tensor` returns a plain tensor when every declared Shard axis has size 1, and FSDP then rejects it

## Body

### Summary

With `parallelism.spmd_backend = "spmd_types"` (the default since #4085), FSDP
fails at init:

```
ValueError: When dp_mesh_dims is provided, all parameters must be DTensors
on the full SPMD mesh (e.g. via distribute_module).
Got plain tensor for parameter 'weight'.
  torch/distributed/fsdp/_fully_shard/_fsdp_param.py:345
```

`model.parallelize()` runs and `_spmd_distribute_state` is invoked for the
parameter, but the parameter is still a plain `torch.Tensor` afterwards. The
conversion silently no-ops.

### Root cause

`torchtitan/distributed/spmd_types.py:353`, `spmd_distribute_tensor`:

```python
shard_types = layout.per_axis_spmd_types()
if layout.partition_spec is None:
    axis_shard_dims = [
        (axis_name, axis_type.dim)
        for axis_name, axis_type in shard_types.items()
        if isinstance(axis_type, spmd.Shard)     # Replicate axes excluded
    ]
...
for axis_name, dim in axis_shard_dims:
    axis_size = mesh.size(...) if axis in mesh.mesh_dim_names else 1
    if axis_size > 1:
        tensor = spmd.shard(tensor, mesh.get_group(axis), src=spmd.I, dst=spmd.S(dim))
return tensor        # <-- unchanged input if the body never ran
```

Two properties combine:

1. `axis_shard_dims` contains only `Shard` axes; `Replicate` axes never enter
   the loop.
2. The body is guarded by `if axis_size > 1`.

So for any parameter whose declared Shard axes are all size 1 at
`parallelize` time, the function returns its input untouched -- no DTensor,
no error. There is no "otherwise, wrap as Replicate" branch.

The caller (`torchtitan/protocols/module.py:313`) registers the result
directly, so the module keeps a plain tensor, and FSDP's SPMD check
(`is_spmd_mesh and not is_dtensor`) then raises.

### Reproduction

Any Decoder model at TP=1. `tok_embeddings.weight` is declared
`dp:Replicate, cp:Replicate, tp:Shard(0)` by
`decoder_sharding.set_decoder_sharding_config`, so its only Shard axis is
`tp`. At TP=1 nothing shards.

```
torchtitan/experiments/ezpz/train.py --module=<any decoder model> \
  --parallelism.spmd-backend=spmd_types \
  --parallelism.tensor-parallel-degree=1
```

Measured at 12 ranks:

```
world=12 dp_shard=12 dp_replicate=1 tp=1 cp=1
spmd_dense_mesh dims=('dp','cp','tp') sizes=[12, 1, 1]
-> Got plain tensor for parameter 'weight'
```

`dp` is 12, but it is declared `Replicate` for this parameter, so it is not
in `axis_shard_dims` and does not save the conversion.

### Clean A/B on core llama3, core trainer, no downstream code

`torchtitan.train --module llama3 --config llama3_debugmodel`, FSDP-only
(TP=1), 8 ranks, same binary and launcher, changing ONLY the backend
(job 12473444):

| backend | steps | result |
|---|---|---|
| `partial_dtensor` | 8 | **PASS** |
| `spmd_types` | 0 | `Got plain tensor for parameter 'weight'` |

This is upstream's own default model config in its FSDP-only shape -- the
same shape `tests/integration_tests/run_tests.py` uses by default
(`--module` defaults to `llama3_debugmodel`) and that `features.py` covers
with tests like `1d_compile` and `fsdp_reshard_always`.

### Suggested fix

`spmd_distribute_tensor` should return a DTensor in all cases -- wrapping
with `Replicate()` on axes that are size 1 or declared Replicate -- rather
than returning the bare input when no shard applies. Alternatively, the
caller could wrap when the result is not already a DTensor.

### Why it may not have shown up in CI

- `tests/integration_tests/h100.py:45` overrides to
  `--parallelism.spmd_backend full_dtensor`
- `b64d3f6a9` pins the rl+hf tests to `partial_dtensor`
- #4085, which made `spmd_types` the default, touched deepseek_v3 /
  kimi_k2_7 / qwen3_5 but not llama3

Note the `models.py` suite is entirely `+tp` / `+ep` -- every entry
(`deepseek_v3_hsdp+ep`, `qwen3_5_moe_fsdp+tp+ep+pp`,
`muse_glimmer_mm_fsdp+tp+sp`, ...) has a live shard axis, so
`tok_embeddings.weight` converts there. The FSDP-only coverage lives in
`features.py`, which is where this should surface.

Also checked and NOT the explanation:

- **Activation checkpointing.** `_configure_spmd_backend_and_typechecking`
  appends `activation-checkpoint:none` to spmd_types runs, so I tested AC
  none / full / selective at TP=1 and TP=2 (job 12473441). All five arms fail
  identically. AC is irrelevant here.
- **A newer PyTorch.** The raise is at
  `torch/distributed/fsdp/_fully_shard/_fsdp_param.py`, introduced by
  pytorch `da19cbd78` (#181519, 2026-06-23). **pytorch main still contains
  the identical check** (verified against the current file on main), and the
  only commit touching that file since our 2026-08-02 build is `f0152d66f`
  (#194114, all-gather output release), which is unrelated. So this is not
  something a torch nightly fixes -- the check is deliberate, and the
  expectation is that torchtitan hands FSDP DTensors.

---

## Upstream already knows about this shape -- the guard is just too coarse

`torchtitan/distributed/fsdp.py`, `resolve_fsdp_mesh` (upstream/main):

```python
if storage_mesh.size() == 1:
    # ``assert_type`` filters out inactive size-1 axes, so params get no
    # annotations under a size-1 full mesh. That leaves ``fully_shard()``
    # with no SPMD annotations to translate to DTensor params, so do not
    # pass a DataParallelMeshDims object to FSDP.
    return storage_mesh, None
```

That comment describes exactly this failure. But the guard triggers only when
the WHOLE dense storage mesh (`dp_replicate, dp_shard, cp, tp`) is size 1. At
TP=1 with FSDP>1 the mesh is size 12, so `DataParallelMeshDims` IS passed --
while a parameter whose only non-Replicate axis is `tp` still ends up with no
surviving annotation.

Suggested refinement: the check wants to be per-parameter (does this param
retain any annotation on the live mesh?) rather than per-mesh.

Correspondingly, the proximate trigger is more precisely the missing
**annotation**, not the missing shard. In
`torch/.../fsdp/_fully_shard/_fsdp_param.py`, a plain-but-annotated tensor is
still converted (`spmd.get_local_type(param)` -> `_resolve_spmd_types_for_storage`);
the raise fires only when the param is plain AND carries no annotation.

## Checked and ruled out

- **A newer `spmd_types` package.** Our venv had `spmd_types==0.2.1` while
  `requirements.txt` (ours and upstream's) pins `0.2.3` -- a real divergence,
  and a good candidate since that package owns `assert_type`. Tested by
  installing 0.2.3 into a throwaway overlay: **fails identically** (job
  12473448). Not the cause. (Worth fixing the venv anyway.)
- **Activation checkpointing.** none / full / selective x TP=1 / TP=2, all
  five arms fail identically (12473441).
- **A torch nightly.** pytorch main still contains the same check; the only
  commit touching that file since our build is unrelated.
- **Unmerged upstream fixes.** No open PR touches `spmd_distribute_tensor`,
  `spmd_types.py`, or `_spmd_distribute_state`. No open issue matches this
  error. PR #4085 contains no statement that spmd_types requires TP>1 or
  AC-off.

## Related upstream movement worth knowing

- **`full_dtensor` is being REMOVED** (`601cf4d23`, #4217, 2026-08-19).
  `partial_dtensor` is the supported fallback. Our agpt configs are currently
  pinned to `full_dtensor`, which is a dead end -- that pin should move to
  `partial_dtensor`.
- `b64d3f6a9` (#4228) pins the rl+hf CI suites to `partial_dtensor`.
- `801fe175f` (#3913, 2026-08-11) "fix spmd->DTensor translation on partial
  mesh" is the same family of size-1-axis bug, fixed earlier.

## Confidence, stated honestly

**Solid:**
- The failure is real, reproducible, and reproduces on stock core llama3 with
  no ezpz code (job 12473433).
- The runtime mesh is `('dp','cp','tp') = [12,1,1]`, measured at 12 ranks
  (12473437).
- `parallelize` and `_spmd_distribute_state` DO fire on the parameter, and it
  is still not a DTensor afterwards (12473429). So the conversion is
  attempted and no-ops -- this is not a missing declaration or a skipped walk.
- The code path above is short and unambiguous on inspection.

**Unresolved, and a maintainer should be told:** upstream's own CI defaults
`--module` to `llama3_debugmodel` and `features.py` carries FSDP-only tests
(`1d_compile`, `fsdp_reshard_always`) that get `spmd_types` applied -- which
is the exact shape that fails here. If those are green upstream, something in
our environment differs and this may not be a universal bug. I could not
determine what from the sources; the A/B above is the strongest evidence I
have that it is not our model code.

**Not proven:**
- I never got a standalone unit-level repro calling `spmd_distribute_tensor`
  directly. Four attempts failed for harness reasons (unlaunched single
  process; patched the wrong namespace; an import-order assert; and finally
  `init_device_mesh` needing backend splitting this CCL build does not
  support). The mechanism is therefore *inferred from code inspection plus
  the observed no-op*, not demonstrated in isolation.
- One earlier draft of the local writeup claimed the mesh was all-ones. That
  was wrong -- an artifact of probing without a launcher -- and it is
  corrected. Worth knowing that this analysis has already been wrong once.

If a maintainer wants the isolated repro before acting, it should be trivial
on CUDA where `init_device_mesh` splitting works.
