# Why `spmd_types` leaves parameters unconverted

**Status:** RESOLVED (2026-08-20). Not a torchtitan bug -- our PyTorch build
predates the FSDP code that consumes spmd_types annotations. Nothing to fix in
torchtitan or ezpz; needs a torch newer than `pytorch_2.13.0_patched_08_02_2026`.
**Affects:** `parallelism.spmd_backend = "spmd_types"` -- which is upstream's
DEFAULT as of #4085 (`5ab3a0fd1`, 2026-08-18).
**Jobs:** 12473427-12473433

## ROOT CAUSE: our torch predates the FSDP consumer (2026-08-20)

Everything below traced torchtitan's side and concluded the plain tensor was
the defect. It is not -- handing FSDP a plain-but-ANNOTATED tensor is exactly
what upstream's design expects. The missing piece is on the PyTorch side.

pytorch `da19cbd78` (#181519, 2026-06-23) added to `FSDPParam.__init__`:

```python
self.is_spmd_types = (
    dist._is_spmd_types_available()
    and bool(spmd_local_type := spmd.get_local_type(param))
    and not isinstance(param, DTensor)
)
if self.is_spmd_types:
    param = self._resolve_spmd_types_for_storage(...)   # plain -> DTensor
```

That block converts the annotated tensor into a DTensor *before* the
`is_spmd_mesh and not is_dtensor` check can fire.

Our build does not have it:

| symbol | our torch | pytorch main |
|---|---:|---:|
| `_is_spmd_types_available` | 0 | 2 |
| `_resolve_spmd_types_for_storage` | 0 | 2 |
| `get_local_type` | 0 | 1 |
| `_fsdp_param.py` length | 1095 | 1373 |

Our `FSDPParam.__init__` goes straight from `self.grad_offload_event = None`
to `self._init_sharded_param(...)` with no `is_spmd_types` branch at all, so
an annotated plain tensor falls through to the raise.

Confusingly, `torch.distributed._is_spmd_types_available` DOES exist in our
build (`distributed/__init__.py:31`) -- the availability hook shipped, the
FSDP consumer did not. So "the symbol exists" is not evidence the path does.

**Consequence:** `spmd_types` cannot work on the shipped stack regardless of
what torchtitan or ezpz do. It needs a torch containing `da19cbd78`.

**PROVEN 2026-08-20** (job 12473463): with `torch 2.14.0.dev20260722+xpu` in
an otherwise identical venv, `spmd_types` runs **3/3** where the shipped
2.13 gives 0/3 with `params-not-DTensors`. See
[spmd-types-newer-torch-attempt.md](spmd-types-newer-torch-attempt.md) for
the working recipe and the narrow window of usable nightlies.

Note our build is named `pytorch_2.13.0_patched_08_02_2026` -- Aug 2, which
should postdate a Jun 23 commit -- so the patched Aurora build is presumably
branched from an older base than its name suggests. Worth raising with
whoever builds it, since it likely affects more than this one path.

**Nothing to file upstream.** The draft issue at
`docs/upstream-issues/spmd-types-plain-tensor-issue.md` should NOT be filed:
it argues torchtitan should wrap the tensor, but upstream deliberately leaves
that to FSDP, and their CI is green because their torch has the consumer.

## Symptom

```
ValueError: When dp_mesh_dims is provided, all parameters must be DTensors
on the full SPMD mesh (e.g. via distribute_module).
Got plain tensor for parameter 'weight'.
  torch/distributed/fsdp/_fully_shard/_fsdp_param.py:345
```

Raised by torch's FSDP, not by torchtitan. Two ways FSDP can shard:

- **legacy:** params are plain tensors; FSDP shards them itself along one DP
  axis and ignores the rest of the mesh.
- **SPMD (`spmd_types` / `full_dtensor`):** params arrive already wrapped as
  DTensors carrying their full-mesh placement, and FSDP reads that placement
  instead of imposing its own.

Under SPMD, FSDP is handed a `DataParallelMeshDims` and checks
`if self.mesh_info.is_spmd_mesh and not self.is_dtensor: raise`. A bare
tensor gives it no placement to read, so it refuses rather than guessing.

## Root cause

`torchtitan/distributed/spmd_types.py:353` `spmd_distribute_tensor`:

```python
for axis_name, dim in axis_shard_dims:
    axis_size = mesh.size(...) if axis in mesh.mesh_dim_names else 1
    if axis_size > 1:
        tensor = spmd.shard(tensor, mesh.get_group(axis), src=spmd.I, dst=spmd.S(dim))
return tensor          # <-- UNCHANGED, still a plain tensor, if nothing fired
```

**If every declared shard axis has size 1, the function returns its input
untouched.** It never wraps the tensor as a DTensor, and it reports no error.
FSDP then rejects the plain tensor it gets back.

Note `axis_shard_dims` is built from `Shard` axes ONLY -- `Replicate` axes
never enter the loop:

```python
axis_shard_dims = [
    (axis_name, axis_type.dim)
    for axis_name, axis_type in shard_types.items()
    if isinstance(axis_type, spmd.Shard)      # <-- Replicate axes excluded
]
```

`tok_embeddings.weight` declares `dp:R, cp:R, tp:S(0)`. Its only Shard axis
is `tp`. At TP=1 that axis has size 1, the loop body never runs, and the bare
input tensor is returned.

Measured at 12 real ranks (job 12473437):

```
[MD] world=12 dp_shard=12 dp_replicate=1 tp=1 cp=1
[MD] spmd_dense_mesh dims=('dp','cp','tp') sizes=[12, 1, 1]
Got plain tensor for parameter 'weight'
```

**Correction to an earlier draft of this page**, which claimed the mesh was
all-ones. It is not -- `dp` is 12. That reading came from a probe run as a
bare `python3 -c` with no launcher, i.e. world_size=1, where every axis is
size 1 by construction. The mechanism survives with the premise fixed: what
matters is that every *Shard* axis is size 1, and `dp=12` is declared
Replicate, so it is irrelevant to this loop.

The function has no "otherwise replicate" branch, so instead of a DTensor
with `Replicate()` on dp/cp and a trivial shard on tp, the caller gets a
plain tensor.

## What it is NOT

Ruled out by instrumenting the chain (jobs 12473427-12473429), because each
of these was a plausible guess:

- **A missing declaration.** `tok_embeddings.sharding_config` IS set, with
  `{'weight': dp:R, cp:R, tp:S(0)}` -- correct for an embedding.
- **The declaration not reaching the module.** The built `Embedding` instance
  carries `_sharding_config`, and it is a direct child of the model, so the
  recursive `parallelize` walk reaches it.
- **`parallelize` skipping it.** It does not. Live probe:
  `Embedding.parallelize fired` and `_spmd_distribute_state on
  Embedding.weight` both print -- and the weight is STILL not a DTensor
  afterwards. Conversion is attempted and silently no-ops.
- **Ordering.** `update_from_config` (trainer.py:347) runs before `build()`
  (367) before `parallelize` (491) before `apply_fsdp`.
- **Something agpt-specific.** See below.

The embedding is not special either -- it is just the first param FSDP
reaches. The same probe shows the FFN weights (14336, 5120) and the norms
coming back unconverted too.

## It is upstream's bug, not ours

Core llama3 debugmodel, no ezpz model code, job 12473433:

| backend | result |
|---|---|
| `spmd_types` | **same `ValueError`, params not DTensors** |

(The other two backends died on an unrelated ezpz-trainer/core-config
mismatch -- `'Config' object has no attribute 'lr_finder'` -- so they are
uninformative here. The `spmd_types` arm got far enough to hit the real
error, which is the point.)

Two corroborating signs upstream knows this path is not ready:

- their own H100 integration test overrides to
  `--parallelism.spmd_backend full_dtensor` (`tests/integration_tests/h100.py:45`)
- `b64d3f6a9` (2026-08-19) pins the rl+hf tests to `partial_dtensor`
- `5ab3a0fd1`, the commit that made `spmd_types` the default, touched
  deepseek_v3 / kimi_k2_7 / qwen3_5 -- but NOT llama3, our base

## Consequences

Do not chase `spmd_types` on the ezpz side; there is nothing to annotate. The
fix belongs in `spmd_distribute_tensor`, which should return a DTensor with
`Replicate()` on the size-1 axes rather than the bare input.

Worth reporting upstream with the one-line repro: any model whose params
declare only size-1 shard axes at `parallelize` time.
