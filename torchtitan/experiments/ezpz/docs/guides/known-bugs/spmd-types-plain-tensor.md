# Why `spmd_types` leaves parameters unconverted

**Status:** UPSTREAM BUG. Reproduces on core llama3 with no ezpz code involved.
**Affects:** `parallelism.spmd_backend = "spmd_types"` -- which is upstream's
DEFAULT as of #4085 (`5ab3a0fd1`, 2026-08-18).
**Jobs:** 12473427-12473433

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

Measured live (job 12473430) by wrapping `spmd_distribute_tensor`:

```
[MESH] NOT-CONVERTED shape=(256128, 5120) shard_axes=['tp']
       mesh_dims=('dp','cp','tp') sizes=[1, 1, 1]
```

The mesh at `parallelize` time is all-ones on every axis, so nothing shards.
`dp` is legitimately 1 there -- FSDP shards that axis later -- but the
function has no "otherwise replicate" branch, so the result is a plain tensor
rather than a DTensor with `Replicate()` placements.

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
