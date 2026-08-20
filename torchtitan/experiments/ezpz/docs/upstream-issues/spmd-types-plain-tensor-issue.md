# DRAFT upstream issue: `spmd_types` yields plain tensors when every Shard axis is size 1

Ready to file, pending your review. Not filed -- I do not have your GitHub
account, and the confidence caveats at the bottom are yours to weigh.

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

Reproduced on stock `llama3 debugmodel` with no custom model code, so this is
not specific to a downstream model.

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

TP>1 configurations would not hit it for `tok_embeddings` (the `tp` axis is
then live), which may be why it survives the sharded test paths.

---

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
