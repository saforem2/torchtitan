# MoE at TP>1: `wo` gets Shard(0) where row-parallel wants Partial(sum)

> **Open.** Found 2026-08-25 while verifying the MLA fold port
> ([`moe-mla-not-ported-to-4121-fold.md`](moe-mla-not-ported-to-4121-fold.md)).
> **Nothing in production is affected**: no agpt chain uses this code and no MoE
> production chain exists. This blocks running MoE at TP>1 at all.

## Symptom

```
ValueError: Linear: output DTensor has placements (Shard(dim=0),),
            but out_src_shardings expects (Partial(sum),).
```

Raised from `moe/model.py:192` -- `return self.wo(output)`, the last line of
the MLA forward -- via `_redistribute_outputs`.

`rowwise_config` declares `out_src_shardings=dense_activation_placement(tp=spmd.P, ...)`:
row-parallel's raw output must be `Partial(sum)`, which is what you get when the
Linear's INPUT is feature-sharded. Ours arrives token-sharded instead.

## What this is NOT

Ruled out by reading the code against upstream `deepseek_v3`, not by guessing:

- **`wo`'s sharding config** -- `rowwise_config(output_sp=enable_sp)`, identical
  to upstream (`deepseek_v3/sharding.py:119`).
- **The MLA tail** -- our `output.view(num_tokens, -1); return self.wo(output)`
  matches upstream line-for-line. The only insert is moe's XPU `pad_v` slice,
  which touches the last dim and cannot change dim-0 placement.
- **local_map argument names** -- the wrapper exposes `q_TNH`/`k_TNH`/`v_TNH`,
  exactly the keys `set_gqa_inner_attention_local_map` matches on.
- **local_map wiring** -- same call, same position as upstream.
- **Sequence parallelism.** Bisected: job `8782843` ran TP=2 with
  `--parallelism.no-enable-sequence-parallel` and produced the IDENTICAL error.
  An earlier theory that the SP path had never executed (moe_debugmodel ships
  `enable_sequence_parallel=True` with `tensor_parallel_degree=1`, so SP is a
  no-op) is therefore wrong.
- **The MLA fold port** (`38a0d595c`). The port is verified at TP=1 and the
  failing line is byte-identical to upstream's.

## The control run was invalid (my mistake)

Job `8782898` ran UNFORKED `deepseek_v3_debugmodel` at TP=2 on the same node
and stack, to test whether the bug is ours or the platform's. It failed
EARLIER and differently:

```
RuntimeError: No backend for the parent process group or its backend
              does not support splitting        (distributed_c10d.py:5568)
```

**That was MY error, not a platform limit.** We already carry a workaround:
`torchtitan/experiments/ezpz/xccl_split_group_workaround.py`, installed by
`trainer.py:898`. It steers `DeviceMesh._init_one_process_group` to the
`new_group` fallback because `ProcessGroupXCCL` inherits
`supportsSplitting() == false`.

The control ran **upstream `deepseek_v3`**, which uses the upstream trainer and
therefore never installs it. So the control was invalid -- it measured the
absence of our workaround, not a property of the stack. Our own moe run shows
zero `split_group` errors and reaches the forward pass, exactly as expected.

**Re-run with the workaround hoisted (`99f3aabe2`), job `8782983`: 0
`split_group` errors -- upstream now clears mesh setup.** It then fails at a
DIFFERENT point, still earlier than ours:

```
ValueError: When dp_mesh_dims is provided, all parameters must be DTensors on
            the full SPMD mesh (e.g. via distribute_module).
            Got plain tensor for parameter 'weight'.
```

Our moe runs (`8782821`, `8782843`) never hit that error -- 0 occurrences --
so the two forks fail at genuinely different stages:

| Run | Fails at |
|---|---|
| ours, TP=2 | `wo` placement, INSIDE the forward |
| upstream dsv3, TP=2 | parameter distribution, BEFORE the forward |

So upstream MoE at TP=2 remains unusable as a comparison on this stack, but for
a second independent reason rather than the platform limit I first claimed.
**Our fork gets further than upstream in both attempts.** Whether the
`dp_mesh_dims` failure is an upstream bug, an XPU gap, or a config mismatch is
not yet established -- it is a separate thread from this page's bug.

## MEASURED (job `8784667`)

Probed the DTensor placement at every step between SDPA and `wo` at TP=2
(`EZPZ_MLA_PLACEMENT_PROBE=1`, rank 0):

```
q (into sdpa)          shape=(512, 16, 192)     placements=(Shard(dim=1),)
k (into sdpa)          shape=(512, 16, 192)     placements=(Shard(dim=1),)
v (into sdpa, pre-pad) shape=(512, 16, 192)     placements=(Shard(dim=1),)
sdpa out               shape=(1, 1024, 8, 192)  placements=(Shard(dim=1),)
after pad_v slice      shape=(1, 1024, 8, 128)  placements=(Shard(dim=1),)
after contiguous       shape=(1, 1024, 8, 128)  placements=(Shard(dim=1),)
after view -> wo in    shape=(512, 2048)        placements=(Shard(dim=0),)
```

Two facts fall out, neither of which any earlier theory predicted:

**1. `Shard(dim=1)` on the way in is CORRECT.**
`attention_activation_placement` declares
`partition_spec=((DP, CP), TP, None)` -- axis 0 is tokens, axis 1 is heads,
and heads are the TP-sharded axis. So the inputs are exactly right, and
`pad_v` / the slice / `contiguous` are all innocent: placement is unchanged
across every one of them.

**2. The SDPA output shape is wrong.** q enters as `(512, 16, 192)` with
`seq_len=512`, so the wrapper's unflatten computes `batch = 512//512 = 1` and
reshapes to `(1, 512, 16, 192)`. The measured output is
**`(1, 1024, 8, 192)`** -- the sequence axis is inflated by exactly the TP
degree while the head axis is halved. The reshape is computed from the GLOBAL
shape but applied to LOCAL storage, so under TP the heads leak into the
sequence axis.

The final `view(num_tokens, -1)` then flattens that mis-shaped tensor and the
placement collapses `Shard(dim=1)` -> `Shard(dim=0)`. That is the value `wo`
rejects. **The `wo` error is a symptom; the defect is upstream of it, in the
unflatten.**

## Why identical code diverges: the attention TYPE

```
moe  attention cfg = Attention.Config        (upstream MLA)
agpt attention cfg = GQAttention.Config
```

Both build the **same** `EzpzScaledDotProductAttention.Config` inner attention,
and both forwards expose the **same** positional names
(`['q_TNH', 'k_TNH', 'v_TNH']`, verified with `inspect`), so the local_map
name-matching contract is satisfied on both sides. The divergence is one layer
up, in the attention block itself.

`set_gqa_attention_sharding` asserts `GQAttention.Config`, so **agpt gets that
whole helper and moe structurally cannot** -- moe hand-rolls the equivalent in
`moe/sharding.py`.

### What was checked and ELIMINATED

- **The hand-rolled sharding is not missing anything.** Diffed assignment for
  assignment against upstream `deepseek_v3/sharding.py`: the same 12 entries in
  the same order (attention block, rope, wkv_a, kv_norm, wkv_b, wo, the
  local_map, wq / wq_a+q_norm+wq_b). Ours adds only a
  `getattr(attention, "rope", None) is not None` guard, and rope is non-None
  here anyway.
- **Positional-arg names match** -- the local_map contract binds by name and
  both are `q_TNH`/`k_TNH`/`v_TNH`.
- **`pad_v` and the output slice are innocent** -- placement is unchanged
  across both (see the probe above).
- **Head count is not it** -- both have `n_heads=16`.

### The one live difference

moe's MLA is **head-dim asymmetric**: `qk_head_dim=192` vs `v_head_dim=128`.
agpt's GQA shares a single head_dim across q/k/v. The unflatten reshapes all
three with the head_dim taken from **q**:

```python
q_TNH = q_TNH.view(batch, seq_len, num_heads, head_dim)
k_TNH = k_TNH.view(batch, seq_len, -1, head_dim)   # same head_dim
v_TNH = v_TNH.view(batch, seq_len, -1, head_dim)   # same head_dim, but v is 128
```

For agpt that is exact. For moe, `v` only survives it because `pad_v` has
already padded v to 192 -- and the `-1` then absorbs any residual mismatch into
the head axis instead of raising.

## The agpt side, measured (job `8785489`)

Ran both arms at TP=2 through `ezpz.launch` directly (no venv yeet), same
flags, same node. **agpt never reached the probe.** It died in `parallelize`:

```
AssertionError: expected all tensors_saved_with_vc_check to be Tensors,
got types: [..., <class 'torch.distributed.device_mesh.DeviceMesh'>]
```

That is the known `compile + AC + TP` AOT-autograd bug on torch 2.13 -- a
`DeviceMesh` leaking into saved-for-backward -- already documented for the 80B
family. `AGPT_RC=139`.

moe, in the same job, reached the probe and reproduced the placement
collapse exactly as before.

### So the premise was wrong

Both of these are true at once, and the difference is the harness:

| how agpt TP=2 is run | result |
|---|---|
| `sync_smoke.sh` arm 2 (jobs `8781696`, `8781776`) | `rc=0` |
| `ezpz.launch` direct, identical flags (`8785489`) | AOT assertion, rc=139 |

`sync_smoke.sh` appends `--debug.seed=42 --debug.deterministic` to every
config; the direct launch does not. That is the only difference in the
invocation.

**"agpt TP=2 passes" was never a statement about agpt.** It is a statement
about agpt under `--debug.deterministic`. Every comparison on this page that
treated agpt as a working TP=2 control needs re-reading with that in mind --
including the head-dim-asymmetry candidate below, which was reasoning from
"identical code, different outcome" when the outcomes were measured under
different conditions.

### What is still true

- moe's placement collapse is measured twice, reproducibly (`8784667`,
  `8785489`).
- The inputs are correctly `Shard(dim=1)`; `pad_v`, the slice and `contiguous`
  leave placement unchanged.
- The SDPA output shape is wrong: sequence inflated by the TP degree, heads
  halved.

## ANSWERED (job `8785537`): agpt gets local tensors, moe gets DTensors

Three arms, same node, same flags, `--debug.deterministic` throughout:

| arm | q entering the unflatten | q leaving it | rc |
|---|---|---|---|
| agpt + det | -- (compile AOT assertion) | -- | 1 |
| **agpt + det + `--compile.no-enable`** | `(512, 8, 16)` **PLAIN** | `(1, 512, 8, 16)` | **0** |
| moe + det | `(512, 16, 192)` **`Shard(dim=1)`** | `(1, 1024, 8, 192)` | 143 |

**agpt's wrapper receives plain local tensors. moe's receives DTensors.**

That is the entire divergence, and it explains every observation:

- agpt sees `8` heads -- already the LOCAL count at tp=2 -- so
  `view(batch, seq_len, num_heads, head_dim)` is exact and the output is the
  expected `(1, 512, 8, 16)`.
- moe sees `16` heads, the GLOBAL count, on a tensor whose local storage holds
  half of it. The reshape is computed from the global shape and applied to
  local storage, so the sequence axis absorbs the missing heads:
  `(1, 1024, 8, 192)` instead of `(1, 512, 16, 192)`.
- The final `view(num_tokens, -1)` then collapses `Shard(dim=1)` ->
  `Shard(dim=0)`, which is what `wo` rejects.

`local_map`'s contract is "convert DTensors to local tensors before the kernel
runs, then wrap outputs back" (`decoder_sharding.py:265`). It is holding for
agpt and **not** for moe.

The traversal in `Module.parallelize` (`protocols/module.py:263-286`) is
unconditional -- it recurses into every child and looks through non-Module
wrappers -- so moe's `inner_attention` IS reached and DOES get a
`LocalMapConfig`. The config is installed; the conversion is not happening.

### So the bug is not the unflatten

The unflatten is correct **given local tensors**, which is what it is
documented to receive and what agpt actually gets. Rewriting it to be
TP-aware would paper over a local_map that is not converting, and would break
agpt, which depends on the current behavior.

Fix the conversion, not the reshape.

### Narrowed to three silent early-returns (2026-08-27, code read)

`local_map` not converting has exactly three exits, and **all three return the
unwrapped forward with no error**:

```python
# protocols/module.py:280   in parallelize()
if self._sharding_config is None:
    return                      # module never parallelized at all

# protocols/module.py:430   in _maybe_wrap_with_local_region()
if sharding_config.local_map is None:
    return fn                   # no local_map on this config

# protocols/module.py:479   in _apply_local_map()
if resolved_mesh is None:
    return fn                   # <-- returns the RAW forward: DTensors flow in
```

The third is the one that matches the measurement: the config IS installed (we
confirmed the traversal is unconditional), yet the wrapper receives DTensors.

`resolve_shared_mesh` returns `None` in two cases -- every entry `None`, or
`resolve_mesh` filtering every axis out. Under `partial_dtensor`,
`resolve_mesh` keeps only `("tp", "ep")`:

```python
in_band = ("dp", "cp", "tp", "ep") if spmd_backend == "spmd_types" else ("tp", "ep")
return self.get_activated_mesh([a for a in axes_list if a in in_band])
```

### Eliminated by reading (do not re-check these)

| hypothesis | why it is dead |
|---|---|
| different spmd_backend | both pin `partial_dtensor` (`{agpt,moe}/config_registry.py`) |
| different lifecycle point | both call their sharding setter from `Config.update_from_config` |
| different object passed | both pass `self` and iterate `config.layers` |
| moe's sharding is incomplete | diffed assignment-for-assignment against upstream `deepseek_v3`: same 12 entries, same order |
| positional-arg names | identical, `['q_TNH','k_TNH','v_TNH']` via `inspect` |
| head-dim asymmetry | agpt head_dim 16 vs moe 192, and agpt reshapes correctly regardless |

### The one live lead

`attention_activation_placement` returns **two structurally different
layouts** depending on its `cp` argument:

```python
if isinstance(cp, spmd.Shard):          # q path: cp defaults to S(0)
    return SpmdLayout({DP: V, CP: V, TP: V},
                      partition_spec=((DP, CP), TP, None))
return SpmdLayout({DP: S(0), CP: cp, TP: S(1)})   # kv path: cp=R, NO partition_spec
```

`set_gqa_inner_attention_local_map` passes `cp=spmd.R` for `kv_dst_placements`
and `cp=spmd.P` for `kv_grad_placements`, so a single boundary mixes a
partition_spec'd layout (q) with two that have none (k/v). `resolve_shared_mesh`
asserts all entries share the same axis keys and then calls `resolve_mesh` on
them -- worth checking whether the no-partition_spec branch resolves to a
different mesh, or to `None`, under `partial_dtensor`.

**Not verified.** `spmd_types` is not installed on the laptop this was read on,
so `aap(cp=R).axes()` could not be evaluated. Do that first -- it is a
three-line check and needs no GPU:

```python
import spmd_types as spmd
from torchtitan.models.common.decoder_sharding import attention_activation_placement as aap
for kw in ({}, {"cp": spmd.R}, {"cp": spmd.P}):
    lay = aap(**kw); print(sorted(a.value for a in lay.axes()),
                           getattr(lay, "partition_spec", None))
```

If the axis sets differ between the q and k/v layouts, `resolve_shared_mesh`'s
assert would fire rather than return `None` -- so equal axes with a differing
mesh resolution is the shape to look for.

## Also settled here

- **`agpt` TP=2 requires `--compile.no-enable`.** The `agpt + det` arm still
  hit the `tensors_saved_with_vc_check` / `DeviceMesh` AOT assertion; only the
  nocompile arm reached the forward. `sync_smoke.sh`'s green agpt TP=2 arm is
  therefore green under whatever compile default that path resolves to -- not
  evidence that compiled agpt TP=2 works. This is the known torch-2.13
  compile+AC+TP bug, previously recorded only for the 80B family; it fires at
  `agpt_debugmodel` scale too.
- The head-dim-asymmetry candidate from the previous revision is **dead**.
  agpt's head_dim is 16 vs moe's 192, and agpt reshapes correctly anyway. The
  difference was never the geometry.

## Probe

The instrumentation is in `moe/model.py` behind
`EZPZ_MLA_PLACEMENT_PROBE=1` (rank 0, no output when unset). Re-run with
`scripts/moe-tp2-probe.sh`. **Remove it once this is fixed.**

## Jobs

| Job | Config | Result |
|---|---|---|
| `8782754` | moe TP=2 | `Unknown spmd type: S(1)` -> fixed in `59b8c9053` |
| `8782821` | moe TP=2 | this bug |
| `8782843` | moe TP=2, SP off | this bug (SP ruled out) |
| `8784667` | moe TP=2 + placement probe | MEASURED: sdpa out is (1,1024,8,192) -- heads leaked into the sequence axis |
| `8782898` | upstream dsv3 TP=2 | INVALID control -- upstream trainer, so our xccl split_group workaround was never installed |
| `8782983` | upstream dsv3 TP=2, workaround hoisted | 0 split_group errors; fails earlier than ours on `dp_mesh_dims` plain-tensor params |
