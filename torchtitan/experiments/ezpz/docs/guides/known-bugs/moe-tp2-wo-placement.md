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

## The open question

`agpt/__init__.py` carries a **byte-identical** unflatten, and **agpt TP=2
passes** (both arms green in `sync_smoke.sh`). Both wrappers are local_mapped
through the same `set_gqa_inner_attention_local_map`. So the unflatten alone
cannot be the whole story -- something differs in how moe's tensors reach it,
or in whether agpt reaches the 3D branch at all under TP.

Resolve that before changing either wrapper: a fix derived from moe alone
risks breaking the agpt path that currently works in production.

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
