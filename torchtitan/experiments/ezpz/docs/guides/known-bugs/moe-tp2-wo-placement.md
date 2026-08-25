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

## The control run did not settle it

Job `8782898` ran UNFORKED `deepseek_v3_debugmodel` at TP=2 on the same node
and stack, to test whether the bug is ours or the platform's. It failed
EARLIER and differently:

```
RuntimeError: No backend for the parent process group or its backend
              does not support splitting        (distributed_c10d.py:5568)
```

That is `split_group`, i.e. XCCL does not support process-group splitting --
an XPU backend limitation, not EP (the config has `expert_parallel_degree=1`).

**Upstream MoE cannot run at TP=2 on this stack for an independent reason, so
it cannot serve as a comparison.** Our fork gets FURTHER: zero `split_group`
errors, reaching the forward pass.

## Next step

Instrument rather than theorize. Print the actual DTensor placement of
`output` immediately before `self.wo(output)` at TP=2, and walk back to the
first op that makes it `Shard(0)` instead of feature-sharded. Four successive
theories about this error were wrong; each time the answer came from one
measurement.

## Jobs

| Job | Config | Result |
|---|---|---|
| `8782754` | moe TP=2 | `Unknown spmd type: S(1)` -> fixed in `59b8c9053` |
| `8782821` | moe TP=2 | this bug |
| `8782843` | moe TP=2, SP off | this bug (SP ruled out) |
| `8782898` | upstream dsv3 TP=2 | `split_group` -- XPU limitation, no comparison |
