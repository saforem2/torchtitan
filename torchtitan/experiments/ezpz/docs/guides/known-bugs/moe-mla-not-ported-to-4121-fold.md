# moe's MLA attention was never ported to the #4121 fold

> **RESOLVED 2026-08-25** by `38a0d595c`. Kept as the record of what the port
> required and how it was verified. Original framing below.
>
> The port mirrors `torchtitan/models/deepseek_v3/model.py` (the upstream MLA
> this forked from, which #4121 DID update) rather than inferring axes from the
> traceback. Verified on CPU by RUNNING it, not by shape:
>
> - full `moe_debugmodel` (163M params): flat `[16]` tokens -> `[16, 256128]`
>   logits, finite
> - trains: loss 4.95 -> 2.33 over 3 steps, finite grads on all 83 param
>   tensors, clean across 5 seeds
> - **causality**: perturbing the last token changes ONLY that token
> - **sequence isolation**: 3 folded seqs of 8, perturbing token 0 changes
>   tokens 0-7 and nothing crosses into seq 1 or 2
>
> The last two are what separate a correct port from a wrong-but-plausible one:
> a scrambled reshape has the right shape and leaks across positions.
> Regression tests: `tests/test_moe_mla_flat_layout.py` (15, CPU-only).
>
> **Still unverified: TP>1** -- and blocked behind a separate bug, see
> [`moe-tp2-wo-placement.md`](moe-tp2-wo-placement.md).
>
> **Original note:** The `expand(-1, k_nope.size(1), -1)` change only
> matters at tp>1 and no single-rank test reaches it. Needs a real multi-rank
> smoke before MoE runs at scale.

## Verdict

`moe_debugmodel` cannot train. It now gets past `config.build()` and into the
training loop, then dies in the first forward:

```
x = x + self.attention(self.attention_norm(x), attention_masks, positions)
bsz, seqlen, _ = x.size()
ValueError: not enough values to unpack (expected 3, got 2)
```

`moe/model.py:123`. Post-`42f4edfaa` the loader folds `[B, L] -> [T]`
unconditionally, so `x` arrives 2D `[T, D]` where this forward expects 3D
`[B, L, D]`.

## Why agpt is fine and moe is not

`agpt/model.py` is 50 lines: it uses upstream's attention, which #4121 updated
along with the layout. `moe/model.py` is 330 lines and carries a FORKED MLA
(multi-head latent attention) forward that upstream's change never touched.

`bsz`/`seqlen` are threaded through four reshapes in that forward:

```
q  = q.view(bsz, seqlen, -1, self.qk_head_dim)
kv = kv.view(bsz, seqlen, -1, self.qk_nope_head_dim + self.v_head_dim)
```

plus the `k_pe.expand(-1, -1, self.n_heads, -1)` that depends on their rank.

## What is already done

- `9adde43a9` -- renamed the local_map'd params `_BLNH` -> `_TNH`, matching
  upstream's `in_dst_shardings` keys. Required for TP>1, unvalidated here
  because moe never reaches attention.
- `e4517ae09` -- ported agpt's unflatten into moe's SDPA wrapper. Correct and
  necessary, but it sits one layer BELOW this failure, so it is also
  unreachable today.
- `5532e227f` -- the `cp` kwarg, which is what let moe get this far.

## What is NOT done

The MLA forward itself. This is a real port, not a rename: every `bsz, seqlen`
use has to become the flat equivalent, and getting one wrong reproduces the
exact failure the wrapper guards exist to catch -- a reshape that is
wrong-but-plausible, which SDPA accepts, showing up only as a degraded loss
curve.

Deliberately not attempted on 2026-08-25. Three fixes that day were "verified
working" and turned out incomplete; a four-reshape layout port written from a
traceback rather than a measured shape is the same mistake with a worse blast
radius.

## Fix direction

Measure first. Get the real `x.shape` at `moe/model.py:123` under the folded
loader (a one-step run with a print, or the shapes the smoke log already
carries), then port each reshape against that -- the way
`tests/test_attn_unflatten.py` was written for agpt, asserting element order
via `torch.equal` rather than shape alone. A wrong reshape has the right shape.

## Related

- `blendcorpus-fold-batch-dim.md` -- the fold itself
- `tests/test_attn_unflatten.py` -- the agpt precedent, and the docstring
  recording why an inferred layout passed its own negative control
