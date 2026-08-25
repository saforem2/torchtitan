# BlendCorpus yielded `[B, L]` after #4121 moved the stack to flat `[T]`

**Status:** FIXED (`42f4edfaa`, 2026-08-25). Smoked on Polaris job 7557829.

Upstream PR #4121 ("fold batch dim", part of the 80th sync) moved the LM
stack to a flat token layout. BlendCorpus is an out-of-tree ALCF loader
that counts **sequences** and kept yielding `[B, L]`. It was never
adapted, so after the sync every production agpt config on the
blendcorpus path died at the first attention layer:

```
ValueError: token count 1 is not a multiple of max_context_length 8192;
  this wrapper assumes the fixed-length rows ConcatThenSplitPacking emits
  and cannot reshape a ragged batch
```

`token count` is `dim 0`. It was **1** -- the batch size, not the 8192
tokens the stack expected.

## Root cause

Two places in `models/common/attention.py` treat `dim 0` as TOKENS:

| line | code |
| --- | --- |
| `:696`, `:704` | `num_tokens = x.shape[0]` then `x.view(num_tokens, -1, self.head_dim)` (`QKVLinear.forward`) |
| `:959` | `out_TD = out_TNH.view(out_TNH.shape[0], -1)` (`GQAttention.forward`) |

Hand those a `[1, 8192, 2048]` batch and `dim 0` stays `B`, so the
sequence and the heads merge into a single axis. Verified by running the
real ops rather than reading the names:

```
OLD (unfixed) | input (1, 8192) | q (1, 131072, 128) -> BROKEN
NEW (fixed)   | input (8192,)   | q (8192, 16, 128)  -> OK
```

`(1, 131072, 128)` is bit-for-bit the shape reported by the earlier
2026-08-24 Polaris crash (agpt-2b: `L=8192`, `N=16`, `head_dim=128`, and
`8192*16 == 131072`).

> [!WARNING]
> Do **not** read a layout off a tensor's name suffix. `q_BLNH` and
> `out_TNH` are the author's intent, not proof of what the tensor holds.
> Two earlier "fixes" in this area were wrong because they reasoned from
> the suffix instead of measuring; both were reverted. Run the reshape.

Core's Grain path never hit this: `TextCollator` emits
`torch.cat([...rows])` (`components/data/collators.py:57`), which is
already flat. Core's loss agrees -- `cross_entropy_loss` documents
`pred[T, V]` and `labels[T]`.

## The `positions` half is the dangerous one

`positions` also had to fold, and unlike the token tensor it **does not
raise**. `get_efficient_causal_mask_mod_for_packed_document`
(`attention.py:485-487`) reads `positions.shape[0]` as the sequence
length and cumsums along `dim 0`:

```
OLD [B,L]  seq_len(core reads)=2   doc_id -> [[0,-1,-1,-1,...],[1,-1,...]]
NEW [T]    seq_len(core reads)=16  doc_id -> [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
```

On `[B, L]` that reads the **batch size** as seq_len and yields document
ids of `-1`. Flex attention would then mask the wrong spans, and the only
symptom would be a quietly worse loss curve. Any config on the flex /
varlen backends was exposed.

## The fix

Three tensors folded at the yield in
`experiments/ezpz/blendcorpus/blendcorpus_builder.py`. Nothing in core
changed -- the experiment absorbs the adaptation, per the golden rule.

`positions` is derived **before** the fold, because
`_document_positions` reduces along `dim 1` to find EOD boundaries.

**Why this is safe for in-flight chains:** `flatten()` is row-major, so
it is identical to the `torch.cat(rows)` core already performs. Token
ORDER is unchanged, so the data stream a resumed chain sees is the same
one it would have seen before the sync. Per-row positions already restart
at 0, so document structure survives.

## Evidence (Polaris job 7557829, 2N, agpt_2b, blendcorpus, `.venv-torch213`)

| step | loss | grad_norm |
| ---: | ---: | ---: |
| 1 | 12.90724 | 1.94 |
| 4 | 15.26466 | 47.04 |
| 8 | 10.20471 | 14.38 |
| 12 | 8.19116 | 18.95 |

`rc=0`, 12/12 steps, loss **12.91 -> 8.19**.

A clean exit alone would prove nothing here -- a wrong reshape scrambles
Q/K and still runs. Two numeric checks back it up:

1. Step-1 loss is `12.907` against the known-good Grain run's `12.900`
   (delta `0.007`). Both sit at `ln(vocab)`; a correctly initialized LM
   must start there.
2. Loss falls ~4.7 nats in 12 steps. A model attending across scrambled
   Q/K would still descend, but not like that.

The grad_norm spike at step 4 (47.0, settling to ~19 by step 12) is the
usual pre-warmup transient for this config, not fold-related -- this
smoke ran without the 200-step warmup production uses.

## Tests

`experiments/ezpz/tests/test_attention_layout_no_op_guard.py` (9 tests,
CPU-only) pins the bad layout directly: the `B=1` identity-on-`v`, the
`B>1` cross-row leak, the GQA ratio check that fails to catch either, and
the `wo` width being `L` times too wide -- plus the correct path's
causality and row isolation.

`experiments/ezpz/tests/test_blendcorpus_fold_batch_dim.py` (7 tests,
CPU-only, no corpus or GPU needed). Covers the fold's order-equivalence
to core's `cat`, both attention shapes including GQA `n_kv != n_heads`,
and -- as a negative control -- the `-1` document-id failure, so the
silent-corruption mode stays pinned.

## The competing fix that was wrong: `[B, L*N, H]`

There is a second, incompatible "fix" for the same
`token count 1 is not a multiple of max_context_length 8192` error, and
it is worth recording because it is plausible, self-consistent, and
trains without ever raising.

Commit `5b4a81803` (2026-08-24, on `origin/ezpz`) patched the ATTENTION
WRAPPER instead of the dataloader: it read dim 0 as the BATCH dim and
unflattened as `[B, L*N, H]`, leaving blendcorpus emitting `[B, L]`. The
commit message reasoned "the batch dim is PRESERVED; it is L and N that
are folded together."

That reading makes SDPA see a sequence of length 1 with 131072 heads.
Causal attention over a single token is `softmax([one score]) = 1.0`, so
the output is EXACTLY `v`, bit for bit. No token attends to any other and
the model degenerates into a position-wise MLP, `wo(wv(x))`. It trains,
the loss descends -- an MLP still learns unigram statistics -- and
nothing ever raises.

Measured on CPU, agpt-2b shapes (`D=2048, H=128, n_q=16, n_kv=4`):

| | A: `[B, L*N, H]` (`5b4a81803`) | B: flat `[T]` fold (`42f4edfaa`) |
| --- | --- | --- |
| q / k | `(1, 131072, 128)` / `(1, 32768, 128)` | `(8192, 16, 128)` / `(8192, 4, 128)` |
| seq len SDPA sees | 1 | 8192 |
| heads SDPA sees | 131072 | 16 |
| `max abs(out - v)` | **0.000e+00** | 5.392 |
| `out.view(dim0, -1)` | `(1, 16777216)` | `(8192, 2048)` |

### Why the obvious sanity check passes

`131072 / 32768 == 4 == n_q / n_kv`. The GQA head ratio is preserved
because both sides scale by `L`, so a ratio check confirms the wrong
reading. Do not use it as evidence. The load-bearing check is
`max abs(out - v) == 0`: if attention output equals `v` exactly, no
attention happened.

The width mismatch at `wo` is the other tell -- `(1, 16777216)` against
the expected `(8192, 2048)`, off by a factor of `L`.

### Two degenerate modes, depending on B

Which failure you get under A depends on the batch, and only one of them
is a clean no-op:

| B | what SDPA sees | failure |
| --- | --- | --- |
| 1 (production blendcorpus) | sequence length 1 | output is EXACTLY `v`; no attention at all |
| > 1 | sequence length B | batch rows attend to EACH OTHER |

The cross-row leak runs FORWARD: with the batch dim standing in for the
sequence, row `i` sits at position `i`, and `is_causal` lets later
positions attend to earlier ones. Perturbing row 0 moves row 1 by ~100;
row 0 is unreachable from row 1 (delta exactly `0.0`). Production runs
`B=1`, so the shipped failure is the silent no-op.

### What HEAD does (correct)

`experiments/ezpz/agpt/__init__.py` reads dim 0 as TOKENS, recovers
`B = T / max_context_length`, and reshapes to `[B, L, N, H]`. Verified
end-to-end at `B=2`: `B` recovered exactly; perturbing the second half of
`v` leaves first-half outputs bit-identical (causality holds); perturbing
row 1 leaves row 0 bit-identical (rows isolated).

### No production impact

The AuroraGPT-project 20B chain
(`agpt-20b-sophiag-dolma-n128-gbs1024`) last checkpointed `step-5600` on
2026-08-16 08:05; `5b4a81803` is dated 2026-08-24 13:05. The chain had
been dry for eight days -- the #4121 break was what stopped it, and that
commit was the attempt to restart it. Zero steps trained under the bad
reading.

### Operational note

The chain lives in `/eagle/AuroraGPT/foremans/projects/saforem2/torchtitan`,
NOT the `/eagle/datascience/...` checkout where the gate smokes ran. The
two clones have separate `outputs/checkpoints/`. Checking one and
concluding "no chain exists" is how a resume turns into a step-0 restart
under the same ckpt-dir name; the `-A AuroraGPT` account and the checkout
go together.
