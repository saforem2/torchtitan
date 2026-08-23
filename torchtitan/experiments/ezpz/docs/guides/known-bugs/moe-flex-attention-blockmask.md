# Flex attention on MoE: two stacked bugs, both fixed

**Status:** FIXED (2026-08-19). Both flex MoE configs train 5/5.
**Affects:** `moe_small`, `moe_10b_2b` (any `attn_backend="flex"` config on blendcorpus)
**Jobs:** 12473373, 12473379, 12473381, 12473382, 12473383, 12473385

## Symptom

Every flex-attention MoE config died immediately:

```
AssertionError: attention_masks must be instance of BlockMask,
                got <class 'NoneType'>
```

while the SDPA sibling of the same model (`moe_10b_2b_sdpa`) trained clean.
That asymmetry is the whole clue: SDPA relies on `is_causal` and ignores
masks, so only the flex path notices a missing one.

## Bug 1: blendcorpus never emitted `positions`

Core builds the mask in `Trainer._prepare_inputs` behind three conditions
(`trainer.py:738`):

1. `isinstance(self.model_config, Decoder.Config)` -- holds, `moeModel(Decoder)`
2. `positions is not None`                          -- **this was the failing one**
3. `inner_attention` is Flex/Varlen                 -- holds, `_10b_2b` sets `attn_backend="flex"`

`positions` was only ever emitted by the HF loader
(`hf_datasets/text_datasets.py:166`). blendcorpus yielded `{"input": ...}`
and nothing else, so the gate never fired, `get_attention_masks` was never
called, and the model got `None`.

Fix (`0a20fd03d`): emit per-document positions from blendcorpus.

Positions must be **per-document**, not a plain `arange`. blendcorpus packs
multiple documents into one sequence separated by EOD (`append_eod=True`),
and a monotonic arange would let attention cross document boundaries --
exactly what the mask exists to prevent. Positions restart at 0 after each
EOD, matching the HF loader's per-sample `range(len(sample_tokens) - 1)`.

The vectorized cumsum/scatter_reduce implementation was unit-tested against
the HF semantics on four cases (mid-sequence EOD, leading EOD, no EOD,
consecutive EODs) before it ever ran on a GPU.

## Bug 2: the EOD id was read off the wrong object

The first fix unit-tested clean and **still failed identically** on hardware.

Measured, not assumed: the gemma-7b tokenizer does resolve `eos_id = 1`
(job 12473379), so the effective EOD was 1 and positions should have been
emitted.

The lookup was the problem. `_document_positions` read the id back off
`self._bc_cfg`, which is `bc_get_config()` -- an object owned by the
blendcorpus library, not our dataclass. It is not guaranteed to expose
`eod_token_id`, and `getattr(..., None)` silently turns that into `None`,
so the method returned `None` and the key was never yielded.

Fix (`059a49d5e`): resolve the id once where it is already computed for
`bc_cfg`, and keep it on `self._eod_token_id`.

Note the red herring: the dumped config shows `"eod_token_id": null`. That
is the *user-facing* field, which is legitimately null here -- the value
comes from the tokenizer fallback. It does not mean the effective EOD is
unset.

## What the fix exposed: MoE routing is not recompute-stable under FullAC

With the mask built, `moe_small` got past the assertion and into a real
backward, where it hit a different failure:

```
CheckpointError: Recomputed values ... have different metadata
  saved:      torch.Size([560, 2048])
  recomputed: torch.Size([559, 2048])
```

560 vs 559 routed tokens -- the MoE router assigns differently on the
recompute pass.

Three arms, 2N, 5 steps each (job 12473385):

| config              | AC        | steps | memory          | result                        |
| ------------------- | --------- | ----- | --------------- | ----------------------------- |
| `moe_small`         | FullAC    | 0/5   | 1.53GiB (2.39%) | recompute mismatch            |
| `moe_small_noac`    | none      | 2/5   | 57.15GiB (89.31%) | level_zero error 40         |
| `moe_small_selac`   | selective | 5/5   | 46.54GiB (72.73%) | **PASS**                    |
| `moe_10b_2b`        | FullAC    | 2/5   | 38.01GiB (59.41%) | recompute mismatch          |
| `moe_10b_2b_noac`   | none      | 0/5   | 2.47GiB (3.86%) | level_zero error 40           |
| `moe_10b_2b_selac`  | selective | 5/5   | 51.16GiB (79.95%) | **PASS**                    |

Selective AC wins for a specific reason, not by luck: `SelectiveAC` lists
`aten.topk.default` as MUST_SAVE with the comment "topk can be
non-deterministic; save to keep MoE expert assignments stable between
forward and recompute" (`activation_checkpoint.py:52`). FullAC has no save
list, so it re-runs the router and is free to route differently.

AC-off is not a workaround: it pushes memory to 89% and dies on level_zero
error 40 (resource exhaustion -- events/command lists, not memory).

**This is now the shipped default** (`8a8847c56`): `moe_small` and
`moe_10b_2b` pass `activation_checkpoint_mode="selective"`. The `_sdpa`
siblings keep `full`, because they pass 5/5 with it -- they were not changed
for symmetry.

## Final state (job 12473388, 2N, 5 steps, defaults only)

| config | steps | memory | |
|---|---|---|---|
| `moe_small` | 5/5 | 72.64% | **PASS** |
| `moe_10b_2b` | 5/5 | 79.95% | **PASS** |
| `moe_10b_2b_sdpa` | 5/5 | 74.05% | PASS (control) |
| `moe_10b_2b_sdpa_bmm` | 5/5 | 94.54% | PASS (control) |

Both flex configs now work with no flags. The two SDPA controls report the
same memory as before the change (74.05% and 94.54%), which is the evidence
that the maskless path was not perturbed.

## Verifying

The SDPA controls (`moe_10b_2b_sdpa`, `moe_10b_2b_sdpa_bmm`) passed 5/5
before and after both fixes -- the maskless path is untouched. Any future
change here should keep them in the test matrix for exactly that reason.

`_document_positions` returns `None` when the EOD id is unknown, so the key
is omitted and SDPA behaves as before rather than silently receiving wrong
positions.
