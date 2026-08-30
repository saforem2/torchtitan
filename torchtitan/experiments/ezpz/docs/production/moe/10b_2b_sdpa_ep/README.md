# Production Training — MoE 10B_2B_sdpa EP=12

> Last updated: 2026-08-30

**Status:** Not planned any more -- **EP=12 has since been measured and does
not work on this stack.** This page is a 2026-04 draft that predates every
MoE expert-parallel result; nothing described here is running, queued, or
being prepared, and it should not be read as a live plan.

The prerequisite it names as unverified ("Verify EP=12 works at multi-node
scale") was tested in the 2026-08-19 MoE sweep (job `12473367`) and the
answer was no. Measured there:

- `moe_10b_2b_sdpa` at its **own default EP passes 5/5**; the same config
  with `--parallelism.expert-parallel-degree=12` forced on dies at step 2.
- `moe_10b_2b_sdpa_bmm_ep` (EP=12) reached 1/5 steps and hit
  `UR_RESULT_ERROR_OUT_OF_RESOURCES` at 86% memory.
- `moe_10b_2b_sdpa_hybridep` (EP=12) never trained at all -- upstream
  hybridep is NVIDIA-only by design
  ([WONTFIX](../../../guides/known-bugs/hybridep-is-nvidia-only.md)).
- The lower-EP arm `moe_10b_2b_sdpa_ep` in the registry is **EP=2**, not 12,
  and even that aborts 2/5 in `all_to_all_single` -- an Intel UR runtime
  fault, not our dispatch code.

Both failure families are written up in
[`moe-ep-a2a-degrades-with-size.md`](../../../guides/known-bugs/moe-ep-a2a-degrades-with-size.md).
Before this page becomes a plan again, that is the blocker to clear.

The draft configuration below is left as originally written.

## Configuration (Draft)

| Field | Value |
|-------|-------|
| Model | moe_10b_2b_sdpa (9.41B total / 1.98B active) |
| Experts | 36, top_k=3 |
| EP | 12 |
| Vocab size | 256,128 (Gemma tokenizer) |
| Seq len | 8,192 |
| Optimizer | TBD (AdamW suggested LR=3.99e-4 from 7B LR finder) |
| Compile | TBD |

The model shape above still matches the registry (36 experts, top_k=3, vocab
256128, seq_len 8192). The LR does not: `moe_10b_2b_sdpa` in
[`moe/config_registry.py`](../../../../moe/config_registry.py) now pins
**2.2e-4** with cosine decay, not the 3.99e-4 this draft carried over from
the 7B LR finder.

## Prerequisites

- LR finder at production scale (GBS >> 24) -- **not done**
- Verify EP=12 works at multi-node scale -- **done, and it fails**; see the
  status note at the top of this page
- Determine optimal TP/EP/FSDP decomposition -- **not done**
