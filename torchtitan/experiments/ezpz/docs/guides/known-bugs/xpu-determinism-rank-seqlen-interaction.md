# `--debug.deterministic` is not bit-reproducible on XPU (2026-08-16)

> [!IMPORTANT]
> **It takes BOTH multi-rank AND seq >= 4096.** Neither alone reproduces it.
> CLAUDE.md requires that two runs with `--debug.seed=42 --debug.deterministic`
> produce bit-identical loss and grad_norm; at production sequence lengths they
> do not. Until this is understood, an "identical loss" check on this stack
> cannot detect a regression smaller than **~0.012 at step 10**.

## The evidence table

Every datapoint collected, 2B, `--debug.seed=42 --debug.deterministic`,
compile off, each config run **twice and compared with itself**:

| job | ranks | seq | workers | result |
|---|---:|---:|---:|---|
| `12473174` | 1 | 2048 | 0 | deterministic |
| `12473174` | 12 | 2048 | 0 | deterministic |
| `12473174` | 24 | 2048 | 0 | deterministic |
| `12473177` | 24 | 2048 | 0 | deterministic |
| `12473177` | 24 | 2048 | 2 | deterministic |
| `12473177` | 24 | **4096** | 0 | **NONDETERMINISTIC** |
| `12473177` | 24 | **4096** | 2 | **NONDETERMINISTIC** |
| `12473187` | **1** | 4096 | 0 | deterministic |
| `12473172` | 24 | **4096** | 2 | **NONDETERMINISTIC** |

Divergence, when it happens, is identical in shape every time: **steps 1-7
bit-identical, first difference at step 8**, growing to ~0.012 by step 10.

## What it is NOT

Three single-variable explanations were tested and refuted. Each looked
convincing on partial data:

- **Not the collectives alone.** 1-, 12- and 24-rank are all bit-identical at
  seq=2048 (job `12473174`). Multi-rank by itself is fine.
- **Not the dataloader.** `num-workers` 0 vs 2 makes no difference in either
  seq column (job `12473177`). This was the most plausible-sounding hypothesis
  -- worker prefetch reordering batches -- and it is simply wrong here.
- **Not sequence length alone.** seq=4096 at 1 rank is deterministic
  (job `12473187`), as are 2560 / 3072 / 3584.

## Working hypothesis

The two required factors together point at **collective payload size**. At
larger sequence length the per-step gradient/activation collective crosses a
size threshold and oneCCL selects a different reduction algorithm; some of
those algorithms do not have a fixed reduction order, so the floating-point
accumulation order varies run to run. That fits the "identical until step 8"
signature -- the weights stay identical until the first reordered reduction,
then diverge and amplify.

**Not yet confirmed.** Job `12473188` runs 1-rank controls (2048/4096/8192) and
a 12-rank ladder (2048/3072/3584/4096) in one job to find where the interaction
switches on. If it tracks a byte threshold rather than a token count, that is
strong support.

## Method note (the actual mistake)

I reported three different causes before building the table above, each from a
single-variable probe: collectives, then dataloader workers, then sequence
length. Every one was consistent with the data in front of it and wrong overall,
because the effect requires an **interaction** and no single-variable sweep can
see one.

The table took minutes to assemble from results already in hand and immediately
made the pattern obvious. For any future "which knob causes X" question here:
**collect the full grid before proposing a cause**, and treat a single-variable
result as a constraint rather than an answer.

## Practical impact

- Any "non-computation change produces identical loss" verification at
  production seq-len is currently **unable to detect** a real regression below
  ~0.012 at step 10. It produced one false positive already (78th sync,
  job `12473171`), where a clean merge was reported as changing numerics.
- **Workaround for verification runs:** compare at **seq=2048**, where
  determinism holds at every rank count tested. A shorter sequence is a weaker
  test of the model but a valid test of a code change.
- Production training is unaffected -- nondeterminism at this magnitude is
  normal for large-scale training and is only a problem for bit-exact
  verification.
