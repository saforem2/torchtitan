# `--debug.deterministic` is not bit-reproducible on XPU (2026-08-16)

> [!IMPORTANT]
> **It takes BOTH inter-node (>1 NODE) AND seq >= 4096.** Neither alone
> reproduces it, and *multi-rank within one node is not enough* -- 12 ranks on
> a single node is deterministic at every sequence length tested, including
> 4096.
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
| `12473188` | 1 | 2048 / 4096 | 0 | deterministic |
| `12473188` | **12** | 2048 / 3072 / 3584 / **4096** | 0 | **deterministic** |
| `12473172` | 24 | **4096** | 2 | **NONDETERMINISTIC** |

Sorted by what actually separates the cases:

| ranks | nodes | seq | result |
|---:|---:|---:|---|
| 1 | 1 | 2048, 4096 | deterministic |
| 12 | **1** | 2048, 3072, 3584, **4096** | **deterministic** |
| 24 | **2** | 2048 | deterministic |
| 24 | **2** | **4096** | **NONDETERMINISTIC** |

Only the bottom row fails. It needs **inter-node** collectives *and* the larger
payload -- 24 ranks at seq=2048 is clean, and 12 ranks at seq=4096 is clean.

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
- **Not multi-rank + seq=4096 either.** This was my fourth wrong answer, and
  it was committed before the confirming run finished: **12 ranks at seq=4096
  is deterministic** (job `12473188`). What distinguishes the failing case is
  that 24 ranks spans **two nodes**.

## Working hypothesis

**Inter-node reduction order.** Intra-node collectives (12 ranks, one node)
are deterministic at every size tested; only crossing to a second node, at
sufficient payload, breaks it. That is consistent with oneCCL selecting a
different multi-node algorithm above a size threshold -- ring vs tree, or a
scatter/gather decomposition -- where the accumulation order across nodes is
not fixed. It fits the "identical until step 8" signature: weights stay
identical until the first reordered inter-node reduction, then diverge and
amplify.

**Still not confirmed.** The obvious next test is 24 ranks at 3072 and 3584 to
find the payload threshold, and 36/48 ranks (3-4 nodes) to check whether it
worsens with node count. Given that four hypotheses have already failed here,
that grid should be collected in ONE job before any further claim.

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
