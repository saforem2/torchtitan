# `--debug.deterministic` is not bit-reproducible on XPU (2026-08-16)

## fp32 COMPUTE makes multi-node deterministic (2026-08-20)

**The multi-node nondeterminism is a bf16 precision artifact, and setting
`training.mixed_precision_param=float32` eliminates it.** Job 12473503,
agpt_20b, seq 2048, 10 steps, `--debug.seed=42 --debug.deterministic`,
comparing two identical `partial_dtensor` runs:

| cell | control | spmd vs partial |
| --- | --- | --- |
| 1 node, bfloat16 | 10/10 | bit-identical |
| 1 node, float32 | 10/10 | bit-identical |
| 2 nodes, **bfloat16** | **0/10**, diverges step 1 | -- |
| 2 nodes, **float32** | **10/10** | **bit-identical** |

Note what is already fp32 and what is not: agpt sets
`training.dtype=float32` (fp32 master weights) and
`mixed_precision_reduce=float32`, but `mixed_precision_param` defaults to
**bfloat16**, so the forward/backward compute is bf16. That is the knob.

Two consequences:

1. **Multi-node backend parity is answerable after all.** `spmd_types` is
   bit-identical to `partial_dtensor` at 2 nodes under fp32 compute -- the
   last open cell in the parity matrix.
2. **Recipe for any future numerics comparison on this stack:** add
   `--training.mixed-precision-param=float32`. It turns a 0/10 control into
   10/10, which is the difference between a test that can resolve something
   and one that cannot. Diagnostic only -- fp32 compute roughly doubles
   activation memory and costs throughput, so it is not a production setting.

### Correction to the section below

That section says the 2N divergence is a loss-reduction effect because
`grad_norm` was identical while loss differed. That held for the one cell
measured (2N seq4096) but does NOT generalize -- at 2N seq2048, grad_norm
differs too:

```
step 1   A  loss=12.90793  grad_norm=6.1255
         B  loss=12.90795  grad_norm=6.1249
```

So the divergence is not confined to the loss all-reduce; the gradients
differ as well. The safe statement is the one at the top: it is a bf16
precision effect that fp32 compute removes.

## Re-measured 2026-08-20 on torch 2.14.0.dev20260722+xpu

The note below was taken on the shipped torch 2.13. Re-measured on the
nightly, with `--debug.seed=42 --debug.deterministic`, agpt_20b, comparing
PRINTED loss and grad_norm between two identical `partial_dtensor` runs:

| config | control | first divergence |
| --- | --- | --- |
| 1 node, seq 1024 / 2048 / 4096 | **10/10 identical** | never |
| 2 nodes, TP=1, seq 4096 | 0/10 | **step 1**, loss only |
| 2 nodes, TP=2, seq 2048 | 11/20 | step 12, 1 ulp of printed precision |

Single-node determinism holds at every size tested, confirming the original
note's claim on this build too.

**The multi-node step-1 divergence is a reduction-order effect, not
data/init.** At 2N seq4096:

```
step 1:  A  loss=12.90411  grad_norm=5.2182
         B  loss=12.90394  grad_norm=5.2182
```

`grad_norm` is IDENTICAL while loss differs in the 4th decimal. Same
gradients means the model, the data, and the backward all agree; what differs
is the cross-node all-reduce ordering of the loss itself. An init or
data-ordering difference would move grad_norm too.

The TP=2 2-node case is different again -- deterministic for 11 steps, then
one-ulp drift -- which is ordinary accumulation rather than a per-step
reduction difference.

### Caveats on the numbers above

The 2N rows come from the last cell of a sweep whose earlier cells were
overwritten by a filename bug (all cells wrote `ns<seq>` instead of
`n<nodes>s<seq>`), so only `2N seq4096` survived with its logs intact. The
1N rows were computed before any overwrite and are sound. 2N at seq 1024 and
2048 were NOT measured -- an earlier version of this section claimed they
were, on verdicts my harness produced from empty step sets.


> [!IMPORTANT]
> **Multi-NODE runs are nondeterministic at seq >= 2560. Single-node is
> deterministic at every size tested, and seq=2048 is deterministic even
> multi-node.**
>
> CLAUDE.md requires two runs with `--debug.seed=42 --debug.deterministic` to
> produce bit-identical loss and grad_norm. At production sequence lengths on
> more than one node, they do not.
>
> **Use `--training.seq-len=2048` for any bit-exact verification.** That is not
> a guess: 24 ranks / 2 nodes / seq=2048 was run **six times** and all **15
> pairwise comparisons are identical** (job `12473191`). Every larger sequence
> length on >1 node fails.

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

### The complete grid

| ranks | nodes | seq | result | first diff |
|---:|---:|---:|---|---:|
| 1 | 1 | 2048, 4096 | deterministic | -- |
| 12 | 1 | 2048, 3072, 3584, 4096 | deterministic | -- |
| 24 | 2 | **2048** | **deterministic** (6 runs, 15/15 pairs) | -- |
| 24 | 2 | 2560 | NONDETERMINISTIC | step 10 |
| 24 | 2 | 3072 | NONDETERMINISTIC | step 11 |
| 24 | 2 | 3584 | NONDETERMINISTIC | step 9 |
| 24 | 2 | 4096 | NONDETERMINISTIC | step 8 |
| 36 | 3 | 4096 | NONDETERMINISTIC | step 8 |
| 48 | 4 | 4096 | NONDETERMINISTIC | step 8 |

Two clean separations:

- **Nodes, not ranks.** 12 ranks on ONE node is deterministic at every size,
  including 4096. Cross a node boundary and it breaks. Adding more nodes does
  not make it worse -- 2N, 3N and 4N all diverge at step 8.
- **A real boundary between 2048 and 2560.** seq=2048 survived six runs and 15
  pairwise comparisons at 24 ranks; 2560 fails. The divergence step wanders
  (8/9/10/11) at the failing sizes, so *when* it shows is probabilistic even
  though *whether* it shows is not.

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

**Inter-node reduction order, above a payload threshold near 2048-2560.**
Intra-node collectives are deterministic at every size; crossing a node
boundary with a large enough message is not. That fits oneCCL switching to a
different multi-node algorithm above a size cutoff -- ring vs tree, or a
scatter/gather decomposition -- where the cross-node accumulation order is not
pinned. The wandering onset step (8/9/10/11) fits too: each step is a fresh
chance to reorder, and larger messages reorder sooner.

The threshold is bracketed to **(2048, 2560]** at 24 ranks and is a *message
size* boundary, not a token count -- so the safe sequence length will differ
for other model sizes and parallelism layouts. Re-measure rather than assuming
2048 is safe elsewhere.

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
