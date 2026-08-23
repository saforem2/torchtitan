# exp05 -- 30B at 2N: first runs and performance tuning

> **2026-08-16, Sunspot, frameworks RC** (oneAPI 2026.1.0, torch
> `2.13.0a0+gitcf30153`). Jobs `12473195` (feasibility), `12473196` (tuning
> round 1), `12473197` (round 2). Follow-ups in flight: `12473198` (scaling),
> `12473199` (HSDP diagnosis), `12473200` (128k-vocab retry).

## What this establishes

**The 30B trains, fits at 2 nodes without tensor parallelism, and reaches
27.89% MFU** -- matching the 2B's small-N 29.26% on a model 17x larger.

Until now `30b-exp/` was a design document with no config, so none of its
performance claims could be tested. `agpt_configs["30B"]` was added
(commit `481a0aebc`) by interpolating the family -- the proposal fixes only
`dim=6144`:

| | dim | layers | heads | kv | ffn | vocab | params |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20B | 5120 | 64 | 40 | 8 | 14336 | 256128 | 20.7B |
| **30B** | **6144** | **64** | **48** | **8** | **16384** | **256128** | **28.1B** |
| 80B | 9216 | 84 | 72 | 8 | 32768 | 256128 | 96.7B |

head_dim is 128, matching both neighbours.

## Tensor parallelism HURTS (job `12473195`, uncompiled)

| TP | tps/GPU | MFU | memory |
|---:|---:|---:|---|
| **1** | **340** | **20.35%** | 73.0% |
| 2 | 253 | 15.16% | 58.1% |
| 4 | 151 | 9.04% | 42.9% |

Monotonic: TP=4 costs **55%** of throughput. A 28B model fits at 2N with
27% headroom to spare, so there is no memory reason to pay for TP. Same
pattern the 80B shows -- TP only earns its keep when the model does not
otherwise fit.

## Tuning (jobs `12473196` round 1, `12473197` round 2; all arms compiled)

| config | tps/GPU | MFU | memory | vs uncompiled |
|---|---:|---:|---|---:|
| uncompiled, LBS=1 | 340 | 20.35% | 73.0% | -- |
| compiled, LBS=1 | 360 | 21.54% | 68.1% | +5.9% |
| compiled, LBS=2 | 434 | 25.99% | 80.6% | +27.6% |
| **compiled, LBS=3** | **466** | **27.89%** | **78.5%** | **+37.1%** |
| compiled, LBS=2, seq=8192 | 375 | 24.87% | 80.9% | +10.3% |

**Batch size is the lever, worth +30% on top of compile.** LBS=3 is both
faster than LBS=2 *and* slightly cheaper in memory (78.5% vs 80.6%) -- larger
batches amortize the activation peak better, so the naive "bigger batch = more
memory" intuition does not hold across this step. Compile is worth ~6%.

At 27.89% MFU the 30B has essentially caught the 2B's small-N 29.26% on a
model 17x larger.

**Noise has two scales, and they differ by 3x.** Round 2 re-ran the LBS=2 arm
as a control: 434 tps / 25.96% against round 1's 434 / 25.99% -- 0.1%. That
looked like a flat "<1% noise floor" and this file said so.

It is not flat. Later runs of the *same* olmo2tok LBS=5 config in two
different jobs gave **512 tps / 29.67%** (job `12473207`) and **498 / 28.85%**
(job `12473208`) -- a **2.8% tps / 0.82pp MFU** spread, with byte-identical
memory (49.16GiB/76.83%) confirming it is the same run shape.

| comparison | spread |
|---|---:|
| same config, same job | **~0.1%** |
| same config, different jobs | **~3%** |

The within-job figure is tighter than the "<1%" first recorded here: job
`12473210` ran LBS=4 twice inside one job and got **499 tps both times**, to
the digit (28.92% / 28.94% MFU). That precision is what makes small effects
decidable -- LBS=5's +13 tps is ~26x the within-job spread.

**So any cross-job difference under ~3% is not evidence.** Same-job A/B is
mandatory below that threshold -- which is what the tokenizer comparisons here
used, and why their ~1pp results stand. Memory, by contrast, reproduces
byte-for-byte across jobs and can be compared freely.

`seq=8192` gives fewer tokens/sec but nearly the same MFU -- longer sequences
do more work per token, so the two roughly cancel. Use whichever the data
pipeline prefers; it is not a throughput lever at this scale.

### Activation checkpointing is not optional (round 2)

| arm | result |
|---|---|
| `activation-checkpoint:none`, LBS=2 | **OOM** -- 746 MiB request, 126 MiB free |
| `activation-checkpoint:selective`, LBS=2 | 0 steps, `RuntimeError: Enqueue process failed` |

The default (`full`) is load-bearing at this size. `none` is a genuine
out-of-memory -- unlike the HSDP failure below, which is not. The `selective`
error is different again and is not yet isolated.

## HSDP fails in gradient clipping, and it is NOT an OOM

Both HSDP attempts died with `UR_RESULT_ERROR_OUT_OF_RESOURCES`, which I twice
recorded as an out-of-memory. Reading the actual traceback:

```
trainer.py:819    grad_norm = dist_utils.clip_grad_norm_(...)
  distributed/utils.py:652  torch.nn.utils.get_total_norm(...)
    clip_grad.py:106  torch.stack([norm.to(first_device) for norm in norms])
      RuntimeError: level_zero backend failed with error: 40
                    (UR_RESULT_ERROR_OUT_OF_RESOURCES)
```

It failed at **45.87 GiB / 71.68% -- 18 GiB free**, in *gradient clipping*,
after the model had built and a full step had run. `UR_RESULT_ERROR_OUT_OF_RESOURCES`
from level_zero is device-**resource** exhaustion (events, command lists), not
device memory. The LBS=1 retry -- which existed only to give HSDP "more room"
-- failed identically, which is itself evidence the memory framing was wrong.

## HSDP: the ceiling is between 20B and 30B (job `12473199`)

Four arms, same 2-node HSDP mesh (`dp_shard=12`, `dp_replicate=2`):

| arm | steps | tps | MFU | memory | verdict |
|---|---:|---:|---:|---|---|
| **20B + HSDP** | **12/12** | **488** | **21.72%** | 76.29% | **works** |
| 30B + HSDP, `foreach=False` | 1 | 101 | 6.08% | 71.68% | fails |
| 30B + HSDP, `max-norm=0` | 1 | 104 | 6.25% | 71.68% | fails (bad test, below) |
| 30B + HSDP (control) | 1 | 105 | 6.32% | 71.68% | fails |

**The 20B runs HSDP fine.** So HSDP is not broken on XPU, and the failure is
specific to the 30B. Model size IS the axis -- which is the opposite of what
the "not an OOM, so probably not size" reasoning above predicted.

**`foreach=True` is exonerated.** The unfused path fails identically, same
frame, same error. The one-time log line confirms `foreach=False` really
reached the ranks (`clip_grad foreach=False` in the arm log, `foreach=True`
in every other arm), so this is a genuine negative rather than a flag that
never landed.

**It is not tensor count.** 20B and 30B have the *same* 579 parameter
tensors (both are 64 layers); only their sizes differ -- largest tensor 2,623M
vs 3,147M elements. So the level_zero limit being hit is a per-allocation or
per-size ceiling, not a count ceiling. That is consistent with a threshold
sitting between the two models.

### Where the ceiling is, and whether it is worth finding

`dim=5632` (44 heads at head_dim 128, ffn 15360, ~24B params) is the clean
midpoint and the obvious bisect point.

**But HSDP may not be worth the node hours.** Measured here, HSDP on the 20B
gives **21.72% MFU** -- well below the 30B's **27.89%** on plain FSDP at
LBS=3. The guide's "+2.7% MFU, prefer HSDP" claim
(`exp03-1b-proxy-design.md:56`) does not reproduce at this scale on XPU. A
bisect would find the ceiling, but the prize behind it looks smaller than the
batch-size lever already delivered. Sequence the scaling results first.


## On the proposal's central claim

The proposal argues per-GPU efficiency is the biggest available lever, citing
the 2B running **8.79% MFU at 512N** and calling a recovery to ~27% a "~3x
effective-compute multiplier".

**This is 2N data and cannot test that claim.** What it shows is that the 30B
reaches 27.89% MFU at small N, versus the 2B's 29.26% at the same node count.
The proposal's argument is about the *512N* regime, where the 2B collapses
because per-rank work is too small to hide communication. Whether the 30B
holds ~26% at 512N is the actual question, and it needs a scaling run.

That run fits entirely on Sunspot and is the obvious next step.

## Failures worth recording (all mine, none the model's)

- **`--activation-checkpoint selective` is not a flag.** The token is
  positional: `activation-checkpoint:selective`. The 80B submit script says so
  at line 175. Cost one arm.
- **"HSDP OOM'd" was wrong, twice.** I recorded
  `UR_RESULT_ERROR_OUT_OF_RESOURCES` as an out-of-memory and reasoned that
  "HSDP needs *more* memory than pure FSDP, and 68% left no room." The
  traceback says otherwise -- see "HSDP fails in gradient clipping". Retrying at LBS=1
  was the wrong experiment, and it failed the same way for the same reason.
- **`compile` defaults ON** in `agpt()` (`config_registry.py:249`). The
  feasibility job passed `--compile.no-enable` and the tuning job did not, so
  those two jobs are not directly comparable. My "compile" arm was a no-op
  against an already-compiled base, and the 340->360 gap I first dismissed as
  run-to-run noise was in fact the compile effect.
- **`--training.max-norm=0` does not bypass gradient clipping.** I built an
  arm around that assumption. `clip_grad_norm_` computes `get_total_norm`
  unconditionally at `distributed/utils.py:651`; `max_norm` is only consumed
  by the *scaling* call at line 675. So the arm was a second copy of the
  control, and the "clipping bypassed" label on it was wrong. To actually
  skip the norm the callsite has to not be reached at all.
- **I talked myself out of the right answer.** From "this is not an OOM" I
  concluded "so model size is probably not the axis, and a bisect would waste
  nodes." Both halves of that were wrong: not-an-OOM was correct, but the
  20B arm ran clean and the 30B did not, so size is exactly the axis. The
  error was treating one refuted mechanism as evidence against an unrelated
  hypothesis.
- **Duplicate flags.** Round 1 appended overrides onto a base array that
  already set them (`--local-batch-size=1 --local-batch-size=2`). Later-wins
  made it work by accident. Round 2 spells out each arm in full.

## Open

- **Scaling run** at 8/32/64N -- the only way to test the proposal's actual
  claim about 512N behaviour.
- **The Llama-3 128k vocab variant now runs** -- see below.

## Tokenizer: what we use, and what "64k" would cost

All arms above use **gemma-7b, vocab 256,128** -- the whole agpt family's
tokenizer (20B, 30B, 80B all inherit it via
`hf_assets_path="./assets/hf/gemma-7b"`). At `dim=6144` with untied
embeddings that is `256128 x 6144 x 2` = **3.15B params, 11.2% of the 28.1B
model**, spent on the vocabulary.

The proposal asks for "~64k custom BPE". **There is no 64k tokenizer on
hand.** What `assets/hf/` actually holds on Sunspot:

| tokenizer | vocab | embedding at dim=6144 |
|---|---:|---:|
| gemma-7b (current) | 256,128 | 3.15B (11.2%) |
| Llama-3.1-8B / 3.2-1B | 128,256 | 1.58B (5.9%) |
| llama-2-7b-hf | 32,000 | 0.39B (1.6%) |

So "use an existing 64k" is not available -- it would mean training one.
`agpt_30b_llama3tok` uses the 128k Llama-3 vocab as the closest real option:
it captures most of the saving (3.15B -> 1.58B; 28.1B -> 26.5B total), and the proposal's own
fertility table already prefers Llama-3 on code (gemma costs +17% on
starcoder, +26% on Python). A true 64k would save only ~0.8B beyond that,
which is a thin return for training and validating a new tokenizer.

Llama-2's 32k would save more still, but a 32k vocab is small by current
standards and would hurt fertility on everything except English prose.

**Neither is a drop-in swap.** Changing vocab means retokenizing the corpus
and invalidates every gemma-trained checkpoint, so this decision belongs to a
fresh flagship, not a continuation of the existing base.

### Measured: 128k vocab at 2N (job `12473201`)

`agpt_30b_llama3tok` had never produced a single step. Two bugs, both mine,
fixed in `4275bdb9b`: it inherited the family's gemma `hf_assets_path` (a
256k tokenizer feeding a 128,256 embedding -- ids the model cannot index),
and its docstring told callers to pass `--tokenizer.path`, which is not a
flag. Every invocation died in argument parsing; the earlier arm's log file
was never created, so the parser error was invisible.

With that fixed, at identical settings (TP=1, LBS=2, seq=4096, compiled, 2N):

| | params | tps/GPU | MFU | memory |
|---|---:|---:|---:|---|
| gemma 256k | 28.1B | 434 | 25.99% | 51.59 GiB (80.63%) |
| **Llama-3 128k** | **26.5B** | **449** | **26.17%** | **39.51 GiB (61.75%)** |

The throughput delta (+3.5% tps, +0.18pp MFU) is roughly what a 5.7% smaller
model should give -- not a per-GPU efficiency win.

**The memory is the real result: 12.08 GiB freed, 80.63% -> 61.75%.** exp05
established batch size as the dominant lever here (LBS 1->3 was worth +30%),
and the gemma variant runs out of room at LBS=3 / 78.5%. Job `12473202` tests
whether the 128k variant reaches LBS=4 or 5 -- if it does, the vocab change
is worth considerably more than its parameter count suggests.
