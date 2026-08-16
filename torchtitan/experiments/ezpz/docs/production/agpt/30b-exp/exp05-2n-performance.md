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

**Run-to-run noise is under 1%, not the ~6% I assumed.** Round 2 re-ran the
LBS=2 arm as a control: 434 tps / 25.96% against round 1's 434 / 25.99%. An
earlier note in this file dismissed a real +5.9% compile effect as noise on
the strength of that wrong assumption.

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

Prime suspect: `foreach=True`, hardcoded at `trainer.py:822`. It fuses the
~500 per-parameter norms into one multi-tensor op; under HSDP each is a
DTensor needing a partial-reduce, so the fused stack claims far more
level_zero handles than the pure-FSDP path does.

If that is the cause, **model size is the wrong axis** -- a smaller model
would fail the same way. Job `12473199` tests it directly: 20B+HSDP (is it
size?), 30B+HSDP with `foreach=False` (is it the fused clip?), 30B+HSDP with
clipping bypassed (diagnostic), and a 30B+HSDP control. `EZPZ_CLIP_NO_FOREACH`
was added for this (commit `74f17090d`); the default is unchanged.

Note this contradicts the guide's "+2.7% MFU, prefer HSDP" advice
(`exp03-1b-proxy-design.md:56`) at 30B scale on XPU.


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
- **Duplicate flags.** Round 1 appended overrides onto a base array that
  already set them (`--local-batch-size=1 --local-batch-size=2`). Later-wins
  made it work by accident. Round 2 spells out each arm in full.

## Open

- **Scaling run** at 8/32/64N -- the only way to test the proposal's actual
  claim about 512N behaviour.
- **The Llama-3 128k vocab variant** (`agpt_30b_llama3tok`, 26.5B) has not run
  successfully yet. Job `12473200` retries it at the best-known config
  (TP=1, LBS=2) instead of the TP=2 / LBS=1 it first used -- which is the
  *slowest* layout measured here -- and captures the failure properly.

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
| Llama-3.1-8B / 3.2-1B | 128,256 | 1.57B (5.9%) |
| llama-2-7b-hf | 32,000 | 0.39B (1.4%) |

So "use an existing 64k" is not available -- it would mean training one.
`agpt_30b_llama3tok` uses the 128k Llama-3 vocab as the closest real option:
it captures most of the saving (3.15B -> 1.57B), and the proposal's own
fertility table already prefers Llama-3 on code (gemma costs +17% on
starcoder, +26% on Python). A true 64k would save only ~0.8B beyond that,
which is a thin return for training and validating a new tokenizer.

Llama-2's 32k would save more still, but a 32k vocab is small by current
standards and would hurt fertility on everything except English prose.

**Neither is a drop-in swap.** Changing vocab means retokenizing the corpus
and invalidates every gemma-trained checkpoint, so this decision belongs to a
fresh flagship, not a continuation of the existing base.
