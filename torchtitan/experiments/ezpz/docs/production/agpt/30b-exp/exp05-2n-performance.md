# exp05 -- 30B at 2N: first runs and performance tuning

> **2026-08-16, Sunspot, frameworks RC** (oneAPI 2026.1.0, torch
> `2.13.0a0+gitcf30153`). Jobs `12473195` (feasibility), `12473196` (tuning
> round 1), `12473197` (round 2, running).

## What this establishes

**The 30B trains, fits at 2 nodes without tensor parallelism, and reaches
~26% MFU** -- close to the 2B's small-N 29.26% on a model 17x larger.

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

## Tuning (job `12473196`, all arms compiled)

| config | tps/GPU | MFU | memory | vs uncompiled |
|---|---:|---:|---|---:|
| uncompiled, LBS=1 | 340 | 20.35% | 73.0% | -- |
| compiled, LBS=1 | 360 | 21.54% | 68.1% | +5.9% |
| **compiled, LBS=2** | **434** | **25.99%** | **80.6%** | **+27.6%** |
| compiled, LBS=2, seq=8192 | 375 | 24.87% | 80.9% | +10.3% |

**Batch size is the lever, worth +20.6% on top of compile.** It spends the
idle memory headroom (68% -> 81%) on arithmetic intensity. Compile is worth a
further ~6%.

`seq=8192` gives fewer tokens/sec but nearly the same MFU -- longer sequences
do more work per token, so the two roughly cancel. Use whichever the data
pipeline prefers; it is not a throughput lever at this scale.

## On the proposal's central claim

The proposal argues per-GPU efficiency is the biggest available lever, citing
the 2B running **8.79% MFU at 512N** and calling a recovery to ~27% a "~3x
effective-compute multiplier".

**This is 2N data and cannot test that claim.** What it shows is that the 30B
reaches 25.99% MFU at small N, versus the 2B's 29.26% at the same node count.
The proposal's argument is about the *512N* regime, where the 2B collapses
because per-rank work is too small to hide communication. Whether the 30B
holds ~26% at 512N is the actual question, and it needs a scaling run.

That run fits entirely on Sunspot and is the obvious next step.

## Failures worth recording (all mine, none the model's)

- **`--activation-checkpoint selective` is not a flag.** The token is
  positional: `activation-checkpoint:selective`. The 80B submit script says so
  at line 175. Cost one arm.
- **HSDP OOM'd** (`UR_RESULT_ERROR_OUT_OF_RESOURCES`) -- a real result, not
  syntax. HSDP replicates across the replicate dimension, so it needs *more*
  memory than pure FSDP, and 68% left no room. Retried at LBS=1 in round 2.
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
- **LBS=3** and **AC=none** (round 2) -- is there more headroom past 80.6%?
- **HSDP at LBS=1** (round 2) -- does it help when it has room to run?
- **The Llama-3 128k vocab variant** (`agpt_30b_llama3tok`, 26.5B) has not run
  successfully yet; its arm produced no log at all. It halves the embedding
  (3.15B -> 1.57B) using a tokenizer we already vendor, and the proposal's own
  fertility table prefers Llama on code.
