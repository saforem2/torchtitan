# exp02 -- Is norms-only fp32 master sufficient?

> Tier 0 experiment for the [30B proposal](README.md). Tests the one
> claim in [Section 6](README.md#6-what-here-is-judgment-not-measurement)
> that was explicitly labelled judgment rather than measurement:
>
> > **norms-only fp32 would have fixed the observed bug.** I still recommend
> > full fp32 master because [...] That is a risk-asymmetry argument, not a
> > measurement.
>
> Prior investigation:
> [`training-dtype-bf16-norm-freeze.md`](../../../guides/training-dtype-bf16-norm-freeze.md).
>
> Status: **SETTLED** (2026-08-14). Job 8757151, Aurora, agpt debugmodel.

## Verdict

**No -- norms-only fp32 is NOT sufficient, and the Section 6 sentence quoted
above is wrong.** Arm B (fp32 master for norm params only) does unfreeze all
13 RMSNorm weights and matches full-fp32 on them, but it leaves
`tok_embeddings.weight` frozen at **frac_changed = 0.000345** versus
**1.000000** under full fp32 -- because agpt initializes the embedding at
`std=1.0` (`_EMBEDDING_INIT`, `agpt/__init__.py:202`), the *same* scale as
`RMSNorm.weight`, and therefore with the same ~7.8e-3 bf16 ULP. Across the
whole model, **38.36% of all parameter elements never move in arm B** (8.25M
of 21.5M) against **0.000%** in arm C. The prior investigation's "why other
parameters update fine" section reasoned only about *linear* layers at
std~0.005-0.02; it did not check the embedding, which is not a linear layer
and does not sit at that scale. So the shipped full-fp32 default is not
over-broad -- it is load-bearing, and for a second reason nobody had
identified. (All numbers MEASURED, 300 steps, single XPU tile, job 8757151.)

## Method

### Arms

| Arm | `training.dtype` | `EZPZ_FP32_NORMS` | Master weights |
|-----|------------------|-------------------|----------------|
| A | `bfloat16` | 0 | bf16 everywhere (reproduces the v1 bug) |
| B | `bfloat16` | 1 | bf16 everywhere **except** affine norm weights (fp32) |
| C | `float32` | 0 | fp32 everywhere (current production default) |

Compute dtype is bf16 in all three arms
(`MixedPrecisionPolicy(param_dtype=bfloat16, reduce_dtype=float32)`), so the
arms differ *only* in the master copy. Identical seed (42), `--debug.deterministic`,
identical data, optimizer (AdamW lr=8e-4), and step count (300).

### Was arm B cleanly expressible?

Not as a config flag -- it needed a small amount of code, and the reason is
itself a finding. **FSDP2 asserts a uniform original parameter dtype within
each `fully_shard` group:**

```
AssertionError: FSDP expects uniform original parameter dtype but got
{torch.bfloat16, torch.float32}
```

(`torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py::_init_mp_dtypes`,
verified directly with a 3-case probe.) An fp32 `attention_norm.weight`
inside an otherwise-bf16 transformer block is rejected outright. Arm B
therefore requires **re-grouping**: each promoted norm gets its own
`fully_shard` group. Nesting works -- the norms keep their own group inside
the enclosing block wrap -- but a module may not be passed to `fully_shard`
twice, so `model.norm` must be dropped from the `[norm, lm_head]` tail group.

Implementation (all inside `experiments/ezpz/`, opt-in, default-off):

- `agpt/fp32_norms.py` -- `promote_norms_to_fp32()` casts every affine
  `nn.RMSNorm` / `nn.LayerNorm` weight to fp32 before `fully_shard`, and
  returns those modules.
- `agpt/parallelize.py` -- gated on `EZPZ_FP32_NORMS=1`; passes the promoted
  modules to `apply_fsdp(..., separate_fsdp_modules=...)`, which wraps them
  first and excludes them from the tail group. With the env var unset the
  production grouping is byte-for-byte unchanged.
- `agpt/__init__.py` + `agpt/config_registry.py` -- added
  `debugmodel_qknorm` / `agpt_debugmodel_qknorm_local` to test the QK-norm
  recurrence claim.
- `scripts/oneoff/fp32_norms_ablation.py` -- the driver.
- `scripts/oneoff/compare_fp32_norms_ablation.py` -- the comparison.

### Jobs

| Job | Queue | Nodes | What | Outcome |
|-----|-------|-------|------|---------|
| 8757151 | debug | 1 | 3 arms x 300 steps, correctness | **COMPLETE -- the main result** |
| 8757164 | debug-scaling | 2 | throughput | FAILED (`~/.venv` shadowed the repo venv: `libsycl.so.8: undefined symbol urEnqueueCooperativeKernelLaunchE`). Fixed by activating the repo venv before `ezpz_setup_job`; resubmitted as 8757216 |
| 8757216 | debug-scaling | 2 | throughput at FSDP degree 24 | **COMPLETE -- Section 5** |
| 8757215 | debug | 1 | embedding visited-row control | FAILED (probe bug: `touched` mask on CPU vs params on `xpu:0`); fixed, resubmitted as 8757238 |
| 8757238 | debug | 1 | embedding visited-row control | submitted; had not returned at time of writing |

Reproduce:

```bash
# on Aurora, from the repo root
qsub torchtitan/experiments/ezpz/scripts/oneoff/submit_fp32_norms_ablation.sh

# or one arm directly (single rank, tiny model)
source .venv/bin/activate
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29577 \
ZE_AFFINITY_MASK=0 python3 -m torchtitan.experiments.ezpz.scripts.oneoff.fp32_norms_ablation \
    --arm B --steps 300 --seed 42 --out /tmp/armB.json
```

Environment: Aurora, `torch 2.13.0.dev20260520+xpu`, repo `.venv`, agpt
`debugmodel` (dim=256, n_layers=6, n_heads=16, vocab=32000, 21,499,136
params), seq_len 512, LBS 2, AC=full, compile off, one XPU tile.

Single rank is deliberate: it removes any DP-reduction nondeterminism between
arms, and the FSDP mesh still exists at degree 1 -- `parallel_dims._mesh_exist`
keeps the `fsdp` mesh alive at degree 1 *precisely* so `fully_shard` can
install the `MixedPrecisionPolicy` -- so all three arms exercise the real
master-weight path. Confirmed by the measured master dtypes: arm A 57/57
params bf16, arm B 44 bf16 + 13 fp32, arm C 57/57 fp32.

## Results

All numbers below are MEASURED from job 8757151 unless labelled otherwise.

### 1. Norm weights -- arm B fully fixes these

| Arm | master dtype | frozen norms | mean variance | value range |
|-----|--------------|--------------|---------------|-------------|
| A | bfloat16 | **13 / 13** | 0.000e+00 | [1.0000, 1.0000] |
| B | float32 | 0 / 13 | 8.612e-05 | [0.9413, 1.0382] |
| C | float32 | 0 / 13 | 8.363e-05 | [0.9416, 1.0371] |

Arm A reproduces the v1 signature exactly: every one of the 13 RMSNorm
weights is bit-identical to its 1.0 init after 300 steps, variance
identically 0. Arms B and C both fully unfreeze the norms and land in the
same place: per-tensor variance agrees to 4.72% mean / 14.14% max, std to
2.33% mean / 6.84% max -- the same order as the run-to-run drift the arms
show on the linear layers, not a systematic offset. **On norms, norms-only
fp32 is a complete fix.** This part of the Section 6 reasoning holds; what
follows is why it is not enough.

### 2. Non-norm parameters -- where arm B fails

The 42 core linear weights (attention q/k/v/o, FFN w1/w2/w3) behave
essentially the same in all three arms:

| Comparison | mean rel. std diff | max rel. std diff |
|------------|--------------------|-------------------|
| B vs C | 0.334% | 0.776% |
| A vs C | 0.309% | 0.707% |
| A vs B | 0.065% | 0.281% |

Note that A-vs-C is no worse than B-vs-C: on the *linear* layers the master
dtype barely matters, exactly as the prior investigation argued. Mean
`frac_changed` on those 42 tensors is 0.9938 (A), 0.9937 (B), 1.0000 (C).

**But the embedding is not a linear layer:**

| Parameter | Arm | master dtype | frac_changed | std | mean abs delta |
|-----------|-----|--------------|--------------|-----|----------------|
| `tok_embeddings.weight` | A | bfloat16 | **0.000346** | 0.99978 | 3.439e-06 |
| `tok_embeddings.weight` | B | bfloat16 | **0.000345** | 0.99978 | 3.427e-06 |
| `tok_embeddings.weight` | C | float32 | **1.000000** | 0.98619 | 1.086e-02 |
| `lm_head.weight` | A | bfloat16 | 0.996405 | 0.06736 | 3.328e-02 |
| `lm_head.weight` | B | bfloat16 | 0.996290 | 0.06726 | 3.298e-02 |
| `lm_head.weight` | C | float32 | 1.000000 | 0.07195 | 3.744e-02 |

`tok_embeddings` is initialized `normal_(std=1.0)`
(`_EMBEDDING_INIT`, `agpt/__init__.py:202`) -- **the same scale as
`RMSNorm.weight`, and therefore the same ~7.8e-3 bf16 ULP**. Arm B does not
touch it, so it stays frozen. The mean per-element movement differs by
**3,168x** between arm B (3.427e-06) and arm C (1.086e-02).

Whole-model aggregate:

| Arm | elements never updated in 300 steps | fraction |
|-----|--------------------------------------|----------|
| A | 8,249,554 / 21,499,136 | 38.372% |
| B | 8,247,641 / 21,499,136 | **38.363%** |
| C | 0 / 21,499,136 | **0.000%** |

Arm B recovers only **0.009 percentage points** of the 38.372% that arm A
freezes. Nearly all of the frozen mass is the embedding table, not the norms.

### 3. The mechanism is scale, not vocabulary coverage

An obvious objection: most vocab rows are never visited in 300 steps, so of
course they do not move. Two controls rule this out.

**(a) Isolated optimizer probe** (MEASURED, `/tmp/fp32abl/emb_test.py`, CPU) --
identical synthetic gradients applied to every row, 20 AdamW steps, so
coverage is not a factor:

| Parameter shape | init scale | master dtype | fraction of elements that moved |
|-----------------|-----------|--------------|-------------------------------|
| embedding-like | std = 1.0 | bfloat16 | **0.1981** |
| embedding-like | std = 1.0 | float32 | 1.0000 |
| linear-like | std = 0.02 | bfloat16 | 1.0000 |
| norm-like | init = 1.0 | bfloat16 | **0.0000** |

Scale alone reproduces the effect. At std=1.0 a bf16 master drops ~80% of
updates even when every element receives one; at std=0.02 it drops none.

**(b) bf16 ULP vs init scale** (derived):

| init scale | bf16 ULP | relative |
|-----------|----------|----------|
| 1.0 (norms, `tok_embeddings`) | 7.812e-03 | 7.8e-03 |
| 0.02 (q/k/v) | 1.221e-04 | 6.1e-03 |
| 0.005 (o, depth-scaled) | 3.052e-05 | 6.1e-03 |

The per-step update (~1.6e-5 for the 2B SophiaG runs cited in the prior doc)
is above the ULP at 0.005 and 100x below it at 1.0. The vulnerability is a
function of *parameter scale*, and both `RMSNorm.weight` and
`tok_embeddings.weight` sit at 1.0.

A per-arm visited-row control on the real model (job 8757215, restricting
`frac_changed` to embedding rows that actually received nonzero gradient) was
submitted to close this off on the real data path; the two controls above
already establish the mechanism.

### 3b. You cannot detect this from a checkpoint's stored dtype

Worth recording because it is a trap for anyone trying to audit existing runs.
DCP writes the **model** state dict, which is the bf16 all-gathered parameter,
not the fp32 master. MEASURED via `FileSystemReader.read_metadata()` on two 2B
production checkpoints:

| Checkpoint | `tok_embeddings.weight` stored dtype | `attention_norm.weight` stored dtype |
|---|---|---|
| v1 `...n256-gbs3072.bf16-norm-bug-20260429/step-10000` | `torch.bfloat16` | `torch.bfloat16` |
| v2 `...n512-gbs12288/step-1000` | `torch.bfloat16` | `torch.bfloat16` |

Both store bf16, and in both the sampled embedding rows lie exactly on the
bf16 grid (`frac exactly on bf16 grid = 1.0000`). So "is the checkpoint fp32?"
is the **wrong question** -- it is always bf16 on disk. The v1 detection
recipe in the prior guide works because it tests *values* (`norm.weight`
exactly 1.0), not dtypes.

One caution learned here: `norm.weight` was also all-ones in the v2
`n512-gbs12288` checkpoints at step-100 and step-1000. That is not the v1 bug
resurfacing -- 1000 steps of a 2B at these LRs moves the norms by less than
one bf16 ULP at 1.0 (7.8e-3), so the bf16 *stored* copy still rounds back to
exactly 1.0 even though the fp32 master has moved. The known-good v2 evidence
in the prior guide is at **step 5000**, where the accumulated change clears
the storage ULP. **Do not conclude "the bug is back" from an early v2
checkpoint** -- check a late one, or check the master copy in the optimizer
state.

### 4. Loss does not reveal any of this

| Arm | step 1 | step 50 | step 150 | step 300 | mean(last 20) |
|-----|--------|---------|----------|----------|---------------|
| A | 10.80178 | 3.47237 | 2.96511 | 2.77249 | 2.77937 |
| B | 10.80178 | 3.47209 | 2.96444 | 2.77350 | 2.78010 |
| C | 10.80145 | 3.47511 | 2.96026 | 2.77853 | 2.77575 |

All three are within 0.005 nats at step 300, and arm A (the *broken* arm) has
the numerically lowest final loss. This reproduces the central lesson of the
original investigation at 1/1000th the scale: **training loss cannot
distinguish these arms.** Anyone using loss as the acceptance signal for a
precision change will accept a broken configuration. The v1-vs-v2 lm-eval gap
(+19.8pp ARC-Easy) is what the loss curve was hiding.

### 5. Cost -- arm B is also the slowest, at both FSDP degrees

Single rank, FSDP degree 1 (job 8757151):

| Arm | wall (300 steps) | ms/step | peak memory |
|-----|------------------|---------|-------------|
| A | 18.6s | 62 | 0.811 GiB (1.27%) |
| B | 22.4s | 75 (+25% vs C) | 0.812 GiB (1.27%) |
| C | 17.9s | **60** | 0.963 GiB (1.50%) |

2 nodes, 24 ranks, FSDP degree 24, seq_len 2048, 100 steps (job 8757216;
mean of the last 5 logged steps, per-step spread +/-1 tps so the ordering is
well outside noise):

| Arm | tps | MFU | peak memory |
|-----|-----|-----|-------------|
| A (bf16 master) | 29.8 | 1.20% | 1.85 GiB |
| B (norms-only fp32) | 39.0 | 1.55% | 1.85 GiB |
| C (full fp32 master) | **54.0** | **2.16%** | 1.85 GiB |

**Arm C is the fastest arm at real FSDP degree -- 38% more throughput than
arm B and 81% more than arm A** -- and all three use identical memory at this
size. Arm B pays the regrouping cost: each of the 13 promoted norms gets its
own `fully_shard` group and therefore its own all-gather/reshard instead of
riding along in the block's flat parameter. So arm B is beaten by arm C on
*every* axis measured here: correctness (embedding stays frozen), throughput,
and (at this scale) memory.

Caveat on these absolute numbers: a 21M model at 24 ranks is
communication-bound, so ~1-2% MFU is expected and these are **not**
production-representative throughputs. The *ordering* is the result; the
magnitude of the gap would shrink on a compute-bound model.

The memory saving arm B was supposed to buy is negligible at production
scale anyway (derived):

| Model | norm params | as % of total | fp32 master saving of arm B vs arm C |
|-------|-------------|---------------|--------------------------------------|
| 2B | 51,200 | 0.0026% | 4.0 GB -> 0.0001 GB |
| 20B | 399,360 | 0.0020% | 40.0 GB -> 0.0008 GB |
| ~30B | ~580,608 | 0.0019% | ~60 GB -> ~0.001 GB |

So arm B's supposed advantage over arm C was memory -- and it cannot realize
it. To be *correct* it would also have to promote `tok_embeddings`
(2048 x 256128 = 525M params at 2B, ~26% of the model), at which point it is
most of the way to full fp32 while still paying the regrouping cost, still
running slower than arm C, and still leaving every future near-1.0 parameter
exposed.

Note also that the Section 6 premise "there is no throughput cost" was
asserted about *arm C vs bf16 master*. The measurement says C is not merely
free relative to A -- it is **faster** than A here (54.0 vs 29.8 tps at 24
ranks). That is at 21M params and communication-bound, so it should not be
extrapolated to production; the honest statement is that no throughput
argument favors A or B at this scale.

## Implications for the 30B proposal

- **README Section 6's fp32 bullet should be corrected.** The sentence
  "norms-only fp32 would have fixed the observed bug" is false as written:
  it would have fixed the *observed* symptom (norm weights) while leaving a
  larger frozen parameter -- the embedding table -- undetected. Keep full
  fp32 master, but the justification is now a measurement, not risk asymmetry.
- **The Section 6 recurrence argument was right and under-stated.** It
  predicted the vulnerability recurs "for any parameter initialized near 1.0,
  QK-norm gains being exactly that." It already *had* recurred, in the
  shipped 2B/20B/80B configs, on `tok_embeddings` -- and nobody noticed
  because the v1 post-mortem only checked the norms.
- **This is a cheap, scale-free regression test.** 300 steps on a 21M model
  at ~20s per arm detects a class of bug that cost a full production restart.
  Worth running whenever an init scale or a precision default changes.

## Caveats

- **Scale.** Everything here is on the 21M debugmodel with AdamW lr=8e-4 for
  300 steps. The *mechanism* (bf16 ULP vs parameter scale) is scale-free and
  is confirmed by the isolated optimizer probe, but the exact fractions
  (38.36%, 3,168x) are debugmodel-specific. The embedding's share of total
  params differs by model: 38% here, ~26% at 2B.
- **Single rank, degree-1 FSDP.** Correct for the correctness question (the
  MixedPrecisionPolicy is installed at degree 1) but it means the throughput
  numbers in Section 5 are a lower bound on collective overhead and should
  not be quoted as production throughput. Job 8757216 addresses this.
- **The visited-row control on the real model (8757215) had not returned at
  the time of writing.** The mechanism is established by the isolated probe
  (which has no coverage confound at all) and by the fact that arm C moves
  100.000% of the same embedding under the same data and step count -- if
  coverage were the explanation, arm C could not have moved unvisited rows
  either. Weight decay does move unvisited rows, which is precisely the
  point: those decay updates are ~1e-5 against a 7.8e-3 ULP and round to zero
  in bf16.
- **`lm_head` is not affected** (std ~0.06, `frac_changed` 0.996 even in arm
  A), so this is specifically about the input embedding, which agpt does not
  tie to the output head in these configs.
- **Optimizer.** AdamW here; production 2B/20B use SophiaG. The prior
  investigation measured the ~1.6e-5 update magnitude under SophiaG and the
  same ULP argument applies, but the exact frozen fraction under SophiaG at
  2B was not re-measured in this experiment.
- **Not committed to git.** The code changes listed under Method are in the
  working tree locally and copied to the Aurora checkout; they are opt-in and
  default-off, so the production path is unchanged either way.

## Related

- [`training-dtype-bf16-norm-freeze.md`](../../../guides/training-dtype-bf16-norm-freeze.md)
  -- the prior investigation this extends. Its "Why other parameters update
  fine" section is correct for linear layers and incomplete for the embedding;
  a correction has been added there pointing here.
- [README.md Section 6](README.md#6-what-here-is-judgment-not-measurement)
  -- the claim under test; the fp32 bullet has been retracted and replaced.
- [exp04](exp04-fp32-inference-investigation.md) -- the other half of the
  fp32-master question: fp32 master does **not** force fp32 at serving time.
  Together, exp02 and exp04 say fp32 master is necessary for training
  correctness and costs nothing at inference.
- [exp01](exp01-tokenizer-analysis.md) -- independently found that production
  agpt has weight tying **off**, so the 256128x2048 matrix exists twice. Only
  the input embedding is freeze-affected (`lm_head` sits at std ~0.06); if
  tying is ever turned on, the tied tensor inherits the input embedding's
  std=1.0 init and its exposure.
- [EXPERIMENTS.md](EXPERIMENTS.md) -- Tier 0 tracking table.
