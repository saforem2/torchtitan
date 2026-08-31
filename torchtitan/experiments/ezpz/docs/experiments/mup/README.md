# muP for agpt: audit, design, and staged plan

**Status:** DESIGN. Nothing implemented. No GPU time committed.

**Question this answers:** what would it take to put agpt on Maximal Update
Parametrization (muP), so that learning rates tuned on a small proxy model
transfer to 30B/80B instead of being re-measured with an LR finder at every
scale and every batch size?

Everything below is either verified at runtime on the cluster (marked
VERIFIED) or cited to a primary source. Uncertainties are flagged as such
rather than smoothed over -- this document exists to decide whether to spend
node-hours, and a confident wrong claim is worse than an open question.

---

## 0. Headline

**The starting position is unusually good: three of the four init groups
already satisfy muP by accident.** The Megatron-DeepSpeed base std agpt uses,
`sqrt(2/(5*dim))`, is exactly `fan_in^-1/2` up to a constant -- which is what
muP asks for. Embedding init is width-independent, norms are `ones_`. Only the
unembedding is off-spec.

**But three features of the current recipe are documented to BREAK muP
transfer**, and agpt has all three: trainable RMSNorm gains, weight decay, and
`1/sqrt(d)` attention. A naive "implement Table 3" port would likely fail its
own coordinate check.

**And muP will not fix SophiaG.** Sophia takes AdamW's muP LR table unchanged
(see 4.3). muP is a width-transfer rule, not a stability mechanism. The
SophiaG divergence documented in
[known-bugs/sophiag-stochastic-divergence-30b.md](../../guides/known-bugs/sophiag-stochastic-divergence-30b.md)
is state-dependent and fires at four distinct steps across four arms including
one with the RNG pinned; nothing in muP addresses that.

---

## 1. What agpt does today

### 1.1 Init -- VERIFIED at runtime on `agpt_30b_olmo2tok_optcmp_adamw`

agpt does **not** use llama3's inits. It has its own complete parallel copy in
`experiments/ezpz/agpt/__init__.py:285-424`, and the base std differs:

| | llama3 | agpt |
|---|---|---|
| linear base std | `0.02` const (`models/llama3/__init__.py:46`) | `sqrt(2/(5*dim))` (`agpt/__init__.py:294`) |
| depth base | `0.02` | `sqrt(2/(5*dim))` (`agpt/__init__.py:310`) |

Measured stds at dim=6144, 64 layers:

```
L0    w1=0.008068715  w2=0.005705443  w3=0.005705443  wo=0.005705443
L1    w1=0.008068715  w2=0.004034358  w3=0.004034358  wo=0.004034358
L63   w1=0.008068715  w2=0.000713180  w3=0.000713180  wo=0.000713180
base  sqrt(2/(5*6144)) = 0.008068715
```

`wo` ratio L0->L63 is exactly 8.00 = sqrt(64), i.e.
`base / sqrt(2*(layer_id+1))`.

**`w3` is depth-scaled, and probably should not be.** `w3` is the *gate*
projection (`FeedForward.forward` computes `w2(silu(w1(x)) * w3(x))`,
`models/common/feed_forward.py:53-54`), not a residual output. It gets the
depth factor because `make_ffn_config` applies a single `w2w3_param_init` to
both (`models/common/config_utils.py:300-305`) and agpt passes `_depth_init`
into it (`agpt/__init__.py:402`). This is upstream behavior -- llama3 does the
same at `models/llama3/__init__.py:106` -- so it is a shared quirk rather than
an ezpz divergence. It is consistent across widths, so it does not break a
coordinate check, but it should be a deliberate decision rather than an
inherited accident.

### 1.2 Attention scale -- VERIFIED

`self.scaling = self.head_dim**-0.5 if config.head_dim is not None else None`
(`models/common/attention.py:947`).

**No agpt config sets `head_dim`.** `make_gqa_config` is called without it
(`agpt/__init__.py:410-420`), so `GQAttention.Config.head_dim` stays `None`,
so `self.scaling` is `None`, so `scale=None` reaches the kernel and PyTorch
applies its implicit `1/sqrt(E)`. Effective scale at 30B: `1/sqrt(128) =
0.0883`. The *effective* head_dim is 128, derived as `dim // n_heads`
(`attention.py:926-930`) -- it is only the config field that is None.

### 1.3 Optimizer groups -- VERIFIED

Every agpt flavor uses exactly one catch-all group:
`ParamGroupConfig(pattern=r".*", ...)` from `default_adamw`
(`components/optimizer/optimizer.py:378-399`), wired at
`agpt/config_registry.py:329`. Confirmed against a production log:

```
Optimizer AdamW (model_part=0): 45 params [.*] {'lr': 0.0008, 'betas': (0.9, 0.95),
  'eps': 1e-08, 'weight_decay': 0.1, ...}
```

### 1.4 Parameter FQNs -- VERIFIED against three production DCP checkpoints

2,164 keys, identical structure across all three:

```
tok_embeddings.weight
layers.<N>.attention_norm.weight
layers.<N>.attention.qkv_linear.w{q,k,v}.weight
layers.<N>.attention.wo.weight            # NOT under qkv_linear
layers.<N>.ffn_norm.weight
layers.<N>.feed_forward.w{1,2,3}.weight
norm.weight
lm_head.weight                            # live models use lm_head, not output
```

All linears are `bias=False` (`models/common/linear.py:33`).

---

## 2. What muP requires, per group

Using Table 8 of Tensor Programs V ([arXiv 2203.03466](https://arxiv.org/abs/2203.03466)),
the multiplier form -- which is the one to use, because Table 3 is
structurally incompatible with weight tying. `m = dim / base_dim`.

| group | muP init | muP AdamW LR | current | change |
|---|---|---|---|---|
| `tok_embeddings` | width-independent | `eta` | `std=1.0` const | **none** |
| `wq/wk/wv/w1` | `1/fan_in` | `eta/m` | `sqrt(2/(5d))` | **LR group only** |
| `wo` | `1/fan_in` | `eta/m` | same, x depth | **LR group only** |
| `w2/w3` | `1/fan_in` | `eta/m` | same, x depth | **LR group only** (+ see 5.1) |
| norms | `1.0` | `eta` | `ones_` | **LR group only** |
| `lm_head` | `1/fan_in^2`, or `1/fan_in` + `1/m` forward mult | `eta` | `d^-1/2` (SP) | **init AND LR group** |
| attention logits | `alpha * sqrt(d_head_0)/d_head` | -- | implicit `1/sqrt(d_head)` | **new knob** |

**The lm_head is the subtle one.** Under Adam, changing the init alone does
not reproduce the multiplier's effect, because Adam is scale-invariant in the
gradient. Both halves -- the `d^-1` init and the `O(1)` LR group -- are
required together. This is the single point most worth confirming empirically
with a coordinate check rather than trusting the analysis.

---

## 3. What would have to change

Five items. **Nothing in core torchtitan.**

| # | item | where |
|---|---|---|
| 1 | `lm_head` init to `d^-1` | `agpt/__init__.py:301-306` |
| 2 | attention scale `1/head_dim` | agpt-local SDPA wrappers, `agpt/__init__.py:51,155,192` |
| 3 | four LR groups instead of one | new `default_mup_adamw()` beside `default_adamw` |
| 4 | purpose-built width ladder | `agpt/__init__.py:488+` |
| 5 | decide `w2` fan_in and `w3` depth-scaling | `agpt/__init__.py:402` |

### 3.1 The attention knob must not be a new positional argument

`set_gqa_inner_attention_local_map` matches `in_dst_shardings` by **parameter
name**, and a mismatch broke TP=2 once already -- see the comment at
`agpt/__init__.py:76-91`. Any new attention-scale control must be a
module-level global or a `Config` field, following the existing
`set_ezpz_max_context_length` pattern (`agpt/__init__.py:15-23`).

### 3.2 The LR-group machinery works, with two traps

**The scheduler preserves per-group ratios -- VERIFIED.** `LambdaLR` computes
`base_lr * lambda(t)` per group from per-group `base_lrs`
(`components/optimizer/lr_scheduler.py:198-203`), and the torchtitan lambda
returns a pure factor in [0,1]. Tested on-cluster with groups at 1e-2 and
1e-4: after decay, 5e-3 and 5e-5, **ratio exactly 100.0 preserved**.

This is worth stating plainly because GPT-NeoX ships the opposite: its
scheduler has an `if self.use_mup and "width_mult" in group` branch that is
**unreachable** (nothing ever sets that key), so the `else` overwrites
MuAdam's scaled LR on the first step and silently reverts to SP. We do not
have that bug.

**Trap 1: activation checkpointing rewrites the FQNs the regex matches
against.** VERIFIED:

```
BEFORE AC: layers.0.attention.weight
AFTER  AC: layers.0._checkpoint_wrapped_module.attention.weight
```

agpt defaults to `FullAC` (`config_registry.py:357`). Matching uses
`pattern.search(name)` on the **raw** name (`optimizer.py:195`), canonicalizing
only afterwards for storage (`:197`). So `^layers\.\d+\.attention\.` matches
zero parameters and raises the empty-group `ValueError`. Leaf-anchored
patterns are safe.

**Trap 2: muP groups cannot be created from the CLI.** VERIFIED: tyro renders
`list[ParamGroupConfig]` at the default's length.
`--param-groups.0.optimizer-kwargs.lr=1.5e-4` works;
`--param-groups.1.pattern=xyz` is rejected as unrecognized. Groups must be
defined in Python.

An AC-safe pattern set, first-match-wins order:

```python
param_groups=[
    ParamGroupConfig(pattern=r"^tok_embeddings\.", ...),   # LR eta
    ParamGroupConfig(pattern=r"^lm_head\.",        ...),   # LR eta
    ParamGroupConfig(pattern=r"(?:^|\.)(?:attention_norm|ffn_norm|q_norm|k_norm)\.weight$|^norm\.weight$", ...),
    ParamGroupConfig(pattern=r".*",                ...),   # LR eta/m
]
```

### 3.3 No existing config can serve as the muP base

muP's coordinate check needs base and target to differ in **width only**. No
two registered agpt configs do:

| flavor | dim | L | head_dim | H/dim | vocab |
|---|---|---|---|---|---|
| debugmodel | 256 | 6 | **16** | 3.00 | 32000 |
| 2B | 2048 | 12 | 128 | **5.375** | 256128 |
| 20B | 5120 | **64** | 128 | 2.80 | 256128 |
| 30B_olmo2tok | 6144 | 64 | 128 | 2.667 | 100352 |

`debugmodel` is the worst candidate: `head_dim = 256/16 = 16`, an 8x mismatch
with production. Since muP's attention change is precisely about head_dim, a
ladder anchored there would validate the wrong scaling law.

A purpose-built ladder holding everything but `dim` fixed at 30B values
(`L=64`, `head_dim=128`, `H/dim=2.667` exactly, `vocab=100352`):

| rung | dim | n_heads | hidden | m vs 30B |
|---|---|---|---|---|
| mup_1536 | 1536 | 12 | 4096 | 4.0 |
| mup_3072 | 3072 | 24 | 8192 | 2.0 |
| target | 6144 | 48 | 16384 | 1.0 |

`4096/1536 = 8192/3072 = 16384/6144 = 2.667` exactly, so the FFN ratio is
rational at every rung -- which matters for 5.1. Compute `hidden_dim` directly
rather than via `compute_ffn_hidden_dim`, whose `multiple_of=1024` rounding is
what makes 20B's ratio 2.80 instead of 2.667.

**Open:** whether to hold `n_kv=8` fixed (making the GQA ratio vary 1.5x to
6x across rungs) or scale it proportionally (1, 2, 4). Unresolved whether
muP's guarantees hold cleanly when the GQA grouping ratio varies.

---

## 4. The coordinate check

This is the validator, and it must exist **before** any muP code does.

### 4.1 It is cheap, and a prototype already ran

A 3-width prototype ran on the **login node, on CPU, ~5 seconds per width**,
no PBS allocation, using `register_forward_hook` and injecting width-varied
flavors at runtime. Current standard parametrization gives:

```
W= 256  layers.3  step1=0.80813  last=0.81251
W= 512  layers.3  step1=0.80392  last=0.82202
W=1024  layers.3  step1=0.80862  last=0.88941
```

Coordinates agree at init (fan-in scaling already handles that) and **diverge
with width by step 4** at deeper layers. That is the SP signature, and it
means the check discriminates on this model -- the prerequisite for it being
able to validate anything.

### 4.2 The trap that would fake a pass

**Top-level `dim` is decorative.** `Decoder.__init__` builds from
`config.tok_embeddings`, `config.layers`, `config.norm`, `config.lm_head`,
each carrying its own baked dimensions (`models/common/decoder.py:253-268`).
VERIFIED: setting `dim` 256 -> 512 leaves the embedding at `(32000, 256)` and
the parameter count at 21,499,136, unchanged.

A width sweep driven that way produces N identical models, and
width-invariant coordinates would "confirm" muP while testing nothing. This is
[the silent-no-op pattern](../../guides/known-bugs/) in its purest form. The
correct entry point is `_build_agpt_config` (`agpt/__init__.py:427`), which
threads `dim` into every sub-config -- VERIFIED to produce genuinely different
models:

```
dim= 256 heads= 4 ffn= 768 params= 21.50M
dim= 512 heads= 8 ffn=1536 params= 53.22M
dim=1024 heads=16 ffn=2816 params=142.62M
```

### 4.3 Two more mechanical blockers

- **torch.compile is fatal, not slow.** A `.item()` inside a hook in a
  compiled block killed job 12473689 at 0 steps with
  `InternalTorchDynamoError` (`912b8deec`, guarded at `attention.py:110`).
  Coordinate checks must run `--compile.no-enable`. Acceptable -- they are
  eager by nature.
- **Activation checkpointing double-fires hooks.** AC replays the forward, so
  3 steps produced `nfire=6` with different values. Silent if unnoticed.

### 4.4 Existing instrumentation does not help

| component | captures | verdict |
|---|---|---|
| `diagnostics/__init__.py:84-254` | weights/grads, per-parameter | irrelevant -- we need activations |
| `diagnostics/attention.py:91-137` | q/k stats at **layer 0 only** | too narrow |
| `diagnostics/attention.py:147-158` `activation_stats()` | rms+absmax of an activation | right quantity, **zero callsites** -- dead code |
| `.claude/skills/numerics_debugging/` | per-op ATen activations | overkill; reportedly broken under DTensor |

`scripts/oneoff/fp32_norms_ablation.py` is the closest structural template:
same "vary one thing, N arms, same seed, dump JSON" shape.

---

## 5. Risks, in the order they are likely to bite

### 5.1 muP as published does not transfer on a modern Llama recipe

This is the finding that should most temper expectations. Three independent
sources: Lingle ([arXiv 2404.05728](https://arxiv.org/abs/2404.05728)), u-muP
([arXiv 2407.17465](https://arxiv.org/abs/2407.17465)), and
[arXiv 2510.19093](https://arxiv.org/abs/2510.19093).

Lingle's ablation table marks transfer as **broken** by:

| feature | agpt has it? |
|---|---|
| trainable RMSNorm gains | **yes** |
| weight decay | **yes** (`wd=0.1` on everything) |
| `1/sqrt(d)` attention | **yes** (today) |

and **preserved** under SwiGLU, GQA/MQA, projection biases, cosine schedule,
zero query init, and 4x batch changes -- so the architecture is otherwise fine.

u-muP's prescription: remove trainable norm parameters, use fully decoupled
AdamW, and stay in the under-fitting regime. [arXiv 2510.19093](https://arxiv.org/abs/2510.19093)
goes further and argues weight decay, not muP, is what actually stabilizes
cross-width dynamics after the first few steps.

**Consequence for planning:** budget for the coordinate check to FAIL on the
first attempt with norms and weight decay left as they are, and treat
"which of the three do we change" as an experimental question the check
answers cheaply.

### 5.2 Muon and Mano already apply a competing width rule

Both partition parameters by **shape** (`p.ndim == 2 and max(p.shape) <=
10000`) and rescale the LR by `0.2 * sqrt(max(A, B))`
(`experiments/ezpz/optimizer/muon.py:130-139`, `mano.py:141`). VERIFIED
multipliers at 30B dims: 15.68x at (6144, 6144), 25.60x at (6144, 16384),
101.22x at (256128, 6144).

Stacking muP's `1/m` on top of that yields the product of two width rules, not
muP. Worse, the `max(shape) > 10000` cutoff is an implicit width-dependent
partition: at 30B, `hidden_dim=16384` means `w1`/`w3` fall to the AdamW branch
while `wq`/`wo` stay on Muon -- and which side a tensor lands on **changes as
width scales in a sweep**.

Per [arXiv 2602.20937](https://arxiv.org/abs/2602.20937) (Gupta, Ngom,
Foreman, Vishwanath -- likely prior art in-house), the correct hidden-weight
LR scaling by optimizer family is:

| optimizer | scaling |
|---|---|
| AdamW, ADOPT, Sophia | `Theta(1/n)` |
| LAMB, Muon | `Theta(1)` |
| Shampoo | `Theta(sqrt(n_l/n_{l-1}))` |

So Muon needs `Theta(1)` -- no muP LR scaling at all -- because its update
normalization already encodes the spectral condition. `dist_muon.py:1810`
already implements `spectral_unclamped = sqrt(d_out/d_in)`, which is the
Bernstein-derived muP-correct form; the ezpz Muon defaults to Kimi's
`0.2*sqrt(max(A,B))` instead.

**Scope decision: do muP for AdamW first.** Muon is a separate parametrization
question, not a second arm of the same one.

### 5.3 Do not use the `mup` package

It attaches `p.infshape` Python attributes to parameters. Those do not survive
`torch.save` ([pytorch#72129](https://github.com/pytorch/pytorch/issues/72129)),
are stripped by `DataParallel`, and have no FSDP story
([mup#59](https://github.com/microsoft/mup/issues/59),
[#72](https://github.com/microsoft/mup/issues/72), both open). GPT-NeoX had to
hand-roll a `MuReadout` replacement because pipeline parallelism broke it.

The Cerebras design -- everything derived from config scalars at construction
time, no per-tensor attributes -- transplants cleanly onto our config-driven
`param_init` dicts and survives FSDP/TP wrapping.

### 5.4 Smaller ones

- **Adam epsilon.** Gradients shrink with width, so for fixed `eps` there is a
  width at which `eps` dominates and Adam stops being scale-invariant
  ([arXiv 2407.05872](https://arxiv.org/abs/2407.05872) 4.3). Current
  `eps=1e-8`. Relevant given the documented 80B bf16 residual-overflow NaN.
- **Weight tying is incompatible** with separate embedding/unembedding LR
  groups -- `named_parameters()` dedupes, so `lm_head.weight` vanishes
  entirely (VERIFIED). agpt production is untied; exclude `agpt_2b_tied`.
- **The LR finder flattens all groups to a single LR**
  (`lr_finder.py:142-143, 194-195`) -- incompatible with muP grouping as
  written.
- **`ezpz/trainer.py:941` logs only group 0's LR**; base `trainer.py:779` uses
  the full `get_metrics()`. Under muP the ezpz trainer would under-report.
- **`TorchMuonOptimizersContainer` bypasses the pattern mechanism entirely**
  (`containers.py:212-307`); its own docstring says the pattern is
  "effectively ignored".
- **Treat a muP run as a fresh chain, not a resume target.** Optimizer state
  is FQN-keyed so regrouping *should* survive DCP, but this repo has a
  documented history of optimizer-state-layout resume breakage and it is
  untested.
- **muP is necessary, not sufficient**, and a flat coordinate check does not
  guarantee a win: [mup#76](https://github.com/microsoft/mup/issues/76)
  reports flat checks to 15 steps with no performance improvement at 1.5B.

---

## 5.5 Stages 1-3 are built, and there is a gap between them

Implemented 2026-08-30:

- **`scripts/mup_coord_check.py`** -- the harness. Verified to reproduce the SP
  divergence signature: at 4 widths x 8 steps, `layers.5.attention` has slope
  1.42 while the width-independent `tok_embeddings` control sits at 0.001.
  Every gate is negative-tested, including a synthetic flat sweep that reports
  "HARNESS FAILURE, not a muP pass" and exits 2.
- **`agpt/mup.py`** -- the parametrization. Six flavors (`mup_1536` /
  `mup_3072` / `mup_6144` plus a CPU-sized `mup_tiny_*` ladder), 18 regression
  tests. Verified: across 1536 -> 6144, lm_head std scales 4.000 (muP wants
  `1/d`) and hidden 2.000 (wants `1/sqrt(d)`). All 56 pre-existing flavors are
  fingerprint-identical.

**They do not yet connect.** The harness varies width by calling
`_build_agpt_config` directly (`mup_coord_check.py:208`) and never references
`agpt_configs` or `register_mup_flavors` -- confirmed by grep, 0 occurrences of
either. So every rung it builds is standard parametrization, and there is no
flag that makes it emit a muP model. **The coordinate check cannot validate muP
until that bridge exists**, and a run today measures only what SP does.

This is a real gap, not a naming detail: the two halves were built
concurrently against the same design and each is correct alone. Bridging it
means either teaching the harness to build through the registered muP flavors,
or giving `_build_agpt_config` a muP switch that the harness can pass through.
The former is cleaner -- the flavors already pin the ladder geometry (L=64,
head_dim=128, H/dim=2.667, vocab=100352) that the harness currently takes as
loose flags, so routing through them also removes a class of
mismatched-geometry error.

Two operational notes for whoever runs it:

- **`TORCH_DEVICE=cpu` is required.** Without it the trainer resolves a CUDA
  device and dies with `AttributeError: module 'torch._C' has no attribute
  '_cuda_setDevice'` on this XPU box.
- **Step count matters more than it looks.** A 6-step run gives max slope 0.114
  against a 0.05 tolerance -- a pass, but only just. The 8-step run gives 1.42.
  Divergence accumulates, so a short check understates it and could read as a
  muP pass when it is really a too-short run. Use at least 8 steps.

## 5.6 FIRST COORDINATE CHECK RUN: hidden layers pass, the readout does not

Bridged the harness to the muP builders (`--mup`, `--mup-base-dim`,
`--mup-independent-wd`) and ran it. 4 widths x 8 steps, lr 1e-4, login node,
CPU. Same geometry both times, so the columns are directly comparable:

| module | SP slope | muP slope |
|---|---:|---:|
| `layers.5.attention` | **1.42** | **-0.010** |
| `layers.4.attention` | 1.32 | 0.020 |
| `layers.3.attention` | 1.13 | (flat) |
| `layers.5.feed_forward` | -- | -0.011 |
| `lm_head` | 0.22 | **-0.130** |
| `tok_embeddings` [control] | 0.001 | 0.001 |

Slope is `d log2(l1) / d log2(width)`; 0 is what muP promises. Tolerance 0.05.

**The hidden layers pass.** Attention went from 1.42 to -0.010 and the
feed-forwards sit at ~0.000. That is the `eta/m` LR grouping doing exactly
what it is supposed to do, and it is the bulk of the parametrization.

**The readout fails, and it fails in the interesting direction.** `lm_head` is
at -0.130: **negative**, meaning over-scaled DOWN, not left un-scaled.
Coordinates fall 0.0696 -> 0.0542 as width grows 8x. Its absolute magnitude
also dropped ~10x from SP (0.785 -> 0.070), which confirms the `d^-1` init is
taking effect -- the problem is not that the change did not land.

A negative slope on the readout specifically is what you would expect if the
`d^-1` init and the `O(1)` LR group are BOTH applying where Table 8 wants the
scaling split differently between init and forward multiplier. This is exactly
the uncertainty flagged in section 4: whether init-plus-LR-group fully
substitutes for the forward multiplier under Adam. The coordinate check says
not quite -- which is what it is for.

**Independent weight decay is not the cause.** `--mup-independent-wd` gives an
identical 0.130. Sensible on reflection: over 8 steps at lr 1e-4 the decay term
moves almost nothing. Section 5.1's warning about weight decay breaking
transfer is about full training runs, not an 8-step check, so this does not
contradict it -- it just means WD is not the knob that fixes the readout.

### What to try next, in order

1. **The forward-multiplier form.** Keep `std = d^-1/2` and apply Table 8's
   `1/m` multiplier on the logits instead of folding it into the init. Route
   (a) in section 4 -- the audit preferred route (b) for blast radius, and the
   measurement now argues for (a).
2. **Readout LR.** If the multiplier form alone does not flatten it, the
   readout group's `O(1)` LR may need to be `O(1/m)` in combination with the
   `d^-1` init. Cheap to test: two runs.
3. **Only then** look at trainable norm gains and the attention scale. Norms
   are already flat (-0.000) and the attention knob is inert on a fixed
   head_dim ladder, so neither is implicated by this data.

Note the failing modality is a single module with a clean, reproducible signal
-- not a diffuse failure. That is a good position: the remaining work is
bounded.

## 5.7 STAGE 4: the two upper rungs transfer; the base rung needs a wider grid

Three runs. The first two are recorded because each was a wrong verdict the
harness produced at rc=0, and the gates that now catch them were written from
these cases.

**12474329 -- grid two decades too high.** Every width bottomed out at the
leftmost point (1536: 5.485, 3072: 5.437, 6144: 5.308) with loss rising
monotonically from there. The harness reported "same argmin at every width --
TRANSFER". Three identical boundary artifacts, not three measurements. Cause:
the grid was reused from the tiny-ladder rehearsal, which is 6 layers where
this ladder is 64, and depth pushes the usable LR band down.

**12474330 -- corrected grid, real minima at the top two rungs:**

| eta | 1536 | 3072 | 6144 |
|---:|---:|---:|---:|
| 1e-6 | 9.826 | 9.540 | 9.158 |
| 4e-6 | 8.713 | 8.059 | 7.345 |
| 1.6e-5 | 7.322 | 6.544 | 6.176 |
| **6.4e-5** | 5.868 | **5.521** | **5.505** |
| 2.56e-4 | **5.672** | 6.010 | 5.966 |

**3072 and 6144 both minimize at 6.4e-5**, with genuine interior minima --
curves turning up on both sides. A 2x width step moved the optimum not at all,
and 6144 IS the production 30B geometry. That is the transfer signal.

1536 is pinned to the right edge, so its optimum is outside the grid and the
boundary gate refuses the run. Correctly: a three-rung claim cannot rest on a
rung that never bracketed its minimum.

**12474332 -- grid extended right** (1.6e-5 .. 1.024e-3) to bracket 1536,
dropping the two leftmost points that were far up the slope at every width.

### What can be said now, and what cannot

**Can:** between 3072 and 6144, at fixed head_dim=128 / L=64 / H/dim=2.667,
the muP optimum does not move. Those are the two widths closest to production
and the ones a transfer claim would be used for.

**Cannot:** that muP transfers across the full 4x ladder. 1536's optimum is
unmeasured. Nor that it transfers at production TOKEN scale -- these are
60-step runs at seq 2048, and the failure modes documented in section 5.1
(trainable norm gains, weight decay) are about full training runs.

**Also worth holding onto:** run 12474329 showed the blow-up threshold moving
sharply with width in the upper LR range -- at 6.4e-3, loss 7.16 / 12.71 /
41.53 across the ladder. Under working muP that should be roughly
width-invariant. The optimum transferring while the divergence threshold does
not is a real tension, not yet resolved.

## 6. Staged plan

Each stage is gated on the previous one and produces a decision, not just an
artifact. Stages 1-3 need **no allocation** -- they run on a login node.

**Stage 1 -- coordinate-check harness (no allocation).**
Build the harness against the CURRENT parametrization and confirm it
reproduces the SP divergence signature. Deliverable: a script plus a baseline
plot showing coordinates growing with width. Gate: if the harness cannot
reproduce SP divergence, it cannot validate muP either, and nothing downstream
is trustworthy.

**Stage 2 -- width ladder (no allocation).**
Register `mup_1536` / `mup_3072` / `mup_6144` via `_build_agpt_config`, with
exact `H/dim = 2.667`. Verify parameter counts differ and that `head_dim=128`
holds at every rung. Gate: the decorative-`dim` trap in 4.2 must be proven
absent -- three genuinely different models.

**Stage 3 -- muP implementation (no allocation).**
Items 1-3 and 5 from section 3. Run the coordinate check. **Expect it to fail
first time** per 5.1. Iterate on the three suspects -- norms, weight decay,
attention scale -- one at a time. Gate: flat coordinates across all three
widths for at least 10 steps.

**Stage 4 -- LR transfer verification (small allocation).**
Sweep LR on `mup_1536`, pick the optimum, apply the muP rule to predict
`mup_6144`'s optimum, then sweep `mup_6144` and check the prediction lands.
This is the actual payoff and the first stage that needs nodes. Gate: the
predicted optimum within one grid step of the measured one.

**Stage 5 -- 30B arm (full allocation).**
Only if stage 4 transfers. Run 30B under muP against the existing AdamW
baseline at 23.59B tokens, which is already measured (loss 2.51357) and is a
clean comparison point.

Stages 1-3 are the bulk of the work and cost nothing but time. **Stage 4 is
the go/no-go**, and it is cheap relative to the 30B runs already completed.

---

## References

- Yang & Hu, Tensor Programs V ([arXiv 2203.03466](https://arxiv.org/abs/2203.03466)) -- Table 8 is the implementable form
- Yang, Simon, Bernstein, spectral condition ([arXiv 2310.17813](https://arxiv.org/abs/2310.17813)) -- why Muon is muP-satisfying by construction
- Gupta, Ngom, Foreman, Vishwanath ([arXiv 2602.20937](https://arxiv.org/abs/2602.20937)) -- muP across AdamW/ADOPT/Sophia/Muon/LAMB/Shampoo
- Lingle ([arXiv 2404.05728](https://arxiv.org/abs/2404.05728)) -- the ablation table; what breaks transfer
- u-muP ([arXiv 2407.17465](https://arxiv.org/abs/2407.17465)) -- why TP-V's setup flattered itself, and the fix
- Everett et al. ([arXiv 2407.05872](https://arxiv.org/abs/2407.05872)) -- per-layer SP beats muP; Adam epsilon
- [arXiv 2510.19093](https://arxiv.org/abs/2510.19093) -- weight decay may matter more than muP
- Essential AI ([arXiv 2505.02222](https://arxiv.org/abs/2505.02222)) -- muP verified for Muon to 3.7B
