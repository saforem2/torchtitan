# RoPE flavor mismatch: a mid-flight convention switch, and the exports it broke

> **Which STEPS are in which convention? -> [Registry](#registry-which-steps-are-in-which-convention).**

> **Canonical record** for this failure mode. Found 2026-08-16; the page has
> been wrong twice and is now rewritten against W&B `metadata.args` and the
> on-disk eval results. See [Corrections](#corrections) for what was wrong and
> why. **Three of four v2 production chains changed RoPE convention
> mid-flight.** The weights are healthy; the **HF exports are not**.

## Verdict

**The 20B chains are usable as training runs, and their checkpoints are
sound.** MEASURED: at each switch the training loss spiked (2.57 -> 6.03 on
20B-512, 2.69 -> 6.14 on 20B-256) and then re-converged to within +0.01 of the
pre-switch trend line inside ~300-500 steps, and the in-process validator loss
(which uses the correct RoPE by construction) has fallen monotonically ever
since. Nothing about the stored weights is corrupt.

**What IS damaged is every HF export made from a cos_sin step.** MEASURED:
`eval-20b-v2.sh:132` hardcodes `--model_flavor "20b"` (complex), and the v2
production clones ship the bare `Llama3StateDictAdapter`, which applies the Q/K
permute unconditionally. So every 20B eval of a cos_sin checkpoint -- and the
2B-512 evals after step 30400 -- was run on a wrongly-permuted export. The
signature is unmistakable and replicates across chains: ARC-Challenge decays
monotonically (20B-512 0.3823 -> 0.2628; 2B-512 0.3456 -> 0.2381) while train
and validator loss keep improving.

**So: trust the loss curves and the checkpoints; discard the post-switch eval
numbers and re-run them with the correct flavor.** The published post-switch
eval tables for 20B-512, 20B-256, and 2B-512 understate those models. 2B-256
never switched and is unaffected.

## The mechanism

`convert_to_hf.py` decides the RoPE convention from the `--model_flavor` you
pass, **not from the checkpoint**. It cannot do otherwise: the rope cache is
registered `persistent=False` (`torchtitan/models/common/rope.py:115`), so
nothing about the convention reaches disk. A checkpoint trained with cos_sin
RoPE and one trained with complex RoPE are byte-indistinguishable.

`AgptStateDictAdapter` (`agpt/state_dict_adapter.py:39-62`) branches on it:

```python
rope = model_config.layers[0].attention.rope     # from the FLAVOR you passed
self._is_cos_sin = isinstance(rope, CosSinRoPE.Config)
...
def to_hf(self, state_dict):
    if not self._is_cos_sin:
        return super().to_hf(state_dict)   # applies the Q/K permute
    # cos_sin: HF-native rotate_half layout, NO permute
```

The permute is not a subtlety. MEASURED, reproducing
`Llama3StateDictAdapter._permute` for the 20B shape (n_heads=40, head_dim=128,
dim=5120), head-0 output rows map to input rows
`[0, 64, 1, 65, 2, 66, 3, 67, ...]` -- the interleaved <-> rotate_half
repairing. Applying it to weights that are already rotate_half-native pairs
each channel with the wrong partner. The export **loads without error** --
shapes and key names are identical -- and only shows up as degraded quality.

**Two things must agree, and they are in different files:** how the run was
launched (`--config=agpt_20b` vs `agpt_20b_real`) and how the checkpoint is
converted (`--model_flavor 20b` vs `20b_real`). Nothing enforced the pairing.

## Registry: which STEPS are in which convention

**This table is the canonical answer, and it is per-STEP, not per-chain.**
Derived from `--config=` in each run's W&B `metadata.args` (the literal argv the
process executed), cross-checked against checkpoint directory mtimes. **Not**
from submit-script defaults, which is how this page got it wrong twice.

| chain | ckpt dir | COMPLEX steps | COS_SIN steps | switch date |
|---|---|---|---|---|
| `20b_v2_512` | `agpt-20b-...-n512-gbs12288` | `<= 4400` | `>= 4401` (head 9,100) | 2026-07-05 |
| `20b_v2_256` | `agpt-20b-...-n256-gbs6144` | `<= 3100` | `>= 3101` (head 10,300) | 2026-07-10 |
| `2b_v2_512`  | `agpt-2b-...-n512-gbs12288`  | `<= 30400` | `>= 30401` (to 46,429 DONE) | 2026-07-10 |
| `2b_v2_256`  | `agpt-2b-...-n256-gbs6144`   | **all** (1..92,859) | none | never switched |
| `80b` | -- | all | none | n/a (compile OFF; `_real` is a compile win, moot) |
| MDS / Megatron-DeepSpeed | -- | n/a -- different codebase | n/a | convert with `2b-mds` |

MEASURED boundary evidence, per chain (last complex-run step vs first
cos_sin-run step, with the first cos_sin loss and grad-norm):

| chain | last complex run | first cos_sin run | first cos_sin loss / grad_norm |
|---|---|---|---|
| `20b_v2_512` | `0pmsn01c` ..4418 (2.5669) | `tu1iseu1` 4401.. | **6.0298** / **19.95** |
| `20b_v2_256` | `rugscgjs` ..3135 (2.6854) | `6yr6ivh4` 3101.. | **6.1367** / **17.31** |
| `2b_v2_512`  | `nv4qwxc8` ..30483 (2.7085) | `9d10mqwb` 30401.. | **6.7674** / **27.72** |

Normal grad_norm on these chains is 0.12-0.35, so the switch step is a 50-100x
grad-norm excursion. Checkpoint mtimes agree: 20B-512 `step-4400` was written
2026-05-29, `step-4500` on 2026-07-05; 20B-256 `step-3100` on 2026-07-05,
`step-3200` on 2026-07-10; 2B-512 `step-30400` on 2026-05-28, `step-30500` on
2026-07-10.

**Exhaustive cross-check (MEASURED).** Rather than trust `trajectories.py`, all
2,024 runs in `aurora_gpt/torchtitan.ezpz.train` since 2026-04-20 were scanned
and bucketed by `--checkpoint.folder`. That finds far more runs per chain than
the registry lists (20B-512: 56, 20B-256: 42, 2B-512: 45, 2B-256: 33). Result:
**zero** complex-config runs after the switch date on any chain, confirming the
boundaries above are clean one-way transitions. Two apparent exceptions were
checked against disk and are **not** exceptions:

- `b3oacbp8` / `eec9nu1r` (20B-512, 2026-07-01, `agpt_20b_real`, 14 steps and 0
  steps, both crashed) predate the 07-05 switch. No `step-*` directory in that
  chain has an mtime between 2026-07-01 and 2026-07-05, so neither wrote a
  checkpoint; the 4400/4500 boundary holds.
- `am4yzmlu` (2B-256, 2026-07-10, `agpt_2b_real`, `finished`, no steps logged)
  is the lone `_real` run against the 2B-256 folder. That chain's newest
  checkpoint is `step-92859` written 2026-06-29, and **no** `step-*` directory
  there has an mtime after 2026-06-30 -- so `am4yzmlu` wrote nothing and
  2B-256 remains complex end to end.

### How to check a chain yourself

```bash
# authoritative: the argv the process actually ran
python3 torchtitan/experiments/ezpz/scripts/eval/rope_flavor_for_step.py \
    --chain 20b_v2_512 --step 4500
```

W&B `metadata.args` is ground truth. **Do not infer from a clone's
`CONFIG_SUFFIX`, from a submit script, or from a clone's `config_registry.py`**
-- a clone carries several submit scripts and the one you read may not be the
one that ran. Both prior versions of this page were wrong for exactly that
reason.

Caveat (MEASURED): the resolver reads its run-id lists from
`trajectories.py`, which is **incomplete** for `2b_v2_512` -- it omits
`9d10mqwb` and `n887c3lk`, so the resolver reports that chain's switch as
2026-08-05 / `vtumb5cb` when the real switch was 2026-07-10 / `9d10mqwb` at
step 30401. Trust this table over the resolver until those run-ids are added.

To avoid depending on the registry at all, bucket every run by its
`--checkpoint.folder` instead of by a curated list:

```python
# runs are ~2k; filter by createdAt to keep it quick
for r in api.runs("aurora_gpt/torchtitan.ezpz.train",
                  filters={"createdAt": {"$gte": "2026-04-20T00:00:00"}}):
    args = (r.metadata or {}).get("args") or []
    ck  = next((a.split("=",1)[1] for a in args if a.startswith("--checkpoint.folder=")), None)
    cfg = next((a.split("=",1)[1] for a in args if a.startswith("--config=")), None)
```

Then confirm any suspicious run actually wrote weights, since a crashed or
zero-step run cannot change a checkpoint's convention:

```bash
find <ckpt_dir> -maxdepth 1 -name 'step-*' -newermt '<run date>' \
     ! -newermt '<next date>' -printf '%f %TY-%Tm-%Td\n'
```

## The mid-flight switch -- what happened

`5ffb850a1` (2026-06-25, "ezpz(scripts): default _real RoPE (2B/20B) + enable
validator (all)") set `CONFIG_SUFFIX="${CONFIG_SUFFIX-_real}"` in the 2B and 20B
autoretry submit scripts. The change was smoke-validated on a *fresh* 2N run,
where cos_sin is simply a faster equivalent. Nobody considered what it does to
a chain **resuming from complex-trained weights**: `ComplexRoPE` and
`CosSinRoPE` pair head dimensions differently, so the same weights compute a
different function. The running chains picked it up on their next resume --
which is why the three chains switched on three different dates (whenever each
next resumed), not on the commit date.

### Loss evidence: recovery, quantified

MEASURED. A least-squares slope was fit to the last 400 steps of the
pre-switch (complex) run and projected forward; `delta` is
`actual_cos_sin - projected_complex_trend`.

`20b_v2_512` (pre-switch trend: mean 2.5308, slope -4.75e-05/step)

| step | cos_sin loss | complex-trend projection | delta |
|---|---|---|---|
| 4418 | 2.8539 | 2.5213 | +0.3326 |
| 4500 | 2.5727 | 2.5174 | +0.0553 |
| 4700 | 2.5204 | 2.5079 | +0.0125 |
| 4900 | 2.5007 | 2.4984 | **+0.0023** |

`20b_v2_256` (pre-switch trend: mean 2.6950, slope -1.21e-04/step)

| step | cos_sin loss | complex-trend projection | delta |
|---|---|---|---|
| 3135 | 2.8355 | 2.6708 | +0.1647 |
| 3300 | 2.6852 | 2.6509 | +0.0343 |
| 3602 | 2.6259 | 2.6143 | **+0.0116** |

`2b_v2_512` (pre-switch trend: mean 2.7086, slope ~0)

| step | cos_sin loss | complex-trend projection | delta |
|---|---|---|---|
| 30401 | 6.7674 | 2.7087 | +4.0588 |
| 30500 | 2.8672 | 2.7087 | +0.1585 |
| 32000 | 2.7244 | 2.7091 | +0.0153 |
| 35046 | 2.7105 | 2.7100 | **+0.0005** |

All three re-converge onto the pre-switch trend and resume the same descent
slope. The switch cost a few hundred steps of progress; it did not change where
the chain was heading.

**Control (MEASURED): same-flavor resumes show no discontinuity at all.**
`17sfemjj` -> `rugscgjs` (both complex) hands off 2.85581 -> 2.85531 at step
2104-2107; every cos_sin-to-cos_sin resume behaves the same. The spike is
specific to the convention change, not to resuming.

## Q1-Q5 findings

### Q1. Did the 20B chains recover, or are they permanently damaged? -- MEASURED

**Recovered, as training runs.** Loss returns to the pre-switch trend line
within +0.0023 (20B-512, by step 4900) and +0.0116 (20B-256, by step 3602) and
resumes the same slope; see the tables above.

Independent confirmation from the **in-process validator** (added at the same
boundary), which computes held-out loss inside the training process with the
correct RoPE and so is immune to any conversion bug:

| run | chain | validator loss, first -> last |
|---|---|---|
| `tu1iseu1` | 20b_v2_512 | 2.6235 (s4500) -> 2.5482 (s5100) |
| `c8zwrlqw` | 20b_v2_512 | 2.6111 (s7700) -> 2.4731 (s8500) |
| `6yr6ivh4` | 20b_v2_256 | 2.7390 (s3200) -> 2.6604 (s3600) |
| `82e1jewm` | 20b_v2_256 | 2.6140 (s7900) -> 2.5615 (s8300) |
| `9d10mqwb` | 2b_v2_512  | 2.8725 (s30500) -> 2.7149 (s35000) |
| `nowkdepb` | 2b_v2_512  | 2.7008 (s41400) -> 2.6979 (s43800) |

Monotone improvement on held-out data in every cos_sin-era run. The models are
learning normally.

**Cost, INFERRED:** roughly 300-500 steps of wasted progress per switch, plus
whatever the transient large gradients (grad_norm 17-28 for a few steps) did to
the optimizer state. The latter is not separately measurable from these logs;
the loss recovery bounds it as small.

### Q2. Why did 2B-512 not spike? -- MEASURED: **it DID spike. The premise was wrong.**

The apparent no-spike was an artifact of a **gap in `trajectories.py`**. That
file lists `nv4qwxc8` (..30483, complex) followed by `vtumb5cb` (39601..,
cos_sin) -- a 9,118-step hole. Two runs that fill the hole are **missing from
the registry**:

| run | created | config | steps | first loss / grad_norm |
|---|---|---|---|---|
| `oijbc7yb` | 2026-07-10 11:13 | `agpt_2b` | none logged (crashed) | -- |
| **`9d10mqwb`** | 2026-07-10 16:07 | **`agpt_2b_real`** | 30401..35046 | **6.7674 / 27.72** |
| `n887c3lk` | 2026-07-17 13:43 | `agpt_2b_real` | 35001..39603 | 2.7147 / 0.34 |

So the real 2B-512 transition is **2026-07-10 at step 30401**, with the
**largest spike of all three chains** (6.77, grad_norm 27.7). `vtumb5cb` looked
smooth because by 2026-08-05 it was resuming from a checkpoint that was
**already cos_sin** -- hypothesis (c) from the brief, confirmed.

Hypotheses (a) and (b) are **excluded**: MEASURED, no run on any chain passes
`--checkpoint.initial-load-path` (checked in every run's `metadata.args`), and
`--checkpoint.folder` is unchanged across every boundary, so there was no
re-seed; and the 2B is not more tolerant -- it spiked hardest.

### Q3. Is there eval evidence of damage? -- MEASURED: yes, and it is the main finding

MEASURED: `eval-20b-v2.sh:132` hardcodes `--model_flavor "20b"`. It converts
from `V2_REPO` (`runs/agpt-20b-v2/torchtitan-ezpz`), whose `agpt/__init__.py:626`
registers `state_dict_adapter=Llama3StateDictAdapter` -- the unconditional
permute; that clone has no `state_dict_adapter.py` at all. So **every** 20B eval
of a cos_sin checkpoint used a wrongly-permuted export. `eval-2b-v2.sh`
defaulted to `MODEL_FLAVOR=2b` at the time the 2B-512 post-30400 evals ran.

20B-512, eval cadence 100 across the switch at 4401:

| step | conv | HellaSwag | ARC-E | ARC-C | PIQA | Wino |
|---|---|---|---|---|---|---|
| 4300 | complex | 0.6271 | 0.6650 | 0.3635 | 0.7601 | 0.5919 |
| **4400** | **complex** | **0.6339** | 0.6646 | **0.3823** | 0.7650 | 0.5912 |
| **4500** | **cos_sin** | **0.5915** | 0.6040 | **0.3422** | 0.7481 | 0.5691 |
| 5300 | cos_sin | 0.6060 | 0.6402 | 0.3541 | 0.7655 | 0.5848 |
| 6500 | cos_sin | 0.6086 | 0.6288 | 0.2875 | 0.7655 | 0.5793 |
| 7600 | cos_sin | 0.6107 | 0.6338 | **0.2628** | 0.7644 | 0.5635 |

A step-change at the boundary (HS -4.2pp, ARC-E -6.1pp, ARC-C -4.0pp), then
HellaSwag partially recovers but **plateaus ~2.3pp below** its pre-switch value
3,200 steps later, while ARC-C **keeps decaying monotonically** 0.3823 -> 0.2628
-- even though train and validator loss improve throughout.

**Replication on 2B-512** (switch at 30401), same signature:

| step | conv | HellaSwag | ARC-E | ARC-C | PIQA |
|---|---|---|---|---|---|
| 30000 | complex | 0.5319 | 0.5947 | 0.3456 | 0.7138 |
| 41000 | cos_sin | 0.4785 | 0.5551 | 0.2474 | 0.7018 |
| 46429 | cos_sin | 0.4753 | 0.5589 | **0.2381** | 0.6997 |

**Control -- 2B-256, which never switched**, over the same token range: no dip
anywhere; HellaSwag climbs 0.5319 (s30k) -> 0.5598 (s76k), ARC-C holds ~0.32-0.33.

**Token-matched control, the cleanest comparison available.** Same model, same
corpus, same optimizer, both at ~4.674T tokens:

| chain | step | tokens | HellaSwag | ARC-C |
|---|---|---|---|---|
| `2b_v2_256` (complex throughout) | 76,100 | 3.83e12 | **0.5591** | **0.3302** |
| `2b_v2_512` (cos_sin after 30400) | 46,429 | 4.67e12 | 0.4753 | 0.2381 |
| `2b_v2_512` **re-converted `2b_real`** | 46,429 | 4.67e12 | **0.5384** | **0.2978** |

The complex-only chain appeared to score ~8pp HellaSwag and ~9pp ARC-C **higher
on fewer tokens** -- while the switched chain's own train loss (2.69) and
validator loss (2.698, still falling) said it was the healthier-trained model.
**Loss fine + converted-eval depressed = the damage is in the conversion, not
the weights.**

**CONFIRMED 2026-08-16 (job `8760307`, MEASURED).** Re-converting that exact
checkpoint with `2b_real` recovers **+0.063 HellaSwag and +0.060 ARC-C**,
closing roughly three quarters of the apparent gap (8.4pp -> 2.1pp on
HellaSwag, 9.2pp -> 3.2pp on ARC-C). The residual is expected and does not
need a bug to explain it: 2B-256 has a 2x smaller global batch at the same
token count, which the campaign already measured as a per-token advantage
(see `project_2b_large_batch_undertraining`). **This row should no longer be
cited as evidence that 2B-256 beat 2B-512** -- most of that gap was the
permute.

MMLU moved much less: 0.2511 -> **0.2579** (+0.007), still inside noise of the
0.25 floor. The asymmetry is consistent with the 20B result -- corruption costs
most on tasks the model actually learned, and MMLU was never learned.

Cadence caveat: on **20B-256** the eval cadence around the switch is 1,000 steps
(3000 then 4000), so the recovery window there is **UNRESOLVED** -- no
switch-localized dip can be seen or excluded. Its ARC-C shows the same
long-run decay (0.3567 -> 0.2799).

Honest limit: HellaSwag *rises* across the switch on 20B-256 and partially
recovers on 20B-512, so the wrong-permute export is **degraded, not destroyed**
-- INFERRED, a fully scrambled basis could not produce a rising HellaSwag
curve. The likely reason is that lm-eval's multiple-choice scoring is a
loglikelihood ranking, which is more robust to a partially-wrong attention
basis than free generation is. **UNKNOWN:** the exact magnitude of the
understatement. Only a re-conversion with `20b_real` and a re-eval of the same
steps will settle it -- that is the experiment to run.

### Q4. Which checkpoints on disk are affected? -- MEASURED

Step ranges are in the [registry](#registry-which-steps-are-in-which-convention).
Restated as a conversion rule:

| ckpt dir | steps | convert with |
|---|---|---|
| `agpt-20b-...-n512-gbs12288` | `step-100` .. `step-4400` | `--model_flavor 20b` |
| `agpt-20b-...-n512-gbs12288` | `step-4500` .. `step-9100` | `--model_flavor 20b_real` |
| `agpt-20b-...-n256-gbs6144` | `step-100` .. `step-3100` | `--model_flavor 20b` |
| `agpt-20b-...-n256-gbs6144` | `step-3200` .. `step-10300` | `--model_flavor 20b_real` |
| `agpt-2b-...-n512-gbs12288` | `step-100` .. `step-30400` | `--model_flavor 2b` |
| `agpt-2b-...-n512-gbs12288` | `step-30500` .. `step-46429` | `--model_flavor 2b_real` |
| `agpt-2b-...-n256-gbs6144` | all (`step-35600` .. `step-92859` on disk) | `--model_flavor 2b` |

The checkpoints themselves are fine in both ranges -- "affected" means only that
the correct `--model_flavor` differs by step. Note the v2 production clones
cannot perform a correct cos_sin conversion at all: they lack
`state_dict_adapter.py` and their `config_registry.py` builds `_real` via a
`rope.backend="cos_sin"` literal on a single `RoPE` class, so there is no
`CosSinRoPE.Config` for the adapter to detect. **Re-conversions must run from a
clone that has `AgptStateDictAdapter`** -- do not just pass `20b_real` to the v2
clone and assume it worked.

### Q5. Did anything else change at the same boundary? -- MEASURED, and yes

This is the confound check, and it is a real one. Full `metadata.args` diffs:

**`20b_v2_256`, `rugscgjs` (19 args) -> `6yr6ivh4` (24 args)**
- removed: `--config=agpt_20b`
- added: `--config=agpt_20b_real`, `--validator.enable`, `--validator.freq=100`,
  `--validator.steps=10`, `--validator.dataloader.dataset-path=...`,
  `--validator.dataloader.data-cache-path=...`

**`20b_v2_512`, `0pmsn01c` (18) -> `tu1iseu1` (24)**
- same as above, plus `--dataloader.num-workers=2`

**`2b_v2_512`, `nv4qwxc8` (18) -> `9d10mqwb` (22)**
- removed: `--config=agpt_2b`, `--debug.print-config`
- added: `--config=agpt_2b_real` + the same five validator flags

Everything else is byte-identical: optimizer (`sophiag`), `--optimizer.lr=2.28e-5`,
`--training.local-batch-size=2`, global batch size, `--training.seq-len=8192`,
`--training.steps`, dataset, `--checkpoint.folder`, `--checkpoint.interval=100`.
No `--checkpoint.initial-load-path` on any run.

**So the validator IS a confound sitting right at the boundary -- and it is
excluded by the data.** The validator only reads a held-out split and logs a
number; it does not touch the training gradient. Three independent
discriminators:

1. **`2b_v2_512`'s `vtumb5cb` boundary** (2026-08-05) has the validator on
   *both* sides and no config change -- loss is continuous (2.7085 -> 2.7027).
   Validator-on is therefore not sufficient to cause a spike.
2. **Same-flavor resumes with the validator already enabled** are all smooth
   (e.g. `cxlt0tpe` -> `2ktrz29u`, `2ktrz29u` -> `82e1jewm`).
3. The spike is a **single-step** grad-norm excursion of 17-28 that decays
   within ~10 steps -- the signature of the *weights* suddenly computing a
   different function, not of an extra eval pass.

`--dataloader.num-workers=2` (20B-512 only) and dropping `--debug.print-config`
(2B-512 only) are not present on all three chains, so neither can explain a
spike common to all three. **The RoPE change is the only variable that
covaries with the spike on every chain.** The RoPE story survives the challenge.

## Blast radius

**Corrupt -- must be redone:**
- Every HF export and lm-eval result produced from a **cos_sin** checkpoint:
  `20b_v2_512` steps >= 4401, `20b_v2_256` steps >= 3101, `2b_v2_512`
  steps >= 30401. On disk under
  `outputs/evals/{agpt-20b-v2-512n,agpt-20b-v2-256n,agpt-2b-v2-512n}/step-*`.
  These understate the models by an amount **now MEASURED** (2026-08-17,
  re-eval sweep). On `20b_v2_256`, 13 matched checkpoint pairs over steps
  4000-5800:

  | | step 4000 | step 5800 | net |
  |---|---|---|---|
  | corrupted export | 0.3481 | 0.2969 | **-0.0512** |
  | corrected export | 0.3635 | 0.3857 | **+0.0222** |

  ARC-C penalty grows +0.015 -> +0.089 (mean +0.041 first half, +0.075
  second); ARC-Easy runs +0.021 to +0.045 and does NOT grow. `20b_v2_512`
  agrees independently over steps 4000-6000.

  **The penalty is monotone in training progress**, which is worse than a
  constant offset: it inverts the trend. By step 5800 the corrupted export
  reads 0.2969 -- within noise of the 0.25 ARC-C random baseline -- while the
  same weights read 0.3857 exported correctly. A reader of the corrupted
  curve concludes the model lost most of its ARC-C ability; it gained. No
  amount of care reading those numbers recovers the truth, because a monotone
  corruption is indistinguishable from a trend. Full series:
  [`20260816-arc-c-decay-vs-rope-permute.md`](../../experiments/agpt/aurora/20260816-arc-c-decay-vs-rope-permute.md).
- Any downstream claim resting on those numbers -- in particular
  "20B-512 beats 2B-256 per token" and any post-switch capability comparison
  between `2b_v2_512` and `2b_v2_256`, which is confounded by conversion.

**Verified CLEAN (checked 2026-08-17, do not re-litigate):**
- The **Q2 INCITE report** (`summaries/2026-Q2-incite.md`). Its headline
  per-token claim -- "20B beats the 2B on every downstream benchmark, ARC-Easy
  0.46 -> 0.66 over the first ~440B tokens" -- rests on **step-4,400**, exactly
  one checkpoint before the 4,401 switch (443.0B tokens). Every eval number in
  that report is pre-switch. An external report came within a single checkpoint
  of publishing corrupted results, which is luck, not process.
- The **2B eval README**: zero rows at or past `2b_v2_512`'s step-30,401
  switch. (A `0.2969` in it coincidentally equals the 20B's corrupted endpoint;
  unrelated.)
- `evals/agpt/2b-mds/README.md`, `evals/eval-landscape-2026-07.md`: no
  post-switch citations.

**Merely discontinuous -- usable with a footnote:**
- The training-loss curves of all three switched chains. There is a real
  300-500 step spike-and-recovery at the switch step. Plots should mark it;
  the trend either side is sound.

**Fine:**
- All checkpoint weights, in both conventions. Nothing on disk is damaged.
- `2b_v2_256` end to end (complex throughout; completed 4.674T at step 92,859)
  and every eval derived from it -- it never switched, and `eval-2b-v2.sh`'s
  old `2b` default was correct *for that chain*.
- Pre-switch evals on all chains.
- 80B (complex throughout; `_real` was never defaulted for it).

## Mitigation

The checkpoint cannot be interrogated, so the fix is to make a wrong flavor
**loud and auditable** instead of silent.

**1. `eval-2b-v2.sh` now REQUIRES `MODEL_FLAVOR`** -- no default, since no
single default is correct for a chain that switches mid-flight
(`eval-2b-v2.sh:93`). It errors out and points at the resolver.

**2. `convert_to_hf.py` announces the convention it is about to use** and
records it, warning that flavor is per-STEP:

```
[convert_to_hf] flavor='2b' -> RoPE=complex (Q/K permute APPLIED). If this
does not match how the checkpoint was TRAINED, the export is silently corrupt
```

**3. Every export writes `ezpz_export.json`** beside the weights recording
`source_dcp`, `model_flavor`, `rope`, and `export_dtype`, so an existing HF
directory can be audited after the fact.

**4. `scripts/eval/rope_flavor_for_step.py`** resolves step -> flavor from W&B
`metadata.args` (`--all`, `--chain K`, `--chain K --step N`).

**Still outstanding (not yet fixed):**
- `eval-20b-v2.sh:132` **still hardcodes `--model_flavor "20b"`.** This is the
  bug that produced the corrupt 20B evals and it is live.
- `trajectories.py` is missing `9d10mqwb`, `n887c3lk`, and `oijbc7yb` from
  `2b_v2_512`, so the resolver reports that chain's switch 9,000 steps late.
- The v2 production clones cannot do a correct cos_sin conversion at all
  (no `AgptStateDictAdapter`), so re-conversions must run from the main repo.

## What would fix this properly

Record the RoPE convention **in the checkpoint** -- a marker buffer, or a field
in the training config saved alongside the shards -- and have
`convert_to_hf.py` read it and *reject* a contradicting `--model_flavor`. The
mitigation above narrows the window; it does not close it, because a caller can
still pass an explicit wrong flavor and `eval-20b-v2.sh` still does.

Second, and cheaper: **never let a submit-script default change the model
function on a resume.** A flavor change is safe on a fresh run and unsafe on a
resume; the resume path should compare the requested flavor against the one
recorded in the checkpoint and refuse to proceed on a mismatch. `5ffb850a1` was
smoke-validated on a fresh 2N job, which is precisely the case that cannot
expose this.

## Detecting a bad export you already have

Generate ~50 greedy tokens. A RoPE mismatch produces fluent-looking token
salad -- correct vocabulary, no coherence -- not a crash and not repetition.
On multiple-choice loglikelihood benchmarks it is subtler: expect a step-change
at the switch step and a slow monotone decay on the harder tasks (ARC-C) while
loss improves. If `ezpz_export.json` is absent the export predates 2026-08-16;
check the producing chain and step against the registry above.

## Corrections

Three wrong answers were given about this on 2026-08-16, all from the same
root cause -- **inferring training config from scripts instead of from the
recorded argv**:

1. **"The two completed 4.674T 2B chains trained complex."** Wrong. Read from
   `runs/agpt-2b-v2`'s *legacy* submit script, which has no `CONFIG_SUFFIX`
   assignment, while those chains actually ran under the umbrella script. The
   proposed `eval-2b-v2.sh` fix encoded this as a name-based special case and
   would have converted both chains with the wrong flavor -- the very
   corruption the page documents.

2. **"Everything agpt on Aurora is cos_sin except 80B."** Wrong in the other
   direction, and the correction to (1) overshot into it. Disproved by the
   mid-flight switch: `2b_v2_256` is complex end to end, and the other three
   chains are complex for their first several thousand steps. A per-chain table
   cannot express this; the registry is now per-STEP.

3. **"2B-512 did not spike, so the 2B is tolerant of the re-pairing."** Wrong,
   and the most instructive of the three. The premise came from
   `trajectories.py`, which is missing two run-ids; the 9,118-step gap between
   the listed runs was the tell. The 2B-512 spiked hardest of all three
   (6.7674, grad_norm 27.7) in the un-listed run `9d10mqwb`. **A gap in a
   hand-maintained registry is not evidence of absence.**

The through-line: a submit script, a clone default, and a curated run-list are
all *secondary* records that drift. W&B `metadata.args` is the argv the process
executed. Use it.

## Related

- [`polaris-20b-tokenizer-mismatch.md`](polaris-20b-tokenizer-mismatch.md) --
  a *different* silent eval-corruption mode (Llama2-tokenized data vs gemma eval
  tokenizer); not this bug.
- [`exp04-fp32-inference-investigation.md`](../../production/agpt/30b-exp/exp04-fp32-inference-investigation.md)
  -- where this was found (H3), including the separate and still-unisolated
  vLLM bf16 gibberish, which is **not** this bug.
- [`state_dict_adapter.py`](../../../agpt/state_dict_adapter.py) -- the branch.
- [`rope_flavor_for_step.py`](../../../scripts/eval/rope_flavor_for_step.py) -- the resolver.
- `agpt/__init__.py:773-774` -- `2b_real` / `20b_real` registration.
