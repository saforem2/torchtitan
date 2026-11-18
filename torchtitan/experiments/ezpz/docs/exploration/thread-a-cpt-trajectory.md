# CPT Trajectory Analysis: Is 60/40 FineWeb-Edu Learning, Plateauing, or Drifting?

> **Scope:** `ezpz-cpt111k-eval-sft` (`bcbca5bd90`, 2026-11-16). Target file:
> `torchtitan/experiments/ezpz/docs/exploration/thread-a-cpt-trajectory.md`.
> Covers the live `agpt_2b_mds_mix_owm_edu_6040` chain (60/40 Open-Web-Math /
> FineWeb-Edu via Grain) forked from `global_step138650/step-0` (MDS 2B).
> Assignment snapshot: **143.4k/190.7k (75% budget)**, training loss
> **~1.856 at ~10k** and validation loss **12.53 flat at ~111k**.
> All claims cite on-disk docs or the CPT script; inferences are labeled.

---

## 0. Executive Summary

**Short answer:** Training loss says "learning"; validation loss says
"instrument is broken, so we cannot tell from loss alone." The trajectory
through ~10k looked healthy (2.068 -> 1.856, MFU ~18%), but a flat
`12.53-12.60` validator across `110700-111200` that is **~6-7x the training
loss** is not a plateau signal — it is a miscalibrated validator reporting a
different quantity. The prior dolmino CPT taught us exactly this lesson:
**loss improved 0.31 nats (2.80 -> 2.49) while HellaSwag fell -7.4pp and
ARC-Easy -10.4pp**. Loss on a new distribution is not a capability metric.

**Recommendation (no new training to decide):**

1. **Do not stop CPT on loss.** Treat the live chain as "presumed healthy
   until downstream eval says otherwise." Keep it running to 190k on the
   12h walltime cadence while we score what we already have.
2. **Score checkpoints now** to break the ambiguity. Convert a sparse ladder
   of DCPs to HF and run the same eval that caught the dolmino regression:
   `hellaswag / arc_easy / piqa / arc_challenge` + `mmlu` + `mmlu_continuation`
   (both, report `max(CF,MCF)`) + post-SFT `GSM8K-CoT` on a fixed SFT probe.
   The converter already exists; the HF roots are on Tegu to avoid `/home`
   fill. The full gate list and thresholds are in Sec. 4.
3. **Decision gate at 150k is real, 190k is correct if held.** If the 150k
   scoring shows commonsense flat-or-better vs the MDS base *and* the SFT
   probe lifts the CoT ceiling above B2's 0.205, continue to 190k. If either
   degrades (HellaSwag <-2pp, ARC-Easy <-3pp, or SFT probe <=0.10), fork an
   SFT immediately from the last-good checkpoint and stop extending CPT —
   exactly the early-detection counterfactual in Sec. 5.
4. **The 143k fork question is premature.** No CPT checkpoint has ever been
   scored on MMLU in either format; a 0/200 raw-CoT score on a pure-CPT base
   is expected (no chat template), not evidence. Forking SFT now at 143k
   without the scoring ladder would repeat B2v2's mistake: training looked
   healthy, eval halved.

Cost to answer: ~8-12 HF conversions + 3 SFT-probe trainings + 2 eval sweeps.
All reuse existing scripts and run on `workq` 1N; no new CPT compute needed
to decide.

---

## 1. Setup — What Is Actually Running

### 1.1 Config and budget

| Field | Value | Source |
|---|---|---|
| Config | `agpt_2b_mds_mix_owm_edu_6040` | `torchtitan/experiments/ezpz/agpt/config_registry.py:1046-1057` |
| Mix | 40% `open-web-math/open-web-math` (HF streaming) + 60% `fineweb_edu_local` (140 local parquets, Tegu `fineweb-edu-100BT/sample/100BT`) | `config_registry.py:970-1021`, `cpt_fineweb_edu_4n.sh:18-19` |
| Base | `agpt-2b-mds-gs138650/step-0` (MDS 2B, pre-stage-3) via `FaultTolerantTrainer` model-only fork | `cpt_fineweb_edu_4n.sh:23,94-96`, `config_registry.py:919-928` |
| Packing | `GrainDataLoader(ConcatThenSplitPacking(DatasetMix(...)))` — weights count post-packing 8192-token rows, i.e. token-weighted | `config_registry.py:973-1019` |
| GBS | `LBS=2 * PPN=8 * 4N=32r = 64`, `seq_len=8192` => `524,288 tok/step` | `cpt_fineweb_edu_4n.sh:72-77` |
| Budget | `100B tok / 524,288 = 190,734 steps` | `cpt_fineweb_edu_4n.sh:77-78`, docs |
| LR | `2e-6` constant, `warmup_steps=20`, `decay_ratio=0.0`, `min_lr_factor=1.0` | `config_registry.py:923-927` |
| CKPT | `interval=200`, `keep_latest_k=0`, `folder=outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64` (live on `/home`, symlink `outputs/checkpoints -> outputs/outputs/checkpoints`) | `cpt_fineweb_edu_4n.sh:79-89` |
| Validator | `validator.enable --validator.freq=100 --validator.steps=10` | `cpt_fineweb_edu_4n.sh:101` |
| VENV | `xpu-torch216-sync-5a7a622f16` + LD-first (`libur_*`, `libsycl.so*`, `libumf*`, `libtcm*`) | `cpt_fineweb_edu_4n.sh:53-70` |
| Launch | `qsub -l select=4 -l walltime=12:00:00 -v NHOSTS_TRAIN=4 cpt_fineweb_edu_4n.sh` | `cpt_fineweb_edu_4n.sh:12,94-96` |

`_agpt_2b_mds_mix_base()` (line 919) is the fork base: it inherits
`_agpt_2b_mds_anneal_base()` and overrides only the LR schedule to flat.
This is the anneal-proven winner — the 75/25, 50/50, 90/10 blend arms all
share it (`config_registry.py:1024-1043`, `1110-1120` commentary). The 60/40
arm is therefore *not* testing a new LR; it is testing a new **data kind**.

### 1.2 Live chain position (from docs at this branch point)

The docs history on `ezpz-cpt111k-eval-sft` records the chain advancing
through three `workq 02:00` hops then a `12:00` hop:

| Job | Steps | Tip loss | MFU | Outcome | Doc source |
|---|---|---:|---:|---:|---|
| `12483048` | `0 -> 4091` | `2.068` | `19.3%` | walltime `02:00` | `journal.md:100`, `POST-TRAINING-2B.md:57` |
| `12483075` | `4091 -> 7869` | `2.463` | `18.35%` | walltime `02:00` | same |
| `12483245` | `7869 -> 10143+` | **1.856** at `10108` (MBH) | `18.16%` | `R` live | `journal.md:96-104`, `POST-TRAINING-2B.md:57` |
| `12483388` onward | `... -> 26742 -> 32800 -> 34298` | `1.93` at `34298` | `17.1%` | chain healthy | `journal.md:84`, `POST-TRAINING-2B.md:35-36` |
| `12483799` | `-> 75000` | `~22.4G` frontier, `375` ckpt dirs | `~18%` | `R` | `POST-TRAINING-2B.md:19` |
| `12483856` | `100000 -> 111200` | `23G` frontier, `556` dirs | `15.8%` | `R` | `POST-TRAINING-2B.md:10` |
| Snapshot | `143400 / 190734 = 75.2%` | — | — | assignment prompt | prompt |

The assignment's `143.4k` is therefore ~32k steps past the last doc entry
at `111.2k`; the chain continued on the corrected `12h` walltime after
`99e918b342` (which fixed the earlier `02:00` churn eating ~4k steps per hop
vs ~24k per 12h hop — `POST-TRAINING-2B.md:12,19`).

### 1.3 What "validation loss 12.53 flat" actually means

At `111200` the doc records `Validation loss 12.53 at 111200 (flat 12.53-12.60
since 110700)` (`POST-TRAINING-2B.md:10`). Compare:

| Signal | Value | Expected |
|---|---|---:|
| Training loss (live) | `1.856` (10108) -> `1.93` (34298) | `log(vocab) ~= ln(256000) ~= 12.45` at step 1, decaying to ~2.6 for a trained 2B |
| Validation loss (reported) | `12.53` flat | Should track training loss within ~0.1-0.3 nats on the same mix |

A 6-7x gap between training and validation loss on the **same mixture**
with `validator.freq=100` and `validator.steps=10` is not a plateau. It is one
of:

1. **Validator reporting a sum instead of a mean** (missing `global_valid_tokens`
   divisor). The loss-reporting guide at
   `torchtitan/experiments/ezpz/docs/guides/loss-reporting-tp-dist-reduce.md`
   documents exactly this class: `_dist_reduce` over the wrong mesh dropped the
   `dp_world_size` divisor (fixed upstream `d64eabcce`, local workaround still
   in `trainer.py`/`validator.py` as of 2026-07-24). That bug under-reported by
   `dp_world_size`; the inverse (over-reporting by `dp_world_size = 32` here)
   would turn `~1.93 * ~6.5 ~= 12.5`.
2. **Validator reading a different mixture or un-packed loss** (e.g. scoring
   raw next-token on a non-Grain path, or scoring with `seq_len` mismatch).
3. **Validator broken by the symlink fixup** (`outputs/outputs/checkpoints`
   double-nesting fix at `cpt_fineweb_edu_4n.sh:41-44`) — validator runs on a
   held-out slice of the same `DatasetMix`, but if it re-creates the dataset
   without the `_fineweb_edu_shards()` registry it could fall back to a
   dummy or stale path and report nonsense.

**Action (does not block the decision):** open the rank-0 JSONL for
`12483856` and diff `loss` vs `val_loss` fields for steps `110700-111200`.
If `val_loss / loss ~= 32` (i.e. `GBS`), it is the divisor bug; file a
`workq 1N` repro with `--validator.steps=10` and `--training.steps=300` on
the same config and patch `validator.py`. Until then, **ignore `val_loss`
as a training signal** and use training loss + downstream eval only.

This matters because the question prompt frames `12.53` as evidence of
plateau. It is not. It is evidence the validator is misconfigured.

---

## 2. Q1 — Is This Learning or Plateauing? What the Trajectory vs the Dolmino Precedent Tells Us

### 2.1 The dolmino precedent in one table

The only completed CPT on this codebase is the 2026-07 dolmino ratio sweep
(`torchtitan/experiments/ezpz/docs/production/cpt/README.md:60-129`):

| | olmo-100 plateau (base) | dolmino-100 (5,960 steps, ~300B tok) | olmo50-dolmino50 |
|---|---|---|---|
| Training start loss | ~2.80 | **7.27** | **7.24** |
| Training end loss | — | **2.480** | **2.590** |
| Final val loss | **~2.80** | **2.492** | **2.601** |
| vs olmo-100 | — | **-0.31** | **-0.20** |
| HellaSwag `acc_norm` | **0.560** | **0.486 (-7.4pp)** | **0.491 (-6.9pp)** |
| ARC-Easy `acc` | **0.651** | **0.547 (-10.4pp)** | **0.610 (-4.1pp)** |
| PIQA `acc_norm` | **0.733** | **0.687** | **~0.69** |

Source: `cpt/README.md:61-98`, job `8647850` full 12-point screen,
loss plot `cpt/figures/cpt_loss.svg` + eval plot `cpt/figures/cpt_eval.svg`.

Three facts to carry:

1. **Loss and capability diverged.** Lower loss on the new distribution did
   **not** mean a better model — it meant a lower-entropy distribution.
   `cpt/README.md:68-88` calls this out explicitly and warns the loss numbers
   are "MISLEADING." The same warning applies verbatim to FWE: FineWeb-Edu is
   also more filtered / lower-entropy than olmo-mix, so a lower NLL is the
   *null hypothesis*, not evidence of learning.
2. **Two distinct damage modes, cleanly separated by ratio:**

   - **Ratio-independent HellaSwag drop** (~-6-7pp at step-1000, never
     recovers, both mixes) — attributed to the **LR re-warm shock** (base at
     convergence re-warmed to full `2.28e-5` with `warmup=200`,
     `decay_ratio=0.8` — `cpt/README.md:118-122,156-157`). The gentle retry
     (`LR=2e-6 constant, warmup=20`, job `8648114`/`8662867`) was launched to
     test this and stopped at step-1200 with 12 checkpoints, never evaluated —
     **hypothesis still open** (`cpt/README.md:81-88,242-248`).

   - **Ratio-dependent ARC-Easy bleed:** dolmino-100 collapses monotonically
     `0.619 -> 0.547` while olmo50-dolmino50 **holds ~0.61** (`cpt/README.md:122-125`).
     Pure dolmino specifically forgets ARC-Easy; keeping olmo in the mix
     prevents it. So **both levers matter**: gentler LR *and* keeping olmo.

   Our FWE arm uses `40% OWM / 60% FWE` — it keeps a large math/OWM slice
   (40%), unlike dolmino-100. If the ARC-Easy bleed was pure-dolmino-specific,
   this arm should be safer on ARC-Easy than dolmino-100. But we have not
   measured it — **no CPT checkpoint has ever been scored on MMLU in either
   format** (`cpt/README.md:100-116`).

3. **The dolmino damage was visible by step-1000.** HellaSwag had already
   dropped ~-6pp by 50B tokens. That is 1/6 of the 300B pilot, and 1/20 of our
   100B FWE budget. **We are far past the point where dolmino showed damage.**
   If FWE had the same shock mode, it would have shown up by ~10k steps.
   The fact that training loss kept falling to 1.856 without a reported
   benchmark drop (because none was run) is *compatible* with health, but
   **cannot confirm it** — loss alone cleared dolmino too.

### 2.2 What the FWE training-loss trajectory actually shows

Reconstruct from docs:

```
step      loss    MFU     walltime notes
0         ~12.45  —       step-1 uniform-softmax expectation (ln 256k)
4091      2.068   19.3%   first hop, still near base (base ~2.65)
7869      2.463   18.35%  second hop spike (resume jitter / data mix warmup)
10108     1.856   18.16%  MBH new low (assignment cites ~1.856 at ~10k)
34298     1.93    17.1%   +24k steps later, loss stable/slightly up
111200    —       15.8%   val 12.53 flat (instrument broken, ignore)
143400    ?       ?       assignment snapshot; no loss quoted at 143k
```

Reading:

- **Not plateauing on training loss through 34k.** 2.068 -> 1.856 in 6k steps
  is real learning; 1.856 -> 1.93 over the next 24k is flat-to-slightly-up
  but well within seed / data-order noise. Compare to the MDS 2B reference:
  the base took ~4.67T tokens to reach 2.65; FWE reaching 1.93 in 34k steps
  (17.8B tokens at `524k tok/step`) is *much* lower — but that is the entropy
  effect, not necessarily a capability gain. The dolmino pilot also beat the
  plateau on loss and still degraded benchmarks.

- **The MFU decay (19.3% -> 18.16% -> 17.1% -> 15.8%) is not a learning
  signal.** MFU includes data loading; the chain's `PPN=8` over `4N` with
  `GrainPacking` over 140 parquet shards is healthy at `tps ~4625` / `~47
  tflops` per the 111k doc. The drop from 19% to 15.8% likely reflects longer
  walltime hops (12h vs 2h) amortizing startup, plus Tegu I/O contention as
  the chain accumulates 556 ckpt dirs (23G frontier) — not model stalling.

- **The missing signal is the downstream curve.** In dolmino, the damage was
  monotonic and ratio-dependent on ARC-Easy, immediate and ratio-independent
  on HellaSwag. For FWE the equivalent would be: does HellaSwag hold through
  10k/50k/100k, and does ARC-Easy hold when the mix is 40% OWM vs 100%? We
  have zero points on either curve. **The training-loss trajectory alone
  cannot distinguish "learning" from "over-specializing away from the eval
  distribution" — which is exactly what dolmino did.**

### 2.3 The plateau test that actually answers Q1

Loss plateaus are secondary. The primary test is whether the model is *still
moving on the benchmarks that matter*:

| Benchmark | Base (MDS gs138650) | Dolmino-100 damage | Dolmino 50/50 held? | FWE decision rule |
|---|---|---|---:|---|
| HellaSwag `acc_norm` | 0.560 (2B-256 plateau) | **-7.4pp** | **-6.9pp (no)** | Hold within **-2pp** at 150k => not the shock mode |
| ARC-Easy `acc` | 0.651 | **-10.4pp** | **-4.1pp (yes, held)** | Hold within **-3pp** at 150k => OWM retention works |
| PIQA `acc_norm` | 0.733 | ~-4.6pp | ~-4pp | Hold within -2pp |
| MMLU `max(CF,MCF)` 5-shot | 0.2413-0.2655 (chance; best constant-D 0.2689) | never scored | never scored | Any move **above 0.2689** + letter-prior collapse is the FWE thesis |
| GSM8K-CoT post-SFT | 0.205 (B2) | not applicable | not applicable | SFT probe (Sec. 4) clears 0.205 => new ceiling |

Thresholds derived in Sec. 4.4; scoring ladder in Sec. 4.1.

**Tentative read (inference, not measurement):** The chain is *probably*
learning in the narrow sense (loss falling, no NaN, MFU healthy, `2e-6`
constant avoids the dolmino re-warm shock by construction), and the 12.53
validator is not evidence of plateau. But "probably learning" is not "not
over-specializing." The dolmino precedent demands we treat loss as
uncorrelated with capability until the downstream ladder says otherwise.

---

## 3. Q2 — The FineWeb-Edu Kind-Shift Rationale: Does 60/40 Have Evidence It Will Help MMLU vs Just Lowering Loss Entropically?

### 3.1 What 94.8% DCLM means

Measured over `data-lists/aurora/olmo-mix-1124.txt` (1,438 files, weights
renormalized):

| corpus | olmo-mix-1124 | dolmino-mix-1124 | delta |
|---|---:|---:|---:|
| dclm | **94.80%** | **89.18%** | -5.62pp |
| starcoder | 2.46% | — | -2.46pp |
| pes2o | 1.47% | 6.88% | +5.41pp |
| flan | — | 1.97% | +1.97pp |
| math (OWM / algebraic-stack / dolmino-math) | 0.64% | 1.36% | +0.72pp |
| arxiv | 0.53% | — | -0.53pp |
| wiki | 0.09% | 0.44% | +0.35pp |

Source: `torchtitan/experiments/ezpz/docs/evals/mmlu-letter-prior-at-chance.md:188-196`
and `torchtitan/experiments/ezpz/docs/production/sft/fineweb-edu-60-40-kind-shift.md:30-42`.

Dolmino is 89% DCLM — a **change of degree** (5.6 points of DCLM
redistributed into peS2o + 2% FLAN + 1.4% math). The
`production/cpt/README.md:46-51` motivation and the letter-prior doc
`why-dolmino-is-not-the-intervention-the-literature-credits` (line 234)
make the same point: dolmino is the same DCLM-dominated web distribution
with a larger academic slice, not a different kind of data.

Our 60/40 FWE mix is `40% OWM + 60% FineWeb-Edu`. OWM is math-web; FWE is
**classifier-filtered educational web** (HuggingFaceFW `fineweb-edu`,
ODC-By 1.0, filter trained for pedagogical content — textbook prose,
explanations, Q&A — vs DCLM's generic quality classifier). Academic content
in olmo-mix is 2.0% and is research papers (peS2o/arxiv), not pedagogy.
SmolLM2 stage 1 is 60% FineWeb-Edu / 40% DCLM — our 60% FWE is therefore a
**change of kind** (majority educational content with a math floor), not a
re-weighting of the same distribution.

### 3.2 The direct evidence: FineWeb-Edu moves MMLU, DCLM moves HellaSwag

Two papers, one ablation:

1. **FineWeb-Edu paper** (arXiv:2406.17557, cited in
   `fineweb-edu-60-40-kind-shift.md:58-64`): **33.6% MMLU at 38B tokens**
   where the next-best dataset needs **300B** (~8x token efficiency). At 350B
   on a 1.7B model: FineWeb-Edu **37.5 vs DCLM 35.5 on MMLU**, while DCLM
   **62.3 vs 60.1 on HellaSwag** — i.e. FWE trades a small HellaSwag deficit
   for a clear MMLU gain.

2. **SmolLM2** (arXiv:2502.02737 Sec 4.3 Fig 6, same source): stage 1 at
   60% FWE / 40% DCLM reaches that 33.6% MMLU; DCLM-vs-FWE split predicts
   exactly our observed profile: MMLU at chance, commonsense intact.

That trade — **MMLU up, HellaSwag slightly down** — is our exact symptom:

| Signal | Our measurement | FWE prediction |
|---|---|---|
| MMLU `mmlu` 5-shot `MCF` (letters) | 0.2413-0.2655, best constant-D **0.2689** beats all 12 configs | FWE improves MMLU over DCLM |
| HellaSwag `acc_norm` | 0.538-0.61 healthy | FWE slightly *below* DCLM |
| ARC-Easy `acc` | 0.61-0.66 healthy | mixed; math slice retention matters |

Sources: `mmlu-letter-prior-at-chance.md:48-57` (MMLU table),
`mmlu-letter-prior-gating.md:40-50`, `evals/README.md:23-35` (commonsense
overlay), `fineweb-edu-60-40-kind-shift.md:66-71`.

The **strongest** external precedent is OLMo-2 1B: same `olmo-mix-1124` base,
Dolmino mid-training **MMLU 26.9 -> 44.3 (+17.4)** (arXiv:2501.00656 Table 9,
cited in `cpt/README.md:104-106` and `fineweb-edu-60-40-kind-shift.md:73-78`).
At 7B/13B the same intervention is only +3.9/+4.1, so at small scale
mid-training is nearly the **entire** MMLU signal. Our dolmino CPT did not
reproduce that (+17.4) — consistent with dolmino being the wrong *kind* of
mid-training at 89% DCLM, not with mid-training being ineffective.

### 3.3 Does 60/40 have evidence it will help *vs just lowering loss entropically*?

This is the right skepticism, and the answer is bifurcated:

**For training loss: no, lowering NLL is expected and is not evidence.**
FWE is more filtered than olmo-mix (pedagogical subset of web), so its
per-token entropy is lower. A model trained on it *should* report lower NLL
even if it has learned nothing generally useful — exactly as dolmino did
(2.49 vs 2.80) while degrading benchmarks. The `cpt/README.md:131-147`
"Loss-only findings (do NOT imply capability gains)" section exists for this
reason. Our 1.856 vs 2.65 base gap is directionally the same phenomenon;
magnitude alone does not distinguish "knows more" from "distribution is more
predictable."

**For downstream MMLU: yes, but the evidence is from a different scale and
setup, and must be tested via the CF/MCF split.**

The FWE literature credits MMLU gains to **educational-content density**
academically, not to lower entropy. Three reasons to take that seriously:

1. **The ablation isolates distribution, not entropy.** The 37.5 vs 35.5
   (FWE vs DCLM on MMLU) at matched token count and model size controls for
   entropy — both models lower their own training loss; only FWE's *eval*
   MMLU moves. The HellaSwag reversal (62.3 vs 60.1 DCLM win) further shows
   this is not a uniform entropy effect.

2. **SmolLM2's 60/40 is not arbitrary.** It is the published stage-1 recipe
   and the FWE paper's recommended operating point where MMLU gains are large
   and HellaSwag regression is small (`fineweb-edu-60-40-kind-shift.md:182-184`).
   Our 60% FWE respects that ratio; the 75/25 and 90/10 blend arms in
   `config_registry.py:1024-1043` exist to bracket it once 60/40 is measured.

3. **OLMo-2's +17.4 at 1B is the closest in-family proof** that a
   distribution shift on `olmo-mix-1124` can unlock MMLU at small scale. The
   fact that our dolmino-based shift did not replicate it strengthens the
   "kind, not degree" diagnosis.

**But** — and this is where the MMLU gating docs matter — **no FWE number
above is MCF at 2B.** SmolLM2's own MCF is 29.62 across 0-6T tokens
(`mmlu-letter-prior-at-chance.md:166-168`); OLMES on OLMo-7B (arXiv:2406.08446)
says MCF only clears chance past ~400B and recommends `max(CF, MCF)`. Every
AuroraGPT MMLU we have is MCF-on-letters and sits at chance *by
construction* at our token scale. So the right test is:

- Score each CPT checkpoint on **both** `mmlu` (MCF, letters `A-D`) and
  `mmlu_continuation` (CF, answer text) at `num_fewshot=5`, per the OLMES
  recommendation. Job `12474382` ran both on 30B step-2000 (`limit 10`)
  and got `MCF 0.2281 / CF 0.2386, delta +0.0105 SE 0.0256` — not significant
  at 570 questions, and at 3.93B tokens arguably too early
  (`mmlu-letter-prior-at-chance.md:268-294`). The flagship test is the CPT
  checkpoints at 50k+.

- Re-fit the letter prior `acc(subject) = sum_L p_L * keyfreq_L(subject)`
  on each new MMLU. If the model is still `p(A,B,C,D) ~= (0.37,0.32,0.20,0.11)`
  with residual variance at the binomial floor, it has not moved off the
  prior — even if the headline number ticks up to 0.27. A real move is:
  **headline >0.2689 *and* prior collapse** (`R2 -> 0`, residual var not
  explained by letter frequency). See `mmlu-letter-prior-at-chance.md:142-158`
  and `mmlu-letter-prior-gating.md:82-95`.

If FWE only lowers entropy, the CF and MCF curves will stay together at
chance. If it adds knowledge, **CF should lift first** (answer-text scoring,
same as HellaSwag/ARC-Easy where we have signal) while MCF stays at chance
until letter binding emerges — exactly the OLMES/SmolLM2 pattern.

**Bottom line for Q2:** The 60/40 kind-shift has the *right* external
evidence for the *right* symptom (MMLU at chance + commonsense intact), and
it is a distribution-kind change where dolmino was a degree tweak. But
"lower training loss" is not that evidence — the MMLU `max(CF,MCF)` ladder
plus the letter-prior fit is. Run that ladder; do not infer from loss.

---

## 4. Q3 — Should We Continue to 190k, Stop Early at 150k, or Fork SFT Now at 143k?

### 4.1 The three options, priced

| Option | Compute | Walltime | What it buys | Risk |
|---|---:|---|---|---|
| **A: Continue to 190k** | +47.3k steps = +24.8B tok | ~2.5 hops of 12h at 1.9k steps/hr (111k record: `+11.2k in 06:04`) | Full 100B FWE thesis; clean budget match to SmolLM2 schedule | If degrading, burns ~2.5 hops on a regressing model |
| **B: Gate at 150k** (6.7k more from 143.4k) | +6.7k steps = +3.5B tok | <0.5 hop | Decision point with scoring in hand; option to stop or continue | If held, pays a resume hop anyway |
| **C: Fork SFT now at 143k** | 1N SFT: `~1k steps` (tulu_math_uc_mix 729-step regime) or `~100 steps` narrow | 1-2h on `workq 1N` | Immediate SFT probe; tests "does this base lift the 0.205 ceiling?" | Forks before knowing if CPT is helping or hurting |

**Costs are asymmetric.** CPT is 4N/12h; scoring is 1N/1-2h per checkpoint.
The cheapest way to de-risk A is to **pay the scoring cost while CPT
continues** — they are independent (CPT writes DCPs; scoring reads them).
Do not block CPT to score; do not fork SFT blind to save a 1N eval.

### 4.2 What the assignment's 143k snapshot actually says

- At 143.4k/190.7k (75.2% budget) we are **past the dolmino damage horizon**
  (damage visible by step-1000 / 6% of budget) and past the prior doc's
  "eval gate ~15k steps" (`journal.md:110`, production `README.md` CPT row).
  If the FWE shock mode existed, we are deep enough to have seen it — *if*
  we were looking. We are not; no checkpoint past 11k has been scored.

- Training loss 1.856 at 10k is **not a stop signal**. It is a healthy
  new-low on the new distribution. The flat 12.53 validator is not a plateau
  signal — it is a broken instrument (Sec. 1.3). Neither argues for stopping.

- The SFT baseline is **B2 = 0.205 GSM8K-CoT** (two-stage:
  `global_step138650 -> tulu_math_uc_mix_full checkpoint-900 (IFEval 0.253) ->
  gsm8k-r1cot 93 steps 3ep -> checkpoint-93`, `POST-TRAINING-2B.md:107-115`).
  Every alternative on that base without CPT (B3 0.05, B4a 0.02, B4b 0.065,
  B2v2 0.09/0.105, narrow SFT-on-CPT-17k 0.015) sits 0.02-0.10 — a 2-10x gap
  that defines the "ceiling without new data" (`POST-TRAINING-2B.md:93-104`,
  `fineweb-edu-60-40-kind-shift.md:97-155`).

**Decision logic:** CPT's value is not its base-model MMLU (which will
likely stay at chance on MCF regardless — Sec. 3). Its value is whether it
**lifts the SFT ceiling** — i.e. whether a new base plus the proven two-stage
SFT recipe clears 0.205. That is measured by the **SFT probe**, not by
CPT loss or CPT-direct MMLU.

### 4.3 Recommendation — Gate at 150k, default to 190k if held

**Continue CPT uninterrupted to 190k** on the 12h cadence:

```bash
qsub -l select=4 -l walltime=12:00:00 \
  -v NHOSTS_TRAIN=4 \
  torchtitan/experiments/ezpz/scripts/cpt_fineweb_edu_4n.sh
# auto-resumes from CKPT_DIR via FaultTolerantTrainer;
# CKPT_INTERVAL=200, walltime 12:00:00 proven at 12483856 (step-111200)
```

**In parallel, score the ladder through 150k.** When 150k finishes (or now
from 143.4k if the chain is live), apply the gate in Sec. 4.4. If held,
continue to 190k. If degraded, fork SFT from the last-good checkpoint per
Sec. 4.4's fallback and **stop extending CPT** — the remaining 40k steps
would be burning compute on a known-regressing trajectory (the dolmino
lesson: loss kept falling while benchmarks bled monotonically).

**Do not fork SFT now at 143k as the primary action.** Forking before
scoring repeats the B2v2 error: training looked healthy (loss 0.8069, acc
0.8198, descending cleanly — `POST-TRAINING-2B.md:82`) while eval halved.
The cost of waiting 6.7k steps (~3.5B tokens, <0.5 hop) to score 150k is
negligible vs the cost of SFT-training on a potentially-regressing base and
having to redo it from an earlier checkpoint.

**Exception:** if the scoring ladder through 110k (already on disk) shows
degradation *now*, fork SFT from the last-good checkpoint immediately and
pause CPT — no need to wait for 150k to act on a known regression.

### 4.4 The 150k gate — thresholds, tests, and fallback

Run the full scoring ladder in Sec. 4.1 *before* applying thresholds. The
gate is a conjunction — all must hold to continue:

| Gate | Metric | Base | Threshold to CONTINUE to 190k | Source / SE |
|---|---|---|---:|---|
| **G1 — No commonsense regression** | HellaSwag `acc_norm` | 0.560 (2B-256 plateau) | `>= 0.540` (within -2pp) | SE ~0.004; -2pp = 5 sigma |
| **G1b** | ARC-Easy `acc` | 0.651 | `>= 0.621` (within -3pp) | SE ~0.006; -3pp = 5 sigma; dolmino 50/50 held -4.1pp, so -3pp is strict |
| **G1c** | PIQA `acc_norm` | 0.733 | `>= 0.713` (within -2pp) | SE ~0.005 |
| **G2 — No catastrophic forgetting** | ARC-Challenge `acc_norm` 25-shot | ~0.31 (from `evals/`) | `>= 0.290` (within -2pp) | eval-landscape peer table |
| **G3 — MMLU not required to pass, but CF must not regress** | MMLU `max(CF,MCF)` 5-shot | 0.2413-0.2655 (chance band) | `>= 0.240` (no regression) | `mcf` band 0.2295-0.2689, SE 0.0036 |
| **G4 — SFT probe lifts the ceiling** | GSM8K-CoT `acc` post-SFT (see Sec. 4.2 probe) | B2 = 0.205 | `>= 0.205` (match or beat) | SE ~0.03 at N=200; this is the primary value test |
| **G4b — SFT probe format held** | GSM8K-CoT `format` (closed `\boxed{}`) | B2 = 0.985 | `>= 0.90` | B2v2 fell to 0.86/0.91 while degrading |

**Interpreting near-misses:**

- `G1-G2` fail by >2pp but `G4` passes (commonsense dips, CoT ceiling
  lifts): **continue to 190k** — the commonsense dip is the expected FWE
  HellaSwag trade (62.3 vs 60.1 DCLM win in the ablation) and is acceptable
  if the SFT ceiling moves. Re-check at 190k; if commonsense keeps bleeding
  monotonically (dolmino pattern), the mix ratio is wrong — revisit 75/25.

- `G1-G2` pass but `G4` fails (no CoT lift, e.g. 0.08-0.12 like B2v2/B3/B4):
  **stop CPT**, fork SFT from the **best intermediate checkpoint** (not
  necessarily 150k — whichever scored highest on `G1-G4`), and pivot per
  `POST-TRAINING-3` broad-distill mix (OpenThoughts-114k-decontam x
  gsm8k-r1cot, pretokenized to `/tegu/datasets/agpt2b-distill-cot-mix-len8192`,
  `POST-TRAINING-2B.md:164-178`). CPT did not move the ceiling; do not pay
  40k more steps hoping it will.

- Everything fails (commonsense tanks + CoT collapses): **stop CPT, revert**
  to the MDS base for SFT. The FWE kind-shift hypothesis is wrong for this
  scale/mix, or the constant-LR regime needs re-tuning. Do not burn 40k steps
  on a known-bad distribution.

**Eval to decide (the single question that matters):**

> Does the CPT base, when run through the *proven* two-stage SFT recipe
> (`checkpoint-900` equivalent -> `gsm8k-r1cot` 93 steps), produce a higher
> GSM8K-CoT accuracy than B2's 0.205, without regressing commonsense?

That is the SFT probe. Everything else is thermometer.

---

## 5. Q4 — What Checkpoints to HF-Convert and Score (Exact Gates, Commands, Budgets)

### 5.1 Which checkpoints exist and where

Live DCP chain (from docs + assignment):

- `DCP_ROOT = /home/foremans/torchtitan/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64`
  (double-nested under `outputs/outputs/checkpoints` with symlink
  `outputs/checkpoints -> outputs/outputs/checkpoints` per
  `cpt_fineweb_edu_4n.sh:41-44`).
- At 111k: `556` ckpt dirs, frontier `step-111200`, `23G`.
- At 143.4k: `~717` ckpt dirs (143400/200), frontier `step-143400`.
- `TEGU_ROOT = /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan`
  holds new HF outputs per user request (avoid `/home` fill):
  `HF_ROOT = $TEGU_ROOT/outputs/evals/cpt-fwe/step-$s/hf`.

Existing HF conversions (from `POST-TRAINING-2B.md:19-22`):

| Batch | Steps | Size | Status |
|---|---|---:|---|
| `hf_batch_convert_cpt_fwe.sh` early | `74000 / 74200` | `3.7G` | `9/9` files verified (patched `config/tokenizer` omission) |
| Next gate (doc: "next convert at 112k gate") | `~111200` | — | pending at doc time |

Converter uses `flavor=2b-mds` (critical — wrong flavor = silent gibberish;
`torchtitan/experiments/ezpz/eval/convert_to_hf.py` + `hf_batch_convert_cpt_fwe.sh:58`).

### 5.2 The scoring ladder — sparse then dense around the decision

Do not convert every 200-step checkpoint. Convert a **log-spaced early +
linear late** ladder that captures both the shock horizon (0-10k) and the
current frontier (110k-150k):

| Phase | Steps to convert | Count | Why |
|---|---:|---:|---|
| **P0 — Shock horizon** | `1000, 5000, 10000` | 3 | Dolmino shock hit by 1000; our flat-LR should not — verify |
| **P1 — Early plateau** | `25000, 50000, 75000` | 3 | 75k already HF'd (74000/74200); add 25k/50k to fill the gap |
| **P2 — Current frontier** | `100000, 111200, 125000, 143400, 150000` | 5 | 111.2k documented; 143.4k assignment snapshot; 150k gate |
| **P3 — If continuing** | `160000, 175000, 190734` | 3 | Full 100B budget; 190,734 is the exact terminal step |

**Total: 14 HF conversions** (8 immediate: P0+P1+P2 through 143.4k; 6
deferred: 150k + P3). Each is `~3.7G` (`9/9` safetensors + `config.json` +
`tokenizer.*`); total HF on Tegu = `~52G` for the full ladder — trivial
against the `23G` DCP frontier.

For the SFT probe, only **3 checkpoints need SFT-training**: the best P2
candidate, the 150k gate, and (if degrading) the last-good earlier step
(Sec. 4.4). The base-model evals (HellaSwag etc. + MMLU) run on all 14
without SFT.

### 5.3 Commands — exact, runnable on Sunspot/Aurora

#### 5.3.1 HF batch convert (Sunspot `workq 1N 02:00`)

The script already handles fixups and staging locks:

```bash
# Sunspot — lives in the repo; PBS filesystems=tegu:home covers both stores
qsub torchtitan/experiments/ezpz/scripts/eval/hf_batch_convert_cpt_fwe.sh
```

Current `hf_batch_convert_cpt_fwe.sh` loops `13000 $(seq 74400 200 136800)`.
For this ladder, override with a single-hop 1N run that converts the P0-P2
set:

```bash
# One-shot ladder convert (Sunspot, workq, 1N 02:00)
# Run on a login node to stage the env, then qsub:
cat > /tmp/hf_ladder_cpt_fwe.sh <<'PBS'
#!/bin/bash --login
#PBS -N hf-ladder-cpt-fwe
#PBS -q workq
#PBS -l walltime=02:00:00
#PBS -l select=1
#PBS -l filesystems=tegu:home
#PBS -A datascience
#PBS -j oe
set -euo pipefail
SUBMIT_DIR=${PBS_O_WORKDIR:-$(pwd)}; cd $SUBMIT_DIR
REPO=/home/foremans/torchtitan
DCP_ROOT=$REPO/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64
TEGU_ROOT=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan
HF_ROOT=$TEGU_ROOT/outputs/evals/cpt-fwe
HF_CONFIG=$REPO/torchtitan/experiments/ezpz/eval/configs/agpt_2b_mds_config.json
TOK_SRC=$REPO/assets/hf/gemma-7b
echo "HF-LADDER $(date -u +%FT%TZ) DCP=$DCP_ROOT HF=$HF_ROOT"
for s in 1000 5000 10000 25000 50000 75000 100000 111200 125000 143400 150000; do
  DCP=$DCP_ROOT/step-$s; HF=$HF_ROOT/step-$s/hf
  [[ -d "$DCP" ]] || { echo "SKIP $s: no DCP"; continue; }
  [[ -f "$HF/ezpz_export.json" ]] && { echo "SKIP $s: HF exists"; continue; }
  echo "CONVERT $s: $DCP -> $HF"
  $REPO/torchtitan/.venv/bin/python -m torchtitan.experiments.ezpz.eval.convert_to_hf \
    "$DCP" "$HF" --model_name experiments.ezpz.agpt --model_flavor 2b-mds \
    --export_dtype bfloat16 || exit 1
  cp "$HF_CONFIG" "$HF/config.json"
  cp "$TOK_SRC"/tokenizer.* "$HF/"
  ls -lh "$HF"/model-*.safetensors
done
echo CONVERTED $(ls $HF_ROOT/step-*/hf/ezpz_export.json 2>/dev/null | wc -l) HF
PBS
qsub /tmp/hf_ladder_cpt_fwe.sh
```

`flavor=2b-mds` is load-bearing — `torchtitan/experiments/ezpz/eval/convert_to_hf.py`
dispatches on it; wrong flavor silently produces gibberish (doc warning at
`production/cpt/README.md` eval section). Verify each HF with
`ls -lh $HF/model-*.safetensors` (expect `9/9` files, `~3.7G`).

#### 5.3.2 Base-model eval — commonsense + MMLU both formats (Aurora `workq`)

Base-model eval uses `lm-eval` on the HF exports. The per-model plotter
`torchtitan/experiments/ezpz/docs/evals/agpt/2b/plot_eval_overview.py` and
`scripts/eval/convert_and_eval.sh` are the canonical pipeline
(`evals/README.md:77-89`). For this ladder, run lm-eval directly on each HF:

```bash
# Aurora — lm-eval on each HF (frameworks/2025.3.1 or 2026.1.0, see note)
# Run once per step-$s HF; aggregate via eval/aggregate_evals.py
for s in 1000 50000 100000 111200 143400 150000; do
  HF=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-$s/hf
  OUT=$HF/../results.json
  # SHOTS_SPEC: hellaswag 0, arc_easy 0, piqa 0, mmlu 5, mmlu_continuation 5, arc_challenge 25
  # Use frameworks/2026.1.0 for vLLM 0.26 XPU fp32 path if available; else 2025.3.1 HF backend
  python -m lm_eval --model hf --model_args pretrained=$HF,dtype=bfloat16 \
    --tasks hellaswag,arc_easy,piqa,arc_challenge,mmlu,mmlu_continuation \
    --num_fewshot 5 --device xpu --batch_size auto \
    --output_path $OUT 2>&1 | tee $HF/../eval.log
done
```

**Framework note:** `POST-TRAINING-2B.md:25` documents `MMLU vLLM level_zero
40 63G OOM (compute_topk_scores) unfixable on Sunspot XPU — Aurora-only`,
and `evals/mmlu-letter-prior-at-chance.md:119-131` documents the tokenizer
check (GemmaTokenizer vocab 256000, `tokenizer.json` md5 `b60c6af2...`,
continuations `_A=586 _B=599 _C=585 _D=608` single tokens). Run MMLU on
**Aurora** (XPU with more HBM or CUDA) to avoid the Sunspot OOM. CoT eval
with vLLM fp32 works on Sunspot (`frameworks/2026.1.0 vLLM 0.26 XPU fp32`,
`submit_cot_eval.sh`).

**What to record per checkpoint:** the `results.json` `mmlu` group key (not
hand-rolled macro — macro double-counts category rollups and reads ~0.005
high, verified 0.2631 vs 0.2579 on step-46429;
`mmlu-letter-prior-at-chance.md:65-70,94-102` and
`mmlu-letter-prior-gating.md:136-139`), `n-shot` field (since `5ab9e50f6`),
and the letter-prior fit `R2` / residual var.

#### 5.3.3 SFT probe — the CoT-after-SFT test (Sunspot `workq 1N` or Aurora)

The probe is a **fixed SFT recipe on each candidate base** to isolate base
quality from recipe tuning. Two variants; run both on the 150k candidate and
one on each intermediate:

**Probe A — B2-faithful (primary):** `checkpoint-900 -> gsm8k-r1cot 93 steps`
replica, but seeded from the CPT HF instead of `global_step138650`.

```bash
# Sunspot workq 1N — narrow SFT probe on CPT-150k HF
# Mirrors torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh
# but base = CPT HF, 3 epochs on 7.4k gsm8k-r1cot examples
# TRL SFTTrainer, FSDP full_shard, LR 2e-5 cosine, batch 2 / GAS 8 / seq 2048 (GBS 384)
# save_strategy=epoch, checkpoints at 2/4/6+final
# VENV: rl-monarch-torch214-locality-20261001 (verified import trl; see POST-TRAINING-2B.md:86)
# Compare: B2 = 0.205 @ format 0.985, gen_len 282; B2v2 0.09/0.105 degraded
```

**Probe B — Broad-distill (if Probe A passes and pretokenized mix is ready):**
`OpenThoughts-114k-decontam x gsm8k-r1cot` via
`/tegu/datasets/agpt2b-distill-cot-mix-len8192` (91,481 rows -> 36,934 packed
@8192, `dataset_info.json` + 8x `data-*.arrow`, `[decontam]` counts verified
— `POST-TRAINING-2B.md:35-37`, `journal.md:82-86`). This is the only open
path that adds new CoT data under the proven two-stage structure.

**Probe budget:** Probe A is `6-12 steps` (3-6 epochs on 7.4k examples at
GBS 384) — minutes on 2N Sunspot (jobs `12482572/73` ran `00:01:38` eval on
the result). Probe B is `~37k` packed seqs — similarly short. The cost is
eval, not training.

**CoT eval on probe output:** `frameworks/2026.1.0 vLLM 0.26 XPU fp32`,
`temperature 0.0 max_tokens 768`, 200-problem GSM8K-CoT split
(`submit_cot_eval.sh`, `POST-TRAINING-2B.md:78-83`):

```
GSM8K-CoT acc (exact \boxed{}), format (closed box rate), gen_len, n_unclosed
# B2 reference: 0.205 / 0.985 / 282 / ~0
# Degraded signatures: B3 0.05, B4a 0.02 (format 0.26, 133/200 unclosed),
#   B4b 0.065 (format restored 0.86 but accuracy stuck), B2v2 0.09/0.105
```

#### 5.3.4 Budget for the full gate

| Step | Action | Nodes | Walltime | Count | Total |
|---|---|---:|---:|---:|---:|
| HF convert P0-P2 | DCP -> HF `2b-mds bf16` | 1N | 00:20 each | 8 | ~2.5h 1N |
| Base eval (commonsense + MMLU) | lm-eval `hellaswag/arc_easy/piqa/arc_challenge/mmlu/mmlu_continuation` | 1N Aurora | 01:00 each (MMLU ~56k req/step) | 8 | ~8h 1N |
| SFT probe A (3 checkpoints) | TRL SFT `gsm8k-r1cot` 6-12 steps | 2N Sunspot | 00:10 each | 3 | ~0.5h 2N |
| CoT eval on probes | vLLM `GSM8K-CoT` 200 probs | 1N Sunspot | 00:02 each | 3 | ~0.1h 1N |
| **Total to 150k gate** | | | | | **~11h single-node-equivalent** |

No CPT allocation needed to decide. The live chain continues in parallel.

### 5.4 Decision thresholds (numeric, with SE)

These are the same as Sec. 4.4, restated as pass/fail for the scoring
pipeline:

```
PASS if all:
  hellaswag acc_norm  >= 0.540   # base 0.560, SE ~0.004, -2pp = 5σ
  arc_easy acc        >= 0.621   # base 0.651, SE ~0.006, -3pp = 5σ
  piqa acc_norm       >= 0.713   # base 0.733, SE ~0.005
  arc_challenge >= 0.290         # base ~0.31, 25-shot, SE ~0.008
  mmlu max(CF,MCF) >= 0.240     # chance band 0.2295-0.2689, SE 0.0036 — no regression
  sft_probe gsm8k_cot acc >= 0.205  # B2, SE ~0.03 at N=200 — must match or beat
  sft_probe format    >= 0.90    # B2 0.985; below 0.90 = format collapse (B2v2/B4a sig)

WARN (continue with caution):
  hellaswag 0.52-0.54 or arc_easy 0.60-0.621 → FWE HellaSwag trade; acceptable if sft_probe passes
  mmlu MCF still at chance but CF lifts >0.30 → knowledge present, format barrier — pass (OLMES pattern)

FAIL (stop CPT, fork last-good):
  any commonsense metric -5pp vs base
  sft_probe acc <= 0.10 (B2v2/B3/B4 regime)
  sft_probe format <= 0.50 (B4a collapse)
```

**Why these thresholds:** Dolmino 50/50 held ARC-Easy at -4.1pp and was
considered "held" (`cpt/README.md:122-125`); -3pp is stricter. HellaSwag
-2pp is tight because dolmino failed it at -6-7pp regardless of ratio —
that was the shock signal. MMLU at 0.24 is intentionally below the
constant-D ceiling 0.2689 because we do not gate on MCF MMLU
(`mmlu-letter-prior-gating.md:156-178`); CF lifting is the thesis, not MCF.

---

## 6. Q5 — The CPT-then-SFT vs SFT-on-MDS-Base Counterfactual: If CPT Hurts ARC-Easy, How Do We Detect Early?

### 6.1 The counterfactual precisely

We have two paths to a post-trained model:

| Path | Base | SFT | Measured | Status |
|---|---|---|---|---|
| **SFT-on-MDS-base (control)** | `global_step138650` (no CPT) | `checkpoint-900` (tulu_math_uc_mix_full 900 steps, IFEval 0.253) -> `checkpoint-93` (gsm8k-r1cot 93 steps, CoT 0.205) | CoT 0.205, format 0.985 | **Deliverable B2** (`POST-TRAINING-2B.md:43-46,107-115`) |
| **CPT-then-SFT (treatment)** | `global_step138650` -> 100B FWE CPT (60/40) -> HF | same SFT recipe on new base | unknown | **Unmeasured** |
| **SFT-on-CPT (confounded)** | CPT-17000 HF | `gsm8k-r1cot` 3ep GBS 384 (same as B2v2) = 0.015 | CoT 0.015 vs B2 0.165 on preserved split | `POST-TRAINING-PLAN.md:38-39` — format collapse, not CPT evidence |

The third row is the trap: `agpt2b_b2v2_distill_cot_2n.sh` on CPT-17000 HF
scored 0.015 vs B2's 0.165 (same harness, same split) — but the doc notes
this is "format collapse on FW-Edu-shifted manifold, hparams not CPT
evidence." Early-CPT + narrow-SFT with a fixed LR schedule is not a clean
test of the base; the base at 17k is still in the shock/warmup transient and
the SFT hparams were tuned for the MDS manifold, not the FWE-shifted one.

The clean counterfactual is: **same SFT recipe, different base, scored on the
same harness.** That requires the probe design in Sec. 5.3.3, not a one-off
narrow SFT.

### 6.2 How the dolmino damage actually presented — and what "early" means for FWE

Dolmino's two damage modes had opposite early-detection signatures:

| Mode | Signal | Onset | Ratio-dependent? | CPT 60/40 analogue |
|---|---|---|---|---|
| **HellaSwag shock** | -6-7pp at step-1000, never recovers, *both* mixes | Immediate (6% of 300B pilot) | No | Should be absent (flat LR 2e-6, warmup 20) — if present, this is the canary |
| **ARC-Easy bleed** | dolmino-100 `0.619 -> 0.547` monotonic; 50/50 holds `~0.61` | Slow, over full 300B | Yes | 40% OWM retention should protect ARC-Easy; bleed = OWM too low or math forgetting |

Translated to FWE at 143.4k (75% of 100B, ~75B tokens):

- **"Early" was 1k-10k steps** (0.5-5B tokens, 1-5% of budget). We are at 143k.
  We are not early — we are late. The shock horizon is 14x behind us.

- **The slow bleed would be visible as a monotonic decline across the P0-P2
  ladder** (1k -> 50k -> 100k -> 143k). If ARC-Easy holds at 100k and dips at
  143k, that is not shock — it is cumulative distribution drift.

- **Math forgetting is the specific risk for a 60% FWE mix.** The blend arms
  commentary in `config_registry.py:960-964` notes the Wave-1 result:
  `owm-100 holds math (FineMath 1.804); edu-100 CATASTROPHICALLY forgets it
  (+0.308 NLL) for a tiny general gain (-0.024 on wikitext)`. The math-loss
  curve is steep, general-gain curve flat -> optimal mix is math-heavy.
  Their 75/25 predicted winner captures ~all edu gain at zero math cost.
  Our 60% FWE is *more aggressive* than that 75/25 winner — it trades more
  math for more general. If ARC-Easy/ARC-Challenge bleed, the mechanism is
  math forgetting, not LR shock.

### 6.3 The four early detectors, in priority order

#### Detector 1: Commonsense ladder (no SFT) — cheapest, most direct

**What:** HellaSwag `acc_norm` + ARC-Easy `acc` + PIQA + ARC-Challenge on the
base-model ladder (P0-P2, no SFT). This is what caught dolmino.

**Why first:** It caught dolmino by step-1000 without any SFT, and it runs on
already-converted HFs. No training. The `evals/README.md:23-35` overlay
(`figures/all_production_evals.svg`) is the canonical thermometer: HellaSwag
`acc_norm`, ARC-Easy `acc`, ARC-C `acc_norm`, Winogrande `acc` vs tokens.

**Thresholds:** Sec. 4.4 / 5.4. HellaSwag -2pp or ARC-Easy -3pp vs the
0.560/0.651 plateau triggers the fallback.

**How early:** By step-1000. At 143.4k we can retroactively fill the ladder
from DCPs already on disk — the detection latency is zero once we run eval.

#### Detector 2: SFT probe CoT (the ceiling test) — definitive for CPT value

**What:** Fixed SFT (`gsm8k-r1cot` 93 steps, LR 2e-5 cosine, GBS 384) on each
ladder checkpoint vs B2's 0.205 (Sec. 5.3.3 Probe A).

**Why second:** Base-model commonsense can hold while the SFT ceiling still
falls (B4b: format restored 0.86, accuracy stuck 0.065 — "style was a
symptom, not the cause", `POST-TRAINING-2B.md:99-104`). Conversely, base
commonsense can dip -2pp (FWE HellaSwag trade) while the SFT ceiling lifts.
The probe separates "CPT changed style" from "CPT moved capability."

**How early:** Probe on 10k and 50k checkpoints — if 10k already underperforms
B2 at 0.205 on the same narrow SFT, the FWE manifold shift broke SFT tuning
and needs hparam re-tuning before longer CPT can be judged.

#### Detector 3: Held-out NLL on frozen corpora — no SFT, no lm-eval

**What:** Perplexity / NLL on **frozen holdouts that no arm trains on**:
`FineMath-4+` (math) + `wikitext` (general). This is the Wave-1 design in
`config_registry.py:908-916`: "Eval reserves FineMath-4+ + wikitext as
FROZEN holdouts that NO arm trains on (disjoint by construction)." The 75/25
vs 50/50 vs 90/10 blend commentary (lines 960-964) reports Wave-1 results
in these terms.

**Why third:** It separates "does FWE forget math?" from "does it help
general?" without any SFT:

- `FineMath NLL rises >0.1` with FWE mix => math forgetting (steep curve).
- `wikitext NLL falls` => general gain (flat curve; small even at 100% edu).

If FineMath NLL spikes at 50k while wikitext barely moves, the 60% FWE is
too aggressive — the 75/25 OWM-heavy mix is the correction (as the commentary
predicts). This detector is cheap (one forward pass per holdout, no generation)
and runs on the DCP directly via the validator path — fix the validator
divisor bug first (Sec. 1.3).

**How early:** By 25k-50k steps. The Wave-1 10B-token anneal arms showed the
split at 10B tokens (~19k steps at GBS 64).

#### Detector 4: MMLU `max(CF,MCF)` + letter-prior fit — confirms the FWE thesis

**What:** MMLU 5-shot both formats + letter-prior `R2` per Sec. 3.3.

**Why last:** MMLU on a 2B base at <100B CPT tokens is *expected* to sit at
chance on MCF (SmolLM2 MCF 29.62 across 0-6T, OLMES MCF only clears ~400B).
A null on MMLU is weak evidence (30B step-2000 CF-MCF gap +0.0105 SE 0.0256
excluded only a large gap, `mmlu-letter-prior-at-chance.md:278-294`). It is
the *last* to move, not the first. Use it to confirm the thesis, not to
gate early.

**How early:** 100k-150k. Below that, MMLU at chance is the null.

### 6.4 Operationalizing early detection — what to watch and when to act

```
Steps  Action                          Detector              Stop rule
─────  ──────                          ────────              ─────────
1k     HF convert + base eval          D1 HellaSwag         if -6pp → stop CPT, LR shock (unexpected at 2e-6 const)
10k    + SFT probe (1k,10k)            D2 CoT probe         if probe <=0.10 → SFT hparams wrong for FWE manifold; retune before judging CPT
50k    + held-out NLL                  D3 FineMath/wikitext if FineMath +0.15 → 60% FWE too aggressive, pivot to 75/25
100k   + MMLU both formats             D1+D4                if commonsense holds but MMLU still chance → normal per OLMES, continue
143k   assignment snapshot             D1+D2 (retro)        if commonsense -3pp or probe fails → fork last-good, stop CPT
150k   gate decision                   D1+D2+D3+D4          Sec. 4.4 thresholds (all must hold to continue to 190k)
190k   terminal                        all                  final SFT: tulu_math_uc_mix on new base vs B2
```

**Alerting (cheap, continuous):** The validator is broken for absolute
values but may still be useful for *trend* if the bug is a constant factor
(Sec. 1.3). Plot `val_loss / train_loss` per 100 steps; if the ratio is a
constant ~32 (or ~6.5), it is the divisor bug — track the de-biased trend.
If `val_loss` diverges from `train_loss` (ratio grows), that is a real
overfitting or distribution-shift signal. Fix the validator to confirm.

---

## 7. Concrete Next Actions — Who Runs What, In What Order

### 7.1 Immediate (no CPT chain change — scoring only)

| # | Action | Command / file | Owner | Blocks |
|---|---|---|---|---|
| 1 | **Verify DCP frontier** — `ls -1 $DCP_ROOT/step-* \| tail -20`, confirm `step-143400` exists and `.metadata` present | `DCP_ROOT=/home/foremans/torchtitan/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64` | Sunspot login | HF convert |
| 2 | **HF batch convert P0-P2 (8 HFs)** | `qsub torchtitan/experiments/ezpz/scripts/eval/hf_batch_convert_cpt_fwe.sh` (or `/tmp/hf_ladder_cpt_fwe.sh` in Sec. 5.3.1) — `1N 02:00 workq`, `flavor=2b-mds bf16`, Tegu `HF_ROOT` | Sunspot `workq` | Base eval |
| 3 | **Base-model eval sweep (commonsense + MMLU both formats)** | Aurora `workq` 1N, `lm-eval --tasks hellaswag,arc_easy,piqa,arc_challenge,mmlu,mmlu_continuation --num_fewshot 5 --device xpu` per HF, output `results.json` with `n-shot` + group key | Aurora `workq` | Gate |
| 4 | **SFT probe A on 50k + 100k + 143k** (3 probes) | Sunspot `workq 2N`, TRL `gsm8k-r1cot` 93 steps on each CPT HF, then `submit_cot_eval.sh` GSM8K-CoT `200/200 temp 0.0 max_tokens 768` | Sunspot `workq` | 150k gate |
| 5 | **Diagnose validator** | Open rank-0 JSONL for `12483856`, diff `loss` vs `val_loss` at `110700-111200`; if ratio ~= 32 or 6.5, file bug with 300-step repro (`--validator.steps=10`) | Sunspot login | None (info) |

ETA to gate: ~11h single-node-equivalent walltime, all 1N except 3x 2N
10-minute probes. CPT chain continues untouched.

### 7.2 At the 150k gate (decision per Sec. 4.4)

| Outcome | Action | Command |
|---|---|---|
| **PASS** (all G1-G4 hold) | Continue CPT to 190k, score P3 | `qsub -l select=4 -l walltime=12:00:00 -v NHOSTS_TRAIN=4 cpt_fineweb_edu_4n.sh` (auto-resumes); then HF convert `160k/175k/190734` + eval + final broad SFT |
| **WARN** (commonsense -2pp but probe passes) | Continue to 190k, flag mix ratio for review | Same + open issue: "60% FWE trades HellaSwag for CoT ceiling — evaluate 75/25" |
| **FAIL commonsense** | Fork SFT from last-good (max `G1-G2` score), stop CPT | SFT on `step-$LAST_GOOD/hf` via `tulu_math_uc_mix` (stage-1 general), then narrow CoT — compare to B2 |
| **FAIL probe** (commonsense holds, CoT <=0.10) | CPT did not move ceiling — pivot to broad distill on MDS base | Stop CPT; run `sft_distill_cot_4n.sh --pretokenized_dataset /tegu/datasets/agpt2b-distill-cot-mix-len8192` on `global_step138650` (no CPT) |

### 7.3 After 190k (if gated through)

1. Final HF: `step-190734/hf` on Tegu.
2. Final base eval: full suite (commonsense + MMLU both formats + `eval/aggregate_evals.py`).
3. Final SFT: `tulu_math_uc_mix` (general) on 190k base -> then `gsm8k-r1cot`
   narrow, matching B2's two-stage structure. Compare IFEval 0.253 / CoT 0.205.
4. If final SFT beats B2: new deliverable. If not: CPT was a distribution
   shift that lowered entropy without lifting the SFT ceiling — document and
   promote the broad-distill-MDS-base path instead.

### 7.4 What NOT to do

- **Do not use validation loss 12.53 to make a stop decision.** It is a
  miscalibrated instrument (Sec. 1.3). Fix it, then re-read.
- **Do not fork SFT at 143k blind.** The CPT-17k SFT at 0.015 already showed
  early-CPT + fixed SFT hparams is confounded. Score the ladder first.
- **Do not gate promotion on MMLU MCF.** Constant-D 0.2689 beats all 12
  measured agpt-2B MMLU configs (0.2413-0.2655) — `mmlu-letter-prior-gating.md:11-20`.
  Gate on ARC-Easy/HellaSwag/GSM8K-CoT; report `max(CF,MCF)` with the
  constant-letter band.
- **Do not compare hand-rolled MMLU macros across runs.** The macro over
  `mmlu_*` double-counts category rollups and reads ~0.005 high (verified
  0.2631 vs 0.2579 on step-46429). Read the `mmlu` group key and `n-shot`
  fields from `results.json` (`mmlu-letter-prior-at-chance.md:65-70,
  mmlu-letter-prior-gating.md:136-139`).
- **Do not train SFT or CPT on HF checkpoints.** CPT/SFT are DCP-native via
  native torchtitan; HF exports are eval-only bridges and TRL `BASE_MODEL`
  inputs. Use `flavor 2b-mds` for HF export or get silent gibberish
  (memory: "HF exports exist ONLY as an eval bridge — never train on HF
  checkpoints directly").
- **Do not edit outside `torchtitan/experiments/ezpz/docs/` or
  `torchtitan/experiments/ezpz/scripts/`** (assignment contract + Golden
  Rule #1: core `trainer.py`/`training_engine.py` never imports
  `torchtitan/experiments/`).

---

## 8. Appendix — Key File Paths

| Path | What |
|---|---|
| `torchtitan/experiments/ezpz/docs/production/cpt/README.md` | Dolmino CPT sweep: loss vs downstream divergence, gentle-LR hypothesis |
| `torchtitan/experiments/ezpz/docs/production/POST-TRAINING-2B.md` | B2/B2v2/B3/B4, CPT 111k chain, SFT/RL status |
| `torchtitan/experiments/ezpz/docs/production/POST-TRAINING-PLAN.md` | 3-phase plan: audit -> CPT 60/40 + broad distill -> RL guardrail |
| `torchtitan/experiments/ezpz/docs/production/sft/fineweb-edu-60-40-kind-shift.md` | 60/40 rationale: 94.8% DCLM table, SmolLM2 33.6% @38B, dolmino degree-vs-kind |
| `torchtitan/experiments/ezpz/scripts/cpt_fineweb_edu_4n.sh` | Live CPT launcher: 4N PPN=8 GBS=64 seq8192 100B/190734 steps LR 2e-6 const warmup20 |
| `torchtitan/experiments/ezpz/agpt/config_registry.py:1046-1057` | `agpt_2b_mds_mix_owm_edu_6040` def (60/40 via Grain `DatasetMixConfig`) |
| `torchtitan/experiments/ezpz/agpt/config_registry.py:919-928` | `_agpt_2b_mds_mix_base` — shared fork (MDS base, constant LR 2e-6) |
| `torchtitan/experiments/ezpz/docs/evals/mmlu-letter-prior-at-chance.md` | 12-config MMLU at chance, letter prior `p(A,B,C,D)`, CF vs MCF `0.2386 vs 0.2281` at 30B 3.93B tok |
| `torchtitan/experiments/ezpz/docs/evals/mmlu-letter-prior-gating.md` | Gating rule: do not gate on MCF MMLU, use ARC-Easy/HellaSwag/GSM8K-CoT |
| `torchtitan/experiments/ezpz/docs/evals/README.md` | Eval pipeline: DCP->HF->lm-eval, 4-panel commonsense thermometer, MMLU/GSM8K backfill |
| `torchtitan/experiments/ezpz/docs/evals/eval-landscape-2026-07.md` | Modern-suite review: 608-result corpus, MMLU/GSM8K coverage audit |
| `torchtitan/experiments/ezpz/docs/production/README.md` | Umbrella + trajectory status (historical + current per-seat jobs) |
| `torchtitan/experiments/ezpz/docs/journal.md` | Dispatch-by-dispatch CPT journal: 0->4091 (2.068) -> 7869 (2.463) -> 10143 (1.856) -> 34298 (1.93) -> 111200 |
| `torchtitan/experiments/ezpz/scripts/eval/hf_batch_convert_cpt_fwe.sh` | HF batch converter: `flavor 2b-mds bf16`, Tegu `HF_ROOT`, seq loop with fixups |
| `torchtitan/experiments/ezpz/docs/guides/loss-reporting-tp-dist-reduce.md` | Loss divisor bug history: `_dist_reduce` under-reported by `dp_world_size` |
| `torchtitan/experiments/ezpz/eval/convert_to_hf.py` | DCP->HF converter (flavor dispatch) |
| `torchtitan/experiments/ezpz/eval/configs/agpt_2b_mds_config.json` | HF config for `2b-mds` exports (copied to each HF) |
| `assets/hf/gemma-7b/tokenizer.*` | Tokenizer for HF exports (Gemma vocab 256000, md5 `b60c6af2`) |

---

## 9. Appendix — Loss Trajectory: What Healthy Looks Like vs What We Have

### 9.1 Training loss reference

```
MDS base (gs138650, 4.674T):  loss ~2.65 (plateau after ~635B flat — cpt/README.md:40-45)
Dolmino-100 start (OOD):      loss  7.27 (distribution shock — cpt/README.md:62-64)
Dolmino-100 end (5,960 steps): loss 2.48  (beats plateau by -0.31 but benchmarks tank)
FWE 60/40 start:              loss  2.068 (4091, barely above plateau — less OOD than dolmino)
FWE 60/40 MBH:                loss  1.856 (10108, new low — lower entropy than olmo-mix)
FWE 60/40 mid:                loss  1.93  (34298, flat/slightly up — within noise)
FWE 60/40 validator:          loss 12.53 (111200, flat — instrument broken, ~6.5x too high)
```

The dolmino shock (7.27) was large because the base had never seen dolmino;
FWE 2.068 is not a shock — the base is already 40% OWM-adjacent and FWE is
web-derived, so the OOD gap is small. That is consistent with health (no
LR re-warm shock) and with the entropy point (FWE is more predictable than
olmo-mix, so loss *should* be lower).

### 9.2 What a real plateau would look like

A plateau on training loss would be: loss flat for >10k steps with
`grad_norm` decaying and `tps` stable, while downstream commonsense is also
flat. The base plateau was 635B tokens flat on HellaSwag 0.560 / ARC-Easy
0.651 with val loss saturated at ~2.80 (`cpt/README.md:40-44`). We have 34k
steps (~17.8B tokens) of FWE — 1/35 of the base plateau window — so calling
a post-10k flattening from 1.856 to 1.93 a "plateau" is premature. The
dolmino pilot ran the full 300B (5,960 steps at GBS 6144 = 51B? actually
~300B with their GBS) and *still* had falling loss while benchmarks fell —
loss plateau is not the right lens.

### 9.3 The question the prompt raises that loss cannot answer

> "Is this learning or plateauing? What does the loss trajectory vs the
> prior dolmino CPT (dropped HellaSwag -7pp) tell us?"

Answer: **Both CPTs lower loss on their new distribution; only downstream
eval tells them apart.** Dolmino lowered loss *and* hurt downstream; FWE
lowers loss and we have not checked downstream. The loss trajectory alone
repeats the dolmino story so far — the divergence is entirely in the
benchmarks we have not run. That is why the entire action plan is about
running them.

---

## 10. Appendix — Why the SFT-then-CPT vs CPT-then-SFT Counterfactual Matters Beyond ARC-Easy

The prompt asks specifically about ARC-Easy because that is where dolmino's
two modes separated (ratio-independent HellaSwag vs ratio-dependent ARC-Easy).
But the deeper counterfactual is:

> **Does continued pre-training on a shifted distribution produce a *better
> starting point for SFT* than the original base?**

B2's 0.205 came from the MDS base *without* CPT. If FWE CPT hurts ARC-Easy
(and especially if it hurts FineMath NLL on the frozen holdout), it has
likely forgotten math-relevant representations — and the SFT that follows will
have to re-learn them, starting from a worse place than the MDS base. The
early detectors in Sec. 6.3 are not just about ARC-Easy; they are about
whether the CPT base is still SFT-able.

Historical precedent for "CPT can hurt downstream even when loss looks good"
extends beyond dolmino:

- `checkpoint-8672` (full OpenMathInstruct-2 SFT to epoch 1.0 at LR 2e-5 cosine)
  forgot HellaSwag 0.593 -> 0.273 and ARC-Easy 0.694 -> 0.298 — at chance —
  because cosine kept LR >1e-5 for ~4000 steps on narrow math
  (`POST-TRAINING-2B.md:50-51,57-58,126-127`). That was SFT, not CPT, but the
  mechanism (too much LR on a narrow distribution destroys general capability)
  is identical to the dolmino LR-shock hypothesis.

- B2v2 (more epochs on narrow `gsm8k-r1cot`): training loss 0.8069, acc 0.8198
  descending cleanly, eval halved 0.205 -> 0.09/0.105
  (`POST-TRAINING-2B.md:68-88`). Healthy training, collapsed eval — same
  divergence as CPT loss vs benchmarks.

The common thread: **narrow-distribution training at non-trivial LR destroys
general capability while improving its own loss.** CPT FWE at 60% FWE is
narrower than the MDS base's olmo-mix (which was already 94.8% DCLM + diverse
minor corpora) but broader than pure-OWM or pure-edu. Whether 60% FWE is
"too narrow" is exactly what the held-out FineMath NLL detector (D3) tests.

If D3 shows FineMath NLL rising while wikitext barely falls (the Wave-1
`+0.308 / -0.024` pattern at 100% edu), the 60% FWE is too aggressive and the
fix is the 75/25 OWM-heavy mix that "already captured ~all edu wikitext gain
at zero FineMath cost" (`config_registry.py:1038-1040`).

---

*Report prepared for `torchtitan/experiments/ezpz/docs/exploration/thread-a-cpt-trajectory.md`
on `ezpz-cpt111k-eval-sft` (`bcbca5bd90`). No files outside
`torchtitan/experiments/ezpz/docs/` or `torchtitan/experiments/ezpz/scripts/`
were modified. All citations are to on-disk docs at this branch point;
inferences are labeled as such.*
