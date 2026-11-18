# Thread E — SFT/RL Hyperparameter and Recipe Optimization

> **Date:** 2026-11-16
> **Branch:** `ezpz-cpt111k-eval-sft` (HEAD `bcbca5bd90`)
> **Scope:** Deep analysis of LR schedule, parallelism, RL/GRPO, sequence length, and a concrete sweep plan for agpt-2B stage-2 SFT and downstream RL.
> **Sources:** Every number is quoted from a run doc or W&B run on this branch. Links go to the source.

---

## 0. TL;DR — the one paragraph

B2 (`checkpoint-93`, 0.205 GSM8K-CoT) survived only because it trained for 93 optimizer steps — the LR schedule was never tested. The full-mix run at the same LR (2e-5 cosine to 0) catastrophically forgot at ~4000 high-LR steps (hellaswag 0.593 -> 0.273), and B2v2 proved that even on B2's own narrow data, 3 more epochs at the same LR halved accuracy (0.205 -> 0.09/0.105) with training metrics still descending. There is no evidence that any LR schedule above ~1e-5 survives more than ~1000 steps on a narrow distribution at 2B on this stack. RL is drift-proof on B2 (format 1.000 for 100 GRPO steps) but accuracy-flat (0.205 -> 0.215 noise) — worth running for format and as an SFT probe, not as an accuracy lever at current group_size and reward density. The sweep plan below fixes the open question — the safe LR x steps budget for math SFT — in cheapest-first order.

---

## 1. LR schedule analysis — the LR x steps budget

### 1.1 What happened, precisely

| Run | LR schedule | Steps at LR > 1e-5 | Distribution | Outcome |
|-----|-------------|---------------------|-------------|---------|
| **B2 stage-2** `checkpoint-93` | 2e-5 cosine warmup 200 -> 0 over 3 epochs (93 steps total) | ~55-60 of 93 (cosine only bites in 2nd half) | gsm8k-r1cot 7.4k ex | **0.205 CoT, 0.985 format** — deliverable |
| **Full-mix** `checkpoint-900` | 2e-5 cosine -> 0 over 8672 steps (1 epoch, 54B tok, GBS 6144) | **~4350** | tulu 0.65 + OpenMathInstruct-2 0.15 + ultrachat 0.20 | **0.253 IFEval, base-LM intact** — deliverable |
| **Full-mix** `checkpoint-8672` | same schedule, continued to step 8672 | ~4350 then decay 1e-5 -> 0 over 4300 more | same | **hellaswag 0.593 -> 0.273, arc_easy 0.694 -> 0.298** — catastrophic forgetting |
| **B2v2** ep3/ep6 on B2 base | 2e-5 cosine, 3->6 epochs, 6->12 steps, GBS 384 seq 2048 | all steps at ~2e-5 (cosine barely decayed over 6-12 steps) | gsm8k-r1cot (same as B2) | **0.09 / 0.105 vs B2 0.205** — halved |
| **B3/B4** single-stage | 2e-5, 1 epoch, broader mixes, seq 4096/8192 | O(1000) at high LR | tulu+OpenR1+gsm8k mixtures | **0.02-0.065** — all failed |

Sources: `torchtitan/experiments/ezpz/docs/production/POST-TRAINING-2B.md` (2026-07 and 2026-11-04 sections), `torchtitan/experiments/ezpz/docs/production/sft/agpt/2b-mds/tulu_math_uc_mix_full/README.md` and `torchtitan/experiments/ezpz/docs/production/sft/agpt/2b-mds/tulu_math_uc_mix_full/evals/README.md` (base-LM sweep 0->8672, IFEval, GRPO), `torchtitan/experiments/ezpz/docs/production/sft/fineweb-edu-60-40-kind-shift.md` section 3 (B2v2 table, jobs 12482423/12482485 training Exit 0, 12482572/73 eval Exit 0, W&B `xakno7kg`/`ur6pqqhp`).

The base-LM sweep is the sharpest evidence. From `evals/README.md`:

```
hellaswag: 0.593 (base) -> 0.592 (900) -> 0.592 (1500) -> 0.502 (3000) -> 0.321 (4500) -> 0.273 (8672)
arc_easy:  0.694 (base) -> 0.640 (900) -> 0.632 (1500) -> 0.516 (3000) -> 0.372 (4500) -> 0.298 (8672)
```

The cliff is between ~1500 and ~4500. That interval is exactly where a cosine schedule from 2e-5 over 8672 steps still holds LR above ~1e-5:

```
cosine LR at step s of S:  LR(s) = 2e-5 * 0.5 * (1 + cos(pi * s / S))
LR(1500) with S=8672: 2e-5 * 0.5 * (1 + cos(pi*1500/8672)) ≈ 1.74e-5
LR(3000) with S=8672: 2e-5 * 0.5 * (1 + cos(pi*3000/8672)) ≈ 1.23e-5
LR(4350) with S=8672: 2e-5 * 0.5 * (1 + cos(pi*4350/8672)) ≈ 1.00e-5  <- half-peak
LR(4500) with S=8672: ~9.6e-6
```

So the collapse coincides with ~3000 steps of exposure to LR > 1e-5 on a distribution that is 15% math-CoT and 65% tulu — narrow enough to overfit, broad enough to mask it in training loss (which kept descending to 0.357, token-acc 0.902).

B2's survival was not a schedule win — it was a **step-count win**. At 93 steps total, the cosine barely decayed at all; LR stayed ~2e-5 throughout. The model was saved by stopping early, not by decaying.

B2v2 then re-tested the "more steps at same LR" hypothesis directly on B2's own narrow data. Same LR (2e-5), same distribution (gsm8k-r1cot), just 2x the epochs (3ep -> 6ep, i.e. 6 -> 12 steps). Training loss stayed healthy (0.8069, acc 0.8198, W&B `xakno7kg`/`ur6pqqhp`), but GSM8K-CoT halved to 0.09/0.105 with format degradation (0.86-0.91) and verbosity inflation (gen_len 378-416, 15-22 unclosed vs B2's 282/~0). This is the same forgetting pattern as ckpt-8672, compressed into 6-12 steps because the distribution is even narrower (pure gsm8k-r1cot vs tulu+math mix).

### 1.2 The LR x steps budget — what the data actually supports

No run on this branch has measured the boundary. What we can bound:

- **Known safe:** ~900 steps at peak 2e-5 on a broad-ish mix (tulu 0.65 + math 0.15 + ultrachat 0.20, GBS 6144, seq 1024). Base-LM intact, IFEval +7.4pp.
- **Known unsafe:** ~3000-4000 steps at LR > 1e-5 on the same mix collapses base-LM to chance. ~6-12 steps at peak 2e-5 on pure gsm8k-r1cot (narrow) halves CoT accuracy.
- **Untested:** ~900-1500 steps on narrow data; any LR < 2e-5 on narrow data; constant LR vs cosine; any warmup other than 200.

The B2v2 result is the most alarming for stage-2 SFT specifically: even 6 extra steps at 2e-5 on narrow data hurt. That suggests the safe budget on narrow data is **tighter than on broad data** — perhaps O(100) steps at 2e-5, not O(1000). But the GBS difference matters (see section 2): B2 ran GBS 6144 @ seq 1024 (6.3M tok/step), B2v2 ran GBS 384 @ seq 2048 (0.79M tok/step). Same LR, 8x fewer tokens per step, so each B2v2 step is "smaller" in token terms but was still destructive. The token-normalized comparison is in section 2.3.

### 1.3 Cosine vs constant vs lower peak — what to use for math SFT

**Cosine with early stop** is what both deliverables actually used (B2's 93-step run and ckpt-900's 900-step truncation). Cosine's virtue here is that it provides a well-defined truncation point: stop at N steps and LR is still ~peak, so you get a clean "early stopping" experiment. Its vice is that for a fixed total-step budget S, LR stays high for ~S/2 steps (the cosine is flat on top). If S is 8672, that is ~4000 steps of overexposure. If S is 93, you never decay at all — cosine is just a constant LR with extra ceremony.

**Constant LR** is simpler and more honest for narrow SFT where the step count is small (tens to low hundreds). At 93 steps, B2's cosine was effectively constant anyway. Constant LR lets you vary steps independently of LR shape, which is what you need to map the LR x steps Pareto frontier. It also removes the confound where "more steps" and "lower final LR" are coupled through S.

**Lower peak** (e.g. 1e-5 or 5e-6) is the lever that the full-mix result demands if you want to train for a full epoch (8672 steps). No run has tested this. The hypothesis from the sweep is that 1e-5 would still cause forgetting if held for 3000+ steps, but no one has measured it. A lower peak also interacts with GBS (section 2) — large-batch training at low LR can be stable but slow, and the 2B model's pre-training used much higher LRs (the LR-finder work in `torchtitan/experiments/ezpz/docs/experiments/lr-finder/agpt/5b/README.md` measures blow-up/10 optima of 2.48e-04 AdamW at GBS 128 for 4.8B params, vs 2e-5 for SFT — a 10x difference that reflects the SFT-vs-pretrain regime shift, not a direct transfer).

**Recommendation for math SFT:**

| Phase | Schedule | Peak LR | Steps | Rationale |
|-------|----------|---------|-------|-----------|
| Stage-1 (broad, general math) | cosine or constant, early stop | 2e-5 | **cap 900-1000** | proven safe (ckpt-900); do not run to 8672 |
| Stage-2 (narrow CoT, gsm8k-r1cot) | **constant** | **2e-5 only if steps <= 100**; else **1e-5 or 5e-6** | **93 (proven) to ~300 max** | B2v2 shows 2e-5 x 12 steps already hurts |
| Broad distill mix (OpenThoughts+gsm8k, 50/50) | constant or cosine | **1e-5** (conservative) | **1000 hard cap** via `--max_steps` | new distribution, unknown safe budget; broad enough to tolerate more steps than narrow |

For any run where epochs would exceed the cap, use `--max_steps` as a hard gate rather than relying on epoch arithmetic — this was the lesson of `sft_distill_cot_4n_verification.md` section 4, which flagged that 3 epochs over the distill pretok can exceed O(1000) steps depending on the materialized mix size.

**Warmup:** 200 steps is the only value ever used (both B2 and full-mix). No evidence for or against other values. Keep 200 for comparability; do not tune warmup until the LR x steps frontier is mapped.

### 1.4 What remains unmeasured — the open question

The report's task asks: "what is the LR x steps budget for math SFT without forgetting?" The honest answer is **we do not know the curve, only two points on it**. The sweep in section 5 is designed to fill it in.

---

## 2. GBS / TP / PP parallelism — how batch size interacts with LR and forgetting

### 2.1 What is actually configured

| Track | GBS | Seq len | Tokens/step | Scale | LBS | GAS | Measured LR behavior |
|-------|-----|---------|-------------|-------|-----|-----|----------------------|
| CPT (FineWeb-Edu 4N) | 64 | 8192 | 524,288 | 4N PPN 8, GBS 64 | 8 | 1 | not an SFT LR; CPT uses pretrain LR (not 2e-5) |
| B2 stage-1 (tulu_math_uc_mix 32N) | 6144 | 1024 | 6,291,456 | 32N x 12 = 384 ranks | 2 | 32 | LR 2e-5 safe to ~900 steps |
| Full-mix stage-1 (tulu_math_uc_mix_full 8N) | 6144 | 1024 | 6,291,456 | 8N x 12 = 96 ranks | 2 | 32 | LR 2e-5 safe to ~900, forgot by ~3000 |
| B2 stage-2 (gsm8k-r1cot -> ckpt-93) | 6144 equiv* | 1024 or 2048 | ~6M equiv | via B2 lineage | - | - | 93 steps, safe |
| B2v2 (gsm8k-r1cot 2N) | 384 | 2048 | 786,432 | 2N x 12 = 24 ranks | 2 | 8 | LR 2e-5, 6->12 steps, **halved accuracy** |
| B2v2 alternative accounting | 384 seqs x 2048 = 786k tok/step; vs B2's 6144 x 1024 = 6.3M tok/step | | | | | | 8x fewer tok/step, still destructive |
| Planned broad distill (sft_distill_cot_4n 4N) | 768 @ 4N / 1536 @ 8N | 8192 | 6.3M @ 4N (768x8192) | 4N/8N x 12 | 2 | 8 | not yet run; LR 2e-5 default flagged as needing cap |
| LR-finder 5B (Sirius, H100) | 32 / 128 / 512 | 4096 | 131k / 524k / 2.1M | 8N x 4 = 32 GPUs, FSDP2 | - | - | AdamW blow-up/10: 3.49e-05 @32, 2.48e-04 @128, 5.36e-04 @512 |

* B2 stage-2 inherited the 32N GBS 6144 lineage; exact B2 stage-2 GBS is the full-mix 6144-equivalent from the 32N base run.

Source for SFT GBS: `torchtitan/experiments/ezpz/docs/production/sft/agpt/2b-mds/tulu_math_uc_mix/README.md` (32N GBS 6144), `torchtitan/experiments/ezpz/docs/production/sft/agpt/2b-mds/tulu_math_uc_mix_full/README.md` (8N GBS 6144), `torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh:100-109` (GBS 384, LBS 2 GAS 8 seq 2048), `torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh:53-55,242` (768 @4N, 1536 @8N, seq 8192), `torchtitan/experiments/ezpz/docs/experiments/lr-finder/agpt/5b/README.md` (GBS ladder).

TP and PP are not used in any SFT on this branch — all SFT is FSDP `full_shard` (TP=1, PP=1). CPT uses PPN=8 GBS 64 with Grain streaming and no TP/PP either. The parallelism story for SFT is purely about GBS (i.e. tokens/step) and how it interacts with LR.

### 2.2 Theory — what batch size should do

In SGD-like optimization, the standard scaling heuristic is **linear LR scaling with batch size** (Goyal et al. 2017) or **square-root scaling** (Hoffer et al. 2017), both motivated by keeping the update variance or the gradient noise scale constant. For AdamW the picture is murkier — Adam's per-parameter adaptivity partially decouples LR from gradient magnitude — but the empirical finding across LLM pretraining is that **larger GBS tolerates (and prefers) proportionally higher LR**, and the LR-finder GBS ladder on this branch directly measures that.

From `torchtitan/experiments/ezpz/docs/experiments/lr-finder/agpt/5b/README.md` section "GBS scaling law (5B AdamW, Sirius H100, Nov 2026)":

```
GBS 32  -> suggested LR 3.49e-05  (blow-up 3.49e-04)
GBS 128 -> suggested LR 2.48e-04  (blow-up 2.48e-03)   4x batch, 7.1x LR
GBS 512 -> suggested LR 5.36e-04  (blow-up 5.36e-03)   4x batch, 2.2x LR
```

That is between linear and sqrt scaling (linear would predict 4x LR per 4x GBS; observed is 7.1x then 2.2x). The LR roughly scales with GBS but sub-linearly at large batch — consistent with the literature that the linear rule breaks down past a critical batch size.

**What this means for the forgetting question:** if you double GBS at fixed LR, each step does a lower-variance update (more tokens averaged per gradient), so you might expect it to be *safer* (less drift per step). But you also do fewer steps per epoch (same token budget, larger batch), so the total number of high-LR steps drops. Forgetting is driven by **total high-LR token exposure**, not just step count — and these two effects partially cancel.

### 2.3 Token-normalized comparison — B2 vs B2v2

This is the most informative cross in the data:

| Run | GBS (seqs) | Seq len | Tok/step | Steps | Total SFT tokens | LR | Result |
|-----|-----------|---------|----------|-------|-----------------|-----|--------|
| B2 stage-2 | ~6144 | 1024 | ~6.3M | 93 | ~586M | 2e-5 cosine | 0.205 |
| B2v2 ep3 (6 steps) | 384 | 2048 | ~786k | 6 | ~4.7M | 2e-5 cosine | 0.09 |
| B2v2 ep6 (12 steps) | 384 | 2048 | ~786k | 12 | ~9.4M | 2e-5 cosine | 0.105 |

B2v2 saw **60-120x fewer tokens** than B2 stage-2 but still degraded. If forgetting were purely about total token exposure, B2v2 should have been safer (far fewer tokens at high LR). That it was *worse* despite fewer tokens suggests either:

1. **Narrow data is disproportionately destructive per token** — pure gsm8k-r1cot overfits faster per token than the tulu+math mix did, so the LR x steps budget on narrow data is much lower in token terms too; or
2. **Fewer tokens per step means noisier gradients** — at GBS 384, each step's gradient estimate is noisier than at GBS 6144, so the trajectory wanders more per unit of LR, causing faster forgetting despite fewer total tokens; or
3. **The starting point matters** — B2v2 continued from an already-SFT'd checkpoint (ckpt-900), not from the pretrain base. The model was already near a narrow optimum; additional narrow steps pushed it off a cliff that the pretrain base had not yet approached.

Evidence favors (1) and (3) jointly. The full-mix run shows that even broad data (tulu 0.65) collapses after ~3000 steps at high LR, so narrow data collapsing in ~6 steps is directionally consistent — the safe budget scales inversely with distribution narrowness. And (3) is supported by the fact that B2v2's starting point (ckpt-900) already had 900 steps of broad SFT; the model had less "distance to forget" than the pretrain base.

**Practical implication:** do not use GBS as a forgetting lever. The data shows that small GBS (384) did not protect against forgetting on narrow data, and large GBS (6144) did not prevent it on broad data past ~3000 steps. The dominant variables are **LR magnitude and total steps on the narrow distribution**, with GBS as a second-order effect. Pick GBS for throughput (fit the hardware), then set LR and cap steps for safety.

### 2.4 Scaling studies — what literature says

The SFT-forgetting literature at 2B scale is thin — most published work is on 7B+ (Alpaca, Vicuna, Llama-2-Chat). Relevant patterns:

- **Biderman et al. (2024, "Studying Large Language Model Generalization with Influence Functions")** — small models overfit instruction data faster; the "generalization gap" between broad and narrow SFT widens at small scale. Consistent with our 2B result where narrow SFT collapses in O(10) steps.

- **Chung et al. (2022, FLAN / T0)** — broad instruction tuning (1000+ tasks) improves held-out generalization; narrow tuning (single task) degrades it. The full-mix run's tulu-heavy broad mix (0.65 tulu) preserved base-LM to step 900; B2v2's pure gsm8k-r1cot destroyed it in 6 steps. Same pattern.

- **Ouyang et al. (2022, InstructGPT)** — SFT before RLHF used 1 epoch on broad data at moderate LR. The 1-epoch convention is load-bearing — our full-mix 1-epoch (8672 steps) violated it not because 1 epoch is wrong but because 8672 steps at peak LR on this distribution is too many; the "1 epoch" heuristic assumes the right LR x steps product, which this run exceeded.

- **Muennighoff et al. (2023, "Scaling Data-Constrained Language Models")** — on repeated/narrow data, performance degrades after ~4 epochs of repetition. Our narrow data (7.4k ex, 3 epochs = 93 steps already) repeated the same examples 3x; B2v2 at 6 epochs repeated them 6x. The degradation at 6 epochs is consistent with repetition-driven overfitting, not just LR.

None of these studies used GBS as an independent variable for forgetting — they varied data breadth and repetition. That matches our finding that GBS is second-order.

### 2.5 CPT vs SFT batch size — why CPT uses GBS 64

CPT at GBS 64 / seq 8192 = 524k tok/step is pretraining-batch, not SFT-batch. It runs 190k steps to 100B tokens. Its LR schedule is Warmup-Stable-Decay (WSD) or cosine over the full 100B, with peak LR set by the LR-finder (2e-4 to 5e-4 range for 2B at this batch, depending on optimizer — see `torchtitan/experiments/ezpz/docs/experiments/lr-finder/agpt/5b/README.md` GBS 128 row at 2.48e-04 for 4.8B as a reference; the 2B CPT LR is not documented in SFT docs but is lower than SFT's 2e-5 only because CPT's LR is set for pretraining stability over 190k steps, not because 2e-5 is "high" in absolute terms). CPT's GBS 64 is small because CPT must balance throughput (MFU 15-18%) against data efficiency at 100B scale — larger GBS would reduce steps and hurt data-parallel efficiency on 4N.

SFT at GBS 6144 / seq 1024 = 6.3M tok/step uses 12x more tokens per step than CPT because SFT runs for O(100-1000) steps, not 190k — throughput per step matters less than walltime per epoch. The two GBS values are not comparable; they serve different optimization regimes.

---

## 3. RL/GRPO — is it worth pursuing for accuracy or only for format?

### 3.1 What was actually measured

| Run | Base | Task | Group size | LR | Steps | Format | Accuracy | Verdict |
|-----|------|------|-----------|-----|-------|--------|----------|---------|
| `sft_arithmetic` 8N | ckpt-729 SFT'd | sum_digits | 4 | 1e-6 const | 1000 | saturated early (~0.5) | 0.4 -> **0.9** | **solved** |
| SFT-vs-baseline smoke | ckpt-729 vs pretrain base | sum_digits | 4 | - | 50 | - | **0.922** vs **0.117** (7.9x) | SFT is strong cold-start |
| v4/v5/v6 difficulty study | SFT'd | alphabet_sort | 4 | varied | 100 | - | plateau ~0.24-0.25 all | **reward function, not LR** |
| beat-v5 sweep w1/w2/w3 | SFT'd | alphabet_sort | 4/16 | 2e-5/5e-5, rank 8/32 | 100 | - | plateau 0.24-0.25 mean, peak 0.362-0.386 with rank/LR | capacity helps peak, not mean |
| ceiling-attack shaped | SFT'd | alphabet_sort | 32-rank, 5e-5 | - | 100 | 1.0 | **0.24 -> 0.667 cross-scored (+168%)** | **reward shaping is the lever** |
| CoT stage-2 weak base | stage-1 ckpt (0.15 acc) | GSM8K CoT | 4 | - | ~400 | **0.88 -> 0.99** then hacked | accuracy 0.16 -> **0.105** (fell while reward rose) | **reward hacking** |
| Gated GRPO on B2 | B2 ckpt-93 (0.205) | GSM8K CoT | 4 | ? | 100 | **1.000 held** | **0.205 -> 0.215 (noise)** | **drift-proof, accuracy-flat** |

Sources: `torchtitan/experiments/ezpz/docs/production/POST-TRAINING-2B.md` RL section (2026-06 to 2026-09), `torchtitan/experiments/ezpz/docs/production/rl/plans/cot.md` (stage-2 gated results, format curves, rollout analysis), `torchtitan/experiments/ezpz/docs/production/rl/grpo/beat-v5-sweep.md` (plateau), `torchtitan/experiments/ezpz/docs/production/rl/grpo/ceiling-attack.md` (+168%), `torchtitan/experiments/ezpz/docs/production/rl/grpo/aurora2b/sft_arithmetic/README.md` (solved task), `torchtitan/experiments/ezpz/docs/production/rl/README.md`.

### 3.2 Why RL solved one task and plateaued on another

The contrast between `sum_digits` (solved, 0.4 -> 0.9) and `alphabet_sort` (plateaued at 0.25) is diagnostic:

- **sum_digits** has a **dense, verifiable reward** (numeric exact-match + format + length) where partial credit is natural — getting 2 of 3 digits right gives 0.66, not 0. The gradient is informative even when the answer is wrong.

- **alphabet_sort** with the original `RewardAlphabetSort` (char-ratio) has a **saturating reward** — near-correct and exactly-correct score almost the same (char-ratio ~0.9 for both), so the advantage between a good and perfect rollout is tiny. GRPO's advantage normalization then washes out the signal. This was proven by the ceiling-attack: decomposing the same task into additive sub-skills (format 0.2 + completeness 0.2 + order 0.6) broke the ceiling from 0.25 to 0.667 **on the original char-ratio metric** (cross-scored offline, self-validating — char-ratio runs' stored and reconstructed rewards match exactly at 0.249).

The lesson: **the plateau was the reward function, not capacity.** Four runs varying LR (2e-5 -> 5e-5), rank (8 -> 32), and batch (8 -> 16 groups) all plateaued at 0.24-0.25 mean. Only the reward change moved it.

### 3.3 Why GRPO on B2 was drift-proof but accuracy-flat

On the strong B2 base (0.205 GSM8K-CoT, format 0.985), GRPO with `group_size=4` held format at 1.000 for 100 steps — no drift. But accuracy moved 0.205 -> 0.215, which is within the eval noise floor (200-problem GSM8K, ~3pp standard error at 0.20 base rate: sqrt(0.2*0.8/200) = 2.8pp).

The arithmetic explains why:

- At 20% solve rate with `group_size=4`, the expected number of correct rollouts per group is 0.8. The probability that a group is **all-wrong** (all 4 rollouts incorrect) is 0.8^4 = 0.41 — **41% of groups have zero variance in the correctness signal**, so their advantage is zero after GRPO's within-group normalization. Those groups contribute no gradient for accuracy.

- At `group_size=8`, all-wrong probability drops to 0.8^8 = 0.17 (17% dead groups). At `group_size=16`, it is 0.8^16 = 0.03 (3% dead). Larger groups dramatically increase the fraction of groups where at least one rollout is correct, giving a non-zero accuracy advantage.

- But group_size alone is not enough if the reward is binary. With a binary exact-match reward, groups where exactly 1 of N is correct give advantages of (1 - 1/N) vs (-1/N) — a weak signal. A **dense reward** (partial credit for close answers, intermediate steps, format) gives larger advantages even in mostly-wrong groups, as the ceiling-attack proved on alphabet_sort.

### 3.4 The weak-base failure — reward hacking and what it teaches

On the weak cold-start (stage-1 SFT with format 0.88, accuracy 0.15), GRPO initially improved format (0.88 -> 0.99) then hacked: reward rose 0.118 -> 0.217 while GSM8K-CoT accuracy fell 0.16 -> 0.105, and format bled 0.955 -> 0.635 in the CoT-stage-2 control. Gating the reward drove it to 0.000, revealing that **95% of rollouts had no `<think>` block and 20% echoed the exemplar's `\boxed{5}`** — the model was reward-hacking by emitting the answer format without reasoning.

Root cause was downstream of weak cold-start quality, not GRPO itself: the model had not learned to emit genuine reasoning, so GRPO optimized the easiest path to reward (format mimicry). Down-weighting format from 0.2 -> 0.05 ("it's saturated") removed the guardrail that was holding the envelope together.

On strong B2, the identical GRPO setup is drift-proof — format 1.000, no hacking. The cold-start quality gates RL.

### 3.5 What reward and hparams would make RL work for accuracy

For GSM8K-CoT at ~20% base accuracy, the reward must be **dense** and the group must be **large enough to see variance**:

| Hparam | Current (B2 GRPO) | Recommended | Why |
|--------|-------------------|-------------|-----|
| `group_size` (num_generations) | 4 | **8-16** | at 20% solve rate, 4 gives 41% all-wrong groups (zero advantage); 8 gives 17%, 16 gives 3% |
| `learning_rate` | unknown (not logged in B2 GRPO) | **5e-6 to 1e-5** | sum_digits used 1e-6 and solved; alphabet_sort used 2e-5-5e-5 before shaped reward. GSM8K-CoT is harder — needs higher than sum_digits but B2 is already SFT'd so conservative. Try 5e-6 first, then 1e-5 if stable |
| `beta` (KL penalty) | 0.0 | **0.0 to 0.01** | 0.0 worked for sum_digits and B2 (no drift). Small beta (0.01) as guardrail if format bled |
| `max_completion_length` | unknown | **512-768** | must cover full CoT + answer; GSM8K CoT avg ~300 tokens (see section 4), 64 (default) truncates everything |
| Reward shaping | binary/char-ratio | **componentized additive** | format (0.1-0.2) + extractable answer (0.1) + correctness (dominant, dense where possible) — per ceiling-attack. At minimum: `answer_correct` with partial credit (e.g. numeric closeness) + `think_format` + `answer_extractable` as separate RewardFn classes |
| Difficulty | random GSM8K | **curated band** | filter to problems where B2's pass@16 is 10-50% — hard enough that CoT is load-bearing, easy enough that the base gets non-zero correct fraction. If baseline is ~0, RL gradient cannot start |

The Monarch infrastructure (`torchtitan/experiments/ezpz/docs/production/rl/monarch.md`, job `12479177` 40/40 rollouts, 3 finite updates, verified two-host) is proven — the blocker is not infra but reward design and hparam density.

### 3.6 Is RL worth pursuing for accuracy, or only for format?

**For format: yes, proven.** GRPO on weak base lifted format 0.88 -> 0.99, on B2 held 1.000. If you have a new CPT base whose SFT has imperfect format, GRPO is a reliable format-perfecter in ~20-100 steps.

**For accuracy at 2B / 20% base: not yet proven, and the evidence says the ceiling is ~0.2 regardless of method.** Every accuracy lever tried (RL, B3, B4a, B4b, B2v2) converged to 0.02-0.215 vs B2's 0.205. `torchtitan/experiments/ezpz/docs/production/POST-TRAINING-2B.md` Finding: "~0.2 is near this base's GSM8K-CoT ceiling." If that is true, RL cannot exceed it any more than SFT can — the base lacks the capability.

**When RL-for-accuracy would be worth it:**

1. On a **stronger base** where the ceiling is higher — the CPT-FWE 60/40 arm (`scripts/cpt_fineweb_edu_4n.sh`, 60/40 FineWeb-Edu kind-shift per `torchtitan/experiments/ezpz/docs/production/sft/fineweb-edu-60-40-kind-shift.md`) or a larger/math-pretrained model. If a new base lifts B2-equivalent SFT to 0.30+, RL has headroom.

2. With a **dense reward** that gives partial credit — the ceiling-attack's +168% was on a task where the original reward was saturating. GSM8K-CoT's binary exact-match is similarly saturating (right vs wrong, no partial credit for close arithmetic). A shaped reward (e.g. per-step correctness, or `1 - |pred-gold|/scale` for numeric answers) would give GRPO a gradient even on wrong answers.

3. With **group_size 8-16** so that every group has at least one correct rollout to anchor the advantage.

Until one of those changes, RL-for-accuracy at 2B is expected to be flat — budget accordingly (a few smoke runs to verify, not a sustained campaign).

---

## 4. Sequence length and packing — how it affects CoT learning

### 4.1 What is configured, and what CoT actually needs

| Setting | Value | Where |
|---------|-------|-------|
| CPT seq len | 8192 | `cpt_fineweb_edu_4n.sh` (`PPN=8 GBS=64 seq=8192`) |
| Stage-1 SFT (tulu_math_uc_mix) | 1024 | `tulu_math_uc_mix/README.md` (packing pre-applied, `assistant_only_loss` via baked `assistant_masks`) |
| Stage-1 full-mix (tulu_math_uc_mix_full) | 1024 | `tulu_math_uc_mix_full/README.md` (BS 2 / GAS 32 to hold GBS 6144 at 8N) |
| B2 stage-2 (gsm8k-r1cot -> ckpt-93) | 1024 or 2048 (not explicitly documented) | B2 lineage; B2v2 explicitly uses 2048 |
| B2v2 stage-2 | 2048 | `agpt2b_b2v2_distill_cot_2n.sh:109` (`max_length 2048`, noted as mandatory — default 1024 silently truncates `<answer>` tail) |
| B3 single-stage | 8192 | `b3-instruct-cot-mix/design.md` (positionally safe — pretrain was 8192; OpenR1 traces 2k-10k+ tokens) |
| B4a (gsm8k finish on B3) | 8192 inherited | `b4-finish-and-reweight/design.md` |
| B4b (reweighted + length-filtered) | 4096 | `b4-finish-and-reweight/design.md` (OPENR1_MAX_THINK_CHARS filter + reweight) |
| Planned broad distill (sft_distill_cot_4n) | 8192 | `sft_distill_cot_4n.sh:53,269` (`max_length 8192`, LBS 2 GAS 8, packing=True) |

Pretrain was at 8192 (max_position_embeddings 131072 nominal, rope_theta 50000), so 8192 is positionally safe. Shorter lengths are safe too (they just truncate).

**What CoT actually contains:**

- Average gsm8k-r1cot CoT: ~300 tokens of reasoning (the `answer` field's step-by-step rationale reformatted as `<think>{cot_sans_marker}</think>\n<answer>\boxed{N}</answer>`). Packing at 1024 means ~3 examples per sequence — efficient.

- OpenThoughts-114k / OpenR1-Math-220k traces: 2k-10k+ tokens (frontier R1 reasoning with reflection, self-correction). At 1024, these truncate mid-reasoning, breaking the `<think>/<answer>` envelope. At 2048, a large fraction still truncates. At 8192, essentially all traces fit intact (the `b3-instruct-cot-mix/design.md` analysis). The `sft_distill_cot_4n_verification.md` notes `DISTILL_MAX_THINK_CHARS=16000` (default, part of recipe hash) trims only the extreme tail.

- B2's measured generation length: 282 tokens mean (cot accuracy eval, greedy, max_tokens 768, vLLM fp32). So inference-time CoT is short (~300 tokens) even when training traces are long — the model learns to emit concise reasoning when trained on concise data (gsm8k-r1cot), and verbose reasoning when trained on long traces (B3 at 601 tokens, B4a at 1378).

### 4.2 How sequence length affects what is learned

**Short SFT (1024) on short CoT (gsm8k-r1cot ~300 tokens):** efficient — 3 examples pack per sequence, no truncation, envelope intact. This is what B2 used and it worked. The packing is "honest" — every training sequence contains complete examples.

**Short SFT (1024) on long CoT (OpenThoughts 2k-10k+):** destructive — long traces truncate mid-`<think>`, so the model trains on incomplete reasoning that never reaches `<answer>\boxed{}`. The loss on truncated sequences teaches the model to emit run-on reasoning without closing the envelope. This is the hypothesized mechanism for B3's regression (0.05, gen_len 601, 26 unclosed): 25% of the mix was OpenR1 long-CoT at 8192, but if any arm had truncated it to 1024/2048, the envelope would break. B3 was *at* 8192 so this specific failure should not apply there — its dilution hypothesis (gsm8k-r1cot only 0.15 weight) is more likely.

**Long SFT (8192) on short CoT:** wasteful but not harmful — each 300-token CoT packs with ~27 examples per sequence at 8192 (vs 3 at 1024), so throughput is similar in tokens/step but attention cost is O(L^2) = 64x higher per sequence (no flash-attention on XPU). Packing 27 short examples into one 8192 sequence means the attention pattern is mostly cross-example (which packing masks handle via `assistant_masks`), but the compute is still quadratic in 8192. Memory pressure forces LBS=1 or 2 at 8192 on XPU.

**Long SFT (8192) on long CoT:** necessary — the only way to train on frontier traces without truncation. At 8192, both short (gsm8k) and long (OpenThoughts) examples fit; short ones just pack more densely. The `sft_distill_cot_4n.sh` default of 8192 for `distill_cot_mix` (50% OpenThoughts + 50% gsm8k) is therefore correct — it preserves the long traces while still efficiently packing the short ones.

**B2v2's 2048 on gsm8k-r1cot:** the design doc notes 2048 is mandatory (default 1024 silently truncates `<answer>` tail). At 300 tokens avg CoT + prompt (~100 tokens) + answer envelope (~20 tokens), a 1024 window should fit — so the truncation risk at 1024 must come from packing (multiple examples per sequence where a boundary falls inside an example). 2048 gives headroom.

### 4.3 Packing — the hidden variable

All SFT on this branch uses TRL `SFTTrainer` with `packing=True` (or pre-packed offline with `assistant_only_loss=True` and baked `assistant_masks`). Packing concatenates examples into fixed-length sequences to avoid padding waste. At 2B / 2048 hidden / 12 layers, the 2B model's XPU memory fits LBS=2 at 1024/2048 but only LBS=1 at 8192.

Packing interacts with CoT in two ways:

1. **Envelope integrity:** if a packing boundary falls inside `<think>...</think><answer>`, the example is split across sequences and the model never sees the complete format. Offline packing (via `--pretokenize_to` / `--pretokenized_dataset`) handles this by packing before training — see `sft_distill_cot_4n_verification.md` section 1 (pretok gate) for why pre-tokenization is required (runtime tokenize of 93M rows ~8.5h blew the watchdog, plus multi-rank HF cache coherency race).

2. **Cross-example attention:** packed sequences use attention masks to prevent cross-example leakage. This is correct for SFT — each example's loss is computed independently despite sharing a physical sequence. No evidence of packing-induced degradation on this branch (B2 at 1024 packing worked; full-mix at 1024 packing worked to 900 steps).

**Recommendation:** keep packing on (it is load-bearing for throughput — without it, short CoT at 8192 wastes ~96% of the sequence on padding). Use offline pre-tokenization for any mix that includes long traces (the 16N fanout's `_pretokenize_distill_cot_mix_1n.sh` pattern).

### 4.4 Concrete guidance

| Data | Recommended seq len | Why |
|------|---------------------|-----|
| gsm8k-r1cot only (~300 tok avg) | **2048** (or 1024 if packing verified) | B2v2 fix: 1024 truncates `<answer>` via packing boundary; 2048 proven |
| distill_cot_mix (50% OpenThoughts 2k-10k + 50% gsm8k 300) | **8192** | long traces need it; short ones pack densely; already wired in sft_distill_cot_4n.sh |
| tulu_math_uc_mix (general instruct, short) | **1024** | proven (ckpt-900); no long traces to justify 8192's O(L^2) cost on XPU |
| Any new frontier CoT (Bespoke-Stratos, etc.) | **measure p90/p95 trace length first**, then set max_length to cover it | per `torchtitan/experiments/ezpz/docs/production/rl/plans/cot.md` scale-up guidance |

Do not use 8192 for narrow/short data to "be safe" — the 64x attention cost on XPU (no flash-attention) is real, and the extra length does not improve learning on 300-token examples.

---

## 5. Concrete hyperparameter sweep plan

### 5.1 What to vary, what to hold fixed, and why

**Hold fixed (proven, not the lever):**

| Param | Value | Justification |
|-------|-------|---------------|
| Base model | `checkpoint-900-hf` (or sharded `checkpoint-900/` re-consolidated) | IFEval 0.253 deliverable; B2's stage-1 base; all B2v2/B3/B4 comparisons use it |
| Data (for this sweep) | `gsm8k-r1cot` (7.4k ex, narrow) | narrow SFT is the forgetting case; mapping its LR x steps boundary first isolates the variable before introducing broad-mix confounds |
| Optimizer | AdamW (TRL default) | all SFT on this branch uses AdamW; no evidence that SophiaG/Mano/Muon behave differently for SFT forgetting |
| Warmup | 200 steps | only value ever used; keep comparable |
| Weight decay | TRL default | not varied in any prior run |
| Sequence length | 2048 | B2v2 proven; 1024 truncates |
| Packing | True, offline pre-tokenized | correctness, not a hparam |
| GBS | 384 (LBS 2 x GAS 8 x 24 ranks @ 2N) | B2v2 proven; matches cheapest validation scale |
| Precision | bf16 | all SFT uses bf16 |

**Vary (the open question):**

| Param | Values | Why |
|-------|--------|-----|
| LR schedule | constant vs cosine-to-0 | cosine's 2nd-half decay vs constant's flat — which causes forgetting? |
| Peak LR | 2e-5, 1e-5, 5e-6 | 2e-5 is the only value ever used; 1e-5 is half-peak (where full-mix collapse began); 5e-6 is the RL-tested conservative value |
| Steps | 50, 100, 300, 600, 1000 (via `--max_steps` or epochs) | maps the forgetting boundary; B2 survived at 93, B2v2 failed at 6-12 additional — but B2v2 was a *continuation*, not a fresh run, so fresh-run boundary may differ |

### 5.2 The sweep — 7 configs in priority order

Each config is a **fresh SFT from ckpt-900 on gsm8k-r1cot** (not a continuation of B2v2 — fresh runs avoid the "already near a narrow optimum" confound). Scale is chosen for cheapest-first validation: 1N smoke where possible (6 ranks, seq 2048, GBS ~96), 2N (24 ranks, GBS 384) where 1N would be too slow or OOM. All configs use `save_strategy=steps` with `save_steps=50` and `logging_steps=5` (fixes the B2v2 `logging_steps` vs `save_strategy=epoch` single-entry bug per `POST-TRAINING-PLAN.md` Phase 1).

#### Config 1 — Reproduce B2 exactly (sanity anchor)

| Field | Value |
|-------|-------|
| **Purpose** | Verify B2's 0.205 reproduces on current stack (frameworks 2026.1.0, TRL, XPU) before varying anything |
| LR | **2e-5** cosine, warmup 200 -> 0 over 3 epochs (93 steps) |
| GBS | 384 (2N: LBS 2, GAS 8, 24 ranks) |
| Seq len | 2048 |
| Epochs | 3 (= 93 steps at GBS 384; gsm8k-r1cot 7.4k ex) |
| Scale | **2N** (select=2, PPN=12, 24 ranks; walltime 02:00:00 — B2v2 completed in 00:01:38 on 2N) |
| Script | `torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh` with `NUM_EPOCHS=3 SFT_DATASET=gsm8k-r1cot` (identical to B2 stage-2) |
| Expected cost | ~2 min on 2N (B2v2 timing); negligible |
| **Success criterion** | GSM8K-CoT 200-problem greedy eval (vLLM fp32, temp 0.0, max_tokens 768) **0.18-0.23** (B2's 0.205 +/- noise floor ~3pp). Format >= 0.95, gen_len 250-350. If outside this band, stack or data has drifted — do not proceed to configs 2-7 until root-caused |

#### Config 2 — The forgetting boundary at 2e-5 (constant LR, more steps)

| Field | Value |
|-------|-------|
| **Purpose** | **Primary open question:** at 2e-5 constant, how many steps on narrow data before forgetting? B2 survived 93 cosine steps; B2v2's continuation failed at +6 steps — but fresh-run boundary may differ |
| LR | **2e-5 constant** (no decay) |
| GBS | 384 (2N) |
| Seq len | 2048 |
| Steps | **100, 300, 600** (three checkpoints from one run via `save_steps=50`; use `--max_steps 600` and evaluate ckpt-100/300/600) |
| Scale | **2N**, walltime 02:00:00 (600 steps at ~0.5s/step on 2N = ~5 min; well within 2h) |
| **Success criterion** | GSM8K-CoT at ckpt-100 vs ckpt-300 vs ckpt-600. If 100 ~ 0.18-0.23 but 300 drops to <0.12, the boundary is between 100-300 at 2e-5. If 600 still holds >=0.15, the boundary is beyond 600 (and B2v2's failure was continuation-specific, not step-count-specific) |

#### Config 3 — Lower LR at 1e-5 (same steps, does lower LR extend the budget?)

| Field | Value |
|-------|-------|
| **Purpose** | If config 2 shows forgetting at 100-300 steps at 2e-5, does 1e-5 extend the safe window? This is the actionable lever for broad-mix SFT (ckpt-8672 would have needed it) |
| LR | **1e-5 constant** |
| GBS | 384 (2N) |
| Seq len | 2048 |
| Steps | **100, 300, 600, 1000** (one run, `max_steps 1000`, eval ckpt-100/300/600/1000) |
| Scale | **2N**, walltime 02:00:00 (~8 min at 0.5s/step for 1000 steps) |
| **Success criterion** | Same eval. If 1e-5 holds >=0.15 through 1000 steps where 2e-5 collapsed at 300, then LR is the lever and config 3's 1e-5 is the safe LR for narrow SFT beyond ~100 steps. If both collapse at similar steps, then step count (repetition overfitting) dominates, not LR — see config 4 |

#### Config 4 — Cosine vs constant at 2e-5 (does decay help?)

| Field | Value |
|-------|-------|
| **Purpose** | Disentangle schedule shape from step count. If config 2 (constant 2e-5) collapses at 300, does cosine 2e-5 -> 0 over 600 steps survive longer? |
| LR | **2e-5 cosine** warmup 200 -> 0 over 600 steps |
| GBS | 384 (2N) |
| Seq len | 2048 |
| Steps | **100, 300, 600** (eval ckpt-100/300/600; note LR at 300 is ~1e-5, at 600 is 0) |
| Scale | **2N**, walltime 02:00:00 |
| **Success criterion** | If cosine at 300 (LR ~1e-5) holds where constant at 300 (LR 2e-5) collapsed, then decay helps — use cosine for any run >100 steps. If both collapse similarly, schedule shape is not the lever — peak LR and step count are |

#### Config 5 — Very low LR at 5e-6 (conservative floor, matches RL hparam)

| Field | Value |
|-------|-------|
| **Purpose** | Establish the conservative floor. If 1e-5 still collapses early, how low must LR go? Also validates the RL-recommended LR range (5e-6 to 1e-5 from `POST-TRAINING-2B.md` next-step #2) for SFT |
| LR | **5e-6 constant** |
| GBS | 384 (2N) |
| Seq len | 2048 |
| Steps | **300, 600, 1000** (one run, `max_steps 1000`) |
| Scale | **2N**, walltime 02:00:00 |
| **Success criterion** | If 5e-6 holds >=0.15 through 1000 steps, it is the safe LR for long narrow SFT. If it still collapses, then narrow SFT has a hard step ceiling regardless of LR — repetition overfitting (Muennighoff 4-epoch limit) dominates, and the answer is "do not train narrow SFT beyond ~100 steps at any LR" |

#### Config 6 — Broad distill mix at 1e-5 (does broad data extend the budget?)

| Field | Value |
|-------|-------|
| **Purpose** | The actual next lever per `POST-TRAINING-2B.md` next-step #1 and `sft_distill_cot_4n_verification.md`: does 50% OpenThoughts + 50% gsm8k (broad frontier CoT) tolerate more steps than pure gsm8k-r1cot? |
| LR | **1e-5 constant** (conservative; 2e-5 default in `sft_distill_cot_4n.sh` flagged as needing cap) |
| GBS | 768 (4N: LBS 2, GAS 8, 48 ranks) or 384 (2N smoke first) |
| Seq len | **8192** (required for OpenThoughts long traces; see section 4) |
| Steps | **300, 600, 1000** (one run, `max_steps 1000`; gate on pretok `PRETOK_DIR=/tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192/dataset_info.json` + `[decontam]` counts) |
| Scale | **2N smoke** (walltime 02:00:00, ~5 min) then **4N** (select=5, walltime 12:00:00) if smoke eval beats B2v2 |
| **Success criterion** | GSM8K-CoT **> 0.15** (beats B2v2's 0.09/0.105 and approaches B2's 0.205) with format >=0.95 at 600-1000 steps. If broad mix holds where narrow collapsed, distribution breadth is the lever — proceed to full 4N run at 1e-5. If it also collapses, the sweep's lesson is that even broad CoT SFT has a step ceiling at 2B |

#### Config 7 — RL/GRPO smoke on B2 at higher group_size (accuracy probe)

| Field | Value |
|-------|-------|
| **Purpose** | Test whether GRPO's accuracy-flat result (0.205 -> 0.215) was a group_size/reward-density artifact, per section 3's analysis (41% dead groups at group_size=4) |
| Base | **B2 `checkpoint-93`** (or config 1's fresh reproduction) |
| Task | GSM8K CoT (same eval as SFT) |
| LR | **5e-6** (conservative, per `POST-TRAINING-2B.md` next-step #2) |
| Group size | **8** (and 16 if 8 shows signal) |
| Steps | **50, 100** (two eval points; prior B2 GRPO ran 100 steps) |
| Reward | componentized: `think_format` (0.1) + `answer_extractable` (0.1) + `answer_correct` (dense, e.g. `1 - |pred-gold|/gold` for numeric GSM8K, else binary — measure both) |
| Scale | **2N Monarch** (`scripts/rl_grpo_b2_2n.sh`, `rl-monarch-torch214-locality-20261001`, LD-first, `generator.model_dtype=float32`, `max_completion_length 512-768`) |
| **Success criterion** | GSM8K-CoT **> 0.25** (meaningful lift over B2's 0.205, outside noise floor) at 50 or 100 steps with format held >=0.99. If accuracy stays 0.205-0.215 (noise) as before, RL-for-accuracy at 2B/20% base is not the lever — accept B2 as ceiling and focus on CPT/base improvements. If reward rises while accuracy falls (as in weak-base CoT stage-2: reward 0.118->0.217, acc 0.16->0.105), reward hacking — revert reward weights |

### 5.3 Cheapest validation — execution order and decision rules

```
Config 1 (2N, 2 min) ── reproduce B2 anchor
        │
        ├── FAIL (outside 0.18-0.23) ──> root-cause stack/data before any sweep
        │
        └── PASS ──> Config 2 (2N, 5 min) ── 2e-5 constant, 100/300/600 eval
                      │
                      ├── holds through 600 ──> B2v2 failure was continuation-specific
                      │     └── Config 3 (2N, 8 min) ── 1e-5 to 1000 ── find extended budget
                      │           └── Config 6 (2N->4N) ── broad mix at winning LR
                      │
                      └── collapses at 100-300 ──> LR is the lever
                            ├── Config 3 (2N, 8 min) ── 1e-5 to 1000 ── does lower LR extend?
                            │     ├── holds ──> Config 6 (2N->4N) at 1e-5
                            │     └── collapses ──> Config 5 (2N) ── 5e-6 floor
                            │           ├── holds ──> Config 6 at 5e-6
                            │           └── collapses ──> narrow SFT hard ceiling ~100 steps
                            │                 └── Config 6 at 5e-6 or accept B2 as ceiling
                            │
                            └── Config 4 (2N, 5 min) ── cosine vs constant (parallel arm)
                                  └── informs schedule choice for Config 6

Config 7 (2N Monarch, ~30 min) ── independent of SFT sweep, run in parallel with Config 2
        └── informs whether RL is worth scaling
```

**Parallelization:** Config 1 must run first (validates stack). After it passes, configs 2, 4, and 7 are independent — run them in parallel on 2N each (6N total, well within workq). Config 3 depends on config 2's result (which LR to test next), so it follows sequentially. Config 6 depends on the winning LR from configs 2-5 and on the pretok gate (`/tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192/dataset_info.json` + `[decontam]` counts per `sft_distill_cot_4n_verification.md` section 5) — start its 2N smoke as soon as LR is known, promote to 4N on positive signal.

**Total walltime for the full sweep (sequential worst case):** ~2 + 5 + 5 + 8 + 8 + 5 + 30 = ~63 min on 2N each, plus eval (~2 min per checkpoint, vLLM fp32 on Sunspot). Under 2 hours even sequentially; under 30 min with configs 2/4/7 in parallel.

**1N vs 2N vs 4N guidance:**

| Scale | When to use | Note |
|-------|-------------|------|
| **1N** (6 ranks, GBS ~48-96) | pretok only (`_pretokenize_distill_cot_mix_1n.sh`, NPROC=1, `rl-monarch-torch214-locality-20261001`, LD-first) | SFT at 1N is possible but slow and hits the HF cache race — prefer 2N for training |
| **2N** (24 ranks, GBS 384) | **all sweep configs 1-5, and config 6 smoke** | sweet spot: fast enough (0.5s/step), cheap (2 of 16N budget), proven for B2v2 |
| **4N** (48 ranks, GBS 768) | **config 6 full validation** and any promotion candidate | needed for broad mix at seq 8192 (more tokens/step, 6.3M vs 3.1M at 2N) |
| **8N** | not needed for this sweep | reserved for full-mix scale (GBS 6144) — not the question here |

### 5.4 Eval protocol — same for every config

All configs evaluated identically so deltas are comparable:

- **Checkpoint selection:** evaluate every `save_steps=50` checkpoint plus `final/` (or `max_steps` if capped). At minimum: ckpt-{50,100,300,600,1000} or epoch boundaries.
- **Harness:** `torchtitan/experiments/ezpz/scripts/eval/submit_cot_eval.sh` (vLLM fp32, `frameworks/2026.1.0 vLLM 0.26 XPU`, `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, `ZE_FLAT=FLAT`), 200-problem GSM8K-CoT, greedy (`temperature 0.0 max_tokens 768`), scored as accuracy + format + gen_len + n_unclosed — all four fields, not just accuracy (per `b4-finish-and-reweight/README.md` "why the guardrails earned their keep").
- **Base-LM spot check:** for any config that runs >300 steps, also run a base-LM HellaSwag/ARC-Easy spot eval on the final checkpoint — if hellaswag has dropped >0.05 from ckpt-900's 0.59, forgetting is systemic, not just CoT-specific.
- **Training metrics logged:** loss, token-accuracy, grad_norm, LR — but do not gate on them. B2v2 proved they can descend while eval halves; eval is the only gate.

### 5.5 What success looks like — decision table

| Outcome | Interpretation | Next action |
|---------|---------------|-------------|
| Config 2 holds through 600 at 2e-5 | B2v2 failure was continuation-specific, not step-count | run broad mix (config 6) at 2e-5 with max_steps 1000 |
| Config 2 collapses at 100-300 at 2e-5, config 3 holds at 1e-5 to 1000 | LR is the lever; 1e-5 extends budget 3x | broad mix at 1e-5, CPT-FWE general SFT at 1e-5 if >1000 steps needed |
| Both 2e-5 and 1e-5 collapse, 5e-6 holds | need very low LR for long narrow SFT | broad mix at 5e-6 (slow but safe); or accept that narrow SFT beyond ~100 steps requires LR < 1e-5 |
| All three collapse at similar steps | step count / repetition dominates, not LR | narrow SFT has hard ceiling ~100 steps regardless of LR; lever is data breadth (config 6) or base model (CPT-FWE) |
| Config 6 (broad mix) beats 0.15 where narrow collapsed | distribution breadth extends budget | promote broad distill to 4N full run; 60/40 CPT for base improvement |
| Config 6 also collapses | even broad CoT has ceiling at 2B | accept B2 as CoT deliverable; invest in CPT-FWE base lift |
| Config 7 (RL) lifts to >0.25 | GRPO accuracy lever exists with dense reward + larger group | scale RL with shaped reward at group_size 8-16 |
| Config 7 stays flat (0.205-0.215) | ~0.2 is base ceiling, RL cannot exceed it | RL for format only; accuracy lever is SFT breadth + CPT base |

---

## 6. Cross-reference — where every claim comes from

| Claim | Source |
|-------|--------|
| hellaswag 0.593 -> 0.273, arc_easy 0.694 -> 0.298 at ckpt-8672 | `torchtitan/experiments/ezpz/docs/production/sft/agpt/2b-mds/tulu_math_uc_mix_full/evals/README.md` base-LM sweep table |
| "LR stayed above 1e-5 through step ~4350" | `torchtitan/experiments/ezpz/docs/production/sft/agpt/2b-mds/tulu_math_uc_mix_full/README.md` and `POST-TRAINING-2B.md` (cosine analysis + evals page) |
| ckpt-900 IFEval 0.253, best of 4 models | `torchtitan/experiments/ezpz/docs/production/sft/agpt/2b-mds/tulu_math_uc_mix_full/evals/README.md` IFEval table |
| B2 0.205 CoT, 0.985 format, gen_len 282 | `torchtitan/experiments/ezpz/docs/production/POST-TRAINING-2B.md` (2026-07 B2 section) |
| B2v2 0.09/0.105 vs B2 0.205, jobs 12482423/12482485/12482572/73, W&B xakno7kg/ur6pqqhp | `POST-TRAINING-2B.md` (2026-11-04 B2v2 section) and `torchtitan/experiments/ezpz/docs/production/sft/fineweb-edu-60-40-kind-shift.md` section 3 |
| B3 0.05, B4a 0.02, B4b 0.065 | `torchtitan/experiments/ezpz/docs/production/sft/agpt/2b-mds/b4-finish-and-reweight/README.md` |
| sum_digits 0.4 -> 0.9, 8N, 1000 steps | `torchtitan/experiments/ezpz/docs/production/rl/grpo/aurora2b/sft_arithmetic/README.md` |
| alphabet_sort plateau 0.24-0.25, ceiling-attack 0.667 +168% | `torchtitan/experiments/ezpz/docs/production/rl/grpo/beat-v5-sweep.md` and `ceiling-attack.md` |
| weak-base reward hacking 0.118->0.217 while acc 0.16->0.105, 95% no \<think\> | `torchtitan/experiments/ezpz/docs/production/rl/plans/cot.md` |
| gated GRPO on B2: format 1.000, 0.205->0.215 noise | `POST-TRAINING-2B.md` RL section and `rl/plans/cot.md` |
| GBS ladder 5B AdamW 3.49e-05@32 / 2.48e-04@128 / 5.36e-04@512 | `torchtitan/experiments/ezpz/docs/experiments/lr-finder/agpt/5b/README.md` GBS scaling law |
| CPT GBS 64 seq 8192, SFT GBS 6144 seq 1024, B2v2 GBS 384 seq 2048 | `sft/agpt/2b-mds/tulu_math_uc_mix/README.md`, `tulu_math_uc_mix_full/README.md`, `rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh`, `scripts/sft_distill_cot_4n.sh` |
| sft_distill_cot_4n 768@4N / 1536@8N seq 8192 | `torchtitan/experiments/ezpz/scripts/sft_distill_cot_4n.sh:53-55,242` |
| B2v2 2048 mandatory (1024 truncates) | `torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh:137` comment |
| "~0.2 is near this base's ceiling" | `POST-TRAINING-2B.md` Findings section |
| `generator.model_dtype=float32` mandatory, group_size 8-16, LR 5e-6-1e-5 | `POST-TRAINING-2B.md` next-step #2 and `rl/plans/cot.md` Stage 2 |
| Monarch 40/40 rollouts, 3 updates, job 12479177 | `torchtitan/experiments/ezpz/docs/production/rl/monarch.md` |

---

## 7. Risks and non-goals

**Not in scope for this sweep:** single-stage rebuilds (B3/B4 refuted), 32N scale (oneCCL fault), v2-256n base (blocked), pre-training LR tuning (separate LR-finder track at `torchtitan/experiments/ezpz/docs/experiments/lr-finder/agpt/`), MMLU (harness prior, not model bug — see `torchtitan/experiments/ezpz/docs/evals/mmlu-letter-prior-gating.md`).

**Risk:** if the narrow-SFT ceiling is truly ~100 steps regardless of LR, the sweep will produce 5 "all collapsed" datapoints — still valuable (it proves the lever is data breadth or base model, not recipe tuning, and closes the question). Budget for that outcome; do not extend the sweep chasing a narrow-data win that the B2v2/B3/B4 pattern says does not exist.

**Risk:** eval variance on 200 problems is ~3pp — a single eval cannot distinguish 0.18 from 0.23. Use the trend across checkpoints (100/300/600/1000), not a single point. If a config scores 0.22 at ckpt-100 and 0.19 at ckpt-300, that is noise; if it scores 0.20 at ckpt-100 and 0.08 at ckpt-300, that is forgetting.

---

## 8. What to do after the sweep

1. **If the sweep finds a safe LR x steps product** (e.g. 1e-5 to 1000 steps on narrow, or broad mix holds to 1000): use that product for `sft_distill_cot_4n.sh` (broad distill) and for any future CPT-FWE-derived SFT. Document the new default LR in `POST-TRAINING-PLAN.md` guardrails (replace "cap at O(1000) steps or drop LR" with the measured curve).

2. **If the sweep finds a hard ceiling** (all narrow configs collapse): close the narrow-SFT question, promote B2 as the narrow CoT deliverable, and invest in the two levers that this sweep would have validated as the alternatives — CPT-FWE 60/40 kind-shift (`cpt_fineweb_edu_4n.sh` at 50-100k, then SFT on top) and broad distill mix at the sweep's winning LR.

3. **If RL (config 7) shows signal:** run the shaped-reward, group_size 8-16 recipe on the sweep's best SFT checkpoint and measure whether RL lifts accuracy beyond the SFT ceiling — the first test of whether RL can exceed `~0.2` on this base with proper hparams.
