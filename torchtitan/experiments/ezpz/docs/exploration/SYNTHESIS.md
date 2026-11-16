# agpt-2B Post-Training Synthesis: Where We Stand and What to Do Next

> **Branch:** `ezpz-cpt111k-eval-sft` HEAD `bcbca5bd90` (2026-11-16).
> **Synthesizes:** Thread A (CPT trajectory), Thread B (broad-distill), Thread C (eval harness), Thread D (B2 autopsy), Thread E (recipe optimization) — **5/5 threads synthesized.**
> **Every number cites its source.** Inferences are labeled.

---

## 1. Executive Summary

**Where agpt-2B stands — one page.**

| Pillar | State | Key number | Source |
|--------|-------|-----------|--------|
| **CPT-FWE 4N** | **143.4k / 190.7k = 75.2%** of 100B budget; chain healthy, 12h walltime, MFU 15-18% | DCP frontier ~717 dirs, train loss ~1.86 at 10k; val loss 12.53 flat = broken validator, not plateau | Thread A §§1.1-1.3; `POST-TRAINING-2B.md` 2026-11-15; `cpt_fineweb_edu_4n.sh` |
| **Base for SFT** | `global_step138650` (MDS stage-3, 1.99B) remains canonical SFT base; ckpt-900 (IFEval 0.253) is its SFT'd derivative | ckpt-900 hellaswag 0.592, arc_easy 0.640 intact | Thread D §1.1.3; `tulu_math_uc_mix_full/evals/README.md` |
| **Best CoT deliverable** | **B2 = checkpoint-93: 0.205 GSM8K-CoT, format 0.985, gen_len 282** — two-stage SFT, the only recipe that worked | 93 steps stage-2 on gsm8k-r1cot (7.4k ex) after ckpt-900 | Thread D §§1-2; `POST-TRAINING-2B.md` B2 |
| **Ceiling evidence** | **4 independent failures converge on ~0.2 ceiling** on this base; every alternative 0.02-0.105 despite healthy training loss | B3 0.05, B4a 0.02, B4b 0.065, B2v2 0.09/0.105; GRPO on B2 0.205->0.215 noise | Thread D §§2,4; Thread E §§1-3 |
| **Catastrophe that defines the guardrail** | **ckpt-8672 forgot** hellaswag 0.593->0.273, arc_easy 0.694->0.298 — training loss 0.357 looked superb | ~4000 steps at LR>1e-5 on narrow mix at 2e-5 cosine | Thread D §3; Thread E §1.1 |
| **Eval harness** | **Two distinct harness problems masquerade as model problems** when misread | MMLU MCF 0.241-0.266 at chance (constant-D 0.2689 beats all); raw-CPT 0/200 CoT expected pre-SFT; 200-problem SE ±0.055 at 0.20 | Thread C §§1-3 |
| **MMLU truth** | At chance is the **correct null** for MCF at 2B — model emits letter prior p=(0.37,0.32,0.20,0.11), not knowledge | 12 configs 0.241-0.266 in 0.229-0.269 band; Llama control 0.312/0.653 proves harness not broken | Thread C §1; `mmlu-letter-prior-at-chance.md` |
| **Broad-distill lever** | **Pre-tokenized** `distill_cot_mix` = 50% OpenThoughts-114k-decontam + 50% OpenR1-Math-220k-decontam per `datasets_sft.py:_build_distill_cot_mix` (91k rows -> 36.9k packed @8192, 3.4G, GSM8K-test-decontam via 13-gram, job 12483443 `cba5c64`) — **landed, parked to CPT 50k, now fully analyzed** | Only open path adding new CoT data under proven two-stage structure; Thread B §§2-5: two-stage framing (gs138650->ckpt-900->distill_cot_mix at seq 8192, LR 2e-5, 48-144 steps) with A0-E7 ablation queue | `POST-TRAINING-2B.md` 2026-11-12; Thread B §§1-5 |

**Why B2 worked and nothing else did:** Stage 1 (tulu 0.65 + OpenMathInstruct-2 0.15 + ultrachat 0.20 at seq 1024, capped at 900 steps) taught instruction-following (IFEval 0.179->0.253) and shallow math without verbosity. Stage 2 (pure gsm8k-r1cot, 93 steps at 2e-5) wrapped that competence in `<think>/<answer>\boxed{}` with 3-epoch early stop. Every failure violated structure (B3/B4: one stage learns competing styles), violated early stop (ckpt-8672/B2v2: more narrow tokens at high LR destroy capability while loss descends), or violated base (B4a: verbosity baked in cannot be un-taught).

**Plateau vs learning:** Training NLL fell 2.07->1.86 on the FWE mix — expected for a lower-entropy distribution, not evidence of capability. The dolmino precedent is the warning: loss 2.80->2.49 while HellaSwag -7.4pp, ARC-Easy -10.4pp. The 12.53 flat validator is a divisor/miscalibration bug (~6.5x train loss at GBS=32), not a plateau. Only downstream eval discriminates.

**Single most important next action:**

> **Do not stop CPT. Do not fork SFT blind. Score what exists while CPT continues, then gate at 150k.**
>
> In parallel with the live 4N chain advancing to 190k on its 12h cadence, (a) HF-convert the P0-P2 ladder (0k-143k, 8 checkpoints, ~52G on Tegu), (b) run the commonsense thermometer + MMLU max(CF,MCF) on every HF, and (c) run one SFT probe (the proven B2 two-stage recipe) on the best intermediate CPT checkpoint. Apply the 150k gate (Sec. 5): if HellaSwag within -2pp, ARC-Easy within -3pp, and the SFT probe matches-or-beats B2's 0.205, continue to 190k; if either degrades, stop CPT and pivot to broad-distill on the last-good checkpoint. Cost to decide: ~8 HF converts + 2 eval sweeps + 3 SFT probes, all 1N, all `workq`, zero new CPT compute.

---

## 2. Ranked Next-Course Matrix

All viable courses, ranked by expected value (EV) per cost. Cost in Sunspot N-hours; EV judged on likelihood of lifting the 0.205 CoT ceiling without regressing commonsense.

| Rank | Course of action | Compute cost | Expected value | Key risk | Verdict |
|------|------------------|-------------|----------------|----------|---------|
| **1** | **Continue CPT to 190k on 12h cadence; gate at 150k via scoring ladder + SFT probe** | CPT: +47k steps / 24.8B tok = ~2.5 hops of 4N/12h (dominant cost already budgeted). Decision cost: ~8 HF converts (1N 2h each) + 2 eval sweeps + 1-3 SFT probes (1N 2h each) — all 1N, parallel with CPT | **Highest.** Only course that tests the FWE kind-shift thesis (>60% educational content, the 60/40 SmolLM2 operating point) at scale. External evidence: FWE 33.6% MMLU at 38B vs next-best 300B (8x efficiency), OLMo-2 +17.4 MMLU at 1B on same olmo-mix base. 40% OWM math floor hedges the dolmino 100% forgetting. If it lifts the SFT ceiling above 0.205, it unlocks the entire post-training stack | If ladder shows commonsense regression (dolmino repeat: HS -7pp by step 1000), ~2.5 hops burned. Mitigated by 150k early gate — <0.5 hop beyond current 143k snapshot | **Do this.** Default path in Thread A §4.3 and `POST-TRAINING-2B.md` phase ordering. Scoring and CPT are independent (DCP writes, HF reads) — never block CPT to score |
| **2** | **LR x steps sweep on narrow SFT (gsm8k-r1cot from ckpt-900)** | **~1.0 N-hour total.** 7 configs, 2N each, ~2-5 min train + ~2 min eval per ckpt. Sequential <2h, parallel with rank-1 work ~30 min | **High.** Maps the forgetting boundary that governs *every* SFT call: at 2e-5 constant, how many steps before collapse? Does 1e-5 extend it? Does cosine help? B2 survived 93 steps, B2v2 halved at +6 steps (continuation), full-mix collapsed at ~1500-4500 (cosine LR>1e-5). No narrow mix has been swept at any LR other than 2e-5. Answers the "cap at O(1000) steps or drop LR" rule with a curve, not a slogan — and sets the LR for course 3. Thread B's A0/A3 ablations run alongside rank 2 to gate rank 3 | Null result ("all collapse at ~100 steps regardless of LR") is still decisive — proves lever is data breadth, not LR tuning | **Do this in parallel with rank 1.** Cheapest experiment with broadest applicability. Thread E §5 is the runnable spec; Thread B A0 (artifact audit) + A3 (decontam n=8) are 0-GPU pre-gates for rank 3 |
| **3** | **Broad-distill SFT on MDS base (OpenThoughts-114k-decontam + OpenR1-Math-220k-decontam via pretokenized `distill_cot_mix`)** | 2N smoke ~5 min at LR 2e-5 (or 1e-5 conservative); 4N full ~1-1.5h, walltime 12h. Pretok already landed: 91k raw rows -> 36.9k packed @8192, 3.4G, GSM8K-test-decontam-verified via 13-gram + normalized-exact (`decontam_traces.py`, `[decontam]` drop counts logged, recipe hash `cba5c64`, job 12483443) — zero pretok cost remaining. Cost is training + eval only | **High.** Only open path adding *new* CoT data under proven two-stage structure. Two-stage framing: `gs138650 -> tulu_math_uc_mix -> checkpoint-900` (stage-1, IFEval 0.253) -> `distill_cot_mix` at seq 8192, LR 2e-5 cosine, 48-144 steps / 1-3 epochs (GBS 768 @4N), envelope `<think>/<answer>\boxed{}` byte-identical to gsm8k-r1cot. Stage-2 swaps narrow 7.4k gsm8k for 50/50 frontier mix — breadth *within* stage 2, not mixing instruct+CoT in one stage (B3/B4's failure mode). Tests whether data breadth, not mix weights, breaks the 0.205 ceiling. Gated by A0-E7 ablation ladder (Sec. 3 E7): A0 artifact audit (0 GPUs) -> E1 1ep gate (Go >=0.15 / Pivot 0.05-0.15 / No-go <0.05) -> E2 3ep + E2b narrow polish + LR/seq ablations. B3/B4's long-CoT dilution does not apply here — Thread B §1.3 structural hypothesis: dilution is a single-stage capacity-allocation problem. Per `POST-TRAINING-2B.md` phase 3, this is the next lever if CPT does not move the ceiling | Distribution shift (CPT-17000 0.015 warning: frontier science/code vocab out-of-distribution vs olmo-mix-1124 base; Thread B §2.3.1), length/packing (8192 OOM risk on XPU without flash-attn, cross-problem attention without block-causal masking; Thread B §2.3.2), quality (OpenThoughts correctness weaker than OpenR1 — unverified `\boxed{}` extraction, dropped code-gen rows reduce breadth; Thread B §2.3.3). Requires rank-2's winning LR (1e-5 conservative alternative) to avoid repeating ckpt-8672 at new scale. **Discrepancy note:** per `datasets_sft.py:_build_distill_cot_mix` the mix is OpenThoughts+OpenR1 (both frontier-distilled), NOT OpenThoughts+gsm8k-r1cot as the task prompt stated — see Thread B §2.1 and Sec. 6 risk 11 | **Queue after rank 2.** Start 2N smoke at winning LR as soon as rank 2 reports; promote to 4N if GSM8K-CoT >=0.15 and format >=0.80 (E1 Go gate). Full A0-E7 ladder in Sec. 3. Do not run at 2e-5 uncapped beyond E1's 48-step gate |
| **4** | **Fork SFT now on 143k CPT checkpoint (SFT probe as primary action, no ladder)** | 1N, 2h walltime (B2 recipe: 93 steps, GBS 384, seq 2048, LR 2e-5). Cost is one SFT run + 200-problem eval (98s) | **Medium.** Directly tests "does 75% CPT lift the ceiling?" — the single question that matters (Thread A §4.2). Cheapest way to get a data point *today* without waiting for 150k | Without the ladder, a 143k SFT number is uninterpretable: if it scores 0.08 (like B2v2/B3), you cannot tell whether CPT hurt, the recipe was wrong, or you sampled a bad checkpoint. Repeats B2v2's mistake (healthy loss 0.8069 while eval halved) — cost of a mis-fork is a wasted SFT cycle plus a wrong pivot decision. Rank 1's ladder disambiguates for <0.5 hop extra | **Do as part of rank 1, not instead of it.** Fork one SFT probe on the best pre-143k HF (e.g. 74k or 112k) *while* the ladder runs; treat 143k itself as P2's frontier point, not a solo bet |
| **5** | **Broad-distill SFT on CPT base (CPT-FWE 112k+ HF + `distill_cot_mix` as stage-2; Thread B E6)** | Same as rank 3 plus CPT HF dependency. 4N, GBS 768 @48 ranks, seq 8192, LR **1e-5 cosine conservative** (distribution-shifted base; 2e-5 re-warm may repeat CPT-17000 0.015 shock), 1ep ~48 steps, tokens/step ~6.3M. Gated on (1) CPT 112k+ HF verified (`config.json`+`tokenizer.json`+`model.safetensors` non-gibberish) and (2) narrow SFT probe on that base scored to calibrate format. Two variables change at once (base + data) — attribution requires rank-3 baseline (Thread B §3.1: MDS base `checkpoint-900` first, CPT base only after) | **High ceiling, high cost.** If CPT lifts the base (commonsense G1-G3 held, MMLU CF lifted), combining it with broad data is the maximum achievable on this stack — the kind-shift + breadth composition from SmolLM2 60/40 + OLMo-2 +17.4 precedent. This is `POST-TRAINING-2B.md` phase ordering (CPT to 50k+ then broad SFT) and Thread B's E6: gated on CPT 112k+ HF availability, conservative LR 1e-5, same `distill_cot_mix` artifact | Triple-blocked: (a) rank-1 gate (no CPT HF proven good yet), (b) rank-3 (broad mix must beat >=0.15 on MDS base first, else composition has no baseline), (c) CPT base itself is harder than MDS — CPT-17000 probe 0.015 shows shifted manifold + narrow SFT collapses. Changing two things at once without baselines is how B3 failed (mix + structure confounded). Precondition: narrow probe on CPT base (gsm8k-r1cot 3ep LR 2e-5 2N ~30 min) must be scored first | **Defer to after rank 1 + rank 3.** Do not run until (a) CPT gate passes at 150k and (b) broad-distill on MDS base (E1) has measured >=0.15 or Pivot result. Then clean A/B: CPT base vs MDS base, same distill mix, same LR (Thread B E6 gate table). If E1 was No-go (<0.05), do not compose — broadening on a shifted base will not rescue a mix that dilutes even on the proven base |
| **6** | **RL/GRPO on B2 (or on rank-3 winner) for accuracy** | 2N Monarch, 30 min per 100-step probe at group_size 8-16. Infra proven (job 12479177 40/40 rollouts, 3 updates, Exit 0) | **Low for accuracy, proven for format.** GRPO on B2 held format 1.000 for 100 steps (drift-proof) but accuracy 0.205->0.215 = noise. Root cause: at 20% solve rate and group_size=4, 41% of groups all-wrong (zero advantage); even group_size 8 leaves 17% dead. No RL run has used a dense reward or group_size 8-16 with correct `max_completion_length` (512-768, not default 64) | Reward hacking is real: on weak base, reward 0.118->0.217 while accuracy 0.16->0.105, 95% rollouts no `<think>`, 20% echoed `\boxed{5}`. Requires componentized reward (format+completeness+correctness as in ceiling-attack +168%) and curated difficulty band — neither wired for GSM8K-CoT yet | **Run one cheap probe (group_size 8, LR 5e-6, 50-100 steps) in parallel with ranks 1-2 to close the question.** If accuracy stays flat (0.205-0.215), accept B2 as ceiling for RL-for-accuracy and keep RL as format-perfector only. Do not scale RL until the SFT ceiling moves (ranks 1/3) — RL reweights rollouts the base already samples |
| **7** | **Kill CPT now (stop chain, write off 75% budget, revert to MDS base)** | Negative cost (saves ~2.5 hops) | **Negative EV unless ladder shows degradation.** Abandons the only kind-shift bet (60% FWE vs 94.8% DCLM olmo-mix) that has external MMLU precedent (+17.4 at 1B, 8x token efficiency at 350B). No replacement base exists — v2-256n 32N blocked on oneCCL fault, and MDS base is known to cap at ~0.205 | Dolmino trauma (300B pilot bled -7pp HS) makes this feel safe, but dolmino was 89% DCLM (degree tweak); FWE is a kind change with a 40% math floor. Killing at 75% without scoring would be the most expensive decision if FWE thesis is right | **Only if 150k gate catastrophically fails** (Sec. 5: commonsense <-3pp *and* SFT probe <=0.10, or monotonic bleed across ladder). Even then, revert to MDS base for SFT — do not kill without the broad-distill lever queued |

**Reading the ranking:** Ranks 1 and 2 are independent and both cheap — run them together. Rank 3 depends on rank 2's LR. Rank 4 is a sub-experiment inside rank 1. Rank 5 depends on ranks 1+3. Rank 6 is independent but low-EV — one probe suffices. Rank 7 is a fallback, not a plan.

## 3. Prioritized Experiment Queue

Ordered by dependency. Each entry states what it tests, exact runnable form, and the decision rule that consumes its output. Thread B's broad-distill deep-dive (data composition §2.1, decontamination audit, length distribution, A0-E7 ablation ladder) is now merged into E7 below — see E7's A0-A4 cheap ablations and ranked E1-E7 queue (Thread B §§2-5).


### E1 — HF conversion ladder (P0-P2 through 143.4k)

| Field | Value |
|-------|-------|
| **Tests** | Whether CPT checkpoints exist as scorable HF artifacts; catches converter flavor/omission bugs early |
| **Depends on** | DCP chain on `/home/foremans/torchtitan/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64` (556 dirs at 111k, ~717 at 143k; 23G frontier) |
| **Run** | Sunspot `workq` 1N 02:00, `VENV=xpu-torch216-sync-5a7a622f16` LD-first, `flavor=2b-mds` (wrong flavor = gibberish), `PBS filesystems=tegu:home` |
| **Convert** | `1000, 5000, 10000` (P0 shock horizon) + `25000, 50000, 75000` (P1; 74000/74200 already 3.7G 9/9 verified) + `100000, 111200, 125000, 143400` (P2 frontier) = **8 converts now**; P3 `150000, 160000, 175000, 190734` deferred |
| **Output** | `TEGU_ROOT/outputs/evals/cpt-fwe/step-$s/hf` (9/9 files: safetensors + config + tokenizer), ~3.7G each, ~52G full ladder — trivial vs 23G DCP frontier |
| **Decision** | If any convert fails or produces degenerate HF (missing config/tokenizer as at 74k pre-patch), fix converter before scoring — do not burn eval walltime on bad HF. Script: `torchtitan/experiments/ezpz/scripts/eval/hf_batch_convert_cpt_fwe.sh` (loop override for P0-P2 set) |

### E2 — Base-model eval sweep on every HF (commonsense + MMLU both formats)

| Field | Value |
|-------|-------|
| **Tests** | The dolmino question: did CPT regress commonsense (HS, ARC-Easy, PIQA) and did knowledge move off the letter prior (MMLU max(CF,MCF) + prior collapse)? |
| **Depends on** | E1 (HF exists) |
| **Run** | Aurora preferred for MMLU (Sunspot `level_zero 40 63G OOM` on `compute_topk_scores` unfixable on XPU — Thread C §4.3). CoT-free: `convert_and_eval.sh` DCP->HF -> `lm-eval 0.4.10` HF backend, `frameworks/2026.1.0` MKL+CCL+compiler+PTI LD as in `submit_cot_eval.sh`. CPT rows 1-4 of Thread C §5.2 |
| **Benchmarks** | HellaSwag `acc_norm` 0-shot, ARC-Easy `acc` 0-shot, PIQA `acc_norm` 0-shot, ARC-Challenge `acc_norm` 25-shot, **MMLU 5-shot MCF (`mmlu`) + CF (`mmlu_continuation`) -> `max(CF,MCF)`** + letter-prior fit `p(A,B,C,D)` and residual dispersion |
| **Scoring discipline** | Read group key `mmlu` from `results.json` (not hand-rolled macro: macro +0.005 high at step-46429 per Thread C §1.5#1). State format (MCF/CF/max), n-shot (5), constant-D band 0.2295-0.2689. Full 14k MMLU (not sampled — sampling drove 0.028 noise in 30B limit-10 vs limit-40) |
| **Decision** | Apply gate G1-G3 of Sec. 5 per checkpoint. Monotonic bleed (HS/ARC-Easy falling across P0->P2) = LR-shock or ratio bug — stop CPT even before 150k. Single -2pp wobble = noise, continue. MMLU recorded informationally; real MMLU gate is post-SFT (E3) + prior collapse (Thread C §1.5#7) |

### E3 — SFT probe on best CPT checkpoint (B2 recipe verbatim)

| Field | Value |
|-------|-------|
| **Tests** | The decisive question: does any CPT base, through the proven two-stage recipe, lift GSM8K-CoT above B2's 0.205 without regressing commonsense? |
| **Depends on** | E2 (which CPT checkpoint scored best on G1-G3; default: highest P2 checkpoint that passed G1-G3, else 74k/112k fallback) |
| **Run** | Sunspot 1N or 2N, `VENV=rl-monarch-torch214-locality-20261001` LD-first (not xpu-torch216 — TRL path), TRL `SFTTrainer` FSDP `full_shard`. Script `rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh`. Two stages: stage-1 `tulu_math_uc_mix_full -> checkpoint-900` equivalent on CPT base, stage-2 `gsm8k-r1cot` 93 steps 3ep LR 2e-5 cosine GBS 384 seq 2048 `assistant_only_loss` |
| **Baseline to beat** | B2 = 0.205 / 0.985 / 282 on same harness (greedy, temp 0.0, max_tokens 768, vLLM fp32 0.26, `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`) — every new SFT measured at 200 problems on the same stack |
| **Decision** | G4 of Sec. 5: promote new base iff CoT accuracy >= 0.265 (B2 +0.06 at 2sigma on 200) *and* format >=0.90 *and* unclosed <=10. If 0.15-0.20 (noise around B2), CPT did not move ceiling — pivot to E7. If <=0.10 (B2v2/B3/B4 band), CPT base is worse — revert to MDS, do not burn to 190k |

### E4 — Reproduce B2 exactly (sweep anchor, Thread E Config 1)

| Field | Value |
|-------|-------|
| **Tests** | Does B2's 0.205 reproduce on current stack (frameworks 2026.1.0 + TRL on XPU) before varying anything? |
| **Depends on** | Nothing — run immediately, in parallel with E1-E2 |
| **Run** | 2N Sunspot, GBS 384, seq 2048, LR 2e-5 cosine warmup 200 ->0 over 3 epochs (93 steps), gsm8k-r1cot from ckpt-900. Same script as E3 stage-2 but from canonical `checkpoint-900-hf` (IFEval 0.253). ~2 min train, ~98s eval (cf. B2v2 jobs 12482572/73 00:01:38 Exit 0) |
| **Decision** | Pass if 0.18-0.23, format >=0.95, gen_len 250-350. **If outside, halt sweep** — stack or data drifted; root-cause before E5-E7. This is the load-bearing sanity check |

### E5 — Forgetting boundary at 2e-5 constant (Thread E Config 2)

| Field | Value |
|-------|-------|
| **Tests** | At 2e-5 constant, how many steps on narrow gsm8k-r1cot before forgetting? Fresh SFT from ckpt-900, not continuation (avoids "already near narrow optimum" confound) |
| **Depends on** | E4 pass |
| **Run** | 2N, 2e-5 constant, GBS 384 seq 2048, `--max_steps 600`, `save_steps=50`, eval ckpt-100 / 300 / 600 on GSM8K-CoT 200 + HellaSwag spot at final |
| **Decision** | If ckpt-100 ~0.20 but ckpt-300 <0.12, boundary 100-300 at 2e-5 — LR is the lever, go to E6. If holds through 600, B2v2's halving was continuation-specific, not step-count — extend to E6 with longer horizon |

### E6 — Lower LR at 1e-5 to 1000 steps (Thread E Config 3) [+ cosine-vs-constant arm E6b]

| Field | Value |
|-------|-------|
| **Tests** | Does 1e-5 extend the safe window? And does cosine decay help vs constant? |
| **Depends on** | E5 result (which LR to test next) |
| **Run** | E6: 2N, 1e-5 constant, 1000 steps (one run, eval 100/300/600/1000). E6b: 2N parallel, 2e-5 cosine warmup 200 ->0 over 600 steps, eval 100/300/600 (at 300 LR~1e-5, at 600 LR=0) |
| **Decision** | If E5 collapsed at 100-300 but E6 holds >=0.15 through 1000, LR is the lever — E7 uses 1e-5. If both collapse, try 5e-6 (Thread E Config 5, ~1 N-hour) — if that also collapses, narrow SFT has a hard ~100-step ceiling regardless of LR (repetition overfitting per Muennighoff 4-epoch limit) and the lever is data breadth (E7). Cosine vs constant (E6b) informs schedule choice for E7 — pick whichever holds longer |

### E7 — Broad-distill on MDS base with A0-E7 ablation ladder (Thread B §§4-5; Thread E Config 6)

> **Mix definition (Thread B §2.1 discrepancy):** `distill_cot_mix` is **50% OpenThoughts-114k-decontam + 50% OpenR1-Math-220k-decontam** per `datasets_sft.py:_build_distill_cot_mix` and `_pretokenize_distill_cot_mix_1n.sh` header — both frontier-distilled (DeepSeek-R1). It is **NOT** OpenThoughts+gsm8k-r1cot as the task prompt stated. The 91k-row artifact matches the OpenThoughts+OpenR1 definition (both ~100k+ raw; 50/50 gsm8k would be much smaller or heavily upsampled). Envelope `<think>/<answer>\boxed{}` byte-identical to gsm8k-r1cot. Two-stage framing: `gs138650 -> tulu_math_uc_mix -> checkpoint-900` (stage-1, IFEval 0.253, seq 1024) -> `distill_cot_mix` at seq 8192, LR 2e-5 cosine, 48-144 steps / 1-3 epochs (GBS 768 @4N, 48 GPUs x LBS2 x GAS8). Pretok landed: 91k raw -> 36.9k packed @8192, 3.4G, GSM8K-test-decontam via 13-gram + normalized-exact (`decontam_traces.py`, `[decontam]` counts, recipe hash `cba5c64`, job 12483443 NPROC=1).

#### E7 pre-gates: cheap ablations A0-A4 (run before or alongside the main SFT)

| Ablation | Question | Setup | Gate |
|----------|----------|-------|------|
| **A0 — Artifact audit** | Is the landed dataset what we think it is? | **0 GPUs, 10 min, login node.** `load_from_disk("/tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192")` -> verify ~91k raw / ~36.9k packed, `input_ids`+`labels` present, `seq_lengths` p50/p90, 3 decoded samples check `<think>/</think>/<answer>/\boxed{}` envelope, `grep [decontam]` non-zero drops. **Blocks all downstream if failed.** | Rows ~91k, packed ~36.9k, envelope well-formed, OpenThoughts science/code rows not truncated, `[decontam]` drops non-zero (114k+220k corpora with GSM8K overlap should drop dozens). If p50 >> 4096 then 8192 justified; if p50 ~1024 then half-empty. |
| **A1 — Base-model probe** | How does `checkpoint-900-hf` behave on CoT before stage-2? | **0 GPUs + 1 XPU tile ~2 min.** `eval_cot_gsm8k.py` via `submit_cot_eval.sh` on `checkpoint-900-hf` itself (no new SFT). B2's stage-1 was never scored on GSM8K-CoT; only its stage-2 derivative (93) was. | If ~0.0 (expected — never saw `<think>/\boxed{}` envelope) then stage-2 does all work and broad mix must teach format+reasoning. If 0.05-0.10 with format 0.3-0.5 then stage-1 already has CoT-adjacent behavior (warmer start). Informs E1's expected delta. |
| **A2 — Mix composition** | Does OpenThoughts add anything beyond OpenR1? | **2N x2, GBS384, 1ep ~1h each.** A2a: OpenThoughts-114k-decontam only. A2b: OpenR1-Math-220k-decontam only. Same hparams as E1 (LR2e-5, seq8192, 2N). Requires one-off pre-tokenize of each component at 8192 or streaming `get_sft_dataset` run. | If A2a or A2b alone matches 50/50 mix on GSM8K-CoT, interleave unnecessary — simpler mix is deliverable. If both alone underperform mix, diversity hypothesis supported. **Run after E1 if E1 ambiguous (e.g. 0.15 — which half did the work?).** |
| **A3 — Decontamination sensitivity** | How many GSM8K-test near-duplicates survive 13-gram? | **0 GPUs, 30 min login node.** Re-run `decontam_traces.py` at `n=8` (more sensitive) on landed artifact vs production `n=13`. Compare drop counts. | Delta <0.5%: negligible contamination. 0.5-2%: note in report, consider re-staging at n=8. >2%: mix meaningfully contaminated — re-stage at n=8 before any "gain" claim. **Run alongside A0; no training.** |
| **A4 — Length truncation sensitivity** | Does 4096 (half attention cost) hurt vs 8192? | **1N pre-tokenize ~1-2h + 2N SFT ~1h.** Stage 4096 pre-tokenize of `distill_cot_mix` (NPROC=4) -> SFT at `max_length 4096, LBS4/GAS4, GBS384` (tokens/step matched to 8192@GBS768 by halving seq_len, doubling micro-batch). Compare to E1's 8192 on GSM8K-CoT (format+acc+gen_len). | If 4096 within 0.02 acc with format >=0.85, long tails not load-bearing — 4096 is efficient default (64x less attention cost O(L^2) on XPU without flash-attn). If format <0.70 or acc -0.05 vs E1, long reasoning tails matter — keep 8192. **Only run if E1 gen_len >500 (verbosity-sensitive).** |

All ablations use `VENV=rl-monarch-torch214-locality-20261001` LD-first, `assistant_only_loss=True`, `packing=True`, `bf16`, `fsdp full_shard`, `dataset_num_proc 8`, `HF_DATASETS_OFFLINE=1` for pre-tokenized path. Thread B §4 is the runnable spec with exact Python snippets.

#### E7 main queue: ranked E1-E7 with base / mix / hparams / gate (Thread B §5)

Ordered by information value per GPU-hour; early exits save downstream allocation. All evals 200-problem GSM8K-CoT `temp 0.0 max_tokens 768` fp32 vLLM on `frameworks/2026.1.0`, reporting `cot_accuracy / format_hit_rate / mean_gen_len / n_unclosed` vs B2 `0.205 / 0.985 / 282 / ~0`. Consolidation via `consolidate_sft_ckpt.sh` (`accelerate merge-weights`) then `submit_cot_eval.sh`.

| # | Experiment | Priority | Base | Mix | Hparams | Scale | Gate |
|---|------------|----------|------|-----|---------|-------|------|
| **E1** | **Broad-distill stage-2 on checkpoint-900, 1ep — THE GATE** | **P0 run first** | `checkpoint-900-hf` (re-consolidate from sharded `checkpoint-900/` if `-hf` missing; verify `config.json eos [1,107]` + `tokenizer.json` + `model.safetensors`) | `distill_cot_mix` = OpenThoughts-114k-decontam + OpenR1-Math-220k-decontam, pre-tokenized `/tegu/datasets/datasets/agpt2b-distill-cot-mix-len8192` (36.9k packed @8192, 91k raw, 3.4G) | LR 2e-5 cosine, 1ep ~48 steps @GBS768, `LBS2 GAS8 max_length 8192 bf16 fsdp full_shard dataset_num_proc 8` | 4N `select=5 NHOSTS_TRAIN=4 PPN=12 GBS768 workq 12h` actual ~1h; script `sft_distill_cot_4n.sh` with `PRETOK_DIR=/tegu/.../len8192 BASE_MODEL=<ckpt-900-hf> NUM_EPOCHS=1` | **Go:** acc >=0.15 with format >=0.80 gen_len <600 -> proceed to E2 (3ep). **Pivot:** acc 0.05-0.15 good format -> broad alone insufficient; E2b narrow polish or MATH/science eval before killing. **No-go:** acc <0.05 or format <0.50 -> mix dilutes like B3; stop, run A2 to isolate poison half. Artifact: `outputs/sft/agpt2b-distill-cot-pretok-4n-ep1/` |
| **E2** | **Same 3ep (B2's duration)** | **P1 if E1 Go/Pivot** | `checkpoint-900-hf` | `distill_cot_mix` | LR 2e-5 cosine **3ep ~144 steps** otherwise E1; `save_strategy epoch` per-epoch eval | 4N GBS768 `12h` actual ~1.5h | **Go:** epoch-3 acc > epoch-1 and >=0.18 -> longer broad training helps; queue E3 (LR sweep) or E6 (CPT-FWE base). **Flat:** epoch-3 ~ epoch-1 within 0.02 -> 1ep sufficient; do not extend. **No-go:** epoch-3 < epoch-1 or format degrades monotonically -> broad also collapses; cap future at 1ep. Artifact: `outputs/sft/agpt2b-distill-cot-pretok-4n-ep3/` 3 per-epoch ckpts |
| **E2b** | **Stage-3 narrow polish on E1/E2's best-epoch HF** | **P1 if E1 Pivot or E2 Flat** | **E1 or E2's best-epoch HF** (not `checkpoint-900` — the broad-distill model) | `gsm8k-r1cot` 7.4k ex `<think>/\boxed{}` inline tokenize (no pretok needed) | LR 2e-5 cosine (or 1e-5 if E1 format-sensitive) **3ep 93 steps @GBS384 `max_length 2048`** match B2 stage-2 proven seq_len; `save_strategy epoch` `dataset_num_proc 8` | 2N `select=3 NHOSTS_TRAIN=2 GBS384 workq 01:00:00` ~30m; script `agpt2b_b4a_gsm8k_finish_2n.sh` pattern on broad base | **Go:** acc > E1/E2 best and >=0.15 format >=0.95 -> 3-stage curriculum (general->broad->narrow) works; deliverable candidate. **No-go:** flat or format collapses -> narrow polish destabilizes even broad base (B4a mechanism, different base — broad manifold not specializable). Artifact: `outputs/sft/agpt2b-broad-distill-gsm8k-polish-2n-ep3/` |
| **E3** | **LR ablation 1e-5 cosine** | **P2 if E1/E2 format-degraded or E1 No-go** | `checkpoint-900-hf` | `distill_cot_mix` | LR **1e-5 cosine** 3ep ~144 steps otherwise E2; `save_strategy epoch` | 4N GBS768 `12h` | **Go:** acc >= E2 best or format >=0.90 gen_len <400 -> lower LR helps; promote 1e-5 for future. **No-go:** flat vs E2 -> LR not lever; data/structure is. Artifact: `outputs/sft/agpt2b-distill-cot-pretok-4n-ep3-lr1e5/` |
| **E4** | **Seq-len ablation 4096 vs 8192** | **P2 cheap alongside E3 or E1 gen_len>500** | `checkpoint-900-hf` | `distill_cot_mix` re-tokenized @`max_length 4096` (one-off 1N pretok NPROC=4 ~1-2h -> `/tegu/.../len4096`) | LR2e-5 cosine 1ep `LBS4 GAS4 max_length 4096 GBS384` tokens/step matched to 8192@GAS8 | 2N GBS384 `01:00:00` (+1N pretok) | **Go:** within 0.02 acc format >=0.85 -> long tails not load-bearing; 4096 efficient default. **No-go:** format <0.70 or acc -0.05 vs E1 -> long tails matter; keep 8192. Artifact: `/tegu/.../len4096/` + `outputs/sft/agpt2b-distill-cot-4096-2n-ep1/` |
| **E5** | **Single-stage control from gs138650 — null hypothesis** | **P3 one control not main line** | **`global_step138650` raw MDS pretrain `~/global_step138650`** NOT `checkpoint-900` | `distill_cot_mix` 36.9k-packed @8192 (tokenizer-compatible) | LR2e-5 cosine 1ep ~48 steps `LBS2 GAS8 max_length 8192` as E1 | 4N GBS768 `12h` ~1h | **Surprise Go:** >0.10 format >=0.80 -> within-trace diversity qualitatively different from B3's across-corpus; revisit structural hypothesis. **Expected No-go:** 0.03-0.07 -> confirms structure is lever; no further single-stage runs. Artifact: `outputs/sft/agpt2b-distill-cot-coldstart-4n-ep1/` |
| **E6** | **CPT-FWE frontier base (gated follow-on)** | **P4 gated on CPT 112k+ HF** | **CPT-FWE 112k+ HF** `step-112000` via `hf_batch_convert_cpt_fwe.sh flavor 2b-mds` — gate verified `config.json`+`tokenizer.json`+`model.safetensors` non-gibberish on 5-problem smoke | `distill_cot_mix` | LR **1e-5 cosine conservative** (shifted base; 2e-5 re-warm may repeat CPT-17000 0.015 shock) 1ep ~48 steps `max_length 8192` | 4N GBS768 `12h`; **precondition** narrow probe on CPT base (gsm8k-r1cot 3ep LR2e-5 2N ~30m) scored to calibrate format before running | **Go:** acc > E1 best or >0.15 if E1 No-go -> CPT+FWE+broad composes; schedule CPT-to-190k then re-train. **Pivot:** acc ~E1 within noise -> CPT did not lift CoT ceiling (may help MMLU — score both). **No-go:** format collapse or acc <0.05 -> shifted base needs general SFT first then broad as stage-2 (mirror B2 two-stage on new base). Artifact: `outputs/sft/agpt2b-cptfwe-distill-cot-4n-ep1-lr1e5/` |
| **E7** | **Scale sensitivity 2N vs 4N vs 8N (optional)** | **P5 only if GBS questioned** | `checkpoint-900-hf` | `distill_cot_mix` | LR2e-5 cosine 1ep `LBS2 GAS8` hold GBS per scale (2N GBS384 / 8N GBS1536 `select=9 NHOSTS_TRAIN=8`) | 2N+8N `workq` | Informational — if 8N degrades vs 4N future production stays 4N; if equivalent 8N is faster. Artifact: `outputs/sft/agpt2b-distill-cot-pretok-{2,8}n-ep1/` |

**Queue timeline (Thread B §5 ordering):**

```
Time 0h ───────────────────────────────── Time ~1h ───── Time ~3h ── Time ~6h
A0 (0 GPU audit) ─┐
A1 (1 tile probe)  ─┤ All three in parallel, <30 min
A3 (0 GPU n=8)    ─┘
                   v
E1 (4N 1ep ~1h) ──> eval (~2 min) ─┐
                                   ├─> branch on gate:
E2 (4N 3ep ~1.5h)  ◄── Go/Pivot ────┤──> E2b (2N polish ~30m)
E3 (4N 1e-5)       ◄── format-sens ─┤
E4 (1N pretok+2N)  ◄── gen_len>500 ─┤
E5 (4N cold-start) ◄── control ────┤──> (informational)
                                   └─> E6 (CPT-FWE gated on 112k+ HF)
```

**Minimum viable queue (tight allocation):** A0 + E1 + E2 = 2 SFT runs (48+144 steps) + 4 evals, ~3h compute, answers "does broad-distill on proven base beat narrow?" Everything else conditional on E1's result.

**Synthesis-queue mapping:** Thread B's E1->SYN E7-smoke, E2->SYN E7-full, E2b->SYN E10 stage-3 polish, E3->SYN E6 LR branch, E4->SYN new sec6 risk-12 ablation, E5->SYN new sec6 risk-11 control, E6->SYN rank-5/CPT-broad (E10 composition). Synthesis E1-E6 (HF ladder, commonsense, SFT probe, B2 repro, forgetting boundary, lower LR) remain the CPT/forgetting track; E7 now carries the full broad-distill track with two-stage `checkpoint-900 -> distill_cot_mix` framing per Thread B §3.

### E8 — RL/GRPO probe on B2 at group_size 8 (Thread E Config 7)

| Field | Value |
| **Tests** | Was GRPO's accuracy-flat (0.205->0.215 noise) a group_size/reward-density artifact? At 20% solve rate, group_size 4 => 41% all-wrong groups (zero advantage); group_size 8 => 17% dead |
| **Depends on** | E4 (B2 reproduces) — otherwise run on stored B2 |
| **Run** | 2N Monarch, VENV `rl-monarch-torch214-locality-20261001` LD-first, `generator.model_dtype=float32` (bf16 -> gibberish on vocab 256k), `max_completion_length 512-768` (default 64 truncates CoT). Base B2 ckpt-93, LR 5e-6 (conservative, per `POST-TRAINING-2B.md` next-step #2), group_size 8 (16 if signal), 50 + 100 step eval, componentized reward: think_format 0.1 + answer_extractable 0.1 + answer_correct dense (binary + numeric closeness), `beta` 0.0->0.01 guardrail, `no_drop_zero_std_groups` |
| **Decision** | If accuracy >0.25 (outside noise) at 50-100 steps with format >=0.99, RL-for-accuracy exists with proper hparams — scale with shaped reward per `ceiling-attack.md` (+168% via decomposition). If flat at 0.205-0.215, accept B2 as RL ceiling at 2B; keep RL as format-perfector only. If reward rises while accuracy falls (as in weak-base 0.118->0.217 / 0.16->0.105), reward hacking — revert weights |

### E9 — 150k gate eval (CPT-scored-to-150k + SFT probe on 150k HF)

| Field | Value |
|-------|-------|
| **Tests** | Does continuation from 143k to 150k hold the G1-G4 gate? |
| **Depends on** | CPT chain advancing (live `qsub -l select=4 -l walltime=12:00:00 -v NHOSTS_TRAIN=4 cpt_fineweb_edu_4n.sh` auto-resumes from CKPT_DIR) + E1-E3 ladder through 143k already in hand |
| **Run** | Identical to E1-E3 but for the single 150k HF: convert, run commonsense+MMLU (E2 protocol), run SFT probe if G1-G3 hold |
| **Decision** | Sec. 5 gate: all G1-G4 hold => continue to 190k (P3). Single -2pp wobble => continue, re-check at 190k. Commonsense <-3pp or SFT probe <=0.10 => stop CPT, fork broad-distill (E7) from last-good checkpoint, do not burn 40k more steps |

### E10 — 190k terminal gate + broad-distill on best CPT base (composition test)

| Field | Value |
|-------|-------|
| **Tests** | Final CPT value: does full-budget FWE base + broad data beat MDS base + broad data (E7 baseline) — isolating CPT gain from data-breadth gain |
| **Depends on** | E9 pass (gate held) + E7 MDS-broad number exists |
| **Run** | Same broad-distill config as E7-full but on 190k HF (or best P3 HF). Same LR, same mix, same harness — only base changes |
| **Decision** | If CPT-broad beats MDS-broad by >=0.06 at 2sigma, CPT thesis validated — promote CPT-broad as deliverable. If flat or worse, CPT did not add beyond broad data alone; deliver MDS-broad. Either way, the 190k ladder closes the "loss vs capability" question with a measured curve (Thread A Fig 5/Fig 6 equivalent for FWE) |

**Walltime summary:** E1 (1N 2h) and E4 (2N 2m) start immediately, in parallel. E2 (Aurora MMLU 48h workq or Sunspot commonsense 1N) overlaps with E5 (2N 5m) and E8 (2N 30m). Sequential worst case <2h of 2N time plus MMLU walltime; parallelized <30 min of active SFT time — all on `workq`, none blocks the 4N CPT chain.

---

## 4. The Scoring Ladder: Which Checkpoints to HF-Convert and Score

Reuse E1's ladder; this section is the single reference for "what gets converted when."

| Phase | Steps to convert | Count | Why this density |
|-------|-----------------|-------|------------------|
| **P0 — Shock horizon** | `1000, 5000, 10000` | 3 | Dolmino shock hit by 1000; constant LR 2e-6 should not shock — verify the hypothesis, not the loss |
| **P1 — Early plateau** | `25000, 50000, 75000` | 3 | 74k/74.2k already HF 3.7G 9/9; fill gap to 50k where broad-distill was parked; 50k was `POST-TRAINING-PLAN.md` phase gate |
| **P2 — Frontier** | `100000, 111200, 125000, 143400, 150000` | 5 | 111.2k documented (`POST-TRAINING-2B.md` `R step-111200 23G 556 dirs`); 143.4k snapshot; 150k gate |
| **P3 — Terminal (if continuing)** | `160000, 175000, 190734` | 3 | Full 100B budget; 190734 exact terminal (100B / 524288) |

Total 14 HF converts. Only **3 need SFT probes**: best P2 checkpoint, 150k gate, and (if degrading) last-good earlier step — per Thread A §4.1 cost note, ~8-12 HF + 3 SFT probes + 2 eval sweeps is the full decision cost.

**Converter contract (critical):**

- `flavor=2b-mds` — wrong flavor = silent gibberish. `torchtitan/experiments/ezpz/eval/convert_to_hf.py` + `hf_batch_convert_cpt_fwe.sh:58`.
- Patch config/tokenizer omission (as at 74k) — verify 9/9 files post-convert before scoring.
- `PBS filesystems=tegu:home` (PBS lacks `/eagle` on Sunspot — `POST-TRAINING-2B.md:13` sync note).
- Tegu output root: `/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-$s/hf` (avoid `/home` fill — 1.2P Lustre 17% full, 1P free, but `/home` holds 15T live CPT).
- Walltime 12h, not 2h — 99e918b342 fix: 2h churn ate ~4k steps/hop vs ~24k per 12h hop (Thread A §1.2).

**What runs on every HF (no SFT needed):** HellaSwag, ARC-Easy, PIQA, ARC-Challenge, MMLU MCF + CF. **What runs post-SFT only:** GSM8K-CoT generative (vLLM fp32, 200 problems, 98s/tile — Thread C §2.1). Raw CPT scores 0/200 expected (no `<think>` contract — Thread C §3.1), not a failure.

---

## 5. Decision Gates: Continue vs Stop vs Fork

### Gate definitions

Apply *before* applying thresholds (Sec. 3 E2/E3 decision rows). Gate is a conjunction — all must hold to continue. Thresholds from Thread A §4.4 / Thread C §5.2, derived from dolmino damage (HS -7.4pp, ARC-Easy -10.4pp at step-5960 job 8647850), MMLU chance band, and GSM8K-CoT power at N=200.

| Gate | Metric | Base | Threshold to **CONTINUE** to 190k | SE / rationale |
|------|--------|------|-----------------------------------|----------------|
| **G1 — No commonsense regression** | HellaSwag `acc_norm` 0-shot | 0.560 (2B-256 plateau) | **>= 0.540** (-2pp) | SE ~0.004; -2pp = 5 sigma; dolmino-100 fell to 0.486 |
| **G1b** | ARC-Easy `acc` 0-shot | 0.651 | **>= 0.621** (-3pp) | SE ~0.006; dolmino 50/50 held 0.610, so -3pp is strict |
| **G1c** | PIQA `acc_norm` 0-shot | 0.733 | **>= 0.713** (-2pp) | SE ~0.005; dolmino only -4.6pp here — weak guard but cheap |
| **G2 — No catastrophic forgetting** | ARC-Challenge `acc_norm` 25-shot | ~0.31 | **>= 0.290** (-2pp) | Peer table in `evals/` |
| **G3 — MMLU must not regress** | MMLU `max(CF,MCF)` 5-shot | 0.241-0.266 chance band | **>= 0.240** (no regression) | Band 0.2295-0.2689 (best constant-D), SE 0.0036. Clearing band requires prior collapse (Thread C §1.5#7) — not a continue-gate until `max(CF,MCF) >0.2689` + `R2->0` |
| **G4 — SFT probe lifts ceiling** | GSM8K-CoT `acc` @200 post-SFT (B2 recipe) | B2 = **0.205** (41/200, 95% CI 0.149-0.260) | **>= 0.265** (B2 +0.06 at 2sigma; Thread C §2.3: 0.06 gap ~2.0 sigma on 200) | SE ~0.028 at 0.20. 0.01 RL move 0.205->0.215 is 0.26 sigma — correctly noise |
| **G4b — Format held** | GSM8K-CoT `format_hit_rate` | B2 = 0.985 | **>= 0.90** | B2v2 fell 0.86/0.91 while degrading; B3 0.86 masking 0.05 |
| **G4c — No verbosity blow-up** | `mean_gen_len` / `n_unclosed` | B2 = 282 / ~0 | **<= 550** / **<= 10** | B2v2 416/378 with 22/15 unclosed; B4a 1378/133 flagged envelope loss |

### Near-miss interpretation (Thread A §4.4 logic)

- **G1-G2 fail by >2pp but G4 passes** (commonsense dips, CoT ceiling lifts): **continue to 190k.** Commonsense dip is the expected FWE HellaSwag trade (ablation: FWE 37.5 vs DCLM 35.5 MMLU, DCLM 62.3 vs 60.1 HS). Re-check at 190k; if monotonic bleed continues (dolmino pattern), ratio is wrong — revisit 75/25 arm (`config_registry.py:1024-1043`).

- **G1-G2 pass but G4 fails** (no CoT lift, e.g. 0.08-0.12 like B2v2/B3/B4): **stop CPT**, fork SFT from best intermediate checkpoint (max over G1-G4, not necessarily 150k), pivot to broad-distill on MDS base (E7). CPT did not move ceiling — do not pay 40k steps hoping it will. This is `POST-TRAINING-2B.md` phase-3 pivot.

- **Everything fails** (commonsense tanks + CoT collapses): **stop CPT, revert to MDS base.** FWE kind-shift wrong at this scale/mix or constant LR 2e-6 needs re-tuning. Remaining 40k steps would be burning compute on a known-bad distribution.

- **G1-G3 pass but SFT probe not yet run** (E3 pending): **continue CPT** — do not make a stop decision without the ceiling test. CPT-direct MMLU/ARC scores are thermometers; the SFT probe is the thermostat.

### SFT-level gates (Thread C §5.2 rows 7-10; Thread E §5.5)

| Condition | Interpretation | Next action |
|-----------|---------------|-------------|
| New SFT GSM8K-CoT >=0.265 at 2sigma + format >=0.90 + unclosed <=10 vs B2 0.205 | **Promotion candidate** — replaces B2 as deliverable | Ship new checkpoint; re-score MMLU with band+prior before external claim (Thread C §1.5#7) |
| New SFT format <0.80 or unclosed >30 | **Envelope collapse** — reject without re-run | Do not down-weight format to "fix" (RL hacking: format 0.2->0.05 drove 95% no-`<think>`) |
| New SFT 0.15-0.20 at noise around B2 | **No lift** — distribution or base not additive | Do not retry same mix at different weights (B4b 0.05->0.065 with reweight proves mix tuning ~+0.015 max) |
| Broad-distill on MDS base beats 0.15 where narrow collapsed | **Breadth extends budget** | Promote broad mix to 4N full run at winning LR (E7) |
| RL on B2 stays 0.205-0.215 flat at group_size 8 | **~0.2 is base ceiling for RL** | RL for format only; accuracy lever is SFT breadth + CPT base (Thread E §3.6) |

### RL-specific gates

- **RL format:** must hold >=0.99 for 100 steps (as B2 GRPO did at group_size 4). If format bleeds, beta 0.0->0.01 guardrail before any other tuning.
- **RL reward sanity:** if reward rises while GSM8K-CoT falls (weak-base hacking pattern 0.118->0.217 / 0.16->0.105), **stop** — reward function is mis-specified. Gate reward drove to 0.000 when format removed, revealing 95% no-`<think>` — Thread D §2.2 weak-base analysis.
- **RL accuracy:** lift must be >=0.05 at 2sigma on 200 to claim, not 0.01 noise (Thread C §2.3: 0.01 at 0.20 is 0.26 sigma at 200, 0.56 sigma even at 1000).

---

Every prohibition below is earned by a measured failure on this branch. Violating any one recreates a known collapse.

| # | Prohibition | Why (failure that earned it) | What to do instead |
|---|-------------|------------------------------|--------------------|
| 1 | **Do NOT attempt another single-stage SFT** from `global_step138650` mixing instruct + long-CoT + gsm8k in one pass | B3 0.05 (dilution: gsm8k at 0.15 vs 0.25 OpenR1 long traces teaching competing styles), B4a 0.02 (verbose base poisoning: cannot un-teach 601->1378 gen_len), B4b 0.065 (reweight 0.15->0.40 + length filter restored format 0.86 but acc only +0.015 — structure, not mix, is the lever) | Any new data goes through the proven two-stage structure: stage-1 tulu_math -> ckpt-900 equivalent, then narrow/broad stage-2. Thread D §§2-3 is the taxonomy |
| 2 | **Do NOT add more epochs on the same narrow data** (gsm8k-r1cot 7.4k) | B2v2: 3ep 0.205 -> 6ep 0.09/0.105 at healthy loss 0.8069 acc 0.8198 — training metrics hid halved eval. Ckpt-8672: 900 steps IFEval 0.253 intact -> 8672 hellaswag 0.593->0.273 at loss 0.357 acc 0.902 | Cap narrow SFT at O(100) steps at 2e-5 or drop LR per E5-E6 curve. Treat early stopping as load-bearing structure, not accident |
| 3 | **Do NOT run narrow SFT to >1000 steps at LR 2e-5** (cosine or constant) expecting cosine decay to save you | Full-mix at LR 2e-5 cosine over 8672 steps kept LR >1e-5 for ~4350 steps (LR(4350)=1e-5 = half-peak at S/2), collapsing hellaswag 0.592->0.273 between 1500-4500. Cosine's shape is flat on top — at 93 steps (B2) it never decayed at all | E5-E6 sweep before any full-mix SFT. For broad distill requiring O(1000) steps, use 1e-5 constant (Thread E §1.3 table) with `--max_steps` hard gate per `sft_distill_cot_4n_verification.md` §4 |
| 4 | **Do NOT read training loss or validation loss as a training signal for capability** | Dolmino: loss -0.31 nats while benchmarks bled -7 to -10pp. FWE: 12.53 flat val vs 1.86 train = miscalibrated validator (likely missing dp_world_size divisor, Thread A §1.3). CPT 17k narrow SFT loss looked healthy before 0.015 CoT | Gate only on downstream eval (G1-G4). Training loss may descend while eval collapses — this is the recurring signature of narrow-data overfitting (B2v2 + ckpt-8672) |

| 5 | **Do NOT gate CPT on MMLU MCF alone** (or compare to SmolLM3's 44.1) | Agpt-2B MCF 0.241-0.266 sits inside constant-letter band 0.229-0.269 — every 2B and 20B config does, plus flagship 2B at 4.674T scoring *lowest* (0.241). Llama 3.2-1B 0.312 controls harness clean. SmolLM3 44.1 is MMLU-CF (contamination-free variant, different task). Below ~400B, MCF is random by construction (SmolLM2 MCF 29.62 across 0-6T; OLMES: "MCF random at 400B, CF has signal") | Report `max(CF,MCF)` with band 0.2295-0.2689 + letter prior `p(A,B,C,D)` + residual dispersion per Thread C §1.5#1-7. Real MMLU gate is post-SFT max-plus-prior, not CPT MCF. Job 12474382 CF+MCF on 30B step-2000 at 3.93B tok: MCF 0.2281 / CF 0.2386 delta +0.0105 SE 0.0256 — not significant at 570 Q, too early and too small to test the format thesis |
| 6 | **Do NOT interpret 0/200 GSM8K-CoT on a raw CPT checkpoint as a failure** | CPT 17k/73k/74k/111k all 0/200 degenerate loops `n_unclosed=200` — expected, not regression. CPT never saw `<think>/<answer>\boxed{}` (taught in SFT stage 1->900->93). Thread C §3.1 category error: gating 2B on CoT pre-SFT is like gating 2B-512 promotion on MMLU MCF | CoT gates only on SFT `final/` via `submit_cot_eval.sh` (vLLM fp32, 200 problems). CPT questions are answered by SFT probe (E3), not direct CoT |
| 7 | **Do NOT use sequence length 1024 for gsm8k-r1cot stage-2, or 8192 for 300-token CoT by default** | B2v2 fix: 1024 silently truncates `<answer>` tail via packing boundary despite 300-token avg CoT — requires 2048. Conversely, 8192 for narrow gsm8k alone is 64x attention cost O(L^2) on XPU (no flash-attn) for 96% padding waste | 2048 for pure gsm8k-r1cot (Thread E §4.1), 8192 only for distill_cot_mix where OpenThoughts traces are 2k-10k (Thread E §§4.2-4.3). Always offline pre-tokenize via `_pretokenize_distill_cot_mix_1n.sh` before long-trace SFT (runtime tokenize of 93M rows ~8.5h + HF cache race) |
| 8 | **Do NOT rebuild without offline pre-tokenization for any mix containing long traces** | `_pretokenize_distill_cot_mix_1n.sh` at NPROC=32/16/8 segfaulted SIGBUS/SIGSEGV; required NPROC=1 + VENV `rl-monarch-torch214-locality-20261001` LD-first + `--pretokenized_dataset` gate on `dataset_info.json` + `[decontam]` counts | Verify pretok gate before every SFT training job — `sft_distill_cot_4n_verification.md` §5 checklist |
| 9 | **Do NOT promote a checkpoint by accuracy alone — check format, gen_len, unclosed** | B4a: acc 0.02 with format 0.26 133/200 unclosed (envelope collapsed — different failure than reasoning). B3/B4b: same 0.05-0.065 band but gen_len 601 vs 611 indistinguishable — verbosity is symptom not cause. RL weak-base: reward 0.118->0.217 while acc 0.16->0.105 with 95% no `<think>` | Guardrail trio (Thread D §2.3 + Thread C §2.2): format >=0.90 + unclosed <=10 + gen_len within [220,550] required alongside acc. Added after B3, load-bearing for diagnosis |
| 10 | **Do NOT use MMLU as a CPT health gate; do NOT run IFEval/BBH/GPQA/HumanEval on base checkpoints** | Tier C deferral in `eval-landscape-2026-07.md`: base-2B scores ~0 on instruct-tuned tasks -- zero gradient, wasted walltime. IFEval on base is 0.179 vs ckpt-900 0.253 only after SFT | Health gate = commonsense thermometer (HS/ARC-Easy/PIQA/ARC-C -- cheap, 0-shot loglikelihood, Dolmino damage visible here first). MMLU informational until max(CF,MCF) clears band |
| 11 | **Do NOT assume `distill_cot_mix` is OpenThoughts+gsm8k-r1cot — it is OpenThoughts+OpenR1 (Thread B §2.1 discrepancy)** | Task prompt described "50% OpenThoughts-114k-decontam + 50% gsm8k-r1cot" but `datasets_sft.py:_build_distill_cot_mix` and `_pretokenize_distill_cot_mix_1n.sh` header register 0.50 OpenThoughts-114k-decontam + 0.50 OpenR1-Math-220k-decontam — both frontier-distilled (DeepSeek-R1), no gsm8k-r1cot. 91k-row artifact matches the OpenThoughts+OpenR1 definition (both ~100k+ raw; 50/50 with 7.4k gsm8k would be much smaller or heavily upsampled). `sft_distill_cot_4n.sh` comment block still references gsm8k-r1cot incorrectly | Verify mix provenance via `datasets_sft.py:_build_distill_cot_mix` + `[decontam]` logs + recipe hash `cba5c64`; landed 36.9k-packed artifact contains no GSM8K problems — if gsm8k-r1cot grounding is desired, stage a separate pre-tokenize with that mix-spec (Thread B §2.1, E5 queue) |
| 12 | **Do NOT run broad-distill SFT on a shifted CPT base without calibration, and do NOT train at 8192 without OOM/packing checks (Thread B §2.3)** | **Distribution shift (Thread B §2.3.1):** CPT-17000 narrow SFT probe 0.015 with format collapse — worse than B3 0.05 on MDS base; OOD science/code vocab + trace structure from frontier distillation shifts the olmo-mix-1124 manifold; shifted-base + narrow SFT is strictly harder. **Length/packing (Thread B §2.3.2):** 8192 is longest TRL/XPU sequence attempted; O(L^2) attention without flash-attn = 64x cost of 1024; batch 2@8192 unproven (B3 survived batch 1@8192; tulu reverted 2048->1024 for OOM); packing at 8192 blends unrelated problems via cross-problem attention (no block-causal masking in TRL) — unmeasured. **Quality (Thread B §2.3.3):** OpenThoughts correctness weaker than OpenR1 — no `correctness_math_verify`, wrong `\boxed{}` can survive regex; dropped code-gen rows (ending in code block not `\boxed{}`) reduce breadth to code-adjacent scalar answers only | A0 artifact audit (0 GPUs, envelope + `[decontam]` + p50 check) before any SFT (Thread B §4 A0). For length: fallback batch 1 + GAS16 (same GBS, half activation); run A4 seq-len 4096 ablation before scaling; estimate 144 steps @30s = 72m well within 12h so OOM not walltime is risk. For quality: audit rows for bogus `\boxed{}` and re-score on MATH/science (broad gain invisible on GSM8K narrow arithmetic — Thread B §2.3.1). CPT-FWE SFT gated on narrow-probe calibration (E6 precondition: score narrow SFT on 112k+ HF first, use LR 1e-5 conservative not 2e-5 re-warm) |
| 13 | **Do NOT claim broad-distill gain without decontamination + broad-eval gates (Thread B §§2.3.1, 2.3.4, 4 A3)** | 13-gram + normalized-exact (`GSM8KDecontaminator`, `decontam_traces.py`) misses reworded GSM8K-test with <13 shared tokens; near-duplicate contamination could fabricate "gain" via leakage; `[decontam]` drops near-zero on 114k+220k corpora would be suspicious. Broad reasoning gain may be invisible on GSM8K (narrow arithmetic execution) and require MATH/science QA to surface — Thread B §2.3.1: "a broad reasoning gain could be invisible on GSM8K" | Materialized-mix recipe hash captures decontam stage so cache hit is guaranteed per that detector, but re-run at n=8 (Thread B A3: CPU-only, 30 min login node; delta <0.5% negligible, 0.5-2% note, >2% re-stage at n=8 before claim). Include broad eval (MATH, science QA) alongside GSM8K-CoT per Thread B §2.3.1 mitigation |

---

## 7. Key Concepts Glossary

Terms recur across the four threads with subtly different emphases. This table pins them for the synthesis.

| Term | Meaning in this synthesis | Canonical source |
|------|---------------------------|------------------|
| **CPT-FWE 4N** | Continued pre-training on 60% FineWeb-Edu + 40% Open-Web-Math via Grain streaming, forked from `gs138650/step-0` (MDS 2B), 4 nodes x 8 XPU, 190.7k-step / 100B-token budget | Thread A §1.1; `cpt_fineweb_edu_4n.sh`; `config_registry.py:1046-1057` |
| **B2 = checkpoint-93** | Two-stage SFT deliverable: `gs138650 -> tulu_math_uc_mix_full ckpt-900 -> gsm8k-r1cot 93 steps` = **0.205 GSM8K-CoT** | Threads D §1, E §1.1; `POST-TRAINING-2B.md` B2 § |
| **Ceiling ~0.20** | Empirical max GSM8K-CoT on `gs138650` base: B2 0.205 vs 0.02-0.105 for every alternative; GRPO on B2 flat 0.205->0.215 at noise | Thread D §4; `POST-TRAINING-2B.md` Findings |
| **Kind-shift vs degree-tweak** | FWE 60% educational content = kind change (SmolLM2 stage-1 recipe); dolmino 89% DCLM with 5.6pp redistributed = degree tweak | Thread A §§3.1-3.2; `fineweb-edu-60-40-kind-shift.md` |
| **MCF vs CF vs max(CF,MCF)** | MCF = letter `A-D` scoring (`mmlu` task); CF = answer-text cloze (`mmlu_continuation`); max per OLMES -- monotonic in either component | Thread C §§1.3-1.5; arXiv:2406.08446 |
| **Constant-letter band** | 0.2295-0.2689 (always-A through always-D on 14k MMLU test) -- the true chance band for MCF, not 0.25 | Thread C §1.1 |
| **Format / gen_len / unclosed** | Guardrail trio: `format_hit_rate` (closed `\boxed{}`), `mean_gen_len`, `n_unclosed` (finish_reason length) -- added after B3 | Thread C §2.2; Thread D §2.3 |
| **SFT probe** | Re-running the B2 two-stage recipe verbatim on a new CPT HF to measure whether CPT lifted the CoT ceiling | Thread A §4.2; E3/E9 |
| **Decontamination gate** | `dataset_info.json` + `[decontam]` count check on `/tegu/datasets/.../len8192` before any broad-distill SFT | `POST-TRAINING-2B.md` 2026-11-12; `sft_distill_cot_4n_verification.md` §5 |
| **G1-G4** | Commonsense (HS/ARC/PIQA/ARC-C) + MMLU + CoT-accuracy + format guard -- the 150k continue/stop/fork gate | Sec. 5 above; Thread A §4.4 |

---

## 8. References: Where Every Key Number Comes From

| Number | Value | Source |
| CPT budget 190,734 steps | 100B / 524,288 (GBS=64 seq=8192, `LBS=2 PPN=8 4N`) | `cpt_fineweb_edu_4n.sh:72-78`, `config_registry.py:1046-1057` (Thread A §1.1) |
| CPT snapshot 143.4k/75.2% | assignment snapshot; prior doc 111.2k/58.3% | `POST-TRAINING-2B.md:10-14` + prompt |
| Train loss 1.856 at 10k, MFU 18.16% | jobs 12483048/75/245 ladder 4091->10143 | `journal.md:96-104`, `POST-TRAINING-2B.md:57` (Thread A §1.2) |
| Val loss 12.53 flat 110700-111200 | job 12483856 step-111200 | `POST-TRAINING-2B.md:10` (Thread A §1.3) |
| `log(vocab)=ln(256000)=12.45` uniform expectation | tokenizer vocab 256k, md5 b60c6af2 | `mmlu-letter-prior-at-chance.md` |
| Mixture 40% OWM + 60% FWE (140 parquets) | `agpt_2b_mds_mix_owm_edu_6040` via Grain Mix | `config_registry.py:970-1021`, `cpt_fineweb_edu_4n.sh:18-19` (Thread A §1.1) |
| LR 2e-6 constant, warmup 20 | CPT fork base | `config_registry.py:919-928` (Thread A §1.1) |
| Dolmino loss 2.80->2.49/2.60, HS 0.560->0.486, AE 0.651->0.547 | 300B pilot steps 5960, job 8647850 | `cpt/README.md:60-129`, `cpt/figures/cpt_loss.svg` (Thread A §2.1) |
| FWE 33.6% MMLU at 38B vs 300B next-best (8x) | 1.7B model, 350B ablation | arXiv:2406.17557 via `fineweb-edu-60-40-kind-shift.md:58-64` (Thread A §3.2) |
| OLMo-2 +17.4 MMLU at 1B (26.9->44.3) | same olmo-mix-1124 base, Dolmino mid-training | arXiv:2501.00656 Table 9 via `cpt/README.md:104-106` (Threads A,C) |
| FWE ablation 37.5 vs DCLM 35.5 MMLU; DCLM 62.3 vs 60.1 HS | matched token count, same scale | arXiv:2406.17557 via `fineweb-edu-60-40-kind-shift.md` (Thread A §3.2) |
| B2 0.205 / 0.985 / 282 | ckpt-900 -> gsm8k-r1cot 93 steps 3ep LR2e-5 | `POST-TRAINING-2B.md` B2 § + `rl/plans/cot.md` (Threads D,E) |
| ckpt-900 IFEval 0.253, HS 0.592, AE 0.640 | 8N full-mix at 900 steps, ~5.7B tok | `tulu_math_uc_mix_full/evals/README.md` (Threads D,E) |
| ckpt-8672 HS 0.273 AE 0.298 loss 0.357 acc 0.902 | same run step 8672, ~54B tok | `tulu_math_uc_mix_full/evals/README.md` (Threads D,E) |
| B3 0.05 / 0.86 / 601 | single-stage tulu+OpenR1 0.25 seq8192 | `b3-instruct-cot-mix/design.md`, `b4/README.md` (Thread D) |
| B4a 0.02 / 0.26 / 1378 unclosed 133 | gsm8k finish on B3-3605 | `b4-finish-and-reweight/README.md` (Thread D) |
| B4b 0.065 / 0.86 / 611 unclosed 27 | reweight gsm8k 0.15->0.40 + length-filter seq4096 | `b4/README.md` (Threads D,E) |
| B2v2 0.09/0.105 format 0.86/0.91 gen_len 416/378 | ckpt-900 -> gsm8k-r1cot 6ep/12ep GBS384 jobs 12482572/73 W&B xakno7kg/ur6pqqhp | `POST-TRAINING-2B.md` B2v2 §, `fineweb-edu-60-40-kind-shift.md` §3 (Threads C,D,E) |
| Narrow SFT-on-CPT-17k 0.015 | gsm8k-r1cot on CPT-17k HF LR2e-5 GBS384 | `POST-TRAINING-2B.md` 2026-11-12 (Threads C,D) |
| Raw CPT 0/200 degenerate 0.0 fmt 200 unclosed | 17k/73k/74k/111k | `POST-TRAINING-2B.md` 2026-11-12/14 (Thread C) |
| MMLU 0.241-0.266 at chance, constant-D 0.2689 | 12 configs spanning 4.6T tok, 2 scales | `mmlu-letter-prior-at-chance.md:48-57` (Threads A,C) |
| Letter prior p=(0.37,0.32,0.20,0.11) r=+0.64-0.76 to keyfreq | per-subject acc ~ sum pL keyfreqL | `mmlu-letter-prior-at-chance.md:142-158` (Thread C) |
| Llama 3.2-1B 0.312 3.1-8B 0.653 controls | same lm-eval XPU 5-shot MCF | job 8736838, `mmlu-letter-prior-at-chance.md` (Thread C) |
| MCF 0.2281 CF 0.2386 delta 0.0105 SE 0.0256 (limit10) | step-2000 3.93B tok 570Q both formats | job 12474382 (Thread C §1.3) |
| SmolLM2 MCF 29.62 0-6T; OLMES "MCF random at 400B, CF has signal" | literature | arXiv:2502.02737 Fig6; arXiv:2406.08446 (Thread C) |
| 200-problem SE ±0.055 at 0.20; 0.10 delta detectable 2.5sigma | binomial SE, delta SE sqrt(SEA2+SEB2) | Thread C §2.3 power table |
| LR-finder 5B blow-up/10: 3.49e-05@32 / 2.48e-04@128 / 5.36e-04@512 | Sirius H100 Sirius ladder Nov 2026 | `lr-finder/agpt/5b/README.md` (Thread E §1.3) |
| Pretok 91k rows -> 36.9k seqs @8192 3.4G decontam-verified | NPROC=1 VENV rl-monarch-... | `_pretokenize_distill_cot_mix_1n.sh` job 12483443 (Thread E §4.3) |
| GRPO B2 0.205->0.215 noise format 1.000; weak-base 0.16->0.105 hacking | group_size 4 41% all-wrong groups | `rl/plans/cot.md` + `POST-TRAINING-2B.md` RL § (Threads D,E) |
| Ceiling-attack alphabet_sort +168% 0.24->0.667 cross-scored | reward decomposition format/completeness/order | `rl/grpo/ceiling-attack.md` (Thread E §3.2) |
| HF converts 74k/74.2k 3.7G 9/9 verified; Tegu root for new HFs |避免 /home fill | `POST-TRAINING-2B.md` 2026-11-14 + `hf_batch_convert_cpt_fwe.sh:58` (Thread A §5.1) |

---

*Thread B (broad-distill analysis) pending — expected to add data-composition audit, decontamination ledger, length-distribution gate, and refined E7/E10 decision thresholds. This synthesis will be updated when Thread B lands.*
