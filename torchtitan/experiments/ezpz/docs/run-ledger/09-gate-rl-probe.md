# 09 — Gate & RL Probe Ledger (150k Decision Gate + GRPO Probe + Ranked Next-Course Matrix)

> **Branch:** `ezpz-cpt111k-eval-sft` @ `a6135a79` (2026-11-16).
> **Scope:** The 150k continue/stop/fork gate (G1-G4), the ranked next-course matrix (ranks 1-7), the E9 150k gate eval chain, the RL/GRPO probe spec (E8), and the decision tree with exact `qsub` commands that feed the gate.
> **Every threshold, cost, and outcome cites its source file + section or job ID.** Inferences labeled `[INFERENCE]`. No gate value is invented — all thresholds are verbatim from SYNTHESIS §5 / Thread E §3.
> **Companion ledgers:** `01-cpt-frontier.md` (chain + budget), `02-hf-conversion.md` (batch HF converts), `03-eval-harness.md` (CoT/MMLU/commonsense harness), `04-sft-broad-distill.md` (broad-distill queue), `05-infra-rl-storage.md` (VENV + Monarch infra). This file is `09`, the gate that consumes their outputs.

---

## 1. G1-G4 gate — metric | base | continue threshold | SE/rationale + near-miss logic

### 1.1 Gate definitions — verbatim from SYNTHESIS §5 `SYNTHESIS.md:225-239`

The gate is a **conjunction**: all rows must hold to **continue to 190k**. Thresholds derived from the dolmino damage record (HS -7.4pp, ARC-Easy -10.4pp at step-5960 job `8647850` per `cpt/README.md:61-98`), the MMLU chance band (`mmlu-letter-prior-at-chance.md:27-38`), and GSM8K-CoT power at N=200 (`thread-c-eval-harness.md:263-287`, `SYNTHESIS.md:236`).

| Gate | Metric (task, shot, key) | Base value | Threshold to **CONTINUE** to 190k | SE / rationale | Source |
|------|--------------------------|------------|--------------------------------------|----------------|--------|
| **G1** | HellaSwag `acc_norm` 0-shot | **0.560** (2B-256 plateau) | **>= 0.540** (-2pp) | SE ~0.004; -2pp = 5 sigma; dolmino-100 fell to **0.486** | `SYNTHESIS.md:231` |
| **G1b** | ARC-Easy `acc` 0-shot | **0.651** | **>= 0.621** (-3pp) | SE ~0.006; dolmino 50/50 held **0.610**, so -3pp is strict | `SYNTHESIS.md:232` |
| **G1c** | PIQA `acc_norm` 0-shot | **0.733** | **>= 0.713** (-2pp) | SE ~0.005; dolmino only -4.6pp here — weak guard but cheap | `SYNTHESIS.md:233` |
| **G2** | ARC-Challenge `acc_norm` 25-shot | **~0.31** (peer table) | **>= 0.290** (-2pp) | Catastrophic-forgetting detector; peer table in `torchtitan/experiments/ezpz/docs/evals/` | `SYNTHESIS.md:234` |
| **G3** | MMLU `max(CF,MCF)` 5-shot | **0.241-0.266** chance band | **>= 0.240** (no regression) | Band **0.2295-0.2689** (best constant-D), SE 0.0036. Clearing band requires prior collapse — not a continue-gate until `max(CF,MCF) >0.2689` + `R2->0` per `SYNTHESIS.md:235` + `thread-c-eval-harness.md:68-82` | `SYNTHESIS.md:235` |
| **G4** | GSM8K-CoT `acc` @200 post-SFT (B2 recipe) | **B2 = 0.205** (41/200, 95% CI 0.149-0.260) | **>= 0.265** (B2 +0.06 at 2sigma) | SE ~0.028 at 0.20; 0.06 gap ~2.0 sigma on 200 per `thread-c-eval-harness.md:263-287`; 0.01 RL move 0.205->0.215 is 0.26 sigma — correctly noise | `SYNTHESIS.md:236` |
| **G4b** | GSM8K-CoT `format_hit_rate` | **B2 = 0.985** | **>= 0.90** | B2v2 fell 0.86/0.91 while degrading (`12482572/73`); B3 0.86 masking 0.05 | `SYNTHESIS.md:237` |
| **G4c** | `mean_gen_len` / `n_unclosed` | **B2 = 282 / ~0** | **<= 550** / **<= 10** | B2v2 416/378 with 22/15 unclosed; B4a 1378/133 flagged envelope loss per `SYNTHESIS.md:239` + `03-eval-harness.md:28-41` | `SYNTHESIS.md:238` |

**How to read the base column:** HellaSwag 0.560, ARC-Easy 0.651, PIQA 0.733 are the 2B-256 plateau values documented in `cpt/README.md:61-98` and `SYNTHESIS.md:229-234`. ARC-Challenge ~0.31 is the peer-table baseline in `torchtitan/experiments/ezpz/docs/evals/`. MMLU base is not a single number but the measured agpt-2B MCF band 0.241-0.266 (`thread-c-eval-harness.md:68-82`); the gate threshold `>=0.240` means "no regression below chance floor." G4 base is B2 = checkpoint-93 = 0.205 (41/200) from `POST-TRAINING-2B.md:45` + `SYNTHESIS.md:17` + `03-eval-harness.md:33`.

**SFT-level sub-gates consumed by G4** — per `SYNTHESIS.md:250-258` (Thread C §5.2 rows 7-10; Thread E §5.5):

| Condition | Interpretation | Next action | Source |
|-----------|---------------|-------------|--------|
| New SFT GSM8K-CoT >=0.265 at 2sigma + format >=0.90 + unclosed <=10 vs B2 0.205 | **Promotion candidate** — replaces B2 as deliverable | Ship new checkpoint; re-score MMLU with band+prior before external claim | `SYNTHESIS.md:254` |
| New SFT format <0.80 or unclosed >30 | **Envelope collapse** — reject without re-run | Do not down-weight format to "fix" (RL hacking: format 0.2->0.05 drove 95% no-`<think>`) | `SYNTHESIS.md:255` |
| New SFT 0.15-0.20 at noise around B2 | **No lift** — distribution or base not additive | Do not retry same mix at different weights (B4b 0.05->0.065 reweight proves mix tuning ~+0.015 max) | `SYNTHESIS.md:256` |
| Broad-distill on MDS base beats 0.15 where narrow collapsed | **Breadth extends budget** | Promote broad mix to 4N full run at winning LR (E7) | `SYNTHESIS.md:257` |
| RL on B2 stays 0.205-0.215 flat at group_size 8 | **~0.2 is base ceiling for RL** | RL for format only; accuracy lever is SFT breadth + CPT base | `SYNTHESIS.md:258` |

### 1.2 Near-miss interpretation — all 4 cases verbatim from `SYNTHESIS.md:240-248`

Thread A §4.4 logic, reproduced exactly:

| # | Condition | Verdict | Rationale | Source |
|---|-----------|---------|-----------|--------|
| 1 | **G1-G2 fail by >2pp but G4 passes** (commonsense dips, CoT ceiling lifts) | **Continue to 190k** | Commonsense dip is the expected FWE HellaSwag trade (FWE 37.5 vs DCLM 35.5 MMLU, DCLM 62.3 vs 60.1 HS per `thread-a-cpt-trajectory.md:300-305`). Re-check at 190k; if monotonic bleed continues (dolmino pattern), ratio is wrong — revisit 75/25 arm (`config_registry.py:1024-1043`) | `SYNTHESIS.md:242` |
| 2 | **G1-G2 pass but G4 fails** (no CoT lift, e.g. 0.08-0.12 like B2v2/B3/B4) | **Stop CPT**, fork SFT from best intermediate checkpoint (max over G1-G4, not necessarily 150k), pivot to broad-distill on MDS base (E7) | CPT did not move ceiling — do not pay 40k steps hoping it will. This is `POST-TRAINING-2B.md` phase-3 pivot | `SYNTHESIS.md:244` |
| 3 | **Everything fails** (commonsense tanks + CoT collapses) | **Stop CPT, revert to MDS base** | FWE kind-shift wrong at this scale/mix or constant LR 2e-6 needs re-tuning. Remaining 40k steps would be burning compute on a known-bad distribution | `SYNTHESIS.md:246` |
| 4 | **G1-G3 pass but SFT probe not yet run** (E3 pending) | **Continue CPT** | Do not make a stop decision without the ceiling test. CPT-direct MMLU/ARC scores are thermometers; the SFT probe is the thermostat | `SYNTHESIS.md:248` |

**Why 4 cases, not 2:** the gate is a conjunction over 3 signal families (commonsense G1-G2, knowledge G3, CoT ceiling G4). A binary pass/fail hides the actionable distinction between "CPT helps reasoning but hurts HellaSwag slightly" (case 1, continue) and "CPT preserves commonsense but does not lift CoT" (case 2, stop-fork). Case 4 prevents premature kills when the most expensive signal (SFT probe, ~2h walltime) has not yet been collected. Source: `SYNTHESIS.md:240-248` + `thread-a-cpt-trajectory.md:400-450` (gate derivation).

---

## 2. Ranked courses 1-7 — cost in N-hours + EV + risk + verdict

### 2.1 Ranked table — verbatim from `SYNTHESIS.md:36-51` (cost/EV/risk/verdict), annotated with ledger stream that delivers each rank

| Rank | Course of action | Compute cost (Sunspot N-hours / walltime) | Expected value | Key risk | Verdict | Ledger stream |
|------|------------------|--------------------------------------------|----------------|----------|---------|---------------|
| **1** | **Continue CPT to 190k on 12h cadence; gate at 150k via scoring ladder + SFT probe** | CPT: **+47k steps / 24.8B tok = ~2.5 hops of 4N/12h** (dominant cost already budgeted; each 12h hop ~24k steps per `99e918b342` fix). **Decision cost: ~8 HF converts (1N 2h each) + 2 eval sweeps + 1-3 SFT probes (1N 2h each) — all 1N, parallel with CPT** | **Highest.** Only course that tests the FWE kind-shift thesis (>60% educational content, the 60/40 SmolLM2 operating point) at scale. External evidence: FWE 33.6% MMLU at 38B vs next-best 300B (8x efficiency), OLMo-2 +17.4 MMLU at 1B on same olmo-mix base. 40% OWM math floor hedges dolmino 100% forgetting. If it lifts the SFT ceiling above 0.205, it unlocks the entire post-training stack | If ladder shows commonsense regression (dolmino repeat: HS -7pp by step 1000), ~2.5 hops burned. Mitigated by 150k early gate — <0.5 hop beyond current 143k snapshot | **Do this.** Default path in Thread A §4.3 and `POST-TRAINING-2B.md` phase ordering. Scoring and CPT are independent (DCP writes, HF reads) — never block CPT to score | **01 + 02 + 03 + 09** — `01` owns the chain, `02` converts HFs, `03` sweeps evals, `09` gates the decision |
| **2** | **LR x steps sweep on narrow SFT (gsm8k-r1cot from ckpt-900)** | **~1.0 N-hour total.** 7 configs, 2N each, ~2-5 min train + ~2 min eval per ckpt. Sequential <2h, parallel with rank-1 work ~30 min | **High.** Maps the forgetting boundary that governs *every* SFT call: at 2e-5 constant, how many steps before collapse? Does 1e-5 extend it? Does cosine help? B2 survived 93 steps, B2v2 halved at +6 steps (continuation), full-mix collapsed at ~1500-4500 (cosine LR>1e-5). No narrow mix has been swept at any LR other than 2e-5. Answers the "cap at O(1000) steps or drop LR" rule with a curve, not a slogan — and sets the LR for course 3 | Null result ("all collapse at ~100 steps regardless of LR") is still decisive — proves lever is data breadth, not LR tuning | **Do this in parallel with rank 1.** Cheapest experiment with broadest applicability. Thread E §5 is the runnable spec | **07** (dedicated LR-sweep ledger; ablations A0/A3 run alongside to gate rank 3) |
| **3** | **Broad-distill SFT on MDS base (OpenThoughts-114k-decontam + OpenR1-Math-220k-decontam via pretokenized `distill_cot_mix`)** | **2N smoke ~5 min at LR 2e-5 (or 1e-5 conservative); 4N full ~1-1.5h, walltime 12h.** Pretok already landed: 91k raw rows -> 36.9k packed @8192, 3.4G, recipe hash `cba5c64`, job `12483443` — zero pretok cost remaining. Cost is training + eval only | **High.** Only open path adding *new* CoT data under proven two-stage structure. Two-stage framing: `gs138650 -> tulu_math_uc_mix -> checkpoint-900` (stage-1, IFEval 0.253) -> `distill_cot_mix` at seq 8192, LR 2e-5 cosine, 48-144 steps / 1-3 epochs (GBS 768 @4N), envelope `<think>/<answer>\boxed{}` byte-identical to gsm8k-r1cot. Stage-2 swaps narrow 7.4k gsm8k for 50/50 frontier mix — breadth *within* stage 2, not mixing instruct+CoT in one stage (B3/B4's failure mode) | Distribution shift (CPT-17000 0.015 warning: frontier science/code vocab OOD vs olmo-mix-1124; `SYNTHESIS.md:46` rank 3 risk col), length/packing (8192 OOM risk on XPU without flash-attn), quality (OpenThoughts correctness weaker than OpenR1). Requires rank-2's winning LR (1e-5 conservative alternative) to avoid repeating ckpt-8672 at new scale | **Queue after rank 2.** Start 2N smoke at winning LR as soon as rank 2 reports; promote to 4N if GSM8K-CoT >=0.15 and format >=0.80 (E1 Go gate). Full A0-E7 ladder in `SYNTHESIS.md:116-167` | **08** (broad-distill ledger; gated on rank-2 LR + A0 audit) |
| **4** | **Fork SFT now on 143k CPT checkpoint (SFT probe as primary action, no ladder)** | **1N, 2h walltime** (B2 recipe: 93 steps, GBS 384, seq 2048, LR 2e-5). Cost is one SFT run + 200-problem eval (98s) | **Medium.** Directly tests "does 75% CPT lift the ceiling?" — the single question that matters (`SYNTHESIS.md:45` Thread A §4.2). Cheapest way to get a data point *today* without waiting for 150k | Without the ladder, a 143k SFT number is uninterpretable: if it scores 0.08 (like B2v2/B3), you cannot tell whether CPT hurt, the recipe was wrong, or you sampled a bad checkpoint. Repeats B2v2's mistake (healthy loss 0.8069 while eval halved) — cost of a mis-fork is a wasted SFT cycle plus a wrong pivot decision. Rank 1's ladder disambiguates for <0.5 hop extra | **Do as part of rank 1, not instead of it.** Fork one SFT probe on the best pre-143k HF (e.g. 74k or 112k) *while* the ladder runs; treat 143k itself as P2's frontier point, not a solo bet | **01 + 09** — one probe on best P2 HF while E9 ladder runs; not a standalone lane |
| **5** | **Broad-distill SFT on CPT base (CPT-FWE 112k+ HF + `distill_cot_mix` as stage-2)** | **Same as rank 3 plus CPT HF dependency.** 4N, GBS 768 @48 ranks, seq 8192, LR **1e-5 cosine conservative** (distribution-shifted base; 2e-5 re-warm may repeat CPT-17000 0.015 shock), 1ep ~48 steps, tokens/step ~6.3M. Gated on (1) CPT 112k+ HF verified (`config.json`+`tokenizer.json`+`model.safetensors` non-gibberish) and (2) narrow SFT probe on that base scored to calibrate format | **High ceiling, high cost.** If CPT lifts the base (commonsense G1-G3 held, MMLU CF lifted), combining it with broad data is the maximum achievable on this stack — kind-shift + breadth composition from SmolLM2 60/40 + OLMo-2 +17.4 precedent. This is `POST-TRAINING-2B.md` phase ordering (CPT to 50k+ then broad SFT) and `SYNTHESIS.md:46` rank 5 | Triple-blocked: (a) rank-1 gate (no CPT HF proven good yet), (b) rank-3 (broad mix must beat >=0.15 on MDS base first, else composition has no baseline), (c) CPT base harder than MDS — CPT-17000 probe 0.015 shows shifted manifold + narrow SFT collapses. Changing two things at once without baselines is how B3 failed | **Defer to after rank 1 + rank 3.** Do not run until (a) CPT gate passes at 150k and (b) broad-distill on MDS base (E1) has measured >=0.15 or Pivot. Then clean A/B: CPT base vs MDS base, same distill mix, same LR | **08 + 09** — `09` gates it (150k pass + MDS-broad baseline from `08`); `08` owns the mix |
| **6** | **RL/GRPO on B2 (or on rank-3 winner) for accuracy** | **2N Monarch, 30 min per 100-step probe at group_size 8-16.** Infra proven (job `12479177` 40/40 rollouts, 3 updates, Exit 0) | **Low for accuracy, proven for format.** GRPO on B2 held format 1.000 for 100 steps (drift-proof) but accuracy 0.205->0.215 = noise. Root cause: at 20% solve rate and group_size=4, 41% of groups all-wrong (zero advantage); even group_size 8 leaves 17% dead. No RL run has used a dense reward or group_size 8-16 with correct `max_completion_length` (512-768, not default 64) | Reward hacking is real: on weak base, reward 0.118->0.217 while accuracy 0.16->0.105, 95% rollouts no `<think>`, 20% echoed `\boxed{5}`. Requires componentized reward (format+completeness+correctness as in ceiling-attack +168%) and curated difficulty band — neither wired for GSM8K-CoT yet | **Run one cheap probe (group_size 8, LR 5e-6, 50-100 steps) in parallel with ranks 1-2 to close the question.** If accuracy stays flat (0.205-0.215), accept B2 as ceiling for RL-for-accuracy and keep RL as format-perfector only | **05 + 09 (this file, §4)** — `05` owns infra; `09` owns the probe spec + hacking gate |
| **7** | **Kill CPT now (stop chain, write off 75% budget, revert to MDS base)** | **Negative cost (saves ~2.5 hops)** | **Negative EV unless ladder shows degradation.** Abandons the only kind-shift bet (60% FWE vs 94.8% DCLM olmo-mix) that has external MMLU precedent (+17.4 at 1B, 8x token efficiency at 350B). No replacement base exists — v2-256n 32N blocked on oneCCL fault, and MDS base caps at ~0.205 | Dolmino trauma (300B pilot bled -7pp HS) makes this feel safe, but dolmino was 89% DCLM (degree tweak); FWE is a kind change with a 40% math floor. Killing at 75% without scoring would be the most expensive decision if FWE thesis is right | **Only if 150k gate catastrophically fails** (commonsense <-3pp *and* SFT probe <=0.10, or monotonic bleed across ladder). Even then, revert to MDS base for SFT — do not kill without the broad-distill lever queued | **01 §5 + 09 §5** — decision, not a compute lane |

Source for every rank's cost/EV/risk/verdict cell: `SYNTHESIS.md:36-51` (the 7-row table). EV judged on "likelihood of lifting the 0.205 CoT ceiling without regressing commonsense" per `SYNTHESIS.md:38`. Cost in Sunspot N-hours per `SYNTHESIS.md:36` header.

### 2.2 Reading the ranking — dependency DAG

Per `SYNTHESIS.md:50`: **Ranks 1 and 2 are independent and both cheap — run them together. Rank 3 depends on rank 2's LR. Rank 4 is a sub-experiment inside rank 1. Rank 5 depends on ranks 1+3. Rank 6 is independent but low-EV — one probe suffices. Rank 7 is a fallback, not a plan.**

```
rank 1 (CPT 150k gate) ─────────┐
rank 2 (LR x steps sweep) ────┐  │  both cheap, both run now
                               v  v
                          rank 3 (broad-distill MDS)  ← needs rank-2 LR
                          rank 4 (143k fork probe)    ← sub-experiment inside rank 1
                               │
                          rank 5 (broad-distill CPT)  ← needs rank 1 gate + rank 3 baseline
                          rank 6 (RL probe)           ← independent, 1 probe in parallel
                          rank 7 (kill CPT)           ← only if 150k catastrophically fails
```

Source: `SYNTHESIS.md:50` + `SYNTHESIS.md:52-195` experiment queue (E1-E10 dependency column).

---

## 3. E9 150k gate eval — CPT to 150k + score (convert+commonsense+MMLU+SFT probe) on single 150k HF

### 3.1 What E9 tests and depends on

| Field | Value | Source |
|-------|-------|--------|
| **Tests** | Does continuation from 143k to 150k hold the G1-G4 gate? | `SYNTHESIS.md:176-183` E9 table |
| **Depends on** | CPT chain advancing (live `qsub` 4N/12h auto-resumes from `CKPT_DIR`) + E1-E3 ladder through 143k already in hand (P0-P2: 1000..143400 scoring) | `SYNTHESIS.md:181` |
| **Run** | **Identical to E1-E3 but for the single 150k HF:** convert one DCP checkpoint, run commonsense+MMLU (E2 protocol), run SFT probe if G1-G3 hold | `SYNTHESIS.md:182` |
| **Decision** | §5 gate (this file §1 + §5): all G1-G4 hold => continue to 190k (P3). Single -2pp wobble => continue, re-check at 190k. Commonsense <-3pp or SFT probe <=0.10 => stop CPT, fork broad-distill (E7) from last-good checkpoint | `SYNTHESIS.md:183` |
| **Why single checkpoint** | E9 is not a ladder — the ladder (P0-P2, 8 converts covering 1000..143400) is E1's job. E9 scores **one new point** (150000) on the same protocol. Cost is one HF convert + one eval sweep + at most one SFT probe, all 1N `workq`, parallel with CPT | `SYNTHESIS.md:58,176-183` + `SYNTHESIS.md:198-209` ladder table (P2 frontier includes `150000` as first P2+ point) |

### 3.2 Qsub chain — exact commands (copy-paste runnable on Sunspot)

All jobs use `PBS filesystems=tegu:home` (required for any job reading `/home` DCP and writing Tegu HF/evals — Sunspot PBS has no `/eagle`; `SYNTHESIS.md:214` + `POST-TRAINING-2B.md:13` + `cpt_fineweb_edu_4n.sh:5` + `sft_distill_cot_4n.sh:5`). VENV mapping per assignment contract (ground truth from verified scripts): CPT `xpu-torch216-sync-5a7a622f16` LD-first, SFT/RL `rl-monarch-torch214-locality-20261001` LD-first, eval `frameworks/2026.1.0` module+PTI.

**Step 0 — CPT chain advances to 150k (no new command needed; auto-resume):**

The live 4N chain auto-resumes from `CKPT_DIR` on every hop. No new `qsub` is needed for E9's CPT leg — the chain is already advancing from 143.4k toward 150k on its 12h cadence. The requeue command (if the chain ever needs manual restart) is:

```bash
# CPT chain requeue — auto-resumes from CKPT_DIR (do NOT hand-edit step numbers)
qsub -l select=4 -l walltime=12:00:00 -v NHOSTS_TRAIN=4 \
  torchtitan/experiments/ezpz/scripts/cpt_fineweb_edu_4n.sh
# Source: cpt_fineweb_edu_4n.sh:12 + SYNTHESIS.md:181 ("auto-resumes from CKPT_DIR")
# CKPT_DIR prefers Tegu if present, else /home live 15T chain — see 01-cpt-frontier.md §1.3
# PBS header inside script: select=4 walltime=12:00:00 filesystems=tegu:home workq datascience
```

CKPT math: at `tps ~4625` and `~1.9k steps/hr` measured at `12483856` (`POST-TRAINING-2B.md:10`), 143.4k -> 150k = +6.6k steps = **~3.5h walltime**, well within one 12h hop. No extra hop is needed.

**Step 1 — Convert the single 150k DCP checkpoint to HF (1N, ~2h):**

```bash
# Single-checkpoint HF convert for the 150k gate (flavor=2b-mds is mandatory — wrong flavor = gibberish)
# Uses the batch converter; override STEPS to target only 150000
qsub -l select=1 -l walltime=02:00:00 -l filesystems=tegu:home -q workq -A datascience \
  -v STEPS=150000 \
  torchtitan/experiments/ezpz/scripts/eval/hf_batch_convert_cpt_fwe.sh
# Script default batch is 74400->136800 every 200; STEPS=150000 targets only the gate checkpoint
# Output: $TEGU_ROOT/outputs/evals/cpt-fwe/step-150000/hf  (9/9 files: safetensors+config+tokenizer, ~3.7G)
# Verify post-convert (before scoring — do not burn eval walltime on bad HF):
#   ls /lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-150000/hf/
#   # expect: config.json, tokenizer.json, tokenizer_config.json, special_tokens_map.json,
#   #         generation_config.json, model.safetensors, model.safetensors.index.json, etc. (9/9)
# Sources: hf_batch_convert_cpt_fwe.sh:1-60 + SYNTHESIS.md:58-66 (E1 Convert row) + 02-hf-conversion.md §2
```

**Step 2 — Score commonsense + MMLU on the single 150k HF (E2 protocol):**

E2's eval protocol per `SYNTHESIS.md:68-78`: **Aurora preferred for MMLU** (Sunspot `level_zero 40 63G OOM` on `compute_topk_scores` unfixable on XPU — `SYNTHESIS.md:74` + `thread-c-eval-harness.md:68-82` + `POST-TRAINING-2B.md:25`). Commonsense can run on either machine. CoT-free: `convert_and_eval.sh` DCP->HF -> `lm-eval 0.4.10` HF backend, `frameworks/2026.1.0` MKL+CCL+compiler+PTI LD as in `submit_cot_eval.sh`.

For the single-150k gate, the two practical options are:

```bash
# Option A — full lm-eval commonsense + MMLU sweep on Aurora (preferred for MMLU)
# (Aurora job; adjust -A/-q/-l filesystems to Aurora's flare:home)
qsub -A AuroraGPT -q workq -l select=1 -l walltime=02:00:00 -l filesystems=flare:home \
  -v CKPT=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-150000/hf \
  torchtitan/experiments/ezpz/scripts/eval/convert_and_eval.sh
# Benchmarks scored (SYNTHESIS.md:75): HellaSwag acc_norm 0-shot, ARC-Easy acc 0-shot,
#   PIQA acc_norm 0-shot, ARC-Challenge acc_norm 25-shot, MMLU 5-shot MCF (mmlu) + CF (mmlu_continuation) -> max(CF,MCF)
# Scoring discipline (SYNTHESIS.md:76): read group key `mmlu` from results.json (not hand-rolled macro:
#   macro +0.005 high at step-46429 per thread-c §1.5#1). State format (MCF/CF/max), n-shot (5), constant-D band 0.2295-0.2689.

# Option B — Sunspot commonsense-only sweep (if Aurora unavailable; skip MMLU until Aurora slot)
# Sunspot can score HS/ARC-Easy/PIQA/ARC-C but OOMs on MMLU's compute_topk_scores at 63G
# Use the same convert_and_eval.sh but note MMLU will OOM — gate G3 stays informational until Aurora MMLU lands
```

**Step 3 — SFT probe on the 150k HF if G1-G3 hold (1N, ~2h, `workq`):**

Per `SYNTHESIS.md:79-87` E3, the SFT probe is **the B2 recipe verbatim** on the CPT base — the only recipe that produced 0.205:

```bash
# SFT probe — B2 recipe verbatim on 150k HF (only if G1-G3 hold per §1 decision tree)
# Two stages: stage-1 tulu_math_uc_mix -> checkpoint-900 equivalent on CPT base,
#   stage-2 gsm8k-r1cot 93 steps 3ep LR 2e-5 cosine GBS 384 seq 2048 assistant_only_loss
# Script pattern: agpt2b_b2v2_distill_cot_2n.sh (1N/2N, VENV rl-monarch-locality, TRL SFTTrainer FSDP full_shard)
qsub -l select=1 -l walltime=02:00:00 -l filesystems=tegu:home -q workq -A datascience \
  -v BASE_MODEL=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-150000/hf \
  torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh
# Then score the SFT output on CoT (same harness as B2):
qsub -v CKPT=<sft-output>/final,LIMIT=200 torchtitan/experiments/ezpz/scripts/eval/submit_cot_eval.sh
# Baseline to beat: B2 = 0.205 / 0.985 / 282 on same harness (greedy, temp 0.0, max_tokens 768, vLLM fp32 0.26)
# Gate: G4 >=0.265 + format >=0.90 + gen_len <=550 + unclosed <=10 — see §1.1 above
# Sources: SYNTHESIS.md:79-87 (E3) + SYNTHESIS.md:225-239 (gate table) + 03-eval-harness.md §2 + 04-sft-broad-distill.md §2
```

**What makes E9 "identical to E1-E3 but single checkpoint":** per `SYNTHESIS.md:182` the Run field is literally "Identical to E1-E3 but for the single 150k HF: convert, run commonsense+MMLU (E2 protocol), run SFT probe if G1-G3 hold." The converter is the same `hf_batch_convert_cpt_fwe.sh` with `flavor=2b-mds`, the commonsense suite is the same 5-benchmark set, the MMLU scoring is the same `max(CF,MCF)` with band, and the SFT probe is the same B2 93-step recipe. Only the checkpoint set changes: E1 had 8 points (P0-P2: 1000..143400), E9 has 1 point (150000). Source: `SYNTHESIS.md:176-183` + `SYNTHESIS.md:52-87` (E1-E3 definitions).

### 3.3 Cost and parallelism

| Step | Resource | Walltime | Blocks CPT chain? |
|------|----------|----------|-------------------|
| 0 CPT 143.4k->150k | 4N/12h `workq` (already running) | ~3.5h within one 12h hop | — (is the chain) |
| 1 HF convert 150k | 1N/2h `workq` | ~2h | **No** — DCP writes, HF reads are independent per `SYNTHESIS.md:43` rank-1 verdict |
| 2 commonsense+MMLU | 1N/2h (Aurora for MMLU) | ~2h + MMLU walltime | **No** — reads HF, parallel with CPT |
| 3 SFT probe (if G1-G3 hold) | 1N/2h `workq` | ~2h + 98s eval | **No** — parallel with CPT |

**Decision cost: ~1 HF convert + 1 eval sweep + 0-1 SFT probes, all 1N, all `workq`, zero new CPT compute beyond the already-budgeted 2.5 hops.** The P0-P2 ladder (8 converts + 2 sweeps + 3 probes) is E1-E3's cost; E9 adds only one more point. Source: `SYNTHESIS.md:181-183` + `SYNTHESIS.md:43` rank-1 cost column + `01-cpt-frontier.md` §1.2 (hop math).

---

## 4. RL probe — 2N Monarch group_size 8 LR5e-6 float32 max_completion 512-768 dense reward

### 4.1 What the probe tests and when to run it

| Field | Value | Source |
|-------|-------|--------|
| **Tests** | Was GRPO's accuracy-flat (B2 0.205->0.215 noise at group_size 4) a group_size/reward-density artifact? | `SYNTHESIS.md:168-174` E8 Tests row |
| **Depends on** | **E4 (B2 reproduces at 0.18-0.23 / >=0.95 format / 250-350 gen_len)** — otherwise run on stored B2 `checkpoint-93-hf` (B2's 0.205/0.985/282 is the anchor) | `SYNTHESIS.md:172` E8 Depends row + `SYNTHESIS.md:89-96` E4 |
| **When to run** | **E4 passes (B2 reproduces on current stack) OR B2 stored HF exists** — either satisfies the base check. Per `SYNTHESIS.md:172`: "E4 (B2 reproduces) — otherwise run on stored B2." Also per `SYNTHESIS.md:47` rank 6: independent of CPT ladder — run one probe in parallel with ranks 1-2 to close the question | `SYNTHESIS.md:172` + `SYNTHESIS.md:47` rank 6 + `thread-e-recipe-optimization.md:229-242` §3.6 |
| **Also valid when** | Rank-3 (broad-distill MDS) produces a winner — then re-run the same probe on the rank-3 winner to test whether broad data lifts RL headroom (per `SYNTHESIS.md:47` rank 6 "or on rank-3 winner" + `thread-e-recipe-optimization.md:236-242` "when RL worth it" condition 2) | `SYNTHESIS.md:47` + `thread-e-recipe-optimization.md:236` |

### 4.2 Group_size dead-group math — the root cause of B2's flat GRPO

At B2's ~20% solve rate (`cot_accuracy 0.205 = 41/200`), the expected correct rollouts per group is `0.2 * group_size`. GRPO's within-group advantage normalization yields **zero gradient for accuracy** on any group where all rollouts are wrong (no variance in the correctness signal).

The all-wrong probability is `(1 - p)^group_size` with `p = 0.20`:

| group_size (`num_generations` / `--async-loop.num-samples-per-prompt`) | All-wrong probability | Dead groups (zero advantage) | Verdict at 20% solve rate |
|---|---|---|---|
| **4 (B2 GRPO as run)** | `0.8^4 = 0.4096` | **41%** | 41% of groups contribute no correctness gradient — accuracy-flat 0.205->0.215 is expected, not surprising |
| **8 (E8 probe)** | `0.8^8 = 0.1678` | **17%** | 2.4x thinner dead mass; first informative group_size for 20% base |
| **16 (if signal at 8)** | `0.8^16 = 0.0281` | **3%** | Near-zero dead groups; only if 8 shows signal — 16 is 2x vLLM memory |

Source: `thread-e-recipe-optimization.md:196-204` (arithmetic + table at lines 200-204) + `SYNTHESIS.md:171` E8 Tests row + `05-infra-rl-storage.md:64-77` §2.1 (same calculation with source attribution).

**Why group_size alone is not sufficient:** with a binary exact-match reward (right vs wrong, no partial credit), groups with exactly 1 correct of N give advantages of `(1 - 1/N)` for the correct rollout vs `(-1/N)` for each wrong one — a weak signal that advantage normalization washes out. The `ceiling-attack` study proved this: on `alphabet_sort` where four runs varying LR (2e-5->5e-5), rank (8->32), and batch (8->16 groups) all plateaued at 0.24-0.25 mean, **only the reward change moved it** — decomposing the same task into additive sub-skills (format 0.2 + completeness 0.2 + order 0.6) lifted cross-scored accuracy 0.24 -> 0.667 (+168%) on the original char-ratio metric (cross-scored offline, self-validating). Source: `thread-e-recipe-optimization.md:184-192` + `POST-TRAINING-2B.md:138-146` + `SYNTHESIS.md:47` rank 6 Key risk column.

### 4.3 Full probe hparams — with fixes over the B2 GRPO run

| Hparam | B2 GRPO as run | **E8 probe (recommended)** | Why / source |
|--------|----------------|-----------------------------|--------------|
| `group_size` (`num_generations`) | **4** (41% dead) — `multihost_grpo_e2e.pbs:151-153` `num_prompts_per_train_step=4, num_samples_per_prompt=4` | **8** (17% dead); 16 only if 8 shows signal | `thread-e-recipe-optimization.md:200-204,220` + `SYNTHESIS.md:173` |
| `learning_rate` | **unknown** (not logged in B2 GRPO) | **5e-6** conservative first; 1e-5 if stable | `thread-e-recipe-optimization.md:221` + `POST-TRAINING-2B.md:175` next-step #2. `sft_arithmetic` 8N solved at 1e-6; `alphabet_sort` used 2e-5-5e-5 before shaped reward. GSM8K-CoT harder than sum_digits but B2 is already SFT'd so conservative. Try 5e-6 first |
| `beta` (KL penalty) | **0.0** (inferred from sum_digits + B2 GRPO no-drift) | **0.0 -> 0.01 guardrail** — 0.0 worked for sum_digits and B2; small 0.01 only if format bleeds | `thread-e-recipe-optimization.md:222` + `SYNTHESIS.md:173` |
| `generator.model_dtype` | **float32 (required on XPU)** | **float32 mandatory** — bf16 through vLLM makes agpt-2b emit gibberish (vocab 256000 + ffn 11008 accumulate enough bf16 error to flip greedy argmax) | `POST-TRAINING-2B.md:158` NOTE (XPU landmine) + `reason_agpt/config_registry.py:27` + `05-infra-rl-storage.md:173-177` |
| `max_completion_length` (`max_tokens`) | **700** in `multihost_grpo_e2e.pbs:156`; default **64** in `train_grpo.py:185` | **512-768** — must cover full CoT + `\boxed{}` answer; GSM8K CoT avg ~300 tokens (B2 gen_len 282), 64 truncates everything | `thread-e-recipe-optimization.md:223` + `train_grpo.py:185` + `reason_agpt/config_registry.py` + `SYNTHESIS.md:173` |
| `reward` | **binary/char-ratio** on the smoke (reward 0.43-0.54 flat) | **Componentized additive** per `ceiling-attack.md` (+168%): `think_format 0.1-0.2` + `answer_extractable 0.1` + `answer_correct` dense (binary + numeric closeness `1 - |pred-gold|/scale`) as separate `RewardFn` classes. At minimum: `answer_correct` with partial credit + `think_format` + `answer_extractable` | `thread-e-recipe-optimization.md:224` + `POST-TRAINING-2B.md:138-150` + `SYNTHESIS.md:47` rank 6 + `05-infra-rl-storage.md:179-180` |
| `difficulty` | **random GSM8K** | **Curated band** — filter to problems where B2's `pass@16` is 10-50% (hard enough CoT is load-bearing, easy enough base gets non-zero correct fraction). If baseline ~0, RL gradient cannot start | `thread-e-recipe-optimization.md:225` |
| Base | **B2 ckpt-93** (0.205 / 0.985) | **B2 ckpt-93** (or rank-3 winner if available) | `SYNTHESIS.md:173` + `thread-e-recipe-optimization.md:172-173` |
| Steps / eval cadence | **100 steps** (B2 GRPO) | **50 + 100 step eval** | `SYNTHESIS.md:173` (50+100) + `05-infra-rl-storage.md:26-27` (in-flight `12483252` staged at `gs=8 lr=5e-6 100 steps ckpt/25`) |
| VENV | `rl-monarch-torch214-locality-20261001` LD-first | **Same** — `rl-monarch-torch214-locality-20261001` + `LD_LIBRARY_PATH=$VENV/lib` prepend + `.cache/xpu-ld-first/` LD-first symlink farm. Do NOT use `xpu-torch216` (no TRL) or bare `xpu-torch214` (UR loader mismatch) or `frameworks/2026.1.0` (eval-only, no `torch_remat`/`trl`/`monarch`) | `05-infra-rl-storage.md:45-54` VENV table + `SYNTHESIS.md:173` + `cpt_fineweb_edu_4n.sh:52-70` |
| Scale | **2N Monarch** (trainer host 0, generator host 1, TorchStore/Gloo weight sync) | **2N Monarch** — same topology; 30 min per 100-step probe at group_size 8-16 (walltime scales with 73s XPU weight-fetch per sync — `05-infra-rl-storage.md:90-94`) | `SYNTHESIS.md:173` + `05-infra-rl-storage.md:12-28` + `monarch.md:24-37` |
| Other invariants | — | `flex` attention `max_autotune=False`, `similarity_power=1`, **`--no-drop-zero-std-reward-groups`** (without it cold-start groups are all dropped and no train step fires) | `POST-TRAINING-2B.md:158` NOTE + `05-infra-rl-storage.md:181` |
| Infra gate | `12479177` 40/40 rollouts, 3 updates, Exit 0 | **Same gate must still hold** — `12479177` `40/40` `auto->Gloo` 4 pushes/4 pulls | `monarch.md:152-165` + `05-infra-rl-storage.md:24-26` |

### 4.4 Decision gate — what counts as signal vs noise vs hacking

Per `SYNTHESIS.md:174` E8 Decision row + `SYNTHESIS.md:260-264` RL-specific gates + `thread-e-recipe-optimization.md:206-212` weak-base analysis:

| Outcome after 50-100 GRPO steps | Interpretation | Action |
|----------------------------------|---------------|--------|
| **Accuracy >0.25** (outside noise) with format >=0.99 | **Signal exists** — RL-for-accuracy exists with proper hparams. Lift is >=0.05 at 2sigma on 200 (SE 0.028 at 0.20 => 0.05 = 1.8 sigma) — outside the 0.205->0.215 noise band that is 0.26 sigma | **Scale** with shaped reward per `ceiling-attack.md` (+168% via decomposition). Try group_size 16, denser reward, curated difficulty band |
| **Accuracy 0.205-0.215 flat** with format 1.000 held | **No lift — accept B2 as RL ceiling at 2B.** Every accuracy lever tried (RL, B3 0.05, B4a 0.02, B4b 0.065, B2v2 0.09/0.105) converged to 0.02-0.215 vs B2 0.205. `POST-TRAINING-2B.md` Finding: "~0.2 is near this base's GSM8K-CoT ceiling." If true, RL cannot exceed it any more than SFT can — base lacks capability | **Keep RL as format-perfector only.** Do not scale RL until SFT ceiling moves (ranks 1/3). RL reweights rollouts the base already samples — if headroom is ~0, no reward shaping can help |
| **Reward rises while accuracy falls** (weak-base hacking pattern: reward 0.118->0.217, accuracy 0.16->0.105, format 0.955->0.635, 95% no `<think>`, 20% echoing `\boxed{5}`) | **Reward hacking — reward function mis-specified.** Gating the reward drove it to 0.000, revealing 95% no-`<think>` — model was reward-hacking by emitting answer format without reasoning. Down-weighting format 0.2->0.05 removed the guardrail that SFT had been holding up | **Revert weights, stop, fix reward.** Do not down-weight format to "fix" — on strong B2 the same reward is drift-proof (format 1.000). Cold-start quality gates RL. Requires componentized reward (format+completeness+correctness as in ceiling-attack) |
| **Format bleeds** (<0.99, e.g. 0.955->0.635 as in weak-base) | **Drift — beta guardrail** | Apply `beta 0.0 -> 0.01` guardrail before any other tuning. If format bleeds on B2 at group_size 8, the reward density is still wrong — fix reward before touching beta |

Sources: `SYNTHESIS.md:174` (E8 Decision row) + `SYNTHESIS.md:260-264` (RL-specific gates) + `thread-e-recipe-optimization.md:206-212` (weak-base hacking) + `POST-TRAINING-2B.md:141-150` (weak-base table) + `05-infra-rl-storage.md:61-66,98-116` (gate summary).

### 4.5 Qsub command — RL probe on B2 at group_size 8

```bash
# RL/GRPO probe — 2N Monarch, group_size 8, LR 5e-6, float32, max_completion 512-768, componentized reward
# VENV: rl-monarch-torch214-locality-20261001 (not xpu-torch216 — TRL path; not frameworks/2026.1.0 — eval-only)
# PBS filesystems must be tegu:home (reads /home DCP if needed, writes Tegu)
qsub -l select=2 -l walltime=00:50:00 -l filesystems=tegu:home -q workq -A datascience \
  -v V=/lus/tegu/projects/datascience/foremans/venvs/rl-monarch-torch214-locality-20261001,CKPT=/path/to/checkpoint-93-hf,STEPS=100 \
  torchtitan/experiments/ezpz/rl/scripts/grpo/multihost_grpo_e2e.pbs
# Required envs: V (RL VENV), CKPT (HF checkpoint dir — B2 checkpoint-93-hf = 0.205/0.985 deliverable)
# Optional overrides (see multihost_grpo_e2e.pbs:19-22):
#   OUT (default $WT/outputs/rl-grpo-e2e-$JOBID), STEPS (default 3 — override to 100 for probe),
#   TORCHTITAN_TORCHSTORE_TRANSPORT (default auto on XPU / gloo on CUDA),
#   LAUNCH_TIMEOUT (default 2400s), EXPECTED_COMMIT (fail unless HEAD matches)
# Config consumed: rl_grpo_lora_agpt_2b_gsm8k_b2smoke (STEPS=3 default, num_prompts=4, num_samples=4, max_tokens=700)
# For the E8 probe override group_size to 8 + LR 5e-6 + max_completion 512 via EXTRA_ARGS or config patch:
#   --rl.group_size=8 --rl.learning_rate=5e-6 --rl.max_completion_length=512 --generator.model_dtype=float32
#   --rl.reward_fns='[think_format(0.1), answer_extractable(0.1), answer_correct(dense)]' --rl.beta=0.0 --no-drop-zero-std-reward-groups
# Walltime: ~30 min per 100 steps at group_size 8 (plus 73s XPU weight-fetch per sync — 05-infra-rl-storage.md:90-94)
# Gate after: score 50-step + 100-step checkpoints on CoT 200-problem eval (same harness as B2 — see §5 below)
# Sources: multihost_grpo_e2e.pbs:1-164 + SYNTHESIS.md:168-174 (E8) + thread-e-recipe-optimization.md:215-245 (§3.5)
#          + 05-infra-rl-storage.md:12-28 (infra) + 05-infra-rl-storage.md:168-183 (§5.1 recipe table)
```

**Infra already proven:** Sunspot `12479177` (`40/40` rollouts across policy versions `0-3`, `auto->Gloo`, 4 pushes/4 pulls, 3 finite nonzero-gradient updates, DCP checkpoints at steps 1-3, `Exit 0`) per `monarch.md:152-165` + `05-infra-rl-storage.md:24-26`. The 3-step cross-machine smoke (`Sirius 33518`, `Sunspot 12483743`, `Aurora 9089138`) re-validated the same stack on `8e63247c` per `journal.md:62-74`. An in-flight B2 GRPO at `12483252` was staged at `gs=8 lr=5e-6 100 steps ckpt/25` on `checkpoint-93-hf` but outcome was `R --` (staged, not yet training) at doc time — no exit-verified result recorded on this branch (see `05-infra-rl-storage.md:27` `[INFERENCE]` note). Source: `05-infra-rl-storage.md:12-28` + `journal.md:122-142`.

### 4.6 When RL is worth scaling — and when it is not (Thread E §3.6)

Per `thread-e-recipe-optimization.md:229-242` **"When RL-for-accuracy would be worth it"**:

1. **On a stronger base** where the ceiling is higher — the CPT-FWE 60/40 arm or a larger/math-pretrained model. If a new base lifts B2-equivalent SFT to 0.30+, RL has headroom. Until then, RL-for-accuracy at 2B is expected flat — budget as a few smoke runs, not a campaign.
2. **With a dense reward** that gives partial credit — the ceiling-attack's +168% was on a task where the original reward was saturating. GSM8K-CoT's binary exact-match is similarly saturating.
3. **With group_size 8-16** so that every group has at least one correct rollout to anchor the advantage.

Source: `thread-e-recipe-optimization.md:229-242` + `SYNTHESIS.md:47` rank 6 + `05-infra-rl-storage.md:64-77` (root-cause framing).

---

## 5. Decision tree — if G1-G4 hold → 190k P3 else if commonsense drop → stop fork else if G4 missing → continue

### 5.1 ASCII flowchart — the 150k gate as a single decision

```
                         ┌─────────────────────┐
                         │  Score 150k HF on   │
                         │  G1-G3 (commonsense │
                         │  + MMLU) — E2       │
                         │  protocol, 1N       │
                         └─────────┬───────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
              G1-G3 PASS     G1-G3 FAIL     G1-G3 MARGINAL
              (all >= thresh) (any < thresh  (single -2pp wobble
                    │         by >2pp)        within SE)
                    │              │              │
                    v              v              v
              ┌──────────┐  ┌──────────────┐ ┌──────────┐
              │ Run SFT  │  │ Was G4 known │ │ CONTINUE │
              │ probe on │  │ to be good   │ │ to 190k  │
              │ 150k HF  │  │ at earlier   │ │ Re-check │
              │ (B2      │  │ checkpoint?  │ │ at 190k  │
              │  recipe) │  │              │ │          │
              └────┬─────┘  └──────┬───────┘ └──────────┘
                   │               │  (SYNTHESIS
                   │         ┌─────┴─────┐   §5: "Single
        ┌──────────┼──────────┤             -2pp wobble
        │          │          │             = noise,
     G4 PASS   G4 FAIL    G4 PASS@        continue"
        │          │       earlier?          │
        v          v          │              │
  ┌──────────┐ ┌──────────┐   │              │
  │ CONTINUE │ │ STOP CPT │   v              │
  │ to 190k  │ │ Fork SFT │ ┌──────────┐     │
  │ (P3)     │ │ from best│ │ CONTINUE │     │
  │ All G1-  │ │ interval │ │ to 190k  │     │
  │ G4 hold  │ │ checkpoint│ │ but note │     │
  │ => P3    │ │ (max over│ │ commonsense│   │
  │ 160k,    │ │ G1-G4,   │ │ dip — if │     │
  │ 175k,    │ │ not nec. │ │ monotonic│     │
  │ 190734   │ │ 150k),   │ │ bleed at │     │
  └──────────┘ │ pivot to │ │ 190k,    │     │
              │ broad-   │ │ ratio is │     │
              │ distill  │ │ wrong —  │     │
              │ on MDS   │ │ revisit  │     │
              │ base (E7)│ │ 75/25 arm│     │
              └──────────┘ └──────────┘     │
                   │                        │
                   v                        │
              ┌──────────┐                  │
              │  STOP CPT│                  │
              │  Revert  │                  │
              │  to MDS  │◄─────────────────┘
              │  base    │   (if monotonic bleed
              │  FWE     │    continues to 190k
              │  kind    │    — dolmino pattern)
              │  wrong   │
              └──────────┘

  ┌──────────────────────────────────────────────────────┐
  │  G1-G3 pass but SFT probe not yet run (E3 pending)  │
  │  => CONTINUE CPT — do not make stop decision         │
  │     without ceiling test. CPT-direct scores are      │
  │     thermometers; SFT probe is thermostat.            │
  │     Source: SYNTHESIS.md:248                          │
  └──────────────────────────────────────────────────────┘
```

Source for every branch: `SYNTHESIS.md:240-248` (4 near-miss cases) + `SYNTHESIS.md:223-239` (gate definitions) + `SYNTHESIS.md:176-193` (E9/E10 dependency).

**Text form of the 4 near-miss leaves** (for copy-paste into decision docs):

| Leaf | Condition | Leaf action | Source |
|------|-----------|-------------|--------|
| A | G1-G3 pass, G4 pass (all G1-G4 hold) | **Continue to 190k** — run P3 ladder (160k, 175k, 190734) per `SYNTHESIS.md:202-209`; then E10 composition test (broad-distill on best CPT base vs MDS base) | `SYNTHESIS.md:240-242` mapped to `SYNTHESIS.md:185-192` E10 |
| B | G1-G3 pass, G4 fail (0.08-0.12 like B2v2/B3/B4) | **Stop CPT**, fork SFT from best intermediate checkpoint (max over G1-G4, not necessarily 150k), pivot to broad-distill on MDS base (E7) | `SYNTHESIS.md:244` |
| C | G1-G2 fail by >2pp but G4 passes | **Continue to 190k** with note — FWE HellaSwag trade expected; if monotonic bleed continues to 190k, ratio wrong, revisit 75/25 arm | `SYNTHESIS.md:242` |
| D | Everything fails (commonsense tanks + CoT collapses) | **Stop CPT, revert to MDS base** — FWE kind-shift wrong at this scale/mix | `SYNTHESIS.md:246` |
| E | G1-G3 pass, SFT probe not yet run | **Continue CPT** — do not decide without ceiling test | `SYNTHESIS.md:248` |

### 5.2 Exact eval commands that feed the gate

The gate consumes **two eval outputs**: the commonsense+MMLU sweep (G1-G3) and the SFT probe's CoT score (G4). Both use the harness documented in `03-eval-harness.md` + `SYNTHESIS.md:68-87`.

**Command 1 — Commonsense + MMLU sweep on the 150k HF (feeds G1-G3):**

```bash
# Commonsense+MMLU sweep (Aurora preferred for MMLU — Sunspot OOMs at compute_topk_scores 63G)
# Benchmarks: HellaSwag acc_norm 0-shot, ARC-Easy acc 0-shot, PIQA acc_norm 0-shot,
#   ARC-Challenge acc_norm 25-shot, MMLU 5-shot MCF (mmlu) + CF (mmlu_continuation) -> max(CF,MCF)
# Scoring discipline: read group key `mmlu` from results.json (not hand-rolled macro — macro +0.005 high
#   at step-46429 per thread-c §1.5#1). State format (MCF/CF/max), n-shot (5), constant-D band 0.2295-0.2689.

# Aurora (MMLU-capable):
qsub -A AuroraGPT -q workq -l select=1 -l walltime=02:00:00 -l filesystems=flare:home \
  -v CKPT=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-150000/hf \
  torchtitan/experiments/ezpz/scripts/eval/convert_and_eval.sh
# Source: SYNTHESIS.md:74 (Aurora preferred) + SYNTHESIS.md:75-76 (benchmark list + scoring discipline)
# Output: results.json with group key `mmlu` + per-task keys (hellaswag, arc_easy, piqa, arc_challenge, mmlu, mmlu_continuation)
# Gate read: compare each benchmark's acc/acc_norm against §1.1 thresholds; MMLU max(CF,MCF) >=0.240

# Sunspot commonsense-only (if Aurora unavailable — G3 stays informational until Aurora MMLU lands):
qsub -l select=1 -l walltime=02:00:00 -l filesystems=tegu:home -q workq -A datascience \
  -v CKPT=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-150000/hf \
  torchtitan/experiments/ezpz/scripts/eval/convert_and_eval.sh
# Same script; Sunspot will OOM on MMLU — gate G1-G2 on HS/ARC-E/PIQA/ARC-C only, G3 deferred
```

**Command 2 — SFT probe on the 150k HF, then CoT eval on its `final/` (feeds G4):**

```bash
# SFT probe — B2 recipe verbatim (only if G1-G3 hold; otherwise skip and apply near-miss case 2/4)
qsub -l select=1 -l walltime=02:00:00 -l filesystems=tegu:home -q workq -A datascience \
  -v BASE_MODEL=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-150000/hf \
  torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh
# Script pattern: GBS 384 seq 2048 LR 2e-5 cosine 93 steps 3ep assistant_only_loss
#   VENV rl-monarch-torch214-locality-20261001 LD-first (not xpu-torch216 — TRL path)
#   TRL SFTTrainer FSDP full_shard — see 04-sft-broad-distill.md §2
# Output: <sft-output>/final/  (HF dir with config+tokenizer+safetensors)
# Sources: SYNTHESIS.md:79-87 (E3 Run row) + 04-sft-broad-distill.md §2 + agpt2b_b2v2_distill_cot_2n.sh:1-120

# CoT eval on the SFT probe's final/ — the G4 read (200-problem GSM8K-CoT, vLLM fp32 0.26, greedy)
qsub -v CKPT=<sft-output>/final,LIMIT=200 torchtitan/experiments/ezpz/scripts/eval/submit_cot_eval.sh
# Expands to: $PY -u eval_cot_gsm8k.py --model "$CKPT" --limit 200 --dtype float32 --out "$OUT"
#   where $PY=/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/bin/python
#         DTYPE=float32, LIMIT=200, OUT=$REPO/outputs/evals/cot/gsm8k-cot-<name>.jsonl
# Output JSON summary: {n:200, format_hit_rate, cot_accuracy, mean_gen_len, n_unclosed} + model/dtype/temperature/max_tokens
# Output JSONL per-example: {idx, question, prompt, generation, format_ok, pred, gold, correct, gen_len, finish_reason}
# Baseline to beat: B2 = 0.205 / 0.985 / 282 on same harness (greedy temp 0.0 max_tokens 768 vLLM fp32 0.26)
# Gate: G4 >=0.265 + format >=0.90 + gen_len <=550 + unclosed <=10 — see §1.1
# Sources: submit_cot_eval.sh:1-35 + eval_cot_gsm8k.py:30-232 + 03-eval-harness.md §1-2 + SYNTHESIS.md:236-239
```

**Gate application after both commands exit:**

```bash
# 1. Read G1-G3 from lm-eval results.json (Aurora MMLU + Sunspot commonsense — same HF, either machine)
#    hellaswag acc_norm >=0.540, arc_easy acc >=0.621, piqa acc_norm >=0.713, arc_challenge acc_norm >=0.290,
#    mmlu max(CF,MCF) >=0.240 — see §1.1 table. If any G1-G2 fails by >2pp, apply near-miss logic (case 1 vs 3).
# 2. If G1-G3 hold, read G4 from CoT JSON summary (cot_accuracy >=0.265, format_hit_rate >=0.90, mean_gen_len <=550, n_unclosed <=10).
# 3. Apply the 5-leaf decision tree above. No invented thresholds — every number is verbatim from SYNTHESIS §5.
# 4. If RL probe ran in parallel (see §4.5), read its 50/100-step CoT evals and apply §4.4 hacking gate:
#    reward up + accuracy down => revert, do not promote.
```

Source for the two-command split: `SYNTHESIS.md:176-183` E9 Run field ("Identical to E1-E3 but for the single 150k HF: convert, run commonsense+MMLU (E2 protocol), run SFT probe if G1-G3 hold") + `SYNTHESIS.md:74-78` E2 (benchmark list + Aurora preference) + `SYNTHESIS.md:84-87` E3 (B2 recipe + baseline to beat). The `qsub` forms are verbatim from `submit_cot_eval.sh:10` + `agpt2b_b2v2_distill_cot_2n.sh` (PBS header pattern `select=1 walltime=00:40:00 filesystems=tegu:home workq datascience`) + `cpt_fineweb_edu_4n.sh:12` (CPT requeue).

### 5.3 What happens after the gate — E10 (for completeness)

If the gate says **continue to 190k**, the P3 ladder (`160000, 175000, 190734` per `SYNTHESIS.md:202-209`) runs the same convert+score protocol at three more points, closing the budget. Then **E10** (`SYNTHESIS.md:185-192`) tests composition: same broad-distill config as `04-sft-broad-distill.md`'s E7-full but on the 190k HF (or best P3 HF), same LR/mix/harness — only the base changes. If CPT-broad beats MDS-broad by >=0.06 at 2sigma, CPT thesis validated — promote CPT-broad; if flat or worse, deliver MDS-broad. Either way, the 190k ladder closes the "loss vs capability" question with a measured curve (Thread A Fig 5/6 equivalent for FWE). Source: `SYNTHESIS.md:185-192`.

---

## Appendix — Source index (every claim traceable)

| Claim | Source file @ `a6135a79` |
|-------|---------------------------|
| G1-G4 thresholds, SE, rationale, near-miss 4 cases, RL-specific gates | `torchtitan/experiments/ezpz/docs/exploration/SYNTHESIS.md:223-265` (§5 Decision Gates) |
| Ranked next-course matrix ranks 1-7 cost/EV/risk/verdict + dependency DAG | `torchtitan/experiments/ezpz/docs/exploration/SYNTHESIS.md:36-51` (§2) — full untruncated rows in `:raw:36-51` |
| E8 RL probe (group_size 8, LR 5e-6, float32, max_completion 512-768, dense reward, when-to-run) | `torchtitan/experiments/ezpz/docs/exploration/SYNTHESIS.md:168-175` (E8) |
| E9 150k gate eval (single 150k HF, identical to E1-E3, qsub chain auto-resume) | `torchtitan/experiments/ezpz/docs/exploration/SYNTHESIS.md:176-184` (E9) |
| E10 190k terminal gate + broad-distill composition | `torchtitan/experiments/ezpz/docs/exploration/SYNTHESIS.md:185-193` (E10) |
| Scoring ladder P0-P3 checkpoints, converter contract (flavor=2b-mds) | `torchtitan/experiments/ezpz/docs/exploration/SYNTHESIS.md:198-218` (§4) |
| RL table (GRPO on B2 0.205->0.215 noise, weak-base hacking 0.118->0.217/0.16->0.105, when RL worth it) | `torchtitan/experiments/ezpz/docs/exploration/thread-e-recipe-optimization.md:168-245` (§3) |
| Group_size dead-group math 41%->17%, dense reward, LR 5e-6-1e-5, generator.dtype fix, ceiling-attack +168% | `thread-e-recipe-optimization.md:196-227` (§3.3-3.5) + `SYNTHESIS.md:47` rank 6 + `POST-TRAINING-2B.md:138-160` |
| GRPO infra 12479177 40/40 Exit0, Monarch 2-host TorchStore Gloo, XPU weight-fetch 73s vs CUDA 3.9s | `torchtitan/experiments/ezpz/docs/run-ledger/05-infra-rl-storage.md:12-30,90-94` + `monarch.md:24-37,152-165` + `journal.md:62-74,122-142` |
| CPT chain budget 100B/524288=190734, 143.4k/190.7k=75.2%, 12h walltime fix 99e918b342 | `torchtitan/experiments/ezpz/docs/run-ledger/01-cpt-frontier.md:18-28,38-68` + `cpt_fineweb_edu_4n.sh:72-78` |
| HF batch converter PBS tegu:home, flavor=2b-mds, 9/9 files | `torchtitan/experiments/ezpz/scripts/eval/hf_batch_convert_cpt_fwe.sh:1-60` + `02-hf-conversion.md` |
| Eval harness vLLM fp32 greedy 200-problem, MMLU band 0.2295-0.2689, SE 0.028@200 | `torchtitan/experiments/ezpz/docs/run-ledger/03-eval-harness.md:18-60` + `submit_cot_eval.sh:1-35` + `eval_cot_gsm8k.py:30-232` |
| SFT B2 recipe (93 steps GBS384 seq2048 LR2e-5, 0.205/0.985/282) + B2v2 halving 0.09/0.105 | `torchtitan/experiments/ezpz/docs/run-ledger/04-sft-broad-distill.md:18-50` + `POST-TRAINING-2B.md:45,77-82` + `SYNTHESIS.md:17,84-87` |
| VENV mapping (xpu-torch216 CPT, rl-monarch-locality SFT/RL, frameworks/2026.1.0 eval) + PBS tegu:home | `05-infra-rl-storage.md:45-54` + `cpt_fineweb_edu_4n.sh:52-70` + `sft_distill_cot_4n.sh:98-113` + `submit_cot_eval.sh:15-20` |
| CPT qsub chain auto-resume via CKPT_DIR | `torchtitan/experiments/ezpz/scripts/cpt_fineweb_edu_4n.sh:12,81-103` + `SYNTHESIS.md:181` |
| RL PBS script host routing, VENV dispatch, STEPS override | `torchtitan/experiments/ezpz/rl/scripts/grpo/multihost_grpo_e2e.pbs:1-164` |

**No invented gate values.** Every threshold (HS 0.560->0.540, ARC-E 0.651->0.621, PIQA 0.733->0.713, ARC-C 0.31->0.290, MMLU 0.241-0.266->0.240, CoT 0.205->0.265, format 0.985->0.90, gen_len 282->550, unclosed ~0->10) is verbatim from `SYNTHESIS.md:229-239`. Every cost (2.5 hops 4N/12h, 1.0 N-hour 7-config sweep, 2N smoke 5 min / 4N full 1.5h, 2N Monarch 30 min/100 steps) is verbatim from `SYNTHESIS.md:40-48`. Every near-miss case (G1-G2 fail but G4 pass => continue; G1-G2 pass G4 fail => stop fork; all fail => revert; G4 missing => continue) is verbatim from `SYNTHESIS.md:240-248`.
