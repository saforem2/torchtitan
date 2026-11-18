# Eval Harness Robustness and Scoring Strategy for agpt-2B Post-Training

> **Thread C — Eval Harness.** Deep analysis on scoring robustness, benchmark
> selection, statistical power, and a concrete eval plan checkpoint x benchmark.
> Covers all five assigned questions.

---

## 0. Scope and Method

This document is a deep analysis of eval-harness design for agpt-2B
post-training on Sunspot/Aurora. Every claim is grounded in artifacts already
on disk or jobs whose stdout/stderr live under
`/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan` or
`/flare/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz` (see
`docs/evals/agpt/2b/README.md`, `docs/evals/mmlu-letter-prior-at-chance.md`,
`docs/production/POST-TRAINING-2B.md`, `scripts/eval/eval_cot_gsm8k.py`,
`scripts/eval/submit_cot_eval.sh`, and the 200-problem GSM8K-CoT evals at
`outputs/evals/cot/`).

Primary sources cited use `CONFIRMED` / `INTERNAL` tagging consistent with
`docs/evals/eval-landscape-2026-07.md`.

---

## 1. The MMLU Letter-Prior Problem

### 1.1 What "at chance" really means for letter-scored MMLU

MMLU as we run it (`lm-eval-harness` task `mmlu`, `num_fewshot=5`,
`device=xpu:0`, HF backend) is **MCF** — multiple-choice letter scoring:

- Prompt: question + four titled choices + few-shot exemplars, then
  `doc_to_choice: ["A","B","C","D"]`.
- Score: the continuation whose single-letter token has highest loglikelihood
  (one of four tokens `_A`=586 / `_B`=599 / `_C`=585 / `_D`=608 for the
  Gemma tokenizer; shared `tokenizer.json` md5 `b60c6af2...`, vocab 256k
  — verified in `mmlu-letter-prior-at-chance.md#why-this-is-not-an-eval-bug`).

The null for this task is **not 0.25**. The answer key of `cais/mmlu`
test split (14,042 questions, `scripts/eval/mmlu_letter_baseline.py`) is
non-uniform:

| strategy     | score |
|------------- |------:|
| always A     | 0.2295 |
| always B     | 0.2465 |
| always C     | 0.2551 |
| **always D** | **0.2689** |

So "chance" for a letter-guessing model is a **band 0.2295–0.2689** — wider
than the nominal 0.25 and wider than the ~0.0036 binomial SE at N=14k.
Every agpt-2B MMLU number on record lands inside it (`INTERNAL`):

| model | tokens | MMLU (5-sh MCF) |
|-------|-------:|----------------:|
| flagship 2B step-46429 | 4.674T | 0.2579 (= group key 0.25787±0.00369; leaf macro 0.2631, §2.3 of that doc) |
| MDS 2B pre-torchtitan | 7.064T | 0.2413 |
| 20B-256 | 0.39T | 0.2599 |
| 20B-512 | 0.71T | 0.2655 |
| mix arm `edu100`/`owm100`/`owm75` | ~600B+10B | 0.2476 / 0.2482 / 0.2393 |
| **always D** | — | **0.2689** |

Twelve configurations over a ~20x token span, two model scales, and five data
treatments, plus the most-trained model scoring *lowest* — **no ordering among
them is meaningful** on MCF alone. The harness control rules out plumbing
bugs: Llama-3.2-1B = **0.3121** and Llama-3.1-8B = **0.6530** through the same
`simple_evaluate(num_fewshot=5, device="xpu:0")` (job 8736838, `CONFIRMED`).

### 1.2 Letter-prior model collapses the "signal"

Fitting the question-blind model

```
acc(subject) = sum_L p_L * keyfreq_L(subject)
```

recovers a fixed prior `p(A,B,C,D) ≈ (0.37, 0.32, 0.20, 0.11)` for all arms
except `owm75` at `(0.57, 0.34, 0.03, 0.06)`, and collapses residual
over-dispersion to the binomial floor (variance/expected 1.69→1.14, 1.90→1.26,
and **0.88** for `owm75`). Per-subject accuracy correlates
**r=+0.64 to +0.76** with answer-key letter frequency; the "strong" subjects
(virology, human_aging) are A/B-heavy, the "weak" ones
(professional_medicine 0.45 D, high_school_statistics 0.47 D) are D-heavy.
Full tables and cross-subject chi2/df = 1.74/2.17/2.24 in
`mmlu-letter-prior-at-chance.md#the-trap-in-the-per-subject-spread`.

**Do not interpret per-subject structure as knowledge without fitting the
letter model first.** The 30B-ladder fan-out that was once read as "not the
shape of chance" (subjects above 50%: 24→23→31, max 0.304→0.423) has exactly
the same shape — a letter-prior/kay-distributions interaction — and was
explicitly retracted in that doc (§"Recorded as an error made…").

### 1.3 Why format, not knowledge, is the bottleneck — and why the cheap cloze fix did NOT rescue the checkpoint we tried it on

Two literature baselines both predict MCF at chance below ~1T tokens while
**CF** (cloze: `doc_to_choice: "{{choices}}"` scoring answer *text*, comparable
to our ARC/HellaSwag setup) already has signal:

- **SmolLM2** (§4.3, Fig 6): above-chance cloze while MCF only clears after
  **6T** on the letter formulation; their own MCF = 29.62 across 0–6T
  (`CONFIRMED` arXiv:2502.02737).
- **OLMES** on OLMo-7B: *"Around 400B tokens, MCF random; prior to that CF
  has signal"*; recommends `max(CF, MCF)` (`CONFIRMED` arXiv:2406.08446).

So MMLU failure on our checkpoints is *consistent with* knowledge-present but
letter-binding-absent — the exact agpt-2B profile (healthy commonsense:
HellaSwag ~0.57, ARC-Easy ~0.66 at same checkpoints). However:

> **Job 12474382 is the result that blocks the cheap reading.** MCF vs CF on
> the *same* 30B step-2000 checkpoint (3.93B tok), 5-shot, `--limit 10`
> (= 570 q/arm) with only `doc_to_choice` varying:
>
> | format | task | macro | above chance |
> |--------|------|------:|------------:|
> | MCF (letters) | `mmlu` | 0.2281 | 21/57 |
> | CF (cloze) | `mmlu_continuation` | 0.2386 | 24/57 |
> | CF − MCF | | **+0.0105** | SE **0.0256** |

Not significant; CF does not clear chance where MCF does not at this
resolution. Caveats that matter more than the p-value:

1. **Token count:** step-2000 = **3.93B tokens** — arguably too early for
   either format to show knowledge. Weakens the null.
2. **Subsample noise:** the same ckpt/shot scored 0.2281 at `--limit 10`
   and **0.2561 at `--limit 40`** (2,280 q) — **0.028 sampling spread**.
   Neither is publishable; the limit-10 pair is internally consistent, which
   is what the *gap* test needs but not what a headline MMLU needs.
3. **Letter-prior fit breaks on subsamples:** A-weighted at limit-10,
   D-weighted at limit-40, R² 0.04–0.11 — neither can test the letter
   hypothesis; only full-size runs can.

Implication: the format explanation is the *live* alternative — no CPT
checkpoint has ever been scored on MMLU in either format — but it is not the
default explanation for our MMLU-at-chance signal until tested on a
well-trained checkpoint (flagship 2B step-46,429 at 4.674T on Aurora).

### 1.4 Data composition is the other, complementary mechanism — Dolmino is not the lever

Measured over the data-lists (file-weighted, renormalized):

| corpus | olmo-mix-1124 | dolmino-mix-1124 |
|--------|:---:|:---:|
| dclm | **94.80%** | **89.18%** |
| pes2o | 1.47% | 6.88% |
| flan | — | 1.97% |
| math | 0.32% | 1.36% |
| wiki | 0.09% | 0.44% |
| starcoder | 2.46% | — |
| arxiv | 0.53% | — |

Dolmino is the same DCLM-dominated web distribution with ~5.6pp
redistributed into peS2o + 2% FLAN — a **change of degree**. By contrast,
SmolLM2 stage-1 is **60% FineWeb-Edu / 40% DCLM** — a *change of kind*:
majority educational content. The FineWeb-Edu paper (arXiv:2406.17557)
reports **33.6% MMLU at 38B tokens** where next-best needs 300B (~8× token
efficiency), and the ablation FineWeb-Edu 37.5 vs DCLM 35.5 on MMLU with DCLM
62.3 vs 60.1 on HellaSwag — exactly our split (commonsense intact, MMLU
down).

The OLMo-2 precedent on this *exact* base (`olmo-mix-1124`): Dolmino
mid-training **MMLU 26.9 → 44.3 (+17.4) at 1B** (Table 9, arXiv:2501.00656)
but only +3.9/+4.1 at 7B/13B. At small scale mid-training is nearly the
entire MMLU signal — yet our torchtitan CPT sweep (300B pilots
dolmino-100 and olmo50-dolmino50 at step-5960) **degraded commonsense**
(HellaSwag 0.560→0.486/0.491, ARC-Easy 0.651→0.547/0.610) despite beating the
plateau on loss (2.80→2.49/2.60; §6 below), and the MDS-continuation arm
scored **0.2413 — the lowest MMLU in the table**. See `docs/production/cpt/README.md`
and `docs/production/sft/fineweb-edu-60-40-kind-shift.md`.

### 1.5 Scoring recommendation: `max(CF, MCF)` with an explicit chance band

**Single number (`MMLU` from lm-eval): do NOT use it as a gate.** The
0.2413–0.2655 band sits inside the constant-letter ceiling (0.2689) and
movements inside it are letter-prior noise — per §"Gating rule for agpt-2B"
in `mmlu-letter-prior-gating.md`. The 30B step-2000 cloze run not clearing
chance at 3.93B does not overturn this reading; it only shows that at *that*
checkpoint and N, CF-vs-MCF is unresolved.

Recommended reporting discipline (drawn from `mmlu-letter-prior-at-chance.md#how-to-report-mmlu-from-here`
and `mmlu-letter-prior-gating.md#how-to-report-mmlu-from-here`):

| Rule | What | Why |
|------|------|-----|
| 1. Read the **group key** `mmlu` from `results.json`, not a hand-rolled macro over `mmlu_*` | hand-rolled macro double-counts four category rollups → +0.005 high (`INTERNAL`: step-46429 macro **0.2631** vs group **0.2579**, artifact at `/flare/AuroraGPT/.../step-46429/.../results.json`) |
| 2. **State the format** (`MCF` vs `CF = mmlu_continuation`) for every number; when possible report **both** and their **`max(CF, MCF)`** per **OLMES** (arXiv:2406.08446) | below ~1T (and on our MC-trained data) MCF alone is close to uninformative; SmolLM2's own MCF 29.62 across 0–6T is on the same formulation |
| 3. **State shot count** (`n-shot`) from `results.json` (persisted since commit `5ab9e50f6`; older artifacts — e.g. step-46429 — omit it) | MMLU is not 0-shot: per-task defaults are `None` and the run must overwrite to 5 (see `eval-landscape-2026-07.md#1.1` few-shot gotcha). Silent 0-shot is a distinct failure mode (flagged in `dead-cli-flags-in-repo-root-pbs.md`) |
| 4. **State the constant-letter band** 0.2295–0.2689 (and which constant letter attains the top) next to every new MMLU number | a reader can see immediately whether a score clears chance **after** key-skew; otherwise the ~0.04-wide skew band masquerades as sampling noise |
| 5. **Do not compare to SmolLM3's 44.1** | That is **MMLU-CF** (contamination-free variant, arXiv:2412.15194) — same acronym, different task. Compare to **OLMo-2** (same data family; token count stated; `CONFIRMED` 63.7/67.5 at 7B/13B) — see `eval-landscape-2026-07.md#3b` caveats |
| 6. Score **every CPT checkpoint** on MMLU in **both** formats (eval-only on existing ckpts) — currently 0 done | dolmino's published MMLU effect (+17.4) has never been measured here because no CPT ckpt was scored on it; whether OLMo-2's gain was MCF/max/CF is also unsettled — cost is eval-time only |
| 7. When a new base clears the band, **re-fit the letter prior** `acc(subject) ~ sum_L p_L * keyfreq_L(subject)` before declaring "knows more" | a shift in `p(A,B,C,D)` can lift the macro without question-conditional knowledge; only collapsing the residual dispersion (R², residual/var-binomial) to floor proves you moved *off* the prior |

Why this specific combination and not the natural alternatives:

- **"Just report `mmlu` (MCF) and wait for it to clear"** — wrong null. At agpt-2B scale MCF under-reports knowledge by design (SmolLM2/OLMES evidence above), and the non-uniform key makes 0.25 the wrong threshold. You would reject a model that *does* know more but lacks letter binding.
- **"Just report `mmlu_continuation` (CF) instead"** — wrong for a different reason. CF is strictly more lenient (scores answer *text*, not binding) but on the 30B early ckpt it *also* did not clear chance — and CF-vs-MCF on our data is precisely the unresolved question you need the gate to settle. Reporting CF alone would over-claim the easy format.
- **"`max(CF, MCF)`"** is the OLMES compromise built for exactly this regime: before MCF emerges, max ≈ CF; after MCF emerges, max ≈ MCF. It is the only statistic that is monotonic in either component and penalizes pure letter-guessing without penalizing cloze knowledge. Using it does not excuse weak models — it correctly attributes signal to the component that has it.

On agpt-2B the gate should **not** be any MMLU variant until it demonstrably clears the band (see §5.3 decision rule tying this to HellaSwag/ARC-Easy + GSM8K-CoT post-SFT from `mmlu-letter-prior-gating.md#gating-rule-for-agpt-2b`).

---

## 2. CoT Eval (GSM8K-CoT): What the Three Signals Mean and Whether 200 Suffices

### 2.1 The current eval stack — what it does, line by line

Source: `scripts/eval/eval_cot_gsm8k.py` (Stage 0, 235 lines) and
`scripts/eval/submit_cot_eval.sh` (single-tile `workq`, 40 min walltime,
40 GB XPU `level_zero:gpu` fp32 `vLLM==0.26`, 768 `max_tokens`, greedy).

- **Dataset:** `openai/gsm8k` `main` `test` (1,319 problems; `INTERNAL`
  `eval_cot_gsm8k.py:152`). Default `--limit 200` = first-200 slice for the
  headline; `LIMIT=200` in the PBS script.
- **Prompting — hand-serialized, not `apply_chat_template`:**
  `build_prompts()` concatenates Gemma-style turns
  `"<start_of_turn>user\n{question}{_PROMPT_SUFFIX}<end_of_turn>\n<start_of_turn>model\n"`
  where `_PROMPT_SUFFIX` =
  `"\nReason step by step inside <think></think>, then give the final answer inside <answer>\\boxed{}</answer>."`
  This matches Stage-1 SFT exactly. The MDS tokenizer intentionally has **no**
  `chat_template`; delegating to `apply_chat_template` would be wrong.
  A fail-closed tokenizer check (first token == 106, last == 108, has 107)
  catches checkpoint-flavor mismatches.
- **Generation:** `vLLM LLM(dtype=args.dtype, gpu_memory_utilization=0.70, enforce_eager=True, max_model_len=2048)` on one tile (`ZE_AFFINITY_MASK=0`), `SamplingParams(temperature=0.0, max_tokens=768, stop_token_ids=[1, 107])` (EOS + AGPT end-of-turn). Stop list explicitly **does not** include textual `</answer>` — a deliberate choice: stopping on `</answer>` hides malformed continuations and diverges from the production inference contract (comment at `eval_cot_gsm8k.py:171-173`).
- **Scoring — two metrics + two health signals:**
  - `format_hit_rate`: fraction where `_FORMAT_RE = r"<think>.*?</think>\s*<answer>(.*?)</answer>"` matches. Inside the span, prefers `\boxed{...}` else first number; `_norm_num` canonicalizes commass/$/`5.0→5`.
  - Strict fallback when no `<think></answer>` envelope: strip any `<think>` block, then look for `\boxed{}` or `#### N` only (comment: `NO last-number fallback — that would score CoT scratch numbers`). So: `0/200 @ 0.0 fmt` genuinely means *no* well-formed answer.
  - `cot_accuracy`: exact string match of normalized extracted number vs
    normalized `#### N` gold. Reported as headline (e.g. 0.205 = 41/200).
  - `mean_gen_len` and `n_unclosed` (finish_reason == `"length"` — i.e. hitting `max_tokens` without `stop` EOS/107).

This is **not** `lm-eval`'s `gsm8k` at all — that task uses loglikelihood or
5-shot prompting with `#### ` extraction. The whole point of Stage 0 is that our
post-training teaches a *generative* `<think>/<answer>\boxed{}` contract, so
only generative scoring is faithful.

On-CPT-67k evidence: `eval-2b-v2.sh`/`eval-20b-v2.sh` explicitly **defer**
`ifeval`, `bbh`, `gpqa`, `math`, `humaneval` as "instruct-tuned OR much more
training" and warn to avoid `*_instruct`, `gsm8k_cot_llama` (needs chat template),
`truthfulqa_gen` (needs judge) — see `eval-landscape-2026-07.md`.

### 2.2 Raw CPT scores 0/200 (degenerate) and B2's family — reading the 4-tuple

`POST-TRAINING-2B.md` (§2026-11-14, §2026-11-04) reports the full B2/B2v2/B3/B4/CPT
table. Canonical numbers at `temperature 0.0 max_tokens 768 frameworks/2026.1.0 vLLM 0.26 XPU fp32 Sunspot workq`:

| Checkpoint | CoT acc | Format | Gen len | Unclosed | Jobs | Notes |
|------------|--------:|-------:|--------:|---------:|------|-------|
| raw CPT 73k/74k (FWE) | **0.000** | 0.000 | — | **200** | `eval 1248xxxx` | degenerate loops, pre-SFT expected |
| raw CPT 17k (FWE) | 0.000 | 0.000 | — | 200 | `POST-TRAINING-2B.md` 2026-11-12 | pure CPT, no `chat_template` |
| cpt34k-gsm8k-3ep (narrow SFT on CPT) | 0.025 | 0.505 | — | — | `cpt34k-gsm8k-3ep` | format-vs-reasoning gap (1N) |
| cpt34k-distill-broad-4N | 0.000 | — | — | — | parked | broad mix on wrong manifold |
| **B2 = checkpoint-93** (stage-2 on ckpt-900) | **0.205** | **0.985** | **282** | ~0 | `xakno7kg` | best; two-stage (see §2.5) |
| B2v2 ep3 final (6 steps) | 0.09 | 0.86 | 416 | 22 | **12482572** `x1922c4s2b0n0` 00:01:38 Exit 0 | W&B `xakno7kg` loss 0.8069, acc 0.8198 — healthy train |
| B2v2 ep6 final (12 steps) | 0.105 | 0.91 | 378 | 15 | **12482573** `x1922c4s5b0n0` 00:01:39 Exit 0 | W&B `ur6pqqhp` — same |
| B3 (tulu + OpenR1 0.25, 1-shot) | 0.05 | — | — | — | 1 SFT seq 8192 | long-CoT dilution |
| B4a (+B2 finish on B3 base) | 0.02 | 0.26 | — | 133 | `b4-finish-and-reweight/README.md` | cannot un-teach verbosity |
| B4b (reweight 0.40 + length-filter) | 0.065 | 0.86 | 611 | 27 | — | format fixed, acc stuck |

Artifacts: `outputs/evals/cot/gsm8k-cot-b2v2-ep{3,6}-final.jsonl` (243K/235K)
and `submit_cot_eval.sh.o12482572/73` on `/lus/tegu`
(`POST-TRAINING-2B.md` §2026-11-04).

**How to read the 4-tuple:**

- **`format_hit_rate` is a gate, not the goal** — but a necessary one.
  B2 holds 0.985; every failure mode that looked like "maybe reasoning"
  actually started with a format collapse (B4a 0.26) or a verbosity blow-up
  (B2v2 282→416 tokens, 0→22 unclosed). **If `format < 0.80` or `unclosed > 30`, stop —
  do not interpret `acc` as reasoning signal.** The RL reward-hacking episode
  (`POST-TRAINING-2B.md` RL § "Reward-hacking is real") is the live
  counterexample: reward 0.118→0.217 while acc 0.16→0.105, driven by
  `ThinkFormatReward 0.2→0.05` down-weight that removed a guardrail SFT had
  been holding up (format 0.955→0.635; 95% of rollouts had *no* `<think>`,
  20% echoed exemplar `\boxed{5}`).
- **`mean_gen_len` and `n_unclosed` are the *early-warning* signals.**
  When B2v2 degraded, gen_len rose and `n_unclosed` went nonzero *before*
  format fully collapsed; B4b restored format (0.86) and brought verbosity
  to 611 tokens + 27 unclosed — and still at 0.065, proving verbosity was a
  symptom, not the cause (§ Findings worth carrying in POST-TRAINING-2B.md).
  Rule: treat `gen_len` > ~1.5× B2 baseline (282) as a drift flag even when
  `format` is still >0.80.
- **`cot_accuracy` is the headline — but at 0.20 it is *not* "above chance".**
  Unlike MMLU there is no key prior; GSM8K is open-response numeric.
  The baseline for a non-CoT model is **0.00** (it cannot produce the
  envelope), and B2's 0.205 is meaningful *because* it was earned under a
  faithful contract — not because it is "far from 0.25".

### 2.3 Is 200 enough? — Power at 200 vs 500 vs 1000

`eval_cot_gsm8k.py` is generation-based (vLLM decode, not loglikelihood) —
the slow part is walltime on XPU tiles, counted in the hundreds of problems
per 10 min per tile. Sample-size choice is a **decision-boundary question**:
what gap must be reliably detected to make a *ship/don't-ship* call between
two model checkpoints.

Binomial SE at the relevant operating points (p = observed `cot_accuracy`,
`SE = sqrt(p(1-p)/N)`) and the half-width of the two-sided 95% CI
(`±1.96·SE` ≈ `±2·SE`) for a *single* checkpoint:

| N | SE at p=0.05 | SE at p=0.10 | SE at p=0.20 | 95% half-width at 0.20 |
|---:|-------------:|-------------:|-------------:|----------------------:|
|  200 | 0.015 | 0.021 | **0.028** | **±0.055** (0.145–0.255) |
|  500 | 0.010 | 0.013 | **0.018** | **±0.035** (0.165–0.235) |
| 1000 | 0.007 | 0.010 | **0.013** | **±0.025** (0.175–0.225) |

For a *comparison* between two checkpoints (A vs B, same N), the SE of the
*delta* is `sqrt(SE_A² + SE_B²)`; a delta is significant at ~2σ if
`Δ > 2·SE_Δ`:

| comparison | N=200 per arm | N=500 | N=1000 |
|-----------|:---:|:---:|:---:|
| B2 0.205 vs B2v2-ep6 0.105 (Δ=0.10) | 2.5σ **detectable** | 4.0σ | 5.5σ |
| B2 0.205 vs B3 0.05 (Δ=0.155) | 4.1σ **detectable** at 200 | 6+σ | 8+σ |
| B4b 0.065 vs B4a 0.02 (Δ=0.045) | 1.5σ — **borderline at 200** | 2.3σ | 3.3σ |
| B2 0.205 vs 0.215 (+0.01 RL gain, `POST-TRAINING-2B` §RL "drift-proof") | 0.26σ **no** | 0.41σ **no** | 0.56σ **no** |
| Any two models separated by <0.04 | **no** at 200 | **no** at 500 | borderline |

GSM8K also has **correlated item difficulty** (not i.i.d. Bernoulli with
uniform difficulty) — the SE derivation above is slightly optimistic. The
30B MMLU-cloze subsample spread (0.2281 at N=570 vs 0.2561 at N=2280,
Δ=0.028) is the lived warning: *sampling alone* moved the estimate by more
than our claimed "close" thresholds.

Concrete consequences for agpt-2B decisions:

- **200 is sufficient for the decisions we actually face.** B2 vs every
  alternative is a **≥0.10** gap — detectable at >2σ on 200. The RL
  "0.205→0.215" noise is *also* correctly flagged as insignificant at 200;
  upgrading to 500/1000 would not rescue it (0.10 vs 0.01 are both on the
  right side of "don't-ship"). So for **SFT gate/shipping decisions, 200 is
  the correct operating point** — fast, one-tile, 2–3 min per checkpoint
  (see `submit_cot_eval.sh` jobs 12482572/73: 98–99 s each).
- **500/1000 would matter for *ranking* near-tied SFT variants**, e.g.
  B4b vs B3 vs future mid-distribution distills separated by 0.03–0.05.
  The right response there is not "run 1000 on GSM8K" but "add a second
  benchmark to get *orthogonal* signal at the same N" — see §2.4.
- **Reporting hygiene at N=200:** never report the point estimate alone.
  The 200-problem file must carry the 95% binomial CI (Wald or Wilson) and,
  for *comparisons*, the Δ + z-score, consistent with the `summarize()`
  dict shape already written by `eval_cot_gsm8k.py`.

### 2.4 What 200 does *not* cover — and why MATH etc. are not drop-in replacements yet

The eval-landscape decision (2026-07) is explicit:

> **Tier A (ADDED)** `mmlu` 5-shot, `gsm8k` 5-shot, `arc_challenge` 25-shot —
> base-valid at 2B/20B/600B and the peer-facing numbers (`docs/evals/eval-landscape-2026-07.md#2`).
>
> **Tier B (stretch, low expected)** `mmlu_pro 5`, `triviaqa`, `drop`,
> `agieval_en`.
>
> **Tier C (DEFER — instruct-tuned or much more training)** `ifeval`
> (*base scores ~0 — do NOT run on base*, `eval-landscape` Tier C note),
> `bbh`, `gpqa`, `math`, `humaneval/mbpp`, `truthfulqa`.

`MATH` is the obvious next math benchmark — but it is **Tier C for a
reason** on our *base* family: it is substantially harder than GSM8K, the
base-2B solve rate is expected near 0, and a base-2B MATH score gives almost
no gradient with respect to model changes. Running it *post-SFT* is reasonable
once SFT GSM8K-CoT reliably clears ~0.20 — but even then:

- `MATH` needs **generative** scoring (like our `eval_cot_gsm8k.py`), not
  `lm-eval` loglikelihood; the `boxed`-answer scorer must be extended
  (LaTeX normalization: `\frac{}{}`, normalization of `5.0→5` already in
  `_norm_num`).
- **Contamination:** the current distill mix (`OpenThoughts-114k-decontam`)
  was *explicitly* decontaminated against GSM8K-test. Any new benchmark's
  decontamination must be re-audited — see `POST-TRAINING-2B.md` 2026-11-05
  gate (`dataset_info.json` + `[decontam]` counts).
- Statistical power at N=500/1000 MATH would still be worse per walltime than
  GSM8K (longer per problem due to harder reasoning traces).

**Recommendation:** GSM8K-CoT @200 (generative, vLLM fp32) remains the
**primary post-SFT math gate**. Add MATH-level eval **only after** B2's
two-stage ceiling is lifted (pre-tokenized broad distill lands, §5), and
then as a *second column* at 200–500 problems, not a replacement.

---

## 3. Why 0/200 on CPT Checkpoints Is Expected — and How to Gate CPT Properly

### 3.1 0/200 is a format gate, not a reasoning gate

A CPT checkpoint (even one at step-111,200 of the FWE 4N run, loss 12.53,
MFU ~16%) has **never seen** the `<think>/<answer>\boxed{}` contract —
that contract is taught in Stage-1 SFT (`tulu_math_uc_mix_full` →
`checkpoint-900`) and Stage-2 narrow CoT (`gsm8k-r1cot` → `checkpoint-93`).
Therefore:

- `format_hit_rate == 0.000`, `n_unclosed == 200` (degenerate looping) on
  `eval_cot_gsm8k.py` is the **correct null** for a pre-SFT checkpoint.
  Documented explicitly as "expected pre-SFT" in `POST-TRAINING-2B.md`
  2026-11-14 and 2026-11-12 (CPT 67k, 17k, 73k/74k all 0/200), and in the
  `pre-tokenize landed` note (SFT stage-2 blocked until HF conversion exists
  precisely because CPT scoring is not meaningful).
- B2v2/B3/B4 are the controls: the *same* harness on SFT checkpoints
  produces a non-degenerate range 0.02–0.205 with near-perfect format — so
  the harness is not "broken on agpt-2B", pre-SFT scoring is.

Treating `0/200` as a failure would be a category error (like gating 2B-512
promotion on MMLU MCF — see §1.5).

### 3.2 The real CPT question is *latent*: does CPT raise the post-SFT ceiling?

The lever we installed CPT-FWE 4N for is **not "make CPT checkpoints solve
GSM8K"** — it is "raise the **post-SFT** GSM8K-CoT ceiling from 0.20 to
significantly above it". That ceiling was:

- 0.205 under the two-stage recipe (stage-1 → 900 → gsm8k-r1cot → 93).
- Refractory to every single-stage SFT variant (0.02–0.105 for B3/B4/B2v2
  even when verbosity/format were restored — B4b at 0.065 proves format is
  not the bottleneck).

So CPT needs a **two-stage evaluation protocol**:

| Stage | When | What to measure | How | Pass condition |
|-------|------|----------------|-----|---------------|
| **CPT-commonsense gate** | every ~10k CPT steps + at budget | HellaSwag `acc_norm`, ARC-Easy `acc`, PIQA — the *text-scored* commonsense thermometer that actually moves (`docs/evals/agpt/2b/README.md` Fig, 0.27→0.66) | `convert_and_eval.sh` DCP→HF → `lm-eval` HF backend XPU `frameworks/2026.1.0` (same path as the July overlay `all_production_evals.svg`) | **Must not degrade** vs olmo-100 plateau (HS 0.560, AE 0.651, PIQA 0.733) by more than a stated guard (`HellaSwag ≥ 0.55`, `ARC-Easy ≥ 0.64` — §4.3). Dolmino-100/50 already failed this (HS -7.4pp, AE -10.4pp) — see `cpt/README.md` §Downstream eval 2026-07-06 job 8647850. CPT-FWE must clear it. |
| **CPT-knowledge gate** | same checkpoints | MMLU **both formats** at 5-shot + `max(CF, MCF)` (see §1.5) | same `convert_and_eval.sh` path; read group key `mmlu` + `mmlu_continuation`, state n-shot and constant-D band per §1.5. Currently **no CPT ckpt has ever been scored on MMLU** in either format (`docs/production/cpt/README.md` Important) — cost is eval-time only | **Informational until it clears 0.2689**; tracked as the OLMo-2-style +17.4 signal. If either CF or max clears the band, refit the letter prior (§1.5#7) before declaring knowledge. |
| **CPT→SFT ceiling test** (decisive) | at budget only (or one mid-budget anchor) | **Re-run the B2 two-stage SFT verbatim on the new CPT HF** (`checkpoint-900 → gsm8k-r1cot → checkpoint-93` recipe, batch 2/GAS 8/seq 2048/GBS 384, LR 2e-5 cosine, 3 epochs) and score GSM8K-CoT @200 (greedy, fp32 vLLM — `submit_cot_eval.sh`) | TRL `SFTTrainer` FSDP `full_shard` path (`rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh`) with `rl-monarch-torch214-locality-20261001` VENV (per §5 wiring) | **Must beat B2 0.205 by ≥0.05 at ≥2σ on N=200 per arm** (see §2.3 power table: 0.05 gap is 1.7σ at 200 — so in practice ≥0.06 to call at 2σ). Format ≥0.95 and n_unclosed ≤10 required. |
| **CPT→broad-distill ceiling** (stretch, after pre-tokenize) | after broad distill mix lands | `OpenThoughts-114k-decontam × gsm8k-r1cot` pre-tokenized via `_pretokenize_distill_cot_mix_1n.sh` → `/tegu/datasets/agpt2b-distill-cot-mix-len8192` (gate: `dataset_info.json` + decontam counts) | same CoT harness | same thresholds vs the on-olmo broad-distill baseline (once it exists) — isolates CPT gain from broader-data gain |

Why this split: CPT loss alone is **not** a quality gate (see §4.3 and
`cpt/README.md` 300B pilots: loss 2.80→2.49 while HellaSwag -7.4pp).
The commonsense gate catches the Dolmino-style collapse before you pay the
2.391T stage-2 budget. The knowledge gate tracks what OLMo-2's +17.4 MMLU
claims without conflating it with letter-binding. The SFT-ceiling test is the
only one that proves CPT actually helped where B2/B3/B4 are known to stall.

---

## 4. What the Harness Needs Next

### 4.1 Scoring robustness — MMLU

- **Persist `n_shot` and `task_key` in every artifact.** Before `5ab9e50f6`
  the MMLU artifacts (e.g. `step-46429/results.json`) had `n-shot: unrecorded`
  — the shot count was recoverable only from PBS stdout. The contract now is
  that `results.json` carries `n-shot` for each task; enforce it at write time
  and fail eval jobs missing it (abort on silent 0-shot fix `60fe6bb24`).
- **Report both MCF and CF, publish `max(CF, MCF)`.** Wire `mmlu` +
  `mmlu_continuation` into `scripts/eval/eval-{2b,20b}-v2.sh`'s `SHOTS_SPEC`
  loop per the landscape plan (2026-07 modern backfill). The backfill audit
  (2026-09-30) already flags that older 2B MMLU artifacts omit shot metadata
  and must not be silently mixed — **retroactively re-score the flagship
  2B step-46429 and step-92859 checkpoints in both formats** as the first
  MMLU eval job.
- **Report the constant-letter band next to every MMLU number.** Template:
  `MMLU MCF 0.2579 ±0.0037 (chance band 0.2295–0.2689, best constant D)`.
  Readers should never see a bare MMLU without it.
- **Letter-prior diagnostic** — on every new MMLU run, emit `p̂(A,B,C,D)`
  from `Σ_L p_L·keyfreq_L(subject)` and the residual-dispersion ratio
  (`var(residual)/var(binomial)`). `R²` and this ratio tell you instantly
  whether a +1pp bump is letter-prior noise. Script:
  `scripts/eval/mmlu_letter_baseline.py` already computes key frequencies;
  extend it to the per-run fit.
- **Do not fold MMLU into the seven-task common curve.** Per
  `docs/evals/README.md` Coverage audit (2026-09-30): the common overview
  (`all_production_evals.svg`) is intentionally the commonsense thermometer;
  MMLU/GSM8K are *endpoint tables / separate panels* until their coverage is
  dense and shot metadata uniform. Keep that contract.

### 4.2 Scoring robustness — CoT GSM8K

- **Keep `extract_cot_answer` strict** — no last-number fallback. The
  `NORM_NUM` canonicalization (`5.0→5`, commas, $ stripping) is correct and
  matches `gold_answer()` on `#### N`. Adding a fallback would score CoT
  scratch arithmetic, not answer production.
- **Emit per-example `finish_reason`.** Already done (`summarize()`
  counts `n_unclosed` as `length`); add a time-series of `mean_gen_len` and
  `n_unclosed` per checkpoint to the aggregate (`aggregate_evals.py`) so the
  verbosity health signal is plottable alongside accuracy.
- **CI-aware reporting.** At N=200, report `0.205 (41/200, 95% CI 0.149–0.260,
  Wald)` and for comparisons `Δ=0.10, z=2.5, p≈0.01` (exact binomial / normal
  approximation). Do not ship on `acc` alone when Δ is near the
  detectability bound (≈0.05 at N=200 per §2.3).
- **Determinism & reproducibility.** Greedy `temperature=0.0` already pins the
  headline; record `DTYPE=fp32`, `max_tokens=768`,
  `stop_token_ids=[1,107]`, `vLLM 0.26`, `frameworks/2026.1.0`, and
  `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` per `submit_cot_eval.sh` in every
  JSONL header so re-scores are byte-comparable. CPT `0/200` degenerate
  runs are already deterministic loops — nothing else in the harness should
  add variance.

### 4.3 Benchmark coverage — what to add, what to skip, and why

The honest answer is **not "more benchmarks" — it is "the right benchmarks
gated on the right checkpoints"**. Each addition must earn its walltime:

| Benchmark | Where it runs | Shots | Harness | Status | Priority |
|-----------|:---:|:---:|--------|:------:|----------|
| **HellaSwag + ARC-Easy + PIQA + ARC-Challenge** | CPT & SFT checkpoints | 0 / 0 / 0 / 25 | `lm-eval` HF XPU (existing `convert_and_eval.sh`) | **production** | **Must** — CPT health gate and commonsense thermometer; the Dolmino damage was visible here first (HS -7pp, AE -10.4pp `8647850`) |
| **MMLU MCF + MMLU CF (`mmlu_continuation`)** | CPT HF + SFT HF (every CPT checkpoint) | 5 / 5 | `lm-eval` HF XPU, same convert path | **needed, 0 CPT ckpts scored yet** | **Must** — the OLMo-2 +17.4 signal lives here; OLMES `max(CF,MCF)` is the reporting standard (§1.5) |
| **GSM8K-CoT generative** | post-SFT only (B2, B2v2, B3/B4, CPT→SFT) | CoT envelope | `eval_cot_gsm8k.py` vLLM fp32 `submit_cot_eval.sh` (1 tile, 98 s/200) | **production** | **Must** — decisive SFT/CPT→SFT gate; §2.2 table is the reference |
| MATH (generative, `\boxed{}` scorer) | post-SFT only, after broad distill | CoT envelope | extend `eval_cot_gsm8k.py` (`_BOXED_RE` + LaTeX normalize) | DEFER | **When** GSM8K-CoT reliably >0.20 (§2.4) — otherwise near-0 and decontam-unchecked |
| MMLU-Pro 5-shot, TriviaQA, DROP, AGIEval-En | any HF checkpoint | 5 / 0 / 3 / 3–5 | `lm-eval` | Tier B stretch (`eval-landscape-2026-07.md#2`) | Stretch — low expected at 2B (MMLU-Pro needs CF/MCF split too: SmolLM3 19.6 CF / 32.7 MCF) |
| IFEval / BBH / GPQA / HumanEval / MBPP | **instruct only** | — | — | **DEFER** (`eval-landscape` Tier C: base scores ~0, do NOT run on base) | Never gate CPT on these |

Operational constraints worth naming:

- **MMLU on Sunspot XPU OOMs**: `level_zero 40 63G OOM (compute_topk_scores)` is unfixable there; MMLU is **Aurora-only** unless the XPU path is fixed (`POST-TRAINING-2B.md` §Eval infra). CoT GSM8K *does* work on Sunspot (vLLM fp32).
- **Walltime cost**: a 57-subject MMLU pass is ~56k lm-eval requests/step — needs **48h** `workq` (the prior 12h pass died Exit -29 walltime mid-MMLU; `eval-landscape-2026-07.md#Backfill runs`). CoT 200 is one tile 40 min; `MAX_GSM8K_STEPS=200` is sized for the queue.
- **Provenance hygiene**: per `docs/evals/README.md` 2026-09-30 audit, older 2B MMLU artifacts omit `n-shot` and `@5shot` task keys and must not be merged into the modern ladder. Re-score them rather than splicing unknown provenance.

### 4.4 Statistical power at 200 vs 500 vs 1000 — where to spend the walltime

See §2.3 power table for GSM8K-CoT; the same arithmetic holds for MMLU-5-shot
on the letter prior (SE 0.0037 at N=14k — very tight, but MMLU moves are all
inside the 0.04-wide letter-prior band, so *absolute* gaps are uninterpretable
without the prior fit).

**Recommendation for where to spend walltime:**

- **GSM8K-CoT:** 200 per checkpoint for every SFT/CPT→SFT eval; promote to
  **500 only when two post-SFT checkpoints are within 0.05 of each other and
  both have `format ≥ 0.90`/`unclosed ≤ 10`** (i.e., you are ranking, not
  gating). Beyond 500, add a second benchmark column (MATH) rather than more
  GSM8K samples — the marginal information per walltime second is higher.
- **MMLU:** full 14,042 every time (not sampled) — the walltime cost is the
  number of *tasks* (57) not N, so subsampling (limit 10/40) is what produced
  the 0.028 sampling noise in §1.3. If walltime is constrained, cut the number
  of CPT checkpoints scored, not N.
- **Commonsense (HellaSwag/ARC-Easy/PIQA):** 0-shot loglikelihood — cheap.
  Score every CPT checkpoint here; this is the gate that catches recipe bugs
  before you pay the 2.391T stage-2 budget (`docs/production/cpt/README.md`:
  two-phase 300B screen → scale the winner).

---

## 5. Concrete Eval Plan

### 5.1 Checkpoints in scope

Every checkpoint is an **HF export** (`convert_to_hf.py` / `mds_to_hf.py`)
from its DCP source. The live CPT-FWE chain at step 111,200 is not scored
directly — its HF converts at 74k/74.2k and next at ~112k gate are.

| Label | Source | Tokens | Flavor | Exists? | Role |
|-------|--------|-------:|--------|--------:|:-----|
| `gs138650` (= 2B-MDS continuation to 7.77T) | MDS `global_step138650` | 4.674T at stage-1 boundary | MDS | Yes (HF) | **SFT base for B2/B2v2/B3/B4** |
| `2B-v2 step-92859` (256N final) | torchtitan DCP v2 256N | **4.674T** (100%) | v2 | Yes (DCP, some destroyed early; 92859 on disk) | **Alternate SFT base**, commonsense plateau evidence |
| `2B-v2 step-46429` (512N flagship) | torchtitan DCP v2 512N | 4.674T | v2 | Yes (HF + results at `/flare/AuroraGPT/.../step-46429/results.json`) | **MMLU/MCF retro-rescore anchor** (§1.5#2) |
| `CPT dolmino-100 step-5960` | torchtitan CPT pilot | 300B | dolmino-100 | Yes | **Degradation negative control** (`8647850`) |
| `CPT olmo50-dolmino50 step-5960` | torchtitan CPT pilot | 300B | olmo50-dolmino50 | Yes | **Milder degradation control** (held AE ~0.61) |
| `CPT-FWE step-74000/74200` HF | torchtitan CPT-FWE 4N | ~39% of 100B | FineWeb-Edu 60/40 | **Yes (3.7G 9/9 verified, 2026-11-14)** | **First kind-shift CPT→SFT candidate** |
| `CPT-FWE step-~112k` HF (next gate) | torchtitan CPT-FWE 4N | ~58% | FWE 60/40 | Gate — chain at 111.2k/190.7k | **Primary CPT→SFT candidate** |
| `CPT-FWE final ~190k` HF | torchtitan CPT-FWE 4N | 100B (190,734 steps) | FWE 60/40 | Future — chain completion | **Final CPT→SFT candidate** |
| `ckpt-900` = `gs138650/tulu_math_uc_mix_full` | SFT stage-1 | 54B SFT tokens | tulu full | Yes | **B2 stage-1 base; IFEval 0.253** |
| `B2 = checkpoint-93` (on ckpt-900) | SFT stage-2 | 7.4k × 3ep | `gsm8k-r1cot` | **Yes (deliverable, 0.205)** | **Shipping baseline** — beat this |
| `B2v2 ep3/ep6`, `B3`, `B4a/b` | SFT variants | varied | narrow/1-shot | Yes (artifacts + W&B) | **Negative controls** — must stay below ceiling test |
| `CPT-FWE-74k → SFT` (B2 recipe on new base) | SFT on CPT-FWE HF | same 7.4k × 3ep | `gsm8k-r1cot` | **Planned** (after pre-tokenize gate) | **CPT→SFT ceiling test #1** (§3.2) |
| `CPT-FWE-112k/190k → SFT` (same) | SFT on CPT-FWE HF | same | `gsm8k-r1cot` | Planned | **CPT→SFT ceiling #2/3** |
| `broad-distill on each base` | SFT via pre-tok mix | 114k decontam | `OpenThoughts×gsm8k-r1cot` | Parked to 50k per `POST-TRAINING-PLAN.md` | **Stretch** — new CoT data lever (§1.4) |

### 5.2 Checkpoint × Benchmark × Metric × Threshold plan

Order is **gate order** — do not proceed to column 3 without passing column 2.

| # | Checkpoint(s) | Benchmark | Metric | Threshold (pass) | Fallback on fail | Decision |
|---|---------------|-----------|--------|-------------------|------------------|----------|
| 1 | Every CPT-FWE HF (74k, 112k, final) | **HellaSwag** (0-shot) | `acc_norm` | **≥ 0.55** (plateau 0.560 — allow -1pp noise) | Hold — do not promote this CPT HF to SFT; diagnose LR warmup vs data (see `cpt/README.md` LR-shock §) | **Degradation guard** — mirrors dolmino-100 HS 0.486 fail (`8647850`) |
| 2 | Every CPT-FWE HF | **ARC-Easy** (0-shot `acc`) | `acc` | **≥ 0.64** (plateau 0.651) | Same as #1 | Same — olmo50-dolmino50 held 0.61 at 300B; pure FWE without olmo may bleed |
| 3 | Every CPT-FWE HF | **PIQA** (0-shot `acc_norm`) | `acc_norm` | **≥ 0.72** (plateau 0.733) | Same | Weak guard (CPT pilots only -4.6pp here) but cheap |
| 4 | Same | **MMLU 5-shot — both MCF + CF** | `max(CF, MCF)` group key (`mmlu` + `mmlu_continuation`) | **Informational** — record both + max + band + letter prior | — (no gate until §5.3) | **Knowledge lever** — whether OLMo-2 +17.4 reproduces as CF/max even if MCF does not |
| 5 | `gs138650`/`2B-v2 256N` base + both 300B CPT pilots (retro) | **MMLU 5-shot MCF+CF** | `max(CF, MCF)` | — | If CF/max clears 0.2689, refit letter prior; if neither does, confirms the early 30B null at larger scale | **Close the open question** of §1.3 (#6 of §1.5) |
| 6 | **B2 = checkpoint-93** | **GSM8K-CoT generative** | `cot_accuracy` (`N=200`, greedy, fp32) | **0.205 baseline** (41/200; 95% CI 0.149–0.260) + `format ≥ 0.95` + `unclosed ≤ 5` + `gen_len 220–350` | — | **Shipping baseline** — every new SFT is measured against this line |
| 7 | Every new SFT / CPT→SFT | **GSM8K-CoT** | `cot_accuracy` + `format` + `gen_len` + `unclosed` | **Acc ≥ B2 + 0.06** at **2σ** (≈ z ≥ 2 per §2.3; requires 0.205 vs 0.265+ on 200) **and** `format ≥ 0.90` **and** `unclosed ≤ 10` | If format fails: reject. If acc fails but format holds (B4b pattern): recipe is wrong, not data — do not "fix format and re-ship" (B4a refuted) | **Promotion gate** — the only gate that moves the deliverable |
| 8 | `B2-then-CPT→SFT` vs raw `CPT` + vs `B2` | **GSM8K-CoT Δ** | `Δacc` + `z` | **Δacc ≥ 0.05 at ≥2σ** and absolute ≥ 0.20 (not degenerate) | If CPT→SFT ≤ B2 + noise: the FWE kind-shift did not raise the ceiling — pivot to proxy-1B ablations or larger/math-pretrained base per `fineweb-edu-60-40-kind-shift.md#7` | **CPT ceiling test** (§3.2) — the thesis of the entire FWE run |
| 9 | Post-broad-distill on best base | **GSM8K-CoT** | Same | **Acc ≥ best narrow-SFT + 0.06 at 2σ** | If acc flat but gen_len healthy: distill data not additive; inspect decontamination (`dataset_info.json` gate) | Stretch — only after pre-tok `agpt2b-distill-cot-mix-len8192` verified (36,934 packed seqs @8192, 91,481 rows) |
| 10 | Post-broad-distill on best base | **MATH (generative)** | `accuracy` (`N=200`) | Informational until GSM8K-CoT crosses 0.30 — then treat MATH > 0.05 as corroboration | If GSM8K high but MATH ~0: reasoning still narrow | §2.4 — second column, not a replacement |

Scoring contracts implicit in rows 6–10:

- Prompt: hand-serialized Gemma turns (never `apply_chat_template`), `_PROMPT_SUFFIX` verbatim.
- Decode: `vLLM` `float32`, `gpu_memory_utilization 0.70`, `enforce_eager`, `max_model_len 2048`, `SamplingParams(temperature 0.0, max_tokens 768, stop_token_ids [1,107])` — no textual `</answer>` stop.
- Extraction: `_FORMAT_RE` strict + `_BOXED_RE`/`####` fallback only; `_norm_num` canonicalization; no last-number fallback.
- Per-checkpoint output: `summarize()` dict (`n`, `format_hit_rate`, `cot_accuracy`, `mean_gen_len`, `n_unclosed`) + model/dtype/temperature/max_tokens + 95% CI + `finish_reason` series.

CPT rows 1–4 use `convert_and_eval.sh` DCP→HF → `lm-eval 0.4.10` HF backend XPU (with `eval/eval-2b-v2.sh` `SHOTS_SPEC` mixed-few-shot loop; `frameworks/2026.1.0` PTI/CCL/compiler/MKL LD as in `submit_cot_eval.sh`), **not** vLLM.

### 5.3 Decision rule tying the columns together

A new CPT→SFT checkpoint **ships** as the replacement for B2 iff:

1. Its CPT parent **passed the commonsense guard** (#1–3: each ≥ threshold).
2. On GSM8K-CoT it **beats B2 by ≥ 0.06 at ≥ 2σ on 200** (row 7) — so a measured
   gap like 0.205 vs 0.265+ is sufficient to promote; a gap like B2v2 0.205 vs
   0.105 or B4b 0.065 is not.
3. `format ≥ 0.90` and `n_unclosed ≤ 10` and `mean_gen_len` within [220, 550]
   (format is necessary but not sufficient — B4b holds format at 0.86 and
   is still a reject).
4. MMLU `max(CF, MCF)` is recorded with band + prior fit before any external
   claim (row 5). A new base does not need to clear MMLU to ship on CoT, but
   whether it clears the letter band must be stated explicitly — per
   `mmlu-letter-prior-gating.md#gating-rule-for-agpt-2b` this avoids
   conflating letter-prior noise with knowledge.

A checkpoint that **fails gate 7 due to format** is rejected without a rerun —
the RL-reward-hacking record shows that down-weighting format ("it's
saturated") removed the SFT-held guardrail and produced 95% no-`<think>`
rollouts; there is no cheaper fix than the two-stage SFT that earned 0.985 in
the first place. A checkpoint that fails gate 7 **while holding format** is
also rejected — it is in the B4b/B2v2 class where training metrics looked
healthy while eval halved, and extra epochs on the same narrow data only
deepen the loss (see `fineweb-edu-60-40-kind-shift.md#3`).

### 5.4 What not to do

- **Do not gate CPT on GSM8K-CoT `0/200`.** See §3.1 — this is a format gate,
  not a reasoning gate. The CPT→SFT ceiling test (§3.2) is the correct
  substitute.
- **Do not average `mmlu_*` leaf keys into a macro.** It double-counts
  category rollups (+0.005 on step-46429, §1.5#1; `mmlu-letter-prior-at-chance.md#the-flagship-number`).
  Read the group key.
- **Do not merge old 2B-v2 MMLU artifacts into the modern ladder** without
  retro-rescoring. They lack `n-shot` and `@5shot` keys and are not at the
  same provenance (`docs/evals/README.md` 2026-09-30 audit; `docs/evals/agpt/2b/README.md`
  backfill note: early v2 256N 23 ckpts destroyed by stray
  `CKPT_KEEP_LATEST_K=10`, `job 8505252`).
- **Do not subsample MMLU to save walltime.** Subsampling drives the 0.028
  spread already observed (§1.3); cutting the number of CPT checkpoints
  scored (not N) is the correct tradeoff.
- **Do not gate any base checkpoint on IFEval/BBH/GPQA/MATH/HumanEval/TruthfulQA**
  — these are instruct-tuned metrics, `eval-landscape-2026-07.md` Tier C, and
  the base scores ~0 by construction.
- **Do not run GSM8K-CoT at 500/1000 to chase a 0.02 separation.** At that
  delta the correct move is a second orthogonal benchmark, not more samples on
  the same one (§2.3, §4.4). For the actual B2-vs-alternatives gaps (≥0.10),
  200 already separates them.

---

## 6. What the Scorecard Actually Says — and What It Still Doesn't

Putting §1–5 together, the current eval state is:

- **Commonsense (HellaSwag/ARC-Easy/PIQA)** says the base is plateaued
  (0.56/0.65/0.73) and that Dolmino-style CPT on that base degraded it — so
  the next CPT's health gate is non-trivial.
- **MMLU (MCF)** says nothing at agpt-2B scale except "inside the letter
  prior band" — and that statement holds across 12 configurations and 20×
  tokens, so treating it as informative would be overreading. Whether CF or
  `max(CF,MCF)` says something different *at maturity* is exactly what rows
  4–5 of §5.2 must measure on existing checkpoints, without new training.
- **GSM8K-CoT** says two-stage SFT is the only recipe that works (0.205) and
  that every narrow-data variation collapses (0.02–0.105) — with diagnosis
  (format vs verbosity vs pure accuracy stall) already in the 4-tuple
  (§2.2). It also says the pre-SFT 0/200 null is healthy, not a bug.

The **open loop** is whether FineWeb-Edu kind-shift CPT lifts the post-SFT
ceiling that B3/B4/B2v2 could not move — that is row 8 of §5.2, gated behind
pre-tokenize for the broad distill and gated behind the commonsense health
check for the FWE HF itself. Until those two gates clear, the deliverables
remain B2 (CoT, 0.205) and `checkpoint-900` (general instruct, IFEval 0.253)
per `POST-TRAINING-2B.md` §Deliverables.

---

## 7. Operational Appendix — Eval Stack Reference

| Need | Script | PBS | VENV / module | Backend | Notes |
|------|--------|-----|---------------|---------|-------|
| CPT → HF conversion | `scripts/eval/convert_to_hf.py` | part of `convert_and_eval.sh` | `xpu-torch216-sync-5a7a622f16` | DCP → safetensors 3.7G/9 files | HF verified at 74000/74200; next at ~112k |
| Commonsense + MMLU loglikelihood | `scripts/eval/eval-2b-v2.sh` `eval-20b-v2.sh` (with `SHOTS_SPEC` mixed-few-shot) | `eval_mds_sweep.sh`, `eval_mds_stage3_ladder.sh`, `eval-20b-v2.sh` (48h for MMLU) | `frameworks/2026.1.0` `pti-gpu` `xpu-torch216` | `lm-eval 0.4.10` HF backend `device=xpu` | MMLU 56k req/step; do NOT use `ezpz_setup_env` user venv (transformers 5.6.2 breaks lm-eval) per `docs/evals/README.md#Environment` |
| CoT GSM8K generative | `scripts/eval/eval_cot_gsm8k.py` | `scripts/eval/submit_cot_eval.sh` (`qsub -v CKPT=…,LIMIT=200`) 1 tile 40 min | `frameworks/2026.1.0 vLLM 0.26` fp32 (bf16 → gibberish due to vocab 256k) | vLLM XPU 1 tile on `/lus/tegu` | `submit_cot_eval.sh` jobs 12482572/73 98 s each; artifacts → `outputs/evals/cot/*.jsonl` |
| CPT→SFT (ceiling test) | `torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh` | `sft_distill_cot_4n.sh` (4N) | `rl-monarch-torch214-locality-20261001` (not stale `xpu-torch214` — missing `trl`) | TRL `SFTTrainer` FSDP `full_shard`, LR 2e-5 cosine | 1N block uses `ezpz launch` not raw `mpiexec` (oneCCL); see `POST-TRAINING-2B.md` §SFT labels fix |
| Pre-tokenize (broad distill) | `scripts/_pretokenize_distill_cot_mix_1n.sh` | 1N `rl-monarch-torch214-locality-20261001` | Grain datasets | HF arrow cache → `/tegu/datasets/agpt2b-distill-cot-mix-len8192` | Gate: `dataset_info.json` + `[decontam]` counts; `dataset_num_proc 32→8` (OOM fix) |

---

## Cross-Refs

- Full MMLU audit (fitted priors, per-subject correlations, cloze experiment,
  reporting checklist): `docs/evals/mmlu-letter-prior-at-chance.md`
- Companion gating note (constant-D 0.2689 > all 2B; do not gate on MMLU):
  `docs/evals/mmlu-letter-prior-gating.md`
- Eval strategy (Tier A/B/C, `SHOTS_SPEC` few-shot gotcha, peer numbers):
  `docs/evals/eval-landscape-2026-07.md`
- Production eval overlay (all trajectories vs tokens, 4 panels):
  `docs/evals/README.md` + `docs/evals/agpt/2b/README.md`
- CPT pilots + LR-shock hypothesis + umbrella handoff:
  `docs/production/cpt/README.md`
- Kind-shift rationale + FWE data availability + narrow-SFT ceiling:
  `docs/production/sft/fineweb-edu-60-40-kind-shift.md`
- Post-training status + B2/B2v2/B3/B4/RL deliverables + SFT VENV map:
  `docs/production/POST-TRAINING-2B.md`
- Eval scripts: `scripts/eval/eval_cot_gsm8k.py`,
  `scripts/eval/submit_cot_eval.sh`, `scripts/eval/mmlu_letter_baseline.py`,
  `scripts/eval/eval-2b-v2.sh`
- RL reward-hacking / Monarch wiring (XPU `float32` landmine):
  `docs/production/rl/monarch.md`, `docs/production/rl/trl.md`,
  `docs/production/POST-TRAINING-2B.md#RL`
