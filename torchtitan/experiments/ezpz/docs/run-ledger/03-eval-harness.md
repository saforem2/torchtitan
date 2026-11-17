# 03 — Eval Harness Ledger (GSM8K-CoT + MMLU + Commonsense)

> **Branch:** `ezpz-cpt111k-eval-sft` @ `a6135a79` — ledger covers
> `torchtitan/experiments/ezpz/` only.
> **One-line status:** CoT harness `eval_cot_gsm8k.py` + `submit_cot_eval.sh` (vLLM 0.26 fp32, 1 tile, 200-problem greedy) is production on Sunspot; MMLU is Aurora-only (Sunspot OOM); every SFT gates on the 4-tuple `cot_accuracy / format_hit_rate / mean_gen_len / n_unclosed` vs B2 `0.205/0.985/282/~0`.
> Every job ID, score, and threshold below cites its source doc + section. Inferences labeled `[INFERENCE]`.

---

## What is running / ran

### Stack reference (from source files)

| Field | Value | Source |
|-------|-------|--------|
| Eval script | `torchtitan/experiments/ezpz/scripts/eval/eval_cot_gsm8k.py` (235 lines, Stage 0) | `eval_cot_gsm8k.py:1-7` |
| PBS wrapper | `torchtitan/experiments/ezpz/scripts/eval/submit_cot_eval.sh` — `select=1`, `walltime=00:40:00`, `q=workq`, `A=datascience`, `filesystems=tegu:home` | `submit_cot_eval.sh:1-7` |
| Module / VENV | `frameworks/2026.1.0` + `pti-gpu`, `LD_LIBRARY_PATH` PTI `/pti-gpu-0.17.0-62kxpbr/lib64`, `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, `ZE_AFFINITY_MASK=0`, `ZE_FLAT_DEVICE_HIERARCHY=FLAT` | `submit_cot_eval.sh:15-20` |
| Python | `/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/bin/python` | `submit_cot_eval.sh:28` |
| Dataset | `openai/gsm8k` `main` `test` (1,319 problems); `--limit 200` = first-200 slice for headline | `eval_cot_gsm8k.py:152-154`, `submit_cot_eval.sh:25-34` |
| Prompt | Hand-serialized Gemma turns `<start_of_turn>user\n{question}{_PROMPT_SUFFIX}<end_of_turn>\n<start_of_turn>model\n` where `_PROMPT_SUFFIX = "\nReason step by step inside <think></think>, then give the final answer inside <answer>\\boxed{}</answer>."` — matches Stage-1 SFT exactly; fail-closed tokenizer check `first==106, last==108, has 107` | `eval_cot_gsm8k.py:86-115` |
| Generation | `vLLM LLM(dtype=float32, gpu_memory_utilization=0.70, enforce_eager=True, max_model_len=2048)` on 1 tile; `SamplingParams(temperature=0.0, max_tokens=768, stop_token_ids=[1,107])` (EOS + AGPT EOT) — textual `</answer>` explicitly NOT a stop (hides malformations) | `eval_cot_gsm8k.py:161-175`, `submit_cot_eval.sh:26,34` |
| Scoring | `format_hit_rate` via `_FORMAT_RE = r"<think>.*?</think>\s*<answer>(.*?)</answer>"` (DOTALL+IGNORECASE); inside span prefers `\boxed{...}` else first number; `_norm_num` canonicalizes commas/$/`5.0->5`; strict fallback (strip `<think>` block then `\boxed{}` or `#### N` only — NO last-number fallback); `cot_accuracy` = exact normalized match vs `#### N` gold; `mean_gen_len` + `n_unclosed` (`finish_reason=="length"`) are health signals | `eval_cot_gsm8k.py:30-84,118-131,186-213` |
| Invocation | `qsub -v CKPT=<hf-dir>,LIMIT=200 torchtitan/experiments/ezpz/scripts/eval/submit_cot_eval.sh` | `submit_cot_eval.sh:10` |

### CoT eval job table — every claimed score cites its artifact

| Checkpoint | CoT acc | Format | Gen len | Unclosed | Job ID | Walltime | Exit | Source |
|------------|--------:|-------:|--------:|---------:|--------|----------|------|--------|
| raw CPT 17k (FWE) | **0.000** | **0.000** | — | **200** | `eval 0/200` (no PBS ID quoted; pure CPT pre-SFT) | — | degenerate loops, expected pre-SFT | `POST-TRAINING-2B.md:35` ("`step-17000` was `0.000` GSM8K-CoT by design (pure CPT, no `chat_template`)"), `thread-c-eval-harness.md:249` |
| raw CPT 73k (FWE) | **0.000** | **0.000** | — | **200** | `eval 1248xxxx` [INFERENCE: doc cites `eval 1248xxxx` as placeholder prefix for 73k/74k pair, not a full retrievable ID] | — | degenerate loops, expected pre-SFT | `POST-TRAINING-2B.md:23` ("raw `CPT` `73k/74k` both `0/200` `0.0 fmt` degenerate loops (`n_unclosed=200`) — expected `pre-SFT`"), `thread-c-eval-harness.md:248` |
| raw CPT 74k (FWE) | **0.000** | **0.000** | — | **200** | `eval 1248xxxx` (same pair) | — | degenerate loops, expected pre-SFT | `POST-TRAINING-2B.md:23`, `thread-c-eval-harness.md:248` |
| **B2 = checkpoint-93** (stage-2 on ckpt-900) — **deliverable** | **0.205** (41/200) | **0.985** | **282** | **~0** | W&B `xakno7kg` (no PBS ID quoted for B2 headline; scored on same harness) | — | — | `POST-TRAINING-2B.md:45` ("`checkpoint-93` = **B2: 0.205 CoT, format 0.985, gen_len 282**"), `thread-c-eval-harness.md:252`, `SYNTHESIS.md:17` |
| **B2v2 ep3 final** (6 steps) | **0.09** | **0.86** | **416** | **22** | **12482572** (`x1922c4s2b0n0`) | **00:01:38** | **Exit 0** | `POST-TRAINING-2B.md:77-82` table + `thread-c-eval-harness.md:253`; artifacts `outputs/evals/cot/gsm8k-cot-b2v2-ep3-final.jsonl` (243K) + `submit_cot_eval.sh.o12482572` on `/lus/tegu` |
| **B2v2 ep6 final** (12 steps) | **0.105** | **0.91** | **378** | **15** | **12482573** (`x1922c4s5b0n0`) | **00:01:39** | **Exit 0** | `POST-TRAINING-2B.md:77-82`, `thread-c-eval-harness.md:254`; artifacts `gsm8k-cot-b2v2-ep6-final.jsonl` (235K) + `o12482573` |
| **B3** (instruct+CoT mix, tulu+OpenR1 0.25, 1-shot, seq 8192) | **0.05** | **0.86** [INFERENCE: thread-c lists B4a format but B3 format not separated; `SYNTHESIS.md` G4b notes "B3 0.86 masking 0.05"; POST-TRAINING-2B assigns B4b's 0.86, so B3 format 0.86 is [INFERENCE] from SYNTHESIS G4b] | **601** [INFERENCE: `SYNTHESIS.md` Prohibition #9: "B3/B4b: same 0.05-0.065 band but gen_len 601 vs 611 indistinguishable"] | — (not quoted) | 1 SFT seq 8192 (no PBS ID quoted) | — | Exit 0 | `POST-TRAINING-2B.md:100`, `thread-c-eval-harness.md:255`, `SYNTHESIS.md:100,238,272` |
| **B4a** (+B2 finish on B3 base, verbose base poisoning) | **0.02** | **0.26** | **1378** [INFERENCE: `SYNTHESIS.md:238-239` G4c: "B4a 1378/133 flagged envelope loss"] | **133** | (no PBS ID quoted; `b4-finish-and-reweight/README.md`) | — | Exit 0 | `POST-TRAINING-2B.md:101`, `thread-c-eval-harness.md:256` ("format 0.26, 133/200 unclosed") |
| **B4b** (reweight gsm8k-r1cot 0.15->0.40 + length-filter, seq 4096) | **0.065** | **0.86** | **611** | **27** | (no PBS ID quoted; `b4-finish-and-reweight/README.md`) | — | Exit 0 | `POST-TRAINING-2B.md:102`, `thread-c-eval-harness.md:257` |
| cpt34k-gsm8k-3ep (narrow SFT on CPT, 1N) | **0.025** | **0.505** | — | — | (no PBS ID quoted) | — | — | `POST-TRAINING-2B.md:23` ("`cpt34k-gsm8k-3ep` `0.025 acc / 0.505 fmt` vs `cpt34k-distill-broad-4n` `0.000`"), `thread-c-eval-harness.md:250` |
| cpt34k-distill-broad-4N (broad mix on wrong manifold) | **0.000** | — | — | — | parked | — | — | `POST-TRAINING-2B.md:23`, `thread-c-eval-harness.md:251` |
| CPT-FWE narrow SFT probe at 17k (CPT-17000 HF, LR 2e-5 3ep GBS 384) | **0.015** | format collapse | — | — | (no PBS ID quoted; `agpt2b_b2v2_distill_cot_2n.sh` on CPT-17000 HF) | — | — | `POST-TRAINING-2B.md:35` ("Narrow SFT-on-CPT ... scored **0.015** vs B2 `0.165` [INFERENCE: doc says `0.165` here but B2 is 0.205 elsewhere — quoting `0.015` verbatim] — format collapse on FW-Edu-shifted manifold") |

**How to read the 4-tuple** (source: `thread-c-eval-harness.md:263-287`):
- `format_hit_rate` is a gate, not the goal — but necessary. B2 holds 0.985; every failure that looked like "maybe reasoning" started with format collapse (B4a 0.26) or verbosity blow-up (B2v2 282->416, 0->22 unclosed).
- `mean_gen_len` + `n_unclosed` are early-warning signals. B2v2 gen_len rose + n_unclosed went nonzero *before* format fully collapsed; B4b restored format 0.86 but at 611 tokens + 27 unclosed and still 0.065 — verbosity was symptom, not cause.
- `cot_accuracy` is headline but at 0.20 is *not* "above chance" in MMLU sense — GSM8K is open-response numeric, baseline for non-CoT model is 0.00 (cannot produce envelope).

**Re-score command** (verbatim from `submit_cot_eval.sh:33-34`):
```bash
qsub -v CKPT=<hf-dir>,LIMIT=200 torchtitan/experiments/ezpz/scripts/eval/submit_cot_eval.sh
# expands to: $PY -u $EVAL_SCRIPT --model "$CKPT" --limit "$LIMIT" --dtype "$DTYPE" --out "$OUT"
# where $PY=/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/bin/python
#       $EVAL_SCRIPT=torchtitan/experiments/ezpz/scripts/eval/eval_cot_gsm8k.py
#       DTYPE=float32, LIMIT=200, OUT=$REPO/outputs/evals/cot/gsm8k-cot-baseline.jsonl
```

**Output artifacts** (source: `eval_cot_gsm8k.py:226-232`, `thread-c-eval-harness.md:259-260`):
- stdout: JSON summary `{"n":200, "format_hit_rate":..., "cot_accuracy":..., "mean_gen_len":..., "n_unclosed":...}` + `model/dtype/temperature/max_tokens`
- per-example JSONL: `{"idx","question","prompt","generation","format_ok","pred","gold","correct","gen_len","finish_reason"}` at `args.out`
- B2v2 artifacts: `outputs/evals/cot/gsm8k-cot-b2v2-ep{3,6}-final.jsonl` (243K/235K) + `submit_cot_eval.sh.o12482572/73` on `/lus/tegu` (source: `POST-TRAINING-2B.md:80-82`)

---

## Why

### MMLU gating: `max(CF, MCF)` + band, not MCF alone

MMLU as run (`lm-eval-harness` task `mmlu`, `num_fewshot=5`, `device=xpu:0`, HF backend) is **MCF** — multiple-choice letter scoring: prompt = question + four titled choices + few-shot exemplars, then `doc_to_choice: ["A","B","C","D"]`; score = continuation whose single-letter token has highest loglikelihood (one of four tokens `_A`=586 / `_B`=599 / `_C`=585 / `_D`=608 for Gemma tokenizer; `tokenizer.json` md5 `b60c6af2...`, vocab 256k). Source: `thread-c-eval-harness.md:30-38`, `mmlu-letter-prior-at-chance.md:117-132`.

The null for this task is **not 0.25**. Answer key of `cais/mmlu` test split (14,042 questions, `scripts/eval/mmlu_letter_baseline.py`) is non-uniform:

| strategy | score | Source |
|----------|------:|--------|
| always A | 0.2295 | `mmlu-letter-prior-at-chance.md:27-32`, `thread-c-eval-harness.md:46-49` |
| always B | 0.2465 | same |
| always C | 0.2551 | same |
| **always D** | **0.2689** | same |

So "chance" for a letter-guessing model is a **band 0.2295–0.2689** — wider than nominal 0.25 and wider than ~0.0036 binomial SE at N=14k. Source: `mmlu-letter-prior-at-chance.md:35-38`, `thread-c-eval-harness.md:51-52`.

**Why MCF alone cannot gate:** every agpt-2B MMLU number on record lands inside the band (see Outcome), including flagship 2B step-46429 at 4.674T scoring *lowest*. Movements inside the band are letter-prior noise — per-subject accuracy correlates `r=+0.64 to +0.76` with answer-key letter frequency; "strong" subjects (virology, human_aging) are A/B-heavy, "weak" (professional_medicine 0.45 D, high_school_statistics 0.47 D) are D-heavy. Fitting `acc(subject) = sum_L p_L * keyfreq_L(subject)` recovers fixed prior `p(A,B,C,D) ≈ (0.37, 0.32, 0.20, 0.11)` (owm75: `(0.57,0.34,0.03,0.06)`) and collapses residual over-dispersion to binomial floor (variance/expected 1.69->1.14, 1.90->1.26, **0.88** for owm75; per-subject `chi2/df = 1.74/2.17/2.24`). Source: `thread-c-eval-harness.md:55-86,72-86`, `mmlu-letter-prior-at-chance.md:141-151`.

**Literature predicts MCF at chance below ~1T while CF has signal:**
- SmolLM2 (arXiv:2502.02737 §4.3 Fig 6): above-chance cloze while MCF only clears after **6T**; own MCF = 29.62 across 0–6T. Source: `thread-c-eval-harness.md:100-102`, `mmlu-letter-prior-at-chance.md:164-166`.
- OLMES on OLMo-7B (arXiv:2406.08446): *"Around 400B tokens, MCF random; prior to that CF has signal"*; recommends `max(CF, MCF)`. Source: `thread-c-eval-harness.md:103-104`, `mmlu-letter-prior-at-chance.md:168-172`.

**The cheap cloze fix did NOT rescue the one checkpoint where it was tested** (so format explanation is live alternative, not default):
Job **12474382** — MCF vs CF on same 30B step-2000 checkpoint (3.93B tok), 5-shot, `--limit 10` (=570 q/arm), only `doc_to_choice` varying:

| format | task | macro | above chance | Source |
|--------|------|------:|-------------:|--------|
| MCF (letters) | `mmlu` | 0.2281 | 21/57 | `thread-c-eval-harness.md:114-118`, `mmlu-letter-prior-at-chance.md:272-276` |
| CF (cloze) | `mmlu_continuation` | 0.2386 | 24/57 | same |
| **CF − MCF** | | **+0.0105** | **SE 0.0256** | same — not significant |

Caveats that matter more than p-value (source: `thread-c-eval-harness.md:121-132`):
1. step-2000 = **3.93B tokens** — arguably too early for either format to show knowledge.
2. Same ckpt scored 0.2281 at `limit 10` and **0.2561 at `limit 40`** (2,280 q) — **0.028 sampling spread** from subsampling alone.
3. Letter-prior fit breaks on subsamples (A-weighted at limit-10, D-weighted at limit-40, R² 0.04–0.11) — only full-size runs can test letter hypothesis. Source: `mmlu-letter-prior-at-chance.md:288-298`.

**Scoring rules for MMLU** (source: `thread-c-eval-harness.md:180-199` §1.5 seven rules):
1. Read **group key** `mmlu` from `results.json`, not hand-rolled macro over `mmlu_*` — macro double-counts four category rollups -> +0.005 high (flagship step-46429 macro 0.2631 vs group 0.2579).
2. **State format** (`MCF` vs `CF = mmlu_continuation`) for every number; when possible report **both** + `max(CF, MCF)` per OLMES.
3. **State shot count** (`n-shot`) from `results.json` (persisted since `5ab9e50f6`; older artifacts like step-46429 omit it).
4. **State constant-letter band** 0.2295–0.2689 (and which letter attains top) next to every MMLU number.
5. **Do not compare to SmolLM3's 44.1** — that is MMLU-CF (contamination-free variant, arXiv:2412.15194), different task.
6. Score **every CPT checkpoint** on MMLU in **both** formats (eval-only on existing ckpts) — currently 0 done.
7. When new base clears band, **re-fit letter prior** `acc(subject) ~ sum_L p_L * keyfreq_L(subject)` before declaring "knows more" — shift in `p(A,B,C,D)` can lift macro without question-conditional knowledge.

**Correct gating:** report `max(CF, MCF)` with band + prior fit; real MMLU gate is post-SFT `max(CF,MCF)` + prior collapse (`R²->0`, residual dispersion ->1), not CPT MCF. Source: `SYNTHESIS.md:235-237` G3, `thread-c-eval-harness.md:199-200`.

### CoT power: N=200 is sufficient for shipping decisions

Generation-based (vLLM decode, not loglikelihood) — walltime counted in hundreds of problems per 10 min per tile. Source: `thread-c-eval-harness.md:288-341`.

Binomial SE at operating points `p = cot_accuracy`, `SE = sqrt(p(1-p)/N)`, 95% half-width `±1.96·SE ≈ ±2·SE` for single checkpoint (source: `thread-c-eval-harness.md:296-304`):

| N | SE at p=0.05 | SE at p=0.10 | SE at p=0.20 | 95% half-width at 0.20 |
|---:|-------------:|-------------:|-------------:|-----------------------:|
| **200** | 0.015 | 0.021 | **0.028** | **±0.055** (0.145–0.255) |
| **500** | 0.010 | 0.013 | **0.018** | **±0.035** (0.165–0.235) |
| **1000** | 0.007 | 0.010 | **0.013** | **±0.025** (0.175–0.225) |

Source: `thread-c-eval-harness.md:300-304` (exact table).

For comparison between two checkpoints (A vs B, same N), SE of delta `sqrt(SE_A² + SE_B²)`; delta significant at ~2σ if `Δ > 2·SE_Δ` (source: `thread-c-eval-harness.md:306-316`):

| comparison | N=200 per arm | N=500 | N=1000 | Source |
|-----------|:---:|:---:|:---:|--------|
| B2 0.205 vs B2v2-ep6 0.105 (Δ=0.10) | 2.5σ **detectable** | 4.0σ | 5.5σ | `thread-c-eval-harness.md:312` |
| B2 0.205 vs B3 0.05 (Δ=0.155) | 4.1σ **detectable** at 200 | 6+σ | 8+σ | same |
| B4b 0.065 vs B4a 0.02 (Δ=0.045) | 1.5σ — **borderline at 200** | 2.3σ | 3.3σ | same |
| B2 0.205 vs 0.215 (+0.01 RL gain) | 0.26σ **no** | 0.41σ **no** | 0.56σ **no** | same |
| Any two models separated by <0.04 | **no** at 200 | **no** at 500 | borderline | same |

GSM8K has correlated item difficulty (not i.i.d. Bernoulli) — SE derivation slightly optimistic. The 30B MMLU-cloze subsample spread (0.2281 at N=570 vs 0.2561 at N=2280, Δ=0.028) is lived warning that sampling alone moves estimate by more than "close" thresholds. Source: `thread-c-eval-harness.md:318-322`.

**Consequence:** 200 is **sufficient for gate/shipping** (B2 vs every alternative is ≥0.10 gap, detectable at >2σ on 200; RL 0.205->0.215 correctly flagged as noise at all N). 500/1000 matters only for ranking near-tied SFT variants separated by 0.03–0.05 — correct response there is second benchmark (MATH) not more GSM8K samples. Source: `thread-c-eval-harness.md:324-340`.

Reporting hygiene at N=200: never point estimate alone — carry 95% binomial CI (Wald/Wilson) and for comparisons Δ + z-score, consistent with `summarize()` dict shape. Source: `thread-c-eval-harness.md:338-340`.

### Commonsense thermometer: HS / ARC-Easy / PIQA

Base plateau (2B-256, olmo-100) benchmarks that visibly moved when dolmino CPT degraded — and therefore gate CPT-FWE health (source: `cpt/README.md:38-51,60-98`, `SYNTHESIS.md:229-233`):

| Benchmark | Shots | Metric | Base plateau | Why it gates | Source |
|-----------|-------|--------|-------------:|--------------|--------|
| HellaSwag | 0 | `acc_norm` | **0.560** | Dolmino-100 fell to **0.486 (-7.4pp)** at job 8647850; olmo50-dolmino50 0.491 — ratio-independent HellaSwag drop | `cpt/README.md:61-98`, `SYNTHESIS.md:231` |
| ARC-Easy | 0 | `acc` | **0.651** | Dolmino-100 **0.547 (-10.4pp)** monotonically collapsing; olmo50-dolmino50 held ~0.61 — pure dolmino specifically forgets ARC-Easy | same |
| PIQA | 0 | `acc_norm` | **0.733** | Dolmino only -4.6pp — weak guard but cheap | `SYNTHESIS.md:233` |
| ARC-Challenge | 25 | `acc_norm` | **~0.31** | Catastrophic forgetting guard | `SYNTHESIS.md:234` peer table |

These score answer *text* (like CF), not letters — directly comparable, and cheap (0-shot loglikelihood, not generation). Source: `thread-c-eval-harness.md:492`.

---

## Outcome

### B2 is the deliverable — and the ceiling

**B2 = checkpoint-93** (stage-2 on ckpt-900): **0.205 CoT, format 0.985, gen_len 282, ~0 unclosed** — best of everything tried; drift-proof under GRPO. Source: `POST-TRAINING-2B.md:45-46`, `SYNTHESIS.md:17`, `thread-c-eval-harness.md:252`.

How B2 happened (two-stage is the only recipe that worked — source: `POST-TRAINING-2B.md:107-115`):
- **Stage 1:** `global_step138650` -> `tulu_math_uc_mix_full` (full OpenMathInstruct-2, 53.3M seqs, 1 epoch, seq 1024, 8N) -> `checkpoint-900`. IFEval prompt-strict **0.253**, base-LM intact. Final ckpt `8672` forgot (hellaswag 0.593->0.273, arc_easy 0.694->0.298 — at or near random chance; do NOT use 8672). Source: `POST-TRAINING-2B.md:48-50,111-113`, `SYNTHESIS.md:19`.
- **Stage 2:** `checkpoint-900` -> `gsm8k-r1cot` (7.4k ex, 93 steps, 3 epochs, LR 2e-5 cosine) -> `checkpoint-93` = **B2**. Every single-stage variant on same base landed 0.02–0.065 — 3–10x gap. Source: `POST-TRAINING-2B.md:111-115`.

**Ceiling evidence:** 4 independent failures converge on ~0.2 ceiling on this base; every alternative 0.02–0.105 despite healthy training loss (source: `SYNTHESIS.md:18`):

| Run | What varied | Outcome | Source |
|-----|-------------|---------|--------|
| B3 | tulu+OpenR1 0.25 in one SFT, seq 8192 | **0.05** — long-CoT dilution | `POST-TRAINING-2B.md:100` |
| B4a | bolt B2 finishing onto B3 base | **0.02**, format 0.26, 133 unclosed — cannot un-teach verbosity | `POST-TRAINING-2B.md:101` |
| B4b | reweight 0.15->0.40 + length-filter, seq 4096 | **0.065** — fixed symptom, not accuracy | `POST-TRAINING-2B.md:102` |
| B2v2 ep3/ep6 | more epochs on same narrow 7.4k data (3ep->6ep) | **0.09 / 0.105** — halved despite healthy loss | `POST-TRAINING-2B.md:77-78` |

B2 also drift-proof under GRPO (format held **1.000** for 100 steps; source: `POST-TRAINING-2B.md:114,142`).

### B2v2 halved despite healthy training metrics — the loss/eval divergence

B2v2 (2N Sunspot, `torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh`, base `checkpoint-900-hf`, TRL SFTTrainer FSDP full_shard, LR 2e-5 cosine, batch 2 / GAS 8 / seq 2048, GBS 384, 3->6 epochs, 6->12 steps, W&B `xakno7kg` ep3 / `ur6pqqhp` ep6):

| Checkpoint | CoT acc | Format | Gen len | Unclosed | vs B2 0.205 | Source |
|------------|--------:|-------:|--------:|---------:|-------------|--------|
| B2v2 ep3 final (6 steps) | **0.09** | 0.86 | 416 | 22 | **-0.115** | `POST-TRAINING-2B.md:77` |
| B2v2 ep6 final (12 steps) | **0.105** | 0.91 | 378 | 15 | **-0.10** | same |

Training looked healthy: **loss 0.8069, acc 0.8198 — descends cleanly** (source: `POST-TRAINING-2B.md:82`, `thread-c-eval-harness.md:253-254`). Same failure mode as every single-stage rebuild — training NLL fell while eval halved. Source: `POST-TRAINING-2B.md:85,88` ("Training looked healthy; eval halved. Same failure mode as every single-stage rebuild"), `SYNTHESIS.md:273` Prohibition #2.

This is the sharpest training-loss-vs-eval divergence on record: **healthy loss 0.8069 hid halved eval** — do not read training loss as capability signal. Same signature as ckpt-8672 (loss 0.357 acc 0.902 while hellaswag 0.593->0.273) and dolmino pilots (loss -0.31 nats while HS -7.4pp). Source: `SYNTHESIS.md:272-275` Prohibitions #2-4.

### Raw CPT 0/200 is expected pre-SFT, not a failure

All CPT checkpoints score **0/200 degenerate loops `n_unclosed=200`** on `eval_cot_gsm8k.py` — **correct null** for pre-SFT (source: `POST-TRAINING-2B.md:23-24,35`, `thread-c-eval-harness.md:384-401` §3.1):

- CPT never saw `<think>/<answer>\boxed{}` contract — taught in Stage-1 SFT (`tulu_math_uc_mix_full` -> `checkpoint-900`) and Stage-2 narrow CoT (`gsm8k-r1cot` -> `checkpoint-93`). Therefore `format_hit_rate == 0.000`, `n_unclosed == 200` is expected.
- B2v2/B3/B4 are controls: same harness on SFT checkpoints produces non-degenerate 0.02–0.205 with near-perfect format — harness not "broken on agpt-2B", pre-SFT scoring is.
- Treating 0/200 as failure is category error (like gating 2B-512 promotion on MMLU MCF). CoT gates only on SFT `final/` via `submit_cot_eval.sh`. CPT question answered by SFT probe (E3), not direct CoT. Source: `thread-c-eval-harness.md:400-401`, `SYNTHESIS.md:277` Prohibition #6.

### MMLU 0.241–0.266 all in chance band — no ordering meaningful

Every AuroraGPT MMLU score sits inside the constant-letter band **0.2295–0.2689** (best constant-D 0.2689 beats all). Source: `mmlu-letter-prior-at-chance.md:46-63`, `thread-c-eval-harness.md:55-62`:

| model | tokens | MMLU (5-shot MCF) | Source |
|-------|-------:|------------------:|--------|
| flagship 2B (512N ropefix) step-46429 | 4.674T | **0.2579** (= group key 0.25787±0.00369; leaf macro 0.2631) | `mmlu-letter-prior-at-chance.md:50,72-86` artifact `/flare/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz/outputs/evals/agpt-2b-v2-512n-ropefix/step-46429/results/results.json` |
| Megatron-DeepSpeed 2B pre-torchtitan | 7.064T | **0.2413** | `mmlu-letter-prior-at-chance.md:51` |
| 20B-256 | 0.39T | **0.2599** | same |
| 20B-512 | 0.71T | **0.2655** | same |
| mix arm edu100 | ~600B+10B | **0.2476** (sample-weighted `mmlu` group key) | same |
| mix arm owm100 | ~600B+10B | **0.2482** | same |
| mix arm owm75 | ~600B+10B | **0.2393** | same |
| **always D** | — | **0.2689** | same — beats all measured arms |

Twelve configurations over ~20x token span, two model scales, five data treatments, plus most-trained (4.674T) scoring *lowest* — **no ordering among them meaningful on MCF alone**. Source: `thread-c-eval-harness.md:64-66`, `mmlu-letter-prior-at-chance.md:59-63`.

The 30B-ladder fan-out once read as "not chance-shaped" (subjects above 50%: 24->23->31, max 0.304->0.423) was explicitly retracted — same letter-prior/key-distribution interaction. Source: `thread-c-eval-harness.md:88-92`, `mmlu-letter-prior-at-chance.md:153-158`.

**Correct reporting discipline** for every MMLU number (source: `thread-c-eval-harness.md:180-199` §1.5 seven rules — see Why section). The gate is `max(CF,MCF)` with band + prior collapse, not MCF ranking.

### Llama controls prove harness clean

Harness control rules out plumbing bugs: **Llama-3.2-1B = 0.3121 and Llama-3.1-8B = 0.6530** through same `simple_evaluate(num_fewshot=5, device="xpu:0")` — **job 8736838**, `CONFIRMED`. Source: `thread-c-eval-harness.md:68`, `mmlu-letter-prior-at-chance.md:128-131`.

Plumbing checks all clean (source: `mmlu-letter-prior-at-chance.md:115-132`):
- Few-shot applied: logs show `Overwriting default num_fewshot of mmlu_abstract_algebra from None to 5` for all 57 subjects, every `dev` split has ≥5 rows.
- Tokenizer matches: all HF exports share `tokenizer.json` md5 `b60c6af2...`, vocab 256000, four continuations single clean tokens `_A`=586 / `_B`=599 / `_C`=585 / `_D`=608, prompt round-trips byte-exact. Not Polaris failure mode.
- No truncation: `LIMIT` empty, all 57 subjects at full size, 14,042 total.
- 1B clears chance where 2B at 4.674T does not — harness not broken, agpt-2B profile (MCF at chance, commonsense intact: HellaSwag ~0.57, ARC-Easy ~0.66 at same checkpoints) is consistent, not contradictory.

---

## What went wrong

### Sunspot MMLU OOM — `level_zero:0 40 63G` in `compute_topk_scores`, Aurora-only

**Symptom:** `MMLU vLLM level_zero 40 63G OOM (compute_topk_scores) unfixable on Sunspot XPU — Aurora-only, documented.` Source: `POST-TRAINING-2B.md:25` Eval infra, `thread-c-eval-harness.md:501`.

**Impact:** No CPT checkpoint has ever been scored on MMLU in either format (source: `mmlu-letter-prior-at-chance.md` Importance note, `SYNTHESIS.md:236` G3). The OLMo-2 +17.4 signal remains unmeasured on our data; closing it requires Aurora routing.

**Cost:** 57-subject MMLU pass is ~56k lm-eval requests/step — needs **48h `workq`** (prior 12h pass died Exit -29 walltime mid-MMLU). CoT 200 is one tile 40 min; `MAX_GSM8K_STEPS=200` sized for queue. Source: `thread-c-eval-harness.md:502`.

**Rule:** MMLU is Aurora-only unless XPU path fixed; CoT GSM8K *does* work on Sunspot (vLLM fp32). Commonsense (HellaSwag/ARC-Easy/PIQA) works on either via `lm-eval` HF backend `frameworks/2026.1.0`. Source: `SYNTHESIS.md:74` E2, `thread-c-eval-harness.md:501,664`.

### Training-loss vs eval divergence — healthy loss hid halved eval (B2v2)

**Signature:** B2v2 training loss **0.8069**, acc **0.8198** — descends cleanly — while CoT **0.205 -> 0.09/0.105** halved. Same as ckpt-8672 (loss 0.357 acc 0.902 while hellaswag 0.593->0.273) and dolmino pilots (loss 2.80->2.49 while HS -7.4pp, AE -10.4pp). Source: `POST-TRAINING-2B.md:82,88`, `SYNTHESIS.md:166-168,272-275`.

**Root cause:** More narrow SFT tokens at high LR destroy capability while NLL descends — overfitting on 7.4k gsm8k-r1cot repetition (Muennighoff 4-epoch limit). At 600 steps LR 2e-5 constant, boundary 100-300 steps; at 1e-5 extends. Early stopping is load-bearing structure, not accident. Source: `SYNTHESIS.md:167`, `POST-TRAINING-2B.md:85-88`.

**Prohibition earned:** Do NOT read training/validation loss as capability signal — gate only on downstream eval (G1-G4). Source: `SYNTHESIS.md:275` Prohibition #4.

### Limit-10 vs limit-40 — 0.028 sampling spread from subsampling

**Observation:** Same checkpoint (30B step-2000, 3.93B tok) scored **0.2281 at `limit 10`** (570 q) and **0.2561 at `limit 40`** (2,280 q) — **0.028 spread from sampling alone**. Neither is publishable MMLU; the 0.028 spread exceeds claimed "close" thresholds and warns that single-point comparisons without CI are noise. Source: `thread-c-eval-harness.md:125-127`, `mmlu-letter-prior-at-chance.md:291-294`.

On subsamples, letter-prior fit also breaks (A-weighted at limit-10, D-weighted at limit-40, R² 0.04–0.11) — neither subsample can test letter hypothesis; only full 14k runs can. Source: `mmlu-letter-prior-at-chance.md:295-298`, `thread-c-eval-harness.md:129-131`.

**Rule:** Do NOT subsample MMLU to save walltime — sampling drove 0.028 noise. Cut number of CPT checkpoints scored, not N. Run full 14,042 every time (walltime cost is number of *tasks* (57) not N). Source: `thread-c-eval-harness.md:519-522,620-622`, `SYNTHESIS.md:276`.

Similarly for CoT: 200 already separates actual B2 gaps (≥0.10); for near-tied variants (Δ 0.03-0.05) add second benchmark (MATH) rather than 500/1000 on same task — marginal information per walltime higher. Source: `thread-c-eval-harness.md:514-518`.

### Format collapse without guardrail — B4a and weak-base RL hacking

**B4a:** format **0.26**, 133/200 unclosed — envelope collapsed, different failure than reasoning. Source: `thread-c-eval-harness.md:256`, `POST-TRAINING-2B.md:101`, `SYNTHESIS.md:271`.

**RL weak-base reward hacking** (separate from SFT but same harness lesson): reward **0.118->0.217** while accuracy **0.16->0.105**, driven by `ThinkFormatReward 0.2->0.05` down-weight that removed SFT-held guardrail — format **0.955->0.635**; **95% of rollouts had no `<think>`**, **20% echoed exemplar `\boxed{5}`**. Gating reward drove to 0.000 when format removed, revealing hack. Source: `thread-c-eval-harness.md:268-274`, `POST-TRAINING-2B.md:144-150`.

On strong B2, identical GRPO setup is drift-proof (format 1.000 for 100 steps). Source: `POST-TRAINING-2B.md:114,142`, `thread-c-eval-harness.md:273`.

**Rules earned:**
- If `format < 0.80` or `unclosed > 30`, stop — do not interpret `acc` as reasoning. Source: `thread-c-eval-harness.md:268`.
- Treat `gen_len` > ~1.5× B2 baseline (282) as drift flag even when `format > 0.80`. Source: `thread-c-eval-harness.md:280-281`.
- Do NOT down-weight format to "fix" collapsed checkpoint — there is no cheaper fix than two-stage SFT that earned 0.985. Source: `SYNTHESIS.md:271` Prohibition #1, `thread-c-eval-harness.md:598-605`.
- Format >=0.90 + unclosed <=10 + gen_len [220,550] required alongside acc for promotion. Source: `SYNTHESIS.md:252-258` SFT-level gates, `thread-c-eval-harness.md:589-591` §5.3.

**Verbosity vs accuracy:** B4b restored format 0.86 + verbosity 611/27 and still only 0.05->0.065 — style was symptom, not cause. Rule: do NOT retry same mix at different weights (B4b proves mix tuning ~+0.015 max). Source: `POST-TRAINING-2B.md:103-104`, `SYNTHESIS.md:272` Prohibition #2, `thread-c-eval-harness.md:278-279`.

**Shot-count hygiene:** Before `5ab9e50f6`, MMLU artifacts (e.g. `step-46429/results.json`) had `n-shot: unrecorded` — shot count recoverable only from PBS stdout. Now `results.json` must carry `n-shot` per task; enforce at write time. Do NOT merge old artifacts lacking `@5shot` keys. Source: `thread-c-eval-harness.md:436-447,503-504,558-566`, `mmlu-letter-prior-at-chance.md:94-101`.

---

## Next steps

### E2 sweep protocol — every HF on HS/ARC/PIQA/ARC-C + MMLU CF+MCF

E2 tests the dolmino question: did CPT regress commonsense and did knowledge move off letter prior. Source: `SYNTHESIS.md:68-77` E2, `thread-c-eval-harness.md:555-571` §5.2 rows 1-4.

| Field | Value | Source |
|-------|-------|--------|
| Tests | Dolmino question — CPT commonsense regression + MMLU letter-prior movement | `SYNTHESIS.md:71` |
| Depends on | E1 HF converts (DCP->HF via `xpu-torch216-sync-5a7a622f16`, `flavor=2b-mds`, `PBS filesystems=tegu:home`, `TEGU_ROOT/outputs/evals/cpt-fwe/step-$s/hf` ~3.7G 9/9) | `SYNTHESIS.md:58-66` E1, `thread-c-eval-harness.md:537-546` |
| Checkpoints scored | `74000/74200` (39% budget, already HF) + next gate `~112k` + final `190734` (100B/524288) — plus retro bases `gs138650`, `2B-v2 256N step-92859`, `2B-v2 512N step-46429` for MMLU anchor | `thread-c-eval-harness.md:538-553` §5.1 table, `SYNTHESIS.md:63` |
| Benchmarks | **HellaSwag** `acc_norm` 0-shot, **ARC-Easy** `acc` 0-shot, **PIQA** `acc_norm` 0-shot, **ARC-Challenge** `acc_norm` 25-shot, **MMLU 5-shot MCF (`mmlu`) + CF (`mmlu_continuation`) -> `max(CF,MCF)`** + letter-prior fit `p(A,B,C,D)` + residual dispersion | `SYNTHESIS.md:75-76`, `thread-c-eval-harness.md:559-565` |
| Harness | `convert_and_eval.sh` DCP->HF -> `lm-eval 0.4.10` HF backend XPU `frameworks/2026.1.0` MKL+CCL+compiler+PTI LD as in `submit_cot_eval.sh`; CPT rows 1-4 of §5.2; CoT-free | `SYNTHESIS.md:74`, `thread-c-eval-harness.md:579` |
| Scoring discipline | Read group key `mmlu` from `results.json` (not hand-rolled macro: +0.005 high at step-46429); state format (MCF/CF/max), n-shot (5), constant-D band 0.2295-0.2689; full 14k MMLU not sampled | `SYNTHESIS.md:76`, `thread-c-eval-harness.md:183-188` rules 1-4 |
| Cost | Every HF 1N cheap (0-shot loglikelihood thermometers); MMLU 57 tasks ~56k requests needs Aurora 48h workq per pass | `thread-c-eval-harness.md:502,523-526` |

### G1-G4 thresholds — continue vs stop vs fork at 150k gate

Apply *before* E3 SFT probe thresholds; gate is conjunction — all must hold to continue. Thresholds from dolmino damage (HS -7.4pp, AE -10.4pp at step-5960 job 8647850), MMLU band, GSM8K-CoT power at N=200. Source: `SYNTHESIS.md:225-239` Gate definitions, `thread-c-eval-harness.md:559-571` §5.2:

| Gate | Metric | Base | Threshold to **CONTINUE** to 190k | SE / rationale | Source |
|------|--------|------|-----------------------------------|----------------|--------|
| **G1** — No commonsense regression | HellaSwag `acc_norm` 0-shot | 0.560 (2B-256 plateau) | **>= 0.540** (-2pp) | SE ~0.004; -2pp = 5σ; dolmino-100 fell to 0.486 | `SYNTHESIS.md:231`, `thread-c-eval-harness.md:561` (>=0.55 variant — SYNTHESIS tightened to 0.540) |
| **G1b** | ARC-Easy `acc` 0-shot | 0.651 | **>= 0.621** (-3pp) | SE ~0.006; dolmino 50/50 held 0.610, so -3pp strict | `SYNTHESIS.md:232`, `thread-c-eval-harness.md:562` |
| **G1c** | PIQA `acc_norm` 0-shot | 0.733 | **>= 0.713** (-2pp) | SE ~0.005; dolmino only -4.6pp — weak guard but cheap | `SYNTHESIS.md:233`, `thread-c-eval-harness.md:563` |
| **G2** — No catastrophic forgetting | ARC-Challenge `acc_norm` 25-shot | ~0.31 | **>= 0.290** (-2pp) | Peer table in `evals/` | `SYNTHESIS.md:234` |
| **G3** — MMLU must not regress | MMLU `max(CF,MCF)` 5-shot | 0.241-0.266 chance band | **>= 0.240** (no regression) | Band 0.2295-0.2689 (best constant-D), SE 0.0036. Clearing band requires prior collapse (R²->0) — not continue-gate until `max(CF,MCF) >0.2689` + prior collapse | `SYNTHESIS.md:235`, `thread-c-eval-harness.md:564` |
| **G4** — SFT probe lifts ceiling | GSM8K-CoT `acc` @200 post-SFT (B2 recipe) | B2 = **0.205** (41/200, 95% CI 0.149-0.260) | **>= 0.265** (B2 +0.06 at 2σ; 0.06 gap ~2.0σ on 200) | SE ~0.028 at 0.20. 0.01 RL move 0.205->0.215 is 0.26σ — correctly noise | `SYNTHESIS.md:236`, `thread-c-eval-harness.md:567-568` |
| **G4b** — Format held | GSM8K-CoT `format_hit_rate` | B2 = 0.985 | **>= 0.90** | B2v2 fell 0.86/0.91 while degrading; B3 0.86 masking 0.05 | `SYNTHESIS.md:237` |
| **G4c** — No verbosity blow-up | `mean_gen_len` / `n_unclosed` | B2 = 282 / ~0 | **<= 550** / **<= 10** | B2v2 416/378 with 22/15 unclosed; B4a 1378/133 flagged envelope loss | `SYNTHESIS.md:238`, `thread-c-eval-harness.md:589-591` |

**Near-miss interpretation** (source: `SYNTHESIS.md:240-249`):
- **G1-G2 fail >2pp but G4 passes** (commonsense dips, CoT ceiling lifts): **continue to 190k** — expected FWE HellaSwag trade (FWE 37.5 vs DCLM 35.5 MMLU, DCLM 62.3 vs 60.1 HS); re-check at 190k; monotonic bleed = ratio wrong -> revisit 75/25 arm (`config_registry.py:1024-1043`).
- **G1-G2 pass but G4 fails** (no CoT lift, e.g. 0.08-0.12 like B2v2/B3/B4): **stop CPT**, fork SFT from best intermediate checkpoint (max over G1-G4, not necessarily 150k), pivot to broad-distill on MDS base (E7).
- **Everything fails** (commonsense tanks + CoT collapses): **stop CPT, revert to MDS base** — FWE kind-shift wrong at this scale/mix or constant LR 2e-6 needs re-tuning.
- **G1-G3 pass but SFT probe not yet run** (E3 pending): **continue CPT** — do not stop without ceiling test. CPT-direct MMLU/ARC are thermometers; SFT probe is thermostat.

**SFT-level gates** (source: `SYNTHESIS.md:250-258`):
| Condition | Interpretation | Next action |
|-----------|---------------|-------------|
| New SFT GSM8K-CoT >=0.265 at 2σ + format >=0.90 + unclosed <=10 vs B2 0.205 | Promotion candidate | Ship; re-score MMLU with band+prior before external claim |
| New SFT format <0.80 or unclosed >30 | Envelope collapse | Reject without re-run; do not down-weight format |
| New SFT 0.15-0.20 at noise around B2 | No lift | Do not retry same mix at different weights (B4b proves ~+0.015 max) |
| Broad-distill on MDS base beats 0.15 where narrow collapsed | Breadth extends budget | Promote broad mix to 4N full run at winning LR |
| RL on B2 stays 0.205-0.215 flat at group_size 8 | ~0.2 is base ceiling for RL | RL for format only |

### Re-score qsub commands

**CoT (Sunspot, 1 tile, ~98s/200):**
```bash
# Single checkpoint
qsub -v CKPT=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/evals/cpt-fwe/step-75000/hf,LIMIT=200 \
  torchtitan/experiments/ezpz/scripts/eval/submit_cot_eval.sh

# B2/B2v2 canonical (already scored — re-score for stack hygiene)
qsub -v CKPT=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900-hf,LIMIT=200 \
  torchtitan/experiments/ezpz/scripts/eval/submit_cot_eval.sh

# With non-default dtype/max_tokens (must match B2: float32/768/greedy)
qsub -v CKPT=<hf-dir>,LIMIT=200,DTYPE=float32 \
  torchtitan/experiments/ezpz/scripts/eval/submit_cot_eval.sh
```
Source: `submit_cot_eval.sh:10,24-34` (CKPT/LIMIT/DTYPE/OUT/PY/EVAL_SCRIPT env overrides).

**Commonsense + MMLU (Aurora for MMLU, Sunspot for commonsense):**
```bash
# Commonsense thermometer — Sunspot or Aurora, frameworks/2026.1.0, lm-eval HF backend
# HellaSwag + ARC-Easy + PIQA per SYNTHESIS E2 — wraps convert_and_eval.sh DCP->HF then lm-eval
# (exact eval-2b-v2.sh invocation per thread-c-eval-harness.md:579,664 — SHOTS_SPEC mixed-few-shot loop)

# MMLU both formats — Aurora only (Sunspot OOM)
# Must run: mmlu (MCF) + mmlu_continuation (CF) at 5-shot, read group keys, compute max(CF,MCF)
# Cost: 48h workq per full MMLU pass (~56k requests); do NOT set --limit — full 14k
```
Source: `thread-c-eval-harness.md:528-526` operational constraints, `SYNTHESIS.md:74` E2.

**Provenance contract for every new artifact** (source: `thread-c-eval-harness.md:436-447,574-577`):
- Persist `n_shot` + `task_key` + `format` (MCF/CF/max) + band 0.2295-0.2689 + letter-prior `p(A,B,C,D)` + residual dispersion in `results.json`.
- CoT JSONL header: `model/dtype/temperature/max_tokens/finish_reason` series + 95% CI — byte-comparable re-scores.
- Do NOT merge old 2B-v2 artifacts lacking `n-shot` / `@5shot` keys — retro-rescore flagship step-46429 and step-92859 in both MMLU formats as first MMLU job.

### Aurora vs Sunspot routing

| Workload | System | Queue | Backend | Why | Source |
|----------|--------|-------|---------|-----|--------|
| **GSM8K-CoT generative** (vLLM fp32) | **Sunspot** | `workq` `40min` 1 tile | `frameworks/2026.1.0` + `vLLM 0.26` `float32` `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` | Works; 98-99s per 200 (jobs 12482572/73); `bf16 -> gibberish` (vocab 256k accumulation) | `submit_cot_eval.sh:15-20,28`, `POST-TRAINING-2B.md:149-160` |
| **HellaSwag / ARC-Easy / PIQA / ARC-C** (loglikelihood) | **Sunspot or Aurora** | `workq` | `lm-eval 0.4.10` HF backend `frameworks/2026.1.0` MKL+CCL+compiler+PTI LD | Cheap 0-shot; catches dolmino-style collapse; do NOT use `ezpz_setup_env` venv (transformers 5.6.2 breaks lm-eval) | `thread-c-eval-harness.md:664`, `docs/evals/README.md#Environment` |
| **MMLU 5-shot MCF+CF -> max** | **Aurora only** | `workq` `48h` | same `lm-eval` HF backend | **Sunspot OOM `level_zero:0 40 63G compute_topk_scores` unfixable** | `POST-TRAINING-2B.md:25`, `thread-c-eval-harness.md:501`, `SYNTHESIS.md:74` |
| **HF conversion** (DCP->HF) | **Sunspot** | `workq` `1N 02:00` | `xpu-torch216-sync-5a7a622f16` LD-first, `flavor=2b-mds` | Tegu output `/lus/tegu/.../outputs/evals/cpt-fwe/step-$s/hf` ~3.7G 9/9; `PBS filesystems=tegu:home` (no `/eagle` on Sunspot); patch config/tokenizer omission as at 74k | `SYNTHESIS.md:60-66` E1, `POST-TRAINING-2B.md:13` |
| **SFT ceiling probe** (B2 recipe on CPT HF) | **Sunspot** | `workq` `2N ~5min + 98s eval` | `rl-monarch-torch214-locality-20261001` LD-first, TRL SFTTrainer FSDP full_shard, LR 2e-5 cosine, GBS 384 seq 2048 | E3 decisive gate; needs fresh HF convert first | `SYNTHESIS.md:78-87` E3, `thread-c-eval-harness.md:420` |

**Minimum viable E2 sweep** (source: `SYNTHESIS.md:70-77`): convert 74000/74200 + 112k HFs (2-3 converts already/soon), run commonsense thermometer on each (1N cheap), queue Aurora 48h MMLU pass for both MCF+CF — cost <1 day walltime, zero new CPT compute, parallel with live CPT chain advancing on 12h cadence.

---

## Cross-refs

- Stack + scoring + power + plan (this doc's spine): `torchtitan/experiments/ezpz/docs/exploration/thread-c-eval-harness.md` — 7 scoring rules §1.5, power table §2.3 (p.300-316), 4-tuple diagnosis §2.2, CPT gating protocol §3.2, operational appendix §7
- MMLU letter-prior audit (band, prior fit, Llama controls, cloze experiment 12474382, per-subject correlations): `torchtitan/experiments/ezpz/docs/evals/mmlu-letter-prior-at-chance.md` — chance band 0.2295-0.2689 p.27-49, Llama 0.312/0.653 p.128-131, CF-MCF 0.2281/0.2386 p.272-276
- SFT deliverables + B2/B2v2/B3/B4/RL + CPT infra + VENV map: `torchtitan/experiments/ezpz/docs/production/POST-TRAINING-2B.md` — CoT table p.73-82 (jobs 12482572/73 00:01:38/39 Exit 0), MMLU OOM p.25, VENV `xpu-torch216-sync-5a7a622f16` vs `rl-monarch-torch214-locality-20261001`
- Decision gates G1-G4 + E2 sweep + thresholds + walltime: `torchtitan/experiments/ezpz/docs/exploration/SYNTHESIS.md` §§3-5 — E2 sweep p.68-77, G1-G4 table p.229-238 (HS>=0.540, AE>=0.621, PIQA>=0.713, MMLU>=0.240, CoT>=0.265)
- Eval scripts: `torchtitan/experiments/ezpz/scripts/eval/submit_cot_eval.sh` (PBS, VENV, qsub contract p.10,33-34), `torchtitan/experiments/ezpz/scripts/eval/eval_cot_gsm8k.py` (dataset/ prompting/ generation 152-175/ scoring 30-84/ summarize 118-131), `torchtitan/experiments/ezpz/scripts/eval/mmlu_letter_baseline.py` (constant-letter baselines)
- Eval strategy Tier A/B/C + SHOTS_SPEC + peer numbers: `torchtitan/experiments/ezpz/docs/evals/eval-landscape-2026-07.md` — Tier C deferral (IFEval/BBH/GPQA/MATH/HumanEval ~0 on base)

