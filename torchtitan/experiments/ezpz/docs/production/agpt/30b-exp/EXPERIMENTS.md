# 30B-exp experiment log

> **Last updated: 2026-08-14.**
>
> Tracking table for every experiment run against the
> [30B-exp proposal](README.md). One row per experiment, one file per
> experiment. **Status is honest**: an experiment that failed, was
> inconclusive, or contradicted the proposal is recorded as such -- a design
> doc whose own tests all conveniently agree with it is not evidence.

## Why these experiments exist

The proposal makes claims of three different kinds, and they need different
treatment:

- **Measured** (Section 1) -- already established by the 2B campaign. Not
  re-tested here.
- **Judgment** (Section 6) -- reasoning from measurements, not measurements.
  These are what the cheap experiments below attack.
- **Untested** -- the central data hypothesis. Needs a real run (exp03).

The tiering is set by *discriminating power*, not by cost. A toy model cannot
answer the MMLU question at all: MMLU has a capability floor around 1-3B
params, so a 20M or 500M model returns ~0.25 on **any** data and a null result
cannot separate "bad mix" from "model too small". Tier 0 experiments are cheap
*because* the questions they ask are scale-free, not because we are economising
on the important one.

## Tier 0 -- scale-free, no production allocation

| # | Experiment | Question | Cost | Status |
|---|---|---|---|---|
| [01](exp01-tokenizer-analysis.md) | Tokenizer analysis | Does gemma-7b's 256k vocab actually hurt us -- embedding cost, digit splitting, fertility, vocab utilisation? | CPU, minutes | **DONE** -- 2 claims refuted |
| [02](exp02-fp32-norms-ablation.md) | fp32 norms-only ablation | Is fp32 for *norm params only* sufficient, or does full fp32 master do real work? | debugmodel, ~2N | **DONE** -- norms-only is NOT sufficient; Section 6 claim refuted |
| [04](exp04-fp32-inference-investigation.md) | fp32 master vs fp32 inference | Does training with fp32 master weights force fp32 serving and double deployment cost? | CPU + cluster probes | **DONE** -- NO; vLLM cause not isolated |

## Tier 1 -- the gate

| # | Experiment | Question | Cost | Status |
|---|---|---|---|---|
| [03](exp03-1b-proxy-design.md) | 1B proxy design | Can a 1B on a candidate mix clear the MMLU floor, and is the 0.28 gate statistically defensible? | design only | **DONE** -- gate redesigned |

## Findings

Filled in as experiments land. Each entry states whether it **supports**,
**undermines**, or **does not settle** the corresponding proposal claim.

### exp01 -- tokenizer analysis (2026-08-14)

**Undermines the proposal's stated rationale; strengthens a different one.**
Section 3's conclusion (shrink the vocab) survives, but two of its four
supporting bullets were measurably false and the headline cost figure was
understated by 2x.

1. **REFUTES "single-digit tokenization" as a reason.** gemma **already**
   tokenizes every numeral into individual digits -- 9,999/9,999 integers,
   one token per digit, **zero exceptions**, context-independent. The
   decisive probe: prepending a `0` leaves gemma's segmentation untouched
   (`2024` -> `2 0 2 4` becomes `0 2 0 2 4`), while **Llama-3.1 re-segments
   the entire number** (`202`,`4` -> `020`,`24`). The pathology the proposal
   described is Llama's, not ours. **A custom tokenizer cannot buy this, and
   digit handling cannot explain GSM8K = 0.0000** -- that failure needs a
   different explanation. *(Measured, 3,000 random numerals.)*

2. **REFUTES "LaTeX / units / indentation" as a reason.** gemma encodes a
   24-space run as one token and *beats* Llama-3.1 on `\begin{equation}`,
   `kPa`, `\nabla^2`. *(Measured.)*

3. **CORRECTS the embedding cost, in the proposal's favour, by 2x.** Not 525M
   / ~26% but **1,049M / 52.8%** -- production agpt has `enable_weight_tying`
   **off**, so the 256128x2048 matrix is instantiated twice (embedding +
   untied `lm_head`). `config_registry.py:163` already said ~53%; the
   proposal cited one matrix. At 30B (dim 6144) it is 10.5%, confirming this
   is a small-model argument.

4. **SUPPORTS the ~64k target, on a different statistic.** **99% of all token
   mass sits in the top 62,108 IDs; 129 IDs cover 50%.** Fertility is within
   2-4% of Llama-3.1 on every prose/science domain (gemma *wins* peS2o at
   0.983x); the only real penalty is code, **+17% starcoder / +26% Python**.
   Qwen3 matches Llama on code at 151k vocab, so that is gemma's merges, not
   vocab size.

5. **Internal contradiction caught.** The section wanted single-digit
   tokenization *and* fewer wasted tokens. The +16% GSM8K fertility **is the
   price of** single digits; any tokenizer keeping that property pays it.

**Cheapest actionable finding: turn on weight tying.** Recovers 525M params
at 2B for free -- no re-tokenization of the 4.674T corpus, no loss of
checkpoint comparability. `agpt_2b_tied` already exists and should be tried
before any tokenizer work.

**Self-flagged caveat (good practice, recorded):** the agent's first
vocab-utilisation pass sampled sequentially from each domain's first shard and
produced "42.6% never used". A randomised re-run roughly **doubled** every
domain's distinct-ID count, so that figure was demoted to an upper bound and
the ~64k argument rests on the sampling-robust mass-concentration statistic
instead. Partial data in the report's Section 5b.

### exp02 -- norms-only fp32 ablation (2026-08-14)

**Refutes the Section 6 concession, in the opposite direction from the one
proposed.** The question was whether full fp32 master is over-broad -- whether
fp32 on the *norm params only* would have sufficed. It would not, and the
reason had gone unnoticed for the entire campaign.

1. **REFUTES "norms-only fp32 would have fixed the observed bug."** Three
   arms, identical seed/data/steps, differing only in the master copy
   (job 8757151, agpt debugmodel, 300 steps). Arm A (bf16) reproduces v1
   exactly: **13/13 RMSNorm weights bit-identical to their 1.0 init**,
   variance identically zero. Arm B (norms-only fp32) fully fixes the norms
   -- statistically indistinguishable from full fp32 on them. **But it leaves
   `tok_embeddings.weight` frozen at `frac_changed` = 0.000345 against
   1.000000 under full fp32.** Across the model, **38.36% of all parameter
   elements never move under norms-only, vs 0.000% under full fp32.**

2. **The cause: agpt initializes the embedding at `std=1.0`
   (`_EMBEDDING_INIT`, `agpt/__init__.py:202`) -- the same scale as
   `RMSNorm.weight`, hence the same ~7.8e-3 bf16 ULP.** The prior
   investigation's "why other parameters update fine" section reasoned only
   about *linear* layers at std~0.005-0.02. The embedding is neither a linear
   layer nor at that scale, and nobody checked it.

3. **Mechanism confirmed as scale, not vocab coverage -- three controls.**
   The obvious objection is that most vocab rows are unvisited in 300 steps.
   **(a) The decisive one (job 8757243):** restrict `frac_changed` to only
   the embedding rows that actually received a nonzero gradient. All three
   arms touch **exactly the same 91/32,000 rows** (same seed, same data);
   among those rows full fp32 moves **1.0000** of elements while arm B moves
   **0.1471** and the fully broken arm A moves 0.1464. **Arm B is
   indistinguishable from the broken arm once coverage is eliminated by
   construction.** (b) An isolated optimizer probe applying identical
   gradients to every element: std=1.0 bf16 moves 19.81%, std=1.0 fp32 moves
   100%, std=0.02 bf16 moves 100% -- scale alone reproduces it. (c) Arm C
   moves 100.000% of the same embedding under the same data, which coverage
   cannot explain.

4. **Loss cannot see any of this.** All three arms land within 0.005 nats at
   step 300, and **the broken arm has the numerically lowest final loss.**
   This reproduces the campaign's central lesson at 1/1000th the scale:
   anyone using loss as the acceptance signal for a precision change will
   accept a broken configuration.

5. **The QK-norm recurrence prediction: CONFIRMED (job 8757254).** Section 6
   warned the vulnerability recurs "for any parameter initialized near 1.0,
   QK-norm gains being exactly that." With QK-norm enabled, arm A freezes
   **25/25 norms including all 12 QK-norms.** The 30B plan's intent to add
   QK-norm would have walked straight into this had the default been bf16.
   The prediction was also *under-stated*: it had already recurred in the
   shipped 2B/20B/80B configs, on the embedding, unnoticed because the v1
   post-mortem only checked norms.

6. **Arm B is also the SLOWEST arm, so it has no remaining advantage**
   (job 8757216, 2 nodes / 24 ranks / FSDP degree 24). tps: **A 29.8,
   B 39.0, C 54.0** -- full fp32 is 38% faster than norms-only and 81%
   faster than bf16 master, at identical memory. FSDP2 forbids mixed dtypes
   in a shard group, so norms-only *requires* splitting each of the 13 norms
   into its own `fully_shard` group and its own collectives. **Arm B loses on
   correctness, throughput, and complexity simultaneously.**

**Net effect on the proposal: full fp32 master stays, and the justification
upgrades from risk-asymmetry judgment to measurement -- it is load-bearing,
for a reason nobody had identified, and it is also the fastest option
measured.** Section 6 amended; a correction added to
`docs/guides/training-dtype-bf16-norm-freeze.md`. Worth keeping as a
scale-free regression test: 300 steps on a 21M model, ~20s per arm, catches a
class of bug that cost a full production restart.

**Operational trap documented:** DCP stores bf16 regardless of master dtype,
and an *early* fp32-master checkpoint is indistinguishable from a broken one
(v2 norms are still all-ones at step 1000, clearly moved by step 5960). Never
judge a precision fix from an early checkpoint.

**Open follow-up, independent of dtype:** `_EMBEDDING_INIT` at `std=1.0` is
what puts the embedding in the danger zone at all, and is unusual against the
`dim**-0.5` agpt already uses for `lm_head`. Worth revisiting for 30B on its
own merits.

**Caveats:** debugmodel scale with AdamW (production uses SophiaG); the
absolute tps figures are from a 21M model at 24 ranks, communication-bound at
~1-2% MFU, so the *ordering* is the result, not the numbers; the embedding's
parameter share is 38% here vs ~26% at 2B; `lm_head` (std~0.06) is
unaffected.

### exp04 -- does fp32 master force fp32 inference? (2026-08-14)

**Answers a direct objection to Section 6: no, and the two are structurally
decoupled.** Also refutes the documented explanation for a real operational
bug, and turns up three unrelated defects.

1. **SUPPORTS Section 6 (the fp32-master recommendation survives).** Two
   independent reasons. Training forward/backward compute is bf16 *either
   way* -- `training.dtype` controls only the optimizer-side master copy,
   while `mixed_precision_param=bfloat16` governs every GEMM
   (`parallelize.py:177`, `configs.py:55,62,70`). And **MEASURED:** the
   exported safetensors are uniformly **BF16, 111/111 tensors** -- the fp32
   master never reaches the served checkpoint at all. Even a fully
   fp32-master run ships bf16 weights. The serving-dtype cost cannot be
   charged against the training-dtype decision.

2. **MEASURED, the key result:** the completed 2B (step-92,859) at bf16 is
   **byte-identical to fp32 over a 200-token greedy decode**. The model is
   not bf16-fragile. bf16 *does* flip ~5% of argmaxes (top1-top2 gap median
   0.82 vs max logit diff 2.75), so the mechanism is real in the small -- but
   it yields a different fluent continuation, never word salad.

3. **REFUTES the recorded root cause of the vLLM gibberish.** Four docs state
   that vocab 256000 + ffn 11008 accumulate enough bf16 error to flip greedy
   argmax. **MEASURED:** that exact architecture (`global_step138650`, direct
   ancestor of the ckpt-900 that gibberished) generates **coherently in bf16**
   through HF. The blamed property is present and produces no gibberish.

4. **Real cause: NOT established.** By elimination the failure is inside
   vLLM, not the model -- but that is an **inference, not a measurement**.
   ckpt-900's safetensors are deleted (dangling symlink) and vLLM will not
   init on a login node, so the A/B could not be re-run. All generation tests
   ran on **CPU**, which exonerates the weights and architecture but not the
   XPU kernels. Keep `--generator.model-dtype=float32` as a workaround, but
   re-label it "cause not isolated" rather than "agpt models need fp32,"
   which would wrongly tax every future deployment.

**Three unrelated bugs found:**

- **No `torch_dtype` in any of 40 exported `config.json`.** vLLM's resolver
  then falls back to safetensors metadata and finally to `torch.float32`,
  before down-casting by platform preference -- so serving dtype is decided
  by fallback logic rather than by us. One-line fix, highest value.
- **RoPE flavor mismatch (H3) -- exposure resolved separately, see below.**
- bos/eos swapped vs the tokenizer (confirmed; ruled out as a bf16 trigger).

**H3 exposure, resolved by follow-up (MEASURED).** The agent left this
inferred from main-repo script defaults; reading the **pinned clones** settles
it. `runs/agpt-2b-v2` -- which produced both completed 4.674T chains -- is
pinned at `f319e3fa`, *predating* the `_real` default (`5ffb850a1`,
2026-06-25). It has no `*_autoretry.sh` at all, and its legacy script's
`${CONFIG_SUFFIX:-}` has **no assignment anywhere in the file**, so those
chains trained **complex** RoPE -- exactly what `MODEL_FLAVOR=2b` converts.
**Every 2B eval in the campaign was converted correctly**, independently
corroborated by HellaSwag rising monotonically 0.405 -> 0.561 (scrambled Q/K
cannot produce a clean learning curve). So near-chance MMLU is **not** a
conversion artifact.

But `runs/agpt-20b-v2` and `runs/agpt-2b-constlr-from9200` both default
`CONFIG_SUFFIX=_real` while `eval-2b-v2.sh:69` still defaults to the complex
flavor -- **the trap is live for every chain newer than the completed 2B.**
Tracked as task #73.

### exp03 -- 1B proxy design (2026-08-14)

Four findings, three of which **undermine** the proposal as written. The
proposal has been amended in place; the originals are preserved here.

1. **UNDERMINES Section 5 ("1B proxy").** There is no 1B to build -- our
   production `agpt_2b` has **0.937B non-embedding params**, *below*
   Llama-3.2-1B's 0.973B. The 256,128-entry gemma vocab puts 525M in the
   embedding and 525M in the untied lm_head; **52.8% of the parameter budget
   does no depth-of-computation work.** Every smaller candidate costed
   (dim1536 L24 = 0.604B, dim2048 L16 = 0.721B) lands *below* the floor. The
   proxy is the production config, unchanged. *(Measured from
   `agpt/__init__.py:434`.)*

2. **UNDERMINES Section 5 ("hard gate 0.28").** Correct arithmetic, wrong
   shape. 0.28 is 8.2 binomial SE above chance at N=14,042 -- but only
   **1.45 pp above the highest near-chance MMLU we have ever recorded**
   (0.2655), and our across-config null is **2.4x overdispersed** (sd 0.00873,
   n=8), making 0.28 a 3.2 sigma gate under the empirical null. Structurally
   worse: a one-arm absolute threshold assumes the null is 0.25 and that
   harness/tokenizer/prompt-format contribute nothing -- all live hypotheses.
   Replaced with a **paired, control-referenced** rule (McNemar, +0.030 vs
   control, plus a same-mix reseed arm that measures the noise floor first).

3. **UNDERMINES Section 6's motivation, STRENGTHENS the design.** The "MMLU
   capability floor at 1-3B" premise appears **false**: Qwen2.5-0.5B scores
   47.5 while TinyLlama-1.1B at 3T tokens scores 25.3. Constraint is data, not
   scale. No published parameter threshold for MMLU emergence was located. A
   null at 0.937B is therefore *more* informative than the proposal assumed.
   *(Recalled, mixed harnesses -- not reproduced on ours.)*

4. **CORRECTS Section 1.3.** The 256N = 9.5% MFU row is a **torch 2.10**
   number; the current 2.13 stack measures **18.77%**. The table overstates
   the mid-scale collapse by ~2x. 512N ~9-11% is current; 64N is still the
   knee.

**Cost finding (supports feasibility):** FineWeb-Edu is *already on Aurora,
already gemma-tokenized, already in blendcorpus format* (1.551T tokens),
plus dolmino `flan` (17.1B) and `math` (11.7B). An arm is a weighted text
file, not a tokenization campaign. Four arms = **~771 node-hours = 0.14% of a
single 512N x 12h production job.** Counter-finding: the `_local` dataset
registrations in `datasets.py` point at `/lus/tegu/` (**Sunspot**), and
Aurora's `Nemotron-CC-Math-v1/4plus` is an **empty directory** -- a Nemotron
arm on Aurora would be planned against data that is not there.

**Provisional:** the 2.4x overdispersion estimate rests on n=8 spanning two
model sizes and five data treatments, so some spread may be real capability
difference. The reseed arm replaces it with a measurement; until then +0.030
is provisional.

## Rules for this directory

1. **One file per experiment**, named `expNN-<slug>.md`, linked from the table
   above.
2. Every experiment file opens with a **## Verdict** section -- two to four
   sentences, stating plainly whether the evidence supports or undermines the
   claim it tested.
3. **Measured vs derived vs recalled** must be labelled. The proposal's
   Section 6 exists because that line was blurred once already.
4. A **negative or inconclusive result is a result** and gets the same
   treatment as a positive one. "Not settled, here is the blocker" is a valid
   verdict.
5. Job IDs and exact commands go in the **## Method** section so anything here
   can be re-run.
