# exp01 -- Tokenizer analysis: does gemma-7b's 256k vocab actually hurt us?

> **Date: 2026-08-14. Status: COMPLETE** (one secondary robustness re-run
> still in flight -- see 5b; it does not change any verdict). Pure CPU
> measurement on Aurora login node `aurora-uan-0012`; no model training, no
> allocation used.
>
> Tests the four claims in [Section 3 of the proposal](README.md#3-tokenizer).

## Verdict

**The evidence is split, and it undermines the proposal's headline
justification.** The single-digit claim is simply **false as applied to
gemma-7b**: gemma already tokenizes every numeral into individual digits,
context-independently and exhaustively (all 9,999 integers tested, one token
per digit, zero exceptions) -- it is the *good* behaviour the proposal asks
for, so a custom tokenizer cannot buy it and cannot explain GSM8K = 0.0000.
The LaTeX/units/indentation claim is likewise **not supported**: gemma matches
or beats Llama-3.1 on LaTeX and units and encodes a 24-space run as one
token. What *does* hold up is the **cost** argument, and it is much stronger
than the proposal states -- untied embeddings at 2B are **1,049M params =
52.8% of the model**, not the 525M / ~26% claimed -- plus a real but modest
code-fertility penalty (**+17% tokens on starcoder**, +26% on Python vs
Llama-3.1) and genuine vocab oversizing (**99% of all token mass sits in the
top ~62,108 IDs**, and just 129 IDs cover half of it -- a ~64k vocab fits our
measured distribution). Note the "42.6% never used" figure from the first pass
is an **upper bound that a larger randomized re-run is actively eroding**
(see 5b); the mass-concentration number is the robust one.

Net: **shrink the vocab for cost and code-efficiency reasons, which are
measured and real. Do not sell it as a digit fix or a GSM8K fix -- that
rationale is measurably wrong, and citing it would put a false statement in a
design doc.** Note also that the cheapest fix to the 2B cost problem is
*weight tying*, not retokenization: tying alone removes 525M params without
touching the data pipeline or invalidating a single existing checkpoint.

## Method

All measurements run on the Aurora login node with the repo venv
(`python3.12`, `transformers 5.9.0`), fully offline
(`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`):

```
PY=/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz/.venv/bin/python3
```

Tokenizers, all from the pre-existing local HF cache (`~/.cache/huggingface/hub`),
nothing downloaded:

| tokenizer | `len(tokenizer)` |
|---|---|
| `google/gemma-7b` (ours) | 256,000 |
| `meta-llama/Llama-3.1-8B` | 128,256 |
| `meta-llama/Llama-2-7b-hf` | 32,000 |
| `Qwen/Qwen3-0.6B` | 151,669 |

Note `len()` is 256,000; the model's `vocab_size=256128` is that plus 128 rows
of alignment padding (confirmed in
`torchtitan/experiments/ezpz/agpt/__init__.py`, flavor `"2B"`).

Scripts (written to `/tmp/tokan/` on Aurora, all re-runnable as-is):

- `tokan.py` -- digit splits, cross-context digit consistency, exhaustive
  sweep of every integer 1..9,999.
- `consist2.py` -- stricter consistency re-test: 20 contexts, per-digit token-id
  stability over 3,000 random numerals, and the leading-zero re-segmentation
  probe.
- `ws.py` -- indentation runs, whitespace-token inventory, LaTeX/unit strings.
- `corpus.py` -- vocab utilisation + text extraction from the **real
  pretokenized production corpus**.
- `fert.py` -- fertility (tokens/word, bytes/token).
- `vocab2.py` -- vocab utilisation robustness re-run at larger scale with
  randomized document sampling.
- `rt.py` -- decode/re-encode round-trip check (fairness control, below).

**Corpus.** Vocab utilisation and per-domain fertility use the actual
production training data, not a proxy:
`/lus/flare/projects/AuroraGPT/datasets/olmo-mix-1124/data_fused_gemma_eod/`
-- Megatron indexed `.bin`/`.idx` pairs of int32 **gemma token IDs**, the exact
bytes the 2B/20B chains consumed. All 7 domains: `wiki`, `arxiv`, `pes2o`,
`dclm`, `open-web-math`, `starcoder`, `algebraic-stack`. Token IDs were read
directly from the `.bin` (no re-tokenization), so utilisation is exactly what
production saw.

**Fairness control.** Per-domain fertility text was obtained by *decoding*
gemma IDs back to text, which could in principle favour gemma. Verified it does
not: for all 7 domains the decode -> re-encode round trip is bit-identical
(`text_identical=True`, token counts equal), so the decoded text is a faithful
reconstruction and the comparison across tokenizers is fair. GSM8K numbers use
the original HF dataset text, never round-tripped.

**Parameter arithmetic** is exact analytic counting (not measured from a
checkpoint) against the shipped configs in
`torchtitan/experiments/ezpz/agpt/__init__.py`, using
`attn = 2*dim^2 + 2*dim*n_kv*head_dim`, `mlp = 3*dim*hidden`, `2*dim` norms per
layer, plus a final norm; embeddings counted twice (untied).

## Results

### 1. Embedding parameter cost (DERIVED -- exact arithmetic, not measured)

The production agpt configs are **untied**. A weight-tying switch does exist
(`model.enable_weight_tying`, used by the `agpt_2b_tied` flavor at
`config_registry.py:162`), but it is **off** in every production flavor
(`2B`/`20B`/`80B`), so the `256128 x dim` matrix is instantiated **twice**:
input embedding + `lm_head`.

Exact counts for the shipped flavors, vocab 256,128:

| flavor | dim | layers | body | embed | lm_head | **total** | **embed+head %** |
|---|---|---|---|---|---|---|---|
| `2B`  | 2048 | 12 | 0.937B | 0.525B | 0.525B | **1.987B** | **52.8%** |
| `20B` | 5120 | 64 | 18.120B | 1.311B | 1.311B | **20.743B** | **12.6%** |
| `80B` | 9216 | 84 | 76.103B | 2.360B | 2.360B | **80.824B** | **5.8%** |

**The proposal understates its own strongest argument.** It says "525M params
(~26% of the model)". 525M is the count for **one** of the two matrices; the
untied pair is **1,049M = 52.8%** of the 1.987B total. The in-repo comment at
`agpt/config_registry.py:163-164` already says the right thing ("~53% of a
2B"). Section 3 of the README should be corrected to 1,049M / ~53%.

Requested grid (embedding params, and share of a 2.0e9 / 3.0e10 budget):

| vocab | dim | 1x matrix | 2x (untied) | % of 2B (untied) | % of 2B (tied) |
|---|---|---|---|---|---|
| 256,128 | 2048 | 524.6M | 1049.1M | **52.46%** | 26.23% |
| 100,000 | 2048 | 204.8M | 409.6M | 20.48% | 10.24% |
| 64,000 | 2048 | 131.1M | 262.1M | 13.11% | 6.55% |
| 32,000 | 2048 | 65.5M | 131.1M | 6.55% | 3.28% |

| vocab | dim | 1x matrix | 2x (untied) | % of 30B (untied) | % of 30B (tied) |
|---|---|---|---|---|---|
| 256,128 | 6144 | 1573.7M | 3147.3M | **10.49%** | 5.25% |
| 100,000 | 6144 | 614.4M | 1228.8M | 4.10% | 2.05% |
| 64,000 | 6144 | 393.2M | 786.4M | 2.62% | 1.31% |
| 32,000 | 6144 | 196.6M | 393.2M | 1.31% | 0.66% |

(dim=6144 is a stand-in for "30B-ish" as specified in the task; no 30B flavor
exists in the registry yet. Percentages use the nominal 2.0e9 / 3.0e10
denominators requested, which is why the 2B row reads 52.46% against the 52.8%
computed from the true 1.987B total.)

**The proposal's own framing is right that this matters far less at 30B**:
going 256k -> 64k at dim=6144 recovers 2,361M params, only ~7.9% of a 30B.
At 2B the same swap recovers 787M -- ~40% of the model. This is an argument
about small models, and the proposal says so.

### 2. How gemma tokenizes numbers (MEASURED) -- the claim fails

Exact splits, no special tokens:

| text | gemma-7b | Llama-3.1-8B | Llama-2-7b | Qwen3-0.6B |
|---|---|---|---|---|
| `123` | **3**: `1 2 3` | 1: `123` | 4: `▁ 1 2 3` | 3: `1 2 3` |
| `4567` | **4**: `4 5 6 7` | 2: `456 7` | 5: `▁ 4 5 6 7` | 4: `4 5 6 7` |
| `12345` | **5**: `1 2 3 4 5` | 2: `123 45` | 6: `▁ 1 2 3 4 5` | 5: `1 2 3 4 5` |
| `0.5` | **3**: `0 . 5` | 3: `0 . 5` | 4: `▁ 0 . 5` | 3: `0 . 5` |
| `3.14159` | **7**: `3 . 1 4 1 5 9` | 4: `3 . 141 59` | 8: `▁ 3 . 1 4 1 5 9` | 7: `3 . 1 4 1 5 9` |
| `1,234,567` | **9**: `1 , 2 3 4 , 5 6 7` | 5: `1 , 234 , 567` | 10: `▁ 1 , 2 3 4 , 5 6 7` | 9: `1 , 2 3 4 , 5 6 7` |

GSM8K-style sentence ("Natalia sold 48 clips..."): gemma = 24 tokens, with
`48` -> `▁`,`4`,`8`. Llama-3.1 = 25 tokens, with `48` as a single token.

**Exhaustive sweep** -- every integer, token-count histogram:

| tokenizer | 100..999 | 1..9,999 |
|---|---|---|
| **gemma-7b** | **3 tokens: 900/900** | 1d->1, 2d->2, 3d->3, 4d->4 (**perfect one-token-per-digit, 9,999/9,999**) |
| Llama-3.1-8B | 1 token: 900/900 | 999 nums ->1 tok, 9,000 ->2 tok (chunked) |
| Llama-2-7b | 4 tokens: 900/900 | one-per-digit **plus** a leading `▁` |
| Qwen3-0.6B | 3 tokens: 900/900 | identical to gemma |

**Consistency** -- same numeral in 10 different contexts (`"{}"`,
`"The answer is {}."`, `"x = {}"`, `"({})"`, `"={}"`, `"\n{}"`, `"a{}"`, ...),
counting distinct tokenizations of the digit-bearing pieces:

| numeral | gemma-7b | Llama-3.1-8B | Llama-2-7b |
|---|---|---|---|
| `7` | **1** | 1 | 2 |
| `42` | **1** (`4`,`2`) | 1 (`42`) | 2 |
| `123` | **1** (`1`,`2`,`3`) | 1 (`123`) | 2 |
| `2024` | **1** (`2`,`0`,`2`,`4`) | 1 (`202`,`4`) | 2 |
| `1000000` | **1** (seven single digits) | 1 (`100`,`000`,`0`) | 2 |

**Stricter re-test** (`consist2.py`, 20 contexts incl. `"0{}"`, `"$-{}$"`,
`"{}th"`, `"[{}]"`, `"#{}"`, `"{}%"`). Two additional checks:

*(a) Is each digit character always the same token id?* For gemma, yes -- all
ten digits map to exactly **1** id each (`0`->235276, `1`->235274, ...,
`9`->235315) across 3,000 random numerals up to 12 digits long.

*(b) Does prepending a single `0` re-segment the number?* This is the sharpest
probe of positional consistency:

| numeral | gemma: `"{}"` -> `"0{}"` | Llama-3.1: `"{}"` -> `"0{}"` |
|---|---|---|
| `2024` | `2 0 2 4` -> `0 2 0 2 4` (**unchanged, just prefixed**) | `202`,`4` -> `020`,`24` (**fully re-segmented**) |
| `1000000` | seven digits -> eight digits (**unchanged**) | `100`,`000`,`0` -> `010`,`000`,`00` (**re-segmented**) |
| `31415926535` | eleven digits -> twelve (**unchanged**) | `314`,`159`,`265`,`35` -> `031`,`415`,`926`,`535` (**re-segmented**) |
| `999` | `9 9 9` -> `0 9 9 9` (**unchanged**) | `999` -> `099`,`9` (**re-segmented**) |

Gemma is **perfectly compositional**: a numeral's tokenization is the
concatenation of its digits' tokens, always, independent of position, length,
or surrounding context. Llama-3.1's chunking shifts with a one-character
offset, so the model must learn that `202`+`4` and `020`+`24` describe
overlapping quantities -- exactly the pattern the arithmetic literature
identifies as harmful.

**This is the decisive negative result.** The proposal asks for
"single-digit tokenization" as a fix. Gemma-7b **already does exactly that**,
perfectly and context-independently -- it is the same scheme Qwen3 uses and is
the configuration the arithmetic literature recommends. The pathological
pattern the proposal is reaching for (inconsistent chunking, where `2024`
becomes `202`+`4` but `1000000` becomes `100`+`000`+`0`) is **Llama-3.1's**
behaviour, not ours. Gemma shows **zero** inconsistency.

Therefore: **GSM8K = 0.0000 cannot be attributed to digit tokenization, and a
custom tokenizer cannot improve digit handling over what we already have** --
the current scheme is already optimal on this axis. The GSM8K zero must be
explained by something else (mix content, model scale, answer-format/parsing in
the eval harness, or absence of any CoT/instruction exposure). That is
consistent with `POST-TRAINING-2B.md` finding ~0.2 to be a 2B ceiling reached
via SFT structure.

### 3. Fertility (MEASURED): tokens per whitespace-word

Real production corpus, ~300KB of decoded text per domain (round-trip verified
lossless), plus synthetic LaTeX/Python and real GSM8K text. **Lower is better.**

| corpus | words | **gemma-7b** | Llama-3.1-8B | Llama-2-7b | Qwen3-0.6B |
|---|---|---|---|---|---|
| wiki | 48,712 | **1.364** | 1.337 | 1.528 | 1.354 |
| dclm | 50,884 | **1.413** | 1.357 | 1.616 | 1.393 |
| pes2o | 39,486 | **1.380** | 1.404 | 1.692 | 1.437 |
| arxiv | 34,321 | **1.875** | 1.833 | 2.078 | 1.887 |
| open-web-math | 48,040 | **1.777** | 1.714 | 1.977 | 1.793 |
| algebraic-stack | 32,147 | **2.045** | 2.003 | 2.251 | 2.079 |
| starcoder | 31,419 | **2.539** | 2.168 | 2.940 | 2.210 |
| latex (synthetic) | 1,400 | **3.344** | 3.258 | 3.458 | 3.429 |
| python (synthetic) | 1,820 | **3.352** | 2.660 | 3.836 | 2.660 |
| gsm8k (real text) | 29,938 | **1.902** | 1.638 | 2.045 | 1.883 |

Gemma tokens relative to Llama-3.1 (>1 = gemma needs more tokens for the same
text, i.e. wastes compute):

| corpus | ratio |
|---|---|
| pes2o | **0.983x** (gemma better) |
| wiki | 1.020x |
| algebraic-stack | 1.021x |
| arxiv | 1.023x |
| latex | 1.026x |
| open-web-math | 1.037x |
| dclm | 1.041x |
| **gsm8k** | **1.161x** |
| **starcoder** | **1.171x** |
| **python** | **1.260x** |

Reading: on prose, scientific text, LaTeX and math, gemma is within **2-4%**
of Llama-3.1 despite carrying 2x the vocab -- essentially a wash, and it beats
Llama-3.1 on peS2o. The real penalty is **code (+17% on real starcoder, +26%
on Python)** and **digit-dense text (+16% on GSM8K)**, the latter being the
direct and expected cost of the one-token-per-digit scheme. Qwen3 matches
Llama-3.1 on code at only 151k vocab, so this gap is a property of gemma's
merges, not of vocabulary size.

**This partially contradicts the proposal's third bullet.** LaTeX is a 2.6%
gap, not a waste; the corpus's science-heavy domains (arxiv, pes2o,
open-web-math, algebraic-stack) are all within 4% or better. Only code is a
real inefficiency.

**Important tension the proposal should absorb:** the +16% GSM8K fertility is
*not* a defect to be engineered away -- it is the arithmetic price of
one-token-per-digit, the very property the proposal asks for. Any custom
tokenizer that keeps single-digit tokenization **will reproduce this
+16% cost**; one that removes it to save tokens would be reintroducing
Llama-style chunked digits. These two goals are in direct conflict, and the
proposal currently asks for both (single digits *and* fewer wasted tokens).
Pick single digits and accept the token cost.

### 4. LaTeX, units, indentation (MEASURED) -- claim not supported

**Indentation.** Tokens needed for a run of k spaces:

| k | gemma-7b | Llama-3.1-8B | Llama-2-7b | Qwen3-0.6B |
|---|---|---|---|---|
| 1,2,4,8,12,16 | **1** | 1 | 1 | 1 |
| 24 | **1** | 1 | 2 | 1 |
| 32 | **2** | 1 | 2 | 1 |

Gemma has 30 multi-space-run tokens and encodes up to **24 spaces as a single
token**. Indentation is *not* mishandled. (Llama-3.1 is marginally better past
24 spaces -- irrelevant at realistic indent depths.)

**LaTeX and units**, gemma vs Llama-3.1 (token counts):

| string | gemma-7b | Llama-3.1-8B |
|---|---|---|
| `\begin{equation}` | **5** | 6 |
| `\sqrt{s}` | 5 | **4** |
| `kPa` | **1** | 2 |
| `\nabla^2` | **4** | 5 |
| `mol/L` | 3 | **2** |
| `298.15 K` | 7 | **4** |
| `101.325 kPa` | 8 | **5** |
| `\frac`, `\alpha`, `\mathrm`, `$\sigma$`, `\end{align}` | tie | tie |

Gemma wins on `\begin{equation}`, `kPa`, `\nabla^2`; loses on numeric literals
(`298.15`, `101.325`) purely because of digit splitting -- which is the
behaviour the proposal *wants*. On LaTeX commands proper it is at parity or
better. **No evidence that gemma "wastes tokens" on LaTeX or units relative to
a modern general-purpose tokenizer.**

### 5. Vocabulary utilisation (MEASURED) -- claim supported

Read directly from production `.bin` files as raw gemma IDs. **56,247,183
tokens** across all 7 domains (~8M per domain, sequential from the first shard):

| metric | value |
|---|---|
| distinct IDs observed | **147,010 / 256,128 = 57.4%** |
| **never observed** | **109,118 = 42.6%** |
| IDs covering 50% of all token mass | **129** |
| IDs covering 90% | 10,120 |
| IDs covering 95% | 22,279 |
| **IDs covering 99%** | **62,108** |
| IDs covering 99.9% | 112,513 |
| IDs seen >= 100 times | 27,158 |
| IDs seen >= 1,000 times | 4,750 |

Per-domain distinct IDs (8M tokens each): dclm 100,813; wiki 97,556;
open-web-math 93,338; pes2o 86,645; algebraic-stack 49,654; arxiv 49,242;
**starcoder 35,672**.

**Why the tail is dead:** of the 109,118 never-used IDs, **62.5% are
non-ASCII**. Overall the gemma vocab is **32.0% non-ASCII (82,034 tokens), of
which 31,118 (12.2% of the whole vocab) are CJK**, plus 100 `<unusedNN>`
placeholders and 255 byte-fallback tokens. Our corpus is English + code +
math; we are paying full embedding and softmax cost for a large multilingual
vocabulary we never use.

**A ~64k vocab is well-matched to the measured distribution**: 62,108 IDs
already carry 99% of our token mass. This is the proposal's best-supported
claim after cost, and the two reinforce each other.

#### 5b. Robustness re-run (PARTIAL -- read this before quoting 42.6%)

The figures above sample **sequentially from the first shard** of each domain.
A larger randomized re-run (`vocab2.py`: 30M tokens/domain, documents drawn at
random from 4 randomly-chosen shards per domain) was still running when this
report was finalized. Completed domains, vs the sequential run:

| domain | sequential (8M tok) | **randomized (30M tok)** |
|---|---|---|
| algebraic-stack | 49,654 | **106,049** |
| arxiv | 49,242 | **122,691** |
| dclm | 100,813 | **144,900** |
| open-web-math | 93,338 | **112,966** |
| pes2o | 86,645 | **109,542** |
| starcoder | 35,672 | _(pending)_ |
| wiki | 97,556 | _(pending)_ |

Every domain roughly doubles its distinct-ID count under randomized sampling
at 4x the tokens. **dclm alone already reaches 144,900 distinct IDs (56.6% of
the vocab)**, which is a hard lower bound on the eventual union -- so the union
across all 7 domains will exceed the sequential run's 147,010, and the true
"never used" fraction is **materially below 42.6%**.

**Consequence for the recommendation:** the *magnitude* of the dead-vocab claim
is softer than the headline 42.6%, and could plausibly land near ~30-40%. The
claim that survives regardless is the **concentration** one -- 129 IDs cover
50% of mass and ~62k cover 99% -- because that is a property of the frequency
distribution, not of how many rare IDs are eventually touched at least once.
**Rest the ~64k sizing argument on the 99%-mass rank, not on the never-used
count.** Refresh from `/tmp/tokan/vocab_util_random.json` (Aurora) when the
job completes.

## Caveats

- **Sample size / sampling bias -- the one genuinely unfinished measurement.**
  The 56M-token utilisation figure reads documents **sequentially from the
  first shard** of each domain (e.g. dclm has 1,380 shards; only `fused_0001`
  was touched). The randomized 30M-tokens/domain re-run completed **5 of 7
  domains** (all but starcoder and wiki) before this report was finalized, and
  every completed domain roughly **doubled** its distinct-ID count -- see
  section 5b. So **57.4% is a lower bound on utilisation and 42.6% is an upper
  bound on waste**, and the true waste is materially lower. The rank-for-99%
  statistic is unaffected by this sampling issue and is what the ~64k
  recommendation rests on. Refresh from `/tmp/tokan/vocab_util_random.json`
  (Aurora, `/tmp` -- copy it off before it is reaped) when the run finishes.
- **Fertility sample is ~300KB per domain**, not the 10-50MB suggested. It is
  enough to separate a 17-26% code gap from a 2-4% prose gap, but the
  sub-5% differences are not resolved to high precision.
- **No custom tokenizer was trained.** The counterfactual "what fertility
  would a 64k BPE trained on *our* mix achieve" is **not measured**. All
  comparisons are against off-the-shelf tokenizers. A well-trained 64k
  domain-specific BPE would very plausibly beat gemma on code while shrinking
  the vocab -- this analysis does not refute that; it only refutes the *stated
  reasons* (digits, LaTeX, indentation).
- **LaTeX and Python fertility use synthetic hand-written samples** (repeated
  20x), not sampled corpus files, because arxiv/algebraic-stack are stored
  pretokenized and mixed. The real-corpus rows (arxiv, algebraic-stack,
  starcoder) are the load-bearing ones; treat the two `*_synthetic` rows as
  illustrative.
- **Parameter counts are analytic, not measured** from a live checkpoint. They
  exclude RoPE buffers and any optimizer state. The 30B row uses an assumed
  dim=6144; no 30B flavor exists in the registry.
- **The `2.0e9` / `3.0e10` denominators** in the requested grid are nominal.
  The true `agpt_2b` total is 1.987B, so percentages against 2.0e9 are ~0.7%
  optimistic.
- **GSM8K = 0.0000 was not re-measured here**; it is taken from the proposal's
  own eval table. This experiment only establishes that *digit tokenization is
  not the cause*, not what the cause is.
- **Vocab utilisation says nothing about training value.** A rarely-used token
  is not necessarily worthless; this measures frequency, not contribution to
  loss.

## Bottom line for the proposal

| Section 3 claim | verdict |
|---|---|
| 256k vocab is expensive at 2B | **SUPPORTED, and understated** -- 1,049M / 52.8%, not 525M / 26% |
| Vocab is oversized for our corpus | **SUPPORTED** -- 99% of mass in top 62,108 IDs, 50% in 129. (The "42.6% never used" figure is an upper bound; see 5b)  **Qualified by [exp07](exp07-custom-tokenizer-feasibility.md):** the mass concentration is real, but a custom 64k fitted to it measures 2.5% WORSE than gemma on fertility -- the oversizing costs embedding params, not tokens. Shrink by adopting Llama-3 128k, not by training a 64k. |
| Single-digit tokenization needed | **REFUTED** -- gemma already does it, perfectly and consistently |
| Digit handling contributes to GSM8K=0 | **REFUTED as stated** -- cannot be improved on this axis |
| Wastes tokens on LaTeX / units | **NOT SUPPORTED** -- at parity or better vs Llama-3.1 |
| Wastes tokens on code indentation | **NOT SUPPORTED** for indentation (1 token up to 24 spaces); **SUPPORTED for code overall** (+17% starcoder, +26% Python) |
| Keep 128-alignment | untested, no reason to doubt |

**Recommended edit to Section 3:** keep the tokenizer change, restate the
rationale as (i) embedding cost at small scale, corrected to ~53%, (ii) the
measured mass concentration (99% of tokens in ~62k IDs, so ~64k is the
data-driven size), (iii) a measured +17% code-token penalty. Delete
the single-digit and LaTeX/units bullets, or replace them with "preserve
gemma's existing one-token-per-digit scheme, which is already correct."
Additionally: **weight tying is a strictly cheaper way to recover 525M params
at 2B** and should be evaluated first, since it needs no new tokenizer, no
re-tokenization of the 4.674T-token corpus, and no loss of checkpoint
comparability. `agpt_2b_tied` already exists in the registry.
