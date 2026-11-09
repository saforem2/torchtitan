# FineWeb-Edu 60/40 kind-shift: the next pre-training lever for agpt-2B

> **Status 2026-11-09.** No new compute. This note synthesizes evidence already
> on disk: the narrow-SFT degradation (W&B `xakno7kg`/`ur6pqqhp`, jobs
> 12482423/12482485 -> eval 12482572/73) plus the corpus-composition audit in
> [`mmlu-letter-prior-at-chance.md`](../../evals/mmlu-letter-prior-at-chance.md)
> and the CPT results in [`../cpt/README.md`](../cpt/README.md).
> Companion gating note:
> [`../../evals/mmlu-letter-prior-gating.md`](../../evals/mmlu-letter-prior-gating.md).

## TL;DR

**Current pre-training is 94.8% DCLM.** Dolmino mid-training at 89% DCLM does
not change its kind. SmolLM2 stage 1 at **60% FineWeb-Edu / 40% DCLM**
reaches **33.6% MMLU at 38B tokens** where the next-best dataset needs 300B,
and its ablation has FineWeb-Edu beating DCLM on MMLU (37.5 vs 35.5) while
DCLM wins HellaSwag (62.3 vs 60.1) — our exact profile: MMLU at chance,
commonsense intact. Extending narrow SFT on the same 7.4k GSM8K-CoT examples
from 3 to 6 epochs **halved** accuracy (0.205 -> 0.09/0.105) despite healthy
training metrics, replicating the B3/B4 collapse. **The lever is data
distribution at CPT, not more SFT tokens on narrow data.**

Wiring for the 60/40 CPT arm already exists on HEAD (`685ce083b` 16N fanout:
`scripts/cpt_fineweb_edu_4n.sh` + `cpt_olmo25_dolmino75_4n.sh` via Grain
streaming, no pre-tokenize). This note is the rationale for running it.

## 1. Current distribution is DCLM-monoculture

Measured over `data-lists/aurora/olmo-mix-1124.txt` (1,438 files, weights
renormalized):

| corpus | olmo-mix-1124 | dolmino-mix-1124 | delta |
|---|---:|---:|---:|
| dclm | **94.80%** | **89.18%** | -5.62pp |
| starcoder | 2.46% | -- | -2.46pp |
| pes2o | 1.47% | 6.88% | +5.41pp |
| flan | -- | 1.97% | +1.97pp |
| math (open-web-math / algebraic-stack / dolmino-math) | 0.64% | 1.36% | +0.72pp |
| arxiv | 0.53% | -- | -0.53pp |
| wiki | 0.09% | 0.44% | +0.35pp |
| stackexchange | -- | 0.16% | +0.16pp |

Source: [`mmlu-letter-prior-at-chance.md`](../../evals/mmlu-letter-prior-at-chance.md#why-dolmino-is-not-the-intervention-the-literature-credits)
and [`../cpt/README.md`](../cpt/README.md).

**Reading:** Dolmino is *still* a DCLM-dominated web distribution with ~5.6
points redistributed into peS2o plus 2% FLAN and 1.4% math. It is a **change of
degree**, not kind. The CPT sweep already ran dolmino-100 and olmo50-dolmino50
for 300B tokens each (jobs 8638977-80, step-5960) and **both degraded
commonsense** (HellaSwag 0.560 -> 0.486 / 0.491, ARC-Easy 0.651 -> 0.547 /
0.610) despite beating the olmo-100 plateau on loss (2.49 / 2.60 vs 2.80).
See [`../cpt/README.md`](../cpt/README.md#result-pilot-complete-2026-07-05).
Loss on a lower-entropy distribution is not a capability gain.

## 2. SmolLM2 60/40 is a change of kind — with a measured MMLU effect

SmolLM2 (arXiv:2502.02737 Sec 4.3, Fig 6) stage 1 is **60% FineWeb-Edu / 40%
DCLM**, and the FineWeb-Edu paper (arXiv:2406.17557) reports:

- **33.6% MMLU at 38B tokens** where the next-best dataset needs **300B**
  (~8x token efficiency on MMLU).
- Controlled ablation at 350B on a 1.7B model: FineWeb-Edu **37.5 vs DCLM
  35.5 on MMLU**, while DCLM **62.3 vs 60.1 on HellaSwag** — i.e. FineWeb-Edu
  trades a small HellaSwag deficit for a clear MMLU gain.

That trade is **exactly our observed profile**: every AuroraGPT MMLU score
sits at chance (0.2413-0.2655, best constant-D 0.2689; see
[`mmlu-letter-prior-at-chance.md`](../../evals/mmlu-letter-prior-at-chance.md)
and the gating companion note) while HellaSwag 0.57-0.61 and ARC-Easy 0.61-0.66
are healthy. The FineWeb-Edu paper's MMLU-vs-HellaSwag split predicts this
pattern when the pre-training filter is wrong for knowledge.

OLMo-2 is the closest precedent on our own data family: same `olmo-mix-1124`
base, Dolmino mid-training **MMLU 26.9 -> 44.3 (+17.4) at 1B**
(arXiv:2501.00656 Table 9), but only +3.9/+4.1 at 7B/13B — at small scale
mid-training is nearly the entire MMLU signal. Our Dolmino CPT at 2B did not
reproduce that gain, consistent with Dolmino being the wrong *kind* of
mid-training mix at 89% DCLM.

> **Caution from the data-strategy memo:** No corpus cures MMLU at
> arbitrary scale. SmolLM2's own MCF number is **29.62 across 0-6T tokens**
> on the letter formulation; OLMES (arXiv:2406.08446) on OLMo-7B sees MCF
> only emerge past ~400B tokens and recommends `max(CF, MCF)`. FineWeb-Edu
> moves the *knowledge* lever; the *format* lever (MMLU vs
> `mmlu_continuation`, letter binding) is separate and is tracked in the
> MMLU gating note. Run both formats when scoring CPT checkpoints.

References:

- FineWeb / FineWeb-Edu: <https://arxiv.org/abs/2406.17557>
- SmolLM2: <https://arxiv.org/abs/2502.02737>
- OLMo-2: <https://arxiv.org/abs/2501.00656>
- OLMES: <https://arxiv.org/abs/2406.08446>

## 3. Narrow SFT cannot substitute for a distribution shift

### B2v2: more epochs on the same 7.4k examples hurts

B2 is the only SFT recipe that cleared 0.10 on GSM8K-CoT:
`global_step138650 -> tulu_math_uc_mix_full checkpoint-900 -> gsm8k-r1cot`
(7.4k ex, 93 steps, 3 epochs) -> **checkpoint-93 = 0.205 CoT, format 0.985,
gen_len 282** ([POST-TRAINING-2B](../../production/POST-TRAINING-2B.md#2026-07--b2-two-stage-sft-is-how-0205-happened)).

B2v2 asked whether more epochs on the same narrow data could lift the ~0.2
ceiling:

| Checkpoint | CoT acc | Format | Gen len | Unclosed | vs B2 0.205 |
|---|---:|---:|---:|---:|---|
| B2 = `checkpoint-93` | **0.205** | 0.985 | 282 | ~0 | -- |
| B2v2 ep3 final (6 steps) | **0.09** | 0.86 | 416 | 22 | **-0.115** |
| B2v2 ep6 final (12 steps) | **0.105** | 0.91 | 378 | 15 | **-0.10** |

Eval: 200-problem GSM8K-CoT, `temperature 0.0 max_tokens 768`,
`frameworks/2026.1.0 vLLM 0.26 XPU fp32` on Sunspot `workq`, jobs
**12482572** (`x1922c4s2b0n0 00:01:38`) and **12482573** (`x1922c4s5b0n0
00:01:39`), both `Exit_status=0`, artifacts
`outputs/evals/cot/gsm8k-cot-b2v2-ep{3,6}-final.jsonl` on `/lus/tegu`.

Training:

- 2N Sunspot, `torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh`,
  base `outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900-hf`,
  TRL `SFTTrainer` FSDP `full_shard`, LR 2e-5 cosine, batch 2 / GAS 8 /
  seq 2048 (GBS 384), 3->6 epochs (6->12 steps), `dataset_num_proc=8`, bf16,
  `save_strategy=epoch`.
- W&B **`xakno7kg`** (ep3) / **`ur6pqqhp`** (ep6): loss **0.8069**, acc
  **0.8198**, descending cleanly.
- Checkpoints at 2/4/6+final and 2/4/6/8/10/12+final under
  `outputs/sft/agpt2b-b2v2-distill-cot-2n-ep{3,6}/`.

**Training looked healthy; eval halved.** Loss and token-accuracy gave no
warning — the same divergence seen in CPT where dolmino loss improved while
benchmarks degraded.

Original intent was the broad distill mix
(`OpenThoughts-114k-decontam x gsm8k-r1cot`, same `<think>/<answer>\boxed{}`
envelope, mix-spec tunable) shelved 2026-11-04 on the TRL SFT path due to
multi-rank HF cache coherency + re-tokenize walltime; pre-tokenize to
re-enable (16N fanout).

### B3/B4: the same failure in a different shape

| Run | What varied | Outcome |
|---|---|---|
| **B3** — `tulu + OpenR1 0.25` in one SFT, seq 8192 | single-stage mix | **0.05** — long-CoT dilution |
| **B4a** — bolt B2's finishing stage onto B3 base | finish-and-reweight | **0.02**, format 0.26, 133/200 unclosed |
| **B4b** — reweight gsm8k-r1cot->0.40 + length-filter, seq 4096 | reweight | **0.065** — format restored (0.86) but accuracy stuck |

Details: [`POST-TRAINING-2B.md`](../POST-TRAINING-2B.md#2026-08--b3b4-every-single-stage-rebuild-failed)
and [`b4-finish-and-reweight`](agpt/2b-mds/b4-finish-and-reweight/README.md).

B4b is the sharpest control: it restored format/verbosity to healthy levels
and still only moved 0.05 -> 0.065. Style was a symptom, not the cause.
B4a shows a short finishing stage cannot un-teach a verbose base. B2v2 shows
more epochs on narrow data is the same trap with a different knob.

**Pattern:** Every attempt to improve GSM8K-CoT by tuning SFT duration,
mix weight, length filtering, or sequence length on narrow data produced
the same result — training metrics adequate or improving, eval at or below
0.10. The ~0.2 ceiling is a property of the *base + data distribution*,
not of the SFT recipe.

## 4. What 60/40 actually changes

**FineWeb-Edu** is a classifier-filtered educational-web corpus (HuggingFace
`HuggingFaceFW/fineweb-edu`, ODC-By 1.0). Its filter is trained to select
for pedagogical content — explanations, textbook-style prose, Q&A — not for
generic web quality (DCLM's fastText). Concretely:

- DCLM selects the "best" web pages by a quality classifier trained on
  high-quality reference corpora; it retains web *distribution* but filtered.
- FineWeb-Edu selects the *educational subset* of web. The two filters agree
  on many pages but diverge where it matters for MMLU: academic knowledge
  density.

SmolLM2 stage 1 at 60% FineWeb-Edu / 40% DCLM is therefore not "more
filtering" — it is a **different data kind**: majority educational content
with a DCLM floor to preserve general web coverage. Our current 94.8% DCLM
has no educational-filtered component at all (academic content is 2.0%, all
research papers via peS2o/arxiv, not pedagogy).

The 60/40 ratio is not arbitrary: it is SmolLM2's published stage-1 recipe
and the FineWeb-Edu paper's recommended operating point where MMLU gains
are large and HellaSwag regression is small.

## 5. Availability on Aurora/Sunspot (no new tokenization)

FineWeb-Edu is already on the system in two forms:

| Form | Path | Size | Tokenizer |
|---|---|---|---|
| **Fused blendcorpus (gemma)** | `/flare/AuroraGPT/datasets/fineweb-edu-v1.4.0/data-fused-tok/` | **1.551T tokens**, 989 `.bin` / 110 CC snapshots | `gemma-7b` (`tk_gemma.sh`) |
| **Parquet sample (local)** | `/lus/tegu/projects/datasets/datasets/fineweb-edu-100BT/` | 267 GB, 140 parquet files | on-the-fly via Grain |

Plus dolmino slices already fused and gemma-tokenized:

| Slice | Tokens | Use in mix |
|---|---|---|
| `flan` | 17.1B | MC/instruction format exposure |
| `math` | 11.7B | math reasoning |
| `pes2o` / `wiki` | 58.7B / small | science / factual |

Source: [`production/agpt/30b-exp/exp03-1b-proxy-design.md`](../agpt/30b-exp/exp03-1b-proxy-design.md)
and [`journal.md`](../../journal.md) (2026-04-27 local cache).

**An arm is a weighted text file, not a tokenization campaign.** The CPT
data-lists under `data-lists/aurora/olmo25-dolmino75.txt` etc. are built by
`utils/build_cpt_mixes.py` renormalizing per-source weights. For a new
60/40-style mix the same builder can emit a list; no `.bin` generation is
needed.

## 6. Wiring that already exists

The 16N Sunspot fanout (`685ce083b`, 2026-11-05) wired:

- `scripts/cpt_fineweb_edu_4n.sh` — 4N FineWeb-Edu CPT (Grain
  `fineweb_edu_local` streaming, no pre-tokenize)
- `scripts/cpt_olmo25_dolmino75_4n.sh` — 4N olmo25/dolmino75 control
- `scripts/sft_distill_cot_4n.sh` + `_pretokenize_distill_cot_mix_1n.sh` —
  SFT broad distill mix (separate track, HF cache race being fixed via
  pre-tokenize to `/tegu/datasets/agpt2b-distill-cot-mix-len8192`)
- `scripts/rl_grpo_b2_2n.sh` — RL on B2

CPT streams correctly via `torchtitan/datasets` Grain
(`olmo-mix-1124` / `fineweb_edu_local` via `register_local_dataset()`),
tokenizing on the fly with no pre-tokenize step, verified at 2N/8N.
SFT's TRL path is the one that needed the pre-tokenize fix; CPT is
unaffected.

VENV map: CPT uses `xpu-torch216-sync-5a7a622f16` with LD-first
(`.cache/xpu-ld-first` linking `libur_*`, `libsycl.so*`, `libumf*`,
`libtcm*` -> `LD_LIBRARY_PATH`) — see
[`journal.md`](../../journal.md) and [`POST-TRAINING-2B.md`](../POST-TRAINING-2B.md).

## 7. What to measure when it runs

1. **Pre-training metrics:** val loss on the *new* mix is not comparable to
   olmo-mix loss (different entropy). Compare instead:
   - Downstream commonsense (HellaSwag, ARC-Easy, PIQA) vs the olmo-100
     plateau (0.560 / 0.651 / 0.733) — must not repeat the CPT degradation.
   - MMLU **both formats**: `mmlu` (MCF, letters A-D) and
     `mmlu_continuation` (CF, answer text), 5-shot, via the OLMES
     `max(CF, MCF)` recommendation. No CPT checkpoint has ever been scored
     on MMLU in either format.
2. **Post-SFT:** Re-run the B2 two-stage recipe (`checkpoint-900 -> gsm8k-r1cot`)
   on the new base and compare GSM8K-CoT to 0.205. If the lever is real,
   the new base should lift the ceiling that B2v2/B3/B4 could not move.
3. **Cost gate:** 1B-proxy design at 50B tokens / 64 GPUs / ~3h / ~190
   node-hours per arm is in
   [`production/agpt/30b-exp/exp03-1b-proxy-design.md`](../agpt/30b-exp/exp03-1b-proxy-design.md)
   — use it to validate the mix before scaling to 2B.

## Cross-refs

- MMLU gating (why MMLU must not gate promotion):
  [`mmlu-letter-prior-gating.md`](../../evals/mmlu-letter-prior-gating.md)
- Full MMLU letter-prior audit:
  [`mmlu-letter-prior-at-chance.md`](../../evals/mmlu-letter-prior-at-chance.md)
- CPT sweep + gentle-LR hypothesis:
  [`cpt/README.md`](../cpt/README.md)
- Data-strategy memo (corpus acquisition ladder):
  [`data-strategy-after-olmo-mix-2026-07.md`](../../notes/data-strategy-after-olmo-mix-2026-07.md)
- B3/B4/B2v2 source of truth:
  [`POST-TRAINING-2B.md`](../POST-TRAINING-2B.md)
