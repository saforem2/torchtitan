# Teaching agpt-2b Chain-of-Thought Reasoning (plan)

> Status: **in progress** (2026-07-20). Stage 0 (eval) + Stage 1 (cold-start CoT-SFT) DONE and validated; Stage 2 (GRPO-RLVR) next. Staged recipe to take the
> instruction-tuned AuroraGPT-2B (`checkpoint-900-hf`, full-mix SFT) from "has
> seen math rationales but does not reliably emit reasoning" to "emits and
> benefits from an explicit `<think>` scratchpad." Everything lives under
> `experiments/ezpz/`; no edits to `experiments/rl/` upstream or `models/**` core.

## TL;DR

Run the **DeepSeek-R1 recipe, 2B-sized**: **cold-start CoT-SFT first, then
GRPO-RLVR** -- on our owned, XPU-validated SFT and GRPO stacks.

- **SFT teaches the model to _emit_ reasoning** (the `<think>...</think>
  <answer>...</answer>` envelope + shallow step-by-step habit).
- **GRPO-RLVR teaches it to reason _well_** (traces that reach verified-correct
  answers), with a **componentized** reward -- the direct application of our
  proven [ceiling-attack result](../grpo/ceiling-attack.md): a saturating/binary
  reward plateaus; decomposing it into additive sub-skills unlocks the gradient
  (+168% there).

**Do not start with RL-only (R1-Zero).** On a 2B model that cannot yet emit
`<think>`, a binary exact-match reward gives near-zero reward variance ->
collapsed advantage -> no learning. R1-Zero is kept only as a *falsifiable
control* in Stage 2. SFT-distillation of CoT is also what R1 itself found best
for small models (R1-Distill-Qwen-1.5B/7B).

**No teacher model is required.** Stage 1 reformats gsm8k's own `#### N`
rationales; the scale-up uses pre-distilled, already-correctness-filtered HF R1
traces (`open-r1/OpenR1-Math-220k`). We distill the *format* from data, not the
*content* from a live teacher.

## Results so far (2026-07-20)

| Stage | checkpoint | format hit-rate | CoT accuracy (greedy, GSM8K/200) |
|-------|-----------|-----------------|----------------------------------|
| baseline (instruct) | `checkpoint-900-hf` | **0.00** | 0.15 |
| Stage 1 cold-start | `agpt2b-gsm8k-r1cot-8n/checkpoint-16-hf` | **0.955** | 0.16 |

**Stage 1 works.** After only 16 optimizer steps of continue-SFT on `gsm8k-r1cot`,
the model went from *never* emitting the envelope (0.0) to **95.5%** well-formed
`<think>...</think><answer>\boxed{}</answer>`, with reasoning accuracy preserved
(0.15 -> 0.16, within noise) and no regression. Generations are compact (median
~220 chars, zero truncation at the 768-token budget) and every correct answer is
now well-formed. This validates the cold-start premise: SFT cheaply teaches the
*format*; lifting *accuracy* is Stage 2's (GRPO) job.

Notes from the run:
- 8N was over-provisioned for a 7473-example smoke -- packing + 96 ranks gave only
  ~16 optimizer steps in 38s. It sufficed to teach the envelope; a longer run (2N
  / 3 epochs, launcher `..._2n.sh`) is available for a stronger cold-start but is
  not required to unblock Stage 2.
- FSDP `trainer.save_model()` leaves `final/` empty; the per-epoch checkpoints are
  sharded. Consolidate with `accelerate merge-weights` then copy config from the
  base and the tokenizer (WITH chat_template) from the staged dir -- see
  `scripts/eval/consolidate_and_eval_cot.sh`.

## The one thing that gates everything: a CoT eval

There is **no generation-based, chat-templated CoT eval today**. The existing
`5:gsm8k` lm-eval path is loglikelihood/EM and is blind to whether the model
*emits* a reasoning trace (`docs/evals/eval-landscape-2026-07.md` deliberately
defers `gsm8k_cot_llama`). Nothing downstream is measurable without it, so
**Stage 0 is built first.**

---

## Stage 0 -- CoT eval + baseline (prerequisite)

Add a generation-based, chat-templated GSM8K-CoT eval under
`experiments/ezpz/scripts/eval/`:

- Render the prompt with the gemma chat template (`add_generation_prompt=True`),
  sample (greedy for the headline number), then score **two things separately**:
  1. **Format hit-rate**: fraction matching
     `<think>.*?</think>\s*<answer>.*?</answer>` with a parseable answer.
  2. **CoT accuracy**: exact-match on the answer extracted **only from the
     `<answer>` / `\boxed{}` span after `</think>`**.
- **Baseline `checkpoint-900-hf` now**, before any training, so Stage 1 has a
  real before/after.

**`extract_answer` caveat (`rl/tasks/common.py:12`).** It tries `\boxed{}` ->
`#### N` -> ... -> **last standalone number** (`:46`). That last-number fallback
will grab intermediate CoT scratch numbers. For CoT scoring (eval *and* the
Stage 2 reward) we must **strip `<think>` first and require an explicit terminal
marker** (`\boxed{}`/`####`) -- do not rely on the last-number fallback.

**Checkpoint identity.** Deliverable = `checkpoint-900-hf` from
`docs/production/sft/agpt/2b-mds/tulu_math_uc_mix_full/` (the full-mix SFT; ~5.7B
tokens; the strongest IFEval + GRPO-start per its evals README). Confirm the HF
dir is staged/reachable on the target cluster and use it as
`--model_name_or_path`. Do **not** target the v2-256n base -- blocked on a
scale-only oneCCL crash at 32N.

---

## Stage 1 -- Cold-start CoT-SFT (the format teacher) -- cheapest first experiment

Rides `rl/train_sft.py` verbatim: `assistant_only_loss=True` (`:218`) + the
`{% generation %}` span (`train_grpo.py:587`) already train the entire
`<think>...</think><answer>...</answer>` envelope as assistant tokens -- **no
masking-code change**.

### Smoke (signal in hours, no teacher, no download)

`gsm8k`'s `answer` field **is** step-by-step CoT ending in `#### N`, and gsm8k is
already registered + cached (`datasets_sft.py:345 _build_gsm8k`). Add one builder:

- `_build_gsm8k_r1cot()` mapping each row to
  - prompt: `[{user, question + "\nReason step by step inside <think></think>,
    then give the final answer inside <answer>\boxed{}</answer>."}]`
  - completion: `[{assistant, "<think>{cot_sans_marker}</think>\n<answer>
    \boxed{N}</answer>"}]`
  - **Completion content must start directly with `<think>`** (no leading `\n`):
    the gemma template's `lstrip('\n')` (`train_grpo.py:584/587`) otherwise trips
    TRL's prompt/completion tokenization-mismatch and silently breaks the mask.
- `register_sft_dataset(SFTDataset("gsm8k-r1cot", _build_gsm8k_r1cot, "..."))`.

Launch on the instruct checkpoint (continue-SFT, so we only teach the envelope),
`<=8N`, via `scripts/submit_sft_aurora.sh`:

```bash
qsub -v NHOSTS_TRAIN=8,SFT_DATASET=gsm8k-r1cot,MODEL_PATH=<checkpoint-900-hf>,\
MAX_LENGTH=2048,NUM_EPOCHS=2,LR=2e-5 \
  torchtitan/experiments/ezpz/scripts/submit_sft_aurora.sh
```

**`MAX_LENGTH=2048` is mandatory** -- the default 1024 (`train_sft.py:221`)
silently truncates the `<answer>` tail and destroys the supervised target.

**Gate:** format hit-rate on the Stage 0 eval goes ~0 -> >90%; CoT accuracy no
worse than the instruct baseline; lm-eval `5:gsm8k` EM not regressed
(no-forgetting check).

### Scale-up (once the envelope is learned)

- Add `_build_openr1_cot()` for `open-r1/OpenR1-Math-220k` (R1 traces, **already
  Math-Verify-filtered by the authors** -- do NOT re-run the numeric-only
  `extract_answer` on it; that would drop symbolic/fractional answers). Strip
  native `<think>` wrappers, re-wrap to our canonical schema.
- Mix to prevent forgetting/memorization (zero code -- mix-spec parser exists):
  `--sft_dataset 'openr1_cot:0.4,tulu-3-sft-mixture:0.4,ultrachat-200k:0.2'`.
- **Measure the real trace-length distribution first**, then set `--max_length`
  to cover ~p90-p95 (likely 4096+); do not hard-drop long traces (biases toward
  shallow reasoning).
- **Use the offline two-phase path** for the big mix: `--pretokenize_to <path>`
  on 1N (bakes `assistant_masks` with `assistant_only_loss=True`), then
  `--pretokenized_dataset <path>` for the job -- avoids the startup-tokenize
  idle-watchdog trip. **Pre-cache both HF corpora from a login node** (compute
  nodes are HF-offline).

Effort: **M**. Honest scope: gsm8k CoT is terse worked-math, not reflective
self-correction. This stage teaches the *envelope + shallow emission*, which
makes Stage 2's gradient dense -- exactly its job.

---

## Stage 2 -- GRPO-RLVR (the content teacher)

Take the Stage 1 checkpoint and run GRPO where reward keys **only on the
extracted `<answer>`**, never on the reasoning text. Start on **Path A (TRL,
`rl/train_grpo.py` + `rl/tasks/`)** -- the lower-friction loop. `tasks/countdown.py`
is already R1-shaped (multi-step arithmetic to a target, verifiable;
`accuracy_reward` at `:166`).

### Reward design -- apply the ceiling-attack lesson

Our proven result: a saturating single reward plateaued (~0.25, no lr/rank/batch
lever helped); a **componentized additive** reward hit +168%. Translate faithfully:

- **Do NOT make the dominant component binary exact-match** -- that reproduces the
  saturating trap. Make correctness **dense** where the task allows
  (`1 - |pred-gold|/scale`, or per-step/per-operation credit) so partial
  reasoning earns partial gradient.
- **Separate additive reward functions** (GRPO sums a list of `reward_funcs`):
  - `think_format_reward` (~0.2): well-formed non-empty `<think>` block.
  - `answer_extractable` (~0.1): a parseable `\boxed{}`/`####` answer **after**
    `</think>`.
  - `answer_correct` (dominant): correctness on the extracted answer, dense where
    possible.
  - task anti-hack (countdown: `uses_given_numbers`, expression present,
    full-expression re-eval -- not the `last '= N'` shortcut).
- **On Path B (Monarch overlay) specifically:** these must be **separate
  `RewardFn` classes in the Rubric list**, not one packed reward fn.
  `reward_breakdown` is keyed by reward-fn class name, so packing collapses the
  per-component metrics and defeats the measurement plan. (Learned from the
  ceiling-attack `shaped_reward.py`, which was single-fn on purpose there.)

### Launch (TRL / Path A)

- No change to `train_grpo.py`; add `think_format_reward` + a `<answer>`-span
  extractor to a task (or new `tasks/gsm8k_reason.py` + one import in
  `tasks/__init__.py`).
- **`--max_completion_length` MUST be raised from its 64 default
  (`train_grpo.py:185`)** to ~512-700, or every CoT rollout is truncated to zero
  reward.
- `--num_generations >= 8` (default 4, `:184`), small `beta` (KL),
  `should_std_normalize=True`, `--model_name_or_path <Stage-1 ckpt>`.
- **Pick a difficulty band where CoT is load-bearing but the base gets a nonzero
  correct fraction** (verify with a rollout probe). If baseline accuracy is ~0,
  the RL gradient can't start -- that itself signals Stage 1 needs to be stronger.

### Falsifiable control: R1-Zero

Run GRPO straight on the instruct checkpoint (skip Stage 1), same reward, and
**log reward std / advantage magnitude, not just mean reward.** Prediction:
collapsed advantage without cold-start, healthy gradient with it. This is the
clean, cheap proof that the *recipe* (not just SFT) did the work.

### Path B (Monarch+TorchStore+vLLM) -- later increment, not the MVP

Mirror `rl/alphabet_sort_agpt/` into a new `rl/reason_agpt/` overlay (subclass
env for the `<think>` prompt + one-shot exemplar a la `few_shot_env.py`; separate
`RewardFn` per above; `config_registry.py` entry fn). Drive via
`train_upstream.py --module`. Copy the XPU knobs **verbatim**:

- `generator.model_dtype='float32'` (**mandatory** -- agpt-2b through vLLM in bf16
  emits gibberish),
- LoRA `target_modules=['wqkv','wo']`, `alpha=2*rank`, the `2b-rl` fused-QKV
  flavor (vocab 256000), `LMHeadCastConverter` fp32 head,
- FlexAttention `max_autotune=False`, TP=1/FSDP=1,
  `--no-drop-zero-std-reward-groups`.
- `enable_thinking` is unusable (gemma/`auto` renderer has no reasoning channel);
  reasoning is prompt-driven + regex-scored.

---

## Stage 3 (later / optional) -- STaR self-improvement or long traces

**STaR-2b** -- teacher-free rejection-sampling: sample k traces from the current
model on verifiable problems, keep only verified-correct ones, SFT on them,
iterate. A good complement to grow the SFT set from the model's own traces, but
**only if retargeted at multi-step substrate** (GSM8K / metamathqa / countdown
>=4 numbers). On 1-3-step arithmetic with a "keep shortest correct trace" filter
it teaches format mimicry, not reasoning:

- Invert the length heuristic (require >=2 genuine intermediate steps).
- Cap rationalized (hint-the-answer) traces.
- **Make GSM8K-CoT _transfer_ the go/no-go** -- in-task pass@1 cannot distinguish
  reasoning from mimicry.
- Generation endpoint: `trl vllm-serve` is **not** OpenAI-compatible -- use TRL's
  `VLLMClient` against `/generate/`, `--dtype float32`.

### On a fancier sampler for the STaR step (HMC / MCMC / SMC)

The natural question "use HMC for the rejection-sampling step" has a verified
answer: **no one has, and it does not port directly.** STaR "rejection sampling"
is a **binary correctness filter on discrete token sequences** judged by a
black-box verifier -- not von Neumann density-ratio rejection sampling. HMC needs
a *continuous* state and a *differentiable* log-density; token sequences are
discrete and the correctness indicator is non-differentiable with zero gradient
almost everywhere, so HMC's gradient advantage is unavailable for the part that
matters (a type mismatch, not an engineering gap). Cast as inference, STaR draws
from `p(trace | correct) ~ p_LM(trace) * 1[verifier=correct]`, and the field's
real "smarter sampler" answer is **SMC / amortized inference, not HMC**:

- STaR-as-discrete-MCMC-EM: Phan et al., NeurIPS 2023
  (https://arxiv.org/abs/2312.02179) -- closest bridge; sampler is *discrete*
  MCMC, explicitly not HMC.
- Twisted SMC for reasoning: Feng et al., ICLR 2025
  (https://arxiv.org/abs/2410.01920); Zhao et al.
  (https://arxiv.org/abs/2404.17546) -- SMC is the type-correct generalization
  (rejection sampling = its degenerate uninformative-twist case).
- GFlowNet fine-tuning: Hu et al., ICLR 2024
  (https://arxiv.org/abs/2310.04363) -- amortized posterior sampling over
  rationales.

**Practical read for a 2B model:** the binding constraint is pass@k / acceptance
rate, not sampler mixing -- a cleverer sampler only reconcentrates mass the model
already assigns; it cannot manufacture correct traces on problems the 2B never
solves. Spend effort on the cheap acceptance-rate levers first (temperature/n
sweep, dedup + diversity, RFT-style weighting) and, above all, a
**process/partial-credit reward** (same ceiling-attack lesson) -- which is also
the prerequisite that would make any smarter sampler useful. If chasing the
sampler idea anyway, the type-correct experiment is **twisted SMC over partial
steps with a lightweight learned twist head, A/B'd against temperature-tuned
best-of-n at matched compute** -- not HMC.

**Longer reflective traces** (OpenThoughts-114k, Bespoke-Stratos) only once
Stage 1's envelope is solid and you've measured you need reflection depth.

---

## How we measure success

1. **Format hit-rate** (Stage 0 eval) -- primary Stage-1 gate.
2. **CoT accuracy** (generation-based) at three points: instruct baseline <
   cold-start < post-GRPO.
3. **GRPO health**: per-component reward curves + advantage magnitude; the
   R1-Zero control shows collapsed advantage.
4. **Cross-task generalization**: train on countdown, measure GSM8K-CoT -- the
   only test that separates reasoning transfer from task-format memorization.
5. **No-regression**: lm-eval `5:gsm8k` EM + IFEval/MMLU on the cold-start ckpt.
6. **Scratchpad ablation** (sharpest reasoning-vs-cosmetic test): accuracy with
   the model's own `<think>` vs forced-empty think. If the scratchpad doesn't
   raise accuracy, the CoT is decorative.

## Top risks

- **Eval doesn't exist** -- largest chunk of real build work; gates everything.
  Stage 0 first.
- **Silent truncation** -- `max_length=1024` (SFT, `train_sft.py:221`) and
  `max_completion_length=64` (GRPO, `train_grpo.py:185`) will silently kill the
  target / zero out rollouts. Override to 2048 / ~512-700 and assert in the
  launch wrapper.
- **Numeric-only `extract_answer`** -- strip `<think>` and require a terminal
  marker for eval + reward.
- **Saturating/binary reward** -- keep correctness dense and componentized.
- **Format mimicry vs real reasoning** -- guard with scratchpad ablation +
  cross-task generalization; never report in-task pass@1 alone.
- **Scale/infra** -- stay `<=8N` on the `2b-mds` `gs138650`/`checkpoint-900-hf`
  lineage; the v2-256n base is blocked at 32N. Pre-cache HF datasets from a login
  node.

## What we deliberately did NOT choose, and why

- **R1-Zero (RL-only) as the primary path** -- feasible to run, but a 2B
  non-emitter + binary reward => collapsed advantage. Kept only as the
  falsifiable control.
- **Real `<think>`/`<answer>` special tokens** (`add_tokens` +
  `resize_token_embeddings`) -- perturbs the base embedding matrix and breaks
  clean DCP/HF reload for a marginal single-token-boundary gain. Use multi-token
  literals; enough traces make emission reliable.
- **`enable_thinking=True` renderer route** -- unusable through the gemma/`auto`
  renderer.
- **search_r1 reason-with-*search*** -- needs a dense retrieval server on XPU;
  drop the tool for a first pure-CoT target (mirror its `<think>`/rubric shapes
  only).
- **Distillation from a live local teacher as the default** -- teacher
  availability on XPU is unconfirmed; the HF pre-distilled-traces path makes it
  unnecessary. (A dormant `scripts/distill/gen_cot_traces.py` can drop in later
  if a served teacher appears.)

## References

- [ceiling-attack: reward shaping breaks the ~0.25 wall](../grpo/ceiling-attack.md)
  -- the +168% componentized-reward result this plan applies.
- [beat-v5 tuning sweep](../grpo/beat-v5-sweep.md) -- no config lever beats a
  reward-shape ceiling.
- [monarch.md](../monarch.md) / [trl.md](../trl.md) -- the two GRPO frameworks.
