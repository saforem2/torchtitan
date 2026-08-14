# AuroraGPT-2B post-training: complete status

**Last updated:** 2026-08-13. Covers all SFT and RL/GRPO work on the 2B model.
Every number here is quoted from a run doc; links go to the source.

> [!IMPORTANT]
> **Accuracy lives in SFT structure. RL perfects format.**
>
> Two-stage SFT (general math -> then GSM8K CoT) scores **0.205** on
> 200-problem GSM8K-CoT. Every single-stage rebuild lands **0.02-0.065** --
> a 3-10x gap that survived varying mix weights, length filtering, and
> sequence length. 100 steps of GRPO on top of the best SFT model moved
> accuracy **0.205 -> 0.215**, i.e. about two problems: noise.
>
> **~0.2 is near this base's GSM8K-CoT ceiling.** The next real lever is a
> stronger cold-start SFT or a larger / math-pretrained base -- not more
> recipe tuning and not more RL.

## Deliverables (use these)

| purpose | checkpoint | why |
|---|---|---|
| **CoT / math reasoning** | **B2 = `checkpoint-93`** | 0.205 GSM8K-CoT, format 0.985. Best of everything tried, and drift-proof under GRPO |
| **General instruct** | **`checkpoint-900-hf`** | IFEval prompt-strict 0.253 **and** base-LM intact. Also B2's stage-1 model |
| **Arithmetic (RL)** | sft_arithmetic 8N final | accuracy_reward ~0.4 -> ~0.9 over 1000 GRPO steps |

> [!WARNING]
> **Do NOT use `checkpoint-8672`.** It is the final checkpoint of the full-mix
> SFT run and it catastrophically forgot: hellaswag 0.593 -> **0.273**,
> arc_easy 0.694 -> **0.298** -- at or near random chance. Use
> `checkpoint-900` from the same run instead. The recipe page's header still
> calls 8672 "the deliverable"; that line is wrong and its own body contradicts
> it.

## SFT experiments

Base is the MDS stage-3 checkpoint `global_step138650` unless noted. (That base
is not a distinct artifact -- the MDS 2B is one continuous 140k-step run whose
"stages" are data-mix transitions.)

| run | what varied | outcome |
|---|---|---|
| [metamathqa-729](sft/agpt/2b-mds/tulu_math_uc_mix/README.md) | tulu 0.65 / metamathqa 0.15 / ultrachat 0.20, seq 1024, 3 ep | **COMPLETE.** IFEval prompt-strict 0.1645 -> **0.2440** (+8pp, ~4-5 sigma); base-LM unchanged; GRPO smoke last-10 0.117 -> **0.922** |
| [full big-mix](sft/agpt/2b-mds/tulu_math_uc_mix_full/README.md) | full OpenMathInstruct-2, 53.3M seqs, 1 ep (~54B tok) | **ckpt-900 is the deliverable** (IFEval 0.253, base-LM intact). **ckpt-8672 catastrophically forgot** |
| [v2-256n](sft/agpt/2b-v2-256n/tulu_math_uc_mix/README.md) | same recipe, v2 production base (`step92859`, 4.674T tok) | **Not delivered** -- blocked on the 32N oneCCL scale crash |
| **B2** (two-stage) | ckpt-900 -> gsm8k-r1cot, 3 ep | **0.205** CoT, format 0.985, gen_len 282 |
| [B3](sft/agpt/2b-mds/b3-instruct-cot-mix/design.md) | one combined SFT, +OpenR1 0.25, seq 8192 | 0.05 -- long-CoT dilution |
| [B4a](sft/agpt/2b-mds/b4-finish-and-reweight/README.md) | bolt B2's finishing stage onto the B3 base | **0.02**, format collapsed to 0.26, 133/200 unclosed |
| [B4b](sft/agpt/2b-mds/b4-finish-and-reweight/README.md) | reweight gsm8k-r1cot to 0.40 + length-filter, seq 4096 | 0.065 -- fixed the symptom, not the accuracy |

### Two findings worth carrying

**Structure beats mix.** B4b is the sharpest datapoint: it restored format
(0.86) and verbosity (gen_len 611, 27 unclosed) to healthy levels and accuracy
still only moved 0.05 -> 0.065. Run-on style was a *symptom*, not the cause.
B4a is explicitly refuted -- a short finishing stage cannot un-teach a verbose
base.

**More SFT tokens actively hurt.** LR 2e-5 stayed above 1e-5 through step ~4350
(cosine only bites in the second half), so ~4000 steps on a narrow math
distribution destroyed general capability. The 729-step run survived *because*
it stopped early -- that stop was load-bearing, not designed. **Rule: cap
full-mix SFT at O(1000) steps or drop the LR substantially.**

## RL / GRPO experiments

Two working XPU stacks. TRL+vLLM-server is validated to 3N/24 ranks;
**Monarch+TorchStore+vLLM (2-tile) is what every agpt-2b LoRA and the whole CoT
campaign actually ran on.** Both are real -- they serve different workloads.

| run | task | outcome |
|---|---|---|
| [sft_arithmetic 8N](rl/grpo/aurora2b/sft_arithmetic/README.md), 1000 steps | sum_digits | **accuracy_reward ~0.4 -> ~0.9** -- a genuinely solved task |
| SFT-vs-baseline smoke | sum_digits | SFT-729 last-10 **0.922** vs baseline **0.117** (7.9x) |
| v4 / v5 / v6 difficulty study | alphabet_sort | lever is **task difficulty, not LR** |
| [beat-v5 sweep](rl/grpo/beat-v5-sweep.md) (lr / rank / batch) | alphabet_sort | all three plateau at **0.24-0.25** mean |
| [ceiling-attack](rl/grpo/ceiling-attack.md) | alphabet_sort | componentized reward -> **0.667 cross-scored, +168%** |
| [CoT stage 2](rl/plans/cot.md), weak base | GSM8K CoT | **reward-hacking**: reward 0.118 -> 0.217 while accuracy 0.16 -> **0.105** |
| [gated GRPO on B2](rl/plans/cot.md), 100 steps | GSM8K CoT | format held **1.000**; accuracy **0.205 -> 0.215 (noise)** |

### Three findings worth carrying

**The alphabet_sort plateau was the reward function, not capacity.** Four runs
varying LR, LoRA rank, and batch all plateau at 0.24-0.25. Componentizing the
reward (format / completeness / order) reached **0.667 on the same metric**.
The comparison is honest: all completions were re-scored offline with v5's own
char-ratio reward, and the reconstruction reproduces the stored rewards exactly.
**Rule: always re-score a shaped reward on the shared original metric.**

**Reward-hacking is real and was caught in the act.** Reward rose while eval
accuracy fell and format bled 0.955 -> 0.635. Self-diagnosed as a design error:
down-weighting format 0.2 -> 0.05 ("it's saturated") removed a guardrail that
the *SFT*, not the reward, had been holding up.

**The hack was downstream of a weak cold-start, not of GRPO.** Gating the reward
drove it to exactly 0.000, revealing that **95% of rollouts emitted no `<think>`
block and 20% literally echoed the exemplar's `\boxed{5}`**. On the strong B2
base the identical setup is drift-proof: format held 1.000 for 100 steps.

### What RL bought, and did not

**Bought:** a working in-tree XPU GRPO capability (zero core edits, runtime
monkeypatches only); one genuinely solved task; format perfection with no drift;
the +168% shaping result; and **RL-as-an-SFT-probe** -- convergence speed cleanly
separates checkpoints, which is arguably the most useful thing it produced.

**Did not buy:** any accuracy gain on the metric we care about. At ~20% solve
rate with `group_size=4`, most groups are all-wrong, so the advantage is zero.
No LoRA adapter has been promoted to a production checkpoint.

> [!NOTE]
> **XPU landmine, worth knowing before any RL run:** set
> `generator.model_dtype=float32`. bf16 through vLLM makes agpt-2b emit
> gibberish -- vocab 256000 + ffn 11008 accumulate enough bf16 error to flip the
> greedy argmax, and it compounds over decode. ~30% slower but correct.
> Also: attention `flex` with `max_autotune=False`, reward `similarity_power=1`,
> and `--no-drop-zero-std-reward-groups` (without it, cold-start groups are all
> dropped and no train step ever fires).

## Next steps (docs' own priority order)

1. **Stronger / longer cold-start SFT** -- more epochs, better CoT data, or STaR
   self-distillation. Accuracy lives here.
2. **If continuing RL:** raise LR into 5e-6 - 1e-5 (safe now that format is
   bulletproof at 1.0) and `group_size` to 8-16 for denser gradient.
3. **Do NOT attempt further single-stage SFT rebuilds** -- refuted across
   B3/B4a/B4b. Either extend *within* B2's two-stage recipe, or move to a larger
   / math-pretrained base. Both are bigger lifts and should be scoped
   separately.

Also open: re-run the ckpt-900 GRPO with more walltime (it hit 6h at ~step 20,
never converged); the 32N oneCCL scale fault that blocks the v2-256n base.

## Known doc inconsistencies

Recorded 2026-08-13 so nobody is misled by them:

- `tulu_math_uc_mix_full/README.md` calls `checkpoint-8672` "the deliverable" in
  its header (line 7) and "NOT 8672" in its body (line 177). **The body is
  right.**
- The [SFT index](sft/README.md) does not list B3 or B4 at all, and its
  checkpoint column points at the forgotten 8672.
- v2-256n is "blocked" in the SFT index and "in progress" in its own README,
  both dated the same day. **Treat it as not delivered.**
- `rl/trl.md` still says the Monarch path is "blocked" (2026-07-06); it was
  verified working 2026-07-19.
- "B1" is never defined -- the B-series labels start at B2.
- The 400-step `cot-long` run was **never evaluated** on the real metric; its
  "likely drifting" status is an inference, not a measurement.
