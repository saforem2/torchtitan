# AuroraGPT-2B post-training: complete status

**Last updated: 2026-11-12.** Every number is quoted from a run doc — links go to the source. Most recent first.

> **One-line status:** B2 (`checkpoint-93`, 0.205 GSM8K-CoT) still the best CoT model. CPT-FWE 4N at 34.3k/190k (17.8% budget, `12483437 R` MFU 17.1% healthy) via `afterany 12483446 H` to 50k. Pretok **landed** `12483443`: `91,481 -> 36,934 packed @8192` (3.4G, `/tegu/datasets/agpt2b-distill-cot-mix-len8192`, mix-cache `cba5c64` hit) — broad SFT unblocked but **parked to CPT 50k** per `POST-TRAINING-PLAN.md` phase 2.

---

## 2026-11-12 — CPT 34.3k/190k healthy + pretok landed (broad SFT parked to 50k)

CPT-FWE 4N (`agpt_2b_mds_mix_owm_edu_6040` 60/40 FWE, `PPN=8 GBS=64 seq=8192`, 100B=190734 steps) now `12483437 R 00:34` `step 34298 loss 1.93 MFU 17.1%` `tps 4,571` — **17.8% of 100B budget**, healthy (MFU recovered 13.4%->17.1% across chain `12483388 step 26742` -> `12483416 32800->33303` -> `12483437`). Chaining every `02:00` via `afterany 12483446 H` to 50k then general SFT; latest on disk `step-34200`. Eval at `step-17000` was `0.000` GSM8K-CoT by design (pure CPT, no `chat_template`) vs B2 `0.205` which required stage-1 SFT first — not a regression. Narrow SFT-on-CPT (`agpt2b_b2v2_distill_cot_2n.sh` on CPT-17000 HF, LR 2e-5 3ep GBS 384) scored **0.015** vs B2 `0.165` — format collapse on FWE-shifted manifold, hparams not CPT evidence.

Pretok **landed**: `_pretokenize_distill_cot_mix_1n.sh` job `12483443` `Exit` (NPROC=1, `rl-monarch-torch214-locality-20261001`, bypass `ezpz_setup_job` MPI wrapping, `LOG_DIR=logs/pretokenize-distill-cot-mix-<id>`) — `91,481 rows -> 36,934 packed seqs @8192` (3.4G, `dataset_info.json` + 8x `data-*.arrow`), mix-cache **hit** `cba5c64` in 5.8s (6 prior jobs e.g. `12483441` segfaulted at NPROC=1 via `3553055 Segmentation fault` until VENV-path fix). Broad SFT via `sft_distill_cot_4n.sh --pretokenized_dataset /tegu/datasets/agpt2b-distill-cot-mix-len8192` now **ready but parked** to CPT 50k per `POST-TRAINING-PLAN.md` phase ordering — no training until 50k checkpoint converts to HF.


---

## Deliverables — use these

| Purpose | Checkpoint | Why |
|---|---|---|
| **CoT / math reasoning** | **B2 = `checkpoint-93`** | 0.205 GSM8K-CoT, format 0.985. Best of everything tried; drift-proof under GRPO |
| **General instruct** | **`checkpoint-900-hf`** | IFEval prompt-strict 0.253 and base-LM intact. Also B2's stage-1 model |
| **Arithmetic (RL)** | `sft_arithmetic` 8N final | accuracy_reward ~0.4 → ~0.9 over 1000 GRPO steps |

> [!WARNING]
> **Do NOT use `checkpoint-8672`.** It is the final checkpoint of the full-mix run and catastrophically forgot: hellaswag 0.593 → **0.273**, arc_easy 0.694 → **0.298** — at or near random chance. Use `checkpoint-900` from the same run. The recipe page's header once called 8672 "the deliverable" — fixed 2026-08-16; its own body was always right.
Base for everything below is the MDS stage-3 checkpoint `global_step138650` (not a distinct artifact — the 2B MDS is one continuous 140k-step run whose "stages" are data-mix transitions) unless noted.

---

---

## 2026-11-11 — Sunspot CPT-FWE 4N + pretok + GRPO launched — _provisional, jobs live_

CPT-FWE 4N (`agpt_2b_mds_mix_owm_edu_6040` 60/40 FWE, `PPN=8 GBS=64 seq=8192`, 100B=190734 steps) chain `12483048 0→4091 → 12483075 4091→7869 → 12483245 7800→10143+ R` (walltime 02:00 chained, ckpt every 200, base `gs138650/step-0`, VENV `xpu-torch216-sync-5a7a622f16`) — new low **loss 1.856 at 10108** MBH, MFU 18.16%, `~5.3%` of budget. Pretok `12483253 R -- 1N` (`_pretokenize_distill_cot_mix_1n.sh` 50/50 OpenThoughts+gsm8k-r1cot → `/tegu/.../len8192 num_proc=4`, gate `dataset_info.json`+`[decontam]`) unblocks stage-2 SFT `sft_distill_cot_4n.sh`. GRPO `12483252 R -- 3(2+1) 06:00` on B2 `checkpoint-93-hf 0.205` (`gs=8 lr=5e-6 100 steps`) — Monarch TorchStore+XPU [monarch.md](rl/monarch.md). SFT narrow `12483074 10/10 11.02→10.72 completed`. _Provisional_: `R` numbers not exit-verified; finalize on `12483245/252/253` exit with pretok `dataset_info.json`+decontam + GSM8K-CoT delta vs `B2=0.205`. Journal: [journal](../../docs/journal.md).

---

## 2026-11-05 — 16N Sunspot fanout wired (CPT now executing, see 2026-11-11 above)

`685ce083b` fans Sunspot to 16N: `scripts/cpt_fineweb_edu_4n.sh`, `cpt_olmo25_dolmino75_4n.sh`, `sft_distill_cot_4n.sh`, `rl_grpo_b2_2n.sh`, plus `sft_distill_cot_pretok_8n.sh` / `_pretokenize_distill_cot_mix_1n.sh`. CPT already streams correctly — Grain (`olmo-mix-1124` / FineWeb-Edu via `torchtitan/datasets`) streams + tokenizes on the fly with no pre-tokenize step and is verified for pre-training/CPT at 2N/8N. **SFT does not use Grain.** SFT goes through TRL `SFTTrainer` / HF `datasets` (`torchtitan/experiments/ezpz/rl/datasets_sft.py` → `train_sft.py`), which re-tokenizes the whole mix at job start and builds HF arrow caches per rank — that path hit both the multi-rank HF cache coherency race and the `~8.5h` re-tokenize walltime at 93M rows (killed `12470281`), so the distill mix was shelved 2026-11-04. The fanout fixes only the SFT data path: `train_sft.py --pretokenize_to` runs once, then training loads `input_ids` via `load_from_disk --pretokenized_dataset` (`skip_prepare_dataset` when `input_ids` present) from shared `/tegu/datasets/agpt2b-distill-cot-mix-len8192` — no HF streaming cache on compute, `HF_DATASETS_OFFLINE=1`.

Wiring repairs `40b29ca`/`d551d04`/`927dd1e`/`d49706b`/`d962fd6`/`39e65d9`/`4867112` fixed VENV (`xpu-torch216` for CPT vs LD-first for RL consolidate), HF consolidate, and `--optim.` tyro namespace. Pre-tok `NPROC 32→16→8→4` (`15f3e19`/`3665e3a`/`11df0ee3a`) fixed `SIGBUS`/`SIGSEGV` on `distill pack`. Scripts present on HEAD; gate is verified `dataset_info.json` + `[decontam]` counts before training jobs start. Monarch XPU gate unchanged — see [monarch.md](rl/monarch.md) (`12479177` 40/40 auto→Gloo Exit 0). Full wiring in [journal 2026-11-05](../../docs/journal.md).


## 2026-11-04 — B2v2: more stage-2 epochs on the same narrow data hurts

**Question:** B2 got 0.205 with 3 epochs on 7.4k gsm8k-r1cot examples (93 steps). Can more epochs on the same narrow data lift the ~0.2 ceiling?

**Answer: no — it went backwards.** Training looked healthy; eval halved. Same failure mode as every single-stage rebuild.

| Checkpoint | CoT acc | Format | Gen len | Unclosed | vs B2 0.205 |
|---|---:|---:|---:|---:|---|
| B2 = `checkpoint-93` | **0.205** | 0.985 | 282 | ~0 | — |
| B2v2 ep3 final (6 steps) | **0.09** | 0.86 | 416 | 22 | **-0.115** |
| B2v2 ep6 final (12 steps) | **0.105** | 0.91 | 378 | 15 | **-0.10** |

Eval: 200-problem GSM8K-CoT, `temperature 0.0 max_tokens 768`, `frameworks/2026.1.0 vLLM 0.26 XPU fp32` on Sunspot `workq`. Jobs **12482572** (`x1922c4s2b0n0 00:01:38`) and **12482573** (`x1922c4s5b0n0 00:01:39`), both `Exit_status=0`. Artifacts: `outputs/evals/cot/gsm8k-cot-b2v2-ep{3,6}-final.jsonl` (243K/235K) and `submit_cot_eval.sh.o12482572/73` on `/lus/tegu`. Also see [journal](../../docs/journal.md) for the eval LD fix chain.

**Run:** 2N Sunspot, `torchtitan/experiments/ezpz/rl/scripts/sft/agpt2b_b2v2_distill_cot_2n.sh`, base `outputs/sft/agpt-2b-gs138650-tulu-math-uc-mix-8n-gbs6144/checkpoint-900-hf`, TRL `SFTTrainer` FSDP `full_shard`, LR 2e-5 cosine, batch 2 / GAS 8 / seq 2048 (GBS 384), 3→6 epochs (6→12 steps), `dataset_num_proc=8`, bf16, `save_strategy=epoch`. W&B **`xakno7kg`** (ep3) / **`ur6pqqhp`** (ep6), checkpoints at 2/4/6+final and 2/4/6/8/10/12+final. Loss 0.8069, acc 0.8198 — descends cleanly.

**Original intent was the broad distill mix:** `distill_cot_mix` = 50% OpenThoughts-114k-decontam + 50% gsm8k-r1cot (frontier CoT, GSM8K-test-decontaminated, same `<think>/<answer>\boxed{}` envelope, mix-spec tunable). **Shelved 2026-11-04 on the TRL SFT path** — OpenThoughts streaming + multi-rank HF arrow-cache coherency fails at 2N (rank 0 writes new cache, others look for old shards) plus 429 storms. Fell back to `gsm8k-r1cot` alone for this test (Grain for CPT is not affected). Pre-tokenize to re-enable — see 16N fanout above.

**Env fixes (5 commits 2026-11-04, `git log -- rl/scripts/sft/agpt2b_b2v2*`):** venv (`ezpz_get_job_env` clobbered `VIRTUAL_ENV` to stale `xpu-torch214` without `trl`; fixed to `rl-monarch-torch214-locality-20261001`, verified `import trl`), LD (`${VENV}/lib` UR/SYCL must win over system oneAPI 2026.0), `dataset_num_proc` 32→8 (16 OOMs in `map`), HF cache race, staged `~/stage/utils.sh` (no outbound net on compute).

**Takeaway:** do not extend narrow stage-2 further. Same verdict as B3/B4 below.


---

## 2026-08 — B3/B4: every single-stage rebuild failed

After B2's two-stage result, we tried to do it in one shot. All collapsed — varying mix weights, length filtering, and sequence length did not help.

| Run | What varied | Outcome |
|---|---|---|
| **B3** — [instruct+CoT mix](sft/agpt/2b-mds/b3-instruct-cot-mix/design.md) | tulu + OpenR1 0.25 in one SFT, seq 8192 | **0.05** — long-CoT dilution |
| **B4a** — [finish-and-reweight](sft/agpt/2b-mds/b4-finish-and-reweight/README.md) | bolt B2's finishing stage onto B3 base | **0.02**, format 0.26, 133/200 unclosed |
| **B4b** — [finish-and-reweight](sft/agpt/2b-mds/b4-finish-and-reweight/README.md) | reweight gsm8k-r1cot→0.40 + length-filter, seq 4096 | **0.065** — fixed the symptom, not the accuracy |

B4b is the sharpest datapoint: it restored format (0.86) and verbosity (gen_len 611, 27 unclosed) to healthy levels and still only moved 0.05→0.065. Run-on style was a symptom, not the cause. B4a is explicitly refuted — a short finishing stage cannot un-teach a verbose base. B2v2 above is the same story with more epochs instead of a new base: training metrics healthy, eval halved.

---

## 2026-07 — B2: two-stage SFT is how 0.205 happened

The only recipe that worked. General math first, then narrow GSM8K CoT — not both at once.

- **Stage 1:** `global_step138650` → `tulu_math_uc_mix_full` (full OpenMathInstruct-2, 53.3M seqs, 1 epoch, seq 1024, 8N) → `checkpoint-900`. This is the general-instruct deliverable: IFEval prompt-strict **0.253**, base-LM intact. The final ckpt `8672` forgot (see warning above) — long cosine kept LR >1e-5 for ~4000 steps on narrow math.
- **Stage 2:** `checkpoint-900` → `gsm8k-r1cot` (7.4k ex, 93 steps, 3 epochs, LR 2e-5 cosine) → `checkpoint-93` = **B2: 0.205 CoT, format 0.985, gen_len 282**. Every single-stage variant on the same base landed 0.02–0.065 — a 3–10x gap.

B2 is also drift-proof under GRPO (format held 1.000 for 100 steps) — see RL below.

---

## 2026-06 to 2026-07 — Stage-1 SFT: how we got the base for B2

| Run | What varied | Outcome |
|---|---|---|
| [metamathqa-729](sft/agpt/2b-mds/tulu_math_uc_mix/README.md) | tulu 0.65 / metamathqa 0.15 / ultrachat 0.20, seq 1024, 3 ep | **Complete.** IFEval 0.1645→**0.2440** (+8pp, ~4–5σ); base-LM unchanged; GRPO smoke 0.117→**0.922** |
| [full big-mix](sft/agpt/2b-mds/tulu_math_uc_mix_full/README.md) | full OpenMathInstruct-2, 53.3M seqs, 1 ep (~54B tok) | **ckpt-900 is the deliverable** (IFEval 0.253, intact). **ckpt-8672 catastrophically forgot** |
| [v2-256n](sft/agpt/2b-v2-256n/tulu_math_uc_mix/README.md) | same recipe, v2 production base (`step92859`, 4.674T) | **Not delivered** — blocked on 32N oneCCL scale crash (2N smoke green 2.08→1.7, 32N deterministic GPU fault, jobs 12470254/258/262) |

**Lesson from ckpt-8672:** LR 2e-5 stayed above 1e-5 through step ~4350 (cosine only bites in the second half), so ~4000 steps on a narrow math distribution destroyed general capability. The 729-step run survived *because* it stopped early — that stop was load-bearing, not designed. **Rule: cap full-mix SFT at O(1000) steps or drop LR substantially.**

---

## 2026-06 to 2026-09 — RL / GRPO: format, not accuracy

Two working XPU stacks. TRL+vLLM-server is validated to 3N/24 ranks; **Monarch+TorchStore+vLLM (2-tile) is what every agpt-2b LoRA and the whole CoT campaign actually ran on.** Both are real — different workloads. Operator guide: [monarch.md](rl/monarch.md) (current) and [trl.md](rl/trl.md) (legacy, deprecated for new runs).

| Run | Task | Outcome |
|---|---|---|
| [sft_arithmetic 8N](rl/grpo/aurora2b/sft_arithmetic/README.md), 1000 steps | sum_digits | **accuracy_reward ~0.4 → ~0.9** — a genuinely solved task |
| SFT-vs-baseline smoke | sum_digits | SFT-729 last-10 **0.922** vs baseline **0.117** (7.9×) |
| v4 / v5 / v6 difficulty study | alphabet_sort | lever is **task difficulty, not LR** |
| [beat-v5 sweep](rl/grpo/beat-v5-sweep.md) (lr / rank / batch) | alphabet_sort | all three plateau at **0.24–0.25** mean |
| [ceiling-attack](rl/grpo/ceiling-attack.md) | alphabet_sort | componentized reward → **0.667 cross-scored, +168%** |
| [CoT stage 2](rl/plans/cot.md), weak base | GSM8K CoT | **reward-hacking**: reward 0.118→0.217 while accuracy 0.16→**0.105** |
| [gated GRPO on B2](rl/plans/cot.md), 100 steps | GSM8K CoT | format held **1.000**; accuracy **0.205→0.215 (noise)** |

**Three findings worth carrying:**

1. **The plateau was the reward function, not capacity.** Four runs varying LR, rank, batch all plateau at 0.24–0.25. Componentizing the reward (format/completeness/order) reached 0.667 on the same metric, re-scored offline with v5's own char-ratio reward. **Rule: always re-score a shaped reward on the shared original metric.**

2. **Reward-hacking is real and was caught.** Reward rose while eval accuracy fell and format bled 0.955→0.635. Root cause: down-weighting format 0.2→0.05 ("it's saturated") removed a guardrail the *SFT*, not the reward, had been holding up.

3. **The hack was downstream of a weak cold-start, not GRPO.** Gating the reward drove it to 0.000, revealing **95% of rollouts had no `<think>` block and 20% echoed the exemplar's `\boxed{5}`**. On strong B2 the identical setup is drift-proof.

**What RL bought, and did not:**

- **Bought:** a working in-tree XPU GRPO capability (zero core edits); one solved task; format perfection with no drift; +168% shaping result; **RL-as-an-SFT-probe** — convergence speed cleanly separates checkpoints.
- **Did not buy:** any GSM8K-CoT accuracy gain. At ~20% solve rate with `group_size=4`, most groups are all-wrong, so the advantage is zero. No LoRA adapter has been promoted to a production checkpoint.

> [!NOTE]
> **XPU landmine before any RL run:** `generator.model_dtype=float32` — bf16 through vLLM makes agpt-2b emit gibberish (vocab 256000 + ffn 11008 accumulate enough bf16 error to flip greedy argmax, compounding over decode; ~30% slower but correct). Also: attention `flex` with `max_autotune=False`, reward `similarity_power=1`, `--no-drop-zero-std-reward-groups` (without it, cold-start groups are all dropped and no train step fires).

**Validated infra:** Sunspot two-host `12479177` (40/40 rollouts v0–3, 4 pushes/pulls, 3 finite updates, DCP checkpoints, Exit 0) — see [monarch.md](rl/monarch.md) and [journal 2026-09-30](../../docs/journal.md). TRL was last validated 2026-07-06 (`trl.md`) — not running now.

---

## Findings worth carrying

- **Structure beats mix.** B4b restored format/verbosity and still only moved 0.05→0.065. Style was a symptom. B4a: a short finishing stage cannot un-teach a verbose base. B2v2: more epochs on the same narrow data also hurt.
- **More SFT tokens on narrow data actively hurts.** The full-mix 4000-step run and B2v2's extra epochs both degraded despite healthy training metrics. Early stopping was load-bearing.
- **~0.2 is near this base's GSM8K-CoT ceiling.** Two-stage SFT at 0.205; GRPO on B2 at 0.205→0.215 (noise); every alternative at 0.02–0.105. The next lever is a stronger cold-start or a larger/math-pretrained base — not recipe tuning.

---

## Next steps — what to do, and what not to

1. **Run the pre-tokenized distill mix.** This is the only open path that adds *new* CoT data under the proven two-stage structure: `OpenThoughts-114k-decontam × gsm8k-r1cot` via the 16N fanout (`scripts/sft_distill_cot_4n.sh` + `_pretokenize_distill_cot_mix_1n.sh`, `685ce083b`). Gate: verified `/tegu/datasets/agpt2b-distill-cot-mix-len8192` (`dataset_info.json` + `[decontam]` counts) on Sunspot; then train — do not invent a new single-stage mix.
2. **If continuing RL on B2, raise LR and group size.** Format now holds 1.000 under GRPO, so it is safe to increase LR to 5e-6–1e-5 and `group_size` to 8–16 for denser gradients. Separate open item: re-run ckpt-900 GRPO with more walltime — it previously hit 6h at ~step 20 and never converged.
3. **Do NOT attempt further single-stage SFT rebuilds.** Refuted across B3/B4a/B4b and now B2v2 (more narrow epochs also hurt). Either extend *within* B2's two-stage recipe with new broad data (item 1), or move to a larger / math-pretrained base — both are larger lifts, scope separately.
4. **Known blocker — v2-256n 32N oneCCL scale fault.** Same recipe on the completed 4.674T base is blocked at 32N (2N smoke green, 32N deterministic GPU fault, jobs 12470254/258/262). Comparison to gs138650 SFT stays open but unblockable until the fault is fixed; see [v2-256n README](sft/agpt/2b-v2-256n/tulu_math_uc_mix/README.md).

---

## Appendix — known doc inconsistencies (all resolved 2026-08-16)

Recorded 2026-08-13 so readers aren't misled. Kept (struck) for audit trail — two were not errors at all.

- ~~`tulu_math_uc_mix_full/README.md` calls `checkpoint-8672` "the deliverable" in its header and "NOT 8672" in its body.~~ **RESOLVED** — header now says NOT the deliverable (hellaswag 0.593→0.273); body was right.
- ~~The [SFT index](sft/README.md) does not list B3/B4 and points at forgotten 8672.~~ **RESOLVED** — warning block now records the 8672 fix and explains why B3/B4 live here, not in that table (different row shape — ablations, not base→recipe pairs).
- ~~v2-256n "blocked" in SFT index and "in progress" in its README, same day.~~ **RESOLVED by checking disk** — `outputs/sft/agpt-2b-v2-256n-tulu-mix-32n-gbs6144/` has zero `checkpoint-*` dirs, so "checkpoint-100 saved" did not survive. README now reads NOT DELIVERED.
- ~~`rl/trl.md` still says the Monarch path is "blocked" (2026-07-06)~~ **RESOLVED** — corrected at [`rl/trl.md`](rl/trl.md#what-this-does-not-solve); Monarch verified working 2026-07-19.
- ~~"B1" is never defined~~ **RESOLVED: there is no B1, and that is correct.** The B-series labels CoT experiments; B2 (two-stage) was the first. Single-stage runs above predate the naming — referred to by recipe, not B-number. Not renumbered to avoid breaking cross-references.
- ~~400-step `cot-long` **never evaluated**; "likely drifting" is an inference.~~ **RESOLVED — measured, and the inference was wrong.** LoRA checkpoints gone, but `rollout_samples.jsonl` survived (6,337 scored rollouts, `AnswerCorrectReward` per sample). Accuracy 0.069→~0.17 over ~120 steps then flat to 400 (held-out 0/20→5/20); format 0.88–0.99. No drift — apparent collapse was `ThinkFormatReward` reading only `content` while vLLM split `<think>` into `reasoning_content`. Full table in [`rl/plans/cot.md`](rl/plans/cot.md); scorer at [`rl/scripts/score_rollouts.py`](../../rl/scripts/score_rollouts.py).
