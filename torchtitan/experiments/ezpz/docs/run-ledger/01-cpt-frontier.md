# 01 — CPT-FWE 4N Frontier Ledger (60/40 FineWeb-Edu Kind-Shift)

> **Branch:** `ezpz-cpt111k-eval-sft` @ `a6135a79` (Sunspot `~/torchtitan`) — ledger
> covers `torchtitan/experiments/ezpz/` only.
> **One-line status:** CPT-FWE 4N at **143.4k / 190.7k (75.2% of 100B budget)** —
> chain healthy on `12h` walltime; frontier `~143400` DCPs live on
> `/home/foremans/torchtitan/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64`
> (Tegu HF root holds `74000/74200` `3.7G` early gates only).
> Every job ID, loss, MFU, and budget number below cites its source doc + section.

---

## 1. What is running — chain snapshot and job table

### 1.1 Current assignment snapshot

| Field | Value | Source |
|---|---|---|
| Config | `agpt_2b_mds_mix_owm_edu_6040` (60% FineWeb-Edu / 40% OWM) | `config_registry.py:1046-1057` |
| Base | `agpt-2b-mds-gs138650/step-0` (MDS 2B, pre-stage-3) via `FaultTolerantTrainer` model-only fork | `cpt_fineweb_edu_4n.sh:16-23` |
| GBS / seq_len | `GBS=64` (`LBS=2 * PPN=8 * 4N=32r`), `seq_len=8192` | `cpt_fineweb_edu_4n.sh:72-76` |
| Tokens/step | `64 * 8192 = 524,288` | derived `cpt_fineweb_edu_4n.sh:76` |
| Budget | `100B / 524288 = 190,734 steps` | `cpt_fineweb_edu_4n.sh:77-78` |
| Assignment progress | **143,400 / 190,734 = 75.2%** | assignment prompt |
| Live PBS header | `select=4`, `walltime=12:00:00`, `q=workq`, `A=datascience`, `filesystems=tegu:home` | `cpt_fineweb_edu_4n.sh:2-7` |
| VENV | `/lus/tegu/.../venvs/xpu-torch216-sync-5a7a622f16` + LD-first | `cpt_fineweb_edu_4n.sh:52-70` |
| Launch | `qsub -l select=4 -l walltime=12:00:00 -v NHOSTS_TRAIN=4 cpt_fineweb_edu_4n.sh` | `cpt_fineweb_edu_4n.sh:12` |

Budget math (from script values, no inference):

```
NGPUS_ACTIVE = NHOSTS_TRAIN * PPN = 4 * 8 = 32
GBS          = NGPUS_ACTIVE * LBS * GAS / (TP*PP*CP) = 32 * 2 * 1 / 1 = 64
tokens/step  = GBS * SEQ_LEN = 64 * 8192 = 524,288
steps        = TRAIN_TOKENS / tokens_per_step = 100,000,000,000 / 524,288 = 190,734
```

Validated by `cpt_fineweb_edu_4n.sh:72-78` computing `TRAINING_STEPS` identically.

### 1.2 Every CPT job in the chain (chronological)

No job ID or loss below is invented. Each row traces to a doc section that
quotes the job's log or `qstat` output at the time of writing. Inferences are
labeled `[INFERENCE]`.

| # | Job ID | Steps | Tip training loss | MFU | Frontier on disk | Walltime | Exit / state | Source |
|---|---|---|---|---:|---:|---|---|---|---|
| 1 | `12483048` | `0 -> 4091` | `2.068` | `19.3%` | `step-4000` | `02:00:00` | `E -29` (walltime) | `POST-TRAINING-2B.md:57` `journal.md:132-134` |
| 2 | `12483075` | `4091 -> 7869` | `2.463` | `18.35%` | `step-7800` | `02:00:00` | `E -29` (walltime) | `POST-TRAINING-2B.md:57` `journal.md:132-134` |
| 3 | `12483245` | `7800 -> 10143+` | **1.856** @ `10108` (new low) | `18.16%` | `50` ckpts `step-200..10000` | `02:00:00` | `R 01:14` live at doc time | `POST-TRAINING-2B.md:57` `journal.md:96-104,130-136` |
| 4 | `12483388` | `10143 -> 26742` | — | ~`13.4%` (chain low, then recovers) | `step-26000` range | `02:00:00` | chained via `afterany` | `POST-TRAINING-2B.md:35` `journal.md:118` |
| 5 | `12483416` | `26742 -> 32800-33303` | — | recovering | `step-32800-33303` | `02:00:00` | chained | `POST-TRAINING-2B.md:35` `journal.md:118` |
| 6 | `12483437` | `33303 -> 34298` | `1.93` @ `34298` | `17.1%` | `step-34200`, `tps 4571`, `tflops ~47` | `02:00:00` (`00:34` elapsed at observation) | `R` | `POST-TRAINING-2B.md:35-36` `journal.md:118` |
| 7 | `12483799` | `34298 -> 75000` | — (`22.4G` frontier, `375` ckpt dirs) | `~18%`, `~2k steps/hr` | `step-75000` | `02:00:00` (last `2h` hop) | `R` at doc time | `POST-TRAINING-2B.md:19` |
| 8 | `12483856` | `100000 -> 111200` | `23G` frontier, `556` ckpt dirs | `15.8%`, `tps 4625`, `tflops 47.03` | `step-111200` | `12:00:00` (`stime 07:29:06`, `06:04` elapsed, `~05:56` remaining) | `R` | `POST-TRAINING-2B.md:5-10` |
| 9 | *(continuation, no new job ID in docs)* | `111200 -> 143400` | [INFERENCE: not quoted in any doc at branch point; assignment snapshot] | [INFERENCE] | `~717` ckpt dirs [INFERENCE: `143400/200`] | `12:00:00` | `R` [INFERENCE: assignment prompt states `143.4k`] | assignment prompt + `thread-a-cpt-trajectory.md:7-9,79-95` |

**How to read this table:**

- Jobs 1-7 ran on `walltime 02:00:00` chaining (`qsub ... walltime=02:00` with
  `afterany` / 600s-poll auto-requeue). Each hop delivered `~4k` steps.
- Job 8 is the first `12:00:00` hop after fix `99e918b342` — delivered `+11.2k`
  steps in `06:04` (`~1.9k steps/hr`). Source: `POST-TRAINING-2B.md:5,10,12`.
- Row 9 is the assignment snapshot advance (`111.2k -> 143.4k = +32.2k steps`);
  no additional job ID is documented on `ezpz-cpt111k-eval-sft` at `a6135a79`
  past `12483856`. The docs note the chain "continued on the corrected 12h
  walltime after `99e918b342`" (`thread-a-cpt-trajectory.md:92-95`).

### 1.3 Checkpoint storage

| Store | Path | Content at branch point | Source |
|---|---|---|---|
| DCP frontier (live) | `/home/foremans/torchtitan/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64` (+ double-nested `outputs/outputs/checkpoints` with symlink `outputs/checkpoints -> outputs/outputs/checkpoints`) | `556` dirs `23G` at `111.2k`; `~717` dirs at `143.4k` [INFERENCE for 143.4k] | `cpt_fineweb_edu_4n.sh:41-44,83-89` `POST-TRAINING-2B.md:5,10` `thread-a-cpt-trajectory.md:79-95` |
| HF converts (Tegu) | `/lus/tegu/.../torchtitan/outputs/evals/cpt-fwe/step-$s/hf` | `74000/74200` `3.7G` `9/9` files verified (patched `config/tokenizer` omission) | `POST-TRAINING-2B.md:5,19-22` `thread-a-cpt-trajectory.md:537-545` |
| HF converts (home) | `~/torchtitan/outputs/hf*` | empty — next convert at `112k` gate per `POST-TRAINING-2B.md:5` | `POST-TRAINING-2B.md:5` |

### 1.4 CKPT_DIR symlink fix

The trainer writes to `outputs/` relative to `SUBMIT_DIR`, so prior runs
double-nested to `outputs/outputs/checkpoints/`. The script fixes resume:

```bash
if [[ -d "${SUBMIT_DIR}/outputs/outputs/checkpoints" && ! -e "${SUBMIT_DIR}/outputs/checkpoints" ]]; then
  ln -s outputs/checkpoints "${SUBMIT_DIR}/outputs/checkpoints"
fi
```

Source: `cpt_fineweb_edu_4n.sh:39-44`. Verified active in `POST-TRAINING-2B.md:10`
(`checkpointer.folder ./outputs/outputs/checkpoints/... at step 100000
(fault-tolerance restart)` — symlink fix active).

---

## 2. Why — kind-shift rationale, budget math, and walltime fix

### 2.1 Kind-shift rationale: 60/40 FWE vs 94.8% DCLM

**The base is DCLM-saturated.** `olmo-mix-1124` is **94.8% DCLM** by weight
(`torchtitan/experiments/ezpz/docs/evals/mmlu-letter-prior-at-chance.md:188-196`
and `production/sft/fineweb-edu-60-40-kind-shift.md:30-42`, cited in
`thread-a-cpt-trajectory.md:262-278`). More olmo-mix tokens cannot help —
the 2B plateau benchmarks were flat over the final ~635B tokens (HellaSwag
`0.560`, ARC-Easy `0.651`, val loss `~2.80`), per `cpt/README.md:38-51`.

**Dolmino was a degree tweak, not a kind change.** Dolmino redistributes
`5.62pp` of DCLM into `peS2o +6.88pp`, `FLAN +1.97pp`, `math +1.36pp`
(`mmlu-letter-prior-at-chance.md:188-196` table reproduced in
`thread-a-cpt-trajectory.md:264-286`). It is the same DCLM-dominated web
distribution with a larger academic slice. The dolmino CPT pilots degraded
benchmarks despite lowering loss (see Sec. 3).

**FineWeb-Edu is a kind change.** FWE is classifier-filtered educational web
(`HuggingFaceFW/fineweb-edu`, ODC-By 1.0, filter trained for pedagogical
content — textbook prose, explanations, Q&A — vs DCLM's generic quality
classifier). Academic content in olmo-mix is `2.0%` and is research papers
(`peS2o`/`arxiv`), not pedagogy. Source: `thread-a-cpt-trajectory.md:287-295`
and `production/sft/fineweb-edu-60-40-kind-shift.md`.

Our **60/40 FWE** mix is `40% OWM + 60% FineWeb-Edu` via
`GrainDataLoader(ConcatThenSplitPacking(DatasetMix(...)))` with weights
counting post-packing 8192-token rows (token-weighted). Source:
`config_registry.py:970-1021` and `cpt_fineweb_edu_4n.sh:18-19`.

**External evidence that this kind matters for MMLU specifically:**

| Evidence | Result | Source |
|---|---|---|
| FineWeb-Edu paper (arXiv:2406.17557) @ 350B, 1.7B model | FWE **37.5 vs DCLM 35.5 on MMLU**; DCLM **62.3 vs 60.1 on HellaSwag** — FWE trades a small HellaSwag deficit for a clear MMLU gain | `thread-a-cpt-trajectory.md:300-305` |
| Same paper @ 38B tokens | **33.6% MMLU at 38B** where next-best dataset needs **300B** (~8x token efficiency) | `thread-a-cpt-trajectory.md:301-303` |
| SmolLM2 (arXiv:2502.02737 Sec 4.3) stage-1 `60% FWE / 40% DCLM` | reaches `33.6% MMLU`; DCLM-vs-FWE split predicts our observed profile (MMLU at chance, commonsense intact) | `thread-a-cpt-trajectory.md:307-310` |
| OLMo-2 1B on `olmo-mix-1124` base, Dolmino mid-training | **MMLU 26.9 -> 44.3 (+17.4)** (arXiv:2501.00656 Table 9) — strongest in-family proof that a distribution shift on this exact base can unlock MMLU at small scale | `cpt/README.md:104-106` `thread-a-cpt-trajectory.md:323-329` |

Our dolmino CPT did **not** reproduce OLMo-2's `+17.4` MMLU gain; the ledger
diagnosis (per `thread-a-cpt-trajectory.md:283-286,328-329`) is that dolmino
was the wrong *kind* of mid-training at `89% DCLM`, not that mid-training is
ineffective.

### 2.2 Budget math (derived from script values)

From `cpt_fineweb_edu_4n.sh:72-78`:

```
LBS=2; GAS=1; SEQ_LEN=8192; TP=1; PP=1; CP=1
GBS = NGPUS_ACTIVE * LBS * GAS / (TP*PP*CP) = 32 * 2 * 1 / 1 = 64
tokens/step = GBS * SEQ_LEN = 64 * 8192 = 524,288
TRAIN_TOKENS = 100_000_000_000
TRAINING_STEPS = 100_000_000_000 / 524_288 = 190,734
```

Divisor is `GBS * seq_len = 64 * 8192` (not `GBS` or `seq_len` alone).
Source: `cpt_fineweb_edu_4n.sh:77-78` `Thread-a Sec. 1.1` table.

Budget share: `143400 / 190734 = 75.2%` (assignment snapshot).

### 2.3 The 12h walltime fix: `99e918b342` — 4k to 24k steps/hop

**Problem:** The chain launched at `walltime=02:00:00` (`PBS -l walltime=02:00:00`
in early `cpt_fineweb_edu_4n.sh`; see `POST-TRAINING-2B.md:10-12,19` and
`journal.md:132-138`). Each hop delivered `~4k` steps (e.g. `0->4091` in
`12483048`, `4091->7869` in `12483075`). With 600s-poll auto-requeue /
`afterany` chaining, the walltime overhead dominated and the chain needed
`~48` hops to reach `190k`.

**Fix:** Commit `99e918b342` synced `cpt_fineweb_edu_4n.sh` to
`#PBS -l walltime=12:00:00` (`workq` cap `23h`; header `A=datascience`).
Source: `POST-TRAINING-2B.md:10,12,19` and `cpt_fineweb_edu_4n.sh:4,12`.

**Impact:**

| Metric | Before (`02:00`) | After (`12:00`, `99e918b342`) |
|---|---|---|
| Steps per hop | `~4k` | `~24k` (`+11.2k in 06:04` observed at `12483856` = `~1.9k/hr` => `~22.8k/12h`) |
| Hops to `190k` | `~48` | `~8` |
| Measured | `12483048 0->4091`, `12483075 4091->7869` | `12483856 100000->111200 (+11.2k in 06:04)` |

Source: `POST-TRAINING-2B.md:10,12,19` `journal.md:132-138`.

The chain watcher (`600s` poll auto-requeue) remains as a safety net but is
no longer the primary hop mechanism; the 12h PBS walltime carries the chain.
Source: `cpt_fineweb_edu_4n.sh:10-11` `POST-TRAINING-2B.md:12`.

---

## 3. Outcome — what the trajectory actually shows

### 3.1 Training loss trajectory

Reconstructed from docs at branch point `a6135a79` (all numbers quoted from
doc-reported `loss` / `tip loss` fields, not re-derived):

```
step      loss     MFU      source
0         ~12.45   —        expectation: ln(256000) uniform-softmax at step 1
4091      2.068    19.3%    POST-TRAINING-2B.md:57 / journal.md:132-134 (12483048)
7869      2.463    18.35%   same (12483075) — resume jitter / mix warmup spike
10108     1.856    18.16%   POST-TRAINING-2B.md:57 / journal.md:130-136 (12483245) — new low
34298     1.93     17.1%    POST-TRAINING-2B.md:35-36 / journal.md:118 (12483437)
75000     —        ~18%     POST-TRAINING-2B.md:19 (12483799) — no loss quoted
111200    —        15.8%    POST-TRAINING-2B.md:10 (12483856) — no training loss quoted at tip
```

**Reading:** `2.068 -> 1.856` in `~6k` steps is real learning on the new
distribution; `1.856 -> 1.93` over the next `24k` is flat-to-slightly-up but
within seed / data-order noise. The base MDS 2B took `~4.67T` tokens to reach
`~2.65`; FWE reaching `1.93` in `34k` steps (`17.8B` tokens at `524k tok/step`)
is *much* lower — but that is the expected entropy effect (FWE is a
pedagogical subset of web vs DCLM's generic web), not necessarily a capability
gain. The dolmino precedent is the caution: dolmino also beat the plateau on
loss (`2.49 vs 2.80`) while degrading benchmarks. Source:
`thread-a-cpt-trajectory.md:196-235` `cpt/README.md:60-88,131-147`.

### 3.2 Validation loss 12.53 flat — miscalibrated validator, not plateau

At `111200` the doc records `Validation loss 12.53 at 111200 (flat
12.53-12.60 since 110700)` (`POST-TRAINING-2B.md:10`, reproduced in
`thread-a-cpt-trajectory.md:97-136 Sec. 1.3`).

**Why this is not a plateau signal:** the validator is configured
`--validator.enable --validator.freq=100 --validator.steps=10`
(`cpt_fineweb_edu_4n.sh:100`) on the same `DatasetMix`, so validation loss
should track training loss within `~0.1-0.3` nats. A `6-7x` gap (`12.53` vs
`~1.93`) is off by an order of magnitude. Expected training-loss range for a
trained 2B on this vocab is `~2.0-2.6` (`log(vocab) = ln(256000) ~12.45` at
step 1 decaying to `~2.6`); `12.53` is near the *untrained* expectation,
not a trained plateau. Source: `thread-a-cpt-trajectory.md:102-106,134-135`.

**Diagnosis (per `thread-a-cpt-trajectory.md:108-133`):** one of

1. **Missing `global_valid_tokens` divisor (sum vs mean)** — the loss-reporting
   guide `torchtitan/experiments/ezpz/docs/guides/loss-reporting-tp-dist-reduce.md`
   documents exactly this class: `_dist_reduce` over the wrong mesh dropped the
   `dp_world_size` divisor (fixed upstream `d64eabcce`, local workaround still
   in `trainer.py`/`validator.py` as of `2026-07-24`). The inverse
   (over-reporting by `dp_world_size=32`) would turn `~1.93 * ~6.5 ~= 12.5`.
   Diagnostic: diff rank-0 JSONL `loss` vs `val_loss` for `110700-111200`; if
   `val_loss / loss ~= 32` it is the divisor bug.
2. **Validator reading a different mixture or unpacked loss** (raw next-token
   on a non-Grain path or `seq_len` mismatch).
3. **Broken by the symlink fixup** (`cpt_fineweb_edu_4n.sh:41-44`) if it
   re-creates the dataset without `_fineweb_edu_shards()`.

**Action:** ignore `val_loss` as a training signal until fixed; use training
loss + downstream eval only. Source: `thread-a-cpt-trajectory.md:127-136`.

### 3.3 MFU health

| Observation | Value | Source |
|---|---|---|
| Early hops | `19.3% -> 18.35% -> 18.16%` | `journal.md:132-136` |
| `34298` | `17.1%`, `tps 4571` | `POST-TRAINING-2B.md:35-36` `journal.md:118` |
| `75000` | `~18%`, `~2k steps/hr` | `POST-TRAINING-2B.md:19` |
| `111200` | `15.8%`, `tps 4625`, `tflops 47.03` | `POST-TRAINING-2B.md:10` |

The `19.3% -> 15.8%` decay is **not a learning signal**. It reflects longer
walltime hops amortizing startup + Tegu I/O contention as the chain
accumulates `556` ckpt dirs (`23G` frontier) under `PPN=8` x `4N` with
`GrainPacking` over `140` parquet shards. `tps ~4625` / `~47 tflops` is
healthy for this config. Source: `thread-a-cpt-trajectory.md:221-227`.

---

## 4. What went wrong / open risks

### 4.1 Validator bug — diagnosis and impact

See Sec. 3.2. The validator reports `12.53` on a distribution where training
loss is `~1.93`. This is a misconfigured instrument, not evidence of plateau.
The question prompt framing `12.53` as plateau evidence is incorrect; treat it
as a bug to be fixed via the JSONL diff + `validator.py` patch without
blocking the chain. Source: `thread-a-cpt-trajectory.md:97-136`.

### 4.2 Dolmino precedent: loss improved while benchmarks bled

The only completed CPT on this codebase is the 2026-07 dolmino ratio sweep
(`cpt/README.md:53-147`):

| Signal | olmo-100 base plateau | dolmino-100 (5960 steps, ~300B tok) | olmo50-dolmino50 |
|---|---|---|---|
| Train loss start -> end | `~2.80 -> ~2.80` (plateau) | `7.27 -> 2.480` (`-0.31` vs base) | `7.24 -> 2.590` (`-0.20`) |
| HellaSwag `acc_norm` | **0.560** | **0.486 (-7.4pp)** | **0.491 (-6.9pp)** |
| ARC-Easy `acc` | **0.651** | **0.547 (-10.4pp)** | **0.610 (-4.1pp)** |
| PIQA `acc_norm` | **0.733** | **0.687** | `~0.69` |

Source: `cpt/README.md:61-98` (table), job `8647850` 12-point screen, plots
`cpt/figures/cpt_loss.svg` + `cpt/figures/cpt_eval.svg`.

**Two distinct damage modes** (`cpt/README.md:117-129`):

1. **Ratio-independent HellaSwag drop** (`~-6-7pp` at `step-1000`, never
   recovers, both mixes) — attributed to **LR re-warm shock** (base at
   convergence re-warmed to full `2.28e-5` with `warmup=200`,
   `decay_ratio=0.8` — `cpt/README.md:118-122,156-157`). The gentle retry
   (`LR=2e-6 constant`, `warmup=20`, jobs `8648114`/`8662867`) stopped at
   `step-1200` with `12` checkpoints, never evaluated — **hypothesis still
   open** (`cpt/README.md:81-88,242-248`).
2. **Ratio-dependent ARC-Easy bleed:** dolmino-100 collapses monotonically
   (`0.619 -> 0.547`) while `olmo50-dolmino50` **holds `~0.61`**
   (`cpt/README.md:122-125`). Pure dolmino specifically forgets ARC-Easy;
   keeping olmo in the mix prevents it.

**Implication for FWE:** our arm uses `40% OWM / 60% FWE` with `LR=2e-6`
constant (`config_registry.py:923-927` / `cpt_fineweb_edu_4n.sh:20`) — it keeps
a large math/OWM slice and avoids the dolmino re-warm shock by construction.
If the ARC-Easy bleed was pure-dolmino-specific, this arm should be safer.
But **we have not measured it** — no CPT checkpoint has ever been scored on
MMLU in either format (`cpt/README.md:100-116`), and the dolmino damage was
visible by `step-1000` (`~50B` tokens, `1/6` of the `300B` pilot), far earlier
than our `143.4k` frontier. The absence of a measured drop is not evidence of
health; loss alone cleared dolmino too. Source: `thread-a-cpt-trajectory.md:182-195`.

### 4.3 No CPT checkpoint has ever been MMLU-scored

> "No CPT checkpoint has ever been scored on MMLU in either format (MCF vs
> CF)." — `cpt/README.md:100-116` `thread-a-cpt-trajectory.md:185-186,370-395`

Every AuroraGPT MMLU number is `MCF` (letter scoring) at `0.2413-0.2655`,
best constant-`D` `0.2689` beats all 12 configs — chance-level with letter
prior `p(A,B,C,D)~(0.37,0.32,0.20,0.11)` (`mmlu-letter-prior-at-chance.md`),
not a knowledge signal. Correct eval is `max(CF,MCF)` per OLMES. No CPT
checkpoint has been converted and scored with both; the dolmino CPT's 12-point
screen covered `HellaSwag/ARC-Easy/PIQA` only (job `8647850`). Source:
`thread-a-cpt-trajectory.md:312-395` `cpt/README.md:100-116`.

### 4.4 Other open risks

- **Early CPT eval showed `0.000` by design, not regression.** `step-17000`
  raw CPT scored `0.000 GSM8K-CoT` — expected for pure CPT with no
  `chat_template` (`POST-TRAINING-2B.md:22,35-36` `journal.md:118`). Narrow
  SFT-on-CPT (`cpt34k-gsm8k-3ep` `0.015`) vs `cpt34k-distill-broad-4n` `0.000`
  format collapse is a hparam/format-vs-reasoning gap, not CPT evidence
  (`POST-TRAINING-2B.md:22-23`).
- **CKPT_DIR double-nesting.** The workaround is a symlink, not a move; new
  chains writing to Tegu (`/lus/tegu/.../outputs/checkpoints/...`) avoid `/home`
  fill per user request. See Sec. 1.4.
- **No downstream curve exists for FWE.** In dolmino, damage was monotonic
  and ratio-dependent on ARC-Easy. For FWE the equivalent (HellaSwag hold
  through `10k/50k/100k`, ARC-Easy hold at `40%` OWM) has zero points on
  either curve — the ledger is a loss-only record until the scoring ladder
  runs. Source: `thread-a-cpt-trajectory.md:232-237`.

---

## 5. Next steps — gates, ladder, and manual requeue

### 5.1 150k gate (G1-G4) — thresholds to continue to 190k

All must hold to continue; failure modes and fallbacks are explicit. Source:
`thread-a-cpt-trajectory.md:475-518` `POST-TRAINING-2B.md:164-178` (for the
broad-distill fallback).

| Gate | Metric | Base | Threshold to CONTINUE to 190k | SE | Source |
|---|---|---|---:|---|---|
| **G1** | HellaSwag `acc_norm` | `0.560` (2B-256 plateau) | `>= 0.540` (within `-2pp`) | `~0.004` | `thread-a:482` `cpt/README.md:95` |
| **G1b** | ARC-Easy `acc` | `0.651` | `>= 0.621` (within `-3pp`) | `~0.006` | `thread-a:483` `cpt/README.md:96` |
| **G1c** | PIQA `acc_norm` | `0.733` | `>= 0.713` (within `-2pp`) | `~0.005` | `thread-a:484` |
| **G2** | ARC-Challenge `acc_norm` 25-shot | `~0.31` | `>= 0.290` (within `-2pp`) | — | `thread-a:485` |
| **G3** | MMLU `max(CF,MCF)` 5-shot | `0.2413-0.2655` (chance band) | `>= 0.240` (no regression) | `0.0036` | `thread-a:486` |
| **G4** | GSM8K-CoT `acc` post-SFT probe | B2 = `0.205` | `>= 0.205` (match or beat) | `~0.03` at `N=200` | `thread-a:487` `POST-TRAINING-2B.md:44-48,107-115` |
| **G4b** | GSM8K-CoT `format` (closed `\boxed{}`) | B2 = `0.985` | `>= 0.90` | — | `thread-a:488` |

**Interpreting near-misses** (`thread-a-cpt-trajectory.md:490-509`):

- `G1-G2` fail by `>2pp` but `G4` passes: **continue to 190k** — commonsense
  dip is the expected FWE HellaSwag trade (DCLM `62.3 vs 60.1` in the ablation);
  re-check at `190k` and revisit the `75/25` blend if it bleeds monotonically.
- `G1-G2` pass but `G4` fails (`0.08-0.12` like B2v2/B3/B4): **stop CPT**, fork
  SFT from the **best intermediate checkpoint** and pivot to the broad-distill
  mix per `POST-TRAINING-2B.md:164-178`.
- Everything fails: **stop CPT, revert** to the MDS base for SFT.

### 5.2 E1 scoring ladder — P0-P3, 14 HF converts

Do not convert every 200-step DCP. Convert a log-spaced early + linear late
ladder capturing the shock horizon (`0-10k`) and the current frontier
(`110k-150k`). Source: `thread-a-cpt-trajectory.md:547-569`.

| Phase | Steps to convert | Count | Why |
|---|---:|---:|---|
| **P0 — Shock horizon** | `1000, 5000, 10000` | 3 | Dolmino shock hit by `1000`; flat-LR should not — verify |
| **P1 — Early plateau** | `25000, 50000, 75000` | 3 | `75k` already HF'd (`74000/74200`); add `25k/50k` to fill the gap |
| **P2 — Current frontier** | `100000, 111200, 125000, 143400, 150000` | 5 | `111.2k` documented; `143.4k` assignment snapshot; `150k` gate |
| **P3 — If continuing** | `160000, 175000, 190734` | 3 | Full `100B` budget; `190734` is the exact terminal step |

Total: **14 HF conversions** (`8` immediate: P0+P1+P2 through `143.4k`; `6`
deferred: `150k + P3`). Each is `~3.7G` (`9/9` safetensors + `config.json` +
`tokenizer.*`); total HF on Tegu `~52G` for the full ladder — trivial against
the `23G` DCP frontier. Source: `thread-a-cpt-trajectory.md:560-569`.

For the SFT probe, only **3 checkpoints need SFT-training**: the best P2
candidate, the `150k` gate, and (if degrading) the last-good earlier step.
Source: `thread-a-cpt-trajectory.md:565-569`.

Converter uses `flavor=2b-mds` — wrong flavor = silent gibberish
(`torchtitan/experiments/ezpz/eval/convert_to_hf.py` +
`hf_batch_convert_cpt_fwe.sh:58`). Source:
`thread-a-cpt-trajectory.md:544-545`.

### 5.3 Manual requeue command

No watcher requeue; manual hop required when the 12h walltime expires.

```bash
# Sunspot — auto-resumes from CKPT_DIR via FaultTolerantTrainer
qsub -l select=4 -l walltime=12:00:00 \
  -v NHOSTS_TRAIN=4,NGPU_PER_HOST=8 \
  torchtitan/experiments/ezpz/scripts/cpt_fineweb_edu_4n.sh
# At 06:04/12:00, 12483856 had ~05:56 remaining (POST-TRAINING-2B.md:10);
# expiry ~13:29 UTC that hop. Resume picks last good CKPT_INTERVAL=200 dir
# (CKPT_DIR auto-resume, no explicit step needed).
```

Source: `POST-TRAINING-2B.md:12` `cpt_fineweb_edu_4n.sh:12,79-80,94-103`
`thread-a-cpt-trajectory.md:448-456`.

### 5.4 CKPT_DIR symlink fix

The script applies the symlink fixup before every launch (see Sec. 1.4).
No manual intervention needed — the guard is idempotent:

```bash
# Inside cpt_fineweb_edu_4n.sh:41-44 — runs before ezpz_setup_job
if [[ -d "${SUBMIT_DIR}/outputs/outputs/checkpoints" && ! -e "${SUBMIT_DIR}/outputs/checkpoints" ]]; then
  ln -s outputs/checkpoints "${SUBMIT_DIR}/outputs/checkpoints"
fi
```

For **new chains** (post-`190k` or fresh experiments), set `CKPT_DIR` to Tegu:

```bash
export CKPT_DIR=/lus/tegu/projects/datascience/foremans/projects/saforem2/torchtitan/outputs/checkpoints/agpt-2b-cpt-fineweb-edu-n4-gbs64
qsub -l select=4 -l walltime=12:00:00 -v NHOSTS_TRAIN=4 cpt_fineweb_edu_4n.sh
```

The live chain stays on `/home` until `190k`; after the budget completes,
`rsync` to Tegu and leave the symlink (Sunspot storage topology: `/home` +
`/lus/tegu` same `1.2P` Lustre, `PBS filesystems=tegu:home`). Source:
`cpt_fineweb_edu_4n.sh:81-89` and memory guidance on Sunspot outputs.

---

## Appendix — source index for this ledger

Every claim above traces to one of these files. No job ID or loss is cited
without a line or section reference in the tables.

| Source file | What it provides |
|---|---|
| `torchtitan/experiments/ezpz/scripts/cpt_fineweb_edu_4n.sh` | PBS header, GBS/LR/CKPT_DIR/VENV, `MDS_ANNEAL_BASE`, `TRAIN_TOKENS/TRAINING_STEPS` budget math, validator flags, symlink fixup — Secs 1.1-1.4, 2.2-2.3 |
| `torchtitan/experiments/ezpz/docs/production/POST-TRAINING-2B.md` | All 2026-11-10..2026-11-16 job IDs + steps + loss + MFU + frontier sizes; B2/B3/B4/B2v2 baselines; `12h` walltime fix `99e918b342`; HF converts `74000/74200` — Secs 1.2-1.3, 2.3, 3.1-3.2, 4.2 |
| `torchtitan/experiments/ezpz/docs/exploration/thread-a-cpt-trajectory.md` | Trajectory analysis, dolmino comparison, validator bug diagnosis, kind-shift evidence tables, 150k gate + P0-P3 ladder + requeue command — Secs 2.1, 3.2-3.3, 4.1-4.4, 5.1-5.4 |
| `torchtitan/experiments/ezpz/docs/journal.md` | Early chain `0->4091->7869->10143` hops, `1.856` new low at `10108`, `34298 loss 1.93`, pretok + GRPO context — Secs 1.2, 3.1 |
| `torchtitan/experiments/ezpz/docs/production/cpt/README.md` | Dolmino precedent: `7.27->2.480` loss, `HellaSwag -7.4pp / ARC-Easy -10.4pp` degradation, `gentleLR 2e-6` open hypothesis, no CPT MMLU ever scored — Secs 4.2-4.3 |
| `torchtitan/experiments/ezpz/agpt/config_registry.py:919-1057` | `_agpt_2b_mds_mix_base()` flat-LR definition, `DatasetMix` 60/40 construction, `GrainDataLoader` packing semantics, `checkpointer.folder` — Secs 1.1, 2.1-2.2 |
| Assignment prompt | Snapshot `143.4k/190.7k (75.2%)` — Secs 1.1-1.2 |

Inferences are labeled `[INFERENCE]` in Sec. 1.2-1.3 (frontier `~717` dirs,
row 9's loss/MFU/state). All other numbers are direct quotes with section
citations.
