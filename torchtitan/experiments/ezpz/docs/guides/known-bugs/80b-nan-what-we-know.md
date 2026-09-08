# The 80B NaN: what is known, what is refuted, what is open

> Last rewritten 2026-08-31; substantially updated 2026-09-06 with the first
> per-tensor measurements (`lm_head` dominance, clipping on every step, the
> 3.87e-9 effective LR at the death point) and a correction to what the
> capture instrumentation was capable of. Conclusions first; the investigation log is at the bottom.
> Every number here is either measured on this stack or cited to the job that
> produced it.

## Answer in one paragraph

**Nobody knows why the 80B NaNs.** The long-standing diagnosis -- "bf16
forward-activation overflow in the 84-layer residual stream" -- is refuted on
three independent grounds, each derived from that diagnosis's own evidence.
What replaces it is not a mechanism but a boundary: the failure is real,
reproducible, dp-sensitive, optimizer-independent, and **not** a
magnitude/overflow phenomenon. It is prevented by fp32 activations at a 3.4x
throughput cost. The site is unmeasured, and the experiments that would locate
it have either not been run or were run wrong.

## What is established

| claim | evidence |
|---|---|
| **The failure is real and reproducible** | SophiaG `8574385` (510N) NaN step 18; mano `8661293` (62N, dp~186) NaN step 17. |
| **It is optimizer-independent** | Three optimizers fail in one controlled corner. mano has no Hessian term, so it is not a SophiaG artifact. |
| **It is NOT an overflow** | bf16 max **3.3895e38** vs fp32 **3.4028e38** -- the same 8-bit exponent (measured on this stack). fp32 cannot fix a range problem; it buys mantissa, 8 bits -> 24. |
| **Nothing is anywhere near a numerical limit** | `qk_q_absmax_local` = **61.2** at 48, 72 and 84 layers, and **62.25** at dp=12 in the clean regime (`12472477`). 36 orders below the bf16 ceiling. `grad_absmax` peaks at 0.03. |
| **fp32 activations prevent it at production dp** | `12473149`: 120/120 steps, zero NaN, loss 12.95 -> 8.098, where bf16 `12473142` died at step 30. Confounded by warmup (40 vs 200) in the favourable direction. ~3.4x slower. |
| **Gradient mass concentrates in `lm_head.weight`** | `12474761` step 1-4: `top0 = lm_head.weight` at **0.2520**, next is `layers.65..attention.wo.weight` at **0.005693** -- **44x** smaller. Skew (max/mean) pinned at **217** across every step of three separate runs. Ranks 1-4 are `attention.wo` from layers 65/36/57/5, spanning 2% -- a flat, depth-independent population. |
| **Clipping fires on 100% of steps** | `clip_fired = 1.0` every step; `grad_norm_preclip ~8.07 -> postclip 1.0`. `max_norm=1.0` is the core default and `agpt_80b` does not override it, so **every reported grad_norm in this document is pre-clip**, including `8574385`'s "flat ~6.17". The optimizer saw 1.0. |
| **`8574385` died at an effective LR of 3.87e-9** | Its warmup was 4650 with peak 1e-6, so `lr(18) = 1e-6 * 18/4650`. That is **four orders of magnitude below** the ~7.4e-7 ceiling `agpt_80b.md` documents. Whatever kills it, an over-large LR is not it. |
| **A single `inf` zeroes every other gradient** | `clip_grad_norm_` runs with `error_if_nonfinite=False`, so `scale = max_norm/inf = 0`. Measured: every finite gradient becomes exactly 0.0 while the offender becomes `nan` (`inf*0`). The step is a no-op **before** the trainer skips it -- which is how a run recovers from a non-finite step (`12474403` L72 recovered at 56). |
| **Gradient concentration scales with dp** | Three arms, matched GBS (384 seqs), seed 42, LR trajectory, optimizer, model, verified hosts: `layer_gradnorm_skew` = **163.2** at dp=48 (`12474806`), **216.9** at dp=96 (`12474768`), **273.0** at dp=192 (`12474803`). Within-arm cv < 0.8% against between-arm gaps of 26-33%, so ~35 sigma. Driven by the DENOMINATOR -- mean layer gradnorm falls 22% per doubling while `lm_head`'s falls 2%. dp=48 was a **pre-registered prediction** (172.7 predicted, 162.3 observed, committed at `aa12e8d77` before the run). Per-doubling ratios 1.329 then 1.259 -- decelerating, so dp=384 projects to 325-344. |
| **dp and GBS have opposite signatures** | A 16x GBS change leaves means invariant (0.03-0.20%) and scales variances by sqrt(16) (`12474765` vs `12474768`). A 4x dp change scales means monotonically and leaves variances alone (cv ratios 0.94-1.05). Batch size averages away noise; parallelism changes structure. |
| **No archived failure was LR-voided** | Audit 2026-09-08 of `8574385`, `8661293`, `8673658`, `8671243`, `8537349`, `8530891`: all ran at 1e-6 or below. The 80B submit script sets `LR="${LR:-1e-6}"` (`submit_agpt_80b_aurora_venv_failover.sh:110`) and passes `--optimizer.lr` explicitly, so `agpt()`'s 8e-4 registry default is never reached by that path. A run whose writeup states no LR took 1e-6, not 8e-4. |
| **The LR trajectory alone does not cause the failure** | `12474765`: 25/25 steps, zero non-finite, at `8574385`'s effective LR *at every step* (3.87096768e-09 at step 18 against an intended 3.871e-09 -- exact to 9 significant figures), same optimizer, same GBS. dp was 96 against ~1530. So the failure is **not** a deterministic function of (LR trajectory, GBS, optimizer, step count) alone; at least one further variable matters. Read narrowly: also consistent with a stochastic miss. |
| **The 80B trains cleanly at a correct LR** | `12474431`: lr 5e-7, 64N, dp=192, GAS=4 -- 12 steps, 0 non-finite gradients, loss 12.948 -> 12.795, grad_norm flat 7.9. |

## What is refuted

**1. "bf16 residual-stream overflow" -- the stated mechanism.**
The direct test failed twice: fp32-ing the residual add NaNs at dp=192 step 19
(`8671243`) and, at full depth, step 37 (`8673658`). If fp32-ing the named
operation does not prevent the failure, that operation is not the site. The
2026-08-03 write-up concedes this in the same section that calls the diagnosis
closed.

**2. "bf16 masks true grad_norms of 21K-79K down to ~5-7" -- the smoking gun.**
Arithmetically impossible as stated, verified here: bf16 represents those
values to **0.29%** (21309 -> 21248, 79284 -> 79360, neither infinite), and a
real overflow yields `inf`, which propagates through a norm
(`norm([inf,1,2]) = inf`). No operation maps a true 21309 to a reported 4.97.
The two runs are on different optimization trajectories; fp32 changes every
rounding, so the weight sequences diverge immediately.

**3. Job `8537349`, the run the diagnosis rests on.**
Ran at n32/GBS=96 -- **inside the region where plain bf16 also trains clean**.
Retracted in the 2026-08-14 summary ("it never tested scale") and still cited
as the smoking gun in the meeting notes 17 days later. It also changes every
tensor in the model, so even taken at face value it localizes nothing.

## What is open

- **The site, at the moment of failure.** Still unmeasured -- no capture has
  fired yet. But the gradient DISTRIBUTION is now measured, and it points
  outside the transformer stack: `lm_head.weight` carries 44x the next tensor
  (see "What is established"). **Both routes on the standing list -- softcap
  and QK-Norm -- bound attention scores INSIDE the blocks, and neither
  touches `lm_head`.** If the failure originates in the output head, they
  were never going to bound it. Note the older "it must be a bf16 sublayer
  GEMM" inference compares `8673658` (64N/dp=192) against `8537349` (TP=4,
  **node count recorded nowhere**), so precision and scale vary together
  there anyway.
- **Why SophiaG failed at a safe LR.** `8574385` ran SophiaG at **1e-6**,
  below the finder's ~2.5e-6 optimum and well below its ~4.6e-6 blow-up onset,
  and NaN'd at step 18 anyway. **The historical failures are not an LR error.**
- **Why the vocabulary projection resists averaging.** This is now the sharpest
  question, and it is about the output layer and the loss rather than about
  distribution. Every other tensor's gradient shrinks ~22% per dp doubling;
  `lm_head.weight`'s shrinks 2%. Whatever makes it different is what
  concentrates gradient there, and concentration is the only quantity measured
  so far that grows monotonically toward the failing regime.
- **Whether concentration causes the failure.** Conjecture. None of the three
  dp arms failed, so this is healthy-regime structure. The hypothesis predicts
  the failure threshold tracks skew rather than dp directly.
- **Whether depth matters.** Untested. See the log below for why the attempt
  failed.
- **Whether large batch matters.** **Tested at dp=96 on 2026-09-07, and the
  answer is no.** Two arms differing ONLY in GBS (25,165,824 vs 1,572,864
  tokens; 6,144 vs 384 seqs), same 32 nodes, same dp, same LR trajectory:
  means agree to **0.03-0.20%** (`top0_gradnorm` 0.08%, skew 0.03%) while
  variances scale **2.75-4.33x**. The variance ratios sit on **sqrt(16) = 4**,
  which is what pure sampling noise predicts for a 16x batch reduction --
  three of four within 8% of it. So the batch does exactly what averaging
  says it should and nothing more, and the gradient STRUCTURE is
  batch-independent. See
  [`experiments/80b-gbs-vs-dp-separation.md`](../../experiments/80b-gbs-vs-dp-separation.md).

  Scope: this holds AT dp=96. It does not exclude a batch-by-parallelism
  interaction at dp~1530, which needs node counts that were not obtainable.
  Also note the older LR-finder observation (AdamW "NaN'd at GBS=192 but runs
  all 15 finder steps finite at GBS=6144") is not contradicted -- that was a
  different optimizer at a different dp.

## What to do next

1. **Per-tensor amax capture in the failing regime.** The instrumentation
   exists and, as of `3c17d3b55`, actually works. **The earlier version of
   this item was wrong**: it claimed the non-finite metrics "name the affected
   tensors". They did not. `collect_param_stats` filtered non-finite layer
   norms out of its per-layer stats and emitted `diag/topN_gradnorm` with no
   companion key for the NAME, so a capture would have logged two global
   aggregates (`grad_absmax_local`, `top0_gradnorm`) and nothing else --
   restating what a non-finite `grad_norm` already said. Driving it with a
   poisoned gradient exposed this; reading it did not. It now prints
   `THE TENSORS THAT WENT NON-FINITE: <name> (gradnorm=inf)`, with 4
   regression tests.

   Second, independent route to the site, from the clipping measurement
   above: post-clip, **exactly one tensor is non-finite and every other is
   exactly 0.0**. That identifies the culprit without relying on gradient
   norms being distinguishable.
2. **Re-run the depth bisect at lr=5e-7.** `80b_depth_bisect.pbs` is correct
   apart from the LR; the ladder (`agpt_50b_wide` 48L / `agpt_70b_wide` 72L /
   `agpt_80b` 84L) is verified to share dim 9216, vocab 256128 and per-layer
   shape.
3. **Reproduce `8574385`'s configuration** -- IN FLIGHT as `12474765`
   (successor to `12474761`). Not at 510N: the trajectory is reproduced
   instead by rescaling the peak LR, since `lr(n) = peak * n / warmup` means
   `5.376344e-09` with warmup 25 equals `1e-6` with warmup 4650 at every step
   (verified to 2.2e-16). 32N on access-verified hosts, GBS held at
   25,165,824 via GAS 64.

   **Do not try to match the warmup by shortening the run.** torchtitan
   clamps `warmup_steps` to `total_steps` and only warns, so a shorter run is
   clamped HARDER -- and the trainer banner prints the unclamped value on the
   next line. Two attempts died on this before the rescale. See
   [`warmup-clamp-silently-voids-short-reproductions.md`](warmup-clamp-silently-voids-short-reproductions.md).

## Investigation log (2026-08-31) -- read this before repeating any of it

Everything in this section is either superseded or void. It is kept because
each error is cheap to repeat.

**All my experiments ran at the wrong LR.** `agpt_80b` inherits `agpt()`'s
shared default of **8e-4**; the guide documents a usable AdamW ceiling of
**~7.4e-7** and records job `8530891` NaN-ing at step 2 at 1e-6. So jobs
`12474403` (depth bisect) and `12474423` (GAS sweep) ran ~1000x past the last
stable point, and their step-2 blow-ups are that. `agpt_80b`'s docstring now
warns (`c4422f24f`).

**Three conclusions published and withdrawn, in order:**

1. *"L84 died at step 40; two back-to-back events are fatal."* It recovered at
   step 41 (loss 6.98) and finished healthy. I read a failure off one log line
   without reading the next, then built a mechanism on it.
2. *"The event rate scales with depth, 0/1/8 across 48/72/84 layers."* The
   replicate came back **2 events against L84's 8** in the identical
   configuration -- a 4x spread from data order alone. Single runs cannot
   support the trend.
3. *"GBS not dp -- the non-recovering state reproduced at fixed dp."* Void with
   the LR.

**A rate metric cannot describe a deterministic failure.** The GAS report
computed events-per-microbatch and concluded "larger batches are MORE stable
per unit work" -- because the G32 arm ran **two steps** and died on the second.
I had tested that verdict logic 5/5 on synthetic *rate* patterns; every case
checked whether the rate was computed correctly and none checked whether a
rate was the right measurement.

**The no-accumulation control cannot be built on this hardware.** Removing GAS
while holding GBS fixed requires larger per-rank microbatches; both 32,768 and
16,384 tokens/rank SIGTERM in the first forward (per-rank activation memory).
Its GAS=4 companion arm did run clean, which does at least show accumulation
is not corrupting gradients at that depth of accumulation.

**Instrumentation defects found and fixed along the way**, each of which had
been silently producing nothing: the QK probe disabled under `torch.compile`
(host sync in `.item()`); diagnostics destroyed by `zero_grad` before
collection; the monitored `grad_norm` being pre-clip (`clip_fired=1` every
step, framework default `max_norm=1.0`) and depth-confounded (global scales
3.3x while `layer_gradnorm_max` stays 0.99).
