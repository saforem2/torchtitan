# The 80B NaN: what is known, what is refuted, what is open

> Last rewritten 2026-08-31, after a day of experiments that were themselves
> invalidated. Conclusions first; the investigation log is at the bottom.
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

- **The site.** No per-tensor capture has ever run in the failing regime. The
  "it must be a bf16 sublayer GEMM" inference compares `8673658` (64N/dp=192)
  against `8537349` (TP=4, **node count recorded nowhere**), so precision and
  scale vary together.
- **Why SophiaG failed at a safe LR.** `8574385` ran SophiaG at **1e-6**,
  below the finder's ~2.5e-6 optimum and well below its ~4.6e-6 blow-up onset,
  and NaN'd at step 18 anyway. **The historical failures are not an LR error.**
- **Whether depth matters.** Untested. See the log below for why the attempt
  failed.
- **Whether large batch matters.** Untested for the same reason. Note the
  LR-finder found the opposite of what one might assume: AdamW "NaN'd at
  GBS=192 but runs all 15 finder steps finite at GBS=6144, where the larger
  batch smooths the Hessian estimate."

## What to do next

1. **Per-tensor amax capture in the failing regime.** The instrumentation now
   exists: the non-finite branch captures `collect_param_stats(per_layer=True)`
   *before* `zero_grad` destroys the gradients (`f01548f6a`), and logs the
   metrics that are themselves non-finite -- which name the affected tensors.
   Nothing has run through it yet.
2. **Re-run the depth bisect at lr=5e-7.** `80b_depth_bisect.pbs` is correct
   apart from the LR; the ladder (`agpt_50b_wide` 48L / `agpt_70b_wide` 72L /
   `agpt_80b` 84L) is verified to share dim 9216, vocab 256128 and per-layer
   shape.
3. **Reproduce `8574385`'s configuration** -- SophiaG at 1e-6, 510N -- since
   that is the failure nobody has explained and it is not an LR artifact.

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
