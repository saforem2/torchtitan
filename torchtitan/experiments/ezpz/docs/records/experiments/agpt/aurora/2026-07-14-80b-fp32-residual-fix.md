# 80B fp32-residual fix — root-cause + prototype

> Last updated: 2026-08-23

## Root cause (task #21)

The agpt 80B production NaN is a **bf16 forward-activation overflow in the deep
residual stream — NOT an optimizer bug** (the `80b/README.md` banner previously
blamed "SophiaG Hessian `grad*grad` overflow at dim=9216"; that is wrong).

Evidence:
- **Both optimizers NaN identically.** mano @ 62N (job 8661293) and SophiaG @
  510N (job 8574385) both show grad_norm dead-flat ~6.0 for 13-16 steps, then a
  sudden inf/nan with no runup (SophiaG's step-14 inf even *recovers* to 6.19
  before nan'ing at 17). mano has no Hessian term -> optimizer-independent.
- **Loss softmax ruled out** — all CE paths upcast logits to fp32 before softmax
  (`components/loss.py:61,128,263`). (bf16 has fp32's ~3.4e38 exponent range; the
  65504 ceiling is fp16. bf16's weakness here is its 8-bit mantissa.)
- **Grad reduction ruled out** — FSDP `reduce_dtype` is locked to fp32
  (`config/configs.py:66`); loss is token-normalized by a divisor that grows with
  dp (`trainer.py:694`), so per-step grad magnitude is dp-invariant (not a
  sum-before-divide overflow).
- **Mechanism:** the 84-layer pre-norm residual `h = x + sublayer(norm(x))`
  accumulates in bf16 (`mixed_precision_param=bfloat16`). RMSNorm rescales
  sublayer *inputs* but never the residual stream itself. At 80B's width x depth
  (dim=9216 x 84L x ffn=25600) the deep bf16 residual reaches the
  overflow/precision regime 2B/20B never hit (structural: `50B_wide`/`70B_wide`
  share the exact 80B per-layer shape at fewer layers — the depth ladder built to
  bisect this family).
- **Smoking gun:** the fp32-activations run (job 8537349) trains clean and
  reveals TRUE grad_norms of **21K-79K** that bf16 silently masks to ~5-7.
- **dp-dependence:** larger dp -> larger effective GBS -> weights reach the
  overflow state faster (LR/seed/master-dtype/clip all proven non-causal in the
  [7-job factorial](20260611-80b-n32-nan-diagnosis.md)).

## Fixes

| # | Fix | Status |
|---|-----|--------|
| 1 | `--training.mixed-precision-param=float32` @ TP=4 | **confirmed** clean (job 8537349, 20 steps); ~3-5x slower (fp32 for ALL activations) |
| 2 | **fp32 residual stream** (this doc) | prototype built + CPU-validated; needs on-XPU numerics + throughput validation |
| 3 | `--nan-abort-consecutive=5` in the 80B autoretry script | done — bails a diverged run instead of burning full walltime |

## fp32-residual prototype (task #24)

`agpt/fp32_residual.py` — `AgptFp32ResidualBlock(Llama3TransformerBlock)`
overrides only `forward` to do the two residual **adds** in fp32 while casting
each sublayer's input back to the bf16 param dtype, so the attention/FFN GEMMs
stay bf16. Block takes bf16 in / emits bf16 out (so the decoder's bf16 `lm_head`
contract at `decoder.py:283` is satisfied with zero core changes). Rationale for
per-block (not full-depth) fp32: the between-block RMSNorm re-standardizes the
stream, and the observed failure is a single-step overflow spike (flat-then-inf),
so protecting each `+` is what matters. If full-depth fp32 proves necessary, it
needs an experiment-local Decoder subclass to cast `h` to the lm_head dtype
before the head.

Wired via a post-hoc transform `_set_fp32_residual(cfg)` (mirrors
`_set_rope_backend`; deep-copies the config first to avoid poisoning the shared
`agpt_configs` template — a leak that was caught and fixed in testing). Flavors:
`agpt_80b_fp32res`, `agpt_80b_real_fp32res`.

### CPU validation (login node, no XPU)
- Block imports, subclasses `Llama3TransformerBlock`, `Config._owner` auto-wired.
- Full debugmodel build + block-swap + forward: **runs, bf16 logits, all finite**
  (caught + fixed two bugs en route: wrong `AttentionMasksType` import path, and
  an fp32-output-into-bf16-lm_head dtype mismatch).
- Config wiring: `agpt_80b_fp32res()` -> all 84 layers are
  `AgptFp32ResidualBlock.Config`; plain `agpt_80b()` stays stock (no leak).

### On-XPU smoke result (job 8671066, 4N, 40 steps) -- PASSED

First on-XPU validation of . Trained all 40 steps clean,
Training completed, zero NaN/inf:
- **Loss 12.79 -> 9.24** (monotonic descent; the bf16 NaN runs stayed dead-flat
  ~12.9 -- the model is actually learning).
- **grad_norm ALIVE + dynamic: 5 -> 22 -> 42 -> 56 (step 24) -> recovers to ~9**
  -- large gradient spikes that bf16 would mask flat at ~6.0 (or NaN on) are
  absorbed by the fp32 residual and training continues. Smoking-gun match to the
  fp32-activations reference.
- MFU ~11.6%, TPS ~63, mem 42.5 GiB/66%, stable throughout.
- (First attempt 8671046 failed on invalid CKPT_INTERVAL=0 -- my config error,
  never trained; resubmitted with CKPT_INTERVAL=100.)

**CAVEAT: 4N is BELOW the NaN wall** (bf16 was also clean at small dp, to step 30
at 8N). This validates the block trains correctly on XPU with sane numerics and
throughput -- it does NOT yet prove it clears the dp>186 wall. That is the next
test.

### Wall test (job 8671243, 64N / dp=192, 40 steps) -- FAILED: still NaNs

The decisive test, in the regime that killed mano@62N (step 17) and SophiaG@512N
(step 14). Result: **the per-block fp32-residual fix does NOT clear the wall.**
- grad_norm rose 5.4 -> 8.1 (steps 10-18) -- ALIVE, not the bf16-masked flat ~6.0,
  so the fp32 residual add is genuinely active -- then **step 19 grad_norm nan,
  step 20 loss nan.** Same flat-ish-then-instant-nan signature, same ~step-17-20
  window as the bf16 runs. Marginally more headroom (grad_norm reached 8.1 vs
  ~6.9, NaN 2 steps later) but the wall stands.
- nan-abort guard worked: bailed at step 27 (8 consecutive NaN), reclaimed
  walltime. Validated in production conditions.

**Interpretation:** the per-block fp32 residual is **necessary-but-insufficient**.
It fixes the residual-add masking (grad_norm now visible; 4N trained clean to
step 40) but the high-dp NaN persists. This means either (a) the FULL-DEPTH fp32
residual stream is needed (my prototype casts back to bf16 at each block boundary,
so cross-layer accumulation is still bf16 -- needs an experiment-local Decoder
subclass to carry fp32 across all 84 layers + cast to lm_head dtype at the end),
or (b) the overflow is elsewhere (attention scores / a specific activation) that
per-block casting does not protect. No run has instrumented per-op activation
maxima to disambiguate -- that is now the decisive next step.

**Fallback that IS confirmed working:** 
@ TP=4 (job 8537349) -- full fp32 activations, ~3-5x slower but NaN-free. If
80B production is urgent, that is the unblock today.

### Full-depth fp32 wall test (job 8673658, 64N/dp=192) -- BETTER but STILL NaNs

Full-depth variant (agpt_80b_fp32res_depth: fp32 residual carried across all 84
layers via AgptFp32ResidualModel, vs per-block which re-truncates each boundary).
Result: **strictly better, wall moved but did not fall.**
- Trained clean through **step 37** (loss 12.49 -> 9.47), CLEARING the step-19
  point where the per-block variant died and the step-14-17 bf16 point.
- grad_norm spiked to ~40 at steps 22-24 and RECOVERED to ~8, then spiked again
  17->20 at steps 36-37, then **NaN at step 38.**
- Ladder: bf16 NaN ~step 14-17 -> per-block fp32 step 19 -> full-depth fp32
  step 38. Each fp32 layer of protection buys ~18 more steps but does not
  eliminate the overflow.

**Conclusion: the residual stream is PART of the problem (full-depth fp32 doubled
the survival window) but NOT the whole story -- there is a second overflow source
fp32-residual does not protect.** The intermittent grad_norm spikes (40 at s22-24,
recover, again at s36-37) point at something that spikes then poisons -- attention
scores are the prime suspect. This is exactly the ambiguity the per-op numerics
capture resolves; it is now clearly warranted (task #29, wiring committed
36892d672). Run: single bf16 run at 32N/TP=2 (dp=192, NaNs ~step 6), scan
rank_0_activations.log for the first non-finite Max -> names the op.

### Outcome (RESOLVED -- work now dormant)
The wall tests above were run (tasks #25-#29):
1. 4N smoke (job 8671046) -- trains NaN-free, loss matches the fp32-acts
   reference. PASS.
2. dp=192 wall test (per-block fp32 residual, job 8671243) -- **STILL NaN'd at
   step 19**; full-depth fp32 residual (job 8673658) also NaN'd at dp=192. So
   per-block fp32 accumulation is **necessary but insufficient** -- the true
   overflow site is elsewhere (attention scores / another activation), needing
   per-op numerics capture (task #29 wiring committed) to localize.
3. Not reached (fix not yet working, so TPS comparison moot).

**Status: dormant.** There is no live 80B production; the fp32-residual
prototype does not yet clear the dp>186 wall, and the effort is paused pending
a decision to invest in per-op instrumentation. See
[production/agpt/80b/README.md](../../../../production/agpt/80b/README.md) Wall 1.
