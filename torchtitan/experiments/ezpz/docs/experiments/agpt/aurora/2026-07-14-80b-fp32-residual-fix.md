# 80B fp32-residual fix — root-cause + prototype

> Last updated: 2026-07-14

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

### Next (needs a GPU allocation)
1. Smoke `agpt_80b_fp32res` at small N (e.g. 4-8N) — confirm it trains NaN-free
   and the loss curve matches the fp32-acts reference (job 8537349).
2. Scale to dp>186 (the NaN regime) — the real test: does per-block fp32
   accumulation clear the wall that killed SophiaG@512N and mano@62N?
3. Measure throughput vs. `mixed-precision-param=float32` — the fp32-residual fix
   is only worth it if the bf16 GEMMs give materially better TPS.
