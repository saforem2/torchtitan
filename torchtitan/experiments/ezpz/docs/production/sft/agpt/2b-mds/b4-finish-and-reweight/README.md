# B4 results -- both paths FAILED to recover B2; two-stage structure is the lever

Date: 2026-07-24
Verdict: NEITHER B4 path recovered B2's 0.205. Do not pursue single-stage rebuilds further.

## Results (200-problem GSM8K CoT, fp32 vLLM, identical eval)

| model | cot_accuracy | format_hit_rate | mean_gen_len | n_unclosed | notes |
| --- | --- | --- | --- | --- | --- |
| B2 (two-stage: tulu-math -> gsm8k-r1cot) | 0.205 | 0.985 | 282 | ~0 | the target; best result |
| B3 (single SFT, broad mix @8192) | 0.05 | 0.86 | 601 | 26 | long-CoT dilution regression |
| B4a (gsm8k-r1cot finish on B3 base) | 0.02 | 0.26 | 1378 | 133 | WORSE: verbose bias baked in |
| B4b (reweighted + len-filtered @4096) | 0.065 | 0.86 | 611 | 27 | fixed run-on symptom, NOT accuracy |

Jobs: B4a 12471671 (ep1/2/3 = 12471676/77/78), B4b 12471672 (eval 12471685).

## Conclusions

1. **B4a REFUTED (finishing stage on the B3 base).** A short gsm8k-r1cot finish did
   NOT un-teach B3's verbose R1 style -- it made it worse (gen_len 601 -> ~1400-2000,
   format collapsed to 0.0-0.26). The B3 base's verbose bias is baked in; a light
   finish destabilizes the envelope. You cannot rehabilitate a verbose-poisoned base.

2. **B4b partially worked but missed the point.** The OPENR1_MAX_THINK_CHARS length
   filter + reweight (gsm8k-r1cot 0.40) fixed the SYMPTOM the guardrails flagged
   (gen_len 611 not B3's 601-ish-but-run-on, unclosed 27 not 133, format 0.86) --
   but accuracy stayed at 0.065, ~B3's level, far from B2's 0.205. The run-on style
   was not the accuracy cause.

3. **The real lever is STRUCTURE, not mix.** Every single-SFT-from-gs138650 recipe
   (B3 0.05, B4b 0.065) lost badly to B2's two-stage tulu-math -> gsm8k-r1cot lineage
   (0.205), regardless of mix weights, OpenR1 length filtering, or seq length. A
   single combined SFT -- even a well-balanced, filtered one -- does not reproduce
   what the specific two-stage sequence produced.

4. **~0.2 is near the 2B GSM8K-CoT ceiling for this base.** Across the whole effort
   -- RL (flat at 0.205->0.215), B3 (0.05), B4a (0.02), B4b (0.065) -- nothing beat
   B2's two-stage 0.205. The accuracy lever is NOT SFT recipe tuning at 2B; it is a
   bigger / math-specialized base model, or accepting B2 as the deliverable.

## Recommendation

- Keep B2 (checkpoint-93 lineage, 0.205) as the agpt-2b CoT cold-start deliverable.
- Do NOT attempt further single-stage SFT rebuilds -- the structure result is
  consistent across B3/B4a/B4b.
- If pushing accuracy remains a goal: (a) reproduce B2's two-stage recipe exactly and
  extend/tune WITHIN it, or (b) move to a larger or math-pretrained base. Both are
  bigger lifts than mix-tuning and should be scoped separately.
- Reusable infra built this session survives regardless: OPENR1_MAX_THINK_CHARS
  filter, b4_reweight_mix, the eval gen_len/n_unclosed guardrails (which correctly
  surfaced the run-on failure that accuracy alone hid), and merge_and_eval tooling.
