# LR Finder — agpt 10B OLMo-tokenizer

Learning-rate evidence for the 9.48B-parameter `agpt_10b_olmo2tok` model.
For the campaign-wide record, see the
[GBS=6144 report](../2026-09-18-olmo2tok-ladder-gbs6144-nexteval.md).
For methodology and acceptance criteria, see the
[LR-finder methodology](../../README.md).

## Status

**Production LR: open.**

| optimizer | evidence | interpretation |
|---|---|---|
| AdamW | `12478509`, 100/100 points over `1e-6`–`1e-3` | observed basin minimum `4.64e-4`; detector `4.16e-5` is a 10× safety heuristic, not a measured optimum |
| SophiaG | `12478569`, 100/100 points over `1.66e-6`–`1.66e-4` | loss minimum is the final sample; detector `3.24e-7` is below the sampled range and is not defensible |
| Muon | excluded | first-update Newton–Schulz path has not passed finite-gradient, finite-weight, real-update validation |

## AdamW

Job `12478509` completed with finite artifacts. Its loss minimum occurred at
`4.64e-4`, then the curve rose. The published `4.16e-5` was the detector
crossing divided by 10. The curve supports a basin near several `1e-4`; it does
not by itself establish a constant production LR.

A fixed-LR validation should follow the 5B matrix once that workflow is
classified, using identical initialization and enough updates to expose
sustained instability.

## SophiaG

Job `12478569` completed 100 points, but the loss was still decreasing at the
upper endpoint `1.66e-4`. The fine window therefore did not bracket the basin.
A replacement must start from the same clean initialization and extend above
`1.66e-4`; independently initialized points must not be appended to this curve.

## Artifacts

- [10B AdamW CSV](../data/2026-09-24-olmo2tok-gbs6144-verified/sunspot-12478509-10b-adamw-fine.csv)
- [10B SophiaG CSV](../data/2026-09-24-olmo2tok-gbs6144-verified/sunspot-12478569-10b-sophiag-fine.csv)
- [Evidence manifest](../data/2026-09-24-olmo2tok-gbs6144-verified/README.md)

The CSV schema does not encode gradient or optimizer-update health. Terminal
logs and fresh artifacts remain part of the acceptance contract.
