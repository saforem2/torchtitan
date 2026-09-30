# 30B SophiaG bounded LR canary — 2026-09-30

Coarse learning-rate evidence for `agpt_30b_olmo2tok_smoke` under SophiaG on
the validated post-`cde3c93227` HSDP topology.

## Provenance

| field | value |
|---|---|
| job | `12479134` (Sunspot, `workq`) |
| scheduler result | `job_state=F`, `Exit_status=0`, walltime `00:19:44` |
| source SHA | `7cc86ebc750d8455e1b8a0f6dc6177797f0d010b` (enforced by `LRF_EXPECTED_SHA`) |
| runtime | `/lus/tegu/.../venvs/xpu-torch215`, torch `2.15.0.dev20260915+xpu`, spmd-types `0.2.5`, grain `0.2.18` |
| model | `agpt_30b_olmo2tok_smoke`, OLMo-2 tokenizer |
| topology | 16 nodes / 192 ranks, TP=1, HSDP `dp_replicate=3` × `dp_shard=64` |
| batch | `num_tokens_per_train_step=3,932,160`, `num_tokens_per_microbatch_per_dp_rank=20,480`, `max_context_length=4096` (GBS 960) |
| activation checkpoint | `full` |
| sweep | 10 points, `1e-7` → `1e-4`, multiplier `2.154435` |

This matches the AdamW canary (`12479108`) in every dimension except the
optimizer, so the two curves are directly comparable.

## Artifacts

| file | sha256 |
|---|---|
| `sunspot-12479134-30b-sophiag-canary.csv` | `b745f47ad71ca107e4da401d9f785a7d4eb8a6071a8cb95ad37500c17768fbaa` |
| `sunspot-12479134-30b-sophiag-canary.png` | `202aa215ff6cb37a748e6c54894aef7b09296ed0f63b787a1abed83ccf54f251` |

## Measured curve

All ten sampled learning rates produced finite smoothed losses.

| LR | smoothed loss |
|---:|---:|
| `1e-7` | 12.026743 |
| `2.154435e-7` | 12.017947 |
| `4.641589e-7` | 12.002591 |
| `1e-6` | 11.975449 |
| `2.154435e-6` | 11.925534 |
| `4.641589e-6` | 11.830914 |
| **`1e-5`** | **11.682948** |
| `2.154435e-5` | 12.131376 |
| `4.641589e-5` | 13.044453 |
| `1e-4` | 13.815876 |

The minimum is interior: three lower samples descend into it and three higher
samples rise away from it, so this window brackets a basin rather than ending
on a boundary.

## Update evidence

Per-step trainer metrics confirm real parameter-changing work rather than
skipped updates, and show the post-basin degradation directly:

```
step:  2  loss: 12.02674  grad_norm:  3.5364
step: 10  loss: 11.73566  grad_norm:  3.6268
step: 14  loss: 10.85527  grad_norm:  5.7747
step: 16  loss: 15.02905  grad_norm: 94.1816
step: 20  loss: 20.10017  grad_norm: 69.6485
```

Two full-state DCP checkpoints were written (`step-10`, `step-20`), each with
193 files and a 15,291,267-byte `.metadata`.

## Interpretation

- **Evidence level:** complete coarse curve with an interior basin.
- **Observed loss-minimizing sampled LR:** `1e-5`.
- This is *not* a production recommendation. It is a coarse sampled minimum on
  a changing-LR trajectory; weights, optimizer state, and data advance while
  the LR moves.
- The gradient-norm jump from ~5.8 to ~94 immediately past the basin is
  consistent with the known 30B SophiaG instability
  (`guides/known-bugs/sophiag-stochastic-divergence-30b.md`) and argues for
  caution above roughly `1e-5`.
- Promoting `1e-5` for SophiaG requires matched fixed-LR arms, exactly as was
  done for AdamW in `12479115`–`12479117`.

## Superseded evidence

This run replaces the earlier incomplete SophiaG material for this model:
the 30-point coarse sweep `12478513` and the truncated 97/100 fine curve
`12478570`. Those remain historical records and are not resumable — no finder
cursor survives on disk.
