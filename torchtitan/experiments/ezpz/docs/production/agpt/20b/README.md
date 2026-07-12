# Production Training — agpt 20B

> Last updated: 2026-07-12
>
> Current v2 production runs on `--training.dtype=float32`.
> Historical v1 (bf16-tainted) runs are archived at
> [`../historical/v1-bf16/`](../historical/v1-bf16/README.md) along
> with the diagnosis link.

## 20B chains overlaid

![all 20B trajectories](../../figures/production_20b_training.svg)

Every 20B trajectory (TT v2 256N + TT v2 512N sync) overlaid on shared
axes vs tokens consumed. Three panels: training loss / TPS-per-GPU /
MFU. Refreshed via `python3 -m
torchtitan.experiments.ezpz.utils.plot_production_combined`.

For the cross-model view (2B + 20B together) with per-token efficiency
comparison, see [`../README.md`](../README.md).

## 🏁 Headline (2026-06-09)

**20B 512N chain beats 2B 256N async on every benchmark per token.**
Advanced to step **5,400** / ~543.6B tokens (11.6%) after the native
auto-retry relaunch (8638793 + 8638795) broke the long sync-chain stall on
2026-07-05..07 (was stuck at step-4,400 since 2026-05-29). The eval headline
below is the step-100 → step-4,400 window (35+ ckpts; a re-eval of the
4,400 → 5,400 tail is pending):

- ARC-Easy `acc` 0.463 → **0.664** (+20pp)
- HellaSwag `acc_norm` 0.296 → **0.635** (+34pp)
- ARC-C `acc_norm` 0.224 → **0.380** (+16pp)
- Winogrande `acc` 0.493 → **0.586** (+9pp)

Monotonic lift across 35+ consecutive ckpts. See
[`evals/agpt/20b/`](../../../evals/agpt/20b/README.md) for the full
per-task table.

## Snapshot

| Trajectory | Status | Cumulative steps | Loss | Tokens |
|------------|--------|-----------------:|-----:|-------:|
| [**v2 512N**](n512/README.md) (canonical chain) | Advancing — native auto-retry relaunch (8638793 resume + 8638795 cont) broke the sync-chain stall, carried step-4,400 → 5,400 (2026-07-05..07). step-100..5,400 persisted every 100. cont chained. | **6,000** (persisted) | **2.47** | **~604.0B (12.9%)** |
| [v2 256N](n256/README.md) | Advancing — carried to step **3,100** (156.0B, 3.3%) via the relocated `agpt-20b-n256` clone chain (8558548/8558549 + sneaks). Per-token comparator to the canonical 512N. | 4,200 | 3.2631 | ~156.0B (4.5%) |
| [v2 1024N](n1024/README.md) | First attempt 8463183 crashed at startup (SIGSEGV at 12,288 ranks); not retried | — | — | — |

**Canonical 512N chain (sync-mode)**: 8505258 (🏁 sync-mode
breakthrough, +6) → 8505259 (+6) → 8507197 (+6) → 8507200 (+6,
walltime, ended step 3,270) → 8508214 (walltime, +5 ckpts, ended step
3,806) → [**8509393**](n512/README.md#log-8509393) (walltime exit
2026-05-29 12:01, +6 ckpts, ended step 4,400). Then: 8513546 / 8514610
(retries, no new ckpts), 8516701 (PBS killed mid-save → empty
`step-4500/` placeholder), 8521624 / 8521625 (trained in-RAM past
4,400 but blocked by stale placeholder), placeholder
renamed `.bak-empty-20260606-170503/` on 2026-06-06 to unblock resume,
cont (8521628) Q, cont (8521632) H.

## Per-trajectory detail

- [n512/](n512/README.md) — **canonical v2 512N sync chain** (stalled
  on queue + persistence-blocked retries)
- [n256/](n256/README.md) — v2 256N (ended step 1,125; no continuation)
- [n1024/](n1024/README.md) — v2 1024N (8463183 crashed at startup,
  std::bad_alloc / SIGSEGV — needs 768N/896N bracket before retry)

## Eval scores

See [`docs/evals/agpt/20b/`](../../../evals/agpt/20b/README.md) for the
current 20B lm-eval tables and the **🏁 headline** finding above. At
step 4,400 (~442.9B tokens) v2 512N sync reaches ARC-Easy **0.6641**
and HellaSwag `acc_norm` **0.6346** — beating the 2B 256N async chain
at step-69,900 / ~3.52T tokens (HSn 0.5552) by a wide per-token margin.
The 20B per-token efficiency advantage is dramatic and the chain has
not begun to plateau yet.
