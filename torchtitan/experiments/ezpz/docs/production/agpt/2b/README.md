# Production Training — agpt 2B

> Last updated: 2026-08-30
>
> Current v2 production runs on `--training.dtype=float32`.
> Historical v1 (bf16-tainted) runs are archived at
> [`../historical/v1-bf16/`](../historical/v1-bf16/README.md) along
> with the diagnosis link.

> [!IMPORTANT]
> **Nothing on this page is running.** Both 2B stage-1 chains **finished**
> their full 4.67T budget -- 256N on 2026-06-29, 512N on 2026-08-13 -- so
> "not running" is the expected end state for them, not a fault. What is
> *idle rather than finished* is the follow-on 2B work (stage-2 dolmino CPT
> seeded from 512N `step-46429`, and the 512N constant-LR fork): Aurora
> production last trained **2026-08-26** and the successor umbrella
> `8784460` (2,098 nodes, `large`) has been `Q` since 2026-08-26 13:22 UTC
> with 101h+ eligible time and `score_boost = 0`, `8784462` held behind it
> on `afterany`. ALCF ticket drafted but **NOT sent**:
> [`ops/alcf-ticket-8784460-not-scheduling-20260830.md`](../../../ops/alcf-ticket-8784460-not-scheduling-20260830.md).

## 2B chains overlaid

![all 2B trajectories](../../figures/production_2b_training.svg)

Every 2B trajectory (MDS-reference + TT v2 256N async + TT v2 512N sync)
overlaid on shared axes vs tokens consumed. Three panels: training loss /
TPS-per-GPU / MFU. Refreshed via `python3 -m
torchtitan.experiments.ezpz.utils.plot_production_combined`.

For the cross-model view (2B + 20B together), see
[`../README.md`](../README.md).

## Snapshot

| Trajectory | Status | Cumulative steps | Loss | Tokens |
|------------|--------|-----------------:|-----:|-------:|
| [**v2 256N (async)**](n256/README.md) (per-token comparator) | **COMPLETE** 2026-06-29 (step-92,859 = 100% of 4.67T target); finished by cont12 (8558531). Now the base for CPT ([../../cpt/](../../cpt/README.md)) + SFT. | **92,859** | **2.6524** | **~4.674T (100.0%)** |
| [**v2 512N (sync)**](n512/README.md) (canonical chain) | **COMPLETE (stage 1)** 2026-08-13 19:11 UTC (step-46,429 = 100% of the 4.67T budget), as umbrella 8744247 trainer 0, `FAILOVER STOP: success`, rc=0. No stage-1 budget left, so no further dispatches. Stage 2 (dolmino CPT from `step-46429`) and the constant-LR fork are **idle** -- last advanced 2026-08-26, now waiting on queued umbrella `8784460`. | **46,429** | **2.6869** | **~4.674T (100.0%)** |
| [v2 1024N](n1024/README.md) | Crashed at startup (12,288-rank init OOM/SIGSEGV); not retried | — | — | — |
| v2 512N sqrt(2)-LR fork | 8467141 → 8467142 (separate ckpt dir `gbs12288-lr3.22e-5`) | 200 | — | ~20B |

**Headline (2026-08-30): both 2B stage-1 chains are done; nothing is
running.**

- **512N stage 1 is COMPLETE, not stalled.** It finished 2026-08-13
  19:11 UTC at step-46,429 / **4.674T** tokens / loss **2.68687**, exit 0,
  as trainer 0 of umbrella 8744247. It has no budget left and is not
  awaiting a continuation -- see [n512/](n512/README.md).
- **256N is COMPLETE** at step-92,859 / 4.674T / loss 2.652, finished by
  cont12 (8558531) on 2026-06-29. It is the base for CPT
  ([../../cpt/](../../cpt/README.md)) + SFT.
- **The idle work is downstream of those two.** The last umbrella to seat,
  `8773440`, ran 5h13m of a 12h slot on 2026-08-26 (`Exit_status=-14`).
  Its three 2B seats: one 2B-512 trained **504 steps, 2.74535 -> 2.73836**,
  one 2B-256 trained **782 steps, 2.56449 -> 2.54807**, and the other 2B-512
  logged **0 steps**. **Nothing on Aurora has trained since.** None of them
  was extending the two completed chains above: as of the
  [2026-08-21 seat audit](../../../experiments/agpt/aurora/20260821-umbrella-seat-audit.md)
  the three 2B umbrella seats were the 512N constant-LR fork (`t3`), the
  256N stage-2 dolmino arm (`t4`), and the 512N stage-2 dolmino arm (`t0`),
  each on its own checkpoint dir. The losses are consistent with that
  mapping -- the 504-step seat's 2.74535 sits on the constant-LR fork's tip
  (2.74509 at step 21,307 in the committed
  [metrics store](../../metrics/README.md)), and neither figure matches a
  completed chain. Per-seat detail: [dispatch log](../../dispatch-log.md).

**Headlines (2026-07-24) -- superseded, kept as the record of that day.
The "LIVE" and step figures in this block are NOT current:**

- **256N async chain COMPLETE** at step-92,859 (100.0% of the 4.67T
  target, loss ~2.6511). Finished by cont12 (8558531) on 2026-06-29; it
  is now the base for CPT ([../../cpt/](../../cpt/README.md)) + SFT.
- **512N sync chain LIVE at step-39,600** (~3.99T tokens, 85.3%, loss
  ~2.71). It advanced past the old step-30,400/30,500 stall via the
  autoretry chain and a HEAD-migration dress rehearsal
  (8686135 -> 8686136); it is no longer pinned or queue-blocked.
  *(It went on to finish at step-46,429 on 2026-08-13.)*

## Per-trajectory detail

- [n256/](n256/README.md) — v2 256N async (canonical per-token comparator;
  COMPLETE at step-92,859)
- [n512/](n512/README.md) — **canonical v2 512N sync chain** (COMPLETE at
  step-46,429 on 2026-08-13; stage 2 + the constant-LR fork are idle)
- [n1024/](n1024/README.md) — v2 1024N (8463182/8463183 both crashed
  at 12,288-rank init; needs 768N/896N bracket before retry)

## Eval scores

See [`docs/evals/agpt/2b/`](../../../evals/agpt/2b/README.md) for the
current 2B lm-eval tables (HellaSwag / ARC-Easy / ARC-Challenge /
Winogrande). Latest entries are at the bottom of that table; step-69900
holds the chain's **best Winogrande** at 0.5627.

**Per-token efficiency note:** the 20B 512N sync chain at step 4,400
(~442B tokens) hits HSn 0.6346 / ARC-E 0.6641 — beating this 2B 256N
plateau on every benchmark per token. The 2B chain has burned ~8×
more tokens to reach a worse score, which is expected; the comparator
exists to confirm the 20B chain is converging *qualitatively faster*
per token, not just per FLOP. See
[`evals/agpt/20b/`](../../../evals/agpt/20b/README.md).
