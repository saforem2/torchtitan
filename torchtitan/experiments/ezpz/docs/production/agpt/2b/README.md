# Production Training — agpt 2B

> Last updated: 2026-09-21
>
> Current v2 production runs on `--training.dtype=float32`.
> Historical v1 (bf16-tainted) runs are archived at
> [`../historical/v1-bf16/`](../historical/v1-bf16/README.md) along
> with the diagnosis link.

> [!IMPORTANT]
> **Nothing on this page is running.** Both 2B stage-1 chains **finished**
> their full 4.67T budget -- 256N on 2026-06-29, 512N on 2026-08-13 -- so
> "not running" is the expected end state for them, not a fault. The
> follow-on 2B heads, verified directly on Aurora on 2026-09-21, are
> **step-22,300** (512N stage-2 dolmino), **step-28,000** (256N stage-2
> dolmino), and **step-39,200** (512N constant-LR fork); every head has the
> expected shard count and `.metadata`.
>
> The latest completed umbrella, `8828612`, ran for 12h on 2026-09-19--20
> but converted only **1/5** seats. Its sole training seat was `t2` (20B-256,
> log through step 16,036; persisted head 16,000). The 2B seats made no
> progress: `t0`, `t3`, and `t4` failed during startup with the repeated-
> import guard (`torch` can only be initialized once per process), while
> `t1` failed with `std::bad_alloc`. Thus this was a launcher/runtime
> regression plus the known allocator failure, not a successful five-seat
> production cycle. Successor `8834528` is queued.

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
| [**v2 512N (sync)**](n512/README.md) (canonical chain) | **COMPLETE (stage 1)** 2026-08-13 19:11 UTC (step-46,429 = 100% of the 4.67T budget), as umbrella 8744247 trainer 0, `FAILOVER STOP: success`, rc=0. No stage-1 budget left, so no further dispatches. Follow-on heads: stage-2 dolmino **22,300** and constant-LR **39,200**, both unchanged by umbrella `8828612`. | **46,429** | **2.6869** | **~4.67T (100.0%)** |
| [v2 1024N](n1024/README.md) | Crashed at startup (12,288-rank init OOM/SIGSEGV); not retried | — | — | — |
| v2 512N sqrt(2)-LR fork | 8467141 → 8467142 (separate ckpt dir `gbs12288-lr3.22e-5`) | 200 | — | ~20B |

**Headline (2026-09-21): both 2B stage-1 chains are done; the latest
umbrella advanced no 2B trajectory.**

- **512N stage 1 is COMPLETE, not stalled.** It finished 2026-08-13
  19:11 UTC at step-46,429 / **4.674T** tokens / loss **2.68687**, exit 0,
  as trainer 0 of umbrella 8744247. It has no budget left and is not
  awaiting a continuation -- see [n512/](n512/README.md).
- **256N is COMPLETE** at step-92,859 / 4.674T / loss 2.652, finished by
  cont12 (8558531) on 2026-06-29. It is the base for CPT
  ([../../cpt/](../../cpt/README.md)) + SFT.
- **The active work is downstream of those two.** Umbrella `8828611`
  previously converted all five seats and left the three 2B follow-ons at
  persisted steps **22,300 / 28,000 / 39,200** (512N stage 2 / 256N stage 2 /
  512N constant-LR). The next umbrella, `8828612`, was much worse: only its
  20B-256 seat trained. All three 2B seats failed before a training step, so
  the verified 2B checkpoint heads did not move. This distinction matters:
  the umbrella itself consumed its full allocation, but its 2B result was
  **zero progress**, not a continuation of the successful 5/5 run.

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
