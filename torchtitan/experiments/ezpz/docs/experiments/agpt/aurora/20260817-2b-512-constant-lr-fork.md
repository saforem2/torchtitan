# 2B-512 constant-LR fork: the decay phase is worth <0.01 nats so far

**Date:** 2026-08-17 (fork launched 2026-08-13)
**Chain key:** `2b_v2_512_constlr_from9200` (`cls: wandb_only`)
**W&B runs:** `ww88slec` (8744247 t3), `ijfo395o` (8756070 t3), `xii94czx` (8756957 t3, live)
**Checkpoints:** `/flare/AuroraGPT/foremans/runs/agpt-2b-constlr-from9200/torchtitan-ezpz/outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288-constlr-from9200`
**Status:** running, step ~21,092

## The question

The canonical 2B-512 chain runs a decaying LR schedule. This fork branches off
it at **step-9200**, right before decay begins, and holds LR **constant at
2.28e-5**. Same corpus (olmo-mix-1124), same GBS (12,288), same seed weights --
the only difference is the schedule. So it is directly comparable step-for-step
to the canonical chain, unlike the stage-2 dolmino chain, which changes corpus.

If the decay phase is what earned the canonical chain its final loss, the two
curves should separate. If it is not, they should not.

## Result: they have not separated

Loss at matched steps, constant-LR vs canonical (canonical values from the
committed ground-truth store, `docs/production/metrics/2b_v2_512.csv`):

| step | constant-LR | canonical | delta | LR |
|-----:|------------:|----------:|------:|----|
| 9,201 | 6.5170 | 2.8506 | +3.6664 | 2.28e-5 |
| 12,000 | 2.8136 | 2.8058 | **+0.0078** | 2.28e-5 |
| 16,000 | 2.7733 | 2.7670 | **+0.0063** | 2.28e-5 |
| 20,000 | 2.7492 | 2.7479 | **+0.0013** | 2.28e-5 |
| 21,000 | 2.7409 | 2.7363 | **+0.0046** | 2.28e-5 |

Across ~12,000 steps the constant-LR arm stays within **+0.001 to +0.008 nats**
of the decaying chain, with no widening trend -- the gap at step 20,000 is
smaller than at 12,000. At this point in training the decay is not buying
measurable loss.

**The step-9201 row is a restart transient, not a gap.** 6.52 is the first
logged step after the fork resumed from the seed; it recovers to ~2.81 within
the first few hundred steps. Reading it as a real +3.67 regression would be
wrong.

## Caveats -- do not over-read this

- **The interesting part has not happened yet.** The canonical chain's decay
  runs all the way to step 46,429 / 4.674T tokens, and LR decay characteristi-
  cally pays off *late*, as the schedule approaches its floor. This comparison
  covers steps 9.2k-21k, less than half the run, while the canonical LR is
  still relatively high. A null result here is entirely compatible with a real
  gap at 40k+.
- **Loss is not the deliverable.** The v1-vs-v2 experience is the standing
  reminder: two chains can sit close on loss and diverge sharply on downstream
  evals. Nothing here has been eval'd. Any claim about the decay phase
  mattering or not should be settled on ARC/HellaSwag/etc., not on this table.
- **Single seed, single arm.** No repeat, so small deltas are not separable
  from run-to-run noise. The measured cross-job noise floor on the 30B work was
  ~3% on throughput; the equivalent for loss at this scale has not been
  established, but 0.005 nats is plainly inside anything that would count.

## Why this page exists

The fork had been running across **three umbrellas** and reached ~21k steps
with **no trajectory entry at all** in `utils/trajectories.py`. Consequences:

- the live board rendered it as a raw truncated checkpoint-dir name
  (`2b-sophiag-olmo-mix-1124-n512-gbs1`), with no token target and no W&B link;
- none of its ~11.8k steps past the fork point appeared on any chart;
- its three W&B run-ids were nowhere on record, so the run history was only
  reachable by grepping umbrella console logs.

Registered 2026-08-17 as `cls: wandb_only` -- it is a deliberate LR ablation
rather than a production chain, so it belongs on the board and in `prod_dash`
without being required on the canonical overlay charts.

Found while checking something unrelated: the live board showed 20B-512 in
state `R` with a six-day-old W&B heartbeat, which turned out to be a missing
run-id on that chain too (`ctbs1be4`). Both were fixed in the same commit.

## Reproduce the table

```bash
# canonical, from the committed store
python3 -c "
import csv
rows=list(csv.DictReader(open(
  'torchtitan/experiments/ezpz/docs/production/metrics/2b_v2_512.csv')))
d={int(float(r['_step'])): float(r['loss_metrics/global_avg_loss'])
   for r in rows if r.get('loss_metrics/global_avg_loss')}
for s in (12000,16000,20000,21000):
    near=[k for k in d if abs(k-s)<120]
    if near:
        k=min(near,key=lambda k:abs(k-s)); print(k, round(d[k],4))"

# constant-LR arm, from W&B (run on Aurora, where credentials live)
./.venv/bin/python3 -c "
import wandb; api=wandb.Api()
for rid in ('ww88slec','xii94czx'):
    r=api.run(f'aurora_gpt/torchtitan.ezpz.train/{rid}')
    h=[x for x in r.scan_history(keys=['_step','loss_metrics/global_avg_loss'])]
    print(rid, r.state, len(h), h[0]['_step'], h[-1]['_step'])"
```

## Next

- Let it run. Re-check the delta at step ~30k and again near 46k, where the
  canonical LR is low and the decay should matter most if it matters at all.
- Eval a matched checkpoint pair (constant-LR vs canonical at the same step)
  before drawing any conclusion about the schedule.
