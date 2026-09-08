# Does gradient concentration track dp? A prediction and its falsifier

**Written before `12474806` returns.** The point of committing this now is that
a 20% drop is easy to rationalize after the fact in either direction.

## The observation

Two arms, matched on GBS (384 seqs), seed (42), LR trajectory
(`5.376344e-09`, warmup 25), optimizer, model and host quality. n=17 steps:

| | dp=96 (`12474768`) | dp=192 (`12474803`) | change |
|---|---|---|---|
| `layer_gradnorm_skew` | 217.06 | 272.79 | **+25.7%** |
| mean layer gradnorm | 0.00115764 | 0.00090154 | **-22.1%** |
| max = `lm_head` | 0.251287 | 0.245944 | -2.1% |

The skew rise is entirely in the **denominator**. Doubling dp suppresses the
typical layer's gradient about ten times harder than it suppresses
`lm_head`'s.

Note the contrast with the batch experiment, which used the same metrics on
the same model: a **16x GBS change left means invariant (0.03-0.20%) and
scaled variances by sqrt(16)**. Doubling dp does the reverse -- means move,
variances do not (cv ratios 0.94-1.05). Two knobs, two distinct signatures.

## The hypothesis

Gradient concentration in `lm_head` intensifies with data parallelism. If
concentration is what eventually destabilises the model, higher dp makes it
monotonically worse -- which would account for a dp-dependent failure without
dp altering gradient noise at all, and the noise is measurably unaltered.

## The prediction

Two independent routes, both extrapolating the 96->192 interval **downward**
to a point outside it:

1. Per-doubling component ratios (mean /0.779, max /0.979) -> skew **172.7**
2. Log-linear in dp, `skew(dp) = 217.06 * 1.2567^log2(dp/96)` -> skew **172.7**

They agree to 0.2%, which is a coincidence worth noting rather than evidence
of correctness -- both derive from the same two points.

**Prediction: `12474806` (dp=48) returns skew ~173, a drop of ~20% from 217.**

For reference the same fit gives dp=384 -> 342.8, testable when 128 nodes are
free.

## The falsifier

**If dp=48 returns skew ~217 (flat) or higher, the hypothesis is wrong** and
skew is not tracking dp. That outcome is as informative as confirmation, which
is why the cheap downward rung ran immediately instead of waiting for 128 free
nodes to test dp=384 upward.

A third possibility: skew drops but by much less than 20% (say to 205). That
would mean the effect is real but not log-linear, and the two-point
extrapolation is unreliable -- in which case dp=384 becomes necessary rather
than confirmatory.

## What it will not settle

- Doubling dp at fixed GBS **halves GAS** (here 4 -> 8 going down to dp=48),
  so the accumulation/reduction split moves with dp. At fixed batch these are
  not separable, and any "dp effect" is really a "dp-and-GAS effect".
- Three points on a monotone curve is still a curve fit, not a mechanism. It
  would motivate looking at *why* the vocabulary projection resists averaging
  when every other tensor does not -- which is a question about the loss and
  the output layer, not about parallelism.
- None of these runs failed, so this describes healthy-regime structure. The
  link from concentration to instability remains conjecture.
