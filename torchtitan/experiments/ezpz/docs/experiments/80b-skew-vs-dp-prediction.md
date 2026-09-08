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

## RESULT: confirmed. skew = 162.30 at dp=48

`12474806`, step 1, matched GBS/seed/LR:

| dp | skew | mean layer gradnorm |
|----|------|---------------------|
| **48** | **162.30** | 0.00156439 |
| 96 | 217.06 | 0.00115764 |
| 192 | 272.79 | 0.00090154 |

**Predicted 172.7, observed 162.30** -- 6.0% error on a value extrapolated
outside the measured interval and committed to git (`aa12e8d77`) before the
run started.

The predicted drop was 20.4%; the observed drop is **25.2%**. The effect is
real and slightly *stronger* than the two-point fit implied.

Per-doubling ratios across three points:

```
48 ->  96 : 1.3374
96 -> 192 : 1.2567
```

Consistent to within 6%, so near log-linear in dp with a mild flattening as dp
grows. Not the third case the falsifier anticipated (a much smaller drop), and
emphatically not the flat-or-higher outcome that would have killed it.

### What is now established

**Gradient concentration in `lm_head.weight` scales monotonically with data
parallelism**, over a 4x range of dp, at fixed GBS, seed, LR trajectory,
optimizer and model. Each doubling of dp raises skew by ~26-34%, driven by the
mean layer gradient falling while `lm_head`'s barely moves.

Combined with the batch result -- 16x GBS leaves means invariant and scales
variances by sqrt(16) -- the two knobs have cleanly separated signatures:

| knob | means | variances |
|------|-------|-----------|
| GBS (16x) | invariant, 0.03-0.20% | scale by sqrt(16) |
| dp (4x) | scale monotonically | unchanged, cv ratios 0.94-1.05 |

### What is still conjecture

That concentration is what *breaks* the model. None of these runs failed, so
this is healthy-regime structure. The hypothesis predicts the failure
threshold should track skew rather than dp directly, and that dp=384 gives
skew ~343 -- testable when 128 nodes free up.

The mechanism question this raises is sharper than the parallelism one: **why
does the vocabulary projection resist averaging when every other tensor does
not?** That is a question about the output layer and the loss, not about
distribution.

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
