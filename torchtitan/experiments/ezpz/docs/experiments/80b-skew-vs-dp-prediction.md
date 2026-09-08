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

### Multi-step means, and a deceleration the two-point fit could not see

The result above reads step 1 of each arm. Across all available steps:

| dp | n | mean skew | sd | cv |
|----|---|-----------|-----|-----|
| 48 | 2 | 163.18 | 0.88 | 0.54% |
| 96 | 40 | 216.88 | 1.64 | 0.75% |
| 192 | 35 | 273.04 | 1.99 | 0.73% |

Within-arm cv is under 0.8% against between-arm gaps of 26-33%, so the arms
are separated by roughly 35 standard deviations. The step-1 reading was not a
fluke: ratios move only from (1.337, 1.257) to (1.329, 1.259).

**The ratios are falling: 1.3291 then 1.2589, a drop of 0.070 per doubling.**
The relationship is close to log-linear but decelerating, which two points
could not have revealed. That changes the dp=384 forecast:

| model | dp=384 skew |
|-------|-------------|
| log-linear (constant 1.2589) | **343.7** |
| deceleration continues (1.1887) | **324.6** |

My committed prediction quoted 342.8 from the log-linear fit. With three
points the honest range is **325-344**, and a 128N run would discriminate at
6%. The direction is unambiguous either way -- both are far above the 273 at
dp=192.

Caveat: the dp=48 arm has n=2 so far. Its mean will firm up as it runs, and
the 48->96 ratio is the one most likely to move.

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

### Correction: the runner-up ranking is NOT depth-independent across dp

Earlier notes in this investigation (including mine) described ranks 1-4 as a
"flat, depth-independent population" scattered through the stack. That holds
*within* an arm. It does not hold *across* dp:

| dp | layers occupying ranks 1-4 |
|----|----------------------------|
| 48 | 82, 50, 14, 11 |
| 96 | 65, 57, 36, 5 |
| 192 | **9, 8, 10, 21** |

**dp=192 is an outlier, not a trend.** Mean runner-up layer index:

| dp | mean layer index | n |
|----|------------------|---|
| 48 | 39.2 | 20 |
| 96 | 44.0 | 160 |
| 192 | **10.6** | 160 |

48 and 96 are both mid-stack and close together; 192 drops to layer ~10. That
is a **step change at dp=192**, not a monotonic migration. An earlier version
of this section said "the high-gradient layers migrate toward the input as dp
rises", which reads a gradient into what is really two similar arms and one
different one. Only `lm_head.weight` holds rank 0 at every dp.

Caveats before anyone builds on this: dp=48 has n=20 (the arm is young), and
each arm runs a different GAS (8 / 4 / 2) because GAS absorbs the dp change at
fixed GBS. So "dp=192 behaves differently" is equally consistent with
"GAS=2 behaves differently", and these data cannot separate them.

Two consequences:

1. **"Depth-independent" was an artifact of looking at one arm.** The scatter
   within a single dp is real, but the *location* of the scatter moves with
   parallelism, which is a stronger and more specific claim than either
   "depth matters" or "depth does not".
2. **The concentration result is unaffected.** Skew measures max/mean, and the
   max is `lm_head` in every arm. Which layers occupy ranks 1-4 does not enter
   it.

Also worth recording: the ranking is not perfectly frozen even within an arm.
At dp=192 the top four are identical across all 40 steps, but rank 5 alternates
between layer 21 (29 steps) and layer 0 (11 steps). "Frozen top-5" was an
overstatement from reading five steps of one arm; "frozen top-4, contested
rank 5" is what the 40-step data shows.

### `tok_embeddings` is the control that makes `lm_head` interesting

`tok_embeddings.weight` has the **same shape** as `lm_head.weight` (vocab x
dim, the two largest tensors in the model) and **never once enters the top-5**
across 40 steps at any dp.

So the concentration is not "the biggest tensor gets the biggest gradient". Two
tensors of identical size sit at opposite ends of the distribution. Whatever
distinguishes them is the mechanism, and the obvious candidate is that one is
on the loss side of the network and the other is not.

(Weight tying would have made them the same tensor and explained nothing --
checked, and `enable_weight_tying` appears exactly once in the registry, in
`agpt_2b_tied`. The 80B is untied.)

## FOUR POINTS: it is a power law, and my first two shape claims were noise

`12474807` (dp=24) returned **131.15**, above BOTH standing predictions
(decelerating 116.6, log-linear 122.8) by 12.5% and 6.8%.

With four points the per-doubling ratios are **1.244, 1.329, 1.259** -- they do
not decline monotonically, they peak in the middle. So "decelerating" was
itself an artifact of having exactly three points, exactly as "log-linear" was
an artifact of having two.

A single power law fits all four to within 2.1%:

```
skew = 41.61 * dp^0.3584
```

| dp | observed | fit | residual |
|----|----------|-----|----------|
| 24 | 131.15 | 129.98 | +0.9% |
| 48 | 163.18 | 166.64 | -2.1% |
| 96 | 216.88 | 213.63 | +1.5% |
| 192 | 273.04 | 273.88 | -0.3% |

Residuals show no trend with dp, so the ratio wobble is scatter around a power
law rather than structure. The exponent is **0.358**, close to 1/3.

### The methodological point

Three successive shape claims, each overturned by the next data point:

| points | inferred shape | prediction for the next point | outcome |
|--------|----------------|-------------------------------|---------|
| 2 | log-linear | dp=48 -> 172.7 | 162.3, off by 6% but right direction |
| 3 | decelerating | dp=24 -> 116.6 | **131.2, off by 12.5%** |
| 4 | power law, exp 0.358 | dp=12 -> **101.4** | pending (`12474808`) |

Every intermediate shape was a real pattern in the data available and wrong
about the data that followed. The underlying finding -- **skew grows
monotonically with dp** -- survived all three revisions unchanged, which is
worth separating from the functional form, where I was wrong twice.

The dp=12 arm is already running and discriminates cleanly: power law 101.4,
decelerating 79.4, log-linear 92.4. Those are 9-28% apart.

### Caveat that has not gone away

Every one of these four arms varies GAS inversely with dp (8/4/2 and now 16 at
dp=24), because GBS is held at 384 seqs. So this is a power law in
*dp-with-inverse-GAS*. `12474809` holds dp=192 fixed and moves GAS 2 -> 1 to
separate them; until it reports, the exponent 0.358 belongs to the pair, not
to dp alone.

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
