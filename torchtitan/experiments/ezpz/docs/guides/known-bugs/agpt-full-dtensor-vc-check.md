# agpt on full_dtensor: vc_check/DeviceMesh, and a pin that was justified uncompiled

**Status:** OPEN upstream; NOT BLOCKING us. Every ezpz config is pinned to
`partial_dtensor` (`b2ff09632`), which is production-validated at 30B --
1589 steps and a compiled resume, see the answer to open question 1 below.
The assertion itself is still unexplained under `full_dtensor`, and upstream
is deleting that backend anyway (`601cf4d23`, #4217).

**Affects:** every agpt config (20B and 30B both confirmed) at TP=1, compiled.
**Jobs:** 12473394, 12473398, 12473403, 12473417, 12473418, 12473419

## Symptom

```
AssertionError: expected all tensors_saved_with_vc_check to be Tensors,
got types: [<class 'torch.Tensor'> x12, <class 'torch.distributed.device_mesh.DeviceMesh'>]
```

Dies before step 1 at ~7% memory -- during graph construction, not from
allocation pressure.

## The matrix (agpt_20b, 2N, TP=1, LBS=2, seq=8192, 3 steps)

| backend | compile | steps | memory | result |
|---|---|---|---|---|
| `full_dtensor` | on | 0/3 | 7.27% | vc_check (DeviceMesh) |
| `full_dtensor` | off | 0/3 | 7.27% | level_zero 40 (in `cross_entropy`) |
| **`partial_dtensor`** | **on** | **3/3** | **84.86%** | **PASS** |
| `partial_dtensor` | off | 0/3 | 7.27% | level_zero 40 |

Two independent axes, and they are easy to conflate:

- **`full_dtensor` is what breaks the compiled path.** `partial_dtensor`
  compiled is the only green cell.
- **`--compile.no-enable` is NOT a workaround here.** It fails under both
  backends with level_zero error 40 inside `cross_entropy` -- a separate
  capacity problem that compile was hiding. The 20B at seq=8192 with gemma's
  256k vocab needs ~16.8 GB of fp32 logits per rank.

## What it is NOT

Each of these was proposed and then killed by measurement, in this order:

- **The blendcorpus `positions` change.** Runs log `"emit_positions": false`
  and fail identically. (Also: `decoder.py:238` appears in the traceback for
  every forward with `positions=None` -- it is the layer loop, not evidence.)
- **"compile is broken", per TODO.md #2.** That entry is about the 80B; the
  workaround it gives does not work here, see above.
- **seq_len / logits capacity for the vc_check arm.** 8192, 4096 and 2048 all
  fail identically at ~7% memory. Failing at 2048 with 4.45 GiB allocated is
  not a memory wall.
- **The env.** Re-run under the user's `frameworks_rc` (miniforge +
  `intel_gpu_umd_aicoe/2026.06.19` + conda) instead of
  `module load frameworks/2026.1.0`: same assertion. A different GPU
  user-mode driver was a reasonable suspect and is now excluded.

## Where the bisect landed

`60009b04d` ("model.parallelize must run unconditionally") vs its parent,
crossed with both backends. `partial_dtensor` is a built-in control -- the
commit only edits the `full_dtensor` branch, so the legacy arm must be
unaffected:

| commit | full_dtensor | partial_dtensor |
|---|---|---|
| `7bd902b5c` (parent) | `ValueError: all parameters must be DTensors` | 3/3 PASS |
| `60009b04d` | vc_check (DeviceMesh) | 3/3 PASS |

Control holds: `partial_dtensor` passes at both commits with byte-identical
memory (54.29 GiB).

So **`full_dtensor` was already broken before that commit**, with a different
error. `60009b04d` fixed the `ValueError` -- parameters really do become
DTensors now, as intended -- and advanced the failure to the next wall.
Reverting it would not restore agpt; it would swap the assertion back for a
`ValueError`.

## The pin that should not have been global

`621ac2406` pinned every agpt config to `full_dtensor`, citing job 12473350:
"full_dtensor runs 4/4 steps, spmd_types 0".

That probe ran `--compile.no-enable`. The measurement was real but its scope
was not: an uncompiled result was used to pin the backend for compiled
production configs, where `full_dtensor` does not work. The pin's own
reasoning ("pin the backend that works until the spmd_types path is
understood") is sound; the evidence just did not cover the compiled case.

## Was full_dtensor ever green? No -- and the rename is why it looked that way

The user pushed back that they remembered a successful full_dtensor run.
Worth checking, and the check is decisive. Scanning every log on disk for
{compile enabled} AND {>2 real step lines}, then reading each one's backend:

| steps | backend | run |
|---:|---|---|
| 482 | `"default"` | 12473304 (30B convergence) |
| 3 | `partial_dtensor` | 12473418 bisect, both arms |

**No compiled `full_dtensor` run has ever produced a single step.**

The 482-step run reported `"spmd_backend": "default"`, which is not one of
today's three legal values -- the validator rejects it. Upstream #4085
(`5ab3a0fd1`, 2026-08-18) RENAMED the values:

```diff
-spmd_backend: Literal["default", "full_dtensor", "spmd_types"] = "default"
+spmd_backend: Literal["partial_dtensor", "full_dtensor", "spmd_types"] = "spmd_types"
-  - "default": use the existing TorchTitan parallelism paths.
+  - "partial_dtensor": use DTensor for model-parallel axes only.
```

`"default"` and `partial_dtensor` are the same code path under two names. So
the remembered successful run WAS on partial_dtensor -- the label changed
underneath it, not the behavior. Consistent with everything above.

One near-miss worth recording: job 12473387 shows `full_dtensor` and 153
steps, which looks like a counterexample until you check `compile.enable`,
which is `false`. Its log also contains the string "torch.compile" -- from a
pytree warning path, not from compiling. Grepping for that string as a proxy
for "was it compiled" gives the wrong answer; read the dumped config instead.

## ROOT CAUSE (2026-08-19, jobs 12473420 / 12473421 / 12473422)

The framing above -- "full_dtensor is broken" -- is wrong. Running all three
backends x TP=1,2 (job 12473420) shows the failure does not track the backend
at all:

| | TP=1 | TP=2 |
|---|---|---|
| `partial_dtensor` | **PASS** | vc_check |
| `full_dtensor` | vc_check | vc_check |
| `spmd_types` | params-not-DTensors | params-not-DTensors |

vc_check fires in exactly the cells where `model.parallelize()` ran:

- `partial_dtensor` TP=1 -- gated on `tp_enabled`, so SKIPPED -> PASS
- `partial_dtensor` TP=2 -- `tp_enabled` true, so RAN -> vc_check
- `full_dtensor` TP=1/2 -- unconditional, so RAN -> vc_check

The trigger is the new `Module.parallelize` + `sharding_config` path, which
is exactly what CLAUDE.md says this bug requires. `partial_dtensor` at TP=1
is not "the backend that works" -- it is the one cell that skips the call.

Job 12473421 isolated the other two legs:

| arm | result |
|---|---|
| FullAC + compile | vc_check |
| AC=none + compile | no vc_check (hits a separate capacity wall) |
| FullAC + no compile | **3/3 PASS** |

So all three are required: **compile + AC + model.parallelize**. Remove any
one and the assertion does not fire.

## Selective AC avoids it

Because AC is load-bearing, the AC *policy* is a real lever. agpt previously
offered only `none` and `full`; `selective` was added (`68946110a`) and
tested (job 12473422, seq=4096):

| AC | backend | TP | steps | memory | result |
|---|---|---|---|---|---|
| **selective** | full_dtensor | 2 | **3/3** | 73.47% | **PASS** |
| full | full_dtensor | 1 | 0/3 | 6.99% | vc_check |
| full | partial_dtensor | 2 | 0/3 | 6.99% | vc_check |

The FullAC controls reproduce the assertion in the same job, so the selective
pass is a real contrast and not a lucky run. **No selective arm has ever
produced the vc_check assertion.**

Selective keeps compile, AC and TP together -- the only configuration so far
that does.

### Its limit is memory, not the bug

At seq=8192 (job 12473423) the selective arms fail on capacity instead:

| arm | steps | memory | result |
|---|---|---|---|
| selac TP1 LBS2 seq8192 | 0/5 | 7.27% | XPU OOM (320 MiB alloc) |
| selac TP1 LBS1 seq8192 | 1/5 | 93.99% | level_zero 40 |
| selac TP2 LBS2 seq8192 | 0/5 | 7.27% | level_zero 40 |
| FullAC TP1 LBS2 seq8192 | 0/5 | 7.27% | vc_check |

That is the expected trade: selective AC saves more activations than FullAC
by design, buying speed with memory. The 20B at seq=8192 does not have the
headroom on 2N. Note the FullAC control still fails with vc_check at the same
settings, so this is not selective being worse -- it is selective trading one
failure mode for a different, well-understood one.

## RESOLVED 2026-08-20: repinned to partial_dtensor

Both config registries now pin `partial_dtensor` (`b2ff09632`). Verified on
committed defaults with NO backend flag (job 12473451):

| config | steps | memory | |
|---|---|---|---|
| `agpt_20b` (the originally-failing command) | 5/5 | 84.86% | **PASS** |
| `moe_small` | 5/5 | 72.55% | PASS |
| `moe_10b_2b` | 5/5 | 79.97% | PASS |
| `moe_10b_2b_sdpa` | 5/5 | 73.98% | PASS |

The MoE arms are the ones that mattered: they were green on `full_dtensor`
and had to stay green. They did, at memory within 0.1pp of their previous
values.

This was forced regardless of the vc_check analysis: upstream is DELETING
`full_dtensor` (`601cf4d23`, #4217) and the file is already gone from
upstream/main, so the next sync would have removed what our parallelize.py
imports.

Also fixed: the venv had `spmd_types==0.2.1` against a pinned `0.2.3`. Now
0.2.3, torch untouched. That was NOT the cause of the spmd_types failure
(0.2.3 tested in an overlay first, job 12473448, fails identically) but the
divergence would have confounded the next investigation.

**Process note.** One arm of the pre-change verification (12473450) reported
a bogus `cannot import name 'no_typecheck'`. I had run the pip install six
seconds before that arm started, so it read a half-swapped package. 0.2.3
does export `no_typecheck` (`runtime.py:984`). Do not install into the shared
venv while jobs are live -- the failure looks like a real result.

## What to do now

Use `--parallelism.spmd-backend=partial_dtensor` for compiled agpt runs.

Open questions, in order:
1. ~~Should the agpt pin be `partial_dtensor`? It needs a convergence check
   first -- this session has only smoke-tested it (3/3 steps).~~
   **ANSWERED 2026-08-21.** The pin landed (`b2ff09632`) and the convergence
   check is done: job 12473515 has run the 30B to step 1589 on it, loss
   12.03 -> 2.263, zero NaN, 28.3% MFU. It also resumed COMPILED from
   step-800 without vc_check and wrote three checkpoints since -- so the
   compiled round trip is verified on this pin, not merely smoke-tested.
   See [exp08](../../production/agpt/30b-exp/exp08-convergence.md).
2. Where exactly does the `DeviceMesh` enter saved-for-backward under
   `full_dtensor`? Not yet localized.
3. Separately: the uncompiled level_zero 40 in `cross_entropy` at seq=8192 is
   a real capacity limit worth its own fix (chunked CE already exists as
   `agpt_20b_chunkedce`).

Note upstream just landed `b64d3f6a9 [ci] pin rl+hf tests to
partial_dtensor` -- they are steering CI away from the new backend too,
though for an unrelated HF-weight-loading reason.
