# agpt on full_dtensor: vc_check/DeviceMesh, and a pin that was justified uncompiled

**Status:** OPEN. Workaround: `--parallelism.spmd-backend=partial_dtensor`.
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

```
-    spmd_backend: Literal["default", "full_dtensor", "spmd_types"] = "default"
+        "partial_dtensor", "full_dtensor", "spmd_types"  ] = "spmd_types"
-    - "default": use the existing TorchTitan parallelism paths.
+    - "partial_dtensor": use DTensor for model-parallel axes only.
```

`"default"` and `partial_dtensor` are the same code path under two names. So
the remembered successful run WAS on partial_dtensor -- the label changed
underneath it, not the behavior. Consistent with everything above.

One near-miss worth recording: job 12473387 shows `full_dtensor` and 153
steps, which looks like a counterexample until you check `compile.enable`,
which is `false`. Its log also contains the string "torch.compile" -- from a
pytree warning path, not from compiling. Grepping for that string as a proxy
for "was it compiled" gives the wrong answer; read the dumped config instead.

## What to do now

Use `--parallelism.spmd-backend=partial_dtensor` for compiled agpt runs.

Open questions, in order:
1. Should the agpt pin be `partial_dtensor` rather than `full_dtensor`? That
   is the only configuration measured green for compiled runs. It needs a
   convergence check first -- `partial_dtensor` is the legacy path and this
   session has only smoke-tested it (3/3 steps).
2. Where exactly does the `DeviceMesh` enter saved-for-backward under
   `full_dtensor`? Not yet localized.
3. Separately: the uncompiled level_zero 40 in `cross_entropy` at seq=8192 is
   a real capacity limit worth its own fix (chunked CE already exists as
   `agpt_20b_chunkedce`).

Note upstream just landed `b64d3f6a9 [ci] pin rl+hf tests to
partial_dtensor` -- they are steering CI away from the new backend too,
though for an unrelated HF-weight-loading reason.
