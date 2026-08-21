# SPMD backends on XPU: what works, what does not, and why

**Last updated:** 2026-08-21 (Sunspot, `frameworks/2026.1.0` / oneAPI 2026.1.0)

The reference environment throughout is the official module plus its ezpz
venv:

```bash
module use /opt/aurora/26.181.0/modulefiles
module load frameworks/2026.1.0
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env
# -> venvs/sunspot/torchtitan-aurora_frameworks-2026.1.0
```

Load the module **first**. `ezpz_setup_env` on its own prints
`[OK] Finished` and activates nothing, leaving `python3` as
`/usr/bin/python3` ([ezpz#216](https://github.com/saforem2/ezpz/issues/216)).

Upstream torchtitan flipped the default `parallelism.spmd_backend` to
`spmd_types` in [#4085][pr4085], and is removing `full_dtensor` in
[#4217][pr4217]. This is where each backend stands on our stack, what broke
along the way, and what to run today.

## TL;DR

| backend | status on the shipped stack | notes |
| --- | --- | --- |
| **`partial_dtensor`** | **use this** | what every production trajectory has always run -- upstream renamed `"default"` to this name in #4085 |
| `full_dtensor` | do not use | being deleted upstream (#4217); compiled agpt hits a `DeviceMesh` assertion |
| `spmd_types` | needs a newer torch | upstream's new default; fails on our build for a PyTorch-side reason, **not** a torchtitan one. On the 2.14 nightly it is throughput-neutral (-0.10% tps) but the bump costs **+6.25pp memory** -- see [1c](#1c-is-the-nightly-performant-throughput-yes-memory-costs-625pp). TP>1 is bit-identical too ([1d](#1d-tp1-on-the-nightly-works-a-retracted-scare-and-how-to-check)) |

Both ezpz config registries pin `partial_dtensor` as of `b2ff09632`.

## 1. `spmd_types`: our PyTorch is too old

Symptom, from torch's FSDP:

```
ValueError: When dp_mesh_dims is provided, all parameters must be DTensors
on the full SPMD mesh (e.g. via distribute_module).
Got plain tensor for parameter 'weight'.
```

Under `spmd_types`, torchtitan annotates parameters and hands FSDP plain
tensors carrying those annotations. FSDP is then supposed to convert them.
That conversion was added by pytorch [`da19cbd78`][da19] (#181519,
2026-06-23):

```python
self.is_spmd_types = (
    dist._is_spmd_types_available()
    and bool(spmd_local_type := spmd.get_local_type(param))
    and not isinstance(param, DTensor)
)
if self.is_spmd_types:
    param = self._resolve_spmd_types_for_storage(...)   # plain -> DTensor
```

Our build does not have it:

| symbol in `_fsdp_param.py` | `frameworks/2026.1.0` | 2.14 nightly | pytorch main |
| --- | ---: | ---: | ---: |
| `_is_spmd_types_available` | 0 | 2 | 2 |
| `_resolve_spmd_types_for_storage` | 0 | 2 | 2 |
| `get_local_type` | 0 | 1 | 1 |
| file length | 1095 | 1329 | 1373 |

The packaged torch is `2.13.0a0+gitcf30153` at
`/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/lib/python3.12/site-packages/torch`
(`_fsdp_param.py` md5 `0fcb55157796`). The earlier RC4 wheelforge conda ships
the **same** build, so neither packaged environment has the consumer.

**Proven by A/B** (jobs 12473463, 12473469) -- same venv, same config, same
command, only torch differs:

| torch | backend | steps | result |
| --- | --- | ---: | --- |
| **2.14.0.dev20260722+xpu** | `spmd_types` | **3/3** | **PASS** |
| 2.14.0.dev20260722+xpu | `partial_dtensor` | 3/3 | PASS |
| 2.13.0a0+gitcf30153 (shipped) | `spmd_types` | 0/3 | `params-not-DTensors` |

So torchtitan is behaving correctly and there is nothing to fix on our side.

> **For whoever builds the Aurora stack:** our torch is named
> `pytorch_2.13.0_patched_08_02_2026` but is missing a **2026-06-23** commit.
> The build is presumably branched from an older base than its name suggests,
> which likely affects more than this one code path.

### Running a newer torch, if you want to try it

`libsycl.so.9` now exists (`/opt/aurora/26.181.0/oneapi/compiler/latest/lib`),
so the old "no newer wheel runs on Aurora" rule is **stale** -- that was
against oneAPI 2025.3.1, which shipped only `.so.8`.

Recipe that works (`venvs/rc-plus-nightly`):

1. **Clone the venv, do not build a fresh one.**
   `venvs/sunspot/torchtitan-aurora_frameworks-2026.1.0` has
   `include-system-site-packages = true` and inherits torch from the
   frameworks conda. A from-scratch venv loses the launcher wiring and dies
   with mpiexec `error parsing parameters`. (That failure is a property of
   hand-built venvs, not of ezpz -- the shipped ezpz 0.26.0 launches fine.)
2. **Repoint the console-script shebangs** (45 of them) at the clone's
   python, or launched commands silently run the original venv.
3. **`pip install --ignore-installed`** -- otherwise pip sees the inherited
   conda torch as satisfying the requirement and no-ops.

> **Install into the CLONE, never the shipped venv.** Because torch is
> inherited rather than vendored, a stray `pip install` puts a `torch/` inside
> `venvs/sunspot/torchtitan-aurora_frameworks-2026.1.0/lib/python3.12/site-packages/`
> that shadows the conda one for every later job. I did this by accident and
> it produced a convincing false positive -- `spmd_types` "passing" on the
> shipped stack -- until I checked `torch.__file__` and found the nightly.
> Check that path if a result looks too good.

The usable nightly window is narrow:

| nightly | verdict |
| --- | --- |
| before 2026-06-23 | no FSDP consumer; same failure as ours |
| 2026-07-01 .. 07-15 | inductor regression (`tensorssa_reduction`) |
| **2026-07-22** | **works** |
| 2026-07-29 onward | links `libpti_view.so.1`; this system has only `.so.0` |

Detail: [`known-bugs/spmd-types-plain-tensor.md`](known-bugs/spmd-types-plain-tensor.md),
[`known-bugs/spmd-types-newer-torch-attempt.md`](known-bugs/spmd-types-newer-torch-attempt.md).

## 1b. Does the backend change the numerics? No -- bit-identical.

With `--debug.seed=42 --debug.deterministic`, agpt_20b, 1 node, TP=1,
seq=2048, 30 steps (job 12473494, torch `2.14.0.dev20260722+xpu`):

| comparison | identical loss | identical grad_norm | max abs delta |
| --- | ---: | ---: | ---: |
| `partial` vs `partial` (control) | **30/30** | **30/30** | 0.000000 |
| `spmd_types` vs `partial_dtensor` | **30/30** | **30/30** | 0.000000 |

**Bit-identical at every step.** The two backends are numerically equivalent;
the choice does not affect convergence.

### The earlier "divergent" reading was my error

A first pass reported max |d| = 11.31 between backends and 2.31 between two
runs of the SAME backend, and concluded the stack was nondeterministic. That
run passed only `--debug.seed=42` and **not** `--debug.deterministic`. A seed
fixes initialization and data order; it does not make XPU kernels
deterministic, so that spread was expected behavior and said nothing about
either backend.

The tell was available before the rerun and I missed it:
[known-bugs/xpu-determinism-rank-seqlen-interaction.md](known-bugs/xpu-determinism-rank-seqlen-interaction.md)
records that **single-node is deterministic at every size tested**, and the
run in question was single-node at seq=2048 -- inside the deterministic
regime. A 2.31-nat spread there contradicted our own documented finding,
which should have prompted a check of the flags rather than a conclusion
about the stack.

Do NOT use `--debug.deterministic_warn_only` to force a pass here: it
downgrades nondeterministic ops to warnings and would produce a fake
bit-identical result.

### Coverage: what is settled and what is not

Bit-identical means every printed loss AND grad_norm matched, under
`--debug.seed=42 --debug.deterministic`, against a same-backend control run
in the same job.

| config | control | spmd vs partial |
| --- | --- | --- |
| TP=1, 1 node, seq 2048 | 30/30 | **bit-identical** |
| TP=1, 1 node, seq 1024 | 10/10 | **bit-identical** |
| **MoE** (`moe_small`), TP=1 | 20/20 | **bit-identical** |
| TP=2 | 10/10 | **bit-identical** (job 12473502, with `584fb3d83` active -- see [1d](#1d-tp1-on-the-nightly-works-a-retracted-scare-and-how-to-check)) |
| 2 nodes | **0/20 -- control itself failed** | not resolvable yet |

TP=1 single-node is the configuration LEAST able to expose a difference --
no real tp axis, no cross-node collective -- so the MoE row (different model
family, expert sharding) carries more weight than the two agpt rows.

The TP>1 arms of the first attempt are void: every ezpz TP>1 run was dying on
a stale `get_spmd_backend()` global (fixed in `9c6073273`), the
`partial_dtensor` control included, so those cells tested nothing.

The 2-node cell is genuinely open. Its control -- two identical
`partial_dtensor` runs -- matched at 0 of 20 steps, while single-node cells
on the same build are fully deterministic. A dedicated sweep
({1,2} nodes x seq {1024, 2048, 4096}) is measuring the envelope on this
build, recording the first divergent step per cell so that "data/init
differs" (step 1) and "kernels accumulate" (later) can be told apart.

### `spmd_types` + TP>1 needs `loss.global_vocab_size` set (OUR config gap)

At TP=2 both `partial_dtensor` controls run 20/20, and `spmd_types` produces
**zero steps**:

```
TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'
  torchtitan/components/loss.py:134, in _LossParallelCrossEntropy.forward
```

`cross_entropy_loss` has two branches that both call
`_LossParallelCrossEntropy.apply`, and they are not equivalent
(`components/loss.py:42-57`):

```python
if isinstance(pred, DTensor):            # partial_dtensor
    ...apply(..., pred.shape[-1], "sum")             # vocab size computed inline
elif get_spmd_backend() == "spmd_types" and spmd_mesh_size("tp") > 1:
    ...apply(..., global_vocab_size)                 # forwarded, and never set
```

**This is not an upstream bug -- it is a config we never set.**
`CrossEntropyLoss.Config` exposes the field explicitly:

```python
class Config(BaseLoss.Config):
    global_vocab_size: int | None = None
    """Full vocabulary size, needed for spmd_types loss-parallel CE."""
```

The DTensor/`partial_dtensor` branch derives the vocab size from
`pred.shape[-1]` and never reads the field, which is why our configs have got
away with leaving it unset. The `spmd_types` branch requires it, and no ezpz
config sets it, so `None` reaches
`chunk_size = (global_vocab_size + tp_world_size - 1) // tp_world_size`.

I initially wrote this up as a second upstream bug. That was wrong: I checked
that no *caller* passes the argument, but not whether the loss *config*
exposes it. It does, with a docstring naming this exact use case. Upstream
also has a unit test covering the path
(`tests/unit_tests/test_loss.py:808`).

Fixed in `584fb3d83`: both ezpz registries now read the value off the model
spec, so the flavors that differ stay correct and cannot drift
(gemma 256128, Llama-3 128256, OLMo-2 100352).

Verified (job 12473502):

```
agpt_20b           loss.global_vocab_size=256128  model.vocab_size=256128  OK
agpt_30b_olmo2tok  loss.global_vocab_size=100352  model.vocab_size=100352  OK
moe_small          loss.global_vocab_size=256128  model.vocab_size=256128  OK
```

and at TP=2, 10 steps, deterministic: control 10/10, `spmd_types` vs
`partial_dtensor` **10/10 identical**. TP>1 parity is now answered.

Conclusion: `spmd_types` is numerically identical to `partial_dtensor`
everywhere the comparison is resolvable -- TP=1 and TP=2, dense and MoE, at
three sequence lengths. It remains unusable on the shipped torch (missing
FSDP consumer); the parity question is settled for whenever that floor
clears.

Multi-node parity is not resolvable by this method: the same-backend control
is itself nondeterministic across nodes (a cross-node loss all-reduce
ordering effect -- grad_norm matches while loss does not). See
[known-bugs/xpu-determinism-rank-seqlen-interaction.md](known-bugs/xpu-determinism-rank-seqlen-interaction.md).

## 1c. Is the nightly PERFORMANT? Throughput yes, memory costs 6.25pp

Ten jobs had run on `venvs/rc-plus-nightly` and every one was
correctness-only -- grep all ten for `mfu:` and you get zero lines. The
nightly was proven to produce the RIGHT numbers and had never been shown to
produce them at an acceptable RATE, which matters because 2.14 is the only
way to run `spmd_types` and upstream has made it the default.

Job 12473548, agpt_20b, 2N, TP=1, LBS=2, seq=2048, compiled, 30 steps.
Steps 1-10 discarded (torchtitan reports a CUMULATIVE average tps, so early
steps carry compile and warmup); the table is the mean of steps 11-30.

| arm | torch | backend | tps | MFU | mem |
| --- | --- | --- | ---: | ---: | ---: |
| A (reference) | 2.13.0a0+gitcf30153 | `partial_dtensor` | 521.6 | 21.80% | 50.31% |
| B | 2.14.0.dev20260722 | `partial_dtensor` | 522.9 | 21.86% | **56.56%** |
| C | 2.14.0.dev20260722 | `spmd_types` | 521.1 | 21.78% | **56.56%** |
| D (control) | 2.13.0a0+gitcf30153 | `partial_dtensor` | 527.8 | 22.06% | 50.31% |

**Throughput: no cost.** Torch bump alone (B vs A) +0.25% tps. Backend on top
of the bump (C vs B) -0.34%. Total cost of adopting upstream's default
(C vs A) **-0.10% tps / -0.020pp MFU**.

D is why those numbers mean anything. It re-runs A's exact configuration at
the end of the same job, and measures a **1.19% tps / 0.261pp noise floor**.
Every delta above is well inside it. exp05 separately measured 2.8% tps
across *different* jobs for one config, so treat anything under ~3% as
unattributable regardless.

**Memory: +6.25pp, and it tracks the TORCH BUMP, not the backend.** B and C
are identical at 56.56%; A and D are identical at 50.31%. 2.14 costs 6.25pp
whichever backend you choose. This is not noise -- A and D agree to the
decimal, so the measurement is exact.

That is free at 50% occupancy and is not free near the ceiling. The 20B at
seq=8192 already runs 84.86%, and the uncompiled 30B path sits at 93.80%:
**+6.25pp on top of 93.80% does not fit.** Check headroom before bumping torch
on anything running hot.

Caveat on absolute numbers: 21.8% MFU here is agpt_20b at LBS=2/seq=2048, not
a production shape -- exp05's 27.89% is the 30B at LBS=3. These four arms are
a valid *relative* A/B and not an absolute throughput claim.

### Verifying the arm actually ran the torch you think

Each arm gates on `torch.__version__` AND prints `torch.__file__`:

```
[B-t214-partial] torch=2.14.0.dev20260722+xpu
[B-t214-partial] from=.../venvs/rc-plus-nightly/lib/python3.12/site-packages/torch
[A-t213-partial] torch=2.13.0a0+gitcf30153
[A-t213-partial] from=.../conda_envs/RC4_.../lib/python3.12/site-packages/torch
```

The version string alone is not sufficient -- a stray install into the shipped
venv shadows the conda torch and has already produced one false positive here.
Print the path.

> **A grep that does NOT work for this.** Auditing which torch an arm used by
> taking the first `site-packages/torch` path out of its log gives the WRONG
> answer, and reports every nightly arm as having run on 2.13. Every log
> contains BOTH paths: a `_pytree.py` deprecation warning is emitted from the
> inherited conda torch before the venv's torch loads. The sound discriminator
> is behavioral -- `spmd_types` cannot produce a single step on 2.13, so any
> `spmd_types` arm with steps > 0 was necessarily on the nightly.

## 1d. TP>1 on the nightly works (a retracted scare, and how to check)

An earlier version of this section claimed TP>1 hit "a third, unidentified
failure" on the nightly. **That was wrong and is retracted.** TP=2 on the
nightly is bit-identical, and the mistake is worth keeping because the
verification method it got wrong is the reusable part.

Job 12473502, agpt_20b, TP=2, torch `2.14.0.dev20260722+xpu`, 10 steps, with
`global_vocab_size` actually set:

| comparison | identical loss + grad_norm |
| --- | ---: |
| `partial` vs `partial` (control) | **10/10** |
| `spmd_types` vs `partial_dtensor` | **10/10** |

Scope: one job, 10 steps, TP=2, single node. That is enough to refute
"TP>1 is broken" and NOT enough to claim TP>1 at production scale.

### The two errors behind the retraction

1. **Timezone.** The fix (`584fb3d83`) is stamped 16:38 **CDT**; the job
   `.o` mtimes read 20:04 and 21:13, which are **UTC** -- 15:04 and 16:13
   CDT, i.e. BEFORE the fix, not after. Comparing a local-time commit against
   UTC file times inverted the whole conclusion.
2. **A grep scoped to one failure mode.** Searching only for
   `must be DTensors`, finding none, and concluding both known modes were
   excluded. The real error was in the log the whole time:
   `chunk_size = (global_vocab_size + tp_world_size - 1)` ->
   `TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'`,
   which IS the vocab bug.

### Check the dumped config, not the clock

Timestamps are the wrong instrument for "did this run have my fix?" -- they
depend on timezone, on whether an mtime is a start or a finish, and on when
the job read the working tree rather than when it was submitted. Every run
dumps its own resolved config, so ask the run:

```bash
grep -ao '"global_vocab_size": [^,]*' <arm>.log | head -1
```

`null` means the fix was NOT active in that arm; `256128` (gemma) or `100352`
(OLMo-2) means it was. Every TP>1 arm that ever failed dumps `null`. **No
failing TP>1 arm has ever run with the fix**, so there is no unexplained
failure mode to chase.

Generalizes past this one field: when a result hinges on whether some
config/fix was live, find the value in the run's own dump. It is evidence the
run produced about itself, and it does not care what time it was.

## 2. `full_dtensor`: dead end, and it broke compiled agpt

Symptom:

```
AssertionError: expected all tensors_saved_with_vc_check to be Tensors,
got types: [..., <class 'torch.distributed.device_mesh.DeviceMesh'>]
```

The trigger is **not** the backend as such. Running all three backends across
TP degrees shows it tracks whether `model.parallelize()` ran:

| | TP=1 | TP=2 |
| --- | --- | --- |
| `partial_dtensor` | PASS (call skipped) | vc_check (call ran) |
| `full_dtensor` | vc_check (call ran) | vc_check (call ran) |
| `spmd_types` | params-not-DTensors | params-not-DTensors |

All three of **compile + AC + `model.parallelize`** are required; drop any one
and it does not fire. Selective AC avoids it while keeping compile and TP,
which is why the flex MoE configs use it.

`full_dtensor` is being deleted upstream regardless (#4217), so it is not
worth working around.

Detail: [`known-bugs/agpt-full-dtensor-vc-check.md`](known-bugs/agpt-full-dtensor-vc-check.md).

## 3. Naming: `"default"` became `partial_dtensor`

#4085 renamed the enum values *and* changed the default:

```diff
-spmd_backend: Literal["default", "full_dtensor", "spmd_types"] = "default"
+spmd_backend: Literal["partial_dtensor", "full_dtensor", "spmd_types"] = "spmd_types"
-  - "default": use the existing TorchTitan parallelism paths.
+  - "partial_dtensor": use DTensor for model-parallel axes only.
```

`"default"` and `partial_dtensor` are the same code path under two names. Old
logs reporting `"spmd_backend": "default"` -- including the 482-step 30B
convergence run -- were on what is now called `partial_dtensor`.

## Upstream signals

- [#4217][pr4217] removes `full_dtensor` (2026-08-19)
- [`b64d3f6a9`][b64d] pins the rl+hf CI suites to `partial_dtensor`
- `tests/integration_tests/h100.py:45` overrides to `full_dtensor`
- No open PR touches `spmd_distribute_tensor` / `spmd_types.py`, and no open
  issue matches this error

## Sync note

`ezpz/fsdp_compat.py` (`bf3c4f47d`) resolves `resolve_fsdp_mesh` /
`resolve_sparse_fsdp_mesh` from either the pre- or post-#4217 location, so the
80th sync will not break on import. `validate_config` was removed upstream and
its ezpz call sites dropped to match -- verified not to change the sharding
plan (all configs reproduce their memory to within 0.15pp).

[pr4085]: https://github.com/pytorch/torchtitan/pull/4085
[pr4217]: https://github.com/pytorch/torchtitan/pull/4217
[b64d]: https://github.com/pytorch/torchtitan/commit/b64d3f6a9
[da19]: https://github.com/pytorch/pytorch/commit/da19cbd78
