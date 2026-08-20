# SPMD backends on XPU: what works, what does not, and why

**Last updated:** 2026-08-20 (Sunspot, `frameworks/2026.1.0` / oneAPI 2026.1.0)

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
| `spmd_types` | needs a newer torch | upstream's new default; fails on our build for a PyTorch-side reason, **not** a torchtitan one |

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
| TP=2 | -- | pending (first attempt void) |
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

Conclusion so far: the backends agree wherever the comparison is resolvable,
which is not yet everywhere.

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
