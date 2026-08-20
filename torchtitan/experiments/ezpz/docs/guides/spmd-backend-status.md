# SPMD backends on XPU: what works, what does not, and why

**Last updated:** 2026-08-20 (Sunspot, frameworks RC4 / oneAPI 2026.1.0)

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

| symbol in `_fsdp_param.py` | ours (2.13) | 2.14 nightly | pytorch main |
| --- | ---: | ---: | ---: |
| `_is_spmd_types_available` | 0 | 2 | 2 |
| `_resolve_spmd_types_for_storage` | 0 | 2 | 2 |
| `get_local_type` | 0 | 1 | 1 |
| file length | 1095 | 1329 | 1373 |

**Proven by A/B** (job 12473463) -- same venv, same config, same command,
only torch differs:

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

1. **Clone the working venv, do not build a fresh one.**
   `venvs/fw-2026.1-rc2` has `include-system-site-packages = true` and
   inherits torch from the RC4 conda env. A from-scratch venv loses the
   launcher wiring and dies with mpiexec `error parsing parameters`.
2. **Repoint the console-script shebangs** (45 of them) at the clone's
   python, or launched commands silently run the original venv.
3. **`pip install --ignore-installed`** -- otherwise pip sees the inherited
   conda torch as satisfying the requirement and no-ops.

The usable nightly window is narrow:

| nightly | verdict |
| --- | --- |
| before 2026-06-23 | no FSDP consumer; same failure as ours |
| 2026-07-01 .. 07-15 | inductor regression (`tensorssa_reduction`) |
| **2026-07-22** | **works** |
| 2026-07-29 onward | links `libpti_view.so.1`; this system has only `.so.0` |

Detail: [`known-bugs/spmd-types-plain-tensor.md`](known-bugs/spmd-types-plain-tensor.md),
[`known-bugs/spmd-types-newer-torch-attempt.md`](known-bugs/spmd-types-newer-torch-attempt.md).

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
-    spmd_backend: Literal["default", "full_dtensor", "spmd_types"] = "default"
+        "partial_dtensor", "full_dtensor", "spmd_types"  ] = "spmd_types"
-    - "default": use the existing TorchTitan parallelism paths.
+    - "partial_dtensor": use DTensor for model-parallel axes only.
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
