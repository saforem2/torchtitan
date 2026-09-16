# Sync 84 on XPU: it trains, on torch 2.14

**Bottom line:** the sync-84 merge trains on Aurora/Sunspot XPU hardware, but
only on a torch carrying pytorch [#181519]. The `frameworks/2026.1.0` module
ships torch `2.13.0a0+gitcf30153`, which predates that patch, and on it every
run dies at FSDP wrapping. `torch 2.14.0+xpu` -- a stable release -- works.

[#181519]: https://github.com/pytorch/pytorch/pull/181519

## The result

`12477656`, sunspot `workq`, 2 nodes / 24 ranks, `torch 2.14.0+xpu`:

```
agpt_debugmodel  TP=1   rc=0    step 1 loss 10.87743
                                step 2       10.75458
                                step 3       10.48057
agpt_debugmodel  TP=2   rc=0    step 1 loss 10.88015
moe_debugmodel          rc=143  RuntimeError: Cannot unflatten unevenly sharded tensor
```

TP=2 passing is the load-bearing part: that is the arm exercising the #4533
`local_map` contract rekey, which asserts only at TP>1.

### Losses match the pre-merge baseline

| arm | pre-merge, torch 2.13 (`8829243`) | merged, torch 2.14 (`12477656`) |
|---|---|---|
| agpt TP=1 step 1 | 10.88382 | 10.87743 |
| agpt TP=1 step 3 | 10.48337 | 10.48057 |
| agpt TP=2 step 1 | 10.88071 | 10.88015 |

Different torch versions, so not bitwise -- but ~3 decimals of agreement says
the merge did not move the math. Consistent with the Perlmutter numerics A/B
(argmax 100%, see `sync84-numerics-perlmutter.md`).

## The controlled A/B: only torch differs

Same tree, same script, same node shape, same deps.

| | `frameworks/2026.1.0` (`12477660`) | `torch 2.14.0+xpu` (`12477656`) |
|---|---|---|
| torch | `2.13.0a0+gitcf30153` | `2.14.0+xpu` |
| `_fsdp_param.py` | 1095 lines, #181519 ABSENT | 1329 lines, PRESENT |
| FSDP wrapping | `ValueError` on 69 ranks | `Applied FSDP to the model` |
| training | no steps | both dense arms rc=0 |

```
torch 2.13:  ValueError: When dp_mesh_dims is provided, all parameters must be
             DTensors on the full SPMD mesh ... Got plain tensor for parameter
torch 2.14:  trains
```

## Why the patch matters

`#181519` (`da19cbd78`, 2026-06-23) adds a block to `FSDPParam.__init__` that
converts an annotated plain tensor into a DTensor:

```python
self.is_spmd_types = (
    dist._is_spmd_types_available()
    and bool(spmd_local_type := spmd.get_local_type(param))
    and not isinstance(param, DTensor)
)
if self.is_spmd_types:
    param = self._resolve_spmd_types_for_storage(...)
```

It runs BEFORE the `is_spmd_mesh and not is_dtensor` check that raises. Without
it, an annotated plain tensor falls straight through to the raise.

ezpz used to dodge this by pinning the `partial_dtensor` backend. Upstream
#4419 deleted that backend AND its config field -- zero references remain in
core -- so there is no flag to set. A patched torch is the only route.

### Six builds checked, one has it

| build | `_fsdp_param.py` | #181519 |
|---|---|---|
| aurora `projects/saforem2/.venv` `2.13.0.dev20260520+xpu` | 1093 | absent |
| aurora `runs/agpt-2b-v2/.venv` `2.13.0.dev20260428+xpu` | 1017 | absent |
| aurora `frameworks/2026.1.0` `2.13.0a0+gitcf30153` | 1095 | absent |
| sunspot `frameworks/2026.1.0` `2.13.0a0+gitcf30153` | 1095 | absent |
| perlmutter `pytorch/2.13.0` `2.13.0+cu130` | 1095 | absent |
| polaris `.venv-torch213` `2.13.0+cu130` | 1095 | absent |
| **`torch 2.14.0+xpu`** (stable, PyTorch XPU index) | **1329** | **PRESENT** |

Detect with the SYMBOLS the patch introduces -- `_resolve_spmd_types_for_storage`,
`self.is_spmd_types`, `get_local_type` -- never with prose. An earlier probe
grepped for `"full SPMD"` / `"plain tensor"` and reported PRESENT on a build
that lacked the patch: those strings are the RAISE TEXT the patch makes
unreachable, so they are ANTI-correlated with the fix.

## Two real sync-84 bugs found getting here

Both are fixed on this branch.

1. **`global_valid_tokens` was a Python float.** `components/loss.py` passes it
   to `spmd.assert_type()`, which does `tensor.ndim`. Core keeps a tensor on
   both branches (`trainer.py:902-908`, `dist_sum_tensor`); our no-DP branch
   did `float(local_valid_tokens.item())`, mirroring PR #3586 which upstream
   has since reverted. NOT a `spmd_types` bug -- downgrading it would not help.
2. **The xccl shim declared 4 positional args.** torch 2.15 added a 5th
   (`preserve_rank_order`) to `DeviceMesh._init_one_process_group`. Now
   `*args/**kwargs` so it survives the next arity change.

## Everything else was environment

Not sync-84 issues, but each cost a job and will cost the next person one:

| symptom | cause |
|---|---|
| `importlib.metadata` missing | prod-BKC venv on a test-BKC node; use a `/home`-based venv |
| `Tokenizer path does not exist` | `assets/hf` is not in git; symlink from a prod clone |
| `HYD_arg_parse_array` error | `impi-rt` (a torch XPU dep) installs its own `mpiexec` into `venv/bin` |
| `PMIX_Init returned -25` | `mpi4py` built against the wheel's MPI; rebuild with `MPICC` on a compute node |
| `No module named 'blendcorpus'` | editable install from `deps/blendcorpus` in the prod clone |

See [[project_torch_xpu_wheel_ships_its_own_runtime]] -- a pip torch XPU wheel
brings its own oneAPI, MPI and launchers, all of which shadow the system stack.

## Open

- **The moe arm.** `RuntimeError: Cannot unflatten unevenly sharded tensor`.
  Reached only after FSDP and the dense arms pass. NOT yet attributable to the
  merge: the pre-merge baseline ran moe on torch 2.13, so that comparison is
  not controlled. The clean test is pre-merge moe on 2.14.
- **`spmd_types 0.2.5` vs post-2.13 torch.** Surfaced once as the float bug
  above; whether more remains is untested.
- **Landing.** Do NOT merge into `ezpz` yet: `runs/agpt-80b-v2` tracks it at 0
  commits behind, and the frameworks module still ships 2.13.

## Reproducing

```bash
# venv (sunspot)
uv venv --python 3.12 $VENV
uv pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/xpu
uv pip install spmd_types==0.2.5 renderers==0.1.11 grain tyro omegaconf \
  tensorboard sentencepiece einops torchao torch_checkpointing wandb \
  <repo>/deps/blendcorpus "ezpz @ git+https://github.com/saforem2/ezpz"
uv pip install --no-deps "torch_remat @ git+https://github.com/meta-pytorch/remat.git@d302699b"
mv $VENV/bin/mpiexec{,.disabled}          # and mpirun, mpiexec.hydra
# then, ON A COMPUTE NODE (mpicc does not exist on login):
MPICC=$(command -v mpicc) uv pip install --no-binary mpi4py mpi4py

# smoke
qsub torchtitan/experiments/ezpz/scripts/sync_smoke_sunspot.sh
```

---

*An earlier revision of this page was a running log written newest-first, with
five successive and mutually contradictory conclusions stacked on top of each
other. The full history is in `git log` for this file; this version states what
is true as of 2026-09-16.*
