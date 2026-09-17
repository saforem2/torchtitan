# Sync 84 on XPU: it trains, on torch 2.14

**Bottom line:** the sync-84 merge trains on Sunspot (XPU) and Polaris (CUDA)
-- all three smoke arms, moe included -- but only on a torch carrying pytorch
[#181519]. Aurora is in flight; Perlmutter has a complete venv but no tokenized
corpus on the machine, so it contributed the numerics A/B rather than a
training run. See "Where this has actually run". The `frameworks/2026.1.0` module
ships torch `2.13.0a0+gitcf30153`, which predates that patch, and on it every
run dies at FSDP wrapping. `torch 2.14.0+xpu` -- a stable release -- works.

[#181519]: https://github.com/pytorch/pytorch/pull/181519

## The result

`12477670`, sunspot `workq`, 2 nodes / 24 ranks, `torch 2.14.0+xpu`.
All three arms pass, zero non-finite loss or grad_norm:

```
VERDICT: ok
agpt_debugmodel  TP=1   rc=0    step 1 loss 10.87743  step 2 10.75458  step 3 10.48057
agpt_debugmodel  TP=2   rc=0    step 1 loss 10.88015  step 2 10.70795  step 3 10.55879
moe_debugmodel          rc=0    step 1 loss 12.95236  step 2 12.59633  step 3 11.47142
```

TP=2 passing is the load-bearing part: that is the arm exercising the #4533
`local_map` contract rekey, which asserts only at TP>1.

The moe arm took three fixes past the torch floor, all in
`experiments/ezpz/moe/`, each hiding the next:

1. **Fused gate-up init under FSDP** (#4535/#4526 interleaved `w13`).
   `Cannot unflatten unevenly sharded tensor`. The per-expert rows are
   gate-on-even / up-on-odd, and shards are 43/42 rows -- odd -- so a shard
   can begin on either parity. Fixed by computing the shard's global row
   offset from its mesh placement and striping the two initializers from
   that parity, rather than assuming every shard starts on a gate row.
   A first attempt that assumed even-aligned shards would have silently
   applied the wrong initializer instead of raising.
2. **`padding_mask` in the block contract** (#4594). `Decoder.forward` now
   passes it by keyword to every layer; `moeTransformerBlock.forward` did
   not accept it. Consume and discard, as llama3 does.
3. That fix, first time, landed on `Attention.forward` instead of the block
   -- same file, wrong class -- so job `12477669` failed identically.
   Verifying by line number rather than by enclosing class is what missed it.

An audit for instance 2 across the tree then found the same defect in
`AgptFp32ResidualBlock` and `AgptFp32ResidualDepthBlock`, which override
`forward` with the pre-#4594 signature. Neither is in this smoke, so it
would have surfaced later as a fresh mystery in an 80B fp32-residual run.
Fixed in `a498c608`; all three block classes in `experiments/ezpz` now
accept it.

### Losses match the pre-merge baseline

| arm | pre-merge, torch 2.13 (`8829243`) | merged, torch 2.14 (`12477656`) |
|---|---|---|
| agpt TP=1 step 1 | 10.88382 | 10.87743 |
| agpt TP=1 step 3 | 10.48337 | 10.48057 |
| agpt TP=2 step 1 | 10.88071 | 10.88015 |

Different torch versions, so not bitwise -- but ~3 decimals of agreement says
the merge did not move the math. Consistent with the Perlmutter numerics A/B
(argmax 100%, see `sync84-numerics-perlmutter.md`).

## moe converges

`12477675`, same node shape and venv, moe only, 200 steps:

```
step   1  loss 12.93973  grad_norm 0.8421
step  40  loss 10.41446  grad_norm 1.2182
step  80  loss  7.53617  grad_norm 0.6193
step 120  loss  6.72880  grad_norm 0.4264
step 160  loss  6.42788  grad_norm 0.3793
step 200  loss  6.09904  grad_norm 0.4218
```

Zero non-finite loss or grad_norm. 66 of 199 steps tick the loss up, which
is ordinary minibatch noise on a debug model, and the trend is a clean
descent well below the ~10.4 uniform-prediction floor for this vocab -- so
the model is learning, not just settling onto the token prior.

Four steps exceed grad_norm 2 (15, 16, 58, 100), the largest 13.07 at step
58. All are single-step transients: loss descends straight through the big
one (9.23408 -> 9.15392 -> 9.00414) and grad_norm is back to 1.04 the next
step. No persistent high-gradient regime, unlike the SophiaG 30B failure
mode.

This is what closes the fused gate-up init question. Three steps proved
construction and one forward/backward; 200 steps exercise the striped `w13`
initializer against real optimizer updates, where a wrong gate/up parity
would show up as a model that fails to descend rather than as a raise.

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

The table below is a torch INVENTORY -- which build each machine ships and
whether it carries the patch. A row here is NOT a claim that sync 84 ran on
that machine. Only Sunspot has run it; see "Where this has actually run".

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

The root cause of three of those: **a pip torch XPU wheel is not a library, it
is a parallel runtime.** `uv pip install torch --index-url .../whl/xpu` also
pulls `intel-sycl-rt`, `dpcpp-cpp-rt`, `onemkl-sycl-*` and `impi-rt`, and the
last of those puts `mpiexec`, `mpirun`, `mpiexec.hydra` and the hydra proxies
into `venv/bin`. Activating the venv shadows the system MPI. Check with
`command -v mpiexec` inside vs outside the activated venv.

## Where this has actually run

| machine | what ran | what it proves |
|---|---|---|
| **sunspot** (XPU) | `12477670` 3/3 arms; `12477675` moe 200 steps | dense TP=1/TP=2 + moe train on XPU |
| **polaris** (CUDA) | `7629879` 3/3 arms, `VERDICT: ok` | the merge is not XPU-specific: same three arms train on A100/CUDA |
| **perlmutter** (CUDA) | numerics A/B, A100 login node | fused QKV/gate-up do not move the math (argmax 100%) |
| **aurora** (XPU) | `8834294` 3/3 arms, `VERDICT: ok` | production XPU machine; losses BIT-IDENTICAL to sunspot |

### Aurora 3/3 (the production XPU machine)

`8834294`, `next-eval`, 2 nodes / 24 ranks, `torch 2.15.0.dev20260915+xpu`
(carries #181519 at 1379 lines). Zero non-finite:

```
agpt_debugmodel TP=1  rc=0  10.87743 -> 10.75458 -> 10.48057
agpt_debugmodel TP=2  rc=0  10.88015 -> 10.70795 -> 10.55879
moe_debugmodel        rc=0  12.95236 -> 12.59633 -> 11.47142
```

All nine values are **bit-identical to the Sunspot run** (`12477670`). Same
backend, same deterministic seed, same torch floor -- so identical is the
correct outcome here, and any drift would have been the finding. Note this is
torch 2.15 nightly against Sunspot's 2.14 stable: both carry #181519, and the
patch is what matters, not the version.

Getting here needed `mpi4py` compiled on a compute node against the test-BKC
MPICH (`/opt/aurora/26.181.0/.../mpich-5.0.0.aurora_test`) -- a PyPI wheel
binds the wrong MPI and dies `PMIX_Init returned -25` -- plus `CXX=/usr/bin/g++`
for blendcorpus, which builds through scikit-build-core and otherwise fails
with `CMAKE_CXX_COMPILER not set`.

### Polaris 3/3 (CUDA control)

`7629879`, 2 nodes / 8 ranks, hand-built `torch 2.14.0+cu130` (every preexisting
Polaris venv is 2.10/2.13 and predates #181519). Zero non-finite:

```
agpt_debugmodel TP=1  rc=0  10.85358 -> 10.76995 -> 10.55864
agpt_debugmodel TP=2  rc=0  10.85318 -> 10.73931 -> 10.60787
moe_debugmodel        rc=0  12.98146 -> 12.62001 -> 11.63856
```

Losses land within ~0.01-0.16 of the Sunspot XPU run at the same steps
(10.87743/10.88015/12.95236 at step 1), which is the agreement you would expect
across different hardware, math libraries and RNG -- not bitwise, but nowhere
near a divergence. **This is the load-bearing cross-backend result: the sync-84
merge is not XPU-specific.**

Four environment bugs stood between the venv and that result, none of them in
the merge:

1. `mpiexec` is absent from a PBS job's PATH (cray-pals module) -- every arm
   `rc=127`.
2. PALS stages the interpreter into `/var/run/palsd/.../files/0/python` without
   its `libpython3.12.so.1.0`; `--no-transfer` fixes it.
3. **Exporting the `mpi-compat` dir into the job shell aborts every coreutil**
   with `*** stack smashing detected ***` -- `mkdir`, `whoami`, `head`, `sleep`.
   That is what a "failed to create LOGDIR" actually was. Pass it to the ranks
   via `mpiexec --env`, never into the shell.
4. `libfabric` is its own module; a login shell has it, a PBS job does not.

Aurora is the significant gap: it is the production machine, and while it
shares Sunspot's Intel XPU lineage, "same vendor, same BKC family" is an
inference, not evidence -- the BKC images differ per queue and venvs are
BKC-bound. Polaris is CUDA, so it exercises a genuinely different backend
and its `.venv-torch213` lacks #181519, meaning it needs a newer torch
before it can run this at all.

## Open

- **moe convergence at production scale.** Cleared at debug scale: see
  "moe converges" below. Untested at real model size.
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
