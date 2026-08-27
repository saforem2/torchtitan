# XPU graphs cannot capture oneCCL collectives (2026-08-16)

> [!CAUTION]
> **WRONG, corrected 2026-08-26 (job `8785582`). Collectives ARE capturable.**
> The blocker is `CCL_OP_SYNC=1`, which `ezpz_setup_env` exports
> unconditionally (three sites in `ezpz-utils`), so every run that reached this
> conclusion had it set without anyone choosing it.
>
> Swept the same `probe_xpu_graph_collective.py` across all three settings:
>
> | `CCL_OP_SYNC` | `all_gather` inside capture |
> |---|---|
> | unset | **OK -- collectives ARE capturable** |
> | `0` | **OK -- collectives ARE capturable** |
> | `1` | FAILS: `wait method cannot be used for an event associated with a command graph` |
>
> `matmul` inside capture passes in all three, so capture itself was never in
> question.
>
> Consequences:
>
> - **The Intel ask at the bottom of this page should not be filed as written.**
>   It describes a limitation that does not exist; the async-collective path
>   captures fine.
> - Anything wanting graph capture must `unset CCL_OP_SYNC` after
>   `ezpz_setup_env`, the way `rl/xpu_overrides.py` and four RL scripts already
>   do -- they hit this from the other direction and worked around it without
>   the connection being made.
> - `CCL_OP_SYNC=1` is presumably load-bearing for determinism elsewhere
>   (`bitwise_sync_check.sh` sets it deliberately), so it should not simply be
>   dropped from `ezpz_setup_env`. Capture and this flag are mutually
>   exclusive; that is the real finding.
>
> The reasoning below is kept, but read it knowing the environment was
> confounded throughout.



> [!NOTE]
> **RESOLVED (2026-08-16): the collective framing is CORRECT after all.** The
> detour below is kept because the reasoning matters.
>
> The "1 rank fails too, so it cannot be collectives" objection was **wrong**:
> `apply_fsdp` is called unconditionally in
> `ezpz/agpt/parallelize.py:182`, so even at `nproc=1` the model is
> `fully_shard`-wrapped on a one-rank mesh and STILL issues collectives. There
> is no collective-free path through the trainer.
>
> Two separate bugs were tangled together, and both are now settled:
>
> | error | cause | status |
> |---|---|---|
> | `it->second->use_count > 0 INTERNAL ASSERT` | **mine** -- shared `graph_pool_handle` across captures | FIXED (`feac16acd`), each graph owns its pool |
> | `wait method cannot be used for an event associated with a command graph` | oneCCL collectives are not capturable | stands; this is the Intel ask |
>
> With fresh pools, capture handles everything **except** collectives
> (job 12473186): `nn.Linear` forward CAPTURES, raw forward+backward CAPTURES,
> `nn.Linear` forward+backward CAPTURES. So forward, backward, autograd and
> Modules are all fine -- the collective is the single remaining blocker, and
> the ticket at the bottom of this page is accurate as written.

<details>
<summary>Superseded intermediate reading (kept for the reasoning)</summary>

> **PARTIALLY SUPERSEDED (intermediate, later retracted).** The collective finding below
> is real and reproducible, but it is **not the only** capture blocker and it is
> **not** what stops the trainer. A **1-rank** run -- which issues no
> collectives at all -- fails too (job `12473181`), from inside
> `loss.backward()` -> the autograd engine, not from `all_gather_single`. And a
> bare `nn.Linear` **forward** fails to capture with a different error entirely
> (job `12473182`):
>
> ```
> forward matmul       CAPTURES OK
> nn.Linear forward    FAILS -> it->second->use_count > 0 INTERNAL ASSERT FAILED
> ```
>
> That is an INTERNAL ASSERT (a torch bug, not a documented restriction), and it
> may yet prove to be **my** fault -- the wrapper reuses one
> `graph_pool_handle` across captures, and `use_count > 0` reads like allocator
> refcounting. Job `12473184` is separating pool-reuse vs `nn.Module` vs
> backward.
>
> **GATE CLEARED 2026-08-20.** Job `12473184` returned (exit 0, 8s) and the
> suspicion was right -- the assert is OURS:
>
> ```
> matmul,        fresh pool   OK
> nn.Linear fwd, fresh pool   OK
> matmul,        shared pool  OK
> nn.Linear fwd, shared pool  FAILS -> use_count > 0 INTERNAL ASSERT
> ```
>
> Pool reuse is the trigger, and it is already fixed (one pool per graph,
> `xpu_graph.py:25`). So the `use_count` assert is excluded from the Intel
> ticket, and the collective finding -- which reproduces with fresh pools --
> is the whole of it. The ticket is filable.

</details>

> [!IMPORTANT]
> **XPU graph capture works for a bare matmul, oneCCL collectives work outside
> capture, but a collective inside a capture region fails.** Since every distributed config issues collectives
> inside the step (FSDP all-gather, DDP/FSDP all-reduce), XPU graphs are
> currently **unusable for any multi-rank training** -- which is all of our
> production work. Single-rank capture is fine.

## The three-way isolation

Same node, same 12 ranks, same buffers, one probe
([`tests/probe_xpu_graph_collective.py`](../../../tests/probe_xpu_graph_collective.py),
job `12473180`):

```
world=12 torch=2.13.0a0+gitcf30153
  all_gather OUTSIDE capture: OK
  all_gather INSIDE  capture: FAILS -> RuntimeError: wait method cannot be
                                       used for an event associated with a
                                       command graph.
  matmul     INSIDE  capture: OK
```

The third line is what makes this precise: capture itself is healthy on this
build, and the collective is healthy outside capture. Only the combination
fails, so this is not "XPU graphs are broken" and not "oneCCL is broken."

## How it surfaces in training

`agpt_2b` at 12 ranks with capture enabled dies at step 1 (job `12473179`,
rc=1, 1/12 steps, 0 graphs captured):

```
group.all_gather_single(output_tensor, input_tensor, opts)
RuntimeError: wait method cannot be used for an event associated with a
              command graph.
```

FSDP's parameter all-gather lands inside the captured region, so the very
first captured step raises.

## Why CUDA does not hit this

Upstream restricts CUDA graphs by *parallelism* -- `_validate_cuda_graphs`
(`torchtitan/trainer.py:179`) rejects pipeline parallelism and all but a couple
of expert-parallel token dispatchers -- but it never forbids FSDP's all-gather,
because **NCCL collectives are capturable** under CUDA graphs. The equivalent
oneCCL path is not.

That asymmetry is the ask for Intel: a documented PyTorch feature
(`torch.xpu.graph`, shipped and functional in this build) is unusable for
distributed workloads on XPU while its CUDA counterpart is not.

## Environment

- torch `2.13.0a0+gitcf30153` (RC4 wheelforge conda env)
- `frameworks/2026.1.0`, oneCCL `/opt/aurora/26.181.0/oneapi/ccl/latest`
- `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, XCCL backend, Sunspot
- `torch.xpu.{XPUGraph, graph, graph_pool_handle, Stream}` all present and
  functional -- `graph_pool_handle()` returns `(0, 1)`, and standalone
  capture+replay of a matmul succeeds (job `12473175`)

## Status of our implementation

[`experiments/ezpz/xpu_graph.py`](../../../xpu_graph.py) is committed and
correct as far as this limitation allows: it wraps the `fwd_bwd_fn`
indirection, is **off by default** (`EZPZ_XPU_GRAPHS=1` to enable), and
degrades to eager when the API is missing. It is dormant rather than removed --
if a future oneCCL supports capture, flipping the env var is the whole change.

Two implementation notes worth keeping:

- **Do not subclass core's `CUDAGraphWrapper`.** Its `__init__` calls
  `_manager.maybe_initialize()` (`cudagraph.py:243` -> `:148`), which
  unconditionally calls `torch.cuda.graph_pool_handle()`; on an XPU build that
  is a dummy stub and construction dies before any override runs. Ours is
  standalone for that reason.
- **`torch.xpu.graph` takes no `capture_error_mode`** (CUDA's does; core passes
  `"thread_local"`). Signature is `(xpu_graph, pool=None, stream=None)`.

## Performance: unmeasured, and probably moot

No throughput number can be quoted -- the graphs-ON arm has never completed a
step. For reference the eager baseline at this config is **~3,626 tps**
(two runs, 3627/3626).

Even if capture worked, the upside for us is unclear: graphs mainly remove
launch overhead, which matters most for small kernels at small scale, whereas
our steps are large and communication-bound at production node counts. Any
future re-test should measure rather than assume -- but it is not worth
scheduling until collectives are capturable.

## Draft Intel ticket

> [!NOTE]
> **Filable version lives at**
> [`upstream-issues/intel-xpu-graphs-cannot-capture-oneccl.md`](../../upstream-issues/intel-xpu-graphs-cannot-capture-oneccl.md).
> That is the copy to send; the text below is the original draft.

> **Subject:** XPU graphs cannot capture oneCCL collectives (torch
> 2.13.0a0+gitcf30153, oneAPI 2026.1.0)
>
> On Sunspot, `torch.xpu.graph` capture works and oneCCL collectives work, but
> a collective issued *inside* a capture region raises:
>
> ```
> RuntimeError: wait method cannot be used for an event associated with a
>               command graph.
> ```
>
> Minimal reproduction (12 ranks, one node, ~20s), attached as
> `probe_xpu_graph_collective.py`:
>
> ```
> all_gather OUTSIDE capture: OK
> all_gather INSIDE  capture: FAILS (above)
> matmul     INSIDE  capture: OK
> ```
>
> The matmul line rules out a general capture problem, and the OUTSIDE line
> rules out a general collective problem.
>
> Impact: this makes XPU graphs unusable for distributed training, since FSDP
> all-gather and DDP/FSDP all-reduce both run inside the step. In a real
> torchtitan run the first captured step fails in
> `ProcessGroup.all_gather_single`. The CUDA equivalent works -- NCCL
> collectives are capturable under CUDA graphs -- so this is an XPU-specific
> gap in an otherwise-shipped feature.
>
> Environment: torch `2.13.0a0+gitcf30153`, `frameworks/2026.1.0`, oneCCL
> `/opt/aurora/26.181.0/oneapi/ccl/latest`, XCCL backend,
> `ZE_FLAT_DEVICE_HIERARCHY=FLAT`.
