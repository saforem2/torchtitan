# XPU graphs cannot capture oneCCL collectives (2026-08-16)

> [!IMPORTANT]
> **XPU graph capture works, oneCCL collectives work, but a collective inside
> a capture region fails.** Since every distributed config issues collectives
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
