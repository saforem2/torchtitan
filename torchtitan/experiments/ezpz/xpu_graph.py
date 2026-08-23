"""XPU graph capture/replay for the ezpz trainer (opt-in).

Core's `torchtitan/distributed/cudagraph.py` implements graph capture but
hard-gates on `utils.device_type == "cuda"` and reaches for `torch.cuda.*`
throughout, so on XPU `wrap_with_cuda_graph` logs a warning and returns the
function unwrapped. Core is off-limits for experiment needs, so this module
provides the XPU twin *here*.

The frameworks RC (oneAPI 2026.1.0, torch 2.13.0a0+gitcf30153) exposes a full
mirror of the CUDA graph surface -- verified on-device:

    torch.xpu.XPUGraph  graph  graph_pool_handle  Stream
    make_graphed_callables  is_current_stream_capturing

This is a STANDALONE reimplementation, not a subclass of core's
`CUDAGraphWrapper`. Subclassing was tried first and cannot work: core's
`__init__` calls `_manager.maybe_initialize()`, which unconditionally calls
`torch.cuda.graph_pool_handle()` -- a dummy stub on an XPU build -- so
construction dies before any override runs (job 12473176).

Two XPU-specific differences from the CUDA path, both found the hard way:

1. `torch.xpu.graph` has **no `capture_error_mode`** (CUDA's does; core passes
   `"thread_local"`). Signature: `(xpu_graph, pool=None, stream=None)`.
2. **Each graph gets its OWN memory pool.** Core shares one pool across
   wrappers; on XPU that trips `it->second->use_count > 0 INTERNAL ASSERT
   FAILED` as soon as the captured region owns persistent parameters
   (job 12473184: nn.Linear captures with a fresh pool, fails with a shared
   one).

Enable with `EZPZ_XPU_GRAPHS=1`. Off by default: graph capture requires static
shapes and stable input addresses, and a violation is silently wrong rather
than loud, so this must be opted into and numerically verified per config
before it is trusted. See `tests/test_xpu_graph.py`.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Sequence

import torch

logger = logging.getLogger(__name__)


def xpu_graphs_available() -> bool:
    """Whether the XPU graph API is present AND functional.

    Deliberately CALLS `graph_pool_handle()` rather than testing `hasattr`.
    torch installs dummy placeholder classes that satisfy `hasattr` and only
    raise "Tried to instantiate dummy base class" when invoked -- so a
    presence check reports a working API on builds that have none. That false
    positive is what let job 12473173 reach a runtime failure instead of
    falling back cleanly.
    """
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return False
    for attr in ("XPUGraph", "graph", "graph_pool_handle", "Stream"):
        if not hasattr(torch.xpu, attr):
            return False
    try:
        torch.xpu.graph_pool_handle()
    except Exception as e:  # dummy stub, or a build without real support
        logger.warning("XPU graph API present but not functional: %s", e)
        return False
    return True


def xpu_graphs_enabled() -> bool:
    """Opt-in gate. Off unless EZPZ_XPU_GRAPHS=1."""
    return os.environ.get("EZPZ_XPU_GRAPHS", "0") == "1"


class _XPUGraphManager:
    """Owns the shared capture stream + memory pool.

    Mirrors core's `_CUDAGraphManager`. One pool shared across wrappers so
    captured graphs can share memory, exactly as the CUDA path does.
    """

    def __init__(self) -> None:
        self._stream: Any = None
        self._graph_pool: Any = None

    @property
    def stream(self) -> Any:
        if self._stream is None:
            self._stream = torch.xpu.Stream()
        return self._stream

    # NOTE: no graph_pool property. Sharing one pool across captures is what
    # broke module capture on XPU (job 12473184); each graph now allocates its
    # own. Kept out entirely so it cannot be reintroduced by habit.

    def reset(self) -> None:
        self._stream = None
        self._graph_pool = None


_manager = _XPUGraphManager()


class XPUGraphWrapper:
    """Capture/replay a callable with an XPU graph.

    STANDALONE, not a subclass of core's CUDAGraphWrapper. Subclassing was the
    first attempt and it does not work: core's `__init__` calls
    `_manager.maybe_initialize()` (cudagraph.py:243 -> :148), which
    unconditionally calls `torch.cuda.graph_pool_handle()`. On an XPU-only
    build that is a dummy stub, so construction dies before any override can
    take effect:

        cudagraph.py:148 maybe_initialize
          torch/cuda/graphs.py:74 graph_pool_handle
            RuntimeError: Tried to instantiate dummy base class
                          _graph_pool_handle

    (Confirmed job 12473176. `torch.xpu.graph_pool_handle()` itself is fine --
    it returns (0, 1); the problem is core reaching for the CUDA one.)

    The capture/replay logic below mirrors core's so the semantics match:
    one eager warmup call, then capture, then replay with inputs copied into
    the captured buffers.

    Like core's, the returned output ALIASES graph-owned storage and is
    overwritten by the next replay -- callers must clone anything they keep.
    """

    def __init__(
        self,
        fn: Callable,
        example_inputs: Sequence[Any],
        static_input_indices: tuple[int, ...] | None = None,
    ) -> None:
        self._fn = fn
        self._static_input_indices = set(static_input_indices or ())
        # Every non-static tensor input must be copied into the captured
        # buffer before each replay; non-tensors are frozen at capture.
        self._input_indices_to_copy = [
            i
            for i, inp in enumerate(example_inputs)
            if isinstance(inp, torch.Tensor) and i not in self._static_input_indices
        ]
        self._non_tensor_inputs = {
            i: inp
            for i, inp in enumerate(example_inputs)
            if not isinstance(inp, torch.Tensor)
        }
        self._graph: Any = None
        self._warmup_remaining = 1
        self._args: tuple | None = None
        self._output: Any = None

    def _validate_inputs(self, args: tuple) -> None:
        """Non-tensor inputs are baked into the graph; they must not change."""
        for i, expected in self._non_tensor_inputs.items():
            actual = args[i]
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(
                    "XPU graph non-tensor inputs must remain constant, but "
                    f"input {i} changed from {expected!r} to {actual!r}"
                )

    def __call__(self, *args):
        self._validate_inputs(args)

        # Warmup on the side stream, same as core: the first call runs eagerly
        # so allocator state settles before capture.
        if self._warmup_remaining > 0:
            self._warmup_remaining -= 1
            current_stream = torch.xpu.current_stream()
            _manager.stream.wait_stream(current_stream)
            with torch.xpu.stream(_manager.stream):
                output = self._fn(*args)
            current_stream.wait_stream(_manager.stream)
            return output

        if self._graph is None:
            self._args = args
            self._graph = torch.xpu.XPUGraph()
            # NOTE: no capture_error_mode -- torch.xpu.graph does not accept it
            # (CUDA's does; core passes "thread_local"). Verified against the
            # RC signature: (xpu_graph, pool=None, stream=None).
            # NO SHARED POOL. Core's CUDA manager hands every wrapper one
            # graph_pool_handle and that is safe on CUDA; copying it here was
            # MY bug. On XPU, capturing a module that owns persistent
            # parameters into a SHARED pool trips an allocator refcount assert:
            #     it->second->use_count > 0 INTERNAL ASSERT FAILED
            # Isolated in job 12473184, which is unambiguous:
            #     matmul    + fresh pool  OK     matmul    + shared pool  OK
            #     nn.Linear + fresh pool  OK     nn.Linear + shared pool  FAILS
            # Only shared-pool + module breaks. Omitting `pool` gives this
            # graph its own. The cost is that graphs cannot share memory with
            # each other -- irrelevant here, since we capture exactly one
            # region per wrapper and have nothing to share with.
            with torch.xpu.graph(self._graph, stream=_manager.stream):
                self._output = self._fn(*args)
            logger.info("Recorded XPU graph")

        assert self._args is not None
        for i in self._input_indices_to_copy:
            self._args[i].copy_(args[i])
        self._graph.replay()
        return self._output

    def teardown(self) -> None:
        self._graph = None
        self._args = None
        self._output = None
        self._non_tensor_inputs.clear()


def make_xpu_graph_wrapper(
    fn: Callable,
    example_inputs: Sequence[Any],
    static_input_indices: tuple[int, ...] | None = None,
) -> XPUGraphWrapper:
    return XPUGraphWrapper(fn, example_inputs, static_input_indices)


def maybe_wrap_with_xpu_graph(fwd_bwd_fn: Callable) -> Callable:
    """Wrap `fwd_bwd_fn` with XPU graph capture, or return it unchanged.

    Returns the input untouched -- with a reason logged -- when graphs are not
    opted into, the API is missing, or the device is not XPU. Mirrors core's
    `wrap_with_cuda_graph` contract: capture is deferred to the first call,
    since the example inputs are not known until then.
    """
    if not xpu_graphs_enabled():
        return fwd_bwd_fn
    if not xpu_graphs_available():
        logger.warning(
            "EZPZ_XPU_GRAPHS=1 but this torch build has no XPU graph API "
            "(need torch.xpu.{XPUGraph,graph,graph_pool_handle,Stream}); "
            "using eager execution."
        )
        return fwd_bwd_fn
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        logger.warning(
            "EZPZ_XPU_GRAPHS=1 but no XPU device is available; "
            "using eager execution."
        )
        return fwd_bwd_fn

    wrapper: Any = None

    def graphed(*args):
        nonlocal wrapper
        if wrapper is None:
            # Static input indices are unknown here (core derives them from the
            # trainer's own bookkeeping), so treat every tensor as copy-in.
            # Correct but slightly slower than the CUDA path's static-weight
            # optimization; revisit once this is proven numerically.
            wrapper = make_xpu_graph_wrapper(fwd_bwd_fn, args)
        return wrapper(*args)

    logger.info("XPU graph capture ENABLED (EZPZ_XPU_GRAPHS=1)")
    return graphed


def xpu_graph_teardown() -> None:
    """Release the shared stream + pool. Safe to call when graphs were unused."""
    _manager.reset()
