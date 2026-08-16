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

Only the *device namespace* differs, not the capture/replay algorithm, so this
subclasses core's `CUDAGraphWrapper` and overrides the handful of methods that
touch `torch.cuda`. Everything else -- input validation, static-address
checking, the copy-in list, teardown -- is inherited, so upstream fixes to that
logic reach XPU for free.

One genuine API difference: `torch.xpu.graph` has **no `capture_error_mode`**
parameter (CUDA's does; core passes `capture_error_mode="thread_local"`). It is
simply omitted here.

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
    """Whether this torch build exposes the XPU graph API we need."""
    return all(
        hasattr(torch, "xpu") and hasattr(torch.xpu, attr)
        for attr in ("XPUGraph", "graph", "graph_pool_handle", "Stream")
    )


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

    @property
    def graph_pool(self) -> Any:
        if self._graph_pool is None:
            self._graph_pool = torch.xpu.graph_pool_handle()
        return self._graph_pool

    def reset(self) -> None:
        self._stream = None
        self._graph_pool = None


_manager = _XPUGraphManager()


def _base_wrapper_cls():
    """Core's CUDAGraphWrapper, imported lazily.

    Lazy because importing `torchtitan.distributed.cudagraph` at module scope
    would drag CUDA-oriented imports into every ezpz run, including ones that
    never touch graphs.
    """
    from torchtitan.distributed.cudagraph import CUDAGraphWrapper

    return CUDAGraphWrapper


def make_xpu_graph_wrapper(
    fn: Callable,
    example_inputs: Sequence[Any],
    static_input_indices: tuple[int, ...] | None = None,
    should_check_address: bool = False,
):
    """Build an XPU-backed graph wrapper by subclassing core's CUDA one.

    Constructed dynamically so the base class is resolved at call time (see
    `_base_wrapper_cls`).
    """
    base = _base_wrapper_cls()

    class XPUGraphWrapper(base):  # type: ignore[misc, valid-type]
        """CUDAGraphWrapper with the torch.cuda calls swapped for torch.xpu.

        Overrides only `__call__` -- the one method that names a device
        namespace. Capture/replay semantics, and therefore the requirement that
        the loss aliases graph-owned storage, are unchanged from core.
        """

        def __call__(self, *args):
            self._validate_inputs(args)

            # Warmup on the side stream, same as core: the first call runs
            # eagerly so allocator state settles before capture.
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
                self._record_static_input_addresses(args)
                self._graph = torch.xpu.XPUGraph()
                # NOTE: no capture_error_mode -- torch.xpu.graph does not take
                # it (CUDA's does; core passes "thread_local"). Verified
                # against the RC's signature:
                #   (xpu_graph, pool=None, stream=None)
                with torch.xpu.graph(
                    self._graph,
                    pool=_manager.graph_pool,
                    stream=_manager.stream,
                ):
                    self._output = self._fn(*args)
                logger.info("Recorded XPU graph")

            if self._should_check_address:
                self._check_static_input_addresses(args)

            assert self._args is not None
            assert self._graph is not None
            for i in self._input_indices_to_copy:
                self._args[i].copy_(args[i])
            self._graph.replay()
            return self._output

    return XPUGraphWrapper(
        fn,
        example_inputs,
        static_input_indices=static_input_indices,
        should_check_address=should_check_address,
    )


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
