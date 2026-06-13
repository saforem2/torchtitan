# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ezpz-local wrapper for ``torchtitan.distributed.activation_checkpoint.apply_ac``.

Drops ``torch.ops._c10d_functional.all_to_all_single.default`` from the SAC
``MUST_SAVE`` op list before delegating to upstream.

Background: PR14's ``_normal_equal_a2a_padding`` fast-path in
``token_dispatcher.py:dispatch`` pads tensors before the EP all-to-all and
slices the pad rows off after. When ``apply_ac(mode="selective")`` includes
``all_to_all_single`` in the save list, SAC captures the comm output at the
dispatch boundary (an ``AsyncCollectiveTensor`` wrapping a buffer that the
caching allocator can reuse before backward). On step-1 backward recompute,
the saved wrapper's underlying GPU memory has been freed and reissued for a
new allocation -> GPU page table inconsistency that surfaces as a
``Segmentation fault from GPU at 0x... type: 0 (NotPresent), level: 1 (PDE)``
on the next forward op (typically step-2's ``_local_reorder``).

Dropping the op from the save list forces recompute of the a2a in backward
(extra comm) but avoids the dangling save. Measured on the affected config
(``moe_10b_2b_sdpa_ep`` EP=12 padding=1 AC=selective): 10 clean steps with
loss matching the upstream-dispatcher baseline within 1e-4 nats.

The standard (non-padded) MoE path is fine with the upstream save list, so
keeping the override narrow to ezpz/moe rather than touching upstream.
"""

from contextlib import contextmanager

import torch

from torchtitan.distributed import activation_checkpoint as _upstream_ac


_A2A_OP = torch.ops._c10d_functional.all_to_all_single.default

# Bind upstream's original at import time so the patch below can call it
# without re-entering itself. (Patching `_upstream_ac._get_save_ops` and then
# having the replacement try to call `_upstream_ac._get_save_ops()` would
# recurse forever.)
_ORIGINAL_GET_SAVE_OPS = _upstream_ac._get_save_ops


def _get_save_ops_for_moe() -> set:
    """Return upstream save_ops minus ``all_to_all_single``."""
    save_ops = _ORIGINAL_GET_SAVE_OPS()
    save_ops.discard(_A2A_OP)
    return save_ops


@contextmanager
def _patch_get_save_ops():
    """Temporarily replace upstream's ``_get_save_ops`` with the MoE variant.

    ``_apply_op_sac`` reads ``_get_save_ops`` via module attribute lookup, so
    swapping the module-level symbol is sufficient to intercept it.
    """
    _upstream_ac._get_save_ops = _get_save_ops_for_moe
    try:
        yield
    finally:
        _upstream_ac._get_save_ops = _ORIGINAL_GET_SAVE_OPS


def apply_ac(*args, **kwargs):
    """Apply AC to the MoE model with the ezpz-local save-list override."""
    with _patch_get_save_ops():
        return _upstream_ac.apply_ac(*args, **kwargs)
