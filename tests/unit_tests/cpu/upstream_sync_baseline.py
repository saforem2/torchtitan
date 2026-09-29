# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic signatures used by upstream-sync regression tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


def _tensor_signature(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().to(device="cpu", dtype=torch.float64).contiguous()
    raw = bytes(value.reshape(-1).view(torch.uint8).tolist())
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sum": value.sum().item(),
        "l2": value.square().sum().sqrt().item(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def model_state_signature(model: nn.Module) -> dict[str, Any]:
    """Record model layout and values without depending on object identities."""
    state = model.state_dict()
    return {
        "keys": list(state),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "tensors": {
            name: (
                {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                if tensor.is_meta
                else _tensor_signature(tensor)
            )
            for name, tensor in state.items()
        },
    }


def training_step_signature(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, Any]:
    """Run one step and record output, loss, gradient, and optimizer signatures."""
    optimizer.zero_grad(set_to_none=True)
    output = model(inputs)
    loss = torch.nn.functional.cross_entropy(output, targets)
    loss.backward()
    gradients = {
        name: _tensor_signature(parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    optimizer.step()
    optimizer_state = {
        str(index): {
            key: _tensor_signature(value) if isinstance(value, torch.Tensor) else value
            for key, value in sorted(state.items())
        }
        for index, state in enumerate(optimizer.state.values())
    }
    return {
        "output": _tensor_signature(output),
        "loss": _tensor_signature(loss),
        "gradients": gradients,
        "optimizer": optimizer_state,
        "model_after_step": model_state_signature(model),
    }


def signature_digest(signature: Mapping[str, Any]) -> str:
    """Return a stable digest suitable for storing as a compact baseline."""
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
