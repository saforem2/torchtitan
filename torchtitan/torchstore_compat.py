# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small, dependency-light helpers for configuring TorchStore."""

import os
from importlib import import_module


def torchstore_transport_from_env():
    """Return a forced TorchStore transport, or ``None`` for auto selection."""
    requested = os.environ.get("TORCHTITAN_TORCHSTORE_TRANSPORT", "auto").lower()
    if requested in ("", "auto", "unset"):
        return None

    names = {
        "gloo": "Gloo",
        "xccl": "XCCL",
        "shared_memory": "SharedMemory",
        "monarch_rpc": "MonarchRPC",
    }
    if requested not in names:
        raise ValueError(
            "TORCHTITAN_TORCHSTORE_TRANSPORT must be one of "
            f"auto, gloo, xccl, shared_memory, monarch_rpc; got {requested!r}"
        )

    # Validate before importing so bad configuration is diagnosed even in a
    # dependency-light environment. TorchStore does not re-export this enum.
    transport_type = import_module("torchstore.transport").TransportType
    return getattr(transport_type, names[requested])