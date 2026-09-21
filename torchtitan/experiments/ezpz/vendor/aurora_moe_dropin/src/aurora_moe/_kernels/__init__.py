# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Experimental exact-routing SYCL kernels for Aurora MoE layers.

The package is intentionally opt-in while it is developed.  The existing
``MOE.py`` implementation remains the correctness oracle.
"""

from .ops import load_route_ops
from .swiglu_ops import load_swiglu_ops
from .tla_ops import load_tla_ops

__all__ = ["load_route_ops", "load_swiglu_ops", "load_tla_ops"]
