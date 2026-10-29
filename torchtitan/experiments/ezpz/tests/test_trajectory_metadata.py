#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for production trajectory metadata."""

from torchtitan.experiments.ezpz.utils.trajectories import by_key


def test_stage2_chains_include_inherited_tokens():
    stage2_512 = by_key("2b_v2_512_stage2_dolmino")
    stage2_256 = by_key("2b_v2_256_stage2_dolmino")

    assert stage2_512["prior_tokens"] > 0
    assert stage2_256["prior_tokens"] == 92_859 * 6_144 * 8_192
