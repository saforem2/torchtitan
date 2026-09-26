# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for static trajectory metadata."""

from torchtitan.experiments.ezpz.utils.trajectories import by_key, SEQ_LEN


def test_stage2_n256_includes_inherited_stage1_tokens():
    trajectory = by_key("2b_v2_256_stage2_dolmino")

    assert trajectory is not None
    assert trajectory["prior_tokens"] == 92_859 * 6144 * SEQ_LEN
