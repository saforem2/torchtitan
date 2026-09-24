# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contracts for the frozen Aurora MoE source image."""

import importlib.util
import unittest
from pathlib import Path


FREEZER = Path(__file__).parents[1] / "submit/aurora/freeze_matched_dense_moe_2n_1100.py"
SPEC = importlib.util.spec_from_file_location("freeze_matched_dense_moe", FREEZER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
is_non_runtime_path = MODULE.is_non_runtime_path


class FreezeMetadataExclusionTest(unittest.TestCase):
    def test_agent_metadata_is_excluded_at_root_and_in_ezpz(self):
        excluded = (
            ".agents/skills/example/SKILL.md",
            ".claude/CLAUDE.md",
            "torchtitan/experiments/ezpz/.agents/skills/alcf-job-preflight/SKILL.md",
            "torchtitan/experiments/ezpz/.claude/CLAUDE.md",
        )

        for path in excluded:
            with self.subTest(path=path):
                self.assertTrue(is_non_runtime_path(path))

    def test_runtime_sources_are_not_excluded(self):
        included = (
            "torchtitan/experiments/ezpz/train.py",
            "torchtitan/experiments/ezpz/agents.py",
            "torchtitan/experiments/ezpz/scripts/run_lr_finder.sh",
        )

        for path in included:
            with self.subTest(path=path):
                self.assertFalse(is_non_runtime_path(path))


if __name__ == "__main__":
    unittest.main()
