# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.experiments.ezpz.grain_checkpoint import (
    _optional_hf_previous_state_missing_keys,
    GrainStreamingCheckpointManager,
)
from torchtitan.experiments.ezpz.train import _translate_legacy_args


class _State(Stateful):
    def __init__(self, state):
        self.state = state

    def state_dict(self):
        return self.state

    def load_state_dict(self, state_dict):
        self.state = state_dict


class TestGrainStreamingCheckpointCompatibility(unittest.TestCase):
    def test_allows_only_nullable_dataloader_previous_state(self):
        states = {
            "train_state": _State({"step": 600}),
            "dataloader": _State(
                {
                    "version": 1,
                    "dp_rank_0": {
                        "hf": {
                            "examples_iterable": {
                                "previous_state": None,
                                "num_chunks_since_previous_state": 0,
                            }
                        }
                    },
                }
            ),
        }
        checkpoint_keys = {
            "train_state.step",
            "dataloader.version",
            "dataloader.dp_rank_0.hf.examples_iterable.num_chunks_since_previous_state",
        }

        self.assertEqual(
            _optional_hf_previous_state_missing_keys(states, checkpoint_keys),
            {"dataloader.dp_rank_0.hf.examples_iterable.previous_state"},
        )

    def test_rejects_missing_required_training_state(self):
        states = {
            "train_state": _State({"step": 600}),
            "dataloader": _State(
                {"dp_rank_0": {"hf": {"examples_iterable": {"previous_state": None}}}}
            ),
        }

        with self.assertRaisesRegex(RuntimeError, "train_state.step"):
            _optional_hf_previous_state_missing_keys(states, set())

    def test_rejects_non_null_previous_state(self):
        states = {
            "dataloader": _State(
                {
                    "dp_rank_0": {
                        "hf": {"examples_iterable": {"previous_state": {"shard": 4}}}
                    }
                }
            )
        }

        with self.assertRaisesRegex(RuntimeError, "previous_state"):
            _optional_hf_previous_state_missing_keys(states, set())

    @mock.patch(
        "torchtitan.experiments.ezpz.agpt.config_registry.Path.is_file",
        return_value=True,
    )
    def test_mds_streaming_recipe_uses_compatible_manager(self, _is_file):
        from torchtitan.experiments.ezpz.agpt.config_registry import (
            agpt_2b_mds154391_tulu_math_uc_streaming,
        )

        cfg = agpt_2b_mds154391_tulu_math_uc_streaming()
        self.assertIsInstance(cfg.checkpointer, GrainStreamingCheckpointManager.Config)
        self.assertIs(cfg.checkpoint, cfg.checkpointer)
        self.assertEqual(
            cfg.checkpointer.folder,
            "checkpoints/agpt2b-mds154391-tulu-math-uc-streaming",
        )
        self.assertTrue(cfg.checkpointer.initial_load_path.endswith("/step-0"))

        cfg.__post_init__()
        self.assertIsInstance(cfg.checkpointer, GrainStreamingCheckpointManager.Config)
        self.assertEqual(
            cfg.checkpointer.folder,
            "checkpoints/agpt2b-mds154391-tulu-math-uc-streaming",
        )

    def test_legacy_checkpoint_options_translate_to_canonical_namespace(self):
        self.assertEqual(
            _translate_legacy_args(
                ["--checkpoint.folder=/tmp/ckpt", "--checkpoint.interval", "7"]
            ),
            ["--checkpointer.folder", "/tmp/ckpt", "--checkpointer.interval", "7"],
        )

    def test_stage1_launchers_target_recipe_checkpoint_series(self):
        scripts = Path(__file__).parents[1] / "rl" / "scripts" / "sft"
        expected = (
            '--checkpointer.folder="$OUT/checkpoints/'
            'agpt2b-mds154391-tulu-math-uc-streaming"'
        )
        for name in (
            "agpt2b_mds154391_broad_sft_smoke_2n.pbs",
            "agpt2b_mds154391_broad_sft900_8n.pbs",
        ):
            with self.subTest(name=name):
                self.assertIn(expected, (scripts / name).read_text())


if __name__ == "__main__":
    unittest.main()
