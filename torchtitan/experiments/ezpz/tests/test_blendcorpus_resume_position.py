# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Stdlib-only regression tests; execute this file with python3.

Compile the real loader class and environment helper from its AST, supplying
only external dependency stubs. No torch, ezpz, BlendCorpus or process group is
required. These tests do not exercise the real distributed sampler/backend.
"""

import ast
from dataclasses import dataclass, field
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch


class Tensor:
    def __init__(self, rows):
        self.rows = rows

    def long(self):
        return self

    def __getitem__(self, index):
        rows, columns = index
        return Tensor([row[columns] for row in self.rows[rows]])

    def contiguous(self):
        return self


@dataclass(kw_only=True)
class BaseConfig:
    dataset: str = "blendcorpus"
    dataset_path: str = "unused"


def loader_class() -> tuple[Any, dict[str, Any]]:
    source = Path(__file__).resolve().parents[1] / "blendcorpus/blendcorpus_builder.py"
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)
                 or isinstance(node, ast.FunctionDef)
                 and node.name == "_env_resume_consumed_samples"]
    namespace: dict[str, Any] = dict(dataclass=dataclass, field=field, os=os, Any=Any,
                     SimpleNamespace=SimpleNamespace, logger=Mock(),
                     BaseDataLoader=type("BaseDataLoader", (), {"Config": BaseConfig}),
                     ezpz=SimpleNamespace(get_torch_device_type=lambda: "cpu"))
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["BlendCorpusDataLoader"], namespace


class ResumePositionTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ.pop("BLENDCORPUS_RESUME_CONSUMED_SAMPLES", None)
        self.addCleanup(self.env.stop)
        self.cls, self.ns = loader_class()
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.batches = [{"text": Tensor([[10, 11, 12], [20, 21, 22]])}] * 3
        self.builder = Mock(side_effect=lambda ds, consumed, cfg: self.batches)
        self.bc = None
        def set_config(cfg):
            self.bc = cfg
        self.ns["_import_blendcorpus_modules"] = lambda: (
            SimpleNamespace(model_parallel_is_initialized=lambda: True,
                            get_data_parallel_world_size=lambda: 4),
            lambda: self.bc, set_config,
            lambda cfg: ("train", "valid", None), self.builder)
        torch: Any = ModuleType("torch")
        torch.__version__ = "2.13.0"
        torch.distributed = ModuleType("torch.distributed")
        self.modules = patch.dict("sys.modules", {"torch": torch,
                                                   "torch.distributed": torch.distributed})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def make(self, **config):
        cfg = self.cls.Config(data_cache_path=self.tmp.name, **config)
        # Intentionally different from sampler dp=4, and global batch=64.
        return self.cls(cfg, dp_world_size=99, dp_rank=0, tokenizer=None,
                        seq_len=2, local_batch_size=2, global_batch_size=64)

    def test_counter_advances_before_yield_with_legacy_tuple_layout(self):
        loader = self.make()
        iterator = iter(loader)
        inputs, labels = next(iterator)
        self.assertEqual(loader.state_dict(), {"consumed_samples": 8})
        self.assertEqual(set(inputs), {"input"})
        self.assertEqual(inputs["input"].rows, [[10, 11], [20, 21]])
        self.assertEqual(labels.rows, [[11, 12], [21, 22]])
        next(iterator)
        self.assertEqual(loader.state_dict(), {"consumed_samples": 16})
        iterator.close()
        self.assertEqual(loader.state_dict(), {"consumed_samples": 16})

    def test_zero_checkpoint_override(self):
        loader = self.make(resume_consumed_samples=80)
        loader.load_state_dict({"consumed_samples": 0})
        self.assertEqual(loader.state_dict()["consumed_samples"], 80)
        self.assertEqual(self.builder.call_args.args[1], 80)
        next(iter(loader))
        self.assertEqual(loader.state_dict()["consumed_samples"], 88)

    def test_missing_counter_override_and_validation_split_rebuild(self):
        loader = self.make(resume_consumed_samples=80, serve_validation=True)
        loader.load_state_dict({})
        self.assertEqual(loader.state_dict()["consumed_samples"], 80)
        self.assertEqual(self.builder.call_args.args[0:2], ("valid", 80))

    def test_no_consumption_without_yield_and_empty_loader(self):
        loader = self.make()
        iterator = iter(loader)
        self.assertEqual(loader.state_dict()["consumed_samples"], 0)
        iterator.close()
        loader._loader = []
        self.assertEqual(list(loader), [])
        self.assertEqual(loader.state_dict()["consumed_samples"], 0)

    def test_nonzero_checkpoint_wins_even_over_invalid_unused_override(self):
        loader = self.make(resume_consumed_samples=-1)
        loader.load_state_dict({"consumed_samples": 24})
        self.assertEqual(loader.state_dict()["consumed_samples"], 24)
        self.assertEqual(self.builder.call_args.args[1], 24)

    def test_unset_and_empty_env_default_none(self):
        for value in (None, "", "  "):
            with self.subTest(value=value):
                if value is None:
                    os.environ.pop("BLENDCORPUS_RESUME_CONSUMED_SAMPLES", None)
                else:
                    os.environ["BLENDCORPUS_RESUME_CONSUMED_SAMPLES"] = value
                self.assertIsNone(self.cls.Config().resume_consumed_samples)
                loader = self.make()
                loader.load_state_dict({})
                self.assertEqual(loader.state_dict()["consumed_samples"], 0)
                self.assertEqual(self.builder.call_count, 1)
                self.builder.reset_mock()

    def test_env_is_read_at_config_creation_and_explicit_value_wins(self):
        os.environ["BLENDCORPUS_RESUME_CONSUMED_SAMPLES"] = " 80 "
        loader = self.make()
        loader.load_state_dict({"consumed_samples": 0})
        self.assertEqual(loader.state_dict()["consumed_samples"], 80)
        self.assertEqual(self.cls.Config(resume_consumed_samples=16).resume_consumed_samples, 16)
        self.assertIsNone(self.cls.Config(resume_consumed_samples=None).resume_consumed_samples)
        os.environ["BLENDCORPUS_RESUME_CONSUMED_SAMPLES"] = "invalid"
        with self.assertRaises(ValueError):
            self.cls.Config()

    def test_invalid_effective_positions_rejected_without_mutation(self):
        for override, stored in ((None, -8), (None, 3), (-8, 0), (3, 0)):
            with self.subTest(override=override, stored=stored):
                loader = self.make(resume_consumed_samples=override)
                self.builder.reset_mock()
                with self.assertRaises(ValueError):
                    loader.load_state_dict({"consumed_samples": stored})
                self.assertEqual(loader.state_dict()["consumed_samples"], 0)
                self.builder.assert_not_called()

    def test_roundtrip_and_repeated_load_do_not_rebuild(self):
        loader = self.make()
        next(iter(loader))
        state = loader.state_dict()
        restored = self.make()
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(), state)
        self.builder.reset_mock()
        restored.load_state_dict(state)
        self.builder.assert_not_called()
        next(iter(restored))
        self.assertEqual(restored.state_dict()["consumed_samples"], 16)

    def test_global_step_setter_preserved(self):
        loader = self.make()
        loader.set_consumed_by_global_step(3, 64)
        self.assertEqual(loader.state_dict()["consumed_samples"], 192)
        next(iter(loader))
        self.assertEqual(loader.state_dict()["consumed_samples"], 200)

    def test_hf_delegate_constructor_iteration_state_and_length_preserved(self):
        delegate = Mock()
        delegate.__iter__ = Mock(return_value=iter([("legacy", "labels")]))
        delegate.__len__ = Mock(return_value=7)
        delegate.state_dict.return_value = {"hf": 1}
        hf_config = Mock(return_value=SimpleNamespace(build=Mock(return_value=delegate)))
        self.ns["HuggingFaceTextDataLoader"] = SimpleNamespace(Config=hf_config)
        loader = self.make(dataset="c4_test", resume_consumed_samples=80)
        self.assertEqual(list(loader), [("legacy", "labels")])
        self.assertEqual(len(loader), 7)
        self.assertEqual(loader.state_dict(), {"hf": 1})
        loader.load_state_dict({"hf": 2})
        delegate.load_state_dict.assert_called_once_with({"hf": 2})
        loader.set_consumed_by_global_step(3, 64)
        self.builder.assert_not_called()
        self.assertEqual(hf_config.call_args.kwargs["dataset"], "c4_test")


if __name__ == "__main__":
    unittest.main(verbosity=2)
