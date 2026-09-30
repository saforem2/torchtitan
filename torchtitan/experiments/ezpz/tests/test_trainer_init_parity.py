#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The ezpz trainer must delegate execution state to TrainingEngine.

Historically ``FaultTolerantTrainer`` copied ``Trainer.__init__``. Upstream now
owns execution state in ``TrainingEngine``; copying trainer initialization is
no longer a supported integration point.

This has now happened at least twice:

- `num_pipeline_parallel_microbatches` (noted in a comment at the assignment).
- `fwd_bwd_fn`, added by #3559 (CUDA-graph capture) + #4146 (in-place loss
  accumulation) in the 78th sync. The merge was CONFLICT-FREE and touched no
  ezpz file, so nothing flagged it; job 12473170 then failed 3/3 arms with
  rc=143 and zero steps: "'FaultTolerantTrainer' object has no attribute
  'fwd_bwd_fn'".

Static (AST) comparison -- imports nothing, so it runs anywhere, including a
laptop with no torch. Run it as part of every upstream sync.

    python3 torchtitan/experiments/ezpz/tests/test_trainer_init_parity.py
"""
from __future__ import annotations

import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(HERE, "..", "..", "..", "trainer.py")
EZPZ = os.path.join(HERE, "..", "trainer.py")
AGPT_PARALLELIZE = os.path.join(HERE, "..", "agpt", "parallelize.py")
MOE_PARALLELIZE = os.path.join(HERE, "..", "moe", "parallelize.py")

# Attributes core sets that ezpz intentionally does not, with the reason.
# Keep this list SHORT and justified -- each entry is a deliberate divergence,
# not a TODO.
INTENTIONAL_OMISSIONS: dict[str, str] = {}


def init_self_attrs(path: str, cls: str) -> set[str]:
    """Attribute names assigned to `self` inside <cls>.__init__."""
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "__init__":
                    return {
                        t.attr
                        for a in ast.walk(fn)
                        if isinstance(a, ast.Assign)
                        for t in ast.walk(a)
                        if isinstance(t, ast.Attribute)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == "self"
                        and isinstance(t.ctx, ast.Store)
                    }
    raise AssertionError(f"{cls}.__init__ not found in {path}")


def class_bases(path: str, cls: str) -> set[str]:
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return {ast.unparse(base) for base in node.bases}
    raise AssertionError(f"{cls} not found in {path}")


def test_seed_checkpoint_delegates_to_training_engine() -> None:
    """The ezpz trainer must save seed checkpoints through its engine."""
    entrypoint = os.path.join(HERE, "..", "train.py")
    source = open(entrypoint).read()
    assert "trainer.engine.save_checkpoint(last_step=True)" in source
    assert "trainer.checkpointer.save(" not in source


def test_ezpz_uses_current_engine_contract() -> None:
    """Reject legacy lifecycle names removed by upstream TrainingEngine."""
    source = open(EZPZ).read()
    assert "ParallelDims" not in source
    assert ".parallel_dims" not in source
    assert "def optimizer_step(" not in source
    assert "forward_backward_microbatch(" not in source
    assert "prepare_step(" not in source
    assert "super().optim_step()" in source
    assert "def as_input_dict(" in source
    assert "self._run_forward_backward = maybe_wrap_with_xpu_graph(" in source


def test_ezpz_does_not_restore_removed_model_compilation() -> None:
    """Upstream #4895 removed standard per-TransformerBlock compilation."""
    for path in (AGPT_PARALLELIZE, MOE_PARALLELIZE):
        source = open(path).read()
        assert '"model" in compile_config.components' not in source
        assert ".compile(backend=compile_config.backend" not in source


def main() -> int:
    bases = class_bases(EZPZ, "FaultTolerantTrainer")
    source = open(EZPZ).read()
    if "TorchFTTrainer" not in bases or "engine_cls" not in source:
        print("FAIL: ezpz must extend TorchFTTrainer and select an engine_cls")
        return 1
    test_seed_checkpoint_delegates_to_training_engine()
    test_ezpz_uses_current_engine_contract()
    test_ezpz_does_not_restore_removed_model_compilation()

    print("PASS: ezpz delegates execution state to a TrainingEngine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
