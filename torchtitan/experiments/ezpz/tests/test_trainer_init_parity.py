#!/usr/bin/env python3
"""FaultTolerantTrainer.__init__ must set everything core's __init__ sets.

`FaultTolerantTrainer` deliberately does NOT call `super().__init__()` -- it
reimplements the body so it can interpose fault-tolerance setup. The cost is
that any attribute upstream adds to `Trainer.__init__` silently goes missing
here, and the run dies at step 1 with a bare AttributeError.

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


def main() -> int:
    core = init_self_attrs(CORE, "Trainer")
    ezpz = init_self_attrs(EZPZ, "FaultTolerantTrainer")
    missing = sorted(core - ezpz - set(INTENTIONAL_OMISSIONS))

    print(f"core Trainer.__init__      sets {len(core)} self attributes")
    print(f"ezpz FaultTolerantTrainer  sets {len(ezpz)}")
    if INTENTIONAL_OMISSIONS:
        print(f"intentional omissions:     {len(INTENTIONAL_OMISSIONS)}")

    if missing:
        print()
        print(f"FAIL: {len(missing)} attribute(s) set by core but NOT by ezpz:")
        for m in missing:
            print(f"    self.{m}")
        print()
        print("Each will AttributeError at run time. Either mirror the")
        print("assignment in FaultTolerantTrainer.__init__ (with a comment")
        print("naming the upstream PR), or add it to INTENTIONAL_OMISSIONS")
        print("with a reason.")
        return 1

    print("\nPASS: ezpz sets every attribute core does.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
