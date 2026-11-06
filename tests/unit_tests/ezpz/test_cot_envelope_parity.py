# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CoT envelope + threshold parity (Grain SFT vs TRL vs eval vs RL)."""

import re


def _extract_suffix(src: str) -> str | None:
    m = re.search(r"_(?:OPENR1|PROMPT)_SUFFIX\s*=\s*\(\s*\"(.*?)\"\s*\"(.*?)\"\s*\)", src, re.DOTALL)
    if m:
        return m.group(1) + m.group(2)
    m2 = re.search(r"_(?:OPENR1|PROMPT)_SUFFIX\s*=\s*\(\s*\"(.*?)\"\s*\)", src, re.DOTALL)
    return m2.group(1) if m2 else None


def _threshold(src: str) -> int | None:
    m = re.search(r"OPENR1_MAX_THINK_CHARS\s*=\s*int\(os\.environ\.get\([^,]+,\s*\"(\d+)\"\)", src)
    return int(m.group(1)) if m else None


_PATHS = {
    "grain": "torchtitan/experiments/ezpz/agpt/sft_data.py",
    "trl": "torchtitan/experiments/ezpz/rl/datasets_sft.py",
    "eval": "torchtitan/experiments/ezpz/scripts/eval/eval_cot_gsm8k.py",
    "rl": "torchtitan/experiments/ezpz/rl/tasks/gsm8k_reason.py",
}


def test_cot_suffix_identical():
    """All four sites must use the same CoT prompt suffix (train/eval/RL)."""
    suffixes = {}
    for name, path in _PATHS.items():
        src = open(path).read()
        suffixes[name] = _extract_suffix(src)
        assert suffixes[name] is not None, f"{name} suffix not found in {path}"
    # All should be identical
    vals = set(suffixes.values())
    assert len(vals) == 1, f"CoT suffix diverged: {suffixes}"


def test_openr1_threshold_not_filtering_all():
    """OPENR1_MAX_THINK_CHARS must be >= 4000 (old 1200 filtered 100% — median ~13k)."""
    for name in ("grain", "trl"):
        src = open(_PATHS[name]).read()
        thresh = _threshold(src)
        assert thresh is not None, f"{name} threshold not found"
        assert thresh >= 4000, f"{name} threshold {thresh} too low — filters median trace"


def test_grain_and_trl_threshold_match():
    """Grain and TRL thresholds must match (both drive CoT SFT)."""
    grain = _threshold(open(_PATHS["grain"]).read())
    trl = _threshold(open(_PATHS["trl"]).read())
    assert grain == trl, f"Grain {grain} vs TRL {trl} threshold mismatch"
