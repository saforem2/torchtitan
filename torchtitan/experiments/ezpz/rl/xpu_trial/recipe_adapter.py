# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Recipe adapter: ezpz-filtered re-export of Guoqiong XPU configs.

Guoqiong's ``alphabet_sort.py`` (ported verbatim as
``torchtitan/experiments/ezpz/rl/xpu_trial/alphabet_sort.py``) defines 22
explicit ``rl_grpo*`` factories plus ~696 dynamically generated host-layout
variants (``rl_grpo_qwen3_0_6b_{full,lora}_{hosts}n_ppn{gpus}_{steps}steps``
with ``_rep``/``_gtp`` suffixes).  Not all of them are meaningful for the
ezpz ``train_upstream`` path on Aurora XPU — MoE/DeepEP, GPT-OSS-20B and
Qwen3-30B-A3B/27B need expert parallelism or >8-GPU meshes not covered by the
trial launcher, and have never been exercised in the XPU regression.

This module imports that source and re-exports only the ezpz-compatible
subset.  It is the ``--module`` target for ConfigManager:

    python -m torchtitan.experiments.ezpz.rl.train_upstream \\
        --module torchtitan.experiments.ezpz.rl.xpu_trial.recipe_adapter \\
        --config rl_grpo_qwen3_0_6b_full_8n_ppn4_10steps --output-dir outputs/...

or via the shorthand:

    --module ezpz.rl.xpu_trial.recipe_adapter --config <name>

Filtering predicate is :func:`_is_ezpz_compatible` (kept in one place so the
allowlist is auditable).  Layout arithmetic itself lives in
:mod:`torchtitan.experiments.ezpz.rl.xpu_trial.layouts`; this module only
filters.

Must stay under ``experiments/ezpz/`` per AGENTS.md.
"""

from __future__ import annotations

import importlib

# Re-export layout helpers for callers that want CLI-sweep arithmetic without
# importing the full recipe (e.g. shell: ``python -c "from ...layouts import
# resolve_dp_degrees; ..."``).
from torchtitan.experiments.ezpz.rl.xpu_trial.layouts import (  # noqa: F401
    ezpz_host_layout,
    generator_count_for_tp,
    resolve_dp_degrees,
)

# ---------------------------------------------------------------------------
# Source module (the verbatim Guoqiong port). Importing it registers the
# dynamic ``globals()[f"rl_grpo_qwen3_0_6b_..."]`` host-layout names.
# ---------------------------------------------------------------------------
_src = importlib.import_module("torchtitan.experiments.ezpz.rl.xpu_trial.alphabet_sort")

# ---------------------------------------------------------------------------
# Compatibility filter
# ---------------------------------------------------------------------------

# Substrings that mark configs not exercised/compatible on the ezpz XPU path.
# - moe / deepep / 30b_a3b : expert parallelism (DeepEPTokenDispatcher, EP=4)
# - gpt_oss / 20b         : separate model family (GptOssRendererConfig)
# - 27b (qwen3_6_27b)     : 64k context, TP4 generator, not in trial launcher
_EXCLUDE_SUBSTRINGS = (
    "moe",
    "deepep",
    "30b_a3b",
    "30b-a3b",
    "gpt_oss",
    "qwen3_6_27b",
    "qwen3-6-27b",
)

# Explicit allowlist for the families we keep (prefix match).  This is the
# complement of the exclude list but written as allowlist so a new upstream
# family does not auto-leak into the trial.
_KEEP_PREFIXES = (
    "rl_grpo_qwen3_0_6b",  # primary XPU regression family (+ all _rep/_gtp)
    "rl_grpo_qwen3_1_7b",
    "rl_grpo_qwen3_14b",
    "rl_grpo_qwen3_5_9b",
    "rl_grpo_qwen3_5_debug",
    "rl_grpo_qwen3_0_6b_varlen",
    "rl_grpo_qwen3_0_6b_flex",
)


def _is_ezpz_compatible(name: str, obj: object) -> bool:
    """Return True if ``name`` should be re-exported for ezpz.

    Criteria:
    * Name looks like a recipe entry point (``rl_grpo*``) and is callable.
    * Not in the exclude list (MoE / GPT-OSS / large).
    * Starts with a kept prefix — catches both explicit defs and the dynamic
      ``_hosts`` / ``_rep`` / ``_gtp`` variants which all share the
      ``rl_grpo_qwen3_0_6b_`` prefix.
    """
    if not name.startswith("rl_grpo"):
        return False
    if not callable(obj):
        return False
    lower = name.lower()
    if any(sub in lower for sub in _EXCLUDE_SUBSTRINGS):
        return False
    # Keep if it matches any allowed prefix; the dynamic host-layout names all
    # start with rl_grpo_qwen3_0_6b_ so they pass here.
    if any(lower.startswith(p.lower()) for p in _KEEP_PREFIXES):
        return True
    # Fallback: if it's a dynamic host-layout name (contains "n_ppn") and not
    # excluded, keep it — this preserves future Qwen3-0.6B variants even if
    # _KEEP_PREFIXES is stale.
    if "n_ppn" in lower and "qwen3_0_6b" in lower:
        return True
    return False


# Populate this module's globals with the filtered subset so ConfigManager
# discovers them via ``getattr(module, config_name)``.  We keep __all__ tight
# so ``from ...recipe_adapter import *`` does not leak helpers.
for _name in dir(_src):
    _obj = getattr(_src, _name)
    if _is_ezpz_compatible(_name, _obj):
        globals()[_name] = _obj

# Also re-export the two base factories that the host-layout loop builds on,
# even though they don't match the dynamic suffix pattern — useful for
# single-node smoke:
for _base in (
    "rl_grpo_qwen3_0_6b_varlen",
    "rl_grpo_qwen3_0_6b_varlen_no_compile",
    "rl_grpo_qwen3_0_6b_varlen_checkpoint_test",
    "rl_grpo_qwen3_0_6b_flex",
    "rl_grpo_qwen3_0_6b_flex_batch_invariant",
    "rl_grpo_qwen3_0_6b_varlen_batch_invariant",
    "rl_grpo_qwen3_5_9b_varlen",
    "rl_grpo_qwen3_5_9b_varlen_batch_invariant",
    "rl_grpo_qwen3_5_debug_varlen",
    "rl_grpo_qwen3_5_debug_varlen_batch_invariant",
    "rl_grpo_qwen3_1_7b",
    "rl_grpo_qwen3_14b",
):
    if hasattr(_src, _base):
        globals()[_base] = getattr(_src, _base)

# Clean up import-time temporaries from module namespace
del _src, _name, _obj, _base, importlib  # type: ignore[name-defined]

# ConfigManager discovery uses ``callable(getattr(module, name))``; __all__
# controls ``from ... import *`` and documents the surface.
__all__ = sorted(
    k for k, v in globals().items() if k.startswith("rl_grpo") and callable(v)
)
