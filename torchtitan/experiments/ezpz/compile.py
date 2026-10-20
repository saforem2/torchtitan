# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ezpz-owned ``compile`` config.

Upstream deleted ``CompileConfig`` and ``TrainingEngine.Config.compile``
(#5026): compilation is now a per-model list of ``@local_compile`` regions
(``BaseModel.Config.local_compile_regions``). ezpz recipes, launchers and the
LR finder still select compilation through ``--compile.enable`` /
``--compile.no-enable`` and ``cfg.compile = CompileConfig(...)``, so this
module keeps that surface and maps it onto ``local_compile_regions`` when the
trainer builds the model.

Semantics preserved from the pre-sync ``CompileConfig``:

- ``compile is None`` (the default, ``--compile.no-enable``): nothing is
  compiled. Upstream's model defaults now enable ``loss`` and
  ``fused_binary_activation`` unconditionally; ezpz keeps the explicit opt-in
  because compiled regions are not portable to every XPU runtime.
- ``compile = CompileConfig()`` (``--compile.enable``): the ``components`` are
  compiled; the pre-sync default (``["loss"]``) is unchanged.
"""

from dataclasses import dataclass, field


# Regions an ezpz run may request through ``compile.components``. They must be
# names registered with ``torchtitan.distributed.local_compile.local_compile``;
# ``apply_local_compile`` rejects unknown names, so this list only documents the
# intended surface.
ALLOWED_COMPONENTS = frozenset(
    {
        "loss",
        "fused_binary_activation",
        "rope",
        "gated_rmsnorm",
        "moe_aux_loss",
        "fp32_to_bf16_split",
    }
)


@dataclass(kw_only=True, slots=True)
class CompileConfig:
    components: list[str] = field(default_factory=lambda: ["loss"])
    """``@local_compile`` regions to compile."""

    backend: str = "inductor"
    """Retained for CLI compatibility; ``local_compile`` fixes the backend."""

    def __post_init__(self) -> None:
        unknown = [c for c in self.components if c not in ALLOWED_COMPONENTS]
        if unknown:
            raise ValueError(
                f"Unknown compile.components entries {unknown}; "
                f"allowed values are {sorted(ALLOWED_COMPONENTS)}"
            )


def local_compile_regions(compile_config: "CompileConfig | None") -> list[str]:
    """Map the ezpz ``compile`` field to ``local_compile_regions``."""
    if compile_config is None:
        return []
    return list(compile_config.components)
