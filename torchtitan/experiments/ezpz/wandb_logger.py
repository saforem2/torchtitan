# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""W&B metric exporter for ezpz runs.

``EzpzWandBLogger`` is upstream's ``WandBLogger`` with the two behaviours ezpz
runs had before #5775:

1. **A failed start does not kill the run.** Before #5775 core built the W&B
   logger inside ``try/except Exception`` and logged "Failed to create WandB
   logger" (missing package, no network, no API key, init timeout), then
   trained without W&B. #5775 builds each ``metrics.loggers`` entry with no
   guard, so the same failure now aborts the trainer at construction. Here a
   failure in ``__init__`` is logged once and the logger becomes a no-op.

2. **``_step`` is the training step.** #5903 logs rows without ``step=`` and
   carries the training step as ``trainer/step``, so W&B's ``_step`` became a
   row counter. ezpz tooling (``utils/wandb_fetch.py``, the production plots,
   run-chain concatenation on resume) keys history on ``_step``. ``log`` passes
   ``step=step`` again and keeps the ``trainer/step`` key, without committing:
   a validation log at an already logged step merges into that step's row
   (pre-#5775 core used ``commit=True`` and W&B dropped it), and the row is
   committed when the next step logs or at ``wandb.finish()``.

Project, entity, run name, id, tags, group, resume/fork and the config payload
still come from upstream's ``WandBLogger.__init__`` (``WANDB_*`` environment).
"""

from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.experiments.ezpz.logging import logger
from torchtitan.observability.metrics import WandBLogger

__all__ = ["EzpzWandBLogger"]


def _is_rank_zero() -> bool:
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


class EzpzWandBLogger(WandBLogger):
    """``WandBLogger`` that degrades to a no-op when W&B cannot start."""

    @dataclass(kw_only=True, slots=True)
    class Config(WandBLogger.Config):
        pass

    enabled: bool

    def __init__(
        self,
        config: Config,
        *,
        log_dir: str,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ):
        self.enabled = False
        self.tag = tag
        try:
            WandBLogger.__init__(
                self, config, log_dir=log_dir, config_dict=config_dict, tag=tag
            )
        except Exception as e:
            self._abandon_partial_run()
            # Loggers are built on the metrics rank, or on every rank with
            # metrics.save_for_all_ranks: one warning from rank 0, and the
            # same text at info level elsewhere (under pipeline parallelism
            # the metrics rank is not rank 0).
            log = logger.warning if _is_rank_zero() else logger.info
            log(
                "Failed to create WandB logger (%s: %s); training continues "
                "without W&B.",
                type(e).__name__,
                e,
            )
            return
        self.enabled = True

    def _abandon_partial_run(self) -> None:
        """Finish a run that started before a later init step failed."""
        wandb = getattr(self, "wandb", None)
        try:
            if wandb is not None and wandb.run is not None:
                wandb.finish(exit_code=1)
        except Exception:  # noqa: BLE001 - best effort, already degrading
            pass

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if not self.enabled:
            return
        wandb_metrics = {
            (k if self.tag is None else f"{self.tag}/{k}"): v
            for k, v in metrics.items()
        }
        self.wandb.log({**wandb_metrics, "trainer/step": step}, step=step)

    def close(self) -> None:
        if self.enabled:
            WandBLogger.close(self)
