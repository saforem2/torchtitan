# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ezpz.History bridge: cross-rank statistics we cannot get today.

WHAT HISTORY ADDS THAT OUR PATH DOES NOT.

Our metrics are already reduced before we see them -- global_avg_loss is a sum
over ranks divided by tokens. So we log the MEAN and have no idea about the
SPREAD. History turns every scalar into `k/mean`, `k/max`, `k/min`, `k/std`
across ranks automatically. That directly answers a question we cannot ask
today: is one rank seeing pathological data? A DP rank with a corrupt shard
raises loss/std and loss/max while loss/mean barely moves.

WHY THIS IS A SEPARATE CHANNEL, NOT A REPLACEMENT.

torchtitan already owns a W&B run through MetricsProcessor. Handing History a
second wandb backend would produce two runs, or worse, two writers on one run
(we have already been bitten by the reinit-swallow: a preflight smoke run
holding the session so the real trainer attaches to IT). So History is
constructed with `backends="none"` by default -- purely local aggregation --
and the aggregated numbers are handed BACK to the existing logger through the
extra_metrics hook. One writer, one run.

The end-of-run artifacts (xarray .h5, markdown report, terminal plots) come
free from finalize() and cost nothing during training.

THE 384-RANK CLIFF, WHICH IS A REAL TRAP FOR US.

History auto-disables cross-rank aggregation above world_size 384 to avoid
all-reduce cost. Our production runs are 512N x 12 = 6144 ranks, so the
feature would SILENTLY do nothing exactly where we most want it. We check
world_size explicitly and log which regime we are in, rather than letting a
config default decide it invisibly.
"""

from __future__ import annotations

import os
from typing import Any

from torchtitan.tools.logging import logger

# 384 is History's documented cutoff for automatic cross-rank aggregation.
_AGG_MAX_WORLD = 384


class HistoryBridge:
    """Thin wrapper: local cross-rank aggregation, no second tracker."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        outdir: str | None = None,
        world_size: int = 1,
    ) -> None:
        self.enabled = enabled
        self.history: Any = None
        self.aggregating = False
        if not enabled:
            return
        try:
            import ezpz
        except ImportError:
            logger.warning("HistoryBridge: ezpz not importable, disabling")
            self.enabled = False
            return

        self.aggregating = world_size <= _AGG_MAX_WORLD
        if not self.aggregating:
            # Say so loudly. A silently-degraded metric is worse than none:
            # loss/std reading 0 could mean "ranks agree" or "not measured".
            logger.warning(
                f"HistoryBridge: world_size={world_size} > {_AGG_MAX_WORLD}, "
                "so ezpz.History will NOT aggregate across ranks. Per-rank "
                "spread metrics (loss/std, loss/max) will be absent, not zero."
            )
        try:
            self.history = ezpz.History(
                backends="none",  # torchtitan owns the W&B run; see docstring
                report_dir=outdir,
                distributed_history=self.aggregating,
            )
            logger.info(
                f"HistoryBridge: enabled (cross-rank aggregation="
                f"{self.aggregating}, outdir={outdir})"
            )
        except Exception as e:
            logger.warning(f"HistoryBridge: init failed ({e}), disabling")
            self.enabled = False

    def update(self, metrics: dict[str, float], step: int) -> dict[str, float]:
        """Feed PER-RANK values in; get cross-rank stats back.

        Returns only the derived stat keys (`*/std`, `*/max`, ...) so the
        caller can merge them into extra_metrics without duplicating the
        scalars torchtitan already logs.
        """
        if not self.enabled or self.history is None:
            return {}
        try:
            self.history.update(metrics, step=step)
        except Exception as e:
            logger.warning(f"HistoryBridge: update failed ({e}), disabling")
            self.enabled = False
            return {}
        if not self.aggregating:
            return {}
        derived: dict[str, float] = {}
        try:
            hist = getattr(self.history, "history", None)
            if isinstance(hist, dict):
                for key in metrics:
                    for stat in ("std", "max", "min"):
                        k = f"{key}/{stat}"
                        seq = hist.get(k)
                        if seq:
                            derived[f"rank_{stat}/{key}"] = float(seq[-1])
        except Exception:
            return {}
        return derived

    def finalize(self, outdir: str | None = None) -> None:
        """Write xarray datasets, markdown report and plots. Rank 0 only."""
        if not self.enabled or self.history is None:
            return
        if int(os.environ.get("RANK", "0")) != 0:
            return
        try:
            self.history.finalize(outdir=outdir)
            logger.info(f"HistoryBridge: finalized to {outdir}")
        except Exception as e:
            logger.warning(f"HistoryBridge: finalize failed ({e})")


def maybe_watch_model(history: Any, model_parts: list[Any]) -> None:
    """Enable wandb.watch for true weight/grad HISTOGRAMS.

    Our scalar norms say a distribution moved; a histogram says how. This is
    the one thing worth routing through W&B directly, via History's documented
    escape hatch.

    NOT free: watch(log="all") hooks every parameter and uploads histograms on
    a schedule. Costly at 26B params, hence opt-in and log_freq-throttled.
    """
    if history is None:
        return
    try:
        tracker = getattr(history, "tracker", None)
        if tracker is None:
            return
        wb = tracker.get_backend("wandb")
        if wb is None:
            return
        for part in model_parts:
            wb.watch(part, log="all", log_freq=500)
        logger.info("HistoryBridge: wandb.watch enabled (histograms, freq=500)")
    except Exception as e:
        logger.warning(f"HistoryBridge: wandb.watch unavailable ({e})")
