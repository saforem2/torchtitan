# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Output z-loss: a penalty on the softmax log-normalizer.

WHAT IT IS. Standard cross-entropy is invariant to a constant shift of the
logits, so nothing stops ``log Z = logsumexp(logits)`` from drifting far from
zero. Adding ``coef * (log Z)^2`` removes that freedom and keeps the logits
near a bounded scale. Introduced for PaLM (arXiv:2204.02311 Sec 5, coef 1e-4)
and used since by Chinchilla, OLMo-2 (arXiv:2501.00656, cited among its
stability changes) and others.

WHAT IT IS NOT. This is the OUTPUT z-loss on the model's vocabulary softmax.
It is unrelated to the MoE ROUTER z-loss of the Switch Transformer work
(arXiv:2202.08906), which regularizes the router's expert-selection logits.
Only the router variant appears anywhere in this tree today, vendored inside
transformers, and it does nothing for a dense model.

WHY IT MIGHT MATTER HERE. The 80B NaN was investigated as a score-bounding
problem and produced ``SoftcappedFlexAttention`` and QK-Norm, both of which
bound ATTENTION scores. Nothing bounds the OUTPUT logits. z-loss is the
standard tool for that failure mode and was never tried. See
``docs/guides/known-bugs/`` for the 80B history.

PARALLELISM. Under loss parallelism the lm_head output is vocab-sharded on
the TP axis, so a naive ``logsumexp(logits, dim=-1)`` would compute a
PER-SHARD normalizer -- a different, smaller quantity than log Z, silently
wrong rather than an error. This module gathers the vocab dimension first,
reusing the same redistribute logic core uses in ``compute_logprobs`` rather
than reimplementing it, and covers both the DTensor and the ``spmd_types``
backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import spmd_types as spmd
import torch
from torch.distributed.tensor import DTensor, Replicate, Shard

from torchtitan.components.loss import BaseLoss, CrossEntropyLoss
from torchtitan.config import CompileConfig
from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.distributed.utils import get_spmd_backend


def _gather_vocab(logits: torch.Tensor) -> torch.Tensor:
    """Return logits with the vocab dimension replicated on the TP axis.

    Mirrors the gather in ``torchtitan.components.loss.compute_logprobs`` so
    the two paths cannot drift. A per-shard logsumexp is not log Z, and it
    would produce a plausible number rather than an error.
    """
    if isinstance(logits, DTensor):
        placements = tuple(
            Replicate()
            if isinstance(p, Shard) and p.dim in (-1, logits.ndim - 1)
            else p
            for p in logits.placements
        )
        return logits.redistribute(placements=placements).to_local()
    if get_spmd_backend() == "spmd_types" and spmd_mesh_size("tp") > 1:
        # dst=I matches compute_logprobs: the all-gather's backward is the
        # replicated upstream grad sliced back to this rank's vocab shard,
        # not an all-reduce (which would over-count by tp_degree).
        return spmd.redistribute(logits, "tp", src=spmd.S(-1), dst=spmd.I)
    return logits


def z_loss_term(logits: torch.Tensor, coef: float) -> torch.Tensor:
    """``coef * sum((log Z)^2)`` over tokens, computed in fp32.

    Sum reduction, not mean, to match the sum-reduced cross entropy this is
    added to -- the trainer divides the total by ``global_valid_tokens``.
    Computing in fp32 matters: ``log Z`` for a 100k-vocab model is O(10) and
    squaring it in bf16 loses precision exactly where the penalty is meant to
    bite.
    """
    gathered = _gather_vocab(logits).float()
    log_z = torch.logsumexp(gathered, dim=-1)
    return coef * (log_z**2).sum()


class CrossEntropyWithZLoss(BaseLoss):
    """Cross entropy plus an output z-loss penalty.

    Delegates the cross-entropy term to core's ``CrossEntropyLoss`` rather
    than reimplementing it, so loss-parallel handling, the spmd annotations
    and the ``global_valid_tokens`` scaling stay in one place.

    ``z_loss_coef=0.0`` is a hard bypass: the penalty is not computed at all
    and the result is bit-identical to plain cross entropy. That is what makes
    this safe to make the default loss.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        z_loss_coef: float = 0.0
        """Weight on (log Z)^2. PaLM and Chinchilla use 1e-4. 0.0 disables it
        entirely (bit-identical to plain cross entropy)."""

        global_vocab_size: int | None = None
        """Full vocabulary size, needed for spmd_types loss-parallel CE."""

    def __init__(
        self, config: Config, *, compile_config: CompileConfig | None = None
    ):
        if config.z_loss_coef < 0.0:
            raise ValueError(
                f"z_loss_coef must be >= 0, got {config.z_loss_coef}. "
                "A negative coefficient rewards logit growth, which is the "
                "opposite of the intended effect."
            )
        ce_config = CrossEntropyLoss.Config(
            global_vocab_size=config.global_vocab_size
        )
        self._ce = CrossEntropyLoss(ce_config, compile_config=compile_config)
        self.fn = self._ce.fn
        self.z_loss_coef = float(config.z_loss_coef)

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss, metrics = self._ce(pred, labels, global_valid_tokens, **kwargs)
        if self.z_loss_coef == 0.0:
            return loss, metrics
        z = z_loss_term(pred, self.z_loss_coef)
        if global_valid_tokens is not None:
            z = z / global_valid_tokens
        metrics = dict(metrics)
        # Report the penalty separately so a run shows whether it is doing
        # anything. A z-loss folded invisibly into the total is untunable.
        metrics["z_loss"] = z.detach()
        return loss + z, metrics
