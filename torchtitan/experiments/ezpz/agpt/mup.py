# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Maximal Update Parametrization (muP) for the agpt model family. AdamW only.

Audit, design, and the staged plan this implements:
``torchtitan/experiments/ezpz/docs/experiments/mup/README.md``.

Everything here is OPT-IN and additive. The muP attention wrappers are new
subclasses rather than edits to the standard ones; the muP unembedding init is a
new function beside ``_output_linear_init``; the muP parameter groups live in
``default_mup_adamw`` beside ``default_adamw``; the muP flavors are new keys in
``agpt_configs``. A config that does not ask for muP is byte-for-byte what it
was before this file existed.

What muP asks for, per group, in the MULTIPLIER form of Table 8 of Tensor
Programs V (arXiv 2203.03466), with ``m = dim / base_dim``:

    group                muP init           muP AdamW LR   agpt today
    tok_embeddings       width-independent  eta            std=1.0        -- ok
    wq/wkv/w1            fan_in^-1/2        eta/m          sqrt(2/(5d))   -- ok
    wo, w2, w3           fan_in^-1/2        eta/m          same, x depth  -- ok
    norms                1.0                eta            ones_          -- ok
    lm_head              fan_in^-1          eta            d^-1/2         -- CHANGED
    attention logits     alpha*sqrt(h0)/h   --             1/sqrt(h)      -- CHANGED

Three of the four init groups already satisfy muP by accident, because agpt's
Megatron-DeepSpeed base std ``sqrt(2/(5*dim))`` is ``fan_in^-1/2`` up to a
width-independent constant. Only the unembedding and the attention logit scale
are off-spec, and only the parameter grouping is missing.

THE LM_HEAD PAIRING IS NOT OPTIONAL. Under Adam the ``d^-1`` init alone does
nothing: Adam is scale-invariant in the gradient, so shrinking the init changes
where the weight starts and nothing about how far each step moves it. The init
change only expresses muP when it is paired with an O(1) learning-rate group for
``lm_head`` while the hidden matrices sit at ``eta/m``. Use
``mup_output_linear_init`` and ``default_mup_adamw`` together or use neither;
:func:`build_mup_agpt_config` and :func:`default_mup_adamw` are wired to make
the mismatch hard to reach by accident.

SCOPE: AdamW only. Muon and Mano already apply a competing width rule -- they
partition by tensor shape and rescale the LR by ``0.2 * sqrt(max(A, B))``
(``experiments/ezpz/optimizer/muon.py:130-139``, ``mano.py:141``), measured at
15.68x to 101.22x on 30B tensor shapes. Stacking muP's ``1/m`` on top yields the
product of two parametrizations, not muP. Per arXiv 2602.20937 Muon wants
Theta(1) hidden-weight scaling -- no muP LR factor at all -- because its update
normalization already encodes the spectral condition. That is a separate
question, not a second arm of this one.

NO PER-TENSOR ATTRIBUTES. Everything is derived from config scalars at build
time -- the Cerebras approach. Deliberately unlike the ``mup`` package, whose
``p.infshape`` attributes do not survive ``torch.save`` (pytorch#72129), are
stripped by ``DataParallel``, and have no FSDP story (mup#59, mup#72, both
open). Config scalars are consumed before any FSDP/TP wrapping happens, so they
survive it trivially.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import torch
import torch.nn as nn

from torchtitan.components.optimizer.optimizer import (
    OptimizersContainer,
    ParamGroupConfig,
)

__all__ = [
    "MUP_PARAM_GROUP_PATTERNS",
    "MupScaledDotProductAttention",
    "MupXPUScaledDotProductAttention",
    "build_mup_agpt_config",
    "classify_mup_group",
    "default_mup_adamw",
    "mup_attention_scale",
    "mup_output_linear_init",
    "mup_width_multiplier",
    "register_mup_flavors",
    "summarize_mup_param_groups",
]


# ---------------------------------------------------------------------------
# The two changed scalars
# ---------------------------------------------------------------------------


def mup_width_multiplier(dim: int, base_dim: int) -> float:
    """``m = dim / base_dim``: the only width-dependent scalar muP needs."""
    if base_dim <= 0:
        raise ValueError(f"base_dim must be positive, got {base_dim}")
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    return dim / base_dim


def mup_output_linear_init(dim: int) -> dict[str, Callable]:
    """Unembedding init at ``fan_in^-1``, replacing agpt's ``fan_in^-1/2``.

    ``agpt/__init__.py:_output_linear_init`` uses ``std = dim**-0.5``. muP wants
    ``dim**-1.0`` for the readout. Truncation stays at 3 sigma so the exponent
    is the only difference from the function this replaces.

    Meaningless on its own under Adam -- see the module docstring.
    """
    s = float(dim) ** -1.0
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def mup_attention_scale(
    head_dim: int,
    base_head_dim: int,
    *,
    form: Literal["multiplier", "absolute"] = "multiplier",
    alpha: float = 1.0,
) -> float:
    """The muP attention logit scale, replacing the implicit ``1/sqrt(head_dim)``.

    Two forms, differing only by the width-independent constant
    ``sqrt(base_head_dim)``:

    ``"multiplier"`` (default) is ``alpha * sqrt(base_head_dim) / head_dim``,
    the Table 8 form. At ``head_dim == base_head_dim`` it reduces EXACTLY to
    ``1/sqrt(head_dim)``, so the base rung is numerically identical to what agpt
    trains today.

    ``"absolute"`` is ``alpha / head_dim``, the Table 3 form.

    DEVIATION FROM THE AUDIT, DELIBERATE. The audit's change list says
    "attention scale ``1/sqrt(head_dim)`` -> ``1/head_dim``", which is the
    ``"absolute"`` form. Taken literally at the production head_dim of 128 that
    is ``1/128 = 0.0078125`` against today's ``1/sqrt(128) = 0.08838835``, an
    11.31x reduction in logit scale AT THE BASE WIDTH, where muP is supposed to
    change nothing. Only the ratio between rungs carries the muP content; the
    absolute constant is free, and spending it on an 11x offset would confound
    the stage-5 comparison against the already-measured standard-parametrization
    30B baseline for reasons that have nothing to do with width. The multiplier
    form is what Table 8 actually specifies and is what this defaults to. Pass
    ``form="absolute"`` for the literal reading.

    IMPORTANT, AND EASY TO MISREAD AS A BUG: on the width ladder registered by
    :func:`register_mup_flavors`, ``head_dim`` is held FIXED (n_heads scales with
    dim, head_dim does not), which is what makes the ladder a width-only sweep.
    Both forms are therefore width-INDEPENDENT constants on that ladder, and the
    ``"multiplier"`` default is bit-identical to the ``scale=None`` the standard
    path passes today. The attention knob is inert there. It is implemented, and
    correct, for the other kind of ladder -- fixed ``n_heads``, growing
    ``head_dim`` -- where it is the difference between transfer and no transfer.
    Do NOT read a flat coordinate check on the fixed-head_dim ladder as evidence
    that this knob works: that ladder cannot test it.
    """
    if head_dim <= 0 or base_head_dim <= 0:
        raise ValueError(
            f"head_dim and base_head_dim must be positive, got "
            f"{head_dim} and {base_head_dim}"
        )
    if form == "multiplier":
        return alpha * (float(base_head_dim) ** 0.5) / float(head_dim)
    if form == "absolute":
        return alpha / float(head_dim)
    raise ValueError(f"form must be 'multiplier' or 'absolute', got {form!r}")


# ---------------------------------------------------------------------------
# Attention wrappers
# ---------------------------------------------------------------------------
#
# SUBCLASSES, not a flag on the standard wrappers, so the standard path is
# untouched by construction rather than by a branch that has to stay correct.
#
# The scale is a Config FIELD. It is NOT a module-level global (which would
# apply process-wide, so a muP config built in the same process as a standard
# one would silently reparametrize the standard one -- the global in
# `set_ezpz_max_context_length` is safe only because that value is genuinely
# process-wide), and it is NOT a new positional parameter. The forwards below
# re-declare the inherited positional parameter NAMES exactly, unchanged:
# `set_gqa_inner_attention_local_map` (models/common/decoder_sharding.py:299-314)
# keys `in_dst_shardings` by positional-arg name, and the local_map contract
# check asserts under TP>1 when a mapped input is missing. Renaming
# q_TNH/k_TNH/v_TNH broke TP=2 once already -- see agpt/__init__.py:76-91.
#
# Only the SDPA path is covered. SoftcappedFlexAttention takes a `scale` too but
# has no muP subclass here; a muP config asking for logit_softcap is rejected in
# build_mup_agpt_config rather than silently keeping the 1/sqrt(h) scale.


def _mup_resolve_scale(
    module: "MupScaledDotProductAttention",
    q: torch.Tensor,
    incoming: float | None,
) -> float:
    """Substitute the baked muP scale for the caller's, loudly."""
    # A non-None incoming scale means some config set GQAttention.head_dim,
    # which makes `self.scaling` non-None (models/common/attention.py:947) and
    # would otherwise silently win over muP's. No agpt config does that today.
    if incoming is not None:
        raise ValueError(
            f"muP attention received an explicit scale={incoming!r} from "
            "GQAttention. That happens when a config sets head_dim, which "
            "makes self.scaling non-None and would override the muP logit "
            "scale. Leave head_dim unset on muP configs."
        )
    head_dim = q.shape[-1]
    if head_dim != module.expected_head_dim:
        raise ValueError(
            f"muP attention scale was baked for head_dim="
            f"{module.expected_head_dim} but the tensors have head_dim="
            f"{head_dim}. The scale is a config scalar resolved at build time, "
            "so a mismatch means the config and the model disagree and the "
            "logit scale would be silently wrong."
        )
    return module.attention_scale


# Imported here rather than at module top so `mup` can be imported without
# pulling the agpt package back in while it is still initializing.
from torchtitan.experiments.ezpz.agpt import (  # noqa: E402
    EzpzScaledDotProductAttention,
    XPUScaledDotProductAttention,
)


class MupScaledDotProductAttention(EzpzScaledDotProductAttention):
    """agpt SDPA with the muP logit scale baked in as a config scalar.

    Everything about the kernel call, the #4121 flat-layout unflatten, and the
    backend list is inherited unchanged; only `scale` is substituted.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(EzpzScaledDotProductAttention.Config):
        attention_scale: float
        """Fully resolved muP logit scale from :func:`mup_attention_scale`."""

        expected_head_dim: int
        """head_dim the scale was computed for; asserted against the tensors."""

    def __init__(self, config: "MupScaledDotProductAttention.Config") -> None:
        super().__init__(config)
        self.attention_scale = float(config.attention_scale)
        self.expected_head_dim = int(config.expected_head_dim)

    # pyrefly: ignore [bad-override]
    def forward(
        self,
        q_TNH: torch.Tensor,
        k_TNH: torch.Tensor,
        v_TNH: torch.Tensor,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
        is_causal: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        # Positional names q_TNH/k_TNH/v_TNH are a TP>1 contract. Do not rename,
        # do not add positionals. See the section comment above.
        return super().forward(
            q_TNH,
            k_TNH,
            v_TNH,
            scale=_mup_resolve_scale(self, q_TNH, scale),
            enable_gqa=enable_gqa,
            is_causal=is_causal,
            **kwargs,
        )


class MupXPUScaledDotProductAttention(XPUScaledDotProductAttention):
    """muP logit scale on the XPU-backend SDPA wrapper. See the sibling class."""

    @dataclass(kw_only=True, slots=True)
    class Config(XPUScaledDotProductAttention.Config):
        attention_scale: float
        expected_head_dim: int

    def __init__(self, config: "MupXPUScaledDotProductAttention.Config") -> None:
        super().__init__(config)
        self.attention_scale = float(config.attention_scale)
        self.expected_head_dim = int(config.expected_head_dim)

    # pyrefly: ignore [bad-override]
    def forward(
        self,
        q_TNH: torch.Tensor,
        k_TNH: torch.Tensor,
        v_TNH: torch.Tensor,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
        is_causal: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        return super().forward(
            q_TNH,
            k_TNH,
            v_TNH,
            scale=_mup_resolve_scale(self, q_TNH, scale),
            enable_gqa=enable_gqa,
            is_causal=is_causal,
            **kwargs,
        )


def _mup_inner_attention_config(
    *,
    attention_scale: float,
    expected_head_dim: int,
):
    """Pick the muP SDPA config for the current device.

    Mirrors ``agpt/__init__.py:_default_inner_attention``.
    """
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return MupXPUScaledDotProductAttention.Config(
            attention_scale=attention_scale,
            expected_head_dim=expected_head_dim,
        )
    return MupScaledDotProductAttention.Config(
        attention_scale=attention_scale,
        expected_head_dim=expected_head_dim,
    )


# ---------------------------------------------------------------------------
# Parameter groups
# ---------------------------------------------------------------------------

# First-match-wins, in this order. Every pattern is LEAF-anchored or matches a
# top-level prefix that no wrapper can displace.
#
# AC TRAP, VERIFIED: agpt defaults to FullAC (config_registry.py:357), and the
# checkpoint wrapper inserts a `_checkpoint_wrapped_module` segment into the
# FQNs that `named_parameters()` yields:
#     layers.0.attention.wo.weight
#  -> layers.0._checkpoint_wrapped_module.attention.wo.weight
# Matching runs `pattern.search()` on the RAW name
# (components/optimizer/optimizer.py:195) and canonicalizes only afterwards for
# storage (:197). So a prefix-anchored pattern like `^layers\.\d+\.attention\.`
# matches ZERO parameters under AC and raises the empty-group ValueError. The
# wrapper only ever wraps transformer BLOCKS, so `^tok_embeddings\.`,
# `^lm_head\.` and `^norm\.weight$` are outside it and stay anchored -- but the
# per-layer norms are inside it, which is why group 3 is leaf-anchored.
MUP_PARAM_GROUP_PATTERNS: dict[str, str] = {
    # O(1) LR: input embedding.
    "embedding": r"^tok_embeddings\.",
    # O(1) LR: readout. Paired with mup_output_linear_init -- see the module
    # docstring on why the init alone is a no-op under Adam.
    "unembedding": r"^lm_head\.",
    # O(1) LR: all norm gains, per-layer (inside the AC wrapper, so leaf-
    # anchored) and the final pre-readout norm (outside it).
    "norm": (
        r"(?:^|\.)(?:attention_norm|ffn_norm|q_norm|k_norm)\.weight$"
        r"|(?:^|\.)norm\.weight$"
    ),
    # eta/m: every hidden matrix -- wq, wkv, wo, w1, w2, w3.
    "hidden": r".*",
}


def classify_mup_group(canonical_name: str) -> str:
    """Independent oracle: which muP group SHOULD a canonical FQN land in?

    Deliberately written from the parameter's ROLE rather than from the regexes
    in :data:`MUP_PARAM_GROUP_PATTERNS`, so that comparing the two is a real
    check and not a tautology. Used by :func:`summarize_mup_param_groups` and by
    the verification harness.

    ``canonical_name`` must already have wrapper segments stripped (see
    ``components/checkpointer/utils.py:canonical_fqn``).
    """
    leaf = canonical_name.rsplit(".", 1)[0].rsplit(".", 1)[-1]
    if canonical_name.startswith("tok_embeddings."):
        return "embedding"
    if canonical_name.startswith("lm_head."):
        return "unembedding"
    if leaf in ("attention_norm", "ffn_norm", "q_norm", "k_norm", "norm"):
        return "norm"
    return "hidden"


def default_mup_adamw(
    lr: float = 8e-4,
    *,
    dim: int,
    base_dim: int,
    independent_weight_decay: bool = False,
    **kwargs: Any,
) -> OptimizersContainer.Config:
    """AdamW with muP's four parameter groups instead of one catch-all.

    Shape follows ``components/optimizer/optimizer.py:default_adamw``
    (378-399); the betas/eps/weight_decay defaults are identical, so at
    ``dim == base_dim`` the only difference from ``default_adamw(lr)`` is that
    the same settings arrive as four groups rather than one.

    Args:
        lr: ``eta``, the O(1) learning rate. Applied as-is to embeddings, norms,
            and the readout; hidden matrices get ``eta / m``.
        dim: this model's width.
        base_dim: the width ``eta`` was tuned at. ``m = dim / base_dim``.
        independent_weight_decay: PyTorch's AdamW decays with
            ``param.mul_(1 - lr * weight_decay)`` (``torch/optim/adam.py:419``),
            so the decay is COUPLED to the per-group lr and scaling the hidden
            LR by ``1/m`` silently scales its effective decay by ``1/m`` too.
            Default ``False`` keeps the literal muP prescription -- only the LR
            is touched. Set ``True`` to multiply the hidden group's
            ``weight_decay`` by ``m`` so that ``lr * weight_decay`` is
            width-independent, which is u-muP's "fully decoupled AdamW". Per the
            audit's section 5.1, weight decay is one of the three features
            documented to break transfer on a recipe like this one, so this is
            the first knob to try when the coordinate check fails.
        **kwargs: forwarded to every group, overriding the shared defaults.

    Note:
        ``lr`` is a named parameter, so a second ``lr=`` in the call is a
        ``TypeError`` from Python before this function runs; there is no guard
        for it here. A ``weight_decay`` in ``kwargs`` DOES apply, to all four
        groups, and is then scaled on the hidden group when
        ``independent_weight_decay`` is set.
    """
    m = mup_width_multiplier(dim, base_dim)

    shared: dict[str, Any] = {
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.1,
        **kwargs,
    }
    hidden = dict(shared)
    hidden["lr"] = lr / m
    if independent_weight_decay:
        hidden["weight_decay"] = shared["weight_decay"] * m

    def _group(key: str, opt_kwargs: dict[str, Any]) -> ParamGroupConfig:
        return ParamGroupConfig(
            pattern=MUP_PARAM_GROUP_PATTERNS[key],
            optimizer_name="AdamW",
            optimizer_kwargs=opt_kwargs,
        )

    return OptimizersContainer.Config(
        param_groups=[
            _group("embedding", {**shared, "lr": lr}),
            _group("unembedding", {**shared, "lr": lr}),
            _group("norm", {**shared, "lr": lr}),
            _group("hidden", hidden),
        ]
    )


def summarize_mup_param_groups(
    model: nn.Module,
    param_groups: list[ParamGroupConfig],
) -> dict[str, Any]:
    """Replay the optimizer's own matching over a real model and audit it.

    Uses the SAME first-match-wins loop and the SAME ``pattern.search()`` on the
    RAW (wrapper-decorated) name as
    ``OptimizersContainer._build_param_groups`` (optimizer.py:188-197), then
    cross-checks each assignment against :func:`classify_mup_group`, which is
    derived independently from the parameter's role.

    Returns a dict with per-group counts and LRs, plus ``unclaimed``,
    ``empty_groups``, and ``misrouted`` -- all three of which must be empty for
    the parametrization to be what it claims.
    """
    import re

    from torchtitan.components.checkpointer.utils import canonical_fqn

    all_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    claimed: set[str] = set()
    groups: list[dict[str, Any]] = []
    misrouted: list[dict[str, str]] = []

    # Group index -> the role key it is meant to implement, by pattern identity.
    pattern_to_key = {v: k for k, v in MUP_PARAM_GROUP_PATTERNS.items()}

    for pg in param_groups:
        rx = re.compile(pg.pattern)
        names = [
            n for n, p in all_named if n not in claimed and rx.search(n)
        ]
        claimed.update(names)
        key = pattern_to_key.get(pg.pattern, "?")
        for n in names:
            expected = classify_mup_group(canonical_fqn(n))
            if key != "?" and expected != key:
                misrouted.append(
                    {"param": n, "landed_in": key, "should_be": expected}
                )
        groups.append(
            {
                "role": key,
                "pattern": pg.pattern,
                "lr": pg.optimizer_kwargs.get("lr"),
                "weight_decay": pg.optimizer_kwargs.get("weight_decay"),
                "num_params": len(names),
                "num_elements": sum(
                    p.numel() for n, p in all_named if n in set(names)
                ),
                "examples": [canonical_fqn(n) for n in names[:3]],
            }
        )

    return {
        "total_params": len(all_named),
        "groups": groups,
        "unclaimed": [n for n, _ in all_named if n not in claimed],
        "empty_groups": [g["pattern"] for g in groups if g["num_params"] == 0],
        "misrouted": misrouted,
    }


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def build_mup_agpt_config(
    *,
    dim: int,
    base_dim: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int | None,
    vocab_size: int,
    hidden_dim: int,
    rope_theta: int = 500000,
    base_head_dim: int | None = None,
    attention_scale_form: Literal["multiplier", "absolute"] = "multiplier",
    attention_scale_alpha: float = 1.0,
    **kwargs: Any,
):
    """Build an agpt model config under muP.

    Delegates to ``agpt/__init__.py:_build_agpt_config`` -- the entry point that
    actually threads ``dim`` into every sub-config -- and then applies the two
    init/scale changes muP needs. Top-level ``dim`` is decorative
    (``models/common/decoder.py:253-268`` builds from the sub-configs, each
    carrying its own baked dimensions), so a width sweep that edits ``dim`` on a
    finished config produces N identical models and a coordinate check that
    "confirms" muP while testing nothing.

    Args:
        dim: this rung's width.
        base_dim: the width ``eta`` is tuned at. Must match the ``base_dim``
            passed to :func:`default_mup_adamw` for the same model.
        base_head_dim: head_dim of the base rung. Defaults to this rung's own
            ``dim // n_heads``, which is correct for a fixed-head_dim ladder and
            makes the attention scale reduce exactly to today's
            ``1/sqrt(head_dim)``. Pass it explicitly on a growing-head_dim
            ladder.
        attention_scale_form: see :func:`mup_attention_scale`. The default
            ``"multiplier"`` deviates deliberately from the audit's literal
            ``1/head_dim``; that docstring explains why.
        **kwargs: forwarded to ``_build_agpt_config``.
    """
    from torchtitan.experiments.ezpz.agpt import _build_agpt_config

    if kwargs.get("logit_softcap") is not None:
        raise ValueError(
            "muP + logit_softcap is not implemented: SoftcappedFlexAttention "
            "has no muP subclass, so the softcap path would silently keep the "
            "standard 1/sqrt(head_dim) logit scale while the rest of the model "
            "was reparametrized"
        )
    if kwargs.get("attn_backend", "sdpa") != "sdpa":
        raise ValueError(
            "muP currently overrides the attention scale only on the agpt SDPA "
            f"wrappers; attn_backend={kwargs.get('attn_backend')!r} would "
            "silently keep the standard 1/sqrt(head_dim) scale"
        )
    if dim % n_heads != 0:
        raise ValueError(
            f"dim ({dim}) must be divisible by n_heads ({n_heads})"
        )
    if kwargs.get("enable_weight_tying"):
        # Weight tying is incompatible with muP's separate embedding and
        # unembedding LR groups: with the two sharing one tensor,
        # named_parameters() dedupes and `lm_head.weight` disappears from the
        # model entirely. The unembedding group then matches nothing and the
        # optimizer raises the empty-group ValueError
        # (components/optimizer/optimizer.py:199-203) -- loud, but at optimizer
        # construction, long after the config looked fine. Reject it here
        # instead, where the message can say why. agpt production is untied.
        raise ValueError(
            "muP does not support weight tying: tok_embeddings and lm_head "
            "need separate LR groups, but a tied model has only one tensor, so "
            "named_parameters() dedupes and lm_head.weight vanishes"
        )

    head_dim = dim // n_heads
    resolved_base_head_dim = (
        head_dim if base_head_dim is None else base_head_dim
    )
    scale = mup_attention_scale(
        head_dim,
        resolved_base_head_dim,
        form=attention_scale_form,
        alpha=attention_scale_alpha,
    )

    config = _build_agpt_config(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        rope_theta=rope_theta,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        **kwargs,
    )

    # muP change 1: unembedding init d^-1/2 -> d^-1. Paired with the O(1)
    # lm_head LR group in default_mup_adamw; neither half works alone.
    config.lm_head.param_init = mup_output_linear_init(dim)

    # muP change 2: substitute the muP logit scale for the implicit
    # 1/sqrt(head_dim). Per-layer, because each layer carries its own
    # inner_attention config object.
    for layer in config.layers:
        layer.attention.inner_attention = _mup_inner_attention_config(
            attention_scale=scale,
            expected_head_dim=head_dim,
        )

    # muP change 4 (audit): NOTHING ELSE NEEDS AN INIT CHANGE, and it is worth
    # being explicit about why, because "we changed nothing" and "we forgot" look
    # identical in a diff.
    #   tok_embeddings  std=1.0, width-independent           -- already muP
    #   wq/wkv/w1       sqrt(2/(5*dim)) == fan_in^-1/2 x c   -- already muP
    #   wo/w2/w3        same, times the depth factor         -- already muP
    #   norms           ones_                                -- already muP
    # The depth factor base/sqrt(2*(layer_id+1)) is width-independent, so it
    # rescales every rung identically and does not disturb transfer.
    #
    # muP change 5 (audit): two pre-existing quirks, decided deliberately and
    # LEFT AS THEY ARE.
    #
    #   (a) w2's init std is computed from `dim` while its fan_in is
    #       `hidden_dim` (agpt/__init__.py:_depth_init is called with dim, and
    #       make_ffn_config hands the same dict to w2, whose in_features is
    #       hidden_dim -- config_utils.py:300-302). Strict muP wants
    #       fan_in^-1/2 == hidden_dim^-1/2. LEFT AS IS: the error is the
    #       constant sqrt(hidden_dim/dim), which is width-independent exactly
    #       when H/dim is held fixed -- which the ladder below does, to the
    #       exact rational 8/3. A width-independent constant does not break
    #       transfer, and changing it would gratuitously alter the base rung
    #       away from the measured 30B baseline. This is only safe while H/dim
    #       is fixed; a ladder that varies the FFN ratio MUST fix this first.
    #
    #   (b) w3 is depth-scaled although it is the GATE, not a residual output
    #       (FeedForward.forward computes w2(silu(w1(x)) * w3(x)) --
    #       models/common/feed_forward.py:53-54). The depth factor arrives
    #       because make_ffn_config applies one w2w3_param_init to both
    #       (config_utils.py:300-305) and agpt passes _depth_init
    #       (agpt/__init__.py:402). LEFT AS IS: it is shared with upstream
    #       llama3 (models/llama3/__init__.py:106), it is width-independent so
    #       it does not break the coordinate check, and changing it here would
    #       make the muP arm differ from the baseline in a second, non-muP way.
    #       Fix it as its own experiment, not folded into this one.
    return config


# ---------------------------------------------------------------------------
# Width ladder
# ---------------------------------------------------------------------------
#
# No two registered agpt flavors differ in width ALONE -- debugmodel has
# head_dim=16 against production's 128, 2B has H/dim=5.375, 20B has 64 layers
# and H/dim=2.80 -- so muP's coordinate check, which requires a width-only
# sweep, has no existing base to anchor on. These rungs hold everything but
# `dim` fixed at the 30B_olmo2tok values: L=64, head_dim=128, vocab=100352, and
# H/dim = 8/3 exactly.
#
# hidden_dim is computed directly rather than via compute_ffn_hidden_dim, whose
# multiple_of=1024 rounding is what makes 20B's ratio 2.80 instead of 8/3. The
# exact ratio matters for quirk (a) above.
#
# n_kv_heads scales with n_heads to hold the GQA grouping ratio fixed at 6
# (48/8, the production value). The audit flags the alternative -- pinning
# n_kv=8 across rungs -- as an open question; pinning it would make the grouping
# ratio vary 1.5x to 6x across the ladder, i.e. a second thing changing besides
# width, which is exactly what the ladder exists to avoid. Scaling it is the
# conservative choice for a width-only sweep, but note it means kv head COUNT is
# a width variable; if a coordinate check fails only in the attention block,
# this is a suspect.
_MUP_LADDER: dict[str, dict[str, int]] = {
    # name          dim   n_heads n_kv  hidden
    "mup_1536": {"dim": 1536, "n_heads": 12, "n_kv_heads": 2, "hidden_dim": 4096},
    "mup_3072": {"dim": 3072, "n_heads": 24, "n_kv_heads": 4, "hidden_dim": 8192},
    "mup_6144": {"dim": 6144, "n_heads": 48, "n_kv_heads": 8, "hidden_dim": 16384},
}

MUP_BASE_DIM: int = 1536
"""Width the muP learning rate is tuned at -- the bottom rung of the ladder."""

MUP_LADDER_N_LAYERS: int = 64
MUP_LADDER_VOCAB: int = 100352

# A cheap ladder for CPU-only structural work (coordinate-check harness
# development, param-group partition checks). Same invariants -- fixed head_dim,
# fixed GQA ratio, exact H/dim -- at a size that instantiates on a login node in
# seconds. NOT for science: head_dim is 64, not production's 128.
_MUP_TINY_LADDER: dict[str, dict[str, int]] = {
    "mup_tiny_256": {"dim": 256, "n_heads": 4, "n_kv_heads": 1, "hidden_dim": 1024},
    "mup_tiny_512": {"dim": 512, "n_heads": 8, "n_kv_heads": 2, "hidden_dim": 2048},
    "mup_tiny_1024": {"dim": 1024, "n_heads": 16, "n_kv_heads": 4, "hidden_dim": 4096},
}

MUP_TINY_BASE_DIM: int = 256
MUP_TINY_N_LAYERS: int = 6
MUP_TINY_VOCAB: int = 32000


def register_mup_flavors(configs: dict) -> list[str]:
    """Add the muP ladder flavors to ``agpt_configs``. Purely additive.

    Called once from the bottom of ``agpt/__init__.py``. Raises rather than
    overwrite, so this can never silently reparametrize an existing flavor.
    """
    added: list[str] = []
    ladders = (
        (_MUP_LADDER, MUP_BASE_DIM, MUP_LADDER_N_LAYERS, MUP_LADDER_VOCAB),
        (_MUP_TINY_LADDER, MUP_TINY_BASE_DIM, MUP_TINY_N_LAYERS, MUP_TINY_VOCAB),
    )
    for ladder, base_dim, n_layers, vocab in ladders:
        for name, shape in ladder.items():
            if name in configs:
                raise ValueError(
                    f"muP flavor {name!r} would overwrite an existing agpt "
                    "flavor; rename the muP rung rather than shadowing it"
                )
            configs[name] = build_mup_agpt_config(
                base_dim=base_dim,
                n_layers=n_layers,
                vocab_size=vocab,
                **shape,
            )
            added.append(name)
    return added


# ---------------------------------------------------------------------------
# Runnable trainer configs
# ---------------------------------------------------------------------------
#
# These live here rather than in agpt/config_registry.py deliberately, and they
# are still reachable from the CLI: ConfigManager falls back to importing the
# --module path directly when `<module>.config_registry` does not exist
# (torchtitan/config/manager.py:126-139), and then getattr's --config off it.
# So:
#
#   --module torchtitan.experiments.ezpz.agpt.mup --config mup_6144_adamw
#
# Keeping them here means the whole muP port is two files -- this one plus the
# additive import block at the bottom of agpt/__init__.py.
#
# Every builder starts from `agpt(flavor)`, so it inherits the same dataloader,
# scheduler, AC, parallelism and checkpoint defaults as every other agpt config;
# only the model flavor and the optimizer differ.


def _mup_trainer_config(flavor: str, dim: int, base_dim: int, *, lr: float, **kw):
    """Shared body for the muP trainer configs. See :func:`default_mup_adamw`.

    Imported lazily: ``agpt/config_registry.py`` imports ``agpt/__init__.py``,
    which imports this module at its bottom, so a module-level import here would
    close the cycle.
    """
    from torchtitan.experiments.ezpz.agpt.config_registry import agpt

    cfg = agpt(flavor, hf_assets_path="./assets/hf/OLMo-2-1124-7B")
    cfg.optimizer = default_mup_adamw(lr, dim=dim, base_dim=base_dim, **kw)
    return cfg


def mup_1536_adamw():
    """muP base rung, dim=1536. This is the width `eta` is TUNED at (m=1).

    Stage 4 sweeps LR here, then applies the muP rule to predict the optimum at
    mup_6144. At m=1 all four groups sit at the same LR, so this config is the
    control: it differs from standard parametrization only in the lm_head init.
    """
    return _mup_trainer_config("mup_1536", 1536, MUP_BASE_DIM, lr=8e-4)


def mup_3072_adamw():
    """muP middle rung, dim=3072, m=2. Hidden matrices at eta/2."""
    return _mup_trainer_config("mup_3072", 3072, MUP_BASE_DIM, lr=8e-4)


def mup_6144_adamw():
    """muP target rung, dim=6144, m=4. Hidden matrices at eta/4.

    Same geometry as the production ``30b_olmo2tok`` flavor -- verified to have
    an identical parameter FQN set -- so a converged run here is comparable to
    the measured AdamW baseline, with the parametrization as the only variable.

    NOT a resume target for a standard-parametrization checkpoint: the weights
    would load (the FQNs match) but they were trained under a different
    parametrization, and the optimizer's param-group layout differs. Fresh
    chain, own checkpoint folder.
    """
    return _mup_trainer_config("mup_6144", 6144, MUP_BASE_DIM, lr=8e-4)


def mup_6144_adamw_independent_wd():
    """mup_6144 with width-independent weight decay (u-muP's decoupled AdamW).

    PyTorch's AdamW decays with ``param.mul_(1 - lr * weight_decay)``
    (``torch/optim/adam.py:419``), so putting the hidden group on ``eta/m``
    silently scales its effective decay by ``1/m`` as well. This arm scales
    ``weight_decay`` by ``m`` to compensate.

    Per the audit's section 5.1, weight decay is one of three features Lingle
    (arXiv 2404.05728) marks as BREAKING muP transfer on a recipe like this one,
    and arXiv 2510.19093 argues decay matters more than muP for cross-width
    dynamics. Expect the plain arm's coordinate check to fail first; this is the
    first thing to try when it does.
    """
    return _mup_trainer_config(
        "mup_6144", 6144, MUP_BASE_DIM, lr=8e-4, independent_weight_decay=True
    )


def mup_tiny_256_adamw():
    """CPU-sized muP base rung for harness development. Not for science."""
    return _mup_trainer_config("mup_tiny_256", 256, MUP_TINY_BASE_DIM, lr=8e-4)


def mup_tiny_512_adamw():
    """CPU-sized muP rung, m=2. Not for science."""
    return _mup_trainer_config("mup_tiny_512", 512, MUP_TINY_BASE_DIM, lr=8e-4)


def mup_tiny_1024_adamw():
    """CPU-sized muP rung, m=4. Not for science."""
    return _mup_trainer_config("mup_tiny_1024", 1024, MUP_TINY_BASE_DIM, lr=8e-4)
