#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for agpt muP (``agpt/mup.py``).

Lives beside the code rather than in ``ezpz/tests/`` because the whole muP port
is deliberately confined to the agpt package.

Run::

    python -m pytest torchtitan/experiments/ezpz/agpt/test_mup.py -x
    python torchtitan/experiments/ezpz/agpt/test_mup.py     # no pytest needed

CPU only, no allocation, a few seconds. The single most important test here is
``test_non_mup_flavors_are_byte_identical``: muP is opt-in, and the constraint
that no existing config changes is the one worth a permanent guard, because a
violation would be silent -- every affected run would simply train slightly
differently and still report success.
"""

import math

import torch

from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.components.optimizer.optimizer import OptimizersContainer
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.ezpz.agpt import agpt_configs
from torchtitan.experiments.ezpz.agpt import mup as M

ETA = 8e-4

# Flavors that existed before muP, with the init quantities that must not move.
# Kept as literals rather than recomputed so this test fails if the FORMULA
# changes too, not merely if the call site does.
_PRE_MUP_LM_HEAD_STD = {
    "30B_olmo2tok": 6144**-0.5,
    "2B": 2048**-0.5,
    "20B": 5120**-0.5,
    "debugmodel": 256**-0.5,
}


def _std(param_init) -> float:
    return param_init["weight"].keywords["std"]


def test_non_mup_flavors_are_byte_identical():
    """muP must not perturb any pre-existing flavor. THE critical invariant."""
    for flavor, expected in _PRE_MUP_LM_HEAD_STD.items():
        cfg = agpt_configs[flavor]
        assert _std(cfg.lm_head.param_init) == expected, flavor
        # standard SDPA wrapper, and no muP scale field anywhere on it
        inner = cfg.layers[0].attention.inner_attention
        assert "Mup" not in type(inner).__qualname__, flavor
        assert not hasattr(inner, "attention_scale"), flavor
        # hidden inits still Megatron-DeepSpeed sqrt(2/(5d))
        dim = cfg.layers[0].attention.dim
        base = math.sqrt(2.0 / (5 * dim))
        assert _std(cfg.layers[0].feed_forward.w1.param_init) == base, flavor


def test_ladder_varies_width_only():
    rungs = [agpt_configs[n] for n in ("mup_1536", "mup_3072", "mup_6144")]
    geo = [
        (
            c.layers[0].attention.dim // c.layers[0].attention.n_heads,  # head_dim
            len(c.layers),
            c.vocab_size,
            c.layers[0].attention.n_heads // c.layers[0].attention.n_kv_heads,
            c.layers[0].feed_forward.w1.out_features / c.layers[0].attention.dim,
        )
        for c in rungs
    ]
    assert len(set(geo)) == 1, f"ladder varies more than width: {geo}"
    assert geo[0][0] == 128 and geo[0][4] == 8 / 3
    dims = [c.layers[0].attention.dim for c in rungs]
    assert dims == [1536, 3072, 6144]


def test_ladder_rungs_are_genuinely_different_models():
    """Guards the decorative-``dim`` trap (mup README 4.2)."""
    counts = []
    for name in ("mup_tiny_256", "mup_tiny_512", "mup_tiny_1024"):
        with torch.device("meta"):
            m = agpt_configs[name].build()
        counts.append(sum(p.numel() for p in m.parameters()))
    assert len(set(counts)) == 3, counts
    assert counts == sorted(counts)


def test_mup_lm_head_init_is_d_inverse():
    for name in ("mup_1536", "mup_3072", "mup_6144"):
        cfg = agpt_configs[name]
        dim = cfg.layers[0].attention.dim
        assert _std(cfg.lm_head.param_init) == dim**-1.0, name


def test_mup_top_rung_matches_production_geometry():
    """mup_6144 must be the production 30B model, or stage 5 compares nothing."""
    with torch.device("meta"):
        a = agpt_configs["mup_6144"].build()
        b = agpt_configs["30B_olmo2tok"].build()
    assert {canonical_fqn(n) for n, _ in a.named_parameters()} == {
        canonical_fqn(n) for n, _ in b.named_parameters()
    }


def test_attention_scale_reduces_to_sp_at_base_width():
    assert M.mup_attention_scale(128, 128) == 128**-0.5
    assert M.mup_attention_scale(128, 128, form="absolute") == 1 / 128
    # and it really scales as 1/head_dim on a growing-head_dim ladder
    assert math.isclose(
        M.mup_attention_scale(64, 128) / M.mup_attention_scale(128, 128), 2.0
    )


def test_attention_scale_reaches_the_kernel():
    import torch.nn.functional as F

    torch.manual_seed(0)
    a = M.MupScaledDotProductAttention.Config(
        attention_scale=0.5, expected_head_dim=8
    ).build()
    q, k, v = (torch.randn(2, 5, 4, 8) for _ in range(3))
    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        scale=0.5, is_causal=True,
    ).transpose(1, 2)
    assert torch.allclose(a(q, k, v), ref, atol=1e-6)


def test_attention_rejects_conflicting_scale_and_wrong_head_dim():
    a = M.MupScaledDotProductAttention.Config(
        attention_scale=0.5, expected_head_dim=8
    ).build()
    q, k, v = (torch.randn(2, 5, 4, 8) for _ in range(3))
    for bad in (
        lambda: a(q, k, v, scale=0.123),
        lambda: a(*(torch.randn(2, 5, 4, 16) for _ in range(3))),
    ):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_lr_groups_scale_hidden_by_one_over_m():
    for dim, m in ((1536, 1.0), (3072, 2.0), (6144, 4.0)):
        oc = M.default_mup_adamw(ETA, dim=dim, base_dim=M.MUP_BASE_DIM)
        lrs = [pg.optimizer_kwargs["lr"] for pg in oc.param_groups]
        assert len(lrs) == 4
        assert lrs[:3] == [ETA, ETA, ETA]
        assert lrs[3] == ETA / m


def test_param_groups_partition_cleanly_with_and_without_ac():
    """The AC trap (mup README 3.2 trap 1): FQNs gain a wrapper segment."""
    oc = M.default_mup_adamw(ETA, dim=512, base_dim=M.MUP_TINY_BASE_DIM)
    reports = []
    for ac in (False, True):
        with torch.device("meta"):
            model = agpt_configs["mup_tiny_512"].build()
        if ac:
            FullAC.Config().build().apply(model)
            assert any(
                "_checkpoint_wrapped_module" in n for n, _ in model.named_parameters()
            ), "FullAC did not wrap anything; the trap is untested"
        rep = M.summarize_mup_param_groups(model, oc.param_groups)
        assert not rep["unclaimed"], rep["unclaimed"][:5]
        assert not rep["empty_groups"], rep["empty_groups"]
        assert not rep["misrouted"], rep["misrouted"][:5]
        assert sum(g["num_params"] for g in rep["groups"]) == rep["total_params"]
        reports.append([(g["role"], g["num_params"]) for g in rep["groups"]])
    assert reports[0] == reports[1], reports


def test_prefix_anchored_pattern_would_break_under_ac():
    """Pins the reason the patterns are leaf-anchored. Control, not a feature."""
    import re

    with torch.device("meta"):
        model = agpt_configs["mup_tiny_512"].build()
    FullAC.Config().build().apply(model)
    rx = re.compile(r"^layers\.\d+\.attention\.")
    assert not any(rx.search(n) for n, _ in model.named_parameters())


def test_param_groups_cover_qk_norm_and_fused_qkv():
    """Both branches change the FQN set, so both can break the partition."""
    for kw in ({"qk_norm": True}, {"fuse_qkv": True}, {"qk_norm": True, "fuse_qkv": True}):
        cfg = M.build_mup_agpt_config(
            dim=512, base_dim=256, n_layers=3, n_heads=8, n_kv_heads=2,
            vocab_size=1000, hidden_dim=2048, **kw,
        )
        with torch.device("meta"):
            model = cfg.build()
        FullAC.Config().build().apply(model)
        rep = M.summarize_mup_param_groups(
            model, M.default_mup_adamw(ETA, dim=512, base_dim=256).param_groups
        )
        assert not rep["unclaimed"] and not rep["empty_groups"] and not rep["misrouted"], kw


def test_real_optimizer_moves_hidden_group_by_one_over_m():
    """End-to-end: the groups must drive a real AdamW step, not just exist."""
    torch.manual_seed(0)
    model = agpt_configs["mup_tiny_512"].build()
    model.init_states()
    FullAC.Config().build().apply(model)
    oc = M.default_mup_adamw(ETA, dim=512, base_dim=M.MUP_TINY_BASE_DIM)
    oc.implementation = "foreach"
    opt = OptimizersContainer(oc, model_parts=[model])
    named = dict(model.named_parameters())
    emb = named["tok_embeddings.weight"]
    hid = named[next(n for n in named if "feed_forward.w1" in n)]
    e0, h0 = emb.detach().clone(), hid.detach().clone()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    de = (emb.detach() - e0).abs().mean().item()
    dh = (hid.detach() - h0).abs().mean().item()
    assert math.isclose(de / dh, 2.0, rel_tol=1e-3), de / dh


def test_scheduler_preserves_group_ratio():
    from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer

    torch.manual_seed(0)
    model = agpt_configs["mup_tiny_512"].build()
    model.init_states()
    oc = M.default_mup_adamw(ETA, dim=512, base_dim=M.MUP_TINY_BASE_DIM)
    oc.implementation = "foreach"
    opt = OptimizersContainer(oc, model_parts=[model])
    sched = LRSchedulersContainer.Config(
        warmup_steps=2, decay_ratio=0.8, decay_type="linear", min_lr_factor=0.0
    ).build(optimizers=opt, training_steps=20)
    seen = []
    for _ in range(6):
        lrs = [g["lr"] for o in opt.optimizers for g in o.param_groups]
        seen.append(lrs)
        assert math.isclose(lrs[0] / lrs[3], 2.0, rel_tol=1e-9), lrs
        sched.step()
    assert len({round(s[0], 12) for s in seen}) > 1, "scheduler never moved"


def test_weight_tying_is_rejected():
    import dataclasses

    # the underlying failure, demonstrated
    tied = dataclasses.replace(agpt_configs["mup_tiny_256"], enable_weight_tying=True)
    with torch.device("meta"):
        m = tied.build()
    assert "lm_head.weight" not in {n for n, _ in m.named_parameters()}
    # and the guard that keeps it from getting that far
    try:
        M.build_mup_agpt_config(
            dim=256, base_dim=256, n_layers=2, n_heads=4, n_kv_heads=1,
            vocab_size=1000, hidden_dim=512, enable_weight_tying=True,
        )
    except ValueError as e:
        assert "weight tying" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_unsupported_attention_paths_are_rejected():
    for kw in ({"logit_softcap": 30.0}, {"attn_backend": "flex"}):
        try:
            M.build_mup_agpt_config(
                dim=256, base_dim=256, n_layers=2, n_heads=4, n_kv_heads=1,
                vocab_size=1000, hidden_dim=512, **kw,
            )
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kw}")


def test_independent_weight_decay_makes_lr_times_wd_width_invariant():
    a = M.default_mup_adamw(ETA, dim=6144, base_dim=1536)
    b = M.default_mup_adamw(ETA, dim=6144, base_dim=1536, independent_weight_decay=True)
    ref = a.param_groups[0].optimizer_kwargs
    ref_prod = ref["lr"] * ref["weight_decay"]
    ah, bh = a.param_groups[3].optimizer_kwargs, b.param_groups[3].optimizer_kwargs
    assert math.isclose(ah["lr"] * ah["weight_decay"], ref_prod / 4)
    assert math.isclose(bh["lr"] * bh["weight_decay"], ref_prod)


def test_trainer_configs_build():
    for name in ("mup_1536_adamw", "mup_3072_adamw", "mup_6144_adamw",
                 "mup_6144_adamw_independent_wd"):
        cfg = getattr(M, name)()
        assert len(cfg.optimizer.param_groups) == 4, name


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
