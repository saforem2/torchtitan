# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from torchtitan.experiments.ezpz.moe import moe_configs
from torchtitan.experiments.ezpz.moe.state_dict_adapter import moeStateDictAdapter


def _adapter() -> moeStateDictAdapter:
    return moeStateDictAdapter(moe_configs["debugmodel"](), hf_assets_path=None)


def test_to_hf_splits_grouped_linear_w13_and_w2_by_expert():
    adapter = _adapter()
    w13 = torch.arange(8 * 2 * 3 * 4).reshape(8, 2, 3, 4)
    w2 = torch.arange(8 * 4 * 3).reshape(8, 4, 3)

    hf_state = adapter.to_hf(
        {
            "layers.1.moe.routed_experts.w13.weight": w13,
            "layers.1.moe.routed_experts.w2.weight": w2,
        }
    )

    assert len(hf_state) == 8 * 3
    for expert in range(8):
        prefix = f"model.layers.1.mlp.experts.{expert}"
        torch.testing.assert_close(
            hf_state[f"{prefix}.gate_proj.weight"], w13[expert, 0]
        )
        torch.testing.assert_close(hf_state[f"{prefix}.up_proj.weight"], w13[expert, 1])
        torch.testing.assert_close(hf_state[f"{prefix}.down_proj.weight"], w2[expert])


def test_hf_experts_roundtrip_to_grouped_linear_layout():
    adapter = _adapter()
    native_state = {
        "layers.1.moe.routed_experts.w13.weight": torch.arange(8 * 2 * 3 * 4).reshape(
            8, 2, 3, 4
        ),
        "layers.1.moe.routed_experts.w2.weight": torch.arange(8 * 4 * 3).reshape(
            8, 4, 3
        ),
    }

    restored = adapter.from_hf(adapter.to_hf(native_state))

    assert restored.keys() == native_state.keys()
    for key, value in native_state.items():
        torch.testing.assert_close(restored[key], value)


def test_grouped_linear_dtensors_roundtrip_with_expert_sharding():
    adapter = _adapter()
    with tempfile.TemporaryDirectory() as directory:
        owns_process_group = not dist.is_initialized()
        if owns_process_group:
            dist.init_process_group(
                "gloo",
                init_method=f"file://{directory}/rendezvous",
                rank=0,
                world_size=1,
            )
        try:
            mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("ep",))
            for placement in (Shard(0), Shard(2)):
                native_state = {
                    "layers.1.moe.routed_experts.w13.weight": DTensor.from_local(
                        torch.arange(8 * 2 * 3 * 4).reshape(8, 2, 3, 4),
                        mesh,
                        (placement,),
                        run_check=False,
                    ),
                    "layers.1.moe.routed_experts.w2.weight": DTensor.from_local(
                        torch.arange(8 * 4 * 3).reshape(8, 4, 3),
                        mesh,
                        (placement,),
                        run_check=False,
                    ),
                }

                restored = adapter.from_hf(adapter.to_hf(native_state))

                for key, value in native_state.items():
                    assert isinstance(restored[key], DTensor)
                    assert restored[key].placements == value.placements
                    torch.testing.assert_close(restored[key], value, rtol=0, atol=0)
        finally:
            if owns_process_group:
                dist.destroy_process_group()
