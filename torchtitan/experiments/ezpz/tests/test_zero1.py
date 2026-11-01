# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.multiprocessing.spawn import spawn

from torchtitan.experiments.ezpz.native_ddp import wrap_native_ddp
from torchtitan.experiments.ezpz.optimizer.containers import default_sophiag
from torchtitan.experiments.ezpz.optimizer.sophia import SophiaG
from torchtitan.experiments.ezpz.zero1 import flatten_parameters_, install_zero1

WORLD = 2
STEPS = 3


def _model() -> nn.Module:
    torch.manual_seed(23)
    # 35 + 7 + 21 = 63 elements: padded to 64, and the shard boundary at 32
    # falls inside the first weight.
    return nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 3, bias=False))


def _batch(step: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(100 + step)
    return (
        torch.randn(8, 5, generator=generator),
        torch.randn(8, 3, generator=generator),
    )


def _loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Normalized by the global sample count, like the trainer's token count.
    return ((model(x) - y) ** 2).sum() / 8


def _sophia_config():
    return default_sophiag(lr=1e-2, betas=(0.9, 0.95), rho=0.01)


def test_flatten_parameters_rebinds_views_and_pads() -> None:
    model = _model()
    expected = {name: p.detach().clone() for name, p in model.named_parameters()}
    buffer = flatten_parameters_(model, world_size=WORLD, rank=1)
    assert buffer.numel == 63
    assert buffer.flat.numel() == 64
    assert buffer.shard_bounds == (32, 64)
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, expected[name], rtol=0, atol=0)
        assert param.untyped_storage().data_ptr() == buffer.flat.untyped_storage().data_ptr()
    # Rank 1 owns the tail of weight 0 (3 elements), bias 0, and weight 2.
    segments = [(lo, hi, dst) for _, lo, hi, dst in buffer.local_segments()]
    assert segments == [(32, 35, 0), (0, 7, 3), (0, 21, 10)]


def test_install_zero1_rejects_coupled_optimizer() -> None:
    model = _model()
    buffer = flatten_parameters_(model, world_size=1, rank=0)
    optimizer = torch.optim.Adagrad(model.parameters())
    with pytest.raises(ValueError, match="elementwise optimizer"):
        install_zero1(optimizer, buffer, process_group=None)


def _worker(rank: int, rendezvous: str, workdir: str, dtype: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=WORLD)
    try:
        reference = _model()
        model = _model()
        buffer = flatten_parameters_(model, world_size=WORLD, rank=rank)
        mesh = dist.device_mesh.init_device_mesh(
            "cpu", (WORLD,), mesh_dim_names=("dp_replicate",)
        )
        ddp = wrap_native_ddp(model, mesh, bucket_cap_mb=1e-4)
        container = _sophia_config().build(model_parts=[model])
        state = install_zero1(
            container,
            buffer,
            process_group=mesh.get_group(),
            allgather_dtype=dtype,
            module=model,
        )
        optimizer = container.optimizers[0]
        assert isinstance(optimizer, SophiaG)

        ref_master = copy.deepcopy(reference)
        ref_optimizer = SophiaG(ref_master.parameters(), lr=1e-2, betas=(0.9, 0.95), rho=0.01)
        for step in range(STEPS):
            x, y = _batch(step)
            local_x, local_y = x[rank * 4 : (rank + 1) * 4], y[rank * 4 : (rank + 1) * 4]
            for micro in range(2):
                ddp.set_requires_all_reduce(micro == 1)
                sl = slice(micro * 2, (micro + 1) * 2)
                # DDP averages over ranks; the loss is already globally normalized.
                (_loss(ddp, local_x[sl], local_y[sl]) * WORLD).backward()
            optimizer.update_hessian()
            container.step()
            container.zero_grad()
            assert all(p.grad is None for p in model.parameters())

            # Reference: one process, full batch, FP32 master with the same
            # compute-weight rounding as the requested mode.
            with torch.no_grad():
                for compute, master in zip(reference.parameters(), ref_master.parameters()):
                    value = master if dtype == "float32" else master.bfloat16().float()
                    compute.copy_(value)
            reference.zero_grad()
            _loss(reference, x, y).backward()
            for compute, master in zip(reference.parameters(), ref_master.parameters()):
                master.grad = compute.grad.clone()
            ref_optimizer.update_hessian()
            ref_optimizer.step()

        assert state.steps == STEPS and state.allgathers == STEPS
        flat_ref = torch.cat([p.detach().reshape(-1) for p in ref_master.parameters()])
        start, end = buffer.shard_bounds
        stop = min(end, buffer.numel)
        torch.testing.assert_close(
            state.master.detach()[: stop - start], flat_ref[start:stop], rtol=1e-5, atol=1e-6
        )
        model_flat = buffer.flat[: buffer.numel]
        if dtype == "bfloat16":
            assert torch.equal(model_flat, model_flat.bfloat16().float())
            torch.testing.assert_close(
                model_flat, flat_ref.bfloat16().float(), rtol=1e-2, atol=1e-3
            )
        else:
            torch.testing.assert_close(model_flat, flat_ref, rtol=1e-5, atol=1e-6)
        gathered = [torch.empty_like(buffer.flat) for _ in range(WORLD)]
        dist.all_gather(gathered, buffer.flat)
        assert torch.equal(gathered[0], gathered[1]), "replicas diverged"
        # Only this rank's shard carries optimizer state.
        assert len(optimizer.state) == 1
        (exp_avg,) = [s["exp_avg"] for s in optimizer.state.values()]
        assert exp_avg.numel() == buffer.shard_numel

        # DCP round trip of the sharded optimizer state.
        checkpoint = os.path.join(workdir, f"ckpt-{dtype}")
        saved = {k: v.clone() if torch.is_tensor(v) else v for k, v in container.state_dict().items()}
        assert any(f"rank{rank:05d}" in key for key in saved)
        dcp.save(container.state_dict(), checkpoint_id=checkpoint)
        exp_avg.zero_()
        loaded = container.state_dict()
        dcp.load(loaded, checkpoint_id=checkpoint)
        container.load_state_dict(loaded)
        (restored,) = [s["exp_avg"] for s in optimizer.state.values()]
        key = next(k for k in saved if k.endswith("exp_avg"))
        torch.testing.assert_close(restored, saved[key], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo backend is unavailable")
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_zero1_native_ddp_matches_unsharded_reference(dtype: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        spawn(
            _worker,
            args=(f"file://{tmpdir}/rendezvous", tmpdir, dtype),
            nprocs=WORLD,
            join=True,
        )
