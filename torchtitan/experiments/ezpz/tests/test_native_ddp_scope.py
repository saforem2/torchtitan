# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
from contextlib import contextmanager, nullcontext
from typing import Any, cast

import pytest
import torch
from torch.multiprocessing.spawn import spawn

from torchtitan.components.loss import CrossEntropyLoss, MSELoss
from torchtitan.config import CompileConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.ezpz.agpt import parallelize as agpt_parallelize
from torchtitan.experiments.ezpz.config import EzpzParallelismConfig
from torchtitan.experiments.ezpz.native_ddp import (
    native_ddp_autocast_context,
    validate_native_ddp,
    wrap_native_ddp,
    wrap_native_ddp_loss,
)


def _parallel_dims(*, pp: int = 1) -> ParallelDims:
    return ParallelDims(
        dp_replicate=2,
        dp_shard=1,
        cp=1,
        tp=1,
        pp=pp,
        ep=1,
        world_size=2 * pp,
    )


def _parallelism(**overrides: object) -> EzpzParallelismConfig:
    values = dict(
        enable_data_parallel_native_ddp=True,
        data_parallel_replicate_degree=2,
        data_parallel_shard_degree=1,
    )
    values.update(overrides)
    return EzpzParallelismConfig(**values)


def _validate(*, pp: int = 1, policy: str = "autocast", loss_fn=None) -> None:
    validate_native_ddp(
        model_name="ezpz.agpt",
        parallel_dims=_parallel_dims(pp=pp),
        training=TrainingConfig(
            dtype="float32",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        ),
        parallelism=_parallelism(native_ddp_compute_policy=policy),
        loss_fn=loss_fn or CrossEntropyLoss(CrossEntropyLoss.Config()),
        gradient_accumulation_steps=1,
        fault_tolerance_enabled=False,
        create_seed_checkpoint=False,
        optimizer_has_param_groups=False,
    )


def test_native_ddp_rejects_pipeline_parallelism() -> None:
    with pytest.raises(ValueError, match="does not support pipeline parallelism"):
        _validate(pp=2)


def test_native_ddp_rejects_non_autocast_policy() -> None:
    with pytest.raises(ValueError, match="only supports the autocast compute policy"):
        _validate(policy="ddp_mixed_precision")


def test_native_ddp_validates_original_loss_before_wrapping() -> None:
    loss = MSELoss(MSELoss.Config())
    with pytest.raises(ValueError, match="standard CrossEntropyLoss"):
        _validate(loss_fn=loss)


def test_native_ddp_loss_wrapper_preserves_type_check_target_and_scales_gradient() -> None:
    original = CrossEntropyLoss(CrossEntropyLoss.Config())
    wrapped = wrap_native_ddp_loss(original, dp_degree=2)
    logits = torch.tensor([[2.0, -1.0]], requires_grad=True)
    labels = torch.tensor([0])

    reference, _ = original(logits, labels)
    expected_grad = torch.autograd.grad(reference, logits, retain_graph=True)[0] * 2
    actual, aux = wrapped(logits, labels)
    actual.backward()

    assert aux == {}
    assert torch.equal(actual.detach(), reference.detach())
    torch.testing.assert_close(logits.grad, expected_grad)


def test_native_ddp_autocast_wraps_existing_context() -> None:
    events: list[str] = []

    @contextmanager
    def base_context():
        events.append("enter")
        yield
        events.append("exit")

    context = native_ddp_autocast_context(base_context, "cpu")
    with context():
        assert events == ["enter"]
        assert torch.is_autocast_enabled("cpu")
        assert torch.get_autocast_dtype("cpu") == torch.bfloat16
    assert events == ["enter", "exit"]


def test_agpt_parallelization_skips_fsdp_for_native_ddp(monkeypatch) -> None:
    class Model:
        def __init__(self) -> None:
            self.parallelized = False

        def parallelize(self, _parallel_dims) -> None:
            self.parallelized = True

    model = Model()
    monkeypatch.setattr(
        agpt_parallelize,
        "apply_fsdp",
        lambda *_args, **_kwargs: pytest.fail("native DDP must not install FSDP"),
    )
    result = agpt_parallelize.parallelize_llama(
        cast(Any, model),
        parallel_dims=_parallel_dims(),
        training=TrainingConfig(max_context_length=8),
        parallelism=_parallelism(),
        compile_config=None,
        ac_config=None,
        dump_folder=".",
    )

    assert result is model
    assert model.parallelized


def _run_two_rank_equivalence(rank: int, rendezvous: str, result_file: str) -> None:
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2
    )
    try:
        torch.manual_seed(17)
        reference = torch.nn.Linear(4, 3)
        distributed = torch.nn.Linear(4, 3)
        distributed.load_state_dict(reference.state_dict())
        mesh = torch.distributed.device_mesh.init_device_mesh(
            "cpu", (2,), mesh_dim_names=("dp_replicate",)
        )
        ddp = wrap_native_ddp(distributed, mesh, bucket_cap_mb=1.0)

        inputs = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 8
        labels = torch.tensor([0, 1, 2, 1])
        local_inputs = inputs[rank * 2 : (rank + 1) * 2]
        local_labels = labels[rank * 2 : (rank + 1) * 2]
        context = native_ddp_autocast_context(nullcontext, "cpu")

        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.1)
        loss_fn = wrap_native_ddp_loss(
            CrossEntropyLoss(CrossEntropyLoss.Config()), dp_degree=2
        )
        with context():
            local_loss, _ = loss_fn(ddp(local_inputs), local_labels, torch.tensor(4))
        local_loss.backward()
        optimizer.step()

        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            reference_loss, _ = CrossEntropyLoss(CrossEntropyLoss.Config())(
                reference(inputs), labels, torch.tensor(4)
            )
        reference_loss.backward()
        reference_optimizer.step()

        for actual, expected in zip(
            ddp.module.parameters(), reference.parameters(), strict=True
        ):
            # BF16 matmul accumulation can differ slightly when the global
            # batch is split into two rank-local GEMMs.
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-4)
        if rank == 0:
            torch.save(ddp.module.state_dict(), result_file)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="Gloo backend is unavailable"
)
def test_native_ddp_cpu_gloo_two_rank_step_matches_single_process() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        rendezvous = f"file://{tmpdir}/rendezvous"
        result_file = os.path.join(tmpdir, "result.pt")
        spawn(
            _run_two_rank_equivalence,
            args=(rendezvous, result_file),
            nprocs=2,
            join=True,
        )
        assert os.path.exists(result_file)
