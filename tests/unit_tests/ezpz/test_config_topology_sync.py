# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import inspect
from types import SimpleNamespace
from unittest.mock import ANY

import pytest
from torchtitan.config.configs import CommConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import DistributedTopology, ParallelDims


@pytest.mark.parametrize(
    "module_name",
    [
        "torchtitan.experiments.ezpz.config",
        "torchtitan.experiments.ezpz.native_ddp",
        "torchtitan.experiments.ezpz.agpt.parallelize",
        "torchtitan.experiments.ezpz.moe.parallelize",
    ],
)
def test_parallelism_config_imports_follow_upstream_move(module_name: str) -> None:
    importlib.import_module(module_name)


@pytest.mark.parametrize(
    "relative_path",
    [
        "torchtitan/experiments/ezpz/rl/alphabet_sort_agpt/config_registry.py",
        "torchtitan/experiments/ezpz/rl/reason_agpt/config_registry.py",
    ],
)
def test_rl_registry_parallelism_imports_follow_upstream_move(
    relative_path: str,
) -> None:
    from pathlib import Path

    source = Path(relative_path).read_text()
    assert "from torchtitan.config.parallelism import ParallelismConfig" in source
    assert (
        "from torchtitan.config import CompileConfig, ParallelismConfig" not in source
    )


def test_ezpz_parallelism_config_constructs_with_core_fields() -> None:
    from torchtitan.experiments.ezpz.config import EzpzParallelismConfig

    config = EzpzParallelismConfig(
        data_parallel_replicate_degree=2,
        data_parallel_shard_degree=2,
        tensor_parallel_degree=2,
        enable_data_parallel_native_ddp=True,
    )

    assert isinstance(config, ParallelismConfig)
    assert config.tensor_parallel_degree == 2
    assert config.enable_data_parallel_native_ddp


def test_parallel_dims_from_topology_matches_direct_construction() -> None:
    from torchtitan.experiments.ezpz.config import EzpzParallelismConfig

    config = EzpzParallelismConfig(
        data_parallel_replicate_degree=2,
        data_parallel_shard_degree=-1,
        context_parallel_degree=1,
        tensor_parallel_degree=2,
        pipeline_parallel_degree=2,
        expert_parallel_degree=1,
        enable_sequence_parallel=False,
    )
    topology = DistributedTopology(world_size=16)

    actual = ParallelDims.from_config(config, topology)
    expected = ParallelDims(
        dp_replicate=2,
        dp_shard=2,
        cp=1,
        tp=2,
        pp=2,
        ep=1,
        world_size=16,
        enable_sequence_parallel=False,
    )

    assert actual == expected


def test_legacy_comm_mode_translates_to_backend() -> None:
    from torchtitan.experiments.ezpz.train import _translate_legacy_args

    assert _translate_legacy_args(["--comm.mode", "fake_backend"]) == [
        "--comm.backend",
        "fake",
    ]
    assert _translate_legacy_args(["--comm.mode=fake"]) == [
        "--comm.backend",
        "fake",
    ]


def test_removed_local_tensor_comm_mode_fails_closed() -> None:
    from torchtitan.experiments.ezpz.train import _translate_legacy_args

    with pytest.raises(ValueError, match="local_tensor.*removed"):
        _translate_legacy_args(["--comm.mode", "local_tensor"])


def test_validator_runs_a_real_validation_pass(monkeypatch) -> None:
    """Execute validate() rather than inspecting its source. The SPMD
    context path is stubbed with a no-op to avoid requiring a distributed
    init; the important behavior (preprocess -> forward -> loss) runs.
    """
    import torch
    import torchtitan.experiments.ezpz.validator as validator_module
    from torchtitan.experiments.ezpz.validator import EzpzValidator

    monkeypatch.setattr(validator_module.utils, "device_type", "cpu")

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Linear(4, 4)

        def preprocess_inputs(self, batch, *, parallel_dims, parallelism):
            labels = batch.pop("labels")
            return batch["input"], labels, {}

        def forward(self, inputs, **kwargs):
            return self.weight(inputs)

    class _Metrics:
        ntokens_since_last_log = 0
        logged: list[tuple[float, int]] = []

        def log_validation(self, loss, step):
            type(self).logged.append((float(loss), step))

    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=1,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        world_size=1,
        enable_sequence_parallel=False,
    )

    validator = object.__new__(EzpzValidator)
    validator.parallel_dims = parallel_dims
    validator.parallelism = ParallelismConfig()
    validator.metrics_processor = _Metrics()
    validator.config = SimpleNamespace(steps=1)
    validator.loss_fn = lambda predictions, labels: (predictions.float().sum(), {})
    batch = {
        "input": torch.ones(2, 4),
        "labels": torch.zeros(2, dtype=torch.long),
    }
    validator._cached_dataloader = [batch]
    validator._get_validation_dataloader = lambda: validator._cached_dataloader
    import torchtitan.distributed.utils as du

    monkeypatch.setattr(
        du,
        "get_spmd_context",
        lambda **kw: __import__("contextlib").nullcontext(),
    )
    model = _Model()
    validator.validate([model], step=3)
    assert _Metrics.logged and _Metrics.logged[-1][1] == 3
    assert model.training


def test_xpu_init_wrapper_preserves_signature_keywords_and_return(monkeypatch) -> None:
    import torchtitan.distributed.utils as dist_utils
    from torchtitan.experiments.ezpz.rl import xpu_overrides

    calls = []
    topology = DistributedTopology(world_size=8)

    def original(
        comm_config,
        enable_cpu_backend=False,
        base_folder="",
        ranks=None,
        *,
        pipeline_parallel_degree=1,
    ) -> DistributedTopology:
        calls.append(
            (
                comm_config,
                enable_cpu_backend,
                base_folder,
                ranks,
                pipeline_parallel_degree,
            )
        )
        return topology

    monkeypatch.setattr(xpu_overrides.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(dist_utils, "init_distributed", original)
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("RANK", "7")

    xpu_overrides.patch_init_distributed_for_xpu()
    patched = dist_utils.init_distributed
    result = patched(
        CommConfig(),
        base_folder="dump",
        ranks=[7],
        pipeline_parallel_degree=4,
    )

    assert result is topology
    assert calls == [(ANY, True, "dump", [7], 4)]
    assert os_environ_subset("PALS_LOCAL_RANKID", "PALS_RANKID") == ("3", "7")
    signature = inspect.signature(patched)
    assert (
        signature.parameters["pipeline_parallel_degree"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert signature.return_annotation is DistributedTopology


def os_environ_subset(*keys: str) -> tuple[str | None, ...]:
    import os

    return tuple(os.environ.get(key) for key in keys)
