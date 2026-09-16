import torch

from torchtitan.experiments.ezpz.moe import moe_configs
from torchtitan.experiments.ezpz.moe.activation_checkpoint import (
    AuroraMoeSelectiveAC,
)
from torchtitan.experiments.ezpz.moe.aurora import AuroraMoE, AuroraRoutedExperts
from torchtitan.experiments.ezpz.moe.config_registry import (
    agpt_12b2a_50k_moe_aurora,
)
from torchtitan.models.common.attention import GQAttention
from torchtitan.models.utils import get_nparams_and_active_nparams

FLAVOR = "AGPT_12B2A_50K_MOE_aurora_full_sonic"


def test_aurora_moe_architecture_and_parameter_contract():
    config = moe_configs[FLAVOR]()
    assert config.dim == 2048
    assert config.vocab_size == 50304
    assert len(config.layers) == 24

    for layer in config.layers:
        assert isinstance(layer.attention, GQAttention.Config)
        assert layer.attention.n_heads == 16
        assert layer.attention.n_kv_heads == 4
        assert layer.feed_forward is None
        assert layer.moe.num_experts == 36
        assert layer.moe.router.top_k == 3
        assert layer.moe.routed_experts.inner_experts.hidden_dim == 2112
        assert layer.moe.shared_experts.w1.out_features == 4224

    with torch.device("meta"):
        model = config.build()
    assert isinstance(model.layers["0"].moe, AuroraMoE)
    assert isinstance(model.layers["0"].moe.routed_experts, AuroraRoutedExperts)

    counts = {"dense": 0, "router": 0, "shared": 0, "routed": 0}
    for name, parameter in model.named_parameters():
        if ".moe.router." in name:
            counts["router"] += parameter.numel()
        elif ".moe.shared_experts." in name:
            counts["shared"] += parameter.numel()
        elif ".moe.routed_experts.inner_experts." in name:
            counts["routed"] += parameter.numel()
        else:
            counts["dense"] += parameter.numel()

    total = sum(counts.values())
    historical_active = (
        counts["dense"]
        + counts["router"]
        + counts["shared"]
        + counts["routed"] * 3 // 36
    )
    assert total == 12_293_801_984
    assert historical_active == 2_016_708_608
    assert get_nparams_and_active_nparams(model) == (total, 1_913_686_016)
    assert config.get_nparams_and_flops(model, 2048) == (total, 12_690_075_648)


def test_aurora_moe_production_config_and_sharding():
    config = agpt_12b2a_50k_moe_aurora()
    assert config.model_spec.flavor == FLAVOR
    assert config.training.num_tokens_per_microbatch_per_dp_rank == 2048
    assert config.training.max_context_length == 2048
    assert config.training.steps == 50000
    assert config.parallelism.data_parallel_replicate_degree == 256
    assert config.parallelism.data_parallel_shard_degree == 12
    assert config.parallelism.expert_parallel_degree == 12
    assert config.parallelism.enable_data_parallel_replicate_module
    assert isinstance(config.activation_checkpoint, AuroraMoeSelectiveAC.Config)
    assert isinstance(config.activation_checkpoint.build(), AuroraMoeSelectiveAC)
    assert config.model_spec.state_dict_adapter is None

    config.model_spec.model.update_from_config(config=config)
    layer = config.model_spec.model.layers[0]
    assert layer.attention.sharding_config is not None
    assert set(
        layer.moe.routed_experts.inner_experts.sharding_config.state_shardings
    ) == {
        "aurora_up_EDF",
        "aurora_gate_EDF",
        "aurora_down_EFD",
    }
