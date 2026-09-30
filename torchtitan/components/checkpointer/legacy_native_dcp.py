# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import _StridedShard


@dataclass(frozen=True)
class OptimizerFusedLayout:
    """Physical parameter layouts needed to migrate historical optimizer FQNs."""

    qkv: dict[str, tuple[int, int, tuple[int, ...]]]
    stacked: dict[str, tuple[int, ...]]
    qkv_bias: dict[str, tuple[int, int, tuple[int, ...]]] = field(default_factory=dict)


def _qkv_optimizer_layouts(layout: OptimizerFusedLayout):
    yield from ((prefix, "weight", spec) for prefix, spec in layout.qkv.items())
    yield from ((prefix, "bias", spec) for prefix, spec in layout.qkv_bias.items())


def _split_qkv(
    tensor: torch.Tensor, *, head_dim: int, heads_per_kv: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group_size = heads_per_kv + 2
    num_kv_heads = tensor.shape[0] // (group_size * head_dim)
    tail = tensor.shape[1:]
    packed = tensor.reshape(num_kv_heads, group_size, head_dim, *tail)
    return (
        packed[:, :heads_per_kv].reshape(-1, *tail).contiguous(),
        packed[:, heads_per_kv].reshape(-1, *tail).contiguous(),
        packed[:, heads_per_kv + 1].reshape(-1, *tail).contiguous(),
    )


def _fuse_qkv(
    wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, *, head_dim: int
) -> torch.Tensor:
    num_kv_heads = wk.shape[0] // head_dim
    heads_per_kv = wq.shape[0] // (num_kv_heads * head_dim)
    tail = wq.shape[1:]
    q = wq.reshape(num_kv_heads, heads_per_kv, head_dim, *tail)
    k = wk.reshape(num_kv_heads, 1, head_dim, *tail)
    v = wv.reshape(num_kv_heads, 1, head_dim, *tail)
    return torch.cat((q, k, v), dim=1).reshape(-1, *tail)


def _fuse_qkv_dtensors_to_local_shard(
    values: tuple[DTensor, DTensor, DTensor],
    *,
    head_dim: int,
    heads_per_kv: int,
    fused_shape: tuple[int, ...],
    mesh: Any,
    placements: tuple[Any, ...],
) -> DTensor:
    global_shape = torch.Size(fused_shape)
    local_shape, _ = compute_local_shape_and_global_offset(
        global_shape, mesh, placements
    )
    local_fused = torch.empty(
        local_shape, dtype=values[0].dtype, device=values[0].to_local().device
    )
    row_indices = torch.arange(global_shape[0])
    coordinate = mesh.get_coordinate()
    assert coordinate is not None
    for mesh_dim, placement in enumerate(placements):
        if isinstance(placement, (Shard, _StridedShard)) and placement.dim == 0:
            shards, _ = placement._split_tensor(
                row_indices,
                mesh.size(mesh_dim=mesh_dim),
                with_padding=False,
                contiguous=False,
            )
            row_indices = shards[coordinate[mesh_dim]]
    assert row_indices.numel() == local_shape[0]

    num_kv_heads = values[1].shape[0] // head_dim
    group_rows = (heads_per_kv + 2) * head_dim
    source_rows = (heads_per_kv * head_dim, head_dim, head_dim)
    source_offsets = (0, heads_per_kv * head_dim, (heads_per_kv + 1) * head_dim)
    row_replicated = tuple(
        Replicate()
        if isinstance(placement, (Shard, _StridedShard)) and placement.dim == 0
        else placement
        for placement in placements
    )
    for source, rows_per_group, offset_in_group in zip(
        values, source_rows, source_offsets, strict=True
    ):
        source = source.redistribute(mesh, row_replicated)
        source_local = source.to_local()
        within_group = row_indices.remainder(group_rows)
        mask = (within_group >= offset_in_group) & (
            within_group < offset_in_group + rows_per_group
        )
        local_positions = mask.nonzero().flatten()
        if local_positions.numel() > 0:
            source_indices = (
                row_indices[mask].div(group_rows, rounding_mode="floor")
                * rows_per_group
                + within_group[mask]
                - offset_in_group
            )
            local_fused.index_copy_(
                0,
                local_positions.to(local_fused.device),
                source_local.index_select(0, source_indices.to(source_local.device)),
            )
        del source, source_local

    assert num_kv_heads * group_rows == global_shape[0]
    return DTensor.from_local(
        local_fused,
        mesh,
        placements,
        shape=global_shape,
        stride=torch.empty(global_shape, device="meta").stride(),
        run_check=False,
    )


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    return left == right


def _copies(value: Any, count: int) -> tuple[Any, ...]:
    if isinstance(value, torch.Tensor):
        return tuple(value.clone() for _ in range(count))
    return (value,) * count


def _empty_dtensor_from_metadata(template: DTensor, metadata: Any) -> DTensor:
    global_shape = torch.Size(metadata.size)
    # A fused QKV tensor can carry TP-produced _StridedShard placement that
    # encodes the packed projection grouping. Historical logical Q/K/V tensors
    # have smaller row dimensions and cannot reuse that fused-layout stride.
    # DCP needs ordinary row shards for those logical destinations; the adapter
    # retains the original fused placement and restores it after loading.
    placements = tuple(
        Shard(placement.dim) if isinstance(placement, _StridedShard) else placement
        for placement in template.placements
    )
    local_shape, _ = compute_local_shape_and_global_offset(
        global_shape, template.device_mesh, placements
    )
    local = torch.empty(
        local_shape,
        dtype=metadata.properties.dtype,
        device=template.to_local().device,
    )
    stride = torch.empty(global_shape, device="meta").stride()
    return DTensor.from_local(
        local,
        template.device_mesh,
        placements,
        shape=global_shape,
        stride=stride,
    )


def _split_optimizer_state(
    state_dict: dict[str, Any],
    layout: OptimizerFusedLayout,
    native_sharding: dict[str, tuple[Any, tuple[Any, ...]]],
    checkpoint_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in state_dict.items():
        for (
            prefix,
            param_name,
            (
                head_dim,
                heads_per_kv,
                fused_shape,
            ),
        ) in _qkv_optimizer_layouts(layout):
            marker = f"{prefix}wqkv.{param_name}."
            if marker not in key:
                continue
            if isinstance(value, torch.Tensor) and tuple(value.shape) == fused_shape:
                if isinstance(value, DTensor):
                    native_sharding[key] = (value.device_mesh, value.placements)
                    logical_keys = tuple(
                        key.replace(marker, f"{prefix}{name}.{param_name}.")
                        for name in ("wq", "wk", "wv")
                    )
                    if all(item in checkpoint_metadata for item in logical_keys):
                        values = tuple(
                            _empty_dtensor_from_metadata(
                                value, checkpoint_metadata[item]
                            )
                            for item in logical_keys
                        )
                    else:
                        value = value.redistribute(
                            value.device_mesh,
                            [Replicate()] * value.device_mesh.ndim,
                        )
                        values = _split_qkv(
                            value, head_dim=head_dim, heads_per_kv=heads_per_kv
                        )
                else:
                    values = _split_qkv(
                        value, head_dim=head_dim, heads_per_kv=heads_per_kv
                    )
            elif not isinstance(value, torch.Tensor) or value.numel() == 1:
                values = _copies(value, 3)
            else:
                raise ValueError(
                    f"unsupported optimizer tensor shape for {key}: "
                    f"{tuple(value.shape)} != {fused_shape}"
                )
            for name, item in zip(("wq", "wk", "wv"), values, strict=True):
                result[key.replace(marker, f"{prefix}{name}.{param_name}.")] = item
            break
        else:
            for prefix, fused_shape in layout.stacked.items():
                marker = f"{prefix}w13.weight."
                if marker not in key:
                    continue
                if (
                    isinstance(value, torch.Tensor)
                    and tuple(value.shape) == fused_shape
                ):
                    if isinstance(value, DTensor):
                        native_sharding[key] = (value.device_mesh, value.placements)
                        logical_keys = tuple(
                            key.replace(marker, f"{prefix}{name}.weight.")
                            for name in ("w1", "w3")
                        )
                        if all(item in checkpoint_metadata for item in logical_keys):
                            values = tuple(
                                _empty_dtensor_from_metadata(
                                    value, checkpoint_metadata[item]
                                )
                                for item in logical_keys
                            )
                        else:
                            value = value.redistribute(
                                value.device_mesh,
                                [Replicate()] * value.device_mesh.ndim,
                            )
                            values = tuple(value.unbind(0))
                    else:
                        values = tuple(value.unbind(0))
                elif not isinstance(value, torch.Tensor) or value.numel() == 1:
                    values = _copies(value, 2)
                else:
                    raise ValueError(
                        f"unsupported optimizer tensor shape for {key}: "
                        f"{tuple(value.shape)} != {fused_shape}"
                    )
                for name, item in zip(("w1", "w3"), values, strict=True):
                    result[key.replace(marker, f"{prefix}{name}.weight.")] = item
                break
            else:
                result[key] = value
    return result


def _collapse_equal(values: tuple[Any, ...], key: str) -> Any:
    if not all(_equal(values[0], value) for value in values[1:]):
        raise ValueError(f"Historical optimizer values disagree for {key}")
    return values[0]


def _fuse_optimizer_state(
    state_dict: dict[str, Any],
    layout: OptimizerFusedLayout,
    native_sharding: dict[str, tuple[Any, tuple[Any, ...]]],
) -> dict[str, Any]:
    result = dict(state_dict)
    for (
        prefix,
        param_name,
        (
            head_dim,
            _heads_per_kv,
            _fused_shape,
        ),
    ) in _qkv_optimizer_layouts(layout):
        marker = f"{prefix}wq.{param_name}."
        for key in [item for item in result if marker in item]:
            keys = tuple(
                key.replace(marker, f"{prefix}{name}.{param_name}.")
                for name in ("wq", "wk", "wv")
            )
            if not all(item in result for item in keys):
                raise ValueError(f"Incomplete historical QKV optimizer state for {key}")
            values = tuple(result.pop(item) for item in keys)
            fused_key = key.replace(marker, f"{prefix}wqkv.{param_name}.")
            if (
                isinstance(values[0], DTensor)
                and values[0].numel() > 1
                and fused_key in native_sharding
            ):
                mesh, placements = native_sharding[fused_key]
                assert all(isinstance(value, DTensor) for value in values)
                fused = _fuse_qkv_dtensors_to_local_shard(
                    values,
                    head_dim=head_dim,
                    heads_per_kv=_heads_per_kv,
                    fused_shape=_fused_shape,
                    mesh=mesh,
                    placements=placements,
                )
            else:
                if isinstance(values[0], DTensor) and values[0].numel() > 1:
                    values = tuple(
                        value.redistribute(
                            value.device_mesh,
                            [Replicate()] * value.device_mesh.ndim,
                        )
                        for value in values
                    )
                fused = (
                    _fuse_qkv(*values, head_dim=head_dim)
                    if isinstance(values[0], torch.Tensor) and values[0].numel() > 1
                    else _collapse_equal(values, fused_key)
                )
                if fused_key in native_sharding:
                    assert isinstance(fused, DTensor)
                    mesh, placements = native_sharding[fused_key]
                    fused = fused.redistribute(mesh, placements)
            result[fused_key] = fused
    for prefix in layout.stacked:
        marker = f"{prefix}w1.weight."
        for key in [item for item in result if marker in item]:
            keys = (key, key.replace(marker, f"{prefix}w3.weight."))
            if not all(item in result for item in keys):
                raise ValueError(f"Incomplete historical FFN optimizer state for {key}")
            values = tuple(result.pop(item) for item in keys)
            fused_key = key.replace(marker, f"{prefix}w13.weight.")
            if isinstance(values[0], DTensor) and values[0].numel() > 1:
                values = tuple(
                    value.redistribute(
                        value.device_mesh,
                        [Replicate()] * value.device_mesh.ndim,
                    )
                    for value in values
                )
            fused = (
                torch.stack(values, dim=0)
                if isinstance(values[0], torch.Tensor) and values[0].numel() > 1
                else _collapse_equal(values, fused_key)
            )
            if fused_key in native_sharding:
                assert isinstance(fused, DTensor)
                mesh, placements = native_sharding[fused_key]
                fused = fused.redistribute(mesh, placements)
            result[fused_key] = fused
    return result


class LogicalOptimizerState:
    """Load-only optimizer view that exposes historical logical parameter FQNs."""

    def __init__(
        self,
        optimizer: Any,
        layout: OptimizerFusedLayout,
        checkpoint_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.layout = layout
        self.checkpoint_metadata = checkpoint_metadata or {}
        self.native_sharding: dict[str, tuple[Any, tuple[Any, ...]]] = {}
        self.current_defaults: dict[str, Any] = {}

    def state_dict(self) -> dict[str, Any]:
        current = self.optimizer.state_dict()
        logical = _split_optimizer_state(
            current,
            self.layout,
            self.native_sharding,
            self.checkpoint_metadata,
        )
        if not self.checkpoint_metadata:
            return logical
        self.current_defaults = current
        return {
            key: value
            for key, value in logical.items()
            if key in self.checkpoint_metadata
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        fused = _fuse_optimizer_state(state_dict, self.layout, self.native_sharding)
        for key, value in self.current_defaults.items():
            fused.setdefault(key, value)
        self.optimizer.load_state_dict(fused)


def _optimizer_layout(
    model_state: dict[str, Any],
    adapter: Any,
    *,
    checkpoint_keys: set[str],
) -> OptimizerFusedLayout:
    from torchtitan.models.common.attention import QKVLinear
    from torchtitan.models.common.feed_forward import FeedForward

    qkv: dict[str, tuple[int, int, tuple[int, ...]]] = {}
    qkv_bias: dict[str, tuple[int, int, tuple[int, ...]]] = {}
    stacked: dict[str, tuple[int, ...]] = {}
    for fqn, config, _parent, _ in adapter.model_config.traverse(QKVLinear.Config):
        prefix = f"{fqn}." if fqn else ""
        key = f"{prefix}wqkv.weight"
        logical_markers = tuple(
            f"{prefix}{name}.weight." for name in ("wq", "wk", "wv")
        )
        has_split = any(
            any(marker in checkpoint_key for marker in logical_markers)
            for checkpoint_key in checkpoint_keys
        )
        has_fused = any(
            f"{prefix}wqkv.weight." in checkpoint_key
            for checkpoint_key in checkpoint_keys
        )
        if has_split and has_fused:
            raise ValueError(
                f"Historical optimizer mixes fused and logical QKV keys for {prefix}"
            )
        if has_split and key in model_state:
            qkv[prefix] = (
                config.head_dim,
                config.n_heads // config.n_kv_heads,
                tuple(model_state[key].shape),
            )
        bias_key = f"{prefix}wqkv.bias"
        bias_markers = tuple(f"{prefix}{name}.bias." for name in ("wq", "wk", "wv"))
        has_split_bias = any(
            any(marker in checkpoint_key for marker in bias_markers)
            for checkpoint_key in checkpoint_keys
        )
        has_fused_bias = any(
            f"{bias_key}." in checkpoint_key for checkpoint_key in checkpoint_keys
        )
        if has_split_bias and has_fused_bias:
            raise ValueError(
                f"Historical optimizer mixes fused and logical QKV keys for {bias_key}"
            )
        if has_split_bias and bias_key in model_state:
            qkv_bias[prefix] = (
                config.head_dim,
                config.n_heads // config.n_kv_heads,
                tuple(model_state[bias_key].shape),
            )
    for fqn, _config, _parent, _ in adapter.model_config.traverse(FeedForward.Config):
        prefix = f"{fqn}." if fqn else ""
        key = f"{prefix}w13.weight"
        logical_markers = tuple(f"{prefix}{name}.weight." for name in ("w1", "w3"))
        has_split = any(
            any(marker in checkpoint_key for marker in logical_markers)
            for checkpoint_key in checkpoint_keys
        )
        has_fused = any(
            f"{prefix}w13.weight." in checkpoint_key
            for checkpoint_key in checkpoint_keys
        )
        if has_split and has_fused:
            raise ValueError(
                f"Historical optimizer mixes fused and logical FFN keys for {prefix}"
            )
        if has_split and key in model_state:
            stacked[prefix] = tuple(model_state[key].shape)
    return OptimizerFusedLayout(qkv=qkv, stacked=stacked, qkv_bias=qkv_bias)


def prepare_legacy_native_load(
    state_dict: dict[str, Any],
    model_keys: set[str],
    checkpoint_metadata: Mapping[str, Any] | set[str],
    adapter: Any,
) -> tuple[dict[str, Any], Any]:
    """Prepare a temporary logical-schema destination for a historical DCP load."""
    checkpoint_keys = set(checkpoint_metadata)
    metadata = (
        checkpoint_metadata
        if isinstance(checkpoint_metadata, Mapping)
        else {key: None for key in checkpoint_metadata}
    )
    has_split = any(
        marker in key
        for key in checkpoint_keys
        for marker in (
            ".qkv_linear.wq.weight",
            ".qkv_linear.wk.weight",
            ".qkv_linear.wv.weight",
            ".feed_forward.w1.weight",
            ".feed_forward.w3.weight",
        )
    )
    if not has_split:
        return state_dict, lambda loaded: loaded

    for fused_key in model_keys:
        if ".qkv_linear.wqkv.weight" in fused_key:
            logical_keys = tuple(
                fused_key.replace("wqkv.weight", f"{name}.weight")
                for name in ("wq", "wk", "wv")
            )
        elif ".feed_forward.w13.weight" in fused_key:
            logical_keys = tuple(
                fused_key.replace("w13.weight", f"{name}.weight")
                for name in ("w1", "w3")
            )
        else:
            continue
        present = tuple(key in checkpoint_keys for key in logical_keys)
        if fused_key in checkpoint_keys and any(present):
            family = "QKV" if "wqkv.weight" in fused_key else "FFN"
            raise ValueError(
                f"Historical checkpoint mixes fused and logical {family} keys "
                f"for {fused_key}"
            )
        if any(present) and not all(present):
            raise ValueError(
                f"Historical checkpoint has incomplete logical keys for {fused_key}"
            )

    auxiliary = {
        key: value for key, value in state_dict.items() if key not in model_keys
    }
    model_state = {key: state_dict[key] for key in model_keys}
    logical_model: dict[str, Any] = {}
    for key, value in model_state.items():
        if key in checkpoint_keys:
            logical_model[key] = value
            continue
        conversion_value = value
        if isinstance(value, DTensor):
            conversion_value = torch.empty(
                value.shape, dtype=value.dtype, device="meta"
            )
            if ".qkv_linear.wqkv." in key:
                adapter._qkv_linear_sharding[key] = (
                    value.device_mesh,
                    value.placements,
                )
            elif ".feed_forward.w13." in key:
                adapter._stacked_linear_sharding[key] = (
                    value.device_mesh,
                    value.placements,
                )
        converted = adapter.native_fused_to_logical({key: conversion_value})
        for logical_key, logical_value in converted.items():
            if (
                isinstance(value, DTensor)
                and logical_key != key
                and metadata.get(logical_key) is not None
            ):
                logical_value = _empty_dtensor_from_metadata(
                    value, metadata[logical_key]
                )
            logical_model[logical_key] = logical_value
    prepared_model = {
        key: value
        for key, value in {**model_state, **logical_model}.items()
        if key in checkpoint_keys
    }
    represented = adapter.native_logical_to_fused(prepared_model)
    missing = sorted(model_keys - set(represented))
    extra = sorted(set(represented) - model_keys)
    if missing or extra:
        raise ValueError(
            "Historical checkpoint cannot represent current model state: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )

    prepared = {**prepared_model, **auxiliary}
    optimizer = auxiliary.get("optimizer")
    if optimizer is not None:
        optimizer_metadata = {
            key.removeprefix("optimizer."): value
            for key, value in metadata.items()
            if key.startswith("optimizer.")
        }
        prepared["optimizer"] = LogicalOptimizerState(
            optimizer,
            _optimizer_layout(
                model_state,
                adapter,
                checkpoint_keys=checkpoint_keys,
            ),
            optimizer_metadata,
        )

    def finish(loaded: dict[str, Any]) -> dict[str, Any]:
        loaded_model = {key: loaded[key] for key in prepared_model}
        restored = adapter.native_logical_to_fused(loaded_model)
        restored.update(auxiliary)
        return restored

    return prepared, finish
