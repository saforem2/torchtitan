"""Benchmark-only audit for loaded AuroraGPT parameters."""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
import re
from typing import Any


EXPECTED_AUDIT_SHA256 = "b1584ff72d8b2fd4f53594cd1bba7b12efb0f6a102bda8407cd315a00fe6bad6"
EXPECTED_STATE_PARAMETER_COUNT = 75
EXPECTED_HF_PARAMETER_COUNT = 111
EXPECTED_PARAMETER_NUMEL = 1_986_054_144


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_expected_audit(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Initial-parameter audit JSON not found: {path}")
    audit_sha256 = _file_sha256(path)
    if audit_sha256 != EXPECTED_AUDIT_SHA256:
        raise ValueError(
            "Initial-parameter audit JSON SHA256 does not match the pinned "
            f"native/DCP audit: {audit_sha256}"
        )
    audit = json.loads(path.read_text())
    if audit.get("format") != "agpt_native_hf_dcp_full_parameter_audit":
        raise ValueError("Unexpected initial-parameter audit format")
    if not audit.get("summary", {}).get("dcp_vs_hf_all_exact_fp32_promotion"):
        raise ValueError("DCP audit did not prove exact FP32 parameter promotion")
    expected = audit.get("dcp_vs_hf")
    if not isinstance(expected, dict) or len(expected) != EXPECTED_HF_PARAMETER_COUNT:
        raise ValueError("Expected exactly 111 audited HF parameter rows")
    if sum(int(row.get("numel", -1)) for row in expected.values()) != EXPECTED_PARAMETER_NUMEL:
        raise ValueError("Audited parameter total does not match the AuroraGPT 2B model")
    for key, row in expected.items():
        if row.get("actual_dtype") != "torch.float32" or not row.get("equal_in_fp32"):
            raise ValueError(f"Audit row is not exact FP32 DCP data: {key}")
        if len(row.get("actual_sha256", "")) != 64:
            raise ValueError(f"Audit row has no SHA256: {key}")
    return expected, audit_sha256


def _tensor_sha256(tensor: Any, torch: Any) -> str:
    value = tensor.detach().to(device="cpu", copy=True).contiguous().view(torch.uint8)
    return hashlib.sha256(memoryview(value.numpy())).hexdigest()


def _validate_native_ddp_geometry(engine: Any) -> None:
    config = engine.config
    if not getattr(config.parallelism, "enable_data_parallel_native_ddp", False):
        raise ValueError("Initial-parameter audit currently requires native DDP")
    dims = engine.parallelism_context
    geometry = {
        "dp_replicate": dims.dp_replicate,
        "dp_shard": dims.dp_shard,
        "tp": dims.tp,
        "cp": dims.cp,
        "pp": dims.pp,
        "ep": dims.ep,
    }
    if geometry["dp_replicate"] <= 1 or any(
        geometry[axis] != 1 for axis in ("dp_shard", "tp", "cp", "pp", "ep")
    ):
        raise ValueError(f"Unsupported initial-parameter audit geometry: {geometry}")


def _audit_live_parameters(engine: Any, audit_path: Path, torch: Any) -> dict[str, Any]:
    from torch.distributed.tensor import DTensor

    expected, audit_sha256 = _load_expected_audit(audit_path)
    _validate_native_ddp_geometry(engine)
    adapter = getattr(engine, "state_dict_adapter", None)
    if adapter is None:
        raise ValueError("Model has no state-dict adapter")

    parameter_count = 0
    parameter_numel = 0
    for model_part in engine.model_parts:
        model = getattr(model_part, "module", model_part)
        for name, parameter in model.named_parameters():
            if isinstance(parameter, DTensor):
                raise TypeError(f"Sharded DTensor parameter is unsupported: {name}")
            if not isinstance(parameter, torch.Tensor):
                raise TypeError(f"Unexpected parameter type for {name}: {type(parameter)}")
            if parameter.numel() == 0:
                raise ValueError(f"Empty local parameter is unsupported: {name}")
            if parameter.dtype != torch.float32:
                raise TypeError(f"Expected FP32 master parameter for {name}, got {parameter.dtype}")
            parameter_count += 1
            parameter_numel += parameter.numel()
    if parameter_count != EXPECTED_STATE_PARAMETER_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_STATE_PARAMETER_COUNT} fused state parameters, "
            f"got {parameter_count}"
        )
    if parameter_numel != EXPECTED_PARAMETER_NUMEL:
        raise ValueError(
            f"Expected {EXPECTED_PARAMETER_NUMEL} parameter elements, got {parameter_numel}"
        )

    model_state = engine.checkpointer.states["model"].state_dict()
    layer_pattern = re.compile(r"^layers\.(\d+)\.")
    groups: dict[str, dict[str, Any]] = {}
    for key, tensor in model_state.items():
        if isinstance(tensor, DTensor):
            raise TypeError(f"Sharded DTensor state is unsupported: {key}")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Unexpected model state type for {key}: {type(tensor)}")
        match = layer_pattern.match(key)
        group = f"layer-{match.group(1)}" if match else "top-level"
        groups.setdefault(group, {})[key] = tensor

    actual: dict[str, str] = {}
    rows: dict[str, dict[str, Any]] = {}
    for group_name in sorted(groups):
        cpu_state = {
            key: value.detach().to(device="cpu", copy=True)
            for key, value in groups[group_name].items()
        }
        hf_state = adapter.to_hf(cpu_state)
        for key, tensor in hf_state.items():
            if key not in expected:
                raise ValueError(f"Unexpected adapter parameter key: {key}")
            if key in actual:
                raise ValueError(f"Adapter emitted duplicate parameter key: {key}")
            row = expected[key]
            shape = list(tensor.shape)
            if shape != row.get("shape"):
                raise ValueError(f"Shape mismatch for {key}: {shape} != {row.get('shape')}")
            if tensor.dtype != torch.float32:
                raise TypeError(f"Expected FP32 adapted parameter for {key}, got {tensor.dtype}")
            if tensor.numel() != int(row["numel"]):
                raise ValueError(f"Element-count mismatch for {key}")
            actual_hash = _tensor_sha256(tensor, torch)
            expected_hash = row["actual_sha256"]
            actual[key] = actual_hash
            rows[key] = {
                "shape": shape,
                "numel": tensor.numel(),
                "actual_sha256": actual_hash,
                "expected_sha256": expected_hash,
                "match": actual_hash == expected_hash,
            }
        del cpu_state, hf_state

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatched = sorted(key for key, row in rows.items() if not row["match"])
    if missing or extra or mismatched:
        raise ValueError(
            f"Initial parameter mismatch: missing={missing[:5]}, extra={extra[:5]}, "
            f"hash_mismatch={mismatched[:5]}"
        )
    return {
        "status": "passed",
        "audit_json_sha256": audit_sha256,
        "parameter_count": parameter_count,
        "parameter_numel": parameter_numel,
        "hf_parameter_key_count": len(actual),
        "rows": rows,
    }


def _xpu_memory_snapshot(engine: Any, torch: Any) -> dict[str, int]:
    if engine.device.type != "xpu":
        raise ValueError(f"DTensor audit requires XPU, got {engine.device.type}")
    device = engine.device
    return {
        "allocated_bytes": int(torch.xpu.memory_allocated(device)),
        "reserved_bytes": int(torch.xpu.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.xpu.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.xpu.max_memory_reserved(device)),
    }


class _DTensorAuditError(RuntimeError):
    def __init__(self, message: str, diagnostics: dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics


def _audit_live_parameters_allow_dtensor(
    engine: Any,
    audit_path: Path,
    torch: Any,
    mpi_comm: Any,
    initial_error: BaseException | None = None,
) -> dict[str, Any]:
    """Audit DP-sharded weights by gathering one DTensor at a time."""
    from torch.distributed.tensor import DTensor, Replicate, Shard

    expected: dict[str, dict[str, Any]] = {}
    audit_sha256 = ""
    adapter = None
    dims = None
    model_state = None
    parameter_count = 0
    parameter_numel = 0
    metadata: dict[str, Any] = {}
    local_error = (
        f"{type(initial_error).__name__}: {initial_error}"
        if initial_error is not None
        else None
    )
    try:
        if initial_error is not None:
            raise initial_error
        expected, audit_sha256 = _load_expected_audit(audit_path)
        config = engine.config
        if getattr(config.parallelism, "enable_data_parallel_native_ddp", False):
            raise ValueError("DTensor audit cannot be combined with native DDP")
        dims = engine.parallelism_context
        geometry = {
            "dp_replicate": dims.dp_replicate,
            "dp_shard": dims.dp_shard,
            "tp": dims.tp,
            "cp": dims.cp,
            "pp": dims.pp,
            "ep": dims.ep,
        }
        if geometry["dp_shard"] <= 1 or any(
            geometry[axis] != 1 for axis in ("tp", "cp", "pp", "ep")
        ):
            raise ValueError(f"Unsupported DTensor audit geometry: {geometry}")
        adapter = getattr(engine, "state_dict_adapter", None)
        if adapter is None:
            raise ValueError("Model has no state-dict adapter")

        for model_part in engine.model_parts:
            model = getattr(model_part, "module", model_part)
            for name, parameter in model.named_parameters():
                if not isinstance(parameter, (DTensor, torch.Tensor)):
                    raise TypeError(
                        f"Unexpected parameter type for {name}: {type(parameter)}"
                    )
                if parameter.dtype != torch.float32:
                    raise TypeError(
                        f"Expected FP32 master parameter for {name}, got {parameter.dtype}"
                    )
                parameter_count += 1
                parameter_numel += parameter.numel()
        if parameter_count != EXPECTED_STATE_PARAMETER_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_STATE_PARAMETER_COUNT} fused state parameters, "
                f"got {parameter_count}"
            )
        if parameter_numel != EXPECTED_PARAMETER_NUMEL:
            raise ValueError(
                f"Expected {EXPECTED_PARAMETER_NUMEL} parameter elements, "
                f"got {parameter_numel}"
            )
        model_state = engine.checkpointer.states["model"].state_dict()
        dp_mesh = dims.get_mesh("dp")
        dp_rank_set = {int(rank) for rank in dp_mesh.mesh.flatten().tolist()}
        schema = []
        for key in sorted(model_state):
            tensor = model_state[key]
            if not isinstance(tensor, (DTensor, torch.Tensor)):
                raise TypeError(f"Unexpected model state type for {key}: {type(tensor)}")
            item: dict[str, Any] = {
                "key": key,
                "type": f"{type(tensor).__module__}.{type(tensor).__qualname__}",
                "dtype": str(tensor.dtype),
                "global_shape": tuple(int(size) for size in tensor.shape),
                "local_shape": None,
                "placements": (),
                "mesh_shape": (),
                "mesh_dim_names": (),
                "mesh_ranks": (),
            }
            if isinstance(tensor, DTensor):
                local_tensor = tensor.to_local()
                item["local_shape"] = tuple(int(size) for size in local_tensor.shape)
                item["placements"] = tuple(
                    f"{type(placement).__name__}:{getattr(placement, 'dim', '')}"
                    for placement in tensor.placements
                )
                mesh = tensor.device_mesh
                mesh_shape = tuple(int(size) for size in mesh.mesh.shape)
                mesh_ranks = tuple(int(rank) for rank in mesh.mesh.flatten().tolist())
                mesh_dim_names = tuple(mesh.mesh_dim_names or ())
                if len(mesh_shape) > 2 or set(mesh_ranks) - dp_rank_set:
                    raise ValueError(f"Non-DP device mesh for {key}")
                if len(tensor.placements) != len(mesh_shape):
                    raise ValueError(f"Placement count differs from mesh rank for {key}")
                if not all(
                    isinstance(placement, (Replicate, Shard))
                    for placement in tensor.placements
                ):
                    raise ValueError(f"Unsupported DTensor placement for {key}")
                if any(
                    local_size > global_size
                    for local_size, global_size in zip(
                        item["local_shape"], item["global_shape"], strict=True
                    )
                ):
                    raise ValueError(f"Local shard exceeds global shape for {key}")
                item["mesh_shape"] = mesh_shape
                item["mesh_dim_names"] = mesh_dim_names
                item["mesh_ranks"] = mesh_ranks
            schema.append(item)
        if not any(
            isinstance(value, DTensor)
            and any(isinstance(placement, Shard) for placement in value.placements)
            for value in model_state.values()
        ):
            raise ValueError("No sharded DTensor parameters found")
        metadata = {
            "schema": schema,
            "parameter_count": parameter_count,
            "parameter_numel": parameter_numel,
            "pre_audit_xpu_memory": _xpu_memory_snapshot(engine, torch),
        }
    except BaseException as error:
        local_error = local_error or f"{type(error).__name__}: {error}"

    # Compare metadata before any DTensor full_tensor() collective. Local shard
    # shapes may differ for uneven layouts, but keys, types, global shapes,
    # placements, and mesh dimensions must agree on every rank.
    gathered = mpi_comm.allgather({"error": local_error, "metadata": metadata})
    errors = [
        f"rank {rank}: {item['error']}"
        for rank, item in enumerate(gathered)
        if item["error"] is not None
    ]
    if errors:
        raise ValueError("DTensor audit preflight failed: " + "; ".join(errors))
    reference = gathered[0]["metadata"]
    reference_schema = [
        {
            key: item[key]
            for key in (
                "key",
                "type",
                "dtype",
                "global_shape",
                "placements",
                "mesh_shape",
                "mesh_dim_names",
            )
        }
        for item in reference["schema"]
    ]
    for rank, item in enumerate(gathered[1:], start=1):
        candidate = item["metadata"]
        candidate_schema = [
            {
                key: entry[key]
                for key in (
                    "key",
                    "type",
                    "dtype",
                    "global_shape",
                    "placements",
                    "mesh_shape",
                    "mesh_dim_names",
                )
            }
            for entry in candidate["schema"]
        ]
        if (
            candidate_schema != reference_schema
            or candidate["parameter_count"] != reference["parameter_count"]
            or candidate["parameter_numel"] != reference["parameter_numel"]
        ):
            raise ValueError(
                f"DTensor model-state metadata differs from rank 0 on rank {rank}"
            )

    # HSDP may expose a different shard submesh for each replica group. Verify
    # that each reported collective group agrees on its ordered global ranks.
    for rank, item in enumerate(gathered):
        peer_schema_by_key = [
            {entry["key"]: entry for entry in candidate["metadata"]["schema"]}
            for candidate in gathered
        ]
        for schema_item in item["metadata"]["schema"]:
            mesh_ranks = schema_item["mesh_ranks"]
            if not mesh_ranks:
                continue
            if rank not in mesh_ranks or any(
                peer < 0 or peer >= len(gathered) for peer in mesh_ranks
            ):
                raise ValueError(
                    f"Invalid DTensor mesh rank set for {schema_item['key']} on rank {rank}"
                )
            for peer in mesh_ranks:
                peer_schema = peer_schema_by_key[peer].get(schema_item["key"])
                if peer_schema is None or peer_schema["mesh_ranks"] != mesh_ranks:
                    raise ValueError(
                        "DTensor mesh group rank ordering differs for "
                        f"{schema_item['key']} on ranks {rank} and {peer}"
                    )

    pre_audit_memory = metadata["pre_audit_xpu_memory"]
    host_state: dict[str, Any] = {}
    materialization_error = None
    with torch.no_grad():
        for item in reference["schema"]:
            key = item["key"]
            tensor = model_state[key]
            try:
                full_tensor = tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
                host_state[key] = full_tensor.detach().to(device="cpu", copy=True)
                del full_tensor
            except BaseException as error:
                if materialization_error is None:
                    materialization_error = error
                # Continue in sorted order so peers execute every remaining
                # full_tensor collective before the failure gate.
                continue
    if materialization_error is not None:
        raise _DTensorAuditError(
            "Failed to materialize DTensor model state on this rank",
            {
                "gpu_collectives_used": True,
                "peak_memory_comparable": False,
                "pre_audit_xpu_memory": pre_audit_memory,
                "post_audit_xpu_memory": None,
            },
        ) from materialization_error

    try:
        post_audit_memory = _xpu_memory_snapshot(engine, torch)
    except BaseException as error:
        raise _DTensorAuditError(
            f"Failed to read post-audit XPU memory counters: {error}",
            {
                "gpu_collectives_used": True,
                "peak_memory_comparable": False,
                "pre_audit_xpu_memory": pre_audit_memory,
                "post_audit_xpu_memory": None,
            },
        ) from error
    diagnostics = {
        "gpu_collectives_used": True,
        "peak_memory_comparable": False,
        "pre_audit_xpu_memory": pre_audit_memory,
        "post_audit_xpu_memory": post_audit_memory,
    }

    try:
        groups: dict[str, dict[str, Any]] = {}
        layer_pattern = re.compile(r"^layers\.(\d+)\.")
        for key, tensor in host_state.items():
            match = layer_pattern.match(key)
            group = f"layer-{int(match.group(1)):05d}" if match else "top-level"
            groups.setdefault(group, {})[key] = tensor

        actual: dict[str, str] = {}
        rows: dict[str, dict[str, Any]] = {}
        ordered_groups = ["top-level"] + sorted(
            (name for name in groups if name != "top-level")
        )
        for group_name in ordered_groups:
            if group_name not in groups:
                continue
            hf_state = adapter.to_hf(groups[group_name])
            for key, tensor in hf_state.items():
                if key not in expected:
                    raise ValueError(f"Unexpected adapter parameter key: {key}")
                if key in actual:
                    raise ValueError(f"Adapter emitted duplicate parameter key: {key}")
                row = expected[key]
                shape = list(tensor.shape)
                if shape != row.get("shape"):
                    raise ValueError(f"Shape mismatch for {key}: {shape} != {row.get('shape')}")
                if tensor.dtype != torch.float32 or tensor.numel() != int(row["numel"]):
                    raise ValueError(f"Dtype or element-count mismatch for {key}")
                actual_hash = _tensor_sha256(tensor, torch)
                expected_hash = row["actual_sha256"]
                actual[key] = actual_hash
                rows[key] = {
                    "shape": shape,
                    "numel": tensor.numel(),
                    "actual_sha256": actual_hash,
                    "expected_sha256": expected_hash,
                    "match": actual_hash == expected_hash,
                }
            del hf_state
        missing = sorted(set(expected) - set(actual))
        mismatched = sorted(key for key, row in rows.items() if not row["match"])
        if missing or mismatched:
            raise ValueError(
                f"Initial DTensor parameter mismatch: missing={missing[:5]}, "
                f"hash_mismatch={mismatched[:5]}"
            )
    except BaseException as error:
        raise _DTensorAuditError(
            f"DTensor parameter comparison failed: {type(error).__name__}: {error}",
            diagnostics,
        ) from error
    return {
        "status": "passed",
        "audit_json_sha256": audit_sha256,
        "parameter_count": parameter_count,
        "parameter_numel": parameter_numel,
        "hf_parameter_key_count": len(actual),
        "rows": rows,
        **diagnostics,
    }


def install_initial_parameter_audit(
    *,
    audit_path: Path,
    output_dir: Path,
    rank: int,
    world_size: int,
    mpi_comm: Any,
    mpi_land: Any,
    allow_dtensor: bool = False,
) -> None:
    """Audit each rank after checkpoint load and gate training with CPU MPI."""
    import importlib
    import torch

    trainer_module = importlib.import_module("torchtitan.experiments.ezpz.trainer")
    engine_class = trainer_module.EzpzTrainingEngine
    original_load_checkpoint = engine_class.load_checkpoint
    output_path = output_dir / f"rank-{rank:03d}.initial-parameter-audit.json"

    @functools.wraps(original_load_checkpoint)
    def audited_load_checkpoint(engine: Any) -> bool:
        loaded = False
        load_error = None
        try:
            loaded = original_load_checkpoint(engine)
        except BaseException as error:
            load_error = error
        result: dict[str, Any] = {
            "rank": rank,
            "world_size": world_size,
            "audit_json": str(audit_path),
            "loaded_checkpoint": bool(loaded),
        }
        local_error = load_error
        try:
            if allow_dtensor:
                initial_error = load_error
                if initial_error is None and not loaded:
                    initial_error = RuntimeError(
                        "Expected checkpoint load, but load_checkpoint returned false"
                    )
                result.update(
                    _audit_live_parameters_allow_dtensor(
                        engine, audit_path, torch, mpi_comm, initial_error
                    )
                )
            else:
                if load_error is not None:
                    raise load_error
                if not loaded:
                    raise RuntimeError(
                        "Expected checkpoint load, but load_checkpoint returned false"
                    )
                result.update(_audit_live_parameters(engine, audit_path, torch))
        except BaseException as error:
            local_error = error
            result.update(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            diagnostics = getattr(error, "diagnostics", None)
            if diagnostics is not None:
                result["audit_diagnostics"] = diagnostics
            elif allow_dtensor:
                result.update(
                    {
                        "gpu_collectives_used": False,
                        "peak_memory_comparable": False,
                    }
                )

        try:
            output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        except BaseException as error:
            local_error = local_error or error
            result.update(
                {
                    "status": "failed",
                    "error": f"audit output write failed: {type(error).__name__}: {error}",
                }
            )

        all_ranks_passed = bool(mpi_comm.allreduce(local_error is None, op=mpi_land))
        if not all_ranks_passed:
            raise RuntimeError(
                "Initial-parameter audit failed on at least one rank; see "
                f"{output_path}"
            ) from local_error
        result["all_ranks_passed"] = True
        output_error = None
        try:
            output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        except BaseException as error:
            output_error = error
        all_outputs_written = bool(mpi_comm.allreduce(output_error is None, op=mpi_land))
        if not all_outputs_written:
            raise RuntimeError(
                "Initial-parameter audit artifact write failed on a rank"
            ) from output_error
        return loaded

    engine_class.load_checkpoint = audited_load_checkpoint
