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


def install_initial_parameter_audit(
    *,
    audit_path: Path,
    output_dir: Path,
    rank: int,
    world_size: int,
    mpi_comm: Any,
    mpi_land: Any,
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
            if load_error is not None:
                raise load_error
            if not loaded:
                raise RuntimeError("Expected checkpoint load, but load_checkpoint returned false")
            result.update(_audit_live_parameters(engine, audit_path, torch))
        except BaseException as error:
            local_error = error
            result.update(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
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
