#!/usr/bin/env python3
"""Run an AGPT benchmark and preserve process-lifetime XPU/compile metrics.

This entrypoint does not reset allocator counters or alter training. Each rank
writes its own diagnostics even when the training entrypoint raises an error.
Compilation timers are nested; their values must not be summed across keys.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback


class SophiaStepWitness:
    """Observe successful optimizer returns and sample changes without step syncs."""

    def __init__(self, *, module: str):
        self.records = {}
        self.model = None
        self.model_samples = []
        self.runtime = {}
        self.communication = {}
        self.ddp_models = []
        self.errors = []
        self.original_step = None
        self.optimizer_class = None
        try:
            optimizer_class = importlib.import_module(module).SophiaG
            self.optimizer_class = optimizer_class
            self.original_step = optimizer_class.step
            original_step = self.original_step

            @functools.wraps(original_step)
            def witnessed_step(optimizer, *args, **kwargs):
                key = id(optimizer)
                if key not in self.records:
                    record = {
                        "optimizer": optimizer,
                        "successful_step_calls": 0,
                        "step_call_attempts": 0,
                        "step_exceptions": 0,
                        "samples": [],
                    }
                    self.records[key] = record
                    try:
                        if self.model is not None and not self.model_samples:
                            for name, param in self.model.named_parameters():
                                local = self.local_tensor(param.detach())
                                if local.numel():
                                    self.model_samples.append(
                                        (name, param, local.reshape(-1)[0].clone())
                                    )
                        for group in optimizer.param_groups:
                            for param in group["params"]:
                                local = self.local_tensor(param.detach())
                                if local.numel():
                                    record["samples"].append(
                                        (param, local.reshape(-1)[0].clone(), local.ndim)
                                    )
                    except Exception as error:
                        self.errors.append(f"initial_samples: {type(error).__name__}: {error}")
                record = self.records[key]
                record["step_call_attempts"] += 1
                try:
                    result = original_step(optimizer, *args, **kwargs)
                except BaseException:
                    record["step_exceptions"] += 1
                    raise
                record["successful_step_calls"] += 1
                return result

            optimizer_class.step = witnessed_step
            if module == "megatron.optimizer.sophia":
                # DeepSpeed can flatten master tensors; retain named model views
                # separately so norm samples are actual norm parameters.
                import deepspeed

                original_initialize = deepspeed.initialize
                original_all_reduce = deepspeed.comm.all_reduce

                @functools.wraps(original_all_reduce)
                def witnessed_all_reduce(tensor, *args, **kwargs):
                    if tensor.numel() >= 1024:
                        key = f"{tensor.dtype}:{tensor.numel()}"
                        entry = self.communication.setdefault(key, {
                            "dtype": str(tensor.dtype), "numel": tensor.numel(),
                            "bytes": tensor.numel() * tensor.element_size(), "calls": 0,
                        })
                        entry["calls"] += 1
                    return original_all_reduce(tensor, *args, **kwargs)

                deepspeed.comm.all_reduce = witnessed_all_reduce

                @functools.wraps(original_initialize)
                def witnessed_initialize(*args, **kwargs):
                    result = original_initialize(*args, **kwargs)
                    engine = result[0]
                    self.model = engine.module
                    self.runtime = {
                        "model_and_grad_accum_dtype": tuple(map(str, engine.get_data_types())),
                        "communication_dtype": str(engine.communication_data_type),
                        "zero_stage": engine.zero_optimization_stage(),
                        "pipeline_parallelism": engine.pipeline_parallelism,
                        "optimizer_wrapper": type(engine.optimizer).__qualname__,
                        "basic_optimizer": type(engine.basic_optimizer).__qualname__,
                        "configured_grad_accum_dtype": str(engine._config.grad_accum_dtype),
                        "configured_communication_dtype": str(engine._config.communication_data_type),
                    }
                    print("DS_RUNTIME " + json.dumps(self.runtime, sort_keys=True), flush=True)
                    expected = os.environ.get("BENCH_EXPECTED_DS_OPTIMIZER_WRAPPER")
                    if expected and self.runtime["optimizer_wrapper"] != expected:
                        raise RuntimeError(
                            f"DeepSpeed optimizer wrapper {self.runtime['optimizer_wrapper']} != expected {expected}"
                        )
                    return result

                deepspeed.initialize = witnessed_initialize
            else:
                native = importlib.import_module("torchtitan.experiments.ezpz.native_ddp")
                original_initialize = native.NativeDDP.__init__

                @functools.wraps(original_initialize)
                def witnessed_ddp_initialize(model, *args, **kwargs):
                    original_initialize(model, *args, **kwargs)
                    self.ddp_models.append(model)

                native.NativeDDP.__init__ = witnessed_ddp_initialize
        except Exception as error:
            self.errors.append(f"install: {type(error).__name__}: {error}")

    @staticmethod
    def local_tensor(tensor):
        # DTensor optimizer state is inspected locally; no collectives are added.
        return tensor.to_local() if hasattr(tensor, "to_local") else tensor

    def diagnostics(self):
        import torch

        result = {"errors": list(self.errors), "optimizers": [],
                  "deepspeed_runtime": self.runtime,
                  "observed_large_all_reduce_payloads": list(self.communication.values())}
        result["native_ddp"] = [
            {"logging": model._get_ddp_logging_data(),
             "activation_dtypes": dict(getattr(model, "_agpt_dtype_probe", {}))}
            for model in self.ddp_models
        ]
        for record in self.records.values():
            try:
                optimizer = record["optimizer"]
                parameters = [
                    self.local_tensor(param.detach())
                    for group in optimizer.param_groups
                    for param in group["params"]
                ]
                momenta = [
                    self.local_tensor(state["exp_avg"].detach())
                    for state in optimizer.state.values()
                    if "exp_avg" in state
                ]
                hessians = [
                    self.local_tensor(state["hessian"].detach())
                    for state in optimizer.state.values()
                    if "hessian" in state
                ]
                state_steps = [
                    float(self.local_tensor(state["step"]).item())
                    if torch.is_tensor(state["step"])
                    else float(state["step"])
                    for state in optimizer.state.values()
                    if "step" in state
                ]
                deltas = [
                    (self.local_tensor(param.detach()).reshape(-1)[0] - before).abs()
                    for param, before, _ in record["samples"]
                ]
                norm_deltas = [
                    delta
                    for delta, (_, _, ndim) in zip(deltas, record["samples"])
                    if ndim == 1
                ]

                def all_finite(tensors):
                    return bool(torch.stack([torch.isfinite(t).all() for t in tensors]).all().item()) if tensors else None

                def nonzero_count(tensors):
                    return int(torch.stack([torch.count_nonzero(t) for t in tensors]).sum().item()) if tensors else 0

                def max_delta(values):
                    return float(torch.stack(values).max().item()) if values else None

                result["optimizers"].append({
                    "class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
                    "successful_step_calls": record["successful_step_calls"],
                    "step_call_attempts": record["step_call_attempts"],
                    "step_exceptions": record["step_exceptions"],
                    "parameter_count": len(parameters),
                    "state_parameter_count": len(optimizer.state),
                    "state_step_count": len(state_steps),
                    "state_step_min": min(state_steps) if state_steps else None,
                    "state_step_max": max(state_steps) if state_steps else None,
                    "all_finite_parameters": all_finite(parameters),
                    "all_finite_momentum": all_finite(momenta),
                    "all_finite_hessian": all_finite(hessians),
                    "nonzero_hessian_elements": nonzero_count(hessians),
                    "sampled_parameters": len(deltas),
                    "changed_sampled_parameters": nonzero_count(deltas),
                    "max_sample_delta": max_delta(deltas),
                    "sampled_1d_parameters": len(norm_deltas),
                    "changed_sampled_1d_parameters": nonzero_count(norm_deltas),
                    "max_sample_1d_delta": max_delta(norm_deltas),
                    "parameter_dtypes": sorted({str(t.dtype) for t in parameters}),
                    "momentum_dtypes": sorted({str(t.dtype) for t in momenta}),
                    "hessian_dtypes": sorted({str(t.dtype) for t in hessians}),
                    "zero1": (
                        optimizer._ezpz_zero1.diagnostics()
                        if hasattr(optimizer, "_ezpz_zero1")
                        else None
                    ),
                })
            except Exception as error:
                result["errors"].append(f"final_state: {type(error).__name__}: {error}")
        if self.model is not None:
            try:
                model_deltas = [
                    (self.local_tensor(param.detach()).reshape(-1)[0] - before).abs()
                    for _, param, before in self.model_samples
                ]
                norm_samples = [
                    (name, delta)
                    for (name, param, _), delta in zip(self.model_samples, model_deltas)
                    if param.ndim == 1 and "norm" in name.lower()
                ]
                result["model_parameters"] = {
                    "sampled_parameters": len(model_deltas),
                    "changed_sampled_parameters": sum(int(delta.ne(0).item()) for delta in model_deltas),
                    "max_sample_delta": float(torch.stack(model_deltas).max().item()) if model_deltas else None,
                    "norm_samples": [
                        {"name": name, "absolute_delta": float(delta.item())}
                        for name, delta in norm_samples
                    ],
                    "parameter_dtypes": sorted({str(param.dtype) for _, param, _ in self.model_samples}),
                }
            except Exception as error:
                result["errors"].append(f"final_model_samples: {type(error).__name__}: {error}")
        result["optimizer_count"] = len(self.records)
        result["successful_step_calls"] = sum(record["successful_step_calls"] for record in self.records.values())
        result["step_exceptions"] = sum(record["step_exceptions"] for record in self.records.values())
        result["note"] = "Only the first call samples parameters; later calls add Python counters. Final reductions run after training."
        return result


def install_mds_batch_witness(rank, world_size):
    """Observe CPU batches consumed by get_batch before MDS device transfer."""
    if not os.environ.get("AGPT_BENCH_BATCH_HASH_DIR"):
        return
    path = os.environ["AGPT_BENCH_BATCH_HELPER"]
    spec = importlib.util.spec_from_file_location("agpt_batch_witness", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    witness = helper.BatchInputWitness.from_env(rank=rank, world_size=world_size)
    training = importlib.import_module("megatron.training")
    original_pretrain = training.pretrain

    @functools.wraps(original_pretrain)
    def witnessed_pretrain(*args, **kwargs):
        bound = inspect.signature(original_pretrain).bind_partial(*args, **kwargs)
        forward = bound.arguments["forward_step_func"]
        original_get_batch = forward.__globals__["get_batch"]

        class ObservedIterator:
            def __init__(self, iterator):
                self.iterator = iterator

            def __iter__(self):
                return self

            def __next__(self):
                import torch
                batch = next(self.iterator)
                if torch.is_grad_enabled():
                    witness(batch["text"])
                return batch

        @functools.wraps(original_get_batch)
        def witnessed_get_batch(iterator, *get_args, **get_kwargs):
            observed = ObservedIterator(iterator) if iterator is not None else None
            return original_get_batch(observed, *get_args, **get_kwargs)

        forward.__globals__["get_batch"] = witnessed_get_batch
        return original_pretrain(*args, **kwargs)

    training.pretrain = witnessed_pretrain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--benchmark-module", dest="module")
    target.add_argument("--benchmark-script", "--script", dest="script", type=Path)
    parser.add_argument("--benchmark-output", "--output", dest="output", type=Path, required=True)
    parser.add_argument("--benchmark-dry-run", action="store_true")
    parser.add_argument("--benchmark-sophia-witness", action="store_true")
    args, child_args = parser.parse_known_args()
    if child_args[:1] == ["--"]:
        child_args = child_args[1:]
    if args.benchmark_dry_run:
        print(json.dumps({"module": args.module, "script": str(args.script), "child_argv": child_args}))
        return

    import torch
    from mpi4py import MPI

    witness = None
    if args.benchmark_sophia_witness:
        module = (
            "torchtitan.experiments.ezpz.optimizer.sophia"
            if args.module
            else "megatron.optimizer.sophia"
        )
        witness = SophiaStepWitness(module=module)

    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()
    if os.environ.get("BENCH_EXPECTED_WORLD_SIZE") and world_size != int(os.environ["BENCH_EXPECTED_WORLD_SIZE"]):
        raise ValueError(f"Unexpected MPI world size: {world_size}")
    if args.script:
        install_mds_batch_witness(rank, world_size)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    completed = False
    diagnostics_written = False

    def collect_diagnostics(*, exit_code=None):
        nonlocal diagnostics_written
        if diagnostics_written:
            return
        from torch._dynamo.utils import compile_times, get_compilation_metrics

        headers, values = compile_times(repr="csv", aggregate=True)
        result = {
            "rank": rank,
            "pbs_jobid": os.environ.get("PBS_JOBID"),
            "torch": torch.__version__,
            "target_module": args.module,
            "target_script": str(args.script) if args.script else None,
            "child_argv": child_args,
            "completed_without_exception": completed,
            "hard_exit_code": exit_code,
            "entrypoint_wall_seconds": time.monotonic() - started,
            "compile_timer_seconds": dict(zip(headers, values)),
            "compile_timer_note": "Nested cumulative timers; do not sum across keys.",
            "compilation_metrics": [
                dataclasses.asdict(metric) for metric in get_compilation_metrics()
            ],
        }
        if torch.xpu.is_initialized():
            # Final witness reductions must not inflate the training peak.
            result["xpu_peak_allocated_bytes"] = torch.xpu.max_memory_allocated()
            result["xpu_peak_reserved_bytes"] = torch.xpu.max_memory_reserved()
        if witness is not None:
            # Witness reductions synchronize the device; after a failure they
            # can wait indefinitely on queued work from the aborted step.
            result["sophia_witness"] = (
                witness.diagnostics()
                if completed
                else "skipped: training did not complete"
            )
        output = args.output / f"rank-{rank:03d}.json"
        output.write_text(json.dumps(result, indent=2, default=str) + "\n")
        diagnostics_written = True
        print(f"BENCHMARK_DIAGNOSTICS rank={rank} path={output}", flush=True)

    def write_diagnostics(*, exit_code=None):
        nonlocal diagnostics_written
        try:
            collect_diagnostics(exit_code=exit_code)
        except BaseException as error:
            # Diagnostics must never replace the training exception or exit code.
            fallback = {
                "rank": rank,
                "completed_without_exception": completed,
                "hard_exit_code": exit_code,
                "diagnostics_error": f"{type(error).__name__}: {error}",
            }
            try:
                output = args.output / f"rank-{rank:03d}.json"
                output.write_text(json.dumps(fallback, indent=2) + "\n")
                diagnostics_written = True
            except BaseException:
                pass
            try:
                print(f"BENCHMARK_DIAGNOSTICS_FAILED rank={rank} error={fallback['diagnostics_error']}", file=sys.stderr, flush=True)
            except BaseException:
                pass

    original_exit = os._exit
    original_pid = os.getpid()

    def record_hard_exit(code):
        nonlocal completed
        # TT deliberately hard-exits after closing training/W&B. Forked helper
        # processes must not overwrite the parent rank's diagnostics.
        if os.getpid() == original_pid:
            completed = code == 0
            write_diagnostics(exit_code=code)
        original_exit(code)

    os._exit = record_hard_exit
    sys.argv = [str(args.script) if args.script else args.module, *child_args]
    try:
        if args.script:
            runpy.run_path(str(args.script), run_name="__main__")
        else:
            runpy.run_module(args.module, run_name="__main__", alter_sys=True)
        completed = True
    except SystemExit as error:
        completed = error.code in (None, 0)
        raise
    except BaseException:
        # Report before diagnostics: their device reads can block forever
        # behind a collective that the failure left unfinished.
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        os._exit = original_exit
        write_diagnostics()


if __name__ == "__main__":
    main()
