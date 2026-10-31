#!/usr/bin/env python3
"""Check native-DDP accumulation and global token normalization on XPU."""

import argparse
import json
import os
from pathlib import Path

import ezpz
import torch
import torch.distributed as dist
from torch.nn import functional as F

from torchtitan.experiments.ezpz.native_ddp import NativeDDP, scale_native_ddp_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ezpz.setup_torch()
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != int(os.environ["BENCH_EXPECTED_WORLD_SIZE"]):
        raise ValueError(f"Unexpected world size: {world}")
    device = torch.device("xpu", ezpz.get_local_rank())
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    template = torch.nn.Linear(16, 8, bias=False).to(device)
    initial = template.weight.detach().clone()
    models = []
    for _ in range(2):
        model = torch.nn.Linear(16, 8, bias=False).to(device)
        model.weight.data.copy_(initial)
        models.append(NativeDDP(model, device_ids=[device.index], broadcast_buffers=False))
    generator = torch.Generator().manual_seed(1729)
    all_inputs = torch.randn(world, 2, 3, 16, generator=generator).to(device)
    all_labels = torch.randint(0, 8, (world, 2, 3), generator=generator).to(device)
    # Unequal valid-token counts exercise the actual global-token denominator.
    all_labels[0, 0, 0] = -100
    num_tokens = int((all_labels != -100).sum().item())
    accumulated, combined = models
    losses = []
    unsynchronized = None
    for microbatch in range(2):
        accumulated.set_requires_all_reduce(microbatch == 1)
        logits = accumulated(all_inputs[rank, microbatch])
        loss = F.cross_entropy(logits, all_labels[rank, microbatch], reduction="sum") / num_tokens
        scale_native_ddp_loss(loss, world).backward()
        losses.append(float(loss.detach().item()))
        if microbatch == 0:
            local_grad = accumulated.module.weight.grad.clone()
            partial_grads = [torch.empty_like(local_grad) for _ in range(world)]
            dist.all_gather(partial_grads, local_grad)
            unsynchronized = any(not torch.equal(partial_grads[0], item) for item in partial_grads[1:])
    combined.set_requires_all_reduce(True)
    logits = combined(all_inputs[rank].flatten(0, 1))
    loss = F.cross_entropy(logits, all_labels[rank].flatten(), reduction="sum") / num_tokens
    scale_native_ddp_loss(loss, world).backward()
    loss_pair = torch.tensor([sum(losses), float(loss.detach().item())], device=device)
    dist.all_reduce(loss_pair)
    loss_error = float((loss_pair[0] - loss_pair[1]).abs().item())
    grad = accumulated.module.weight.grad
    reference_grad = combined.module.weight.grad
    max_abs = float((grad - reference_grad).abs().max().item())
    # Accumulation changes FP32 summation order; the comparison is not bitwise.
    equivalent = torch.allclose(grad, reference_grad, atol=1e-6, rtol=1e-5)
    all_grads = [torch.empty_like(grad) for _ in range(world)]
    dist.all_gather(all_grads, grad)
    rank_agreement = all(torch.equal(all_grads[0], item) for item in all_grads[1:])
    result = dict(rank=rank, world_size=world, seed=42, deterministic=True,
                  num_valid_tokens=num_tokens, local_losses=losses,
                  first_microbatch_unsynchronized=unsynchronized,
                  final_gradients_identical_across_ranks=rank_agreement,
                  gas2_vs_combined_max_abs_grad=max_abs,
                  gas2_vs_combined_allclose=equivalent, atol=1e-6, rtol=1e-5,
                  global_loss_gas2=float(loss_pair[0].item()),
                  global_loss_combined=float(loss_pair[1].item()),
                  global_loss_abs_error=loss_error,
                  parameter_dtype=str(accumulated.module.weight.dtype),
                  gradient_dtype=str(grad.dtype), torch=torch.__version__)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank-{rank:03d}.json").write_text(json.dumps(result, indent=2) + "\n")
    passed = bool(unsynchronized and rank_agreement and equivalent and loss_error <= 1e-6)
    status = torch.tensor(int(passed), device=device)
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    if rank == 0:
        import wandb
        with wandb.init(project="torchtitan.ezpz.train", entity="aurora_gpt",
                        group="agpt-replicated-20261031", job_type="accumulation-probe",
                        config={"world_size": world, "torch": torch.__version__,
                                "source_sha": os.environ.get("BENCH_EXPECTED_TT_HEAD")}) as run:
            run.log({"probe/pass": int(status.item()), "probe/max_abs_grad": max_abs})
            artifact = wandb.Artifact("agpt-replicated-accumulation-" + os.environ["PBS_JOBID"].split(".")[0], type="verification")
            artifact.add_dir(str(args.output))
            run.log_artifact(artifact).wait()
    dist.barrier()
    print("ACCUMULATION_PROBE " + json.dumps(result), flush=True)
    if not status.item():
        raise RuntimeError("native DDP accumulation probe failed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
