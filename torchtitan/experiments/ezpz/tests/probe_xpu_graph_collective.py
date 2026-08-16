"""Can a oneCCL collective be captured in an XPU graph at all?

The trainer failed with:
  RuntimeError: wait method cannot be used for an event associated with a
                command graph.
raised from group.all_gather_single inside capture. Test the primitive
directly: same collective, once OUTSIDE capture, once INSIDE.
"""
import os, torch, torch.distributed as dist, ezpz

rank = ezpz.setup_torch()
world = dist.get_world_size()
dev = torch.device(f"xpu:{torch.xpu.current_device()}")
if rank == 0:
    print(f"world={world} torch={torch.__version__}", flush=True)

x = torch.ones(1024, dtype=torch.bfloat16, device=dev)
out = torch.empty(1024 * world, dtype=torch.bfloat16, device=dev)

# 1. outside capture -- control
dist.all_gather_into_tensor(out, x); torch.xpu.synchronize()
if rank == 0: print("  all_gather OUTSIDE capture: OK", flush=True)

# 2. inside capture
g = torch.xpu.XPUGraph()
s = torch.xpu.Stream()
try:
    with torch.xpu.graph(g, stream=s):
        dist.all_gather_into_tensor(out, x)
    torch.xpu.synchronize()
    if rank == 0: print("  all_gather INSIDE  capture: OK -- collectives ARE capturable", flush=True)
except Exception as e:
    if rank == 0:
        print(f"  all_gather INSIDE  capture: FAILS -> {type(e).__name__}: {str(e)[:100]}", flush=True)

# 3. pure compute inside capture -- proves capture itself works here
g2 = torch.xpu.XPUGraph(); s2 = torch.xpu.Stream()
y = torch.randn(512, 512, device=dev); z = torch.empty_like(y)
try:
    with torch.xpu.graph(g2, stream=s2):
        z.copy_(y @ y)
    g2.replay(); torch.xpu.synchronize()
    if rank == 0: print("  matmul     INSIDE  capture: OK -- capture works, collectives are the issue", flush=True)
except Exception as e:
    if rank == 0: print(f"  matmul     INSIDE  capture: FAILS -> {str(e)[:90]}", flush=True)

dist.barrier()
if rank == 0: print("CCL_CAPTURE_PROBE_DONE", flush=True)
