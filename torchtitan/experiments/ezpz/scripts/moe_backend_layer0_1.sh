#!/bin/bash
# Layer 0 (CPU unit tests) + Layer 1 (XPU probes) for the MoE backend
# comparison. Run on a compute node, not the login node.
# DON'T use `bash --login` and DON'T `set -euo pipefail` — venv activate
# has unbound vars that trip both.
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
cd /lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz
source <(curl -fsSL https://bit.ly/ezpz-utils)
ezpz_setup_job >/dev/null 2>&1
ezpz_setup_xpu >/dev/null 2>&1
source .venv/bin/activate

echo "=== LAYER 0: CPU unit tests ==="
python3 -m unittest torchtitan.experiments.ezpz.tests.test_moe_expert_backends -v 2>&1 | tail -30

echo
echo "=== LAYER 1a: grouped_mm on XPU ==="
python3 - <<'PYEOF'
import torch
print("xpu available:", torch.xpu.is_available(), "count:", torch.xpu.device_count())
x = torch.randn(14, 16, device="xpu", dtype=torch.bfloat16)
w = torch.randn(4, 16, 32, device="xpu", dtype=torch.bfloat16)
offs = torch.tensor([4, 4, 11, 14], device="xpu", dtype=torch.int32)
try:
    out = torch._grouped_mm(x, w, offs=offs)
    print("grouped_mm works on XPU:", out.shape, out.dtype)
except Exception as e:
    print("grouped_mm fails on XPU:", type(e).__name__, repr(str(e))[:400])
PYEOF

echo
echo "=== LAYER 1b: for_loop vs batched_mm_padded parity on XPU ==="
python3 - <<'PYEOF'
import torch
from torchtitan.experiments.ezpz.moe.experts import (
    _run_experts_for_loop, _run_experts_batched_mm_padded
)
torch.manual_seed(0)
counts = torch.tensor([4, 0, 7, 3], device="xpu")
w1 = torch.randn(4, 32, 16, device="xpu") * 0.02
w2 = torch.randn(4, 16, 32, device="xpu") * 0.02
w3 = torch.randn(4, 32, 16, device="xpu") * 0.02
x = torch.randn(14, 16, device="xpu")
out_l = _run_experts_for_loop(w1, w2, w3, x, counts)
out_b = _run_experts_batched_mm_padded(w1, w2, w3, x, counts)
diff = (out_l - out_b).abs().max().item()
print("max abs diff:", diff)
print("match (atol=1e-4):", torch.allclose(out_l, out_b, atol=1e-4))
PYEOF
