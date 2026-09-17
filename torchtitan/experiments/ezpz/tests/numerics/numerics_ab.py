"""Seeded fused-vs-unfused QKV/gate-up numerics A/B across the sync-84 merge.

Runs ONE tree per invocation (argv[1] = tree root) so the two trees never share
a process -- torchtitan is imported by absolute name and cannot be loaded twice.
Emits a JSON line the caller diffs.

What this actually tests: #4526 made QKV fusion mandatory (1 GEMM, not 3) and
#4535 made gate-up fusion the default. Both change the arithmetic grouping, so
bitwise equality is NOT expected; the question is whether the difference is at
float-rounding scale or something larger.
"""
import sys, json, os
tree = sys.argv[1]
sys.path.insert(0, tree)

import torch
import torchtitan
assert tree in torchtitan.__file__, torchtitan.__file__

from torchtitan.experiments.ezpz.agpt import (
    agpt_configs,
    set_ezpz_max_context_length,
)

FLAVOR = os.environ.get("AB_FLAVOR", "debugmodel")
SEED = 42
T = 256

# The ezpz SDPA wrapper unflattens a 3D [T, H, K] batch with this and
# raises if the trainer never set it. Pin it to the token count so the
# synthetic batch is exactly one sequence -- identical on both trees.
set_ezpz_max_context_length(T)

torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True, warn_only=False)

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = agpt_configs[FLAVOR]
model = cfg.build()
model.init_states()
model = model.to(dev).to(torch.float32)
model.eval()

# fixed synthetic batch -- same tokens both sides, derived from SEED only
g = torch.Generator(device="cpu").manual_seed(SEED)
vocab = model.tok_embeddings.weight.shape[0]
ids = torch.randint(0, vocab, (T,), generator=g).to(dev)

out = {"tree": os.path.basename(tree.rstrip("/")), "flavor": FLAVOR,
       "device": dev, "torch": torch.__version__,
       "params": sum(p.numel() for p in model.parameters())}

with torch.no_grad():
    logits = model(ids)
    out["logits_shape"] = list(logits.shape)
    out["logits_sum"] = float(logits.double().sum())
    out["logits_absmax"] = float(logits.abs().max())
    out["logits_mean"] = float(logits.double().mean())
    out["logits_std"] = float(logits.double().std())

# one backward, so the fused paths are exercised in reverse too
model.train()
logits = model(ids)
loss = logits.float().pow(2).mean()
loss.backward()
gn = torch.sqrt(sum((p.grad.double()**2).sum() for p in model.parameters() if p.grad is not None))
out["loss"] = float(loss)
out["grad_norm"] = float(gn)
# per-tensor fingerprint of the largest few, to localize any divergence
tops = sorted(((float(p.grad.double().norm()), n) for n,p in model.named_parameters()
               if p.grad is not None), reverse=True)[:5]
out["top5_grad"] = [[n, v] for v,n in tops]

print("JSONLINE" + json.dumps(out))
