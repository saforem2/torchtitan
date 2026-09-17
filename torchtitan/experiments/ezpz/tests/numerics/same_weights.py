"""Decisive test: give BOTH trees bit-identical weights, then compare outputs.

The earlier A/B differed because #4535's fused w13 consumes the RNG stream in a
different order than separate w1/w3 -- different DRAW, not different math. This
removes init from the equation: build on the pre-merge tree, save the state
dict (whose keys the hooks keep as logical w1/w3 on BOTH sides), load it into
the other tree, and compare the forward.

If the merge is numerically sound, the remaining delta is float-regrouping
scale (one fused GEMM vs three) and nothing more.
"""
import sys, json, os
tree = sys.argv[1]
mode = sys.argv[2]          # "save" or "load"
path = sys.argv[3]
sys.path.insert(0, tree)

import torch
import torchtitan
assert tree in torchtitan.__file__, torchtitan.__file__
from torchtitan.experiments.ezpz.agpt import agpt_configs, set_ezpz_max_context_length

T = 256
SEED = 42
set_ezpz_max_context_length(T)
torch.manual_seed(SEED)

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = agpt_configs["debugmodel"].build()
model.init_states()

if mode == "save":
    torch.save(model.state_dict(), path)
else:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=True), None
    print("LOADSTATUS" + json.dumps({"missing": list(getattr(missing, "missing_keys", [])),
                                     "unexpected": list(getattr(missing, "unexpected_keys", []))}))

model = model.to(dev).to(torch.float32).eval()
g = torch.Generator(device="cpu").manual_seed(SEED)
vocab = model.tok_embeddings.weight.shape[0]
ids = torch.randint(0, vocab, (T,), generator=g).to(dev)

with torch.no_grad():
    logits = model(ids)

out = {"tree": os.path.basename(tree.rstrip("/")), "mode": mode,
       "logits_sum": float(logits.double().sum()),
       "logits_absmax": float(logits.abs().max()),
       "logits_mean": float(logits.double().mean()),
       "logits_std": float(logits.double().std())}
torch.save(logits.cpu(), path + f".logits.{os.path.basename(tree)}")
print("JSONLINE" + json.dumps(out))
