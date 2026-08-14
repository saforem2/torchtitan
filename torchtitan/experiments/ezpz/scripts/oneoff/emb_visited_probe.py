"""Restrict to embedding rows the model ACTUALLY saw, using the same data.

Runs the debugmodel briefly, records which vocab rows received nonzero
gradient, then reports frac_changed over ONLY those rows -- removing the
"unvisited vocab row" confound entirely.
"""
import os, json, torch
os.environ.setdefault("TORCH_DEVICE","cpu")
import sys
sys.argv = ["x"]
from torchtitan.config import ConfigManager
from torchtitan.experiments.ezpz.logging import init_logger
init_logger()

ARM = os.environ["ARM"]
dt = "float32" if ARM == "C" else "bfloat16"
if ARM == "B":
    os.environ["EZPZ_FP32_NORMS"] = "1"
steps = int(os.environ.get("STEPS","30"))

argv = ["--module","ezpz.agpt","--config","agpt_debugmodel_local",
        f"--training.dtype={dt}", f"--training.steps={steps}",
        "--compile.no-enable","--checkpoint.no-enable","--validator.no-enable",
        "--debug.seed","42","--debug.deterministic",
        "--metrics.log-freq=100","--metrics.no-enable-wandb",
        "--lr-scheduler.warmup-steps=5"]
cfg = ConfigManager().parse_args(argv)
from torchtitan.components.optimizer import default_adamw
cfg.optimizer = default_adamw(lr=8e-4)
tr = cfg.build()

emb = tr.model_parts[0].tok_embeddings.weight
def _full(t):
    t = t.detach()
    if hasattr(t, "full_tensor"): t = t.full_tensor()
    return t.float().cpu()
before = _full(emb).clone()
touched = torch.zeros(emb.shape[0], dtype=torch.bool)

orig = tr.train_step
def step(*a, **k):
    out = orig(*a, **k)
    g = emb.grad
    if g is not None:
        gg = g.detach()
        if hasattr(gg,"full_tensor"): gg = gg.full_tensor()
        touched.logical_or_((gg.float().abs().sum(dim=1) > 0).cpu())
    return out
tr.train_step = step
tr.train()

after = _full(emb)
rows = touched.cpu().nonzero().flatten()
sub_b, sub_a = before[rows], after[rows]
moved = (sub_a != sub_b).float().mean().item()
print(json.dumps({
  "arm": ARM, "dtype": str(emb.dtype), "steps": steps,
  "rows_touched": int(rows.numel()), "vocab": int(emb.shape[0]),
  "frac_moved_touched_rows": moved,
  "frac_moved_all": float((after != before).float().mean()),
  "max_abs_delta_touched": float((sub_a-sub_b).abs().max()),
}))
