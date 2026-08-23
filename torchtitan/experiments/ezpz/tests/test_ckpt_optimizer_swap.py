"""Optimizer-key swap: does it fire, rename, and clean up?

Uses a fake checkpointer + fake optimizer container, so it runs without torch.
"""
import importlib.util
import os
import sys

_spec = importlib.util.spec_from_file_location(
    "ckc", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "ckpt_key_compat.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

fails = []
def check(name, cond, detail=""):
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond: fails.append(name)


class FakeOpt:
    """Emits CONTAINER-level keys, exactly as OptimizersContainer does."""
    def state_dict(self):
        return {
            "state.layers.0.attention.qkv_linear.wq.weight.step": "S1",
            "state.layers.0.attention.wo.weight.step": "S2",
            "state.lm_head.weight.exp_avg": "S3",
            "param_groups.lm_head.weight.lr": "S4",
            "param_groups.layers.0.moe.output.weight.lr": "S5",
        }
    def load_state_dict(self, sd):
        self.loaded = sd


class FakeCkpt:
    def __init__(self):
        self.seen = None
        self.opt_keys_at_load = None
    def dcp_load(self, state_dict, *a, **kw):
        self.seen = dict(state_dict)
        o = state_dict.get("optimizer")
        if o is not None:
            self.opt_keys_at_load = sorted(o.state_dict().keys())


print("=== head regex now matches CONTAINER-level keys ===")
check("state.lm_head -> state.output",
      m.head_to_old("state.lm_head.weight.exp_avg") == "state.output.weight.exp_avg",
      m.head_to_old("state.lm_head.weight.exp_avg"))
check("param_groups.lm_head -> output",
      m.head_to_old("param_groups.lm_head.weight.lr") == "param_groups.output.weight.lr")
check("MoE trap still NOT touched",
      m.head_to_new("param_groups.layers.0.moe.output.weight.lr")
      == "param_groups.layers.0.moe.output.weight.lr")
check("composite spelling still works",
      m.head_to_old("optimizer.state.lm_head.weight.exp_avg")
      == "optimizer.state.output.weight.exp_avg")

print()
print("=== swap fires on the NORMAL path (model keys need renaming) ===")
c = FakeCkpt(); opt = FakeOpt()
m.install_flat_attention_compat(c, attention=True, head=True)
sd = {"lm_head.weight": "M", "layers.0.attention.qkv_linear.wq.weight": "M2",
      "optimizer": opt}
c.dcp_load(sd)
got = c.opt_keys_at_load or []
check("optimizer emitted FLAT attention key",
      "state.layers.0.attention.wq.weight.step" in got)
check("optimizer emitted OLD head key", "state.output.weight.exp_avg" in got)
check("no nested attention key leaked",
      not any("qkv_linear" in k for k in got))
check("MoE key untouched",
      "param_groups.layers.0.moe.output.weight.lr" in got)

print()
print("=== swap ALSO fires on the EARLY-RETURN path (the reviewer's blocker) ===")
c2 = FakeCkpt(); opt2 = FakeOpt()
m.install_flat_attention_compat(c2, attention=True, head=True)
# top-level keys that _down leaves alone -> early return
sd2 = {"norm.weight": "M", "optimizer": opt2}
c2.dcp_load(sd2)
got2 = c2.opt_keys_at_load or []
check("early-return path STILL renamed optimizer keys",
      "state.layers.0.attention.wq.weight.step" in got2,
      f"got {got2[:2]}")

print()
print("=== cleanup: save path must see the REAL method again ===")
check("state_dict restored after load (normal path)",
      opt.state_dict() == FakeOpt().state_dict())
check("state_dict restored after load (early-return path)",
      opt2.state_dict() == FakeOpt().state_dict())

print()
print("=== no optimizer in state_dict is a no-op, not a crash ===")
c3 = FakeCkpt()
m.install_flat_attention_compat(c3, attention=True, head=True)
try:
    c3.dcp_load({"lm_head.weight": "M"})
    check("model-only load works", True)
except Exception as e:
    check("model-only load works", False, repr(e))

print()
if fails:
    print(f"FAILED: {fails}"); sys.exit(1)
print("ALL PASS")
