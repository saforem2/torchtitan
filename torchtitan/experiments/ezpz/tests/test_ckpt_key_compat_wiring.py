"""Verify maybe_install_flat_attention_compat picks the right checkpoints.

The dangerous failure is a FALSE POSITIVE: installing the remap on a healthy
production chain would rename keys that do not need renaming. Every live chain
is checked here, not just the two forks.
"""
import sys
sys.path.insert(0, ".")
from torchtitan.experiments.ezpz.ckpt_key_compat import (  # noqa: E402
    maybe_install_flat_attention_compat,
)

RUNS = "/flare/AuroraGPT/foremans/runs"
CASES = [
    # (label, checkpoint folder, load_step, expect_shim)
    ("fork from9500 (trainer 4, BROKEN)",
     f"{RUNS}/agpt-2b-constlr-from9200/torchtitan-ezpz/outputs/checkpoints/"
     "agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144-constlr-from9500", -1, True),
    ("fork from9200 (trainer 3)",
     f"{RUNS}/agpt-2b-constlr-from9200/torchtitan-ezpz/outputs/checkpoints/"
     "agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288-constlr-from9200", -1, False),
    ("PROD 2B-512",
     f"{RUNS}/agpt-2b-v2/torchtitan-ezpz/outputs/checkpoints/"
     "agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288", -1, False),
    ("PROD 20B-512",
     f"{RUNS}/agpt-20b-v2/torchtitan-ezpz/outputs/checkpoints/"
     "agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288", -1, False),
    ("PROD 20B-256",
     f"{RUNS}/agpt-20b-n256/torchtitan-ezpz/outputs/checkpoints/"
     "agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144", -1, False),
    ("nonexistent folder", "/nonexistent/checkpoints/nope", -1, False),
]

MAIN = "/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz"
# The converter rewrote OPTIMIZER format only; the model keys in the t4 seed
# are still pre-refactor, so a seeded fresh chain needs the shim just as much
# as an in-place resume does.
T4_SEED = f"{MAIN}/outputs/checkpoints/_convert/t4-step9500-newfmt/step-9500"
CURRENT_FMT_SEED = (
    f"{RUNS}/agpt-2b-v2/torchtitan-ezpz/outputs/checkpoints/"
    "agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288/step-46429"
)
# (label, folder, initial_load_path, expect_shim)
SEED_CASES = [
    # The 8771774 failure: empty folder + pre-refactor seed. Probing only
    # `folder` found nothing and the load died on a missing qkv_linear key.
    ("empty folder + pre-refactor seed",
     "/nonexistent/checkpoints/_smoke/fresh", T4_SEED, True),
    # False positive guard: a current-format seed must be left alone.
    ("empty folder + current-format seed",
     "/nonexistent/checkpoints/_smoke/fresh", CURRENT_FMT_SEED, False),
    # Precedence: a resumable folder WINS over the seed (checkpoint.py:686), so
    # the verdict must come from the folder. Here the folder is current-format
    # and the seed is pre-refactor -- reading the seed would wrongly install.
    ("resumable current-format folder beats pre-refactor seed",
     f"{RUNS}/agpt-2b-v2/torchtitan-ezpz/outputs/checkpoints/"
     "agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288", T4_SEED, False),
    ("no folder, no seed", "/nonexistent/checkpoints/nope", "", False),
]


class FakeCheckpointer:
    """Minimal stand-in: only needs a dcp_load attribute to wrap."""

    def __init__(self):
        self.calls = []

    def dcp_load(self, state_dict, *a, **k):
        self.calls.append(state_dict)


fails = []
for label, folder, step, expect in CASES:
    ck = FakeCheckpointer()
    got = maybe_install_flat_attention_compat(ck, folder, step)
    wrapped = getattr(ck, "_flat_attention_compat", False)
    ok = (got == expect) and (wrapped == expect)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label:38s} shim={'ON ' if got else 'off'} "
          f"(expected {'ON' if expect else 'off'})")
    if not ok:
        fails.append(label)

print()
print("seed (initial_load_path) cases:")
for label, folder, seed, expect in SEED_CASES:
    ck = FakeCheckpointer()
    got = maybe_install_flat_attention_compat(
        ck, folder, -1, initial_load_path=seed
    )
    wrapped = getattr(ck, "_flat_attention_compat", False)
    ok = (got == expect) and (wrapped == expect)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label:52s} shim={'ON ' if got else 'off'} "
          f"(expected {'ON' if expect else 'off'})")
    if not ok:
        fails.append(label)

print()
# Idempotence: a second call must not double-wrap.
ck = FakeCheckpointer()
f = CASES[0][1]
maybe_install_flat_attention_compat(ck, f, -1)
first = ck.dcp_load
maybe_install_flat_attention_compat(ck, f, -1)
same = ck.dcp_load is first
print(f"  [{'ok  ' if same else 'FAIL'}] second install is a no-op (no double-wrap)")
if not same:
    fails.append("idempotence")

print()
if fails:
    print(f"FAILED: {fails}")
    sys.exit(1)
print("ALL PASS -- shim fires only on the pre-refactor fork; every production "
      "chain is untouched")
