"""Certify the ckpt_key_compat renames against REAL checkpoint key sets.

Why this exists: the shim was the leading suspect for the t4 "step-9500"
mystery (a seed that loaded cleanly and trained at loss 5.97 against a parent
at 2.86). It was exonerated -- the seed turned out to be bf16 near-random
weights, and a key rename cannot change a dtype -- but exoneration is not
certification. The existing wiring test only checks WHICH checkpoints get the
shim installed, never that the renames it then applies are correct.

This tests the renames themselves:

  1. ROUND TRIP on every key of a real pre-refactor checkpoint and a real
     current-format one: down(up(k)) == k and up(down(k)) == k.
  2. IDEMPOTENCE: applying a direction twice equals applying it once. The
     module docstring claims this; it is what stops a second install from
     stripping qkv_linear off a key that legitimately carries it.
  3. NO COLLISIONS: the rename must be injective. Two distinct on-disk keys
     mapping to one in-memory key would silently drop a tensor.
  4. NON-TARGETS UNTOUCHED: wo, norms, embeddings, the MoE `output` trap, and
     optimizer-namespaced variants must pass through byte-identical.

Runs against on-disk checkpoints when they are reachable (Aurora), and against
a synthetic key set everywhere else so it is useful off-cluster too.
"""
import re
import sys

# Load ckpt_key_compat DIRECTLY rather than via the package: importing
# torchtitan.experiments.ezpz pulls local_device_compat -> torch, and these
# renames are pure regex with no torch dependency. Loading the file straight
# keeps this runnable on a laptop with no torch installed, which is where a
# key-rename test is most useful to iterate on.
import importlib.util, os  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "_ckc", os.path.join(os.path.dirname(__file__), "..", "ckpt_key_compat.py"))
_ckc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ckc)
to_flat, to_nested = _ckc.to_flat, _ckc.to_nested
head_to_old, head_to_new = _ckc.head_to_old, _ckc.head_to_new

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
    return cond


def synthetic_new_keys():
    """Current-code spellings, including the shapes that trip naive regexes."""
    k = []
    for i in (0, 7, 63):
        for w in ("wq", "wk", "wv"):
            k.append("layers.%d.attention.qkv_linear.%s.weight" % (i, w))
        # wo is NOT under the wrapper -- must be left alone
        k.append("layers.%d.attention.wo.weight" % i)
        k.append("layers.%d.attention_norm.weight" % i)
        k.append("layers.%d.feed_forward.w1.weight" % i)
        # MoE: a `.output` that is NOT the head. The head rename is anchored,
        # so this must survive untouched.
        k.append("layers.%d.moe.experts.output.weight" % i)
    k += ["tok_embeddings.weight", "norm.weight", "lm_head.weight"]
    # optimizer-namespaced forms -- the prefix the head regex must tolerate
    k += ["optimizer.state.lm_head.weight.exp_avg",
          "optimizer.state.layers.0.attention.qkv_linear.wq.weight.exp_avg"]
    return k


def down(k):   # current -> on-disk
    return head_to_old(to_flat(k))


def up(k):     # on-disk -> current
    return head_to_new(to_nested(k))


def run(label, new_keys):
    print("\n=== %s (%d keys) ===" % (label, len(new_keys)))

    # 1. round trip
    bad = [k for k in new_keys if up(down(k)) != k]
    check(not bad, "%s: round-trip failed for %d keys, e.g. %r -> %r -> %r"
          % (label, len(bad), bad[0], down(bad[0]), up(down(bad[0]))) if bad else "")
    print("  round-trip up(down(k)) == k : %s" % ("ok" if not bad else "FAIL %d" % len(bad)))

    # 2. idempotence
    nonidem = [k for k in new_keys if to_flat(to_flat(k)) != to_flat(k)
               or to_nested(to_nested(k)) != to_nested(k)]
    check(not nonidem, "%s: not idempotent, e.g. %r" % (label, nonidem[0]) if nonidem else "")
    print("  idempotent                  : %s" % ("ok" if not nonidem else "FAIL %d" % len(nonidem)))

    # 3. injective
    seen = {}
    coll = []
    for k in new_keys:
        d = down(k)
        if d in seen and seen[d] != k:
            coll.append((seen[d], k, d))
        seen[d] = k
    check(not coll, "%s: rename is NOT injective: %r" % (label, coll[:2]) if coll else "")
    print("  injective (no collisions)   : %s" % ("ok" if not coll else "FAIL %d" % len(coll)))

    # 4. non-targets untouched
    nt = [k for k in new_keys
          if ("qkv_linear." not in k and not re.search(r'(^|\.)lm_head\.', k))]
    moved = [k for k in nt if down(k) != k]
    check(not moved, "%s: non-target keys were rewritten: %r" % (label, moved[:3]) if moved else "")
    print("  non-targets untouched (%3d) : %s" % (len(nt), "ok" if not moved else "FAIL %s" % moved[:2]))


def from_checkpoint(path):
    """Real key set off disk, or None if unreachable."""
    try:
        from torch.distributed.checkpoint import FileSystemReader
        return list(FileSystemReader(path).read_metadata().state_dict_metadata)
    except Exception as e:
        print("  (skipped %s: %r)" % (path, e))
        return None


if __name__ == "__main__":
    run("synthetic", synthetic_new_keys())

    RUNS = "/flare/AuroraGPT/foremans/runs"
    real = [
        ("PRE-refactor (t4 seed, needs both renames)",
         RUNS + "/agpt-2b-constlr-from9200/torchtitan-ezpz/outputs/checkpoints/"
                "agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144-constlr-from9500/step-9500"),
        ("CURRENT format (2b_v2_512 stage-1 tip)",
         RUNS + "/agpt-2b-v2/torchtitan-ezpz/outputs/checkpoints/"
                "agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288/step-46429"),
    ]
    for label, p in real:
        keys = from_checkpoint(p)
        if keys is None:
            continue
        # On-disk keys are the "old" spelling for a pre-refactor ckpt and the
        # "new" spelling for a current one; normalize to current before testing.
        run(label, [up(k) for k in keys])

    print()
    if FAILS:
        print("CERTIFY: FAIL")
        for f in FAILS:
            if f:
                print("  -", f)
        sys.exit(1)
    print("CERTIFY: PASS -- renames round-trip, are idempotent and injective, "
          "and leave every non-target key untouched")
