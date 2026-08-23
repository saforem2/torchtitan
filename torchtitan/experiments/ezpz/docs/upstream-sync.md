# Upstream Sync Log

> **This file is now an index.** Per-sync entries moved to
> `records/upstream-sync/YYYY-MM.md`. The standing notes below stay here,
> because they are current guidance rather than history. Every inbound link
> to this path still resolves.

## HEADS-UP for the 80th sync: `full_dtensor.py` is deleted upstream

`601cf4d23` (#4217, 2026-08-19) removes `torchtitan/distributed/full_dtensor.py`
entirely. Two ezpz files import from it and will fail at import time:

- `agpt/parallelize.py:44` -- `resolve_fsdp_mesh, validate_config`
- `moe/parallelize.py:50` -- `resolve_fsdp_mesh, resolve_sparse_fsdp_mesh, ...`

Where things went:

| symbol | upstream/main |
|---|---|
| `resolve_fsdp_mesh` | moved to `distributed/fsdp.py:32`, logic unchanged |
| `resolve_sparse_fsdp_mesh` | moved to `distributed/fsdp.py:65`, logic unchanged |
| `validate_config` | **GONE** -- no definition anywhere in `distributed/` |

**Already handled (`bf3c4f47d`), so the sync should not trip on this.**

`ezpz/fsdp_compat.py` resolves the two survivors from whichever location
exists -- post-#4217 `distributed.fsdp` first, falling back to
`distributed.full_dtensor` on this tree -- and both models import from it.

`validate_config` was NOT shimmed. Upstream removed it outright and its
callers just dropped the call (`llama3/parallelize.py:40` is now
`if spmd_backend == "spmd_types" or tp_enabled: model.parallelize(...)`;
deepseek_v3 likewise). Both ezpz call sites are dropped to match.

That is a behavior change, so it was verified on hardware rather than by an
import test (job 12473452), against the pre-change numbers:

| config | steps | memory | was |
|---|---|---|---|
| `agpt_20b` | 5/5 | 84.86% | 84.86% |
| `moe_small` | 5/5 | 72.70% | 72.55% |
| `moe_10b_2b` | 5/5 | 79.97% | 79.97% |
| `moe_10b_2b_sdpa` | 5/5 | 74.03% | 73.98% |

Identical to within 0.15pp, i.e. the sharding plan is unchanged -- which is
the property `validate_config` existed to guard.

Remaining at sync time: `"full_dtensor"` stops being a legal `spmd_backend`
value, so drop it from the `spmd_backend in (...)` tuples in both
`parallelize.py` files. Our config pins already moved to `partial_dtensor`
(`b2ff09632`).

## How to use this document

After each `git merge upstream/main`, check if the incoming commits touch:

1. **`models/llama3/`** — replay changes onto `experiments/ezpz/agpt/`
2. **`models/deepseek_v3/`** — replay changes onto `experiments/ezpz/moe/`
3. **`models/qwen3/`** — replay changes onto `experiments/ezpz/qwen3/`
4. **`models/common/`** — check if ezpz models depend on changed APIs
5. **`distributed/`** — check if ezpz trainer or parallelize files use removed/renamed APIs
6. **`trainer.py`** — check if ezpz trainer mirrors the same code path

Add an entry below with the date, upstream commits, what changed, and what
was required in ezpz.

---

## Sync history

- [2026-08](records/upstream-sync/2026-08.md) -- 6 syncs
- [2026-07](records/upstream-sync/2026-07.md) -- 13 syncs
- [2026-06](records/upstream-sync/2026-06.md) -- 20 syncs
- [2026-05](records/upstream-sync/2026-05.md) -- 15 syncs
- [2026-04](records/upstream-sync/2026-04.md) -- 28 syncs
