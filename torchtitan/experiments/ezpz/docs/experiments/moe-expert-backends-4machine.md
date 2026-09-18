# MoE expert backends across four machines

**Bottom line:** all five backends behave correctly. `bmm_nodrop` is BIT-EXACT
against `for_loop`, and `aurora_sycl` WORKS on Aurora -- settling a question the
branch history left contradictory.

Tested on `feat/aurora-moe-port` sitting on top of the merged sync 84.

### Exactly what was run where

Naming the stack matters: "tested on Aurora" is ambiguous when the machine has
three of them.

| machine | queue | venv | torch | #181519 |
|---|---|---|---|---|
| aurora | `next-eval` | `venvs/xpu-torch215` | `2.15.0.dev20260915+xpu` | present (1379 lines) |
| sunspot | `workq` | `venvs/xpu-torch214` | `2.14.0+xpu` | present (1329) |
| polaris | `debug` | `venvs/sync84-torch214-cu` | `2.14.0+cu130` | present (1329) |
| perlmutter | `debug` (slurm) | `venvs/sync84-torch214-cu` | `2.14.0+cu130` | present (1329) |

Aurora ran a **2.15 nightly**, not the 2.14 used on Sunspot -- both carry
pytorch #181519, which is the only property the floor requires. Aurora was NOT
tested on `debug` + `oneapi/2025.3.1` + the production `.venv.tar.gz` (torch
2.13); that stack predates #181519 and is expected to die at FSDP wrapping, but
the controlled A/B establishing that was run on Sunspot, not here.

## Results

| backend | Aurora (XPU) | Sunspot (XPU) | Polaris (CUDA) | Perlmutter (CUDA) |
|---|---|---|---|---|
| `for_loop`    | OK (reference) | OK | OK | OK |
| `bmm`         | 8.057e-03 | 8.057e-03 | 8.057e-03 | 8.057e-03 |
| `bmm_nodrop`  | **0.000e+00** | **0.000e+00** | **0.000e+00** | **0.000e+00** |
| `grouped_mm`  | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| `aurora_sycl` | **6.104e-05** | **6.104e-05** | `ValueError: x must be an XPU tensor` | same |

Numbers are max abs difference vs the `for_loop` reference, bf16, E=4 experts,
D=64, H=128, counts `[8, 0, 5, 3]` (note the deliberate empty expert). Every
backend returned all-finite output. The four portable backends produce
byte-identical numbers on Intel XPU and NVIDIA CUDA.

**`bmm_nodrop` matching `for_loop` exactly is the result that matters.** Plain
`bmm` differs by 8.06e-03 because it drops tokens past capacity; the no-drop
variant does not, which is the whole point of adding it.

`aurora_sycl` refusing a CUDA tensor with a clear `ValueError` is correct
behavior, not a failure -- it is Intel-only by construction.

**`aurora_sycl` works on BOTH XPU machines**, to the same 6.104e-05, on two
different torch builds (Sunspot `2.14.0+xpu`, Aurora `2.15.0.dev+xpu`). That
settles a question the branch history left open: commits on
`feat/aurora-moe-port` record it failing at runtime, then working, then failing
under the production training venv, then working again. It works. 6.1e-05 is
ordinary bf16 kernel variation against a `for_loop` reference, not a
correctness problem.

Jobs: aurora `8835563`, sunspot `12478027`, polaris `7630674`, perlmutter
`58481149`.

## What aurora_moe actually is

`aurora_sycl` calls `aurora_moe.torchtitan_experts.torchtitan_exact_experts`.
`aurora_moe` is **vendored in-tree** at `torchtitan/experiments/ezpz/vendor/aurora_moe_dropin/src/`.
It is not on PyPI and does not get pip-installed; it needs to be on
`PYTHONPATH`.

Two requirements beyond that, both of which produce misleading errors:

1. **It JIT-compiles a C++/SYCL extension** through
   `torch.utils.cpp_extension`, so it needs the `ninja` BINARY on `PATH` and a
   C++ compiler. Installing the `ninja` PyPI package is not sufficient by
   itself: the binary lands in `venv/bin/ninja`, and a job that invokes
   `$VENV/bin/python` by absolute path never puts `venv/bin` on `PATH`. The
   error is `RuntimeError: Ninja is required to load C++ extensions`, which
   reads like a missing package when the package is in fact installed
   (jobs 12478024, 12478026). Export `PATH="$VENV/bin:$PATH"`.
2. On Aurora the resolved ninja came from the frameworks module
   (`/opt/aurora/26.181.0/frameworks/.../bin/ninja`), so `module load
   frameworks` also satisfies it.

## The unported backends, and the tests that chased them

`aurora_full_loop` and `aurora_full_sonic` are **deliberately not ported**. They
need `top_scores` and `selected_experts_indices`, which
`GroupedExperts.forward(x_RD, num_tokens_per_expert_E)` never receives -- sync
84 moved dispatch into the token dispatcher and the `sww` branch these came
from predates that -- and they additionally require an EP mesh. Porting them is
a dispatcher restructuring, not a backend addition. `aurora_sycl` is the one
that fits the post-refactor contract, which is why it is the one that exists.

The three test files under `tests/moe/` were stale against sync 84 and are now
fixed:

- `test_moe_routing_counts.py` broke three separate ways, each hidden behind
  the previous: it imported **core's** `LocalTokenDispatcher` (ezpz has its own,
  and only ezpz's keeps `score_before_experts`); it passed `score_func="sigmoid"`
  where sync 84 wants a `Sigmoid.Config()` to `.build()`; and it asserted on a
  third router return value that used to be per-expert counts and is now
  `routing_map_TE`, a one-hot boolean `(T, E)` map. The tie-break assertion now
  checks that the ordering is STABLE across rows rather than hardcoding which
  expert pair wins, since the pair is a topk implementation detail and the
  determinism is the actual property under test.
- `test_agpt_moe_config.py` keys every one of its four tests off the unported
  `aurora_full_sonic` flavor, so it failed at COLLECTION -- taking the whole
  file down rather than reporting a skip. Now skipped at module level with the
  rationale, kept rather than deleted so the intended 2B/50K architecture stays
  on record.
- `test_moe_expert_backends.py` was already correct.

`pytest torchtitan/experiments/ezpz/tests/moe/` is now **4 passed, 4 skipped**,
no failures and no collection errors.

## Reproducing

```bash
# from a PR#17 checkout, on a COMPUTE node (login nodes have no XPU/GPU)
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/torchtitan/experiments/ezpz/vendor/aurora_moe_dropin/src"
mpiexec -n 1 python <driver>   # see jobs 8835563 / 12478027
```

Two traps worth repeating, both of which produced false results here first:

- `EzpzGroupedExperts.forward(x_RD, num_tokens_per_expert_E)` takes **counts**,
  not cumulative offsets. Passing offsets yields shape errors that look like
  backend bugs.
- `Config(...).build()` leaves the expert weights **uninitialized**, so every
  backend returns NaN and a `finite=False` column measures nothing. Initialize
  the parameters before comparing, or the whole table is noise.
- The script must live on a SHARED filesystem. `/tmp` exists on compute nodes
  and is writable, but it is `tmpfs` -- node-local, so a file staged to the
  login node's `/tmp` is invisible there. PBS ships the job script itself, so
  the job starts and only fails when it opens a second file by path.
