# MoE expert backends across four machines

**Bottom line:** all five backends behave correctly. `bmm_nodrop` is BIT-EXACT
against `for_loop`, and `aurora_sycl` WORKS on Aurora -- settling a question the
branch history left contradictory.

Tested on PR #17 (`feat/aurora-moe-port`) sitting on top of the merged sync 84.

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
variant does not, which is the whole point of PR #17 adding it.

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
`aurora_moe` is **vendored inside PR #17** at `torchtitan/experiments/ezpz/vendor/aurora_moe_dropin/src/`.
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
