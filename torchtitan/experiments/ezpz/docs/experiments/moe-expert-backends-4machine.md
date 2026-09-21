# MoE expert backends across four machines

**Bottom line:** current HEAD registers six expert compute backends. The four
portable paths were compared locally/on CUDA, while the two SYCL paths require
Intel XPU hardware. In the recorded XPU runs, `bmm_nodrop` was bit-exact against
`for_loop`, `aurora_sycl` ran on both XPU machines, and `aurora_full_sonic`
trained at ~5e-05 from the reference. These are historical hardware results,
not claims that every backend can execute on every platform.

Correctness is settled. A matched compiled Sunspot run now gives the first
performance comparison: Sonic is roughly even with BMM in the initially stable
steps, but 21% slower end-to-end because both arms show large mid-run compile
stalls. See "Performance" at the end; this is directional, not a production-
shape EP=12 result.

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
| `aurora_full_sonic` | **~5e-05** (EP=2, job `8837281`) | **training + perf measured** (EP=2, job `12478121`) | XPU-only | XPU-only |

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

## aurora_full_sonic: PORTED AND TRAINING

**Superseded:** this section previously said `aurora_full_sonic` was
"deliberately not ported" because the routing tensors "never reach" our
forward. That reasoning was wrong on both counts -- the tensors are available,
and the backend now trains.

The routing tensors DO reach us, one layer above the expert backend: core's
`RoutedExperts.forward` receives `topk_scores_TK` and `topk_expert_ids_TK`
(`models/common/moe.py:143-144`) and discards them at line 163 before calling
`inner_experts`. `EzpzRoutedExperts` forwards them instead. No dispatcher
restructuring was needed.

Two feasibility questions were settled first, both on hardware:

| question | job | answer |
|---|---|---|
| EP>1 on torch 2.15? | `8836014` | yes -- losses match EP=1 to ~4 decimals, no `ur_die` |
| rank orders agree? | `8836277` | yes at EP=2, 8 AND 12 |

The second mattered because `create_dp_ep_groups` assumes
`rank = dp_rank * EP + ep_rank`; a mismatch would have crossed ranks in the
all-to-all with no error, only wrong gradients.

Three integration bugs then surfaced in sequence, each only after the previous
was fixed -- the port chain runs further every time:

1. BF16 router scores (`c59d20182`). The kernel hard-requires bf16 for both `x`
   and `topk_scores`. Cast the scores at the boundary; `x` deliberately raises
   instead, since wrong-dtype activations mean a real model problem.
2. `local_count` is experts PER RANK, not the total (`fdbd5eee3`). The kernel
   derives `dest = topk_indices // local_count`, so the full count made every
   local id exceed its range.
3. **The backward segfault was a JIT race** (job `8837164`, `rc=139`, crash in
   `_engine_run_backward`). `aurora_moe` JIT-compiles four SYCL extensions on
   first use, and 24 ranks doing that concurrently corrupted the backward --
   which is why it presented as SIGSEGV rather than a clean error. Prebuilding
   them fixes it: `aurora_moe.build.prebuild_torchtitan_full_sonic()` inside an
   XPU allocation (~6 min, job `8837246`), then point
   `AURORA_MOE_SYCL_BUILD_DIR` at the result. The dir must be on a shared
   filesystem so every rank reads the same `.so`.

## It trains

`8837281`, `next-eval`, 2 nodes / 24 ranks, EP=2, `rc=0`, zero non-finite:

```
step: 1  loss: 12.95230
step: 2  loss: 12.59623
step: 3  loss: 11.47120
```

| step | EP=2 reference (`8836014`) | sonic | delta |
|---|---|---|---|
| 1 | 12.95224 | 12.95230 | 6e-05 |
| 2 | 12.59619 | 12.59623 | 4e-05 |
| 3 | 11.47116 | 11.47120 | 4e-05 |

**This agreement is the result, not `rc=0`.** ~5e-05 is bf16 kernel variation,
the same magnitude `aurora_sycl` shows (6.104e-05). It says the expert-id
mapping, the `ParallelMesh` built from torchtitan's EP group, and the
all-to-all are all routing tokens to the correct experts. A wrong mapping
would have produced plausible-but-different losses -- running while silently
wrong -- and it did not.

### Required to run it

```bash
export AURORA_MOE_ALLTOALLV=1                 # the kernel refuses without it
export AURORA_MOE_SYCL_BUILD_DIR=<shared dir> # prebuilt kernels, see above
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
# EP>1 is mandatory: sonic owns its own expert-parallel all-to-all
--config=moe_debugmodel_sonic        # EP=2
--config=moe_10b_2b_sdpa_sonic_ep    # EP=12, mirrors bmm_ep for timing
```

Expect a long startup: `torch.compile` graph-breaks on every SYCL custom op,
so the first steps take minutes. Job `8837281` ran ~20 min wall for 3 steps.
The smaller Sunspot comparison below also shows compile/recompile cliffs within
the measured window, so individual step TPS must not be presented as a single
steady-state throughput number.

`aurora_full_loop` remains unported and is a separate question.

## Current test coverage

The tests under `tests/moe/` now separate host-testable contracts from XPU-only
execution:

- `test_moe_routing_counts.py` broke three separate ways, each hidden behind
  the previous: it imported **core's** `LocalTokenDispatcher` (ezpz has its own,
  and only ezpz's keeps `score_before_experts`); it passed `score_func="sigmoid"`
  where sync 84 wants a `Sigmoid.Config()` to `.build()`; and it asserted on a
  third router return value that used to be per-expert counts and is now
  `routing_map_TE`, a one-hot boolean `(T, E)` map. The tie-break assertion now
  checks that the ordering is STABLE across rows rather than hardcoding which
  expert pair wins, since the pair is a topk implementation detail and the
  determinism is the actual property under test.
- `test_agpt_moe_config.py` no longer imports the removed
  `AGPT_2B_50K_MOE_sdpa_aurora_full_sonic` flavor. It covers every current
  Sonic trainer config, EP requirements, `EzpzRoutedExperts` wiring, the six
  registered backend names, registry ownership, checkpoint adapter contracts,
  and meta-device weight shapes for non-square D/F models.
- `test_moe_expert_backends.py` covers portable numerical backends and the
  Sonic weight-layout conversion without invoking SYCL.

Sonic forward/backward, all-to-all rank mapping, and `aurora_sycl` kernel
execution remain explicitly XPU-only. Their historical jobs are recorded above;
they are not implied by a local pytest pass.

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

## Performance

### First matched result: Sunspot debugmodel, EP=2

Jobs `12478114` (BMM) and `12478121` (Sonic) form a controlled comparison:

- Sunspot `workq`, `venvs/xpu-torch214`, torch `2.14.0+xpu`
- commit `fb062f51aa443086b254096fbec11f14c14ecd8f`
- 1 node / 12 ranks, EP=2, seed 42
- max context and tokens per microbatch per DP rank both 512
- 20 steps, compile ON, activation checkpointing off
- `CCL_SYCL_KERNEL_SYNC=0` in both arms

| metric | `bmm` (`12478114`) | `aurora_full_sonic` (`12478121`) | Sonic vs BMM |
|---|---:|---:|---:|
| return code | 0 | 0 | -- |
| wall time | 103 s | 125 s | **21.4% slower** |
| mean TPS, steps 2-6 | 3,332 | 3,307 | **0.8% slower** |
| median TPS, steps 2-6 | 3,382 | 3,235 | **4.3% slower** |
| mean TPS, steps 17-20 | 1,148 | 3,718 | not comparable; BMM recompiled/stalled |
| final loss | 9.29467 | 9.28677 | -0.00790 |
| maximum absolute loss delta | -- | 0.01107 | at step 19 |
| peak memory | 1.45 GiB | 1.46 GiB | +0.01 GiB |
| oneCCL occupancy errors | 0 | 0 | -- |
| segfaults | 0 | 0 | -- |

The **steps 2-6 window is the cleanest directional comparison** before either
trace hits a severe compile/recompile cliff. It shows no Sonic speedup at this
small shape: mean throughput is effectively tied (-0.8%), while total wall
clock is 21.4% worse for Sonic. The complete traces are highly non-stationary:
BMM falls to 326-332 TPS at steps 18-20, while Sonic falls to 90-473 TPS at
steps 7-16 before recovering to 3,428-3,832 TPS. Consequently neither the
whole-run arithmetic mean nor the final four steps are defensible as a stable
kernel benchmark.

The losses remain close but are not bit-identical. Across 20 steps the mean
absolute delta is 0.00192 and the maximum is 0.01107 at step 19. Final losses
are 9.29467 (BMM) and 9.28677 (Sonic). This is consistent with the previously
established bf16/reduction-order drift; there are no non-finite values or
qualitative divergence.

### oneCCL issue exposed by the first attempt

The original two-node comparison, job `12478094`, was not a valid performance
result. Its BMM arm reached step 10 and then rank 4 segfaulted in compiled
backward; Sonic never started. The log contained 360 instances of:

```text
oneCCL: allgatherv_small_sycl_impl.hpp:195 operator(): EXCEPTION:
sycl threads : 4096 > hw threads : 3584 is not allowed in allgatherv small
```

A 1-node control (`12478113`) reproduced 252 of these messages but completed 20
steps without a segfault, proving that reducing the topology alone does not
remove the invalid oneCCL small-allgather path and that the error does not
predict a deterministic crash. The matched BMM control (`12478114`) set:

```bash
export CCL_SYCL_KERNEL_SYNC=0
```

and completed with zero occupancy errors and zero segfaults. The Sonic arm used
the same setting. This tells oneCCL to split synchronization across multiple
kernels instead of requiring all 4,096 SYCL threads to be resident
simultaneously on a 3,584-thread PVC tile.

### What remains

This is a valid first performance comparison, but not the production-shape
answer. The remaining benchmark is
`moe_10b_2b_sdpa_bmm_ep` versus `moe_10b_2b_sdpa_sonic_ep` at EP=12, with the
same node count, compile enabled, `CCL_SYCL_KERNEL_SYNC=0`, and enough post-
compile steps to identify a genuinely stable window. The failed Aurora jobs
`8837353` and `8837465` do not answer that question: the former spent its
90-minute allocation compiling and logged no steps; the latter disabled
compile, logged no steps in 48 minutes, and was deliberately deleted.
