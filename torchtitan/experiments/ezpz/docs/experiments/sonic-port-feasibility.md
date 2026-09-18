# Porting aurora_full_sonic: feasibility, settled

**Bottom line:** both blockers are cleared. EP>1 runs on torch 2.15, and
torchtitan's EP rank ordering matches what aurora_moe assumes at every EP
degree we register. The port is a day of glue, not a dispatcher restructuring.

## The two gating questions

| question | job | answer |
|---|---|---|
| Does EP>1 survive on torch 2.15? | `8836014` | **Yes.** EP=2, 24 ranks, rc=0, losses 12.95224 -> 12.59619 -> 11.47116, matching the EP=1 run to ~4 decimals. No `ur_die`, no `urEventWait`, no a2a abort. |
| Do the rank orderings agree? | `8836277` | **Yes, at EP=2, 8 AND 12.** All 24 ranks AGREE at each degree. |

The second matters because `create_dp_ep_groups` documents "fixed-order process
groups for **rank = dp_rank * EP + ep_rank**". Had torchtitan ordered its mesh
differently, the all-to-all would have exchanged tokens between the WRONG ranks
-- no crash, no error, just wrong gradients. It does not: `pd.get_mesh("ep")
.get_local_rank()` equals `rank % EP` for every rank at every degree tested,
covering `moe_debugmodel_ep` (2), `moe_10b_2b_sdpa_hybridep` (8) and
`moe_10b_2b_sdpa_bmm_ep` (12).

## What the port actually needs

`aurora_sycl` was easy because `torchtitan_exact_experts(w1,w2,w3,x,rows)` is a
pure function over already-routed rows. `sycl_sonic` is not: it needs the
routing tensors and a mesh.

- `_routed_moe(x, topk_scores, topk_indices, local_expert_ids, mesh, up, gate,
  down, expert_backend="sycl_sonic")` in `vendor/.../aurora_moe/_core.py:3735`.
- `mesh` is aurora_moe's **`ParallelMesh`**, not a torch `DeviceMesh` -- the
  kernel does `mesh.group_size["ep_dispatch"]`. Build it from
  `MoEProcessGroups`.
- The routing tensors ARE available, one layer above where the expert backend
  sits: `MoE.forward` (`models/common/moe.py:728`) passes `topk_scores_TK` and
  `topk_expert_ids_TK` into `RoutedExperts.forward`, which holds them at lines
  143-144 and then drops them, calling `inner_experts(routed_input_RD, counts)`
  at line 163. That single discard is the whole gap.

Steps: an `EzpzRoutedExperts` subclass that forwards scores/indices/mesh; widen
`EzpzGroupedExperts.forward` with optional kwargs (2-arg path untouched so the
four working backends do not move); build the `ParallelMesh`; register the
flavor; prebuild four SYCL kernels (`ep_route_ops`,
`one_mkl_exact_expert_gemm`, `segment_expert_reorder`, `swiglu_ops`) inside an
XPU allocation.

## Traps confirmed on the way

- **XCCL cannot split a process group.** Building a mesh outside `train.py`
  dies with `No backend for the parent process group or its backend does not
  support splitting`; call `maybe_install_xccl_split_group_workaround()` first
  (job `8836176`). `create_dp_ep_groups` makes its own groups, so the port will
  hit this.
- `sycl_sonic` additionally gates on `AURORA_MOE_ALLTOALLV=1`.
- Nine jobs were spent reaching these two answers and eight failed on the job
  script rather than the system. Those lessons are now executable in
  `.claude/skills/alcf-job-preflight/`.
