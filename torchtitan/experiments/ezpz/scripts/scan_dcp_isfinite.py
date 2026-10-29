#!/usr/bin/env python3
"""DCP checkpoint NaN/Inf scan gate for Aurora production chains.

Scans a DCP checkpoint directory (``.metadata`` + ``__*_0.distcp`` shards)
for corrupt tensors and validates structural invariants before a boosted
umbrella is allowed to resume from it.

Checks (every run):

* ``.metadata`` exists and is non-empty.
* Shard count matches ``--expected-shards`` (default 6144 = 512N * 12).
* Every tensor in *model* and *optimizer* state is finite and
  ``abs().max() < --abs-max`` (default 1e6).

Additional checks when ``--baseline`` is given (SophiaG Hessian guard):

* For each ``hessian`` tensor in optimizer state, report ``h.max`` and
  ``h.median`` and compare against the same key in the baseline ckpt.
  A large spike (``> --hessian-spike-factor`` x baseline) is flagged as
  corrupt -- it indicates the Hessian EMA has absorbed a NaN/Inf that
  has not yet propagated to weights.

Exit codes
----------

* ``0``  all checks passed (finite).
* ``1``  NaN / Inf / structural corruption detected.
* ``2``  usage / I/O error (missing path, unreadable metadata, etc.).

Usage
-----

Run with ``python -I`` (isolated mode) so a stray ``json.py`` in the
invocation directory cannot shadow stdlib imports -- checkpoints live on
shared Lustre where other users' files are visible::

    python -I torchtitan/experiments/ezpz/scripts/scan_dcp_isfinite.py \\
        --ckpt /path/to/step-11000 \\
        --baseline /path/to/step-9000

    python -I torchtitan/experiments/ezpz/scripts/scan_dcp_isfinite.py \\
        --ckpt /path/to/step-11100          # no Hessian comparison

The script loads on a single CPU rank via ``torch.distributed.checkpoint``
(``FileSystemReader``); no ``torch.distributed`` init or GPU is required.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Structural helpers (no torch import needed)
# ---------------------------------------------------------------------------

SHARD_RE = re.compile(r"^__(\d+)_0\.distcp$")


def _scan_dir(ckpt: Path, expected_shards: int) -> list[str]:
    """Validate ``ckpt`` directory structure.  Returns list of error strings."""
    errors: list[str] = []

    if not ckpt.is_dir():
        errors.append(f"checkpoint dir does not exist or is not a directory: {ckpt}")
        return errors

    meta = ckpt / ".metadata"
    if not meta.is_file():
        errors.append(f"missing .metadata in {ckpt}")
    elif meta.stat().st_size == 0:
        errors.append(f".metadata is empty (0 bytes) in {ckpt}")

    try:
        names = os.listdir(ckpt)
    except OSError as exc:
        errors.append(f"cannot list {ckpt}: {exc}")
        return errors

    shards = [n for n in names if SHARD_RE.match(n)]
    n_shards = len(shards)
    if n_shards == 0:
        errors.append(f"no __*_0.distcp shards found in {ckpt}")
    elif n_shards != expected_shards:
        errors.append(
            f"shard count {n_shards} != expected {expected_shards} in {ckpt} "
            f"(partial save or wrong scale?)"
        )

    # Detect duplicate / missing shard indices
    if shards:
        indices = sorted(int(SHARD_RE.match(n).group(1)) for n in shards)  # type: ignore[union-attr]
        expected_indices = list(range(expected_shards))
        # Only check continuity when count matches -- otherwise the count
        # error already covers it.
        if n_shards == expected_shards and indices != expected_indices:
            missing = sorted(set(expected_indices) - set(indices))
            dup = len(indices) != len(set(indices))
            if missing:
                errors.append(f"missing shard indices (first 10): {missing[:10]}")
            if dup:
                errors.append("duplicate shard indices detected")

    return errors


# ---------------------------------------------------------------------------
# DCP tensor scan (requires torch)
# ---------------------------------------------------------------------------


def _load_state_dict(ckpt: Path) -> dict:
    """Load full checkpoint state dict via DCP on a single CPU rank.

    Uses ``FileSystemReader.read_metadata()`` to discover keys, allocates
    CPU placeholder tensors, then ``dcp.load`` into them.  Works without
    ``dist.init_process_group``.
    """
    try:
        import torch
        import torch.distributed.checkpoint as dcp
    except ImportError as exc:
        print(f"ERROR: torch is required to scan tensor contents: {exc}", file=sys.stderr)
        sys.exit(2)

    reader = dcp.FileSystemReader(str(ckpt))

    try:
        metadata = reader.read_metadata()
    except Exception as exc:
        print(f"ERROR: failed to read DCP metadata from {ckpt}: {exc}", file=sys.stderr)
        sys.exit(2)

    if not metadata.state_dict_metadata:
        print(f"ERROR: DCP metadata has no state_dict entries in {ckpt}", file=sys.stderr)
        sys.exit(2)

    # Build placeholder state dict from metadata so dcp.load knows shapes/dtypes.
    state_dict: dict = {}
    for fqn, md in metadata.state_dict_metadata.items():
        # md.properties holds dtype, md.size holds shape (may be ChunkStorageMetadata)
        try:
            # Newer torch: md is TensorStorageMetadata with .properties, .size
            props = md.properties  # type: ignore[attr-defined]
            size = md.size  # type: ignore[attr-defined]
            dtype = props.dtype
            # size may contain None for unsharded dims -- treat as-is
            shape = tuple(s if s is not None else 0 for s in size)
            # Skip scalar / empty placeholders that have no storage
            if any(s == 0 for s in shape) and len(shape) > 0 and shape != (1,):
                # Scalar step counters are often stored as () or (1,)
                # Try to handle gracefully; fall through to generic
                pass
            state_dict[fqn] = torch.empty(shape, dtype=dtype)
        except AttributeError:
            # Fallback: metadata entry may be BytesStorageMetadata (non-tensor)
            # e.g. optimizer param_groups, scheduler state -- skip non-tensor
            # entries; they cannot be NaN.
            # We detect this by checking if the entry has tensor-like attrs.
            # BytesStorageMetadata has no .properties; skip it.
            continue
        except Exception as exc:
            print(f"  WARNING: could not allocate placeholder for {fqn}: {exc}", file=sys.stderr)
            continue

    if not state_dict:
        print(
            "WARNING: no tensor placeholders could be built from metadata; "
            "metadata may contain only BytesStorageMetadata entries",
            file=sys.stderr,
        )
        return {}

    # dcp.load mutates state_dict in-place, filling tensors from shards.
    try:
        dcp.load(state_dict, storage_reader=reader)
    except Exception as exc:
        print(f"ERROR: dcp.load failed for {ckpt}: {exc}", file=sys.stderr)
        sys.exit(2)

    return state_dict


def _is_hessian_key(fqn: str) -> bool:
    """Heuristic: SophiaG hessian keys contain 'hessian' in the FQN."""
    return "hessian" in fqn.lower()


def _scan_tensors(
    state_dict: dict,
    abs_max: float,
    label: str,
) -> tuple[list[str], dict[str, dict]]:
    """Check every tensor for finiteness and abs-max.

    Returns (errors, hessian_stats) where hessian_stats maps hessian FQN ->
    {max, median, mean, numel} for the optional baseline comparison.
    """
    import torch

    errors: list[str] = []
    hessian_stats: dict[str, dict] = {}

    for fqn, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        # --- finiteness ---
        if not torch.isfinite(tensor).all().item():
            n_nan = int(torch.isnan(tensor).sum().item())
            n_inf = int(torch.isinf(tensor).sum().item())
            errors.append(f"  NON-FINITE  {fqn}: {n_nan} NaN, {n_inf} Inf  (shape={tuple(tensor.shape)})")

        # --- abs-max guard ---
        try:
            amax = float(tensor.abs().max().item())
        except Exception:
            # Empty tensor or unsupported dtype for abs()
            continue

        if amax >= abs_max:
            errors.append(f"  ABS-MAX  {fqn}: abs().max()={amax:.4g} >= {abs_max:.4g}  (shape={tuple(tensor.shape)})")

        # --- collect Hessian stats ---
        if _is_hessian_key(fqn):
            try:
                h_max = float(tensor.max().item())
                h_median = float(tensor.median().item())
                h_mean = float(tensor.float().mean().item())
                hessian_stats[fqn] = {
                    "max": h_max,
                    "median": h_median,
                    "mean": h_mean,
                    "numel": int(tensor.numel()),
                    "abs_max": amax,
                }
            except Exception:
                pass

    return errors, hessian_stats


def _compare_hessian(
    cur_stats: dict[str, dict],
    base_stats: dict[str, dict],
    spike_factor: float,
) -> list[str]:
    """Compare Hessian EMA stats against baseline. Returns error strings."""
    errors: list[str] = []

    for fqn, cur in cur_stats.items():
        base = base_stats.get(fqn)
        if base is None:
            # Key exists in cur but not baseline -- could be a new param
            # or metadata mismatch; warn but don't fail.
            print(f"  [hessian] {fqn}: no baseline entry (new key?), cur max={cur['max']:.4g}", file=sys.stderr)
            continue

        # Check for NaN/Inf in stats themselves (should already be caught above)
        for k in ("max", "median", "mean"):
            if base[k] == 0 and cur[k] == 0:
                continue
            if base[k] != 0:
                ratio = cur[k] / base[k] if base[k] > 0 else float("inf")
                # Only flag large *increases* -- decreases are benign (decay).
                if ratio > spike_factor:
                    errors.append(
                        f"  HESSIAN-SPIKE  {fqn}  {k}: cur={cur[k]:.4g} vs baseline={base[k]:.4g} "
                        f"(ratio {ratio:.2f}x > {spike_factor}x)"
                    )

        # Also flag if baseline was finite but cur has non-finite stat
        for k in ("max", "median"):
            import math

            if not math.isfinite(cur[k]):
                errors.append(f"  HESSIAN-NONFINITE  {fqn}  {k}={cur[k]}")

    # Check for keys that disappeared (baseline has, cur doesn't)
    for fqn in base_stats:
        if fqn not in cur_stats:
            print(f"  [hessian] {fqn}: in baseline but missing in candidate", file=sys.stderr)

    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "DCP checkpoint NaN/Inf scan gate -- validates .metadata, shard count, "
            "and tensor finiteness for an Aurora DCP checkpoint. "
            "Optionally compares SophiaG Hessian EMA stats against a baseline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -I %(prog)s --ckpt /lus/flare/.../step-11000\n"
            "  python -I %(prog)s --ckpt /lus/flare/.../step-11100 --baseline /lus/flare/.../step-11000\n"
            "  python -I %(prog)s --ckpt ./step-100 --expected-shards 24 --abs-max 1e7\n"
            "\nexit codes: 0=finite/ok  1=NaN or corrupt  2=usage/I/O error"
        ),
    )
    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="path to DCP checkpoint dir (the step-N directory containing .metadata + __*_0.distcp shards)",
    )
    p.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="path to a prior (known-good) checkpoint dir for Hessian EMA comparison (SophiaG)",
    )
    p.add_argument(
        "--expected-shards",
        type=int,
        default=6144,
        help="expected number of __*_0.distcp shards (default: 6144 = 512N*12)",
    )
    p.add_argument(
        "--abs-max",
        type=float,
        default=1e6,
        help="tensors with abs().max() >= this are flagged (default: 1e6)",
    )
    p.add_argument(
        "--hessian-spike-factor",
        type=float,
        default=10.0,
        help="Hessian max/median ratio vs baseline above this is flagged (default: 10.0)",
    )
    p.add_argument(
        "--skip-tensor-scan",
        action="store_true",
        help="only check structure (.metadata + shard count), skip tensor contents",
    )
    p.add_argument(
        "--allow-empty-tensor-scan",
        action="store_true",
        help="allow PASS when no tensor payload loads (structural-only); default is FAIL",
    )
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    baseline = Path(args.baseline) if args.baseline else None

    # ---- structural checks (no torch needed) ----
    print(f"[scan] ckpt: {ckpt}")
    print(f"[scan] expected shards: {args.expected_shards}  abs-max: {args.abs_max:.4g}")

    struct_errors = _scan_dir(ckpt, args.expected_shards)
    if struct_errors:
        for e in struct_errors:
            print(f"  FAIL  {e}", file=sys.stderr)
        print(f"\n[scan] STRUCTURAL FAILURE: {len(struct_errors)} error(s)", file=sys.stderr)
        return 1
    else:
        print(f"  OK  .metadata present and non-empty")
        # Count shards for the log line
        n_shards = len([n for n in os.listdir(ckpt) if SHARD_RE.match(n)])
        print(f"  OK  shard count: {n_shards} == {args.expected_shards}")

    if args.skip_tensor_scan:
        print("\n[scan] PASS (structural only, --skip-tensor-scan)")
        return 0

    # ---- tensor scan ----
    print(f"\n[scan] loading tensors via dcp (this may take a minute for 6144 shards)...")
    state_dict = _load_state_dict(ckpt)

    if not state_dict:
        if args.allow_empty_tensor_scan:
            print("  WARNING: no tensors loaded; cannot verify finiteness", file=sys.stderr)
            print("  (metadata may be BytesStorageMetadata-only -- treating as structural pass)", file=sys.stderr)
            print("\n[scan] PASS (structural; no tensor payload to scan -- --allow-empty-tensor-scan)")
            return 0
        print("  FAIL: no tensor payload to scan -- treating as corrupt", file=sys.stderr)
        print("  (use --allow-empty-tensor-scan to override)", file=sys.stderr)
        print("\n[scan] FAIL (no tensor payload)")
        return 1

    print(f"  loaded {len(state_dict)} tensor(s) from DCP")
    tensor_errors, hessian_stats = _scan_tensors(state_dict, args.abs_max, label=str(ckpt))

    if tensor_errors:
        print(f"\n[scan] TENSOR FAILURES ({len(tensor_errors)}):", file=sys.stderr)
        for e in tensor_errors:
            print(e, file=sys.stderr)
    else:
        print(f"  OK  all {len(state_dict)} tensor(s) finite and abs().max() < {args.abs_max:.4g}")

    # ---- Hessian summary (always print when present) ----
    if hessian_stats:
        print(f"\n[scan] Hessian EMA stats ({len(hessian_stats)} hessian tensor(s)):")
        # Aggregate across all hessian tensors for a quick overview
        all_max = max(s["max"] for s in hessian_stats.values())
        all_median_max = max(s["median"] for s in hessian_stats.values())
        all_median_median = sorted(s["median"] for s in hessian_stats.values())[len(hessian_stats) // 2]
        print(f"  overall h.max={all_max:.4g}  h.median(max)={all_median_max:.4g}  h.median(median)={all_median_median:.4g}")
        # Per-tensor detail (first 10 + worst)
        sorted_h = sorted(hessian_stats.items(), key=lambda kv: kv[1]["max"], reverse=True)
        print(f"  top h.max tensors:")
        for fqn, s in sorted_h[:5]:
            print(f"    {fqn}: max={s['max']:.4g}  median={s['median']:.4g}  mean={s['mean']:.4g}")
        if len(sorted_h) > 5:
            print(f"    ... and {len(sorted_h) - 5} more")
    else:
        print(f"\n[scan] no Hessian (hessian) tensors found -- not a SophiaG checkpoint or keys differ")

    # ---- baseline comparison ----
    hessian_errors: list[str] = []
    if baseline is not None:
        print(f"\n[scan] baseline: {baseline}  (spike factor {args.hessian_spike_factor}x)")

        b_struct = _scan_dir(baseline, args.expected_shards)
        if b_struct:
            print(f"  WARNING: baseline has structural issues:", file=sys.stderr)
            for e in b_struct:
                print(f"    {e}", file=sys.stderr)
            print(f"  continuing with baseline tensor scan anyway...", file=sys.stderr)

        b_state = _load_state_dict(baseline)
        if not b_state:
            print(f"  WARNING: could not load baseline tensors; skipping Hessian comparison", file=sys.stderr)
        else:
            _, b_hessian_stats = _scan_tensors(b_state, args.abs_max, label=str(baseline))
            if not b_hessian_stats:
                print(f"  WARNING: baseline has no Hessian tensors; skipping comparison", file=sys.stderr)
            elif not hessian_stats:
                print(f"  WARNING: candidate has no Hessian tensors; skipping comparison", file=sys.stderr)
            else:
                hessian_errors = _compare_hessian(hessian_stats, b_hessian_stats, args.hessian_spike_factor)
                if hessian_errors:
                    print(f"\n[scan] HESSIAN FAILURES ({len(hessian_errors)}):", file=sys.stderr)
                    for e in hessian_errors:
                        print(e, file=sys.stderr)
                else:
                    print(f"  OK  Hessian EMA within {args.hessian_spike_factor}x of baseline ({len(hessian_stats)} keys)")

    # ---- verdict ----
    all_errors = tensor_errors + hessian_errors
    if all_errors:
        print(f"\n[scan] FAIL: {len(all_errors)} check(s) failed -- checkpoint is NOT safe to resume", file=sys.stderr)
        return 1

    print(f"\n[scan] PASS: checkpoint is finite and structurally sound")
    if baseline and hessian_stats:
        print(f"[scan] Hessian EMA gate: PASS (vs {baseline})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
