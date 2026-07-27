"""Apply the pre-registered anneal A/B decision rule and print PASS/FAIL.

Consumes, for one base (mds or olmo), the three eval points (base / flat / wsd):
  - held-out math val loss json(s) from held_out_math_valloss.py
  - lm-eval results.json from eval-2b-v2.sh (gsm8k full-1319, hellaswag)

Pre-registered rule (frozen before the run; see the anneal design):
  PRIMARY (decisive): valloss(wsd) < valloss(flat) on the FineMath holdout.
      This is the generalization signal; NEVER compare final train loss.
  FORGETTING GUARD: HellaSwag(wsd) drop <= 1.5pp vs BASE.
  SECONDARY (confirmatory only): GSM8K(wsd) full-1319 >= GSM8K(flat) + 2.0pp
      with NON-OVERLAPPING bootstrap CIs. Because 2B GSM8K sits near floor,
      this is a bonus signal -- it does NOT flip a PASS to FAIL when the
      primary val-loss tripwire is positive and the CIs merely overlap.

VERDICT:
  PASS = primary positive (wsd val loss strictly below flat) AND forgetting
         guard satisfied (hellaswag drop <= 1.5pp).
  FAIL = primary not positive, OR hellaswag regresses > 1.5pp vs base.

Usage (paths are per (base,arm,step) directories that already hold the eval
outputs; pass the FineMath val-loss json for each arm + the results.json dirs):

    python3 -m torchtitan.experiments.ezpz.scripts.eval.anneal_ab_verdict \\
        --base-hellaswag  outputs/evals/anneal/<base>-base/step-0/results/results.json \\
        --flat-valloss    outputs/evals/anneal/<base>-flat/step-1600/valloss_finemath.json \\
        --wsd-valloss     outputs/evals/anneal/<base>-wsd/step-1600/valloss_finemath.json \\
        --flat-results    outputs/evals/anneal/<base>-flat/step-1600/results/results.json \\
        --wsd-results     outputs/evals/anneal/<base>-wsd/step-1600/results/results.json \\
        --label mds

The GSM8K secondary and its CI are reported when --flat-results/--wsd-results
are given; the primary + forgetting guard alone determine PASS/FAIL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 95% bootstrap CI half-width in units of the reported stderr.
_CI_Z = 1.96
# Forgetting guard: HellaSwag may drop at most this many percentage POINTS.
_HS_MAX_DROP_PP = 1.5
# Secondary confirmatory bar: GSM8K uplift in percentage POINTS.
_GSM8K_MIN_UPLIFT_PP = 2.0


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _valloss(path: Path) -> float:
    return _load(path)["mean_nll"]


def _metric_and_stderr(results_path: Path, task: str) -> tuple[float, float]:
    """Pull (metric, stderr) for a task from an lm-eval results.json.

    Prefers acc_norm, then acc, then exact_match (gsm8k). Returns fractions
    in [0,1] (lm-eval's native scale); callers convert to pp as needed.
    """
    results = _load(results_path)
    m = results.get(task)
    if m is None:
        # mmlu-style group prefix, or task nested; fall back to a prefix match.
        for k, v in results.items():
            if k == task or k.startswith(task + "_"):
                m = v
                break
    if m is None:
        raise SystemExit(f"task {task!r} not in {results_path}")

    for metric_key, se_key in (
        ("acc_norm,none", "acc_norm_stderr,none"),
        ("acc,none", "acc_stderr,none"),
        ("exact_match,strict-match", "exact_match_stderr,strict-match"),
        ("exact_match,flexible-extract", "exact_match_stderr,flexible-extract"),
        ("exact_match,none", "exact_match_stderr,none"),
    ):
        if metric_key in m:
            return float(m[metric_key]), float(m.get(se_key, 0.0))
    raise SystemExit(f"no known metric key for task {task!r} in {results_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--label", required=True, help="Base label, e.g. mds or olmo.")
    parser.add_argument("--flat-valloss", required=True, type=Path)
    parser.add_argument("--wsd-valloss", required=True, type=Path)
    parser.add_argument(
        "--base-hellaswag",
        required=True,
        type=Path,
        help="results.json for the BASE (step-0) -- the forgetting reference.",
    )
    parser.add_argument(
        "--wsd-results",
        type=Path,
        default=None,
        help="results.json for the WSD arm (hellaswag guard + gsm8k secondary).",
    )
    parser.add_argument(
        "--flat-results",
        type=Path,
        default=None,
        help="results.json for the FLAT arm (gsm8k secondary only).",
    )
    args = parser.parse_args()

    print(f"=== anneal A/B verdict: base={args.label} ===")

    # ---- PRIMARY: held-out FineMath val loss (lower = better) ----
    flat_vl = _valloss(args.flat_valloss)
    wsd_vl = _valloss(args.wsd_valloss)
    primary_pass = wsd_vl < flat_vl
    print(f"PRIMARY  FineMath val-loss  flat={flat_vl:.5f}  wsd={wsd_vl:.5f}  "
          f"delta={wsd_vl - flat_vl:+.5f}  -> {'wsd better' if primary_pass else 'NOT better'}")

    # ---- FORGETTING GUARD: HellaSwag(wsd) vs BASE ----
    base_hs, _ = _metric_and_stderr(args.base_hellaswag, "hellaswag")
    guard_pass = True
    if args.wsd_results is not None:
        wsd_hs, _ = _metric_and_stderr(args.wsd_results, "hellaswag")
        drop_pp = (base_hs - wsd_hs) * 100.0
        guard_pass = drop_pp <= _HS_MAX_DROP_PP
        print(f"GUARD    HellaSwag  base={base_hs*100:.2f}pp  wsd={wsd_hs*100:.2f}pp  "
              f"drop={drop_pp:+.2f}pp  (max {_HS_MAX_DROP_PP}pp)  -> "
              f"{'ok' if guard_pass else 'REGRESSION'}")
    else:
        print("GUARD    HellaSwag  (wsd results not provided -- guard SKIPPED)")

    # ---- SECONDARY (confirmatory only): GSM8K uplift + non-overlapping CI ----
    if args.flat_results is not None and args.wsd_results is not None:
        flat_g, flat_se = _metric_and_stderr(args.flat_results, "gsm8k")
        wsd_g, wsd_se = _metric_and_stderr(args.wsd_results, "gsm8k")
        uplift_pp = (wsd_g - flat_g) * 100.0
        # Non-overlapping 95% CIs: wsd lower bound > flat upper bound.
        wsd_lo = (wsd_g - _CI_Z * wsd_se) * 100.0
        flat_hi = (flat_g + _CI_Z * flat_se) * 100.0
        non_overlap = wsd_lo > flat_hi
        sec_confirm = uplift_pp >= _GSM8K_MIN_UPLIFT_PP and non_overlap
        print(f"SECONDARY GSM8K  flat={flat_g*100:.2f}pp(+-{flat_se*100:.2f})  "
              f"wsd={wsd_g*100:.2f}pp(+-{wsd_se*100:.2f})  uplift={uplift_pp:+.2f}pp  "
              f"CI-nonoverlap={non_overlap}  -> "
              f"{'CONFIRMS' if sec_confirm else 'inconclusive (near-floor; confirmatory only)'}")
    else:
        print("SECONDARY GSM8K  (flat/wsd results not provided -- SKIPPED)")

    # ---- VERDICT: primary + guard decide; secondary never flips a pass ----
    verdict = primary_pass and guard_pass
    print()
    print(f"VERDICT  base={args.label}: {'PASS' if verdict else 'FAIL'}  "
          f"(primary={'+' if primary_pass else '-'} guard={'+' if guard_pass else '-'})")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
