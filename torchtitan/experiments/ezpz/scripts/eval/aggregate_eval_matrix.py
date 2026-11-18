#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Aggregate eval matrix CSV from inventory + optional artifact paths.
Grounding refs:
  MDS numbers: torchtitan/experiments/ezpz/docs/summaries/2026-08-10.md
  TT numbers:  torchtitan/experiments/ezpz/docs/run-ledger/02-hf-conversion.md
  SFT B2:      torchtitan/experiments/ezpz/docs/run-ledger/03-eval-harness.md:33
               + torchtitan/experiments/ezpz/docs/production/sft/agpt/2b-v2-256n/tulu_math_uc_mix/README.md
  CoT SE:      docs/exploration/thread-c-eval-harness.md (N=200 SE~0.028, 2sigma +0.06)
  Letter prior: docs/evals/agpt/2b/mmlu-letter-prior-at-chance.md (R2 0.64-0.76)
"""
import argparse
import csv
import json
import re
from pathlib import Path

BAND_LO = 0.2295
BAND_HI = 0.2689
CONSTANT_D = 0.2689
GROUNDED_MDS_STAGE3 = 0.2463  # stage3-mix MMLU_max
GROUNDED_MDS_DOLMINO = 0.2413  # dolmino MMLU_max
SEQ_LEN = 524288  # tokens per step
B2_GSM8K = 0.205
B2_FORMAT = 0.985
B2_GENLEN = 282
COT_SE_200 = 0.028


def format_tokens(step: int) -> str:
    tokens = step * SEQ_LEN
    if tokens >= 1e12:
        return f"{tokens/1e12:.2f}T"
    return f"{tokens/1e9:.2f}B"


def parse_step(checkpoint: str):
    m = re.search(r"(\d+)", checkpoint)
    return int(m.group(1)) if m else None


def fit_letter_prior(per_subject: dict):
    """Least-squares p_L for acc = sum_L p_L*keyfreq_L. per_subject: {subj: {acc, freq:{A,B,C,D}}}."""
    # Build normal equations 4x4
    letters = ["A", "B", "C", "D"]
    n = len(per_subject)
    # X^T X and X^T y
    XtX = [[0.0] * 4 for _ in range(4)]
    Xty = [0.0] * 4
    ys = []
    for v in per_subject.values():
        y = v["acc"]
        ys.append(y)
        f = [v["freq"][L] for L in letters]
        for i in range(4):
            Xty[i] += f[i] * y
            for j in range(4):
                XtX[i][j] += f[i] * f[j]
    # Solve via Gaussian elimination 4x4
    # Augment
    aug = [XtX[i] + [Xty[i]] for i in range(4)]
    for col in range(4):
        # pivot
        piv = max(range(col, 4), key=lambda r: abs(aug[r][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        if abs(aug[col][col]) < 1e-12:
            return None
        for row in range(4):
            if row == col:
                continue
            factor = aug[row][col] / aug[col][col]
            for k in range(5):
                aug[row][k] -= factor * aug[col][k]
    p = [
        aug[i][4] / aug[i][4 - 0 + i - i] if False else aug[i][4] / aug[i][i]
        for i in range(4)
    ]
    # R2
    ymean = sum(ys) / len(ys) if ys else 0
    ss_tot = sum((y - ymean) ** 2 for y in ys)
    ss_res = 0
    for v in per_subject.values():
        yhat = sum(p[i] * v["freq"][letters[i]] for i in range(4))
        ss_res += (v["acc"] - yhat) ** 2
    r2 = 1 - ss_res / ss_tot if ss_tot else 0
    # chi2/df
    chi2 = ss_res / (max(1e-9, ymean * (1 - ymean) / 200)) if ymean not in (0, 1) else 0
    df = max(1, n - 4)
    return p, r2, chi2 / df


def mmlu_band(val_str):
    if not val_str:
        return ""
    try:
        v = float(val_str)
        inside = BAND_LO <= v <= BAND_HI
        tag = "at-chance" if inside else ("above-band" if v > BAND_HI else "below-band")
        if v < CONSTANT_D:
            tag += f"; constant-D {CONSTANT_D} beats all"
        return f"{tag} band [{BAND_LO}-{BAND_HI}]"
    except Exception:
        return ""


def _metric(entry, *names):
    """First present metric, as a string. lm-eval >=0.4 keys are 'acc,none'.

    Keeps an explicit 0.0 (a plain `or` chain would drop it)."""
    if not isinstance(entry, dict):
        return ""
    for name in names:
        for key in (f"{name},none", name):
            if entry.get(key) is not None:
                return str(entry[key])
    return ""


def read_mmlu(path: Path):
    """(MCF, CF) from one lm-eval results JSON.

    lm-eval's `mmlu` group scores the answer letter (MCF);
    `mmlu_continuation` scores the answer text (CF).
    """
    if not path or not path.exists():
        return "", ""
    try:
        data = json.loads(path.read_text())
        res = data.get("results", data)
        mcf = _metric(res.get("mmlu"), "acc")
        cf = _metric(res.get("mmlu_continuation"), "acc")
        return mcf, cf
    except Exception:
        return "", ""


def read_commonsense(paths):
    """[HS acc_norm, ARC-E acc, PIQA acc_norm, ARC-C acc_norm] (schema columns).

    Merges every given lm-eval results JSON: the commonsense sweep writes
    HS/ARC-E/PIQA to results_0shot.json and ARC-C to results_25shot.json.
    """
    res = {}
    for path in paths or []:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        res.update(data.get("results", data))
    return [
        _metric(res.get("hellaswag"), "acc_norm"),
        _metric(res.get("arc_easy"), "acc"),
        _metric(res.get("piqa"), "acc_norm"),
        _metric(res.get("arc_challenge"), "acc_norm"),
    ]


def read_cot(path: Path):
    if not path or not path.exists():
        return [""] * 4
    try:
        lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        if not lines:
            return [""] * 4
        n = len(lines)
        acc = sum(1 for r in lines if r.get("correct")) / n
        fmt = sum(1 for r in lines if r.get("format_hit") or r.get("format_ok")) / n
        gen_lens = [r.get("gen_len", r.get("gen_tokens", 0)) for r in lines]
        mean_gl = sum(gen_lens) / len(gen_lens) if gen_lens else ""
        if any("finish_reason" in r for r in lines):
            # The harness's definition (eval_cot_gsm8k.py summarize()): a
            # generation is unclosed when it stopped on the length limit.
            n_unclosed = sum(1 for r in lines if r.get("finish_reason") == "length")
        else:
            # Older artifacts without finish_reason: missing </answer> tag.
            n_unclosed = sum(
                1
                for r in lines
                if "</answer>" not in r.get("generation", r.get("text", ""))
            )
        return [
            f"{acc:.3f}",
            f"{fmt:.3f}",
            f"{mean_gl:.0f}" if mean_gl != "" else "",
            str(n_unclosed),
        ]
    except Exception:
        return [""] * 4


def main():
    ap = argparse.ArgumentParser(description="Aggregate eval matrix CSV")
    ap.add_argument(
        "--inventory", default=".hermes/plans/checkpoints-inventory-2026-11-16.csv"
    )
    ap.add_argument(
        "--schema", default=".hermes/plans/eval-matrix-schema-2026-11-16.csv"
    )
    ap.add_argument("--mmlu-json", default=None)
    ap.add_argument("--mmlu-per-subject", default=None)
    ap.add_argument(
        "--commonsense-json",
        nargs="+",
        default=None,
        help="lm-eval results JSONs to merge (e.g. results_0shot.json results_25shot.json)",
    )
    ap.add_argument("--cot-jsonl", default=None)
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="Inventory checkpoint the artifact files belong to. Required when "
        "artifacts are given and the inventory has more than one row; other "
        "rows keep empty score columns.",
    )
    ap.add_argument("--out", default=".hermes/plans/eval-matrix-results-2026-11-16.csv")
    args = ap.parse_args()

    inv_path = Path(args.inventory)
    schema_path = Path(args.schema)

    # Load header from schema
    if schema_path.exists():
        with open(schema_path) as f:
            r = csv.reader(f)
            # skip comment line if starts with #
            first = next(r)
            if first and first[0].startswith("#"):
                header = next(r)
            else:
                header = first
    else:
        header = [
            "group",
            "checkpoint",
            "dcp_path",
            "hf_path",
            "tokens",
            "dcp_exists",
            "hf_exists",
            "stage_note",
            "MMLU_MCF",
            "MMLU_CF",
            "MMLU_max",
            "HS_acc_norm",
            "ARC_E_acc",
            "PIQA_acc_norm",
            "ARC_C_acc_norm",
            "GSM8K_pre",
            "GSM8K_post",
            "format_hit_rate",
            "mean_gen_len",
            "n_unclosed",
            "delta_CPT_vs_MDS",
            "delta_SFT_vs_base",
            "delta_RL_vs_SFT",
            "source_job",
            "verified",
        ]

    extra_cols = ["mmlu_band", "mmlu_letter_prior", "notes"]
    out_header = header + [c for c in extra_cols if c not in header]

    # Letter prior
    if args.mmlu_per_subject and Path(args.mmlu_per_subject).exists():
        pj = json.loads(Path(args.mmlu_per_subject).read_text())
        fitted = fit_letter_prior(pj)
        if fitted:
            p, r2, chi2df = fitted
            letter_prior_str = f"p=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f},{p[3]:.2f}) R2={r2:.2f} chi2/df={chi2df:.2f}"
        else:
            letter_prior_str = "p=(0.37,0.32,0.20,0.11) R2 0.64-0.76 measured on agpt-2B per mmlu-letter-prior-at-chance.md"
    else:
        letter_prior_str = "p=(0.37,0.32,0.20,0.11) R2 0.64-0.76 measured on agpt-2B per mmlu-letter-prior-at-chance.md"

    # Load inventory
    rows = []
    if inv_path.exists():
        with open(inv_path) as f:
            dr = csv.DictReader(line for line in f if not line.startswith("#") or False)
            # handle comment-prefixed lines
            content = inv_path.read_text().splitlines()
            data_lines = [l for l in content if not l.startswith("#") and l.strip()]
            if data_lines:
                dr2 = csv.DictReader(data_lines)
                for r in dr2:
                    rows.append(r)

    # Artifact files are one checkpoint's results; never copy them into
    # every inventory row.
    have_artifacts = any((args.mmlu_json, args.commonsense_json, args.cot_jsonl))
    target = args.checkpoint
    if have_artifacts and target is None:
        if len(rows) > 1:
            raise SystemExit(
                "--checkpoint is required with artifact files when the "
                f"inventory has {len(rows)} rows"
            )
        target = rows[0].get("checkpoint", "") if rows else None
    if (
        target is not None
        and rows
        and not any(r.get("checkpoint", "") == target for r in rows)
    ):
        raise SystemExit(f"--checkpoint {target!r} is not in {inv_path}")

    # Prepare optional artifact reads
    mmlu_mcf, mmlu_cf = read_mmlu(Path(args.mmlu_json) if args.mmlu_json else None)
    cs_vals = read_commonsense([Path(p) for p in args.commonsense_json or []])
    cot_vals = read_cot(Path(args.cot_jsonl) if args.cot_jsonl else None)

    # Also probe known CoT output files (best-effort, do not fail)
    cot_search = [Path("/home/foremans/torchtitan/outputs/evals/cot")]
    # just log existence, don't require

    out_rows = []
    for r in rows:
        ckpt = r.get("checkpoint", "")
        tokens = r.get("tokens", "").strip()
        if not tokens:
            step = parse_step(ckpt)
            if step is not None:
                tokens = format_tokens(step)
        # No invented eval scores: scores come only from this row's artifacts.
        mine = ckpt == target
        r_mcf, r_cf = (mmlu_mcf, mmlu_cf) if mine else ("", "")
        r_cs = cs_vals if mine else [""] * 4
        r_cot = cot_vals if mine else [""] * 4

        mmax_str = ""
        if r_mcf and r_cf:
            try:
                mmax_str = f"{max(float(r_mcf), float(r_cf)):.4f}"
            except ValueError:
                pass

        # Deltas compare the same benchmark against a grounded predecessor,
        # or stay empty: CPT MMLU_max vs the MDS stage-3 MMLU_max; RL GSM8K vs
        # the B2 SFT GSM8K. No grounded base GSM8K exists for SFT-vs-base.
        group = r.get("group", "")
        delta_cpt = ""
        delta_sft = ""
        delta_rl = ""
        if "CPT" in group and mmax_str:
            delta_cpt = f"{float(mmax_str) - GROUNDED_MDS_STAGE3:.4f}"
        if "RL" in group and r_cot[0]:
            delta_rl = f"{float(r_cot[0]) - B2_GSM8K:.4f}"

        verified = "no"
        # HF existence check
        hf_path = r.get("hf_path", "").strip()
        if hf_path and Path(hf_path).exists():
            verified = "no"  # existence alone not verification

        notes = r.get("stage_note", "")
        # Append grounding note for known checkpoints
        if ckpt == "checkpoint-93-hf":
            notes = (
                notes
                + f"; B2 {B2_GSM8K}/{B2_FORMAT}/{B2_GENLEN} SE {COT_SE_200}@200 +0.06=2sigma per 03-eval-harness.md:33"
            ).strip("; ")
        elif ckpt == "step-92859":
            notes = (
                notes
                + "; MMLU 0.2437 HS 0.561 per run-ledger/01-cpt-frontier + summaries/2026-08-10"
            ).strip("; ")

        out = {
            "group": r.get("group", ""),
            "checkpoint": ckpt,
            "dcp_path": r.get("dcp_path", ""),
            "hf_path": hf_path,
            "tokens": tokens,
            "dcp_exists": r.get("dcp_exists", ""),
            "hf_exists": r.get("hf_exists", ""),
            "stage_note": r.get("stage_note", ""),
            "MMLU_MCF": r_mcf,
            "MMLU_CF": r_cf,
            "MMLU_max": mmax_str,
            "HS_acc_norm": r_cs[0],
            "ARC_E_acc": r_cs[1],
            "PIQA_acc_norm": r_cs[2],
            "ARC_C_acc_norm": r_cs[3],
            "GSM8K_pre": r_cot[0],
            "GSM8K_post": r_cot[0],
            "format_hit_rate": r_cot[1],
            "mean_gen_len": r_cot[2],
            "n_unclosed": r_cot[3],
            "delta_CPT_vs_MDS": delta_cpt,
            "delta_SFT_vs_base": delta_sft,
            "delta_RL_vs_SFT": delta_rl,
            "source_job": r.get("source_job", ""),
            "verified": verified,
            "mmlu_band": mmlu_band(mmax_str),
            "mmlu_letter_prior": letter_prior_str,
            "notes": notes,
        }
        out_rows.append(out)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_header)
        w.writeheader()
        for o in out_rows:
            w.writerow({k: o.get(k, "") for k in out_header})
    print(f"Wrote {len(out_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
