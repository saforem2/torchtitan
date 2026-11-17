#!/usr/bin/env python3
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
import math
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
    XtX = [[0.0]*4 for _ in range(4)]
    Xty = [0.0]*4
    ys = []
    for v in per_subject.values():
        y = v["acc"]
        ys.append(y)
        f = [v["freq"][L] for L in letters]
        for i in range(4):
            Xty[i] += f[i]*y
            for j in range(4):
                XtX[i][j] += f[i]*f[j]
    # Solve via Gaussian elimination 4x4
    # Augment
    aug = [XtX[i]+[Xty[i]] for i in range(4)]
    for col in range(4):
        # pivot
        piv = max(range(col, 4), key=lambda r: abs(aug[r][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        if abs(aug[col][col]) < 1e-12:
            return None
        for row in range(4):
            if row == col:
                continue
            factor = aug[row][col]/aug[col][col]
            for k in range(5):
                aug[row][k] -= factor*aug[col][k]
    p = [aug[i][4]/aug[i][4-0+i-i] if False else aug[i][4]/aug[i][i] for i in range(4)]
    # R2
    ymean = sum(ys)/len(ys) if ys else 0
    ss_tot = sum((y-ymean)**2 for y in ys)
    ss_res = 0
    for v in per_subject.values():
        yhat = sum(p[i]*v["freq"][letters[i]] for i in range(4))
        ss_res += (v["acc"]-yhat)**2
    r2 = 1 - ss_res/ss_tot if ss_tot else 0
    # chi2/df
    chi2 = ss_res / (max(1e-9, ymean*(1-ymean)/200)) if ymean not in (0,1) else 0
    df = max(1, n-4)
    return p, r2, chi2/df


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


def read_mmlu(path: Path):
    if not path or not path.exists():
        return "", ""
    try:
        data = json.loads(path.read_text())
        # group key mmlu not leaf macro
        mmlu = data.get("results", {}).get("mmlu") or data.get("mmlu") or {}
        # try MCF/CF keys
        mcf = mmlu.get("acc_mcf") or mmlu.get("mcf") or mmlu.get("MCF") or ""
        cf = mmlu.get("acc") or mmlu.get("cf") or mmlu.get("CF") or ""
        # also check top-level
        if not mcf:
            mcf = data.get("MMLU_MCF", "")
        if not cf:
            cf = data.get("MMLU_CF", "")
        return str(mcf) if mcf != "" else "", str(cf) if cf != "" else ""
    except Exception:
        return "", ""


def read_commonsense(path: Path):
    if not path or not path.exists():
        return [""]*4
    try:
        data = json.loads(path.read_text())
        res = data.get("results", data)
        def get(k):
            v = res.get(k, {})
            if isinstance(v, dict):
                return str(v.get("acc_norm", v.get("acc", "")))
            return str(v) if v else ""
        return [get("hellaswag"), get("arc_easy"), get("piqa"), get("arc_challenge")]
    except Exception:
        return [""]*4


def read_cot(path: Path):
    if not path or not path.exists():
        return [""]*4
    try:
        lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        if not lines:
            return [""]*4
        n = len(lines)
        acc = sum(1 for r in lines if r.get("correct"))/n
        fmt = sum(1 for r in lines if r.get("format_hit") or r.get("format_ok"))/n
        gen_lens = [r.get("gen_len", r.get("gen_tokens", 0)) for r in lines]
        mean_gl = sum(gen_lens)/len(gen_lens) if gen_lens else ""
        n_unclosed = sum(1 for r in lines if not r.get("closed", True) or r.get("n_unclosed"))
        # fallback: count missing </answer>
        if n_unclosed == 0:
            n_unclosed = sum(1 for r in lines if "</answer>" not in r.get("generation", r.get("text","")))
        return [f"{acc:.3f}", f"{fmt:.3f}", f"{mean_gl:.0f}" if mean_gl != "" else "", str(n_unclosed)]
    except Exception:
        return [""]*4


def main():
    ap = argparse.ArgumentParser(description="Aggregate eval matrix CSV")
    ap.add_argument("--inventory", default=".hermes/plans/checkpoints-inventory-2026-11-16.csv")
    ap.add_argument("--schema", default=".hermes/plans/eval-matrix-schema-2026-11-16.csv")
    ap.add_argument("--mmlu-json", default=None)
    ap.add_argument("--mmlu-per-subject", default=None)
    ap.add_argument("--commonsense-json", default=None)
    ap.add_argument("--cot-jsonl", default=None)
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
        header = ["group","checkpoint","dcp_path","hf_path","tokens","dcp_exists","hf_exists","stage_note",
                   "MMLU_MCF","MMLU_CF","MMLU_max","HS_acc_norm","ARC_E_acc","PIQA_acc_norm","ARC_C_acc_norm",
                   "GSM8K_pre","GSM8K_post","format_hit_rate","mean_gen_len","n_unclosed",
                   "delta_CPT_vs_MDS","delta_SFT_vs_base","delta_RL_vs_SFT","source_job","verified"]

    extra_cols = ["mmlu_band","mmlu_letter_prior","notes"]
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

    # Prepare optional artifact reads
    mmlu_mcf, mmlu_cf = read_mmlu(Path(args.mmlu_json) if args.mmlu_json else None)
    cs_vals = read_commonsense(Path(args.commonsense_json) if args.commonsense_json else None)
    cot_vals = read_cot(Path(args.cot_jsonl) if args.cot_jsonl else None)

    # Also probe known CoT output files (best-effort, do not fail)
    cot_search = [Path("/home/foremans/torchtitan/outputs/evals/cot")]
    # just log existence, don't require

    out_rows = []
    for r in rows:
        ckpt = r.get("checkpoint","")
        tokens = r.get("tokens","").strip()
        if not tokens:
            step = parse_step(ckpt)
            if step is not None:
                tokens = format_tokens(step)
        # grounding: grounded MDS/TT SFT numbers injected via stage_note matching
        # No invented eval scores: MMLU/CoT filled only from artifact paths; otherwise empty
        # Delta columns
        delta_cpt = ""
        delta_sft = ""
        delta_rl = ""
        # Only compute if we have mmlu values; otherwise leave empty (no invention)
        if mmlu_mcf and mmlu_cf:
            try:
                mmax = max(float(mmlu_mcf), float(mmlu_cf))
                group = r.get("group","")
                if "CPT" in group:
                    delta_cpt = f"{mmax - GROUNDED_MDS_STAGE3:.4f}"
                if "SFT" in group:
                    delta_sft = f"{mmax - B2_GSM8K:.4f}"
                if "RL" in group:
                    delta_rl = f"{mmax - B2_GSM8K:.4f}"
            except Exception:
                pass

        mmax_str = ""
        if mmlu_mcf and mmlu_cf:
            try:
                mmax_str = f"{max(float(mmlu_mcf), float(mmlu_cf)):.4f}"
            except Exception:
                pass

        verified = "no"
        # HF existence check
        hf_path = r.get("hf_path","").strip()
        if hf_path and Path(hf_path).exists():
            verified = "no"  # existence alone not verification

        notes = r.get("stage_note","")
        # Append grounding note for known checkpoints
        if ckpt == "checkpoint-93-hf":
            notes = (notes + f"; B2 {B2_GSM8K}/{B2_FORMAT}/{B2_GENLEN} SE {COT_SE_200}@200 +0.06=2sigma per 03-eval-harness.md:33").strip("; ")
        elif ckpt == "step-92859":
            notes = (notes + "; MMLU 0.2437 HS 0.561 per run-ledger/01-cpt-frontier + summaries/2026-08-10").strip("; ")

        out = {
            "group": r.get("group",""),
            "checkpoint": ckpt,
            "dcp_path": r.get("dcp_path",""),
            "hf_path": hf_path,
            "tokens": tokens,
            "dcp_exists": r.get("dcp_exists",""),
            "hf_exists": r.get("hf_exists",""),
            "stage_note": r.get("stage_note",""),
            "MMLU_MCF": mmlu_mcf,
            "MMLU_CF": mmlu_cf,
            "MMLU_max": mmax_str,
            "HS_acc_norm": cs_vals[0],
            "ARC_E_acc": cs_vals[1],
            "PIQA_acc_norm": cs_vals[2],
            "ARC_C_acc_norm": cs_vals[3],
            "GSM8K_pre": cot_vals[0],
            "GSM8K_post": cot_vals[0],
            "format_hit_rate": cot_vals[1],
            "mean_gen_len": cot_vals[2],
            "n_unclosed": cot_vals[3],
            "delta_CPT_vs_MDS": delta_cpt,
            "delta_SFT_vs_base": delta_sft,
            "delta_RL_vs_SFT": delta_rl,
            "source_job": r.get("source_job",""),
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
            w.writerow({k: o.get(k,"") for k in out_header})
    print(f"Wrote {len(out_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
