"""Build frozen held-out math evaluation sets for the anneal A/B.

The anneal experiment forks a converged base and continues training on
open-web-math with two LR schedules (flat control vs WSD decay-to-0). The
decision metric is held-out math val loss (see held_out_math_valloss.py),
which requires FIXED, FROZEN holdout sets so every checkpoint (base, flat,
wsd) is scored on byte-identical text.

Two holdouts are built (see the anneal design):

  finemath_holdout.jsonl -- PRIMARY. A slice of FineMath-4+, a DIFFERENT
    math corpus that neither base trained on and the anneal never streams.
    This is the true cross-corpus generalization signal: valid for all
    three eval points (base, flat, wsd).

  owm_holdout.jsonl -- SECONDARY. The first N documents of open-web-math
    (the anneal training corpus). Because both anneal arms stream the
    IDENTICAL seed-42 token order, any overlap with this slice is symmetric
    across arms, so the flat-vs-wsd delta on this set is still valid (the
    overlap cancels). It measures in-distribution fit, NOT generalization,
    and the base-vs-arm comparison on this set is confounded by the arms
    having trained on it -- use it only for the flat-vs-wsd delta.

Run ONCE on a node with HF hub access (ALCF proxy + HF_HUB_ENABLE_HF_TRANSFER=0),
then commit the jsonl files so the scoring is reproducible:

    python3 -m torchtitan.experiments.ezpz.scripts.eval.build_math_holdout \\
        --out-dir torchtitan/experiments/ezpz/eval/holdouts \\
        --num-docs 2000

Determinism: documents are taken from the head of each stream (take(N)),
which is a stable order for a given dataset revision, so re-running produces
the same holdout. The revision is recorded in the sidecar meta json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

# (dataset_path, config_name, split, text_column) for each holdout source.
# FineMath-4+ needs its config_name; open-web-math has a single default config.
_FINEMATH = ("HuggingFaceTB/finemath", "finemath-4plus", "train", "text")
_OWM = ("open-web-math/open-web-math", None, "train", "text")
# General-domain anti-forgetting holdout for the data-mix experiment: NO arm
# trains on wikitext, so it is a clean disjoint judge of whether a stage-2 mix
# forgets general ability. wikitext-103 docs are short (articles split into
# lines), so it uses a lower min-chars via --wikitext-min-chars.
_WIKITEXT = ("Salesforce/wikitext", "wikitext-103-raw-v1", "test", "text")


def _take_docs(
    dataset_path: str,
    config_name: str | None,
    split: str,
    text_column: str,
    num_docs: int,
    min_chars: int,
) -> list[str]:
    """Stream the head of a corpus and collect the first num_docs non-empty texts."""
    kwargs = {"split": split, "streaming": True}
    if config_name is not None:
        kwargs["name"] = config_name
    ds = load_dataset(dataset_path, **kwargs)
    texts: list[str] = []
    for row in ds:
        text = row.get(text_column) or ""
        # Skip near-empty rows so every holdout doc carries real signal; this
        # keeps the token budget honest (no padding-dominated documents).
        if len(text) < min_chars:
            continue
        texts.append(text)
        if len(texts) >= num_docs:
            break
    if len(texts) < num_docs:
        raise RuntimeError(
            f"only collected {len(texts)}/{num_docs} docs from {dataset_path} "
            f"(config={config_name}); lower --num-docs or --min-chars"
        )
    return texts


def _write_jsonl(texts: list[str], path: Path, source: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for text in texts:
            f.write(json.dumps({"text": text}) + "\n")
    meta = {
        "num_docs": len(texts),
        "total_chars": sum(len(t) for t in texts),
        **source,
    }
    meta_path = path.with_name(path.stem + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {path} ({len(texts)} docs, {meta['total_chars']} chars)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("torchtitan/experiments/ezpz/eval/holdouts"),
        help="Directory for the frozen holdout jsonl files.",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=2000,
        help="Documents per holdout (default 2000).",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=512,
        help="Skip documents shorter than this many characters (default 512).",
    )
    parser.add_argument(
        "--wikitext",
        action="store_true",
        help="Also build a wikitext general-domain anti-forgetting holdout "
        "(for the data-mix experiment). Uses --wikitext-min-chars.",
    )
    parser.add_argument(
        "--wikitext-min-chars",
        type=int,
        default=256,
        help="min-chars for the wikitext holdout (docs are short; default 256).",
    )
    parser.add_argument(
        "--wikitext-num-docs",
        type=int,
        default=1500,
        help="docs for the wikitext holdout (test split has only ~1656 docs "
        ">=256 chars, so this defaults BELOW the math --num-docs 2000).",
    )
    args = parser.parse_args()

    fm_path, fm_cfg, fm_split, fm_col = _FINEMATH
    finemath = _take_docs(
        fm_path, fm_cfg, fm_split, fm_col, args.num_docs, args.min_chars
    )
    _write_jsonl(
        finemath,
        args.out_dir / "finemath_holdout.jsonl",
        {"dataset": fm_path, "config": fm_cfg, "split": fm_split, "role": "primary"},
    )

    owm_path, owm_cfg, owm_split, owm_col = _OWM
    owm = _take_docs(
        owm_path, owm_cfg, owm_split, owm_col, args.num_docs, args.min_chars
    )
    _write_jsonl(
        owm,
        args.out_dir / "owm_holdout.jsonl",
        {"dataset": owm_path, "config": owm_cfg, "split": owm_split, "role": "secondary"},
    )

    if args.wikitext:
        wt_path, wt_cfg, wt_split, wt_col = _WIKITEXT
        wikitext = _take_docs(
            wt_path, wt_cfg, wt_split, wt_col,
            args.wikitext_num_docs, args.wikitext_min_chars,
        )
        _write_jsonl(
            wikitext,
            args.out_dir / "wikitext_holdout.jsonl",
            {
                "dataset": wt_path,
                "config": wt_cfg,
                "split": wt_split,
                "role": "anti-forgetting",
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
