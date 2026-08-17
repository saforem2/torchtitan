#!/usr/bin/env python3
"""Train a candidate 64k BPE and measure fertility against gemma / Llama-3.

Fertility here is **tokens per MB of UTF-8 text** -- lower is better, because
fewer tokens for the same content means the same knowledge for less compute.
Reported per domain on HELD-OUT text (disjoint source shards from what the
candidate was trained on; see sample_olmo_text.py).

Also reports a corpus-weighted total, because the domains are sampled equally
but the real corpus is 94.8% dclm -- an unweighted average would overstate the
value of a code/math win.

Decision rule this exists to settle (from exp07):
    >= 8-10% better than the incumbent -> a custom tokenizer is worth building
    ~5%                                -> marginal, prefer the vendored 128k
    < 5%                               -> do not build one

Usage:
    python3 measure_tokenizer_fertility.py \\
        --sample-dir /path/to/olmo-sample \\
        --gemma assets/hf/gemma-7b \\
        --llama3 assets/hf/Llama-3.1-8B \\
        --vocab-size 64000 --out /path/to/results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Real token share of the corpus, from exp07 (measured off the .bin sizes).
# Used to weight per-domain fertility into a single corpus-level number.
CORPUS_WEIGHTS = {
    "dclm": 3764.5,
    "starcoder": 97.8,
    "pes2o": 58.5,
    "arxiv": 21.2,
    "open-web-math": 12.8,
    "algebraic-stack": 12.5,
    "wiki": 3.8,
}


def train_candidate(sample_dir: str, vocab_size: int, out_path: str) -> str:
    """Train a byte-level BPE on the train-* splits. Returns the saved path."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    files = sorted(
        os.path.join(sample_dir, f)
        for f in os.listdir(sample_dir)
        if f.startswith("train-") and f.endswith(".txt")
    )
    if not files:
        raise SystemExit(f"no train-*.txt in {sample_dir}")
    print(f"training {vocab_size}-token BPE on {len(files)} files:")
    for f in files:
        print(f"  {os.path.basename(f):28s} {os.path.getsize(f)/1e6:7.1f} MB")

    tok = Tokenizer(models.BPE(unk_token=None))
    # Byte-level throughout: same family as GPT-2/Llama-3, no unknown tokens,
    # and no language-specific assumptions. add_prefix_space=False matches how
    # Llama-3 treats leading whitespace.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    t0 = time.time()
    tok.train(files, trainer)
    print(f"trained in {time.time()-t0:.0f}s, vocab={tok.get_vocab_size()}")
    tok.save(out_path)
    return out_path


def load_tokenizers(candidate_path: str, gemma: str | None, llama3: str | None):
    from tokenizers import Tokenizer

    toks = {}
    toks["custom64k"] = Tokenizer.from_file(candidate_path)
    for name, path in (("gemma256k", gemma), ("llama3-128k", llama3)):
        if not path:
            continue
        f = os.path.join(path, "tokenizer.json")
        if not os.path.exists(f):
            print(f"  warn: no tokenizer.json under {path}, skipping {name}",
                  file=sys.stderr)
            continue
        toks[name] = Tokenizer.from_file(f)
    return toks


def measure(toks: dict, sample_dir: str) -> dict:
    """tokens-per-MB per (tokenizer, domain) on held-out text."""
    results: dict[str, dict[str, float]] = {name: {} for name in toks}
    held = sorted(
        f for f in os.listdir(sample_dir)
        if f.startswith("heldout-") and f.endswith(".txt")
    )
    for fn in held:
        domain = fn[len("heldout-"):-len(".txt")]
        path = os.path.join(sample_dir, fn)
        text = open(path, encoding="utf-8").read()
        nbytes = len(text.encode("utf-8"))
        if nbytes == 0:
            continue
        print(f"\n{domain}  ({nbytes/1e6:.1f} MB held-out)")
        for name, tok in toks.items():
            # encode_batch on chunks: a single multi-MB encode can blow memory
            # and gives no progress signal.
            chunks = [text[i:i + 200_000] for i in range(0, len(text), 200_000)]
            n = sum(len(e.ids) for e in tok.encode_batch(chunks))
            tpmb = n / (nbytes / 1e6)
            results[name][domain] = tpmb
            print(f"  {name:14s} {n:10,d} tokens  {tpmb:9.0f} tok/MB")
    return results


def report(results: dict, out_dir: str) -> None:
    domains = sorted({d for r in results.values() for d in r})
    names = list(results)
    # The baseline must be an INCUMBENT, never the candidate. Falling back to
    # names[0] once made this compare custom64k against itself and print a
    # confident "+0.0% -- NOT WORTH IT" verdict from a meaningless comparison.
    baseline = next(
        (n for n in ("gemma256k", "llama3-128k") if n in results), None
    )

    print("\n" + "=" * 78)
    print("FERTILITY: tokens per MB of held-out text (LOWER IS BETTER)")
    print("=" * 78)
    hdr = f"{'domain':18s}" + "".join(f"{n:>14s}" for n in names)
    print(hdr)
    for d in domains:
        row = f"{d:18s}"
        for n in names:
            row += f"{results[n].get(d, float('nan')):14,.0f}"
        print(row)

    # Corpus-weighted: the sample is stratified, the corpus is not.
    print("\n" + "-" * 78)
    print("CORPUS-WEIGHTED (weights = real token share, dclm is 94.8%)")
    print("-" * 78)
    weighted = {}
    for n in names:
        num = den = 0.0
        for d, w in CORPUS_WEIGHTS.items():
            if d in results[n]:
                num += w * results[n][d]
                den += w
        if den:
            weighted[n] = num / den
            print(f"  {n:14s} {weighted[n]:9,.0f} tok/MB")

    if baseline is None:
        print("\nNO INCUMBENT LOADED (need --gemma and/or --llama3).")
        print("Fertility numbers above are absolute and cannot be compared to")
        print("anything, so NO VERDICT is printed. Re-run with a baseline.")
    elif baseline in weighted:
        print(f"\nvs {baseline} (negative = FEWER tokens = better):")
        for n in names:
            if n == baseline or n not in weighted:
                continue
            delta = 100 * (weighted[n] - weighted[baseline]) / weighted[baseline]
            print(f"  {n:14s} {delta:+6.1f}%")

        if "custom64k" in weighted:
            gain = 100 * (weighted[baseline] - weighted["custom64k"]) / weighted[baseline]
            print("\n" + "=" * 78)
            print(f"VERDICT: custom 64k is {gain:+.1f}% better than {baseline}")
            if gain >= 8:
                print("  >= 8%: COMPELLING -- compute saving alone justifies retokenizing")
            elif gain >= 5:
                print("  5-8%: MARGINAL -- embedding/storage wins stand, but the")
                print("        vendored 128k gets most of them at lower risk")
            else:
                print("  < 5%: NOT WORTH IT -- use the vendored Llama-3 128k")
            print("=" * 78)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "fertility.json"), "w") as fh:
        json.dump({"tokens_per_mb": results, "weighted": weighted}, fh, indent=2)
    print(f"\nwrote {os.path.join(out_dir, 'fertility.json')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", required=True)
    ap.add_argument("--gemma", default=None)
    ap.add_argument("--llama3", default=None)
    ap.add_argument("--vocab-size", type=int, default=64000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--candidate", default=None,
                    help="skip training, use this tokenizer.json")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cand = args.candidate or train_candidate(
        args.sample_dir, args.vocab_size,
        os.path.join(args.out, f"custom{args.vocab_size//1000}k.json"),
    )
    toks = load_tokenizers(cand, args.gemma, args.llama3)
    print(f"\nmeasuring {len(toks)} tokenizers: {', '.join(toks)}")
    report(measure(toks, args.sample_dir), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
