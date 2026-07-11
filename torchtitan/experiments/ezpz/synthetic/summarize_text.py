#!/usr/bin/env python3
"""Summarize source documents into high-quality synthetic text.

Step 2 of the "summarize olmo-mix" synthetic-data POC. Reads the JSONL text
slice produced by detok_to_text.py ({"id", "n_tok", "text"}), and for each
document generates a concise, faithful, self-contained summary with an
instruction-tuned model. Writes JSONL {"id", "n_tok_src", "summary"}.

The summaries are the synthetic corpus: dense restatements of the source facts
in clean prose, meant to be retokenized (step 3) and used as mid-training data.
The prompt is tuned for FAITHFULNESS (no new facts, no opinions) so the
synthetic tokens carry the source's information at higher density.

This POC uses plain HuggingFace `transformers` generation on a single XPU node
(the frameworks .venv already ships transformers). vLLM would be faster at
scale, but there is no XPU vLLM venv in this clone and building one touches the
torch stack (forbidden by the ezpz env rules); for a ~2k-doc pilot, batched HF
generate is adequate. Swap in vLLM later once the recipe is validated.

Usage (on a compute node with an XPU, via the PBS wrapper submit_summarize.sh):
  python -m torchtitan.experiments.ezpz.synthetic.summarize_text \\
    --in outputs/synthetic/wiki_slice.jsonl \\
    --out outputs/synthetic/wiki_summaries.jsonl \\
    --model /flare/AuroraGPT/azton/models/Llama-3.1-8B-exvocab \\
    --max-src-chars 6000 --max-new-tokens 512 --batch-size 8
"""
from __future__ import annotations

import argparse
import json
import os
import time

SYSTEM_PROMPT = (
    "You are a precise summarizer. Given a source document, write a concise, "
    "self-contained summary that faithfully preserves the key facts, names, "
    "numbers, and relationships in the source. Do not add information that is "
    "not in the source, do not speculate, and do not include opinions or "
    "meta-commentary. Write clear, well-structured prose."
)

USER_TEMPLATE = (
    "Summarize the following document. Preserve the important factual content "
    "and keep it self-contained.\n\n--- DOCUMENT ---\n{doc}\n--- END ---\n\n"
    "Summary:"
)


def _read_jsonl(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input JSONL from detok_to_text")
    ap.add_argument("--out", required=True, help="output JSONL of summaries")
    ap.add_argument(
        "--model",
        default="/flare/AuroraGPT/azton/models/Llama-3.1-8B-exvocab",
        help="instruction-tuned HF model dir (has a chat_template)",
    )
    ap.add_argument(
        "--max-src-chars",
        type=int,
        default=6000,
        help="truncate each source doc to this many chars before summarizing",
    )
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="only summarize the first N docs (0 = all); for quick smokes",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0.0 = greedy (deterministic, most faithful)",
    )
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # XPU is the only accelerator on Aurora compute nodes. Fail loudly if it's
    # absent (e.g. accidentally launched on a login node) rather than silently
    # crawling on CPU for an 8B model.
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError(
            "no XPU available -- run this on an Aurora compute node "
            "(submit_summarize.sh), not a login node"
        )
    device = "xpu"

    rows = _read_jsonl(args.inp)
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"loaded {len(rows)} docs from {args.inp}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # left-pad so batched generation aligns the newest tokens at the right edge
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    # Some of the on-disk instruct checkpoints ship a chat_template that
    # IGNORES add_generation_prompt (it never appends the assistant-turn cue,
    # so the model isn't told it's its turn to speak and generation degrades).
    # Detect that case once and, for ChatML-style templates, append the
    # assistant header ourselves.
    _probe_msgs = [{"role": "user", "content": "x"}]
    _with = tok.apply_chat_template(_probe_msgs, tokenize=False, add_generation_prompt=True)
    _without = tok.apply_chat_template(_probe_msgs, tokenize=False, add_generation_prompt=False)
    _needs_manual_cue = _with == _without
    _is_chatml = "<|im_start|>" in _with
    if _needs_manual_cue and _is_chatml:
        _gen_suffix = "<|im_start|>assistant\n"
        print("note: chat_template ignores add_generation_prompt; "
              "appending ChatML assistant cue manually", flush=True)
    elif _needs_manual_cue:
        # Non-ChatML template that also ignores the flag: fall back to a plain
        # newline cue rather than guessing a header we don't recognize.
        _gen_suffix = "\n"
        print("warning: chat_template ignores add_generation_prompt and is not "
              "ChatML; using a bare newline cue", flush=True)
    else:
        _gen_suffix = ""

    def build_prompt(doc: str) -> str:
        doc = doc[: args.max_src_chars]
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(doc=doc)},
        ]
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return p + _gen_suffix

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    written = 0
    t0 = time.time()
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id)
    if args.temperature and args.temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=args.temperature)
    else:
        gen_kwargs.update(do_sample=False)

    with open(args.out, "w") as f:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            prompts = [build_prompt(r["text"]) for r in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=False).to(device)
            with torch.no_grad():
                out = model.generate(**enc, **gen_kwargs)
            # slice off the prompt tokens; decode only the newly generated span
            gen = out[:, enc["input_ids"].shape[1]:]
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            for r, s in zip(batch, texts):
                f.write(
                    json.dumps(
                        {"id": r["id"], "n_tok_src": r.get("n_tok"), "summary": s.strip()}
                    )
                    + "\n"
                )
                written += 1
            f.flush()
            rate = written / max(time.time() - t0, 1e-6)
            print(f"  {written}/{len(rows)} summarized ({rate:.2f} docs/s)", flush=True)

    dt = time.time() - t0
    print(f"=== wrote {written} summaries to {args.out} in {dt:.0f}s ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
