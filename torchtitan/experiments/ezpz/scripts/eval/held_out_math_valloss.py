"""Held-out math val-loss scorer -- the PRIMARY metric for the anneal A/B.

The anneal experiment decides flat (constant-LR control) vs WSD (decay-to-0)
on GENERALIZATION, not training loss (comparing final train loss is invalid:
the WSD arm ends at LR ~0 so its train loss is structurally lower -- the exact
apples-to-oranges error of the first A/B). Held-out math val loss is the smooth,
statistically powerful signal: it averages NLL over tens of thousands of tokens,
where 5-shot GSM8K at a 2B base sits near floor (~2-4%) and is underpowered.

This scorer loads ONE converted HF checkpoint and computes mean token-level
NLL (natural-log cross-entropy) over a FROZEN holdout jsonl (built by
build_math_holdout.py). Run it for every eval point -- base, flat, wsd -- on
the SAME holdout with the SAME token budget so the numbers are comparable.

Decision (pre-registered): PASS if valloss(wsd) < valloss(flat) on the primary
(FineMath) holdout, AND HellaSwag(wsd) drop <= 1.5pp vs base (measured
separately via eval-2b-v2.sh). See anneal_ab_verdict.py.

Determinism: documents are read in file order (the holdout is frozen), packed
into fixed-length seq_len windows in order (no shuffle), and the same number of
windows is scored for every checkpoint (--max-windows). Loss is the token-count
weighted mean NLL over all scored windows, so it is invariant to how tokens
happen to fall across the last window.

Environment: run in the tt-lm-eval venv (HF transformers on XPU), the same env
eval-2b-v2.sh uses for lm-eval. Example (single XPU tile):

    python3 -m torchtitan.experiments.ezpz.scripts.eval.held_out_math_valloss \\
        --hf-dir outputs/evals/anneal/mds-wsd/step-1600/hf \\
        --holdout torchtitan/experiments/ezpz/eval/holdouts/finemath_holdout.jsonl \\
        --seq-len 8192 --max-windows 64 --device xpu:0 \\
        --out outputs/evals/anneal/mds-wsd/step-1600/valloss_finemath.json

The tokenizer is read from --hf-dir (the converted checkpoint carries the
gemma-7b tokenizer copied by eval-2b-v2.sh), so scoring uses the exact
tokenization the model was trained with. The MDS model (vocab 256000) and the
olmo model (vocab 256128) share the first 256000 gemma ids, so the same
holdout ids are valid for both -- each is scored with its own head width.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _read_holdout(path: Path) -> list[str]:
    texts = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            texts.append(json.loads(line)["text"])
    if not texts:
        raise SystemExit(f"no documents in holdout {path}")
    return texts


def _pack_windows(
    token_ids: list[int], seq_len: int, max_windows: int
) -> list[list[int]]:
    """Pack a flat token stream into non-overlapping seq_len windows, in order.

    Drops the final partial window so every scored window has exactly seq_len
    tokens -- this keeps the token budget identical across checkpoints and
    avoids a short-window bias. Returns at most max_windows windows.
    """
    windows = []
    for start in range(0, len(token_ids) - seq_len + 1, seq_len):
        windows.append(token_ids[start : start + seq_len])
        if len(windows) >= max_windows:
            break
    return windows


@torch.inference_mode()
def score(
    hf_dir: Path,
    holdout: Path,
    seq_len: int,
    max_windows: int,
    device: str,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(str(hf_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(hf_dir), torch_dtype=torch.bfloat16
    )
    model.to(device)
    model.eval()

    texts = _read_holdout(holdout)
    # Tokenize each doc and concatenate with an EOS boundary between docs, so a
    # window never silently bridges two unrelated documents without a separator
    # (matches how packed pretraining streams delimit documents).
    eos_id = tokenizer.eos_token_id
    flat_ids: list[int] = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        flat_ids.extend(ids)
        if eos_id is not None:
            flat_ids.append(eos_id)

    windows = _pack_windows(flat_ids, seq_len, max_windows)
    if not windows:
        raise SystemExit(
            f"holdout too short: {len(flat_ids)} tokens < seq_len {seq_len}; "
            "lower --seq-len or add docs to the holdout"
        )

    # The gemma-7b tokenizer has vocab 256128, but the MDS model is vocab 256000.
    # Any token id in [256000, 256128) (gemma <unused*> extras) is out of range
    # for the MDS embedding/lm_head -> a device-side index assert mid-run. Rare
    # in math text, but fail loud + early with a clear message rather than a
    # cryptic CUDA/XPU assert deep in the forward. (Ids are identical across
    # checkpoints so this does not affect comparability -- only crash safety.)
    model_vocab = int(model.get_input_embeddings().weight.shape[0])
    max_id = max(flat_ids)
    if max_id >= model_vocab:
        raise SystemExit(
            f"holdout contains token id {max_id} >= model vocab {model_vocab}; "
            "the tokenizer emits ids the model cannot embed (gemma 256128 vs "
            "MDS 256000). Rebuild the holdout without those tokens, or score a "
            "model whose vocab covers them."
        )

    total_nll = 0.0
    total_tokens = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    for window in windows:
        input_ids = torch.tensor([window], device=device)
        logits = model(input_ids).logits
        # Standard next-token shift: predict token t+1 from tokens <= t.
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        nll = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
        )
        total_nll += nll.item()
        total_tokens += shift_labels.numel()

    mean_nll = total_nll / total_tokens
    result = {
        "hf_dir": str(hf_dir),
        "holdout": str(holdout),
        "seq_len": seq_len,
        "num_windows": len(windows),
        "num_tokens": total_tokens,
        "mean_nll": mean_nll,
        "perplexity": float(torch.exp(torch.tensor(mean_nll))),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--hf-dir", required=True, type=Path, help="Converted HF checkpoint dir."
    )
    parser.add_argument(
        "--holdout", required=True, type=Path, help="Frozen holdout jsonl."
    )
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument(
        "--max-windows",
        type=int,
        default=64,
        help="Score at most this many seq_len windows (fixes the token budget "
        "across checkpoints; default 64 = ~0.5M tokens at seq_len 8192).",
    )
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument(
        "--out", type=Path, default=None, help="Write result json here (also printed)."
    )
    args = parser.parse_args()

    result = score(
        args.hf_dir, args.holdout, args.seq_len, args.max_windows, args.device
    )
    print(json.dumps(result, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
