"""Write the config.json that convert_to_hf does not emit.

convert_to_hf writes weights, an index, and ezpz_export.json -- but no
config.json, so `lm_eval --model hf` fails on the export with:

    ValueError: Unrecognized model in <dir>. Should have a `model_type` key
    in its config.json.

The tokenizer assets dir cannot supply it: assets/hf/OLMo-2-1124-7B/config.json
describes OLMo-2-*7B* (hidden 4096, 32 layers), not our 26.2B model. Copying it
would produce a config that loads and silently mismatches the weights.

So derive the config FROM THE WEIGHTS. Every field is read out of the
safetensors header rather than transcribed from a registry, which means the
config cannot disagree with the tensors it describes -- the failure mode this
whole eval path is most exposed to. The registry values are used only as a
cross-check, and a mismatch aborts rather than writing.

agpt exports use Llama tensor names (model.layers.N.self_attn.q_proj.weight,
mlp.gate_proj, ...), so model_type is "llama" regardless of whose TOKENIZER
the run used. olmo2tok names the vocab, not the architecture.

Usage:
    python3 -m torchtitan.experiments.ezpz.eval.write_hf_config <hf_dir> \\
        [--head-dim 128] [--rope-theta 500000] [--expect k=v,k=v]
"""
import argparse
import json
import struct
from pathlib import Path


def read_header(safetensors_path: Path) -> dict:
    with open(safetensors_path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def build_config(hf_dir: Path, head_dim: int, rope_theta: float) -> dict:
    shards = sorted(hf_dir.glob("model-*-of-*.safetensors")) or sorted(
        hf_dir.glob("*.safetensors")
    )
    if not shards:
        raise SystemExit(f"no safetensors found in {hf_dir}")

    hdr: dict = {}
    for s in shards:
        hdr.update(read_header(s))

    def shape(key):
        if key not in hdr:
            raise SystemExit(f"missing tensor {key} -- is this an agpt export?")
        return hdr[key]["shape"]

    vocab, dim = shape("model.embed_tokens.weight")
    q = shape("model.layers.0.self_attn.q_proj.weight")
    k = shape("model.layers.0.self_attn.k_proj.weight")
    gate = shape("model.layers.0.mlp.gate_proj.weight")
    n_layers = 1 + max(
        int(name.split(".")[2]) for name in hdr if name.startswith("model.layers.")
    )

    if q[0] % head_dim or k[0] % head_dim:
        raise SystemExit(
            f"q/k out-features ({q[0]}/{k[0]}) not divisible by head_dim "
            f"{head_dim} -- pass the right --head-dim"
        )

    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": dim,
        "intermediate_size": gate[0],
        "num_hidden_layers": n_layers,
        "num_attention_heads": q[0] // head_dim,
        "num_key_value_heads": k[0] // head_dim,
        "head_dim": head_dim,
        "max_position_embeddings": 131072,
        "rope_theta": float(rope_theta),
        "rms_norm_eps": 1e-05,
        "vocab_size": vocab,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir", type=Path)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--rope-theta", type=float, default=500000.0)
    ap.add_argument(
        "--expect",
        default="",
        help=(
            "Comma-separated k=v the derived config MUST match, e.g. "
            "'hidden_size=6144,num_hidden_layers=64'. A mismatch aborts "
            "without writing -- use this to assert the export is the model "
            "you think it is."
        ),
    )
    args = ap.parse_args()

    cfg = build_config(args.hf_dir, args.head_dim, args.rope_theta)
    for field in (
        "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "vocab_size",
    ):
        print(f"  {field:22} {cfg[field]}")

    if args.expect:
        bad = []
        for pair in args.expect.split(","):
            key, _, want = pair.partition("=")
            key, want = key.strip(), want.strip()
            if str(cfg.get(key)) != want:
                bad.append(f"{key}: derived {cfg.get(key)}, expected {want}")
        if bad:
            print("  MISMATCH, refusing to write:")
            for line in bad:
                print(f"    {line}")
            return 1
        print(f"  cross-check OK ({args.expect})")

    out = args.hf_dir / "config.json"
    out.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
