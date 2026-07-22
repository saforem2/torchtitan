# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Merge a Monarch/train_upstream GRPO+LoRA DCP checkpoint into a plain HF model.

The RL engine (torchtitan.experiments.rl, driven by the ezpz reason_agpt overlay)
saves a full-training-state DCP checkpoint whose *model* side holds the FROZEN base
weights PLUS the trained LoRA adapters:

  layers.{i}.attention.qkv_linear.wq.weight   (2048, 2048)  base, SPLIT on save
  layers.{i}.attention.qkv_linear.wk.weight   ( 512, 2048)  base, SPLIT on save
  layers.{i}.attention.qkv_linear.wv.weight   ( 512, 2048)  base, SPLIT on save
  layers.{i}.attention.qkv_linear.wqkv.lora_a.weight  (rank, 2048)  adapter on FUSED wqkv
  layers.{i}.attention.qkv_linear.wqkv.lora_b.weight  (3072, rank)  adapter on FUSED wqkv
  layers.{i}.attention.wo.weight              (2048, 2048)  base
  layers.{i}.attention.wo.lora_a.weight       (rank, 2048)  adapter
  layers.{i}.attention.wo.lora_b.weight       (2048, rank)  adapter
  ... plus tok_embeddings / norm / lm_head / feed_forward / norms (no LoRA).

The base fused ``wqkv`` weight was split into wq/wk/wv by FusedQKVLinear's
``_split_qkv_on_save`` state-dict hook, but the LoRA adapters (children of the
wqkv module) are NOT touched by that hook, so they remain on the FUSED wqkv.
convert_to_hf.py alone would silently DROP every lora_* key (they are unmapped in
the Llama3 from_hf_map), yielding the un-trained base. So we MUST merge here.

Merge math (LoRA delta W' = W + (alpha/rank) * B @ A):
  - wo:   base += scaling * (wo.lora_b @ wo.lora_a)                    -> (2048, 2048)
  - wqkv: delta = scaling * (wqkv.lora_b @ wqkv.lora_a)               -> (3072, 2048),
          then split delta the SAME way FusedQKVLinear._split_qkv_on_save splits
          the fused weight (reshape (n_kv, r, hd, in); q=[:, :hpk], k=[:, hpk],
          v=[:, hpk+1]) and add each slice to the base wq/wk/wv.

The result is a pure-base torchtitan state dict (wq/wk/wv/wo/...), which is then
remapped to HF layout by Llama3StateDictAdapter.to_hf (which applies HF's Q/K rope
permutation) and written as a single model.safetensors. config.json + tokenizer
(with gemma chat_template + eos_token_id=[1,107] fix) are copied from the base HF
dir the LoRA was trained on.

Single-process, CPU-only. Run on a login node in the rl-vllm venv (needs torch +
safetensors); the ~12 GB fp32 model fits comfortably in login-node RAM.

  venvs/rl-vllm/bin/python \
    torchtitan/experiments/ezpz/scripts/eval/merge_lora_dcp_to_hf.py \
      --dcp outputs/rl_lora_agpt2b_cot/checkpoint/step-100 \
      --base-hf outputs/sft/agpt2b-gsm8k-r1cot-8n/checkpoint-16-hf \
      --out outputs/evals/cot/rl_lora_step100_merged_hf \
      --lora-rank 8 --lora-alpha 16 --model-flavor 2b-rl
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader

# Model-side top-level prefixes in the RL DCP (everything else is optimizer /
# train_state / dataloader / lr_scheduler and is not part of the HF export).
_MODEL_PREFIXES = ("tok_embeddings", "layers", "norm", "lm_head")
# Tokenizer / config files copied verbatim from the base HF dir (carries the
# gemma chat_template + the eos_token_id=[1,107] fix that the eval needs).
_ASSET_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


@torch.inference_mode()
def load_model_state(dcp_dir: str) -> dict[str, torch.Tensor]:
    """dcp.load only the model-side keys (base + LoRA), single process."""
    reader = FileSystemReader(dcp_dir)
    sd_md = reader.read_metadata().state_dict_metadata
    model_keys = [k for k in sd_md if k.split(".")[0] in _MODEL_PREFIXES]
    state_dict = {
        k: torch.empty(tuple(sd_md[k].size), dtype=sd_md[k].properties.dtype)
        for k in model_keys
    }
    dcp.load(state_dict, checkpoint_id=dcp_dir)
    return state_dict


@torch.inference_mode()
def merge_lora(
    sd: dict[str, torch.Tensor],
    *,
    scaling: float,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Fold LoRA adapters into the split base weights; return pure-base tt sd."""
    hpk = n_heads // n_kv_heads  # heads_per_kv (Q heads per KV group)
    r = hpk + 2  # r_dim: Q-per-group + K + V
    merged: dict[str, torch.Tensor] = {}
    n_wo, n_wqkv = 0, 0

    # collect layer indices from wo base weights (every layer has one)
    layer_ids = sorted(
        int(k.split(".")[1])
        for k in sd
        if k.startswith("layers.") and k.endswith(".attention.wo.weight")
    )

    for i in layer_ids:
        p = f"layers.{i}.attention"

        # --- wo: base += scaling * (B @ A) ---
        wo = sd[f"{p}.wo.weight"].to(torch.float32)
        la = sd.get(f"{p}.wo.lora_a.weight")
        lb = sd.get(f"{p}.wo.lora_b.weight")
        if la is not None and lb is not None:
            wo = wo + scaling * (lb.to(torch.float32) @ la.to(torch.float32))
            n_wo += 1
        merged[f"{p}.wo.weight"] = wo

        # --- wqkv (fused adapter) -> split delta onto base wq/wk/wv ---
        wq = sd[f"{p}.qkv_linear.wq.weight"].to(torch.float32)
        wk = sd[f"{p}.qkv_linear.wk.weight"].to(torch.float32)
        wv = sd[f"{p}.qkv_linear.wv.weight"].to(torch.float32)
        qa = sd.get(f"{p}.qkv_linear.wqkv.lora_a.weight")
        qb = sd.get(f"{p}.qkv_linear.wqkv.lora_b.weight")
        if qa is not None and qb is not None:
            in_dim = qa.shape[1]
            delta = scaling * (qb.to(torch.float32) @ qa.to(torch.float32))
            # split exactly like FusedQKVLinear._split_qkv_on_save (weight, ndim=4)
            w = delta.reshape(n_kv_heads, r, head_dim, in_dim)
            wq = wq + w[:, :hpk].reshape(-1, in_dim)
            wk = wk + w[:, hpk].reshape(-1, in_dim)
            wv = wv + w[:, hpk + 1].reshape(-1, in_dim)
            n_wqkv += 1
        merged[f"{p}.qkv_linear.wq.weight"] = wq
        merged[f"{p}.qkv_linear.wk.weight"] = wk
        merged[f"{p}.qkv_linear.wv.weight"] = wv

        # --- carry the rest of this layer's non-lora params through ---
        for suf in (
            "attention_norm.weight",
            "ffn_norm.weight",
            "feed_forward.w1.weight",
            "feed_forward.w2.weight",
            "feed_forward.w3.weight",
        ):
            merged[f"layers.{i}.{suf}"] = sd[f"layers.{i}.{suf}"].to(torch.float32)

    # --- non-layer params (embeddings, final norm, lm_head) ---
    for k in ("tok_embeddings.weight", "norm.weight", "lm_head.weight"):
        merged[k] = sd[k].to(torch.float32)

    print(f"[merge] folded LoRA into {n_wo} wo + {n_wqkv} wqkv layers "
          f"(scaling={scaling}, hpk={hpk}, r={r}, head_dim={head_dim})")
    # sanity: no lora keys should survive
    leftover = [k for k in merged if "lora" in k]
    assert not leftover, f"lora keys leaked into merged sd: {leftover[:4]}"
    return merged


@torch.inference_mode()
def to_hf_and_save(
    merged_tt: dict[str, torch.Tensor],
    *,
    base_hf: str,
    out_dir: str,
    model_flavor: str,
    export_dtype: torch.dtype,
) -> None:
    from torchtitan.experiments.ezpz.agpt import model_registry as agpt_registry
    from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter

    # base spec (NO converters) -> plain base model_config for the adapter.
    model_spec = agpt_registry(model_flavor)
    model_config = model_spec.model
    adapter = Llama3StateDictAdapter(model_config, base_hf)

    hf_sd = adapter.to_hf(merged_tt)
    hf_sd = {k: v.to(export_dtype).contiguous() for k, v in hf_sd.items()}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(hf_sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    print(f"[save] wrote {len(hf_sd)} tensors -> {out / 'model.safetensors'} "
          f"(dtype={export_dtype})")

    # copy config + tokenizer (chat_template + eos fix) from the base HF dir.
    base = Path(base_hf)
    for f in _ASSET_FILES:
        src = base / f
        if src.exists():
            shutil.copy2(src, out / f)
    # config.json carries dtype -- align with the export dtype so vLLM loads it.
    cfg_path = out / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        cfg["torch_dtype"] = str(export_dtype).replace("torch.", "")
        cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"[save] copied config+tokenizer from {base_hf}")
    print(f"[done] merged HF checkpoint at: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcp", required=True, help="RL DCP checkpoint dir (step-N)")
    ap.add_argument("--base-hf", required=True, help="base HF dir (config+tokenizer)")
    ap.add_argument("--out", required=True, help="output merged HF dir")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--model-flavor", default="2b-rl")
    ap.add_argument("--n-heads", type=int, default=16)
    ap.add_argument("--n-kv-heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument(
        "--export-dtype",
        default="float32",
        choices=["float16", "bfloat16", "float32"],
    )
    args = ap.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    scaling = args.lora_alpha / args.lora_rank
    print(f"[load] dcp={args.dcp}")
    sd = load_model_state(args.dcp)
    print(f"[load] {len(sd)} model-side tensors loaded")
    merged = merge_lora(
        sd,
        scaling=scaling,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        head_dim=args.head_dim,
    )
    to_hf_and_save(
        merged,
        base_hf=args.base_hf,
        out_dir=args.out,
        model_flavor=args.model_flavor,
        export_dtype=dtype_map[args.export_dtype],
    )


if __name__ == "__main__":
    main()
