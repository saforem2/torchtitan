# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Isolate the bf16-master freeze mechanism: it is parameter SCALE, not layer type.

Gives four parameters identical synthetic gradients and runs the same AdamW
for the same number of steps, varying only the init scale and the master
dtype. Because every element receives a gradient on every step, this removes
the "maybe those rows were simply never visited" confound that muddies the
embedding measurement on a real model.

Measured 2026-08-14, torch 2.13:

    embedding std=1.0  bf16: frac moved after 20 AdamW steps = 0.1981
    embedding std=1.0  fp32: frac moved after 20 AdamW steps = 1.0000
    linear    std=0.02 bf16: frac moved after 20 AdamW steps = 1.0000
    norm      init=1.0 bf16: frac moved after 20 AdamW steps = 0.0000

i.e. a bf16 master silently drops updates for ANY parameter initialized near
1.0 -- `RMSNorm.weight` (uniform 1.0, so it never moves at all) and
`tok_embeddings.weight` (agpt's `_EMBEDDING_INIT` is `normal_(std=1.0)`, so
~80% of its elements never move). At std=0.02 the bf16 ULP is small enough
relative to the update that nothing is lost, which is why the original
investigation concluded the linear layers were fine -- they are.

Context:
  docs/records/proposals/30b-exp/exp02-fp32-norms-ablation.md
  docs/reference/guides/training-dtype-bf16-norm-freeze.md

Runs on CPU in seconds; no distributed setup required.
"""

import torch
import torch.nn as nn

CASES = [
    ("embedding std=1.0  bf16", 1.0, torch.bfloat16),
    ("embedding std=1.0  fp32", 1.0, torch.float32),
    ("linear    std=0.02 bf16", 0.02, torch.bfloat16),
    ("norm      init=1.0 bf16", None, torch.bfloat16),
]


def main() -> None:
    torch.manual_seed(0)
    for label, std, dtype in CASES:
        if std is None:
            w = torch.ones(4096, 256, dtype=dtype)
        else:
            w = (torch.randn(4096, 256) * std).to(dtype)
        w = nn.Parameter(w)
        opt = torch.optim.AdamW([w], lr=8e-4, betas=(0.9, 0.95), weight_decay=0.1)
        before = w.detach().float().clone()
        # Same gradient scale for every case, applied to every element.
        grad = torch.randn(4096, 256) * 1e-3
        for _ in range(20):
            w.grad = grad.to(dtype).clone()
            opt.step()
        moved = (w.detach().float() != before).float().mean().item()
        print(f"{label}: frac moved after 20 AdamW steps = {moved:.4f}")


if __name__ == "__main__":
    main()
