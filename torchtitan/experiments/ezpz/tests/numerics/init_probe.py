# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import sys

import torch

tree = sys.argv[1]
sys.path.insert(0, tree)
from torchtitan.experiments.ezpz.agpt import agpt_configs, set_ezpz_max_context_length

set_ezpz_max_context_length(256)
torch.manual_seed(42)
m = agpt_configs["debugmodel"].build()
m.init_states()
out = {"tree": tree.split("/")[-1]}
# compare tensors that are IDENTICALLY shaped on both sides (not w1/w3 vs w13)
for n in (
    "tok_embeddings.weight",
    "lm_head.weight",
    "norm.weight",
    "layers.0.attention_norm.weight",
    "layers.0.feed_forward.w2.weight",
):
    p = dict(m.named_parameters())[n]
    out[n] = [float(p.double().sum()), float(p.double().std())]
print("JSON" + json.dumps(out))
