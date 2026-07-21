# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Fix a consolidated agpt-2b HF checkpoint's eos_token_id.

FSDP consolidation inherits config.json from the pretrain base, which carries the
stock Llama defaults (eos_token_id=2, bos_token_id=1). Those are WRONG for the
gemma-7b tokenizer this model uses (real ids: <pad>=0, <eos>=1, <bos>=2,
<start_of_turn>=106, <end_of_turn>=107) AND not what the SFT'd model emits: the
gemma chat template ends the assistant turn with <end_of_turn> (107), so that is
the token the model learned to stop on.

Left unfixed, GRPO/vLLM generation never stops (it waits for token 2 = <bos>,
which the model never emits), so every rollout runs to max_completion_length --
wasted compute + the HBM/oneCCL OOM that killed early CoT GRPO runs. Setting
eos_token_id to BOTH <eos> (1) and <end_of_turn> (107) makes every consumer
(vLLM server, HF .generate, evals) terminate correctly with no per-launcher flag.

Usage: python fix_ckpt_eos.py <hf_checkpoint_dir>
"""

import json
import sys


def main() -> None:
    ck = sys.argv[1]
    cfg_path = ck + "/config.json"
    cfg = json.load(open(cfg_path))
    cfg["eos_token_id"] = [1, 107]  # <eos> + <end_of_turn>
    cfg["bos_token_id"] = 2
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    json.dump(
        {
            "eos_token_id": [1, 107],
            "bos_token_id": 2,
            "pad_token_id": 0,
            "do_sample": True,
        },
        open(ck + "/generation_config.json", "w"),
        indent=2,
    )
    print("[eos-fix] %s: eos_token_id -> [1, 107], bos -> 2, +generation_config" % ck)


if __name__ == "__main__":
    main()
