# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "rl" / "scripts" / "repair_agpt_hf_artifact.py"
_spec = importlib.util.spec_from_file_location("repair_agpt_hf_artifact", SCRIPT)
assert _spec and _spec.loader
repair = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repair)


def test_exact_training_chat_template_contract() -> None:
    template = repair.AGPT_GEMMA_CHAT_TEMPLATE
    assert "{{ bos_token }}" not in template
    assert "message['content'].lstrip('\\n')" in template
    assert "| trim" not in template
    assert "message['role'] == 'system' or message['role'] == 'user'" in template
    assert "{% generation %}" in template
    assert "{% endgeneration %}" in template


def test_repair_metadata(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"vocab_size": 256000, "bos_token_id": 1, "eos_token_id": 2})
    )
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({}))

    repair.repair_artifact(tmp_path)

    config = json.loads((tmp_path / "config.json").read_text())
    tokenizer_config = json.loads((tmp_path / "tokenizer_config.json").read_text())
    generation_config = json.loads((tmp_path / "generation_config.json").read_text())
    assert config["bos_token_id"] == 2
    assert config["eos_token_id"] == [1, 107]
    assert config["pad_token_id"] == 0
    assert generation_config == {
        "bos_token_id": 2,
        "eos_token_id": [1, 107],
        "pad_token_id": 0,
        "do_sample": True,
    }
    assert tokenizer_config["chat_template"] == repair.AGPT_GEMMA_CHAT_TEMPLATE
    assert (tmp_path / "chat_template.jinja").read_text() == (
        repair.AGPT_GEMMA_CHAT_TEMPLATE + "\n"
    )
