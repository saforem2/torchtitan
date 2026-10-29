# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AuroraGPT RL renderer helpers (``experiments/ezpz/rl/renderer.py``)."""

import pytest
from renderers import DefaultRendererConfig, Qwen3RendererConfig

from torchtitan.components.renderer import from_renderers, RenderersConfigAdapter
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.experiments.ezpz.rl.renderer import ExtraStopTokensRendererConfig

_TOKENIZER_PATH = "tests/assets/tokenizer"


def test_extra_stop_tokens_preserve_order_and_deduplicate() -> None:
    tokenizer = HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)
    base = from_renderers(Qwen3RendererConfig(enable_thinking=False))
    base_stops = base.build(tokenizer=tokenizer).get_stop_token_ids()
    renderer = ExtraStopTokensRendererConfig(
        renderer=base,
        extra_stop_token_ids=(base_stops[-1], 123456),
    ).build(tokenizer=tokenizer)
    assert renderer.get_stop_token_ids() == [*base_stops, 123456]


def test_default_renderer_matches_hf_chat_template() -> None:
    transformers = pytest.importorskip("transformers")
    from renderers import create_renderer

    messages = [{"role": "user", "content": "hi"}]
    config = DefaultRendererConfig()
    hf = create_renderer(
        transformers.AutoTokenizer.from_pretrained(_TOKENIZER_PATH), config
    )
    titan = ExtraStopTokensRendererConfig(
        renderer=RenderersConfigAdapter(renderers_config=config),
        extra_stop_token_ids=(),
    ).build(tokenizer=HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH))
    assert titan.render_ids(messages, add_generation_prompt=True) == hf.render_ids(
        messages, add_generation_prompt=True
    )
