# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AuroraGPT renderer support for RL.

Two things AuroraGPT RL needs that upstream's renderer layer does not provide:

1. ``DefaultRendererConfig`` (Hugging Face ``apply_chat_template``). Upstream's
   ``RenderersConfigAdapter`` rejects it because TorchTitan's template
   rendering lacks the tokenizer's special-token variables (``bos_token`` ...).
   :class:`HFTemplateTokenizerWrapper` supplies them from the tokenizer config
   TorchTitan already loaded, so the rendered text is identical to Hugging
   Face's without loading a second tokenizer.
2. Extra generation stop ids. ``DefaultRendererConfig`` does not infer
   AuroraGPT's ``<end_of_turn>`` boundary, so :class:`ExtraStopTokensRendererConfig`
   adds model-specific stop ids to whatever renderer it wraps.

Both used to live in ``torchtitan/components/renderer.py``; upstream (#4908
era) rewrote that file, so the additions are kept here instead of patching core.
"""

from dataclasses import dataclass
from typing import cast

from renderers import create_renderer, Renderer

from torchtitan.components.renderer import (
    from_renderers,
    RendererConfig,
    RenderersConfigAdapter,
    RendererTokenizerWrapper,
)
from torchtitan.components.tokenizer import HuggingFaceTokenizer


class HFTemplateTokenizerWrapper(RendererTokenizerWrapper):
    """``RendererTokenizerWrapper`` that can run the checkpoint's chat template."""

    def __init__(self, tokenizer: HuggingFaceTokenizer):
        super().__init__(tokenizer)
        # Keep the TorchTitan wrapper for its compiled chat template.
        self._tokenizer = tokenizer
        self.all_special_tokens = list(
            dict.fromkeys(self._special_token_variables().values())
        )

    def _special_token_variables(self) -> dict[str, str]:
        return {
            key: value
            for key, value in (self._tokenizer._hf_config or {}).items()
            if key.endswith("_token") and isinstance(value, str)
        }

    def apply_chat_template(self, messages, **kwargs) -> list[int] | str:
        """Render the checkpoint's Jinja template with Hugging Face semantics.

        ``DefaultRenderer`` needs the tokenizer's special-token variables and
        asks for token ids. TorchTitan already loaded the same template and
        tokenizer backend, so provide those variables here without silently
        changing the rendered text.
        """
        tokenize = kwargs.pop("tokenize", True)
        kwargs.pop("return_dict", None)
        special_tokens = self._special_token_variables()
        rendered = self._tokenizer.apply_chat_template(
            messages, **special_tokens, **kwargs
        )
        if not tokenize:
            return rendered
        return self.encode(rendered, add_special_tokens=False)


def _build_renderer(
    config: RendererConfig, tokenizer: HuggingFaceTokenizer
) -> Renderer:
    """Build ``config``, allowing the Hugging Face default renderer."""
    if (
        isinstance(config, RenderersConfigAdapter)
        and config.renderers_config.name == "default"
    ):
        return create_renderer(
            tokenizer=HFTemplateTokenizerWrapper(tokenizer),
            config=config.renderers_config,
        )
    return config.build(tokenizer=tokenizer)


class _RendererWithExtraStopTokens:
    """Delegate renderer behavior while extending role-boundary stop tokens."""

    def __init__(self, renderer: Renderer, extra_stop_token_ids: tuple[int, ...]):
        self._renderer = renderer
        self._extra_stop_token_ids = extra_stop_token_ids

    def __getattr__(self, name: str):
        return getattr(self._renderer, name)

    def get_stop_token_ids(self) -> list[int]:
        return list(
            dict.fromkeys(
                [
                    *self._renderer.get_stop_token_ids(),
                    *self._extra_stop_token_ids,
                ]
            )
        )


@dataclass(kw_only=True, slots=True)
class ExtraStopTokensRendererConfig(RendererConfig):
    """Wrap another renderer and add model-specific generation stop IDs."""

    renderer: RendererConfig
    extra_stop_token_ids: tuple[int, ...]

    def build(self, *, tokenizer: HuggingFaceTokenizer) -> Renderer:
        # The wrapper delegates the renderer protocol through __getattr__.
        return cast(
            Renderer,
            _RendererWithExtraStopTokens(
                _build_renderer(self.renderer, tokenizer), self.extra_stop_token_ids
            ),
        )


__all__ = [
    "ExtraStopTokensRendererConfig",
    "HFTemplateTokenizerWrapper",
    "from_renderers",
]
