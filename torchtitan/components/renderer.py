# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, cast

import tyro
from renderers import create_renderer, Renderer
from renderers.configs import RendererConfig as PrimeRendererConfig

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import Configurable


@dataclass(kw_only=True, slots=True)
class RendererConfig(Configurable.Config):
    """Base config of a renderer; `build` returns a `renderers.Renderer` on TorchTitan's tokenizer.

    Subclasses: `RenderersConfigAdapter` for a renderer from the `renderers` library, and
    in-tree renderers such as `MuseGlimmerRendererConfig`.
    """

    # pyrefly: ignore[bad-override]
    def build(self, *, tokenizer: HuggingFaceTokenizer) -> Renderer:
        raise NotImplementedError


@dataclass(kw_only=True, slots=True)
class RenderersConfigAdapter(RendererConfig):
    """TorchTitan config adapter for a `renderers` library config.

    Example:

        from renderers import Qwen3RendererConfig

        from torchtitan.components.renderer import from_renderers
        from torchtitan.components.tokenizer import HuggingFaceTokenizer

        renderer = from_renderers(
            Qwen3RendererConfig(enable_thinking=False)
        ).build(tokenizer=HuggingFaceTokenizer(tokenizer_path="./Qwen3-0.6B"))
        prompt_ids = renderer.render_ids(
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=True,
        )
    """

    renderers_config: Annotated[PrimeRendererConfig, tyro.conf.Suppress]
    """The library's typed config for the model, e.g. `Qwen3RendererConfig(enable_thinking=False)`.
    Renderers and their options:
    https://github.com/PrimeIntellect-ai/renderers/blob/renderers-v0.1.11/docs/renderer-config.md"""

    def to_dict(self) -> dict[str, Any]:
        return {"renderers_config": self.renderers_config.model_dump(mode="json")}

    def build(self, *, tokenizer: HuggingFaceTokenizer) -> Renderer:
        if self.renderers_config.name == "auto":
            raise ValueError(
                f"AutoRendererConfig resolves by exact match of tokenizer.name_or_path ({tokenizer.tokenizer_path!r}) "
                "against renderers' MODEL_RENDERER_MAP, else falls back to DefaultRenderer (unsupported here). "
                "Pick the model's renderer, e.g. Qwen3RendererConfig(...)."
            )
        return create_renderer(
            tokenizer=RendererTokenizerWrapper(tokenizer), config=self.renderers_config
        )


def from_renderers(config: PrimeRendererConfig) -> RendererConfig:
    """Adapt a `renderers` config to TorchTitan's renderer config interface."""
    return RenderersConfigAdapter(renderers_config=config)


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
                self.renderer.build(tokenizer=tokenizer), self.extra_stop_token_ids
            ),
        )


class RendererTokenizerWrapper:
    """Adapt TorchTitan's loaded tokenizer to `renderers.OffsetTokenizer`.

    Protocol and bring-your-own-tokenizer guide:
    https://github.com/PrimeIntellect-ai/renderers/blob/renderers-v0.1.11/renderers/base.py#L668-L699
    https://github.com/PrimeIntellect-ai/renderers/blob/renderers-v0.1.11/README.md#install

    `renderers` needs Hugging Face-style special-token attributes, raw encoding
    without automatic BOS/EOS, token-to-id lookup, and character offsets. The
    offsets identify tokens from message content (`is_content`). This adapter
    exposes that interface from TorchTitan's underlying `tokenizers.Tokenizer`;
    it does not load a second tokenizer.

    Example:

        from torchtitan.components.tokenizer import HuggingFaceTokenizer
        from torchtitan.components.renderer import RendererTokenizerWrapper

        tokenizer = RendererTokenizerWrapper(
            HuggingFaceTokenizer(tokenizer_path="./Qwen3-0.6B")
        )
        tokenizer.encode("hi")  # [6023]
        tokenizer(
            "hi",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )  # ids + character offsets
        tokenizer.convert_tokens_to_ids("<|im_end|>")  # 151645
    """

    def __init__(self, tokenizer: HuggingFaceTokenizer):
        # Keep the TorchTitan wrapper for its compiled chat template and the
        # `tokenizers.Tokenizer` backend for offsets and token -> id lookup.
        self._tokenizer = tokenizer
        self._tokenizer_backend = tokenizer.tokenizer
        self.name_or_path = tokenizer.tokenizer_path
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.bos_token_id = tokenizer.bos_id
        self.eos_token_id = tokenizer.eos_id
        self.all_special_tokens = list(
            dict.fromkeys(self._special_token_variables().values())
        )
        # `tokenizers` returns None for unknown tokens; it has no unk id.
        self.unk_token_id = None

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
        tokenizer backend, so provide those variables here without loading a
        second tokenizer or silently changing the rendered text.
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

    def encode(
        self, text: str, add_special_tokens: bool = False, **kwargs
    ) -> list[int]:
        return self._tokenizer_backend.encode(
            text, add_special_tokens=add_special_tokens
        ).ids

    def decode(self, token_ids, skip_special_tokens: bool = False, **kwargs) -> str:
        return self._tokenizer_backend.decode(
            list(token_ids), skip_special_tokens=skip_special_tokens
        )

    def convert_tokens_to_ids(
        self, tokens: str | list[str]
    ) -> int | None | list[int | None]:
        if isinstance(tokens, str):
            return self._tokenizer_backend.token_to_id(tokens)
        return [self._tokenizer_backend.token_to_id(token) for token in tokens]

    def __call__(
        self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool
    ) -> dict:
        encoding = self._tokenizer_backend.encode(
            text, add_special_tokens=add_special_tokens
        )
        output: dict[str, Any] = {"input_ids": encoding.ids}
        if return_offsets_mapping:
            output["offset_mapping"] = encoding.offsets
        return output
