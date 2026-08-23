# RoPE-aware DCP<->HF state-dict adapter for the ezpz agpt models.
#
# WHY THIS EXISTS
# ---------------
# The core ``Llama3StateDictAdapter`` (torchtitan/models/llama3/
# state_dict_adapter.py) unconditionally applies HuggingFace's
# interleaved->rotate_half permutation to the Q and K projection weights in
# ``to_hf`` (and the reverse in ``from_hf``). That permutation is correct ONLY
# for checkpoints trained with ``ComplexRoPE`` (the interleaved / adjacent-pair
# convention that matches original Meta Llama).
#
# The agpt ``_real`` flavors train with ``CosSinRoPE``, whose weight layout is
# the rotate_half convention -- i.e. ALREADY HF-native (CosSinRoPE._rotate_half
# is byte-identical to HF's rotate_half). Running such a checkpoint through the
# unconditional permute WOULD scramble Q/K channel pairing, corrupting the
# exported weights, so this adapter skips the permute for cos_sin flavors. A
# DCP-vs-HF weight compare confirmed the skip-permute export is bit-faithful.
#
# NOTE: this adapter fixes a real (latent) export-correctness issue, but it was
# NOT the cause of the Polaris 20B "eval gibberish" -- that was a separate
# training-data / tokenizer mismatch (Llama2-tokenized data vs a gemma eval
# tokenizer). See docs/reference/known-bugs/polaris-20b-tokenizer-mismatch.md.
#
# This adapter detects the RoPE type from the built ``model_config`` and:
#   - CosSinRoPE  -> SKIP the Q/K permute (weights are already HF-native).
#   - ComplexRoPE -> defer to the parent (unchanged, correct complex path).
#
# So it is safe for ALL agpt flavors: complex checkpoints convert exactly as
# before; cos_sin (``_real``) checkpoints now convert correctly. The convert
# flavor still must match how the model was trained, because the RoPE type is
# read from the built ``model_config``, not from the checkpoint on disk -- use
# ``--model_flavor <size>_real`` for cos_sin chains.

from typing import Any

from torchtitan.models.common.rope import ComplexRoPE, CosSinRoPE
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter


class AgptStateDictAdapter(Llama3StateDictAdapter):
    """Llama3 adapter that is aware of the CosSinRoPE (``_real``) convention."""

    def __init__(self, model_config, hf_assets_path):
        super().__init__(model_config, hf_assets_path)
        # Detect RoPE convention from the built config -- same source
        # StateDictAdapter._validate_hf_rope_config inspects. All layers share
        # one convention, so the first layer is representative.
        rope = model_config.layers[0].attention.rope
        self._is_cos_sin = isinstance(rope, CosSinRoPE.Config)
        if not self._is_cos_sin and not isinstance(rope, ComplexRoPE.Config):
            # Unknown RoPE subclass: fall back to the parent's complex behavior
            # (its permute) rather than silently skipping -- but flag it.
            import logging

            logging.getLogger().warning(
                "AgptStateDictAdapter: unrecognized RoPE type %s; using the "
                "complex (parent) permute path.",
                type(rope).__name__,
            )

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        # Complex checkpoints: parent already does the right thing.
        if not self._is_cos_sin:
            return super().to_hf(state_dict)

        # CosSinRoPE: HF-native rotate_half layout -> map keys, NO Q/K permute.
        to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
        hf_state_dict: dict[str, Any] = {}
        import re

        for key, value in state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = to_hf_map.get(abstract_key)
                if new_key is None:
                    continue
                # NOTE: intentionally NO _permute on wq/wk here.
                new_key = new_key.format(layer_num)
            else:
                if (
                    self.model_config.enable_weight_tying
                    and key == "lm_head.weight"
                ):
                    if self.fqn_to_index_mapping:
                        self.fqn_to_index_mapping.pop("lm_head.weight", None)
                    continue
                new_key = to_hf_map[key]
            hf_state_dict[new_key] = value
        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        # Complex checkpoints: parent (includes the ComplexRoPE guard + reverse
        # permute).
        if not self._is_cos_sin:
            return super().from_hf(hf_state_dict)

        # CosSinRoPE: HF-native -> map keys, NO reverse permute, and DON'T assert
        # ComplexRoPE (the parent's guard would wrongly reject a cos_sin model).
        import re

        if (
            self.model_config.enable_weight_tying
            and "lm_head.weight" not in hf_state_dict
        ):
            assert "model.embed_tokens.weight" in hf_state_dict
            hf_state_dict["lm_head.weight"] = hf_state_dict[
                "model.embed_tokens.weight"
            ]

        state_dict: dict[str, Any] = {}
        for key, value in hf_state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = self.from_hf_map[abstract_key]
                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = self.from_hf_map[key]
                if new_key is None:
                    continue
            state_dict[new_key] = value
        return state_dict
