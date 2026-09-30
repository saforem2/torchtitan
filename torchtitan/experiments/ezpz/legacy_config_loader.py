# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility loader for maintained ezpz dotted-command-line launchers.

Core TorchTitan deliberately exposes only operational command-line options via
``ConfigLoader``. ezpz still has maintained scheduler scripts that pass typed
recipe mutations such as ``--training.steps=3``. This module keeps that
compatibility local to ezpz; it does not restore the removed core
``ConfigManager`` API.
"""

from __future__ import annotations

import json
import pprint
import sys
from collections.abc import Sequence
from typing import Any

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config import ConfigLoader
from torchtitan.config.configs import CompileConfig
from torchtitan.config.override import parse_cli_imports


class LegacyConfigLoader:
    """Load a Python recipe and apply ezpz's validated legacy CLI mutations."""

    def parse_args(self, args: list[str] | None = None):
        tokens = list(sys.argv[1:] if args is None else args)
        module, config_name, tokens = self._extract_recipe(tokens)
        config = ConfigLoader._load_config(module, config_name)
        self._apply_tokens(config, tokens)
        ConfigLoader._validate_assets_path(config)
        return config

    @staticmethod
    def _extract_recipe(tokens: list[str]) -> tuple[str, str, list[str]]:
        values: dict[str, str] = {}
        remaining: list[str] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            matched = False
            for name in ("module", "config"):
                flag = f"--{name}"
                if token.startswith(f"{flag}="):
                    values[name] = token.split("=", 1)[1]
                    matched = True
                    break
                if token == flag:
                    if i + 1 >= len(tokens) or tokens[i + 1].startswith("--"):
                        raise ValueError(f"{flag} requires a value")
                    values[name] = tokens[i + 1]
                    i += 1
                    matched = True
                    break
            if not matched:
                remaining.append(token)
            i += 1

        for name in ("module", "config"):
            if name not in values:
                raise ValueError(
                    f"--{name} is required. Example: --module llama3 "
                    "--config llama3_debugmodel"
                )
        return values["module"], values["config"], remaining

    def _apply_tokens(self, config: object, tokens: list[str]) -> None:
        legacy_training_geometry: dict[str, str] = {}
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.startswith("activation-checkpoint:"):
                mode = token.partition(":")[2]
                if mode in {"none", "disabled"}:
                    config.activation_checkpoint = None  # type: ignore[attr-defined]
                else:
                    raise ValueError(
                        "Legacy activation-checkpoint subcommands currently "
                        f"support only :none, got {token!r}"
                    )
                i += 1
                continue
            if not token.startswith("--"):
                raise ValueError(f"Unexpected positional argument: {token!r}")

            option, inline, value = token.partition("=")
            key = option.removeprefix("--").replace("-", "_")
            if not inline and i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                value = tokens[i + 1]
                i += 1
            elif not inline:
                value = "true"

            if key == "override":
                trainer = ConfigLoader._trainer_config(config)
                imports = parse_cli_imports([value])
                trainer.override.imports.extend(imports)
                generator = getattr(config, "generator", None)
                generator_override = getattr(generator, "override", None)
                if generator_override is not None:
                    generator_override.imports.extend(imports)
            elif key == "comm_backend":
                ConfigLoader._trainer_config(config).comm.backend = value
            elif key in {"output_dir", "dump_folder"}:
                ConfigLoader._set_attr(config, "dump_folder", value)
            elif key == "resume_step":
                checkpointer = ConfigLoader._trainer_config(config).checkpointer
                if checkpointer is None:
                    raise ValueError("--resume-step requires checkpointing to be configured.")
                checkpointer.load_step = int(value)
            elif key in {"print_config", "debug.print_config"}:
                pprint.pprint(config)
            elif key in {
                "training.local_batch_size",
                "training.global_batch_size",
                "training.seq_len",
                "training.max_context_length",
            }:
                # Upstream #4121 replaced sequence-count geometry with token
                # budgets. Delay these aliases so argv ordering cannot alter
                # their meaning (all conversions depend on seq_len).
                legacy_training_geometry[key.removeprefix("training.")] = value
            else:
                self._apply_legacy_mutation(config, key, value)
            i += 1

        self._apply_legacy_training_geometry(config, legacy_training_geometry)
        post_init = getattr(config, "__post_init__", None)
        if callable(post_init):
            post_init()

    @staticmethod
    def _apply_legacy_training_geometry(
        config: object, values: dict[str, str]
    ) -> None:
        if not values:
            return

        training = LegacyConfigLoader._get(config, "training")
        old_seq_len = training.max_context_length
        raw_seq_len = values.get("seq_len", values.get("max_context_length"))
        seq_len = int(raw_seq_len) if raw_seq_len is not None else old_seq_len
        if seq_len <= 0:
            raise ValueError("training.seq_len must be greater than 0")

        if "local_batch_size" in values:
            local_batch_size = int(values["local_batch_size"])
            if local_batch_size <= 0:
                raise ValueError("training.local_batch_size must be greater than 0")
        elif "seq_len" in values:
            local_batch_size = (
                training.num_tokens_per_microbatch_per_dp_rank // old_seq_len
            )
        else:
            local_batch_size = None

        training.max_context_length = seq_len
        if local_batch_size is not None:
            training.num_tokens_per_microbatch_per_dp_rank = local_batch_size * seq_len

        if "global_batch_size" in values:
            global_batch_size = int(values["global_batch_size"])
            if global_batch_size <= 0:
                raise ValueError("training.global_batch_size must be greater than 0")
            training.num_tokens_per_train_step = global_batch_size * seq_len

        # The old seq_len option governed both trainer geometry and model RoPE
        # capacity. Preserve that single-knob behavior under the new schema.
        model = LegacyConfigLoader._get(config, "model")
        if hasattr(model, "max_context_length"):
            model.max_context_length = seq_len
        for layer in getattr(model, "layers", ()):  # AGPT dense and MoE
            attention = getattr(layer, "attention", None)
            rope = getattr(attention, "rope", None)
            if rope is not None and hasattr(rope, "max_context_length"):
                rope.max_context_length = seq_len

    @staticmethod
    def _apply_legacy_mutation(config: object, key: str, raw_value: str) -> None:
        key = key.replace("checkpoint.", "checkpointer.", 1)
        key = key.replace("lr_scheduler.", "optim.lr_scheduler.", 1)
        if key in {"checkpoint_enable", "checkpointer.enable"}:
            enabled = LegacyConfigLoader._coerce(raw_value, True)
            if enabled and getattr(config, "checkpointer", None) is None:
                config.checkpointer = CheckpointManager.Config()  # type: ignore[attr-defined]
            elif not enabled:
                config.checkpointer = None  # type: ignore[attr-defined]
            return
        if key in {
            "checkpoint_no_enable",
            "checkpointer.no_enable",
            "no_checkpoint.enable",
        }:
            config.checkpointer = None  # type: ignore[attr-defined]
            return
        if key == "checkpoint_create_seed_checkpoint":
            config.create_seed_checkpoint = True  # type: ignore[attr-defined]
            if getattr(config, "checkpointer", None) is None:
                config.checkpointer = CheckpointManager.Config()  # type: ignore[attr-defined]
            return
        if key in {"compile.enable", "compile_enable"}:
            if getattr(config, "compile", None) is None:
                config.compile = CompileConfig()  # type: ignore[attr-defined]
            return
        if key in {"compile.no_enable", "compile_no_enable", "no_compile.enable"}:
            config.compile = None  # type: ignore[attr-defined]
            return

        path = key.split(".")
        negate = path[-1].startswith("no_")
        if negate:
            path[-1] = path[-1].removeprefix("no_")
        parent, leaf = LegacyConfigLoader._resolve_parent(config, path)
        current = LegacyConfigLoader._get(parent, leaf)
        value = LegacyConfigLoader._coerce(raw_value, current)
        LegacyConfigLoader._set(parent, leaf, not value if negate else value)
        post_init = getattr(parent, "__post_init__", None)
        if callable(post_init):
            post_init()

    @staticmethod
    def _resolve_parent(root: object, path: Sequence[str]) -> tuple[object, str]:
        if not path:
            raise ValueError("Empty config option")
        current = root
        for part in path[:-1]:
            current = LegacyConfigLoader._get(current, part)
            if current is None:
                raise ValueError(f"Cannot traverse unset config field {part!r}")
        return current, path[-1]

    @staticmethod
    def _get(parent: object, key: str) -> Any:
        if isinstance(parent, (list, tuple)):
            try:
                return parent[int(key)]
            except (ValueError, IndexError) as error:
                raise ValueError(f"Invalid config list index {key!r}") from error
        if isinstance(parent, dict):
            if key not in parent:
                raise ValueError(f"Unknown config key {key!r}")
            return parent[key]
        if not hasattr(parent, key):
            raise ValueError(
                f"Unknown config option {key!r} on {type(parent).__qualname__}"
            )
        return getattr(parent, key)

    @staticmethod
    def _set(parent: object, key: str, value: Any) -> None:
        if isinstance(parent, list):
            parent[int(key)] = value
        elif isinstance(parent, dict):
            parent[key] = value
        elif isinstance(parent, tuple):
            raise ValueError("Tuple-valued config fields cannot be mutated by index")
        else:
            setattr(parent, key, value)

    @staticmethod
    def _coerce(raw: str, current: Any) -> Any:
        if isinstance(current, bool):
            lowered = raw.lower()
            if lowered not in {"true", "false"}:
                raise ValueError(f"Expected boolean value, got {raw!r}")
            return lowered == "true"
        if isinstance(current, int) and not isinstance(current, bool):
            return int(raw)
        if isinstance(current, float):
            return float(raw)
        if isinstance(current, tuple):
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError(f"Expected JSON list for tuple field, got {raw!r}")
            return tuple(parsed)
        if isinstance(current, list):
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError(f"Expected JSON list, got {raw!r}")
            return parsed
        if current is None:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        return raw
