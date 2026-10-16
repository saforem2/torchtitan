# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ezpz tyro ``ConfigManager``.

Upstream replaced its tyro CLI with ``ConfigLoader`` (#4908), which accepts
only ``--module``, ``--config``, ``--override``, ``--output-dir``,
``--comm-backend``, ``--resume-step`` and ``--print-config``. Every ezpz
launcher, smoke and LR-finder script drives training through
``--section.field value`` flags, so the pre-removal ``ConfigManager`` lives here
(``torchtitan/config/manager.py`` at ``5112abcb6f``) with the changes the new
config schema needs:

- ``--module`` resolution: upstream deleted the ``_supported_models`` /
  ``_supported_experiments`` shorthand tables and moved recipes to
  ``torchtitan_recipes``. ``ezpz.<name>`` resolves under
  ``torchtitan.experiments.ezpz``; the core model names resolve under
  ``torchtitan_recipes``; fully qualified paths work as before.
- Optional components: upstream stopped annotating optional dataclass fields
  with ``tyro.conf.AvoidSubcommands`` (it has no tyro). They are applied
  globally here so ``--checkpointer.interval 7`` keeps working; the frozen
  ``--checkpoint.enable`` family and the ``activation-checkpoint:<policy>``
  subcommand tokens are translated before tyro runs.
- ``list[OverrideImport]`` has no tyro primitive; ``format_cli_imports`` was
  deleted upstream, so the inverse of ``parse_cli_imports`` is local.
"""

import importlib
import logging
import os
import sys
from typing import Any

import tyro

from torchtitan.observability.logging import init_logger

logger = logging.getLogger(__name__)


# --module shorthands: name -> importable module that holds the config
# functions. Anything not listed here is treated as a fully qualified module
# path (tried with ``.config_registry`` appended first, then as given).
_RECIPE_PACKAGES = (
    "torchtitan_recipes.models",
    "torchtitan_recipes.rl",
    "torchtitan_recipes",
)
_LEGACY_ACTIVATION_CHECKPOINT = {
    # tyro subcommand token -> activation_remat policy
    "none": "none",
    "full": "full",
    # Upstream deleted selective-op AC (#4893 migrated it to torch_remat). The
    # closest policy is the default remat policy, which saves every
    # model-declared region and recomputes everything else.
    "selective": "default",
}


def _format_cli_imports(imports: list) -> list[str]:
    """Inverse of ``torchtitan.config.override.parse_cli_imports``."""
    import json

    out: list[str] = []
    for entry in imports:
        if isinstance(entry, str):
            out.append(entry)
        else:
            target, kwargs = entry
            out.append(f"{target}={json.dumps(kwargs)}")
    return out


class ConfigManager:
    """Parse ``--module``/``--config`` and CLI overrides into a trainer config.

    Precedence: CLI args > config function defaults.
    """

    def __init__(self):
        self.register_tyro_rules(custom_registry)

    def parse_args(self, args: list[str] | None = None):
        if args is None:
            args = sys.argv[1:]
        loaded_config, args = self._load_config(args)
        args = self._migrate_legacy_args(loaded_config, args)
        config_cls = type(loaded_config)

        self.config = tyro.cli(
            config_cls,
            args=args,
            default=loaded_config,
            registry=custom_registry,
            config=(tyro.conf.AvoidSubcommands,),
        )

        self._validate_config()

        return self.config

    # ------------------------------------------------------------------
    # legacy flag translation
    # ------------------------------------------------------------------
    @staticmethod
    def _migrate_legacy_args(config: Any, args: list[str]) -> list[str]:
        """Map frozen flags onto the current optional config fields."""
        from torchtitan.components.checkpointer import CheckpointManager
        from torchtitan.distributed.activation_remat import (
            DefaultActivationRematConfig,
            FullActivationRematConfig,
        )
        from torchtitan.experiments.ezpz.compile import CompileConfig

        migrated: list[str] = []
        for arg in args:
            if arg.startswith("activation-checkpoint:"):
                policy = arg.split(":", 1)[1]
                if policy not in _LEGACY_ACTIVATION_CHECKPOINT:
                    raise ValueError(
                        f"Unsupported legacy activation-checkpoint policy "
                        f"{policy!r}; supported: "
                        f"{sorted(_LEGACY_ACTIVATION_CHECKPOINT)}"
                    )
                mapped = _LEGACY_ACTIVATION_CHECKPOINT[policy]
                if mapped == "none":
                    config.activation_remat = None
                elif mapped == "full":
                    config.activation_remat = FullActivationRematConfig()
                else:
                    config.activation_remat = DefaultActivationRematConfig()
                continue
            if arg in ("--checkpoint.enable", "--checkpoint-enable"):
                if getattr(config, "checkpointer", None) is None:
                    config.checkpointer = CheckpointManager.Config()
                continue
            if arg in (
                "--checkpoint.no-enable",
                "--no-checkpoint.enable",
                "--no-checkpoint-enable",
            ):
                config.checkpointer = None
                continue
            if arg == "--checkpoint.create-seed-checkpoint":
                config.create_seed_checkpoint = True
                if getattr(config, "checkpointer", None) is None:
                    config.checkpointer = CheckpointManager.Config()
                continue
            if arg in (
                "--checkpoint.no-create-seed-checkpoint",
                "--no-checkpoint.create-seed-checkpoint",
            ):
                config.create_seed_checkpoint = False
                continue
            if arg == "--compile.enable":
                if getattr(config, "compile", None) is None:
                    config.compile = CompileConfig()
                continue
            if arg in ("--compile.no-enable", "--no-compile.enable"):
                config.compile = None
                continue
            if arg.startswith("--checkpoint."):
                arg = "--checkpointer." + arg.removeprefix("--checkpoint.")
            if arg.startswith("--activation-checkpoint."):
                raise ValueError(
                    f"{arg}: activation checkpointing is configured by "
                    "'activation-checkpoint:{none,full,selective}' or "
                    "--activation-remat.* (selective-op AC no longer exists)."
                )
            migrated.append(arg)
        return migrated

    # ------------------------------------------------------------------
    # module/config resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_module(module_name: str):
        """Import the module holding config functions for ``module_name``."""
        candidates: list[str] = []
        if module_name.startswith("ezpz."):
            candidates.append(
                f"torchtitan.experiments.{module_name}.config_registry"
            )
        elif module_name.startswith("graph_trainer.") or module_name.startswith(
            "torchft."
        ):
            candidates.append(f"torchtitan_recipes.{module_name}")
        else:
            candidates.extend(f"{pkg}.{module_name}" for pkg in _RECIPE_PACKAGES)
            candidates.append(f"torchtitan.experiments.{module_name}.config_registry")
        candidates.extend([f"{module_name}.config_registry", module_name])

        last_error: ImportError | None = None
        for candidate in candidates:
            try:
                module = importlib.import_module(candidate)
            except ModuleNotFoundError as error:
                # Only skip when the candidate itself (or a parent package of
                # it) is what's missing; a missing dependency inside an
                # existing module must surface with its real name.
                missing = error.name or ""
                if candidate == missing or candidate.startswith(missing + "."):
                    last_error = error
                    continue
                raise
            return module, candidate
        raise ImportError(
            f"Cannot resolve config module {module_name!r}; tried {candidates}"
        ) from last_error

    def _load_config(self, args: list[str]) -> tuple[object, list[str]]:
        """Parse --module and --config from args and call the config function.

        Both --module and --config are required.
        Returns (loaded_config, filtered_args) with --module/--config stripped.
        """
        module_name = None
        config_name = None
        filtered_args = []

        i = 0
        while i < len(args):
            arg = args[i]

            if arg.startswith("--module="):
                module_name = arg.split("=", 1)[1]
            elif arg == "--module":
                if i + 1 < len(args):
                    module_name = args[i + 1]
                    i += 1
                else:
                    raise ValueError("--module requires a value")
            elif arg.startswith("--config="):
                config_name = arg.split("=", 1)[1]
            elif arg == "--config":
                if i + 1 < len(args):
                    config_name = args[i + 1]
                    i += 1
                else:
                    raise ValueError("--config requires a value")
            else:
                filtered_args.append(arg)

            i += 1

        if module_name is None:
            raise ValueError(
                "--module is required. Example: --module ezpz.agpt --config agpt_debugmodel"
            )
        if config_name is None:
            raise ValueError(
                "--config is required. Example: --module ezpz.agpt --config agpt_debugmodel"
            )

        module, module_path = self._resolve_module(module_name)

        config_fn = getattr(module, config_name, None)
        if config_fn is None or not callable(config_fn):
            available = [
                name
                for name in dir(module)
                if not name.startswith("_")
                and callable(getattr(module, name))
                and name[0].islower()
            ]
            raise ValueError(
                f"Config function '{config_name}' not found in {module_path}. "
                f"Available config functions: {available}"
            )

        loaded_config = config_fn()
        return loaded_config, filtered_args

    def _validate_config(self) -> None:
        assets_path = getattr(self.config, "hf_assets_path", None)
        if assets_path is None:
            return
        if not os.path.exists(assets_path):
            logger.warning(f"HF assets path {assets_path} does not exist!")
        elif assets_path.endswith("tokenizer.model"):
            raise ValueError(
                "hf_assets_path must name an assets directory, not the legacy "
                "tokenizer.model file. Download the Hugging Face assets "
                "(python scripts/download_hf_assets.py) and update the config."
            )

    @staticmethod
    def register_tyro_rules(registry: tyro.constructors.ConstructorRegistry) -> None:
        @registry.primitive_rule
        def list_str_rule(type_info: tyro.constructors.PrimitiveTypeInfo):
            """Support for comma separated string parsing"""
            if type_info.type != list[str]:
                return None
            return tyro.constructors.PrimitiveConstructorSpec(
                nargs=1,
                metavar="A,B,C,...",
                instance_from_str=lambda args: args[0].split(","),
                is_instance=lambda instance: all(isinstance(i, str) for i in instance),
                str_from_instance=lambda instance: [",".join(instance)],
            )

        @registry.struct_rule
        def opaque_struct_rule(info: tyro.constructors.StructTypeInfo):
            """Keep configs that tyro cannot introspect out of the CLI.

            Upstream dropped the ``Annotated[..., tyro.conf.Suppress]`` markers
            when it dropped tyro, so some config classes (``ShardingConfig`` ->
            ``spmd_types.SpmdType`` with an unresolvable ``ProcessGroup``
            forward reference, ``Callable``/pydantic fields, ...) now fail in
            ``typing.get_type_hints``. The whole model config is also opaque:
            no ezpz launcher addresses ``--model.*`` and its sharding/param-init
            subtree is not CLI-shaped. Opaque structs expose no flags and keep
            their default instance.
            """
            import dataclasses
            import typing

            from torchtitan.models.base.module import Module

            typ = info.type
            if not (isinstance(typ, type) and dataclasses.is_dataclass(typ)):
                return None
            opaque = issubclass(typ, Module.Config)
            if not opaque:
                try:
                    typing.get_type_hints(typ)
                except Exception:
                    opaque = True
            if not opaque:
                return None
            default = info.default
            if default is tyro.constructors.MISSING or (
                default is tyro.constructors.MISSING_NONPROP
            ):
                return None
            return tyro.constructors.StructConstructorSpec(
                instantiate=lambda: default, fields=()
            )

        @registry.primitive_rule
        def override_imports_rule(type_info: tyro.constructors.PrimitiveTypeInfo):
            """Parse ``--override.imports`` targets, each with optional kwargs.

            ``OverrideConfig.imports`` is a ``str | tuple`` union, which tyro
            cannot render as a CLI flag on its own. Tokens are space- or
            comma-separated ``module.function`` targets; attach kwargs to a
            target with ``target=<json>`` (see ``parse_cli_imports``).
            """
            from torchtitan.config.override import OverrideImport, parse_cli_imports

            if type_info.type != list[OverrideImport]:
                return None
            return tyro.constructors.PrimitiveConstructorSpec(
                nargs="*",
                metavar="TARGET[='{\"k\": v}'] ...",
                instance_from_str=lambda args: parse_cli_imports(args),
                is_instance=lambda instance: isinstance(instance, list)
                and all(
                    isinstance(i, str)
                    or (
                        isinstance(i, tuple)
                        and len(i) == 2
                        and isinstance(i[0], str)
                        and isinstance(i[1], dict)
                    )
                    for i in instance
                ),
                str_from_instance=lambda instance: _format_cli_imports(instance),
            )


# Initialize the custom registry for tyro
custom_registry = tyro.constructors.ConstructorRegistry()


if __name__ == "__main__":
    # Debug/inspect configuration parsing:
    #   python -m torchtitan.experiments.ezpz.config_manager \
    #       --module ezpz.agpt --config agpt_debugmodel [--help]
    init_logger()
    config = ConfigManager().parse_args()
    try:
        from rich import print as rprint
        from rich.pretty import Pretty

        rprint(Pretty(config))
    except ImportError:
        logger.info(config)
        logger.warning("rich is not installed, show the raw config")
