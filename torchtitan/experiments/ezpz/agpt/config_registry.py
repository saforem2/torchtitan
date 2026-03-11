import json
import os
from dataclasses import is_dataclass
from typing import Any

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import ActivationCheckpointConfig, CommConfig, TrainingConfig
from torchtitan.experiments.ezpz.blendcorpus.blendcorpus_builder import (
    BlendCorpusDataLoader,
)
from torchtitan.experiments.ezpz.blendcorpus.build_tokenizer import EZPZTokenizer
from torchtitan.experiments.ft.config.job_config import FaultTolerance
from torchtitan.experiments.ft.trainer import FaultTolerantTrainer

from . import model_registry

TT_CONFIG_JSON_ENV = "TT_CONFIG_JSON"


def _base_config(flavor: str) -> FaultTolerantTrainer.Config:
    return FaultTolerantTrainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry(flavor),
        tokenizer=EZPZTokenizer.Config(backend="hf"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=200,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10000,
        ),
        dataloader=BlendCorpusDataLoader.Config(dataset="c4_test"),
        metrics=MetricsProcessor.Config(log_freq=10),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
        comm=CommConfig(train_timeout_seconds=100),
        fault_tolerance=FaultTolerance(enable=False),
        validator=Validator.Config(enable=False),
    )


def _load_json_overrides() -> dict[str, Any]:
    path = os.environ.get(TT_CONFIG_JSON_ENV, "").strip()
    if not path:
        raise ValueError(
            f"{TT_CONFIG_JSON_ENV} must point to a JSON file when using *_from_json configs."
        )

    with open(path, encoding="utf-8") as f:
        overrides = json.load(f)

    if not isinstance(overrides, dict):
        raise ValueError(
            f"Expected top-level JSON object in {path!r}, got {type(overrides).__name__}."
        )

    return overrides


def _apply_config_overrides(
    target: Any,
    overrides: dict[str, Any],
    path: str = "",
) -> None:
    for key, value in overrides.items():
        if not hasattr(target, key):
            raise KeyError(f"Unknown config field {key!r} at path {path or '<root>'}.")

        current_value = getattr(target, key)
        field_path = f"{path}.{key}" if path else key

        if isinstance(value, dict):
            if not is_dataclass(current_value):
                raise TypeError(
                    f"Expected dataclass at {field_path!r} for nested override, "
                    f"got {type(current_value).__name__}."
                )
            _apply_config_overrides(current_value, value, field_path)
            continue

        setattr(target, key, value)


def _config_from_json(flavor: str) -> FaultTolerantTrainer.Config:
    cfg = _base_config(flavor)
    _apply_config_overrides(cfg, _load_json_overrides())
    return cfg


def ezpz_agpt_debugmodel() -> FaultTolerantTrainer.Config:
    return _base_config("debugmodel")


def ezpz_agpt_2b() -> FaultTolerantTrainer.Config:
    return _base_config("2b")


def ezpz_agpt_7b() -> FaultTolerantTrainer.Config:
    return _base_config("7b")


def ezpz_agpt_8b() -> FaultTolerantTrainer.Config:
    return _base_config("8B")


def ezpz_agpt_debugmodel_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json("debugmodel")


def ezpz_agpt_2b_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json("2b")


def ezpz_agpt_7b_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json("7b")


def ezpz_agpt_8b_from_json() -> FaultTolerantTrainer.Config:
    return _config_from_json("8B")


def ezpz_agpt_blendcorpus_debugmodel() -> FaultTolerantTrainer.Config:
    cfg = _base_config("debugmodel")
    cfg.dataloader.dataset = "blendcorpus"
    if isinstance(cfg.tokenizer, EZPZTokenizer.Config):
        cfg.tokenizer.backend = "sptoken"
    return cfg
