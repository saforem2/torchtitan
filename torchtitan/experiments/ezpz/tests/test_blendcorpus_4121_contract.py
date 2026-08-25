"""BlendCorpus must satisfy core's post-#4121 dataloader/validator contract.

Four separate #4121 renames broke the blendcorpus path after the 80th sync,
and each one surfaced only at job-launch time on a cluster, minutes into an
allocation:

  1. dataloader batch units          (tokens vs sequences)
  2. attention layout                ([B, L] vs flat [T]) -- see
     test_blendcorpus_fold_batch_dim.py
  3. Validator.__init__ kwarg        local_batch_size -> num_tokens_per_batch
  4. dataloader Config field         infinite         -> repeat

They share a signature: they only fire on a code path the 2N sync smokes did
not exercise (#3 and #4 both need --validator.enable, which every smoke had
off). These tests pull that feedback back to CPU so the next rename is caught
before it costs an allocation.

Deliberately checks the CONTRACT -- the argument and field names core reaches
for -- rather than any training behaviour, so it runs with no GPU, no corpus,
and without importing the blendcorpus package.
"""

import dataclasses
import inspect

import pytest


def _blendcorpus_cls():
    """Import the loader class, or skip when torch/deps are unavailable."""
    try:
        from torchtitan.experiments.ezpz.blendcorpus.blendcorpus_builder import (
            BlendCorpusDataLoader,
        )
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"blendcorpus_builder not importable here: {exc}")
    return BlendCorpusDataLoader


class TestConfigAcceptsRepeat:
    """Core does replace(config.dataloader, repeat=...) on OUR config."""

    def test_config_has_repeat_field(self):
        """`dataclasses.replace` raises TypeError on an unknown field.

        components/validate.py:132 runs
            replace(config.dataloader, repeat=config.steps != -1)
        unconditionally, so a missing `repeat` makes the validator
        unbuildable on this path -- an abort in config.build(), before
        step 1. That is exactly how gate job 7558682 died.
        """
        cfg_cls = _blendcorpus_cls().Config
        names = {f.name for f in dataclasses.fields(cfg_cls)}
        assert "repeat" in names, (
            "core sets `repeat` via dataclasses.replace; without the field "
            "the blendcorpus validator path cannot be constructed"
        )

    @pytest.mark.parametrize("repeat", [True, False])
    def test_replace_repeat_reconciles_with_infinite(self, repeat):
        """`repeat` and `infinite` are one concept; they must not diverge."""
        cfg_cls = _blendcorpus_cls().Config
        cfg = dataclasses.replace(cfg_cls(), repeat=repeat)

        assert cfg.repeat == repeat
        assert cfg.infinite == repeat, (
            "the rest of the loader reads `infinite`; if replace(repeat=False) "
            "leaves infinite=True the validator silently gets an endless "
            "dataloader and never terminates its pass"
        )

    def test_setting_infinite_alone_still_works(self):
        """The pre-#4121 spelling must keep working for existing configs."""
        cfg_cls = _blendcorpus_cls().Config
        cfg = cfg_cls(infinite=False)
        assert cfg.infinite is False
        assert cfg.repeat is False


class TestInitAcceptsBothSpellings:
    """Two callers reach __init__ with different kwarg names."""

    @staticmethod
    def _params():
        return inspect.signature(_blendcorpus_cls().__init__).parameters

    @pytest.mark.parametrize(
        "name",
        ["seq_len", "local_batch_size", "max_context_length", "num_tokens_per_batch"],
    )
    def test_accepts_kwarg(self, name):
        assert name in self._params(), (
            f"{name} missing: ezpz's validator sends the old pair and core's "
            f"Validator.validate() sends only the new pair, so both must bind"
        )

    @pytest.mark.parametrize(
        "name",
        ["seq_len", "local_batch_size", "max_context_length", "num_tokens_per_batch"],
    )
    def test_kwarg_is_optional(self, name):
        """Neither caller sends all four, so none may be required."""
        p = self._params()[name]
        assert p.default is not inspect.Parameter.empty, (
            f"{name} has no default; core's Validator.validate() passes only "
            f"max_context_length/num_tokens_per_batch and would TypeError"
        )


class TestCoreCallPathBinds:
    """The exact kwarg sets each caller uses must bind against the signature."""

    CORE_VALIDATE = {
        "dp_world_size": 1,
        "dp_rank": 0,
        "tokenizer": None,
        "max_context_length": 8192,
        "num_tokens_per_batch": 8192,
    }
    EZPZ_VALIDATOR = {
        "dp_world_size": 1,
        "dp_rank": 0,
        "tokenizer": None,
        "seq_len": 8192,
        "local_batch_size": 1,
        "max_context_length": 8192,
        "num_tokens_per_batch": 8192,
        "parallel_dims": None,
    }

    @pytest.mark.parametrize(
        "kwargs, label",
        [
            (CORE_VALIDATE, "core Validator.validate()"),
            (EZPZ_VALIDATOR, "ezpz _get_validation_dataloader()"),
        ],
    )
    def test_binds(self, kwargs, label):
        """Bind arguments without running __init__ (which needs a corpus)."""
        sig = inspect.signature(_blendcorpus_cls().__init__)
        try:
            sig.bind(self=None, config=None, **kwargs)
        except TypeError as exc:
            pytest.fail(f"{label} does not bind: {exc}")
