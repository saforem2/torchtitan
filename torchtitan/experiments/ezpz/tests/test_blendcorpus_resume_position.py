# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A restarted BlendCorpus loader must continue the data stream, not replay it.

The 20B production chains restarted from checkpoints dozens of times while the
pinned clone never incremented ``_consumed_samples``, so every checkpoint stored
``dataloader.consumed_samples == 0`` and every restart re-read the stream from
sample 0. These tests run the REAL ``BlendCorpusDataLoader`` iteration and
state-dict code against a fake sampler that copies the semantics of
``blendcorpus.data.data_samplers.MegatronPretrainingSampler`` (a sequential
index stream starting at ``consumed_samples``, one batch of
``micro_batch_size * dp`` indices per yield). CPU only; no corpus, no
blendcorpus package.
"""

from types import SimpleNamespace

import pytest
import torch

from torchtitan.experiments.ezpz.blendcorpus.blendcorpus_builder import (
    BlendCorpusDataLoader,
)

MBS = 2
DP = 3
PER_YIELD = MBS * DP  # global samples consumed per yielded micro-batch
SEQ = 4


def _sampler_stream(consumed_samples, total_samples=10_000):
    """Rank-0 view of MegatronPretrainingSampler: rows hold their sample idx."""
    batch = []
    for idx in range(consumed_samples, total_samples):
        batch.append(idx)
        if len(batch) == PER_YIELD:
            rows = batch[:MBS]  # data_parallel_rank == 0
            yield {"text": torch.tensor([[r] * (SEQ + 1) for r in rows])}
            batch = []


def _make_loader(resume_consumed_samples=None, global_batch_size=PER_YIELD):
    loader = object.__new__(BlendCorpusDataLoader)
    loader._delegate = None
    loader._emit_positions = False
    loader._eod_token_id = None
    loader._consumed_samples = 0
    loader._samples_per_yield = PER_YIELD
    loader._resume_consumed_samples = resume_consumed_samples
    loader._served_ds = object()
    loader._bc_cfg = SimpleNamespace(global_batch_size=global_batch_size)
    loader._build_pretraining_data_loader = lambda ds, consumed, cfg: _sampler_stream(
        consumed
    )
    loader._loader = _sampler_stream(0)
    return loader


def _first_sample_ids(loader, num_batches):
    """Sample index carried by the first token of each yielded batch."""
    it = iter(loader)
    ids = []
    for _ in range(num_batches):
        ids.append(next(it)["input"][0].item())
    return ids


class TestResumeContinuesStream:
    def test_restart_reads_next_samples_not_sample_zero(self):
        original = _make_loader()
        full = _first_sample_ids(original, 8)
        # checkpoint after 5 batches, as the trainer would take it
        before = _make_loader()
        _first_sample_ids(before, 5)
        saved = before.state_dict()
        assert saved == {"consumed_samples": 5 * PER_YIELD}

        restarted = _make_loader()
        restarted.load_state_dict(saved)
        assert _first_sample_ids(restarted, 3) == full[5:8]
        assert full[5] != full[0], "fixture must distinguish replay from resume"

    def test_zero_counter_replays_from_sample_zero(self):
        """The pre-fix failure mode, pinned so the contrast stays visible."""
        full = _first_sample_ids(_make_loader(), 8)
        restarted = _make_loader()
        restarted.load_state_dict({"consumed_samples": 0})
        assert _first_sample_ids(restarted, 3) == full[0:3]

    def test_counter_follows_sampler_with_gradient_accumulation(self):
        """Each yield is one microbatch, so +micro*dp, not +global_batch_size."""
        loader = _make_loader(global_batch_size=4 * PER_YIELD)  # GAS = 4
        _first_sample_ids(loader, 8)  # 2 optimizer steps of 4 microbatches
        assert loader.state_dict() == {"consumed_samples": 8 * PER_YIELD}


class TestLegacyCheckpointOverride:
    def test_override_applies_when_checkpoint_counter_is_zero(self):
        full = _first_sample_ids(_make_loader(), 8)
        loader = _make_loader(resume_consumed_samples=6 * PER_YIELD)
        loader.load_state_dict({"consumed_samples": 0})
        assert loader.state_dict() == {"consumed_samples": 6 * PER_YIELD}
        assert _first_sample_ids(loader, 2) == full[6:8]
        assert loader.state_dict() == {"consumed_samples": 8 * PER_YIELD}

    def test_nonzero_checkpoint_counter_wins_over_override(self):
        """Auto-retry re-passes the same CLI args; newer checkpoints must win."""
        full = _first_sample_ids(_make_loader(), 12)
        loader = _make_loader(resume_consumed_samples=6 * PER_YIELD)
        loader.load_state_dict({"consumed_samples": 10 * PER_YIELD})
        assert _first_sample_ids(loader, 2) == full[10:12]

    def test_no_override_leaves_zero_checkpoint_at_zero(self):
        loader = _make_loader()
        loader.load_state_dict({"consumed_samples": 0})
        assert loader.state_dict() == {"consumed_samples": 0}

    @pytest.mark.parametrize("bad", [1, PER_YIELD + 1, 2 * PER_YIELD - 1])
    def test_misaligned_resume_position_is_rejected(self, bad):
        loader = _make_loader(resume_consumed_samples=bad)
        with pytest.raises(ValueError, match="not a multiple"):
            loader.load_state_dict({"consumed_samples": 0})


def test_env_default_for_resume_position(monkeypatch):
    """The production launcher cannot pass a flag, so the env var supplies the default."""
    from torchtitan.experiments.ezpz.blendcorpus import blendcorpus_builder as b

    monkeypatch.delenv("BLENDCORPUS_RESUME_CONSUMED_SAMPLES", raising=False)
    assert b.BlendCorpusDataLoader.Config().resume_consumed_samples is None
    monkeypatch.setenv("BLENDCORPUS_RESUME_CONSUMED_SAMPLES", "")
    assert b.BlendCorpusDataLoader.Config().resume_consumed_samples is None
    monkeypatch.setenv("BLENDCORPUS_RESUME_CONSUMED_SAMPLES", "9523200")
    assert b.BlendCorpusDataLoader.Config().resume_consumed_samples == 9523200
    # an explicit value still wins over the environment
    assert (
        b.BlendCorpusDataLoader.Config(resume_consumed_samples=6144).resume_consumed_samples
        == 6144
    )
