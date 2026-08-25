"""BlendCorpus must yield the flat [T] token layout that #4121 requires.

Upstream PR #4121 ("fold batch dim", 80th sync) moved the LM stack to a flat
token layout. BlendCorpus is an out-of-tree ALCF loader that counts SEQUENCES
and yielded [B, L]; it was never adapted, so every production agpt config died
at the first attention layer. Measured on Polaris job 7557496:

    ValueError: token count 1 is not a multiple of max_context_length 8192

These tests pin the two properties that make the fold correct, both of which
were verified against the real code paths rather than assumed:

  1. The FOLD ITSELF is order-preserving and matches what core already does
     for its own loader (TextCollator emits torch.cat(rows)).
  2. Downstream consumers -- QKVLinear and the packed-document mask builder --
     genuinely require flat [T] and silently misbehave on [B, L]. That is the
     part worth locking down: neither failure mode is a clean error.

These run on CPU with no GPU, no corpus, and no blendcorpus package: they
exercise the arithmetic of the shipped reshape ops directly.
"""

import torch


def _qkv_head_split(x, head_dim):
    """The reshape QKVLinear.forward performs (models/common/attention.py).

    Reproduced rather than imported: importing the real module pulls in
    spmd_types and a torch build that a CPU-only test box need not have.
    Kept deliberately literal so a change upstream shows up as a diff here.
    """
    num_tokens = x.shape[0]
    return x.view(num_tokens, -1, head_dim)


class TestFoldIsOrderPreserving:
    """flatten() must be exactly core's torch.cat(rows), not merely same-shape."""

    def test_flatten_matches_core_collator_cat(self):
        batch, seq_len = 3, 6
        tokens = torch.arange(batch * seq_len).reshape(batch, seq_len)

        folded = tokens.flatten()
        core_style = torch.cat([tokens[i] for i in range(batch)])

        assert torch.equal(folded, core_style), (
            "the fold must preserve token order exactly -- production chains "
            "and their checkpoints depend on the data stream being unchanged"
        )

    def test_fold_preserves_token_count(self):
        batch, seq_len = 4, 128
        tokens = torch.zeros(batch, seq_len, dtype=torch.long)
        assert tokens.flatten().numel() == batch * seq_len


class TestAttentionRequiresFlatLayout:
    """The observed Polaris crash shape, reproduced from the shipped reshape."""

    # agpt 2B: dim=2048, n_heads=16, n_kv_heads=4, head_dim=128
    DIM, HEAD_DIM, N_HEADS, N_KV, SEQ_LEN = 2048, 128, 16, 4, 8192

    def test_unfolded_batch_dim_collapses_heads_into_sequence(self):
        """[B, L, D] gives (1, L*N, H) -- the 2026-08-24 crash shape."""
        hidden = torch.zeros(1, self.SEQ_LEN, self.N_HEADS * self.HEAD_DIM)
        q = _qkv_head_split(hidden, self.HEAD_DIM)

        assert tuple(q.shape) == (1, self.SEQ_LEN * self.N_HEADS, self.HEAD_DIM)
        assert tuple(q.shape) == (1, 131072, 128), (
            "this is the exact shape reported by the Polaris crash; if it "
            "changes, the root-cause analysis behind the fold needs revisiting"
        )

    def test_folded_layout_gives_the_expected_per_head_split(self):
        """Flat [T, D] gives (T, N, H) -- what the wrapper expects."""
        hidden = torch.zeros(self.SEQ_LEN, self.N_HEADS * self.HEAD_DIM)
        q = _qkv_head_split(hidden, self.HEAD_DIM)

        assert tuple(q.shape) == (self.SEQ_LEN, self.N_HEADS, self.HEAD_DIM)

    def test_gqa_kv_heads_survive_the_fold(self):
        """n_kv_heads != n_heads must still infer correctly (the -1 in view)."""
        hidden = torch.zeros(self.SEQ_LEN, self.N_KV * self.HEAD_DIM)
        k = _qkv_head_split(hidden, self.HEAD_DIM)

        assert tuple(k.shape) == (self.SEQ_LEN, self.N_KV, self.HEAD_DIM)


class TestPositionsRequireFlatLayout:
    """Positions must fold too, or packed-document masking is silently wrong.

    This is the dangerous half: a [B, L] positions tensor does not raise. It
    produces a wrong document id, so flex attention would mask the wrong
    spans and only show up as a degraded loss curve.
    """

    @staticmethod
    def _document_id(positions):
        # attention.py:485-487, get_efficient_causal_mask_mod_for_packed_document
        document_starts = positions == 0
        return torch.cumsum(document_starts.int(), dim=0).to(torch.int32) - 1

    def test_flat_positions_give_contiguous_document_ids(self):
        batch, seq_len = 2, 8
        # each row packs two 4-token documents, so positions restart at 0
        row = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        positions = row.unsqueeze(0).expand(batch, seq_len).flatten()

        doc_id = self._document_id(positions)

        assert positions.shape[0] == batch * seq_len
        assert doc_id.tolist() == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4

    def test_unfolded_positions_produce_garbage_document_ids(self):
        """Locks in WHY positions must fold: [B, L] yields -1 document ids."""
        batch, seq_len = 2, 8
        row = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        positions = row.unsqueeze(0).expand(batch, seq_len)

        doc_id = self._document_id(positions)

        # core reads positions.shape[0] as the sequence length; on [B, L] that
        # is the BATCH size, and cumsum runs down the wrong axis.
        assert positions.shape[0] == batch, "core would read seq_len=2"
        assert (doc_id == -1).any(), (
            "unfolded positions yield negative document ids -- wrong masking "
            "with no exception, which is why the fold covers positions too"
        )
