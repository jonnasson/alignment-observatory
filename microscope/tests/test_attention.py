"""Tests for AttentionPattern class."""

import pytest
import numpy as np
from alignment_microscope import AttentionPattern, Microscope


class TestAttentionPattern:
    """Unit tests for AttentionPattern."""

    def test_create_pattern(self, sample_attention_pattern):
        """Test creating an attention pattern."""
        pattern = AttentionPattern(layer=0, pattern=sample_attention_pattern)
        assert pattern.layer == 0
        assert pattern.num_heads == 12
        assert pattern.seq_len == 10

    def test_get_head_pattern(self, sample_attention_pattern):
        """Test extracting single head pattern."""
        pattern = AttentionPattern(layer=0, pattern=sample_attention_pattern)
        head_pattern = pattern.head(head_idx=0)
        assert head_pattern.shape == (10, 10)  # [seq_q, seq_k]

    def test_entropy_computation(self, sample_attention_pattern):
        """Test attention entropy computation."""
        pattern = AttentionPattern(layer=0, pattern=sample_attention_pattern)
        entropy = pattern.entropy()
        assert entropy.shape == (1, 12, 10)  # [batch, heads, seq_q]
        assert np.all(entropy >= 0)  # Entropy is non-negative

    def test_entropy_uniform(self, uniform_attention):
        """Test entropy is high for uniform attention."""
        pattern = AttentionPattern(layer=0, pattern=uniform_attention)
        entropy = pattern.entropy()
        # Last position attends uniformly to all 10 positions -> high entropy
        # Entropy of uniform(10) = log(10) ~ 2.3
        assert entropy[0, 0, -1] > 2.0

    def test_entropy_focused(self, previous_token_attention):
        """Test entropy is low for focused attention."""
        pattern = AttentionPattern(layer=0, pattern=previous_token_attention)
        entropy = pattern.entropy()
        # Attention is deterministic -> entropy ~ 0
        assert np.all(entropy < 0.1)

    def test_top_attended(self, sample_attention_pattern):
        """Test finding top attended positions."""
        pattern = AttentionPattern(layer=0, pattern=sample_attention_pattern)
        top = pattern.top_attended(k=3)
        # Check result structure
        assert len(top) > 0

    def test_pattern_sums_to_one(self, sample_attention_pattern):
        """Test that attention patterns sum to 1 along key dimension."""
        pattern = AttentionPattern(layer=0, pattern=sample_attention_pattern)
        sums = pattern.pattern.sum(axis=-1)
        np.testing.assert_array_almost_equal(sums, np.ones_like(sums), decimal=5)


class TestHeadClassification:
    """Tests for attention head classification."""

    def test_classify_previous_token(self, previous_token_attention):
        """Test classification of previous-token heads."""
        scope = Microscope.for_llama(num_layers=12, num_heads=1, hidden_size=768)
        classifications = scope.classify_heads(previous_token_attention)
        assert classifications[0] == "previous_token"

    def test_classify_bos(self, bos_attention):
        """Test classification of BOS-attention heads."""
        scope = Microscope.for_llama(num_layers=12, num_heads=1, hidden_size=768)
        classifications = scope.classify_heads(bos_attention)
        assert classifications[0] == "bos"

    def test_classify_uniform(self, uniform_attention):
        """Test classification of uniform attention heads."""
        scope = Microscope.for_llama(num_layers=12, num_heads=1, hidden_size=768)
        classifications = scope.classify_heads(uniform_attention)
        # Uniform attention may be classified as "uniform" or "other" depending on thresholds
        # The key is that it's NOT previous_token or bos
        assert classifications[0] not in ["previous_token", "bos"]

    def test_classify_multiple_heads(self, sample_attention_pattern):
        """Test classification of multiple heads at once."""
        scope = Microscope.for_llama(num_layers=12, num_heads=12, hidden_size=768)
        classifications = scope.classify_heads(sample_attention_pattern)
        assert len(classifications) == 12
        assert all(
            c in ["previous_token", "bos", "uniform", "other"] for c in classifications
        )


class TestAttentionPatternEdgeCases:
    """Edge case tests for AttentionPattern."""

    def test_single_head(self):
        """Test pattern with single head."""
        pattern_data = np.ones((1, 1, 5, 5), dtype=np.float32) / 5
        pattern = AttentionPattern(layer=0, pattern=pattern_data)
        assert pattern.num_heads == 1

    def test_single_position(self):
        """Test pattern with single position."""
        pattern_data = np.ones((1, 4, 1, 1), dtype=np.float32)
        pattern = AttentionPattern(layer=0, pattern=pattern_data)
        assert pattern.seq_len == 1

    def test_large_batch(self):
        """Test pattern with large batch."""
        np.random.seed(42)
        raw = np.random.randn(32, 8, 100, 100).astype(np.float32)
        exp_x = np.exp(raw - raw.max(axis=-1, keepdims=True))
        pattern_data = exp_x / exp_x.sum(axis=-1, keepdims=True)
        pattern = AttentionPattern(layer=0, pattern=pattern_data)
        entropy = pattern.entropy()
        assert entropy.shape == (32, 8, 100)
