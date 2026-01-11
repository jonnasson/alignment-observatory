"""Tests for ActivationTrace class."""

import pytest
import numpy as np
from alignment_microscope import ActivationTrace


class TestActivationTrace:
    """Unit tests for ActivationTrace."""

    def test_create_empty_trace(self):
        """Test creating an empty trace."""
        trace = ActivationTrace()
        assert trace.layers == []

    def test_add_activation(self, sample_activations):
        """Test adding activations to trace."""
        trace = ActivationTrace()
        trace._activations = sample_activations

        assert 0 in trace.layers
        assert 1 in trace.layers

    def test_get_activation(self, sample_trace):
        """Test retrieving activations."""
        residual = sample_trace.get(0, "residual")
        assert residual is not None
        assert residual.shape == (1, 10, 768)

    def test_get_nonexistent_activation(self, sample_trace):
        """Test retrieving non-existent activation returns None."""
        result = sample_trace.get(99, "residual")
        assert result is None

    def test_residual_shortcut(self, sample_trace):
        """Test residual() convenience method."""
        direct = sample_trace.get(0, "residual")
        shortcut = sample_trace.residual(0)
        np.testing.assert_array_equal(direct, shortcut)

    def test_attention_out_shortcut(self, sample_trace):
        """Test attention_out() convenience method."""
        direct = sample_trace.get(0, "attn_out")
        shortcut = sample_trace.attention_out(0)
        np.testing.assert_array_equal(direct, shortcut)

    def test_mlp_out_shortcut(self, sample_trace):
        """Test mlp_out() convenience method."""
        direct = sample_trace.get(0, "mlp_out")
        shortcut = sample_trace.mlp_out(0)
        np.testing.assert_array_equal(direct, shortcut)

    def test_attention_pattern(self, sample_trace):
        """Test retrieving attention patterns."""
        pattern = sample_trace.attention(0)
        assert pattern is not None
        assert pattern.shape == (1, 12, 10, 10)

    def test_token_norms(self, sample_trace):
        """Test computing token norms."""
        norms = sample_trace.token_norms(0, "residual")
        assert norms is not None
        assert norms.shape == (1, 10)  # [batch, seq]
        assert np.all(norms >= 0)  # Norms are non-negative

    def test_token_norms_nonexistent(self, sample_trace):
        """Test token_norms for non-existent layer."""
        norms = sample_trace.token_norms(99, "residual")
        assert norms is None

    def test_layers_property(self, sample_trace):
        """Test layers property returns sorted unique layers."""
        layers = sample_trace.layers
        assert layers == sorted(layers)
        assert len(layers) == len(set(layers))

    def test_to_dict(self, sample_trace):
        """Test exporting trace to dictionary."""
        d = sample_trace.to_dict()
        assert "activations" in d
        assert "attention_patterns" in d
        assert "input_tokens" in d
        assert isinstance(d["activations"], dict)


class TestActivationTraceEdgeCases:
    """Edge case tests for ActivationTrace."""

    def test_empty_trace_layers(self):
        """Test empty trace returns empty layers."""
        trace = ActivationTrace()
        assert trace.layers == []

    def test_single_layer(self):
        """Test trace with single layer."""
        trace = ActivationTrace()
        trace._activations = {"0_residual": np.ones((1, 5, 128), dtype=np.float32)}
        assert trace.layers == [0]

    def test_multiple_components_same_layer(self):
        """Test multiple components at same layer."""
        trace = ActivationTrace()
        trace._activations = {
            "0_residual": np.ones((1, 5, 128), dtype=np.float32),
            "0_attn_out": np.ones((1, 5, 128), dtype=np.float32),
            "0_mlp_out": np.ones((1, 5, 128), dtype=np.float32),
        }
        # Should only have one unique layer
        assert trace.layers == [0]

    def test_large_activation(self):
        """Test with larger activation shapes."""
        trace = ActivationTrace()
        trace._activations = {
            "0_residual": np.random.randn(8, 512, 4096).astype(np.float32)
        }
        norms = trace.token_norms(0, "residual")
        assert norms.shape == (8, 512)
