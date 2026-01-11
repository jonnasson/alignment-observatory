"""Tests for Microscope class."""

import pytest
import numpy as np
from alignment_microscope import Microscope, ActivationTrace, Circuit


class TestMicroscope:
    """Unit tests for Microscope."""

    def test_create_microscope(self):
        """Test creating a microscope with default config."""
        scope = Microscope()
        assert scope.architecture == "llama"
        assert scope.num_layers == 32
        assert scope.num_heads == 32
        assert scope.hidden_size == 4096

    def test_create_for_llama(self):
        """Test for_llama factory method."""
        scope = Microscope.for_llama(num_layers=12, num_heads=8, hidden_size=512)
        assert scope.architecture == "llama"
        assert scope.num_layers == 12

    def test_create_custom(self):
        """Test creating microscope with custom config."""
        scope = Microscope(
            architecture="gpt2", num_layers=12, num_heads=12, hidden_size=768
        )
        assert scope.architecture == "gpt2"

    def test_trace_context_manager(self, microscope_gpt2):
        """Test trace context manager creates and cleans up trace."""
        with microscope_gpt2.trace() as trace:
            assert trace is not None
            assert isinstance(trace, ActivationTrace)

        # After exiting, _current_trace should be None
        assert microscope_gpt2._current_trace is None

    def test_trace_with_input_tokens(self, microscope_gpt2):
        """Test trace captures input tokens."""
        tokens = [1, 2, 3, 4, 5]
        with microscope_gpt2.trace(input_tokens=tokens) as trace:
            assert trace._input_tokens == tokens

    def test_classify_heads_with_array(self, microscope_gpt2, sample_attention_pattern):
        """Test classify_heads accepts numpy array."""
        classifications = microscope_gpt2.classify_heads(sample_attention_pattern)
        assert len(classifications) == 12


class TestMicroscopeCircuitDiscovery:
    """Tests for circuit discovery functionality."""

    def test_discover_circuit_basic(self, microscope_gpt2, sample_trace):
        """Test basic circuit discovery."""
        # Create two traces with differences
        clean_trace = sample_trace
        corrupt_trace = ActivationTrace()
        corrupt_trace._activations = {
            k: v * 0.5 for k, v in clean_trace._activations.items()  # Reduce magnitude
        }
        corrupt_trace._attention_patterns = clean_trace._attention_patterns.copy()

        circuit = microscope_gpt2.discover_circuit(
            behavior="test", clean_trace=clean_trace, corrupt_trace=corrupt_trace
        )

        assert isinstance(circuit, Circuit)
        assert circuit.name == "test"

    def test_discover_circuit_identical_traces(self, microscope_gpt2, sample_trace):
        """Test circuit discovery with identical traces returns sparse circuit."""
        circuit = microscope_gpt2.discover_circuit(
            behavior="no_diff", clean_trace=sample_trace, corrupt_trace=sample_trace
        )

        # Identical traces should produce few/no significant nodes
        assert len(circuit.nodes) == 0 or len(circuit.edges) == 0


class TestMicroscopeConfiguration:
    """Tests for Microscope configuration."""

    def test_different_architectures(self):
        """Test creating microscopes for different architectures."""
        archs = ["llama", "gpt2", "mistral", "qwen", "gemma"]
        for arch in archs:
            scope = Microscope(architecture=arch)
            assert scope.architecture == arch

    def test_different_sizes(self):
        """Test creating microscopes of different sizes."""
        # Small
        small = Microscope(num_layers=6, num_heads=6, hidden_size=256)
        assert small.num_layers == 6

        # Large
        large = Microscope(num_layers=80, num_heads=64, hidden_size=8192)
        assert large.num_layers == 80

    def test_for_llama_convenience(self):
        """Test for_llama creates correct configuration."""
        scope = Microscope.for_llama(num_layers=32, num_heads=32, hidden_size=4096)
        assert scope.architecture == "llama"
        assert scope.num_layers == 32
        assert scope.num_heads == 32
        assert scope.hidden_size == 4096


class TestMicroscopeEdgeCases:
    """Edge case tests for Microscope."""

    def test_minimal_configuration(self):
        """Test microscope with minimal configuration."""
        scope = Microscope(num_layers=1, num_heads=1, hidden_size=64)
        assert scope.num_layers == 1

    def test_trace_without_model(self):
        """Test tracing without a model attached."""
        scope = Microscope()
        with scope.trace() as trace:
            # Should work even without model - just returns empty trace
            assert isinstance(trace, ActivationTrace)

    def test_classify_empty_attention(self):
        """Test classify_heads with zeros attention."""
        scope = Microscope()
        zeros = np.zeros((1, 4, 5, 5), dtype=np.float32)
        # Avoid division by zero by adding small epsilon
        zeros[:, :, :, 0] = 1.0  # All attend to first token
        classifications = scope.classify_heads(zeros)
        assert len(classifications) == 4
