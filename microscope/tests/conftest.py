"""Shared pytest fixtures for alignment_microscope tests."""

import pytest
import numpy as np
from typing import Dict, List, Optional

# Try importing transformers for integration tests
try:
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from alignment_microscope import Microscope, ActivationTrace, AttentionPattern, Circuit


@pytest.fixture
def sample_activations() -> Dict[str, np.ndarray]:
    """Generate sample activation data."""
    np.random.seed(42)
    return {
        "0_residual": np.random.randn(1, 10, 768).astype(np.float32),
        "0_attn_out": np.random.randn(1, 10, 768).astype(np.float32),
        "0_mlp_out": np.random.randn(1, 10, 768).astype(np.float32),
        "1_residual": np.random.randn(1, 10, 768).astype(np.float32),
        "1_attn_out": np.random.randn(1, 10, 768).astype(np.float32),
        "1_mlp_out": np.random.randn(1, 10, 768).astype(np.float32),
    }


@pytest.fixture
def sample_attention_pattern() -> np.ndarray:
    """Generate sample attention pattern (softmax over keys)."""
    np.random.seed(42)
    raw = np.random.randn(1, 12, 10, 10).astype(np.float32)
    # Apply softmax over last dimension
    exp_x = np.exp(raw - raw.max(axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


@pytest.fixture
def previous_token_attention() -> np.ndarray:
    """Generate attention pattern that attends to previous token."""
    pattern = np.zeros((1, 1, 10, 10), dtype=np.float32)
    for i in range(10):
        if i > 0:
            pattern[0, 0, i, i - 1] = 1.0
        else:
            pattern[0, 0, i, i] = 1.0
    return pattern


@pytest.fixture
def bos_attention() -> np.ndarray:
    """Generate attention pattern that attends to first token (BOS)."""
    pattern = np.zeros((1, 1, 10, 10), dtype=np.float32)
    pattern[0, 0, :, 0] = 1.0
    return pattern


@pytest.fixture
def uniform_attention() -> np.ndarray:
    """Generate uniform attention pattern (causal mask applied)."""
    pattern = np.zeros((1, 1, 10, 10), dtype=np.float32)
    for i in range(10):
        pattern[0, 0, i, : i + 1] = 1.0 / (i + 1)
    return pattern


@pytest.fixture
def sample_trace(sample_activations, sample_attention_pattern) -> ActivationTrace:
    """Create a sample activation trace."""
    trace = ActivationTrace()
    trace._activations = sample_activations
    trace._attention_patterns = {0: sample_attention_pattern}
    trace._input_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return trace


@pytest.fixture
def microscope_gpt2() -> Microscope:
    """Create a Microscope configured for GPT-2 small."""
    return Microscope(
        architecture="gpt2", num_layers=12, num_heads=12, hidden_size=768
    )


@pytest.fixture
def microscope_llama() -> Microscope:
    """Create a Microscope configured for Llama-style model."""
    return Microscope.for_llama(num_layers=32, num_heads=32, hidden_size=4096)


@pytest.fixture
def sample_circuit() -> Circuit:
    """Create a sample circuit with known structure."""
    circuit = Circuit(
        name="Test Circuit",
        description="A test circuit for unit testing",
        behavior="test_behavior",
    )
    circuit.add_node(0, "attention", head=3)
    circuit.add_node(1, "attention", head=5)
    circuit.add_node(2, "mlp")
    circuit.add_edge((0, "attention", 3), (1, "attention", 5), 0.8)
    circuit.add_edge((1, "attention", 5), (2, "mlp", None), 0.6)
    return circuit


# Integration test fixtures
@pytest.fixture(scope="module")
def gpt2_model():
    """Load GPT-2 small model (cached for session)."""
    if not HAS_TRANSFORMERS:
        pytest.skip("transformers not installed")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return model


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    """Load GPT-2 tokenizer (cached for session)."""
    if not HAS_TRANSFORMERS:
        pytest.skip("transformers not installed")
    return GPT2Tokenizer.from_pretrained("gpt2")


@pytest.fixture
def gpt2_input(gpt2_tokenizer):
    """Generate sample input for GPT-2."""
    text = "The quick brown fox jumps over the lazy"
    return gpt2_tokenizer(text, return_tensors="pt")


def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (requires model download)"
    )
    config.addinivalue_line("markers", "slow: marks tests as slow")
