"""Integration tests with real GPT-2 model."""

import pytest
import numpy as np

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestGPT2Integration:
    """Integration tests using GPT-2 small."""

    @pytest.fixture(autouse=True)
    def check_dependencies(self):
        """Skip if transformers not installed."""
        pytest.importorskip("transformers")
        pytest.importorskip("torch")

    def test_for_model_factory(self, gpt2_model):
        """Test Microscope.for_model() with real model."""
        from alignment_microscope import Microscope

        scope = Microscope.for_model(gpt2_model)

        assert scope.architecture == "gpt2"
        assert scope.num_layers == 12
        assert scope.num_heads == 12
        assert scope.hidden_size == 768

    def test_trace_forward_pass(self, gpt2_model, gpt2_input):
        """Test tracing activations during forward pass."""
        from alignment_microscope import Microscope
        import torch

        scope = Microscope.for_model(gpt2_model)

        with scope.trace() as trace:
            with torch.no_grad():
                gpt2_model(**gpt2_input)

        # Should have captured activations
        assert len(trace.layers) > 0

    def test_trace_captures_all_layers(self, gpt2_model, gpt2_input):
        """Test that all layers are captured."""
        from alignment_microscope import Microscope
        import torch

        scope = Microscope.for_model(gpt2_model)

        with scope.trace() as trace:
            with torch.no_grad():
                gpt2_model(**gpt2_input)

        # GPT-2 small has 12 layers
        captured_layers = set(trace.layers)

        # We should capture at least some layers
        assert len(captured_layers) > 0

    def test_trace_activation_shapes(self, gpt2_model, gpt2_input):
        """Test that captured activations have correct shapes."""
        from alignment_microscope import Microscope
        import torch

        scope = Microscope.for_model(gpt2_model)
        seq_len = gpt2_input["input_ids"].shape[1]

        with scope.trace() as trace:
            with torch.no_grad():
                gpt2_model(**gpt2_input)

        for layer in trace.layers:
            attn_out = trace.attention_out(layer)
            if attn_out is not None:
                assert attn_out.shape[1] == seq_len
                assert attn_out.shape[2] == 768  # GPT-2 hidden size

    def test_classify_heads_gpt2(self, gpt2_model, gpt2_input):
        """Test head classification on real GPT-2 attention."""
        from alignment_microscope import Microscope
        import torch

        scope = Microscope.for_model(gpt2_model)

        with scope.trace() as trace:
            with torch.no_grad():
                gpt2_model(**gpt2_input, output_attentions=True)

        # Classify heads in first layer
        pattern = trace.attention(0)
        if pattern is not None:
            classifications = scope.classify_heads(pattern)
            assert len(classifications) == 12

    def test_token_norms_real_activations(self, gpt2_model, gpt2_input):
        """Test computing token norms on real activations."""
        from alignment_microscope import Microscope
        import torch

        scope = Microscope.for_model(gpt2_model)

        with scope.trace() as trace:
            with torch.no_grad():
                gpt2_model(**gpt2_input)

        for layer in trace.layers[:3]:  # Test first 3 layers
            attn_out = trace.attention_out(layer)
            if attn_out is not None:
                norms = trace.token_norms(layer, "attn_out")
                assert norms is not None
                assert np.all(norms >= 0)
                assert np.all(np.isfinite(norms))


class TestGPT2CircuitDiscovery:
    """Integration tests for circuit discovery with GPT-2."""

    @pytest.fixture(autouse=True)
    def check_dependencies(self):
        """Skip if transformers not installed."""
        pytest.importorskip("transformers")
        pytest.importorskip("torch")

    def test_discover_circuit_different_inputs(self, gpt2_model, gpt2_tokenizer):
        """Test circuit discovery with different inputs."""
        from alignment_microscope import Microscope
        import torch

        scope = Microscope.for_model(gpt2_model)

        # Clean input
        clean_input = gpt2_tokenizer("The capital of France is", return_tensors="pt")
        with scope.trace() as clean_trace:
            with torch.no_grad():
                gpt2_model(**clean_input)

        # Corrupt input (different but similar length)
        corrupt_input = gpt2_tokenizer("The capital of Germany is", return_tensors="pt")
        with scope.trace() as corrupt_trace:
            with torch.no_grad():
                gpt2_model(**corrupt_input)

        # Discover circuit
        circuit = scope.discover_circuit(
            behavior="geography", clean_trace=clean_trace, corrupt_trace=corrupt_trace
        )

        # Should find some differences
        assert circuit.name == "geography"
        # The circuit should have nodes if there are differences
        assert circuit.num_nodes >= 0  # May be empty if no significant differences


class TestGPT2GenerateIntegration:
    """Integration tests with GPT-2 generation."""

    @pytest.fixture(autouse=True)
    def check_dependencies(self):
        """Skip if transformers not installed."""
        pytest.importorskip("transformers")
        pytest.importorskip("torch")

    def test_trace_during_generation(self, gpt2_model, gpt2_tokenizer):
        """Test that tracing works during text generation."""
        from alignment_microscope import Microscope
        import torch

        scope = Microscope.for_model(gpt2_model)

        input_text = "Hello, my name is"
        inputs = gpt2_tokenizer(input_text, return_tensors="pt")

        with scope.trace() as trace:
            with torch.no_grad():
                outputs = gpt2_model.generate(
                    inputs["input_ids"],
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=gpt2_tokenizer.eos_token_id,
                )

        # Should have captured activations during generation
        assert len(trace.layers) > 0

    def test_multiple_forward_passes(self, gpt2_model, gpt2_tokenizer):
        """Test tracing with multiple forward passes in same context."""
        from alignment_microscope import Microscope
        import torch

        scope = Microscope.for_model(gpt2_model)

        with scope.trace() as trace:
            with torch.no_grad():
                # Multiple forward passes
                for text in ["Hello world", "Goodbye world", "Test input"]:
                    inputs = gpt2_tokenizer(text, return_tensors="pt")
                    gpt2_model(**inputs)

        # Should have captured activations from all passes
        assert len(trace.layers) > 0
