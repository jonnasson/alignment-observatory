"""Tests for architecture adapters."""

import pytest
import numpy as np

from alignment_microscope.architectures import (
    Architecture,
    detect_architecture,
    get_adapter,
    AdapterRegistry,
    ArchitectureAdapter,
    LlamaAdapter,
    GPT2Adapter,
    MistralAdapter,
    QwenAdapter,
    GemmaAdapter,
)
from alignment_microscope.architectures.detection import (
    architecture_to_string,
    string_to_architecture,
)


class TestArchitectureEnum:
    """Tests for Architecture enum."""

    def test_all_architectures_defined(self):
        """Test that all expected architectures are in enum."""
        expected = ["LLAMA", "MISTRAL", "QWEN", "GEMMA", "GPT2", "GPTJ", "GPTNEO", "FALCON", "PHI", "UNKNOWN"]
        actual = [a.name for a in Architecture]
        for arch in expected:
            assert arch in actual

    def test_architecture_to_string(self):
        """Test converting enum to string."""
        assert architecture_to_string(Architecture.LLAMA) == "llama"
        assert architecture_to_string(Architecture.GPT2) == "gpt2"
        assert architecture_to_string(Architecture.UNKNOWN) == "unknown"

    def test_string_to_architecture(self):
        """Test converting string to enum."""
        assert string_to_architecture("llama") == Architecture.LLAMA
        assert string_to_architecture("LLAMA") == Architecture.LLAMA
        assert string_to_architecture("gpt2") == Architecture.GPT2
        assert string_to_architecture("gpt-2") == Architecture.GPT2

    def test_string_to_architecture_invalid(self):
        """Test invalid string raises error."""
        with pytest.raises(ValueError):
            string_to_architecture("nonexistent_arch")


class TestAdapterRegistry:
    """Tests for AdapterRegistry."""

    def test_list_architectures(self):
        """Test listing registered architectures."""
        registered = AdapterRegistry.list_architectures()
        assert Architecture.LLAMA in registered
        assert Architecture.GPT2 in registered
        assert Architecture.MISTRAL in registered

    def test_get_adapter_by_architecture(self):
        """Test getting adapter class by architecture."""
        adapter_class = AdapterRegistry.get(Architecture.LLAMA)
        assert adapter_class is LlamaAdapter

        adapter_class = AdapterRegistry.get(Architecture.GPT2)
        assert adapter_class is GPT2Adapter

    def test_get_nonexistent_adapter(self):
        """Test getting non-existent adapter returns None."""
        adapter_class = AdapterRegistry.get(Architecture.UNKNOWN)
        assert adapter_class is None


class TestLlamaAdapter:
    """Tests for LlamaAdapter."""

    def test_architecture_name(self):
        """Test adapter returns correct architecture name."""
        adapter = LlamaAdapter()
        assert adapter.architecture_name == "llama"

    def test_is_architecture_adapter(self):
        """Test adapter is instance of base class."""
        adapter = LlamaAdapter()
        assert isinstance(adapter, ArchitectureAdapter)

    def test_supports_attention_output(self):
        """Test adapter supports attention output."""
        adapter = LlamaAdapter()
        assert adapter.supports_attention_output() is True


class TestGPT2Adapter:
    """Tests for GPT2Adapter."""

    def test_architecture_name(self):
        """Test adapter returns correct architecture name."""
        adapter = GPT2Adapter()
        assert adapter.architecture_name == "gpt2"

    def test_is_architecture_adapter(self):
        """Test adapter is instance of base class."""
        adapter = GPT2Adapter()
        assert isinstance(adapter, ArchitectureAdapter)

    def test_supports_attention_output(self):
        """Test adapter supports attention output."""
        adapter = GPT2Adapter()
        assert adapter.supports_attention_output() is True


class TestMistralAdapter:
    """Tests for MistralAdapter."""

    def test_architecture_name(self):
        """Test adapter returns correct architecture name."""
        adapter = MistralAdapter()
        assert adapter.architecture_name == "mistral"

    def test_has_sliding_window_method(self):
        """Test adapter has sliding window method."""
        adapter = MistralAdapter()
        assert hasattr(adapter, "get_sliding_window_size")


class TestQwenAdapter:
    """Tests for QwenAdapter."""

    def test_architecture_name(self):
        """Test adapter returns correct architecture name."""
        adapter = QwenAdapter()
        assert adapter.architecture_name == "qwen"


class TestGemmaAdapter:
    """Tests for GemmaAdapter."""

    def test_architecture_name(self):
        """Test adapter returns correct architecture name."""
        adapter = GemmaAdapter()
        assert adapter.architecture_name == "gemma"

    def test_has_head_dim_method(self):
        """Test adapter has head_dim method."""
        adapter = GemmaAdapter()
        assert hasattr(adapter, "get_head_dim")


class TestDetectArchitecture:
    """Tests for architecture detection."""

    def test_detect_unknown_object(self):
        """Test detecting architecture of unknown object."""
        class FakeModel:
            pass

        model = FakeModel()
        arch = detect_architecture(model)
        assert arch == Architecture.UNKNOWN

    def test_detect_from_config_model_type(self):
        """Test detecting architecture from config.model_type."""
        class FakeConfig:
            model_type = "llama"

        class FakeModel:
            config = FakeConfig()

        model = FakeModel()
        arch = detect_architecture(model)
        assert arch == Architecture.LLAMA

    def test_detect_gpt2_from_model_type(self):
        """Test detecting GPT-2 from config.model_type."""
        class FakeConfig:
            model_type = "gpt2"

        class FakeModel:
            config = FakeConfig()

        model = FakeModel()
        arch = detect_architecture(model)
        assert arch == Architecture.GPT2

    def test_detect_mistral_from_model_type(self):
        """Test detecting Mistral from config.model_type."""
        class FakeConfig:
            model_type = "mistral"

        class FakeModel:
            config = FakeConfig()

        model = FakeModel()
        arch = detect_architecture(model)
        assert arch == Architecture.MISTRAL

    def test_detect_from_class_name(self):
        """Test detecting architecture from class name."""
        class LlamaForCausalLM:
            pass

        model = LlamaForCausalLM()
        arch = detect_architecture(model)
        assert arch == Architecture.LLAMA

    def test_detect_gpt2_from_class_name(self):
        """Test detecting GPT-2 from class name."""
        class GPT2LMHeadModel:
            pass

        model = GPT2LMHeadModel()
        arch = detect_architecture(model)
        assert arch == Architecture.GPT2


class TestGetAdapter:
    """Tests for get_adapter convenience function."""

    def test_get_adapter_for_fake_llama(self):
        """Test getting adapter for fake Llama model."""
        class FakeConfig:
            model_type = "llama"
            num_hidden_layers = 12
            num_attention_heads = 8
            hidden_size = 512

        class FakeAttn:
            pass

        class FakeMLP:
            pass

        class FakeLayer:
            self_attn = FakeAttn()
            mlp = FakeMLP()

        class FakeInner:
            layers = [FakeLayer() for _ in range(12)]

        class FakeModel:
            config = FakeConfig()
            model = FakeInner()

        model = FakeModel()
        adapter = get_adapter(model)
        assert adapter is not None
        assert adapter.architecture_name == "llama"

    def test_get_adapter_fallback_disabled(self):
        """Test get_adapter returns None when fallback disabled."""
        class FakeModel:
            pass

        model = FakeModel()
        adapter = get_adapter(model, fallback=False)
        assert adapter is None


@pytest.mark.integration
class TestArchitectureIntegration:
    """Integration tests with real models."""

    def test_detect_real_gpt2(self, gpt2_model):
        """Test detecting architecture of real GPT-2 model."""
        arch = detect_architecture(gpt2_model)
        assert arch == Architecture.GPT2

    def test_get_adapter_for_real_gpt2(self, gpt2_model):
        """Test getting adapter for real GPT-2 model."""
        adapter = get_adapter(gpt2_model)
        assert adapter is not None
        assert adapter.architecture_name == "gpt2"

    def test_adapter_gets_layers(self, gpt2_model):
        """Test adapter can get layers from real model."""
        adapter = get_adapter(gpt2_model)
        layers = adapter.get_layers(gpt2_model)
        assert len(layers) == 12  # GPT-2 small has 12 layers

    def test_adapter_gets_attention(self, gpt2_model):
        """Test adapter can get attention from layer."""
        adapter = get_adapter(gpt2_model)
        layers = adapter.get_layers(gpt2_model)
        attn = adapter.get_attention_module(layers[0])
        assert attn is not None

    def test_adapter_gets_mlp(self, gpt2_model):
        """Test adapter can get MLP from layer."""
        adapter = get_adapter(gpt2_model)
        layers = adapter.get_layers(gpt2_model)
        mlp = adapter.get_mlp_module(layers[0])
        assert mlp is not None

    def test_adapter_gets_config(self, gpt2_model):
        """Test adapter can get model config."""
        adapter = get_adapter(gpt2_model)
        config = adapter.get_model_config(gpt2_model)
        assert config["architecture"] == "gpt2"
        assert config["num_layers"] == 12
        assert config["num_heads"] == 12
        assert config["hidden_size"] == 768
