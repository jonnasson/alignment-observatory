"""Gemma architecture adapter."""

from typing import Any, Dict, List

from .base import ArchitectureAdapter
from .detection import Architecture
from .registry import AdapterRegistry


@AdapterRegistry.register(Architecture.GEMMA)
class GemmaAdapter(ArchitectureAdapter):
    """Adapter for Gemma-family models.

    Supports: Gemma, Gemma 2, etc.

    Model structure (similar to Llama):
        model.model.layers[i].self_attn
        model.model.layers[i].mlp
        model.model.layers[i].input_layernorm
        model.model.layers[i].post_attention_layernorm

    Key differences from Llama:
        - Uses RMSNorm with learned scaling
        - Different activation function (GELU approximation)
        - Different head dimension calculations
    """

    @property
    def architecture_name(self) -> str:
        return "gemma"

    def get_layers(self, model: Any) -> List[Any]:
        """Get transformer layers from Gemma model."""
        if hasattr(model, "model"):
            inner = model.model
        else:
            inner = model

        if hasattr(inner, "layers"):
            return list(inner.layers)
        raise AttributeError("Cannot find layers in Gemma model")

    def get_attention_module(self, layer: Any) -> Any:
        """Get self-attention module from layer."""
        if hasattr(layer, "self_attn"):
            return layer.self_attn
        raise AttributeError("Cannot find self_attn in layer")

    def get_mlp_module(self, layer: Any) -> Any:
        """Get MLP module from layer."""
        if hasattr(layer, "mlp"):
            return layer.mlp
        raise AttributeError("Cannot find mlp in layer")

    def get_layer_norm_modules(self, layer: Any) -> Dict[str, Any]:
        """Get layer normalization modules.

        Gemma uses RMSNorm with a learned scale parameter.
        """
        norms = {}
        if hasattr(layer, "input_layernorm"):
            norms["pre_attn"] = layer.input_layernorm
        if hasattr(layer, "post_attention_layernorm"):
            norms["pre_mlp"] = layer.post_attention_layernorm
        # Gemma 2 may have additional norms
        if hasattr(layer, "pre_feedforward_layernorm"):
            norms["pre_mlp"] = layer.pre_feedforward_layernorm
        if hasattr(layer, "post_feedforward_layernorm"):
            norms["post_mlp"] = layer.post_feedforward_layernorm
        return norms

    def get_num_heads(self, model: Any) -> int:
        """Get number of attention heads."""
        config = getattr(model, "config", None)
        if config is not None:
            if hasattr(config, "num_attention_heads"):
                return config.num_attention_heads
            if hasattr(config, "n_head"):
                return config.n_head
        raise AttributeError("Cannot determine num_heads from model config")

    def get_hidden_size(self, model: Any) -> int:
        """Get hidden dimension size."""
        config = getattr(model, "config", None)
        if config is not None:
            if hasattr(config, "hidden_size"):
                return config.hidden_size
        raise AttributeError("Cannot determine hidden_size from model config")

    def get_num_layers(self, model: Any) -> int:
        """Get number of transformer layers."""
        config = getattr(model, "config", None)
        if config is not None:
            if hasattr(config, "num_hidden_layers"):
                return config.num_hidden_layers
        return len(self.get_layers(model))

    def get_head_dim(self, model: Any) -> int:
        """Get attention head dimension.

        Gemma may have different head_dim calculation than hidden_size / num_heads.
        """
        config = getattr(model, "config", None)
        if config is not None:
            if hasattr(config, "head_dim"):
                return config.head_dim
            # Fall back to standard calculation
            return self.get_hidden_size(model) // self.get_num_heads(model)
        raise AttributeError("Cannot determine head_dim from model config")
