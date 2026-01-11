"""Mistral architecture adapter."""

from typing import Any, Dict, List

from .base import ArchitectureAdapter
from .detection import Architecture
from .registry import AdapterRegistry


@AdapterRegistry.register(Architecture.MISTRAL)
class MistralAdapter(ArchitectureAdapter):
    """Adapter for Mistral-family models.

    Supports: Mistral 7B, Mixtral, etc.

    Model structure (very similar to Llama):
        model.model.layers[i].self_attn
        model.model.layers[i].mlp
        model.model.layers[i].input_layernorm
        model.model.layers[i].post_attention_layernorm

    Key differences from Llama:
        - Uses sliding window attention
        - Different rotary embedding configuration
    """

    @property
    def architecture_name(self) -> str:
        return "mistral"

    def get_layers(self, model: Any) -> List[Any]:
        """Get transformer layers from Mistral model."""
        if hasattr(model, "model"):
            inner = model.model
        else:
            inner = model

        if hasattr(inner, "layers"):
            return list(inner.layers)
        raise AttributeError("Cannot find layers in Mistral model")

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
        """Get layer normalization modules."""
        norms = {}
        if hasattr(layer, "input_layernorm"):
            norms["pre_attn"] = layer.input_layernorm
        if hasattr(layer, "post_attention_layernorm"):
            norms["pre_mlp"] = layer.post_attention_layernorm
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

    def get_sliding_window_size(self, model: Any) -> int:
        """Get sliding window size for attention.

        Returns:
            Window size, or -1 if full attention
        """
        config = getattr(model, "config", None)
        if config is not None:
            if hasattr(config, "sliding_window"):
                return config.sliding_window or -1
        return -1
