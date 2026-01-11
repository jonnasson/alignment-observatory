"""Qwen architecture adapter."""

from typing import Any, Dict, List

from .base import ArchitectureAdapter
from .detection import Architecture
from .registry import AdapterRegistry


@AdapterRegistry.register(Architecture.QWEN)
class QwenAdapter(ArchitectureAdapter):
    """Adapter for Qwen-family models.

    Supports: Qwen, Qwen2, Qwen1.5, etc.

    Model structure (similar to Llama):
        model.model.layers[i].self_attn
        model.model.layers[i].mlp
        model.model.layers[i].input_layernorm
        model.model.layers[i].post_attention_layernorm

    Qwen2 may use:
        model.transformer.h[i] for older versions
    """

    @property
    def architecture_name(self) -> str:
        return "qwen"

    def get_layers(self, model: Any) -> List[Any]:
        """Get transformer layers from Qwen model."""
        # Try Qwen2/newer structure first (same as Llama)
        if hasattr(model, "model"):
            inner = model.model
            if hasattr(inner, "layers"):
                return list(inner.layers)

        # Try older Qwen structure
        if hasattr(model, "transformer"):
            transformer = model.transformer
            if hasattr(transformer, "h"):
                return list(transformer.h)

        raise AttributeError("Cannot find layers in Qwen model")

    def get_attention_module(self, layer: Any) -> Any:
        """Get attention module from layer."""
        # Qwen2 style
        if hasattr(layer, "self_attn"):
            return layer.self_attn
        # Older Qwen style
        if hasattr(layer, "attn"):
            return layer.attn
        raise AttributeError("Cannot find attention module in layer")

    def get_mlp_module(self, layer: Any) -> Any:
        """Get MLP module from layer."""
        if hasattr(layer, "mlp"):
            return layer.mlp
        raise AttributeError("Cannot find mlp in layer")

    def get_layer_norm_modules(self, layer: Any) -> Dict[str, Any]:
        """Get layer normalization modules."""
        norms = {}
        # Qwen2 style
        if hasattr(layer, "input_layernorm"):
            norms["pre_attn"] = layer.input_layernorm
        if hasattr(layer, "post_attention_layernorm"):
            norms["pre_mlp"] = layer.post_attention_layernorm
        # Older Qwen style
        if hasattr(layer, "ln_1"):
            norms["pre_attn"] = layer.ln_1
        if hasattr(layer, "ln_2"):
            norms["pre_mlp"] = layer.ln_2
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
            if hasattr(config, "n_embd"):
                return config.n_embd
        raise AttributeError("Cannot determine hidden_size from model config")

    def get_num_layers(self, model: Any) -> int:
        """Get number of transformer layers."""
        config = getattr(model, "config", None)
        if config is not None:
            if hasattr(config, "num_hidden_layers"):
                return config.num_hidden_layers
            if hasattr(config, "n_layer"):
                return config.n_layer
        return len(self.get_layers(model))
