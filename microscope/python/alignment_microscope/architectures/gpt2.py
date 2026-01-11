"""GPT-2 architecture adapter."""

from typing import Any, Dict, List, Optional

import numpy as np

from .base import ArchitectureAdapter
from .detection import Architecture
from .registry import AdapterRegistry


@AdapterRegistry.register(Architecture.GPT2)
class GPT2Adapter(ArchitectureAdapter):
    """Adapter for GPT-2 family models.

    Supports: GPT-2 (all sizes)

    Model structure:
        model.transformer.h[i].attn
        model.transformer.h[i].mlp
        model.transformer.h[i].ln_1
        model.transformer.h[i].ln_2
    """

    @property
    def architecture_name(self) -> str:
        return "gpt2"

    def get_layers(self, model: Any) -> List[Any]:
        """Get transformer layers from GPT-2 model."""
        # Handle both raw model and wrapped model
        if hasattr(model, "transformer"):
            transformer = model.transformer
        else:
            transformer = model

        if hasattr(transformer, "h"):
            return list(transformer.h)
        raise AttributeError("Cannot find layers (h) in GPT-2 model")

    def get_attention_module(self, layer: Any) -> Any:
        """Get attention module from layer."""
        if hasattr(layer, "attn"):
            return layer.attn
        raise AttributeError("Cannot find attn in layer")

    def get_mlp_module(self, layer: Any) -> Any:
        """Get MLP module from layer."""
        if hasattr(layer, "mlp"):
            return layer.mlp
        raise AttributeError("Cannot find mlp in layer")

    def get_layer_norm_modules(self, layer: Any) -> Dict[str, Any]:
        """Get layer normalization modules."""
        norms = {}
        if hasattr(layer, "ln_1"):
            norms["pre_attn"] = layer.ln_1
        if hasattr(layer, "ln_2"):
            norms["pre_mlp"] = layer.ln_2
        return norms

    def get_num_heads(self, model: Any) -> int:
        """Get number of attention heads."""
        config = getattr(model, "config", None)
        if config is not None:
            if hasattr(config, "n_head"):
                return config.n_head
            if hasattr(config, "num_attention_heads"):
                return config.num_attention_heads
        raise AttributeError("Cannot determine num_heads from model config")

    def get_hidden_size(self, model: Any) -> int:
        """Get hidden dimension size."""
        config = getattr(model, "config", None)
        if config is not None:
            if hasattr(config, "n_embd"):
                return config.n_embd
            if hasattr(config, "hidden_size"):
                return config.hidden_size
        raise AttributeError("Cannot determine hidden_size from model config")

    def get_num_layers(self, model: Any) -> int:
        """Get number of transformer layers."""
        config = getattr(model, "config", None)
        if config is not None:
            if hasattr(config, "n_layer"):
                return config.n_layer
            if hasattr(config, "num_hidden_layers"):
                return config.num_hidden_layers
        # Fall back to counting layers
        return len(self.get_layers(model))

    def extract_attention_pattern(
        self,
        model: Any,
        layer_idx: int,
        attention_output: tuple
    ) -> Optional[np.ndarray]:
        """Extract attention weights from GPT-2 attention output.

        GPT-2 attention output format:
        - (attn_output, present_key_value) when not outputting attentions
        - (attn_output, present_key_value, attn_weights) when outputting attentions

        Args:
            model: The transformer model instance
            layer_idx: Layer index
            attention_output: Output tuple from attention forward pass

        Returns:
            Attention weights as numpy array [batch, heads, seq_q, seq_k]
        """
        if len(attention_output) >= 3 and attention_output[2] is not None:
            attn_weights = attention_output[2]
            if hasattr(attn_weights, 'detach'):
                return attn_weights.detach().cpu().numpy()
            return np.array(attn_weights)
        return None
