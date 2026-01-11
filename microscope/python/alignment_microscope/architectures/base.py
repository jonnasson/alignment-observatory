"""Base architecture adapter interface."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class ArchitectureAdapter(ABC):
    """Abstract base class for architecture-specific adapters.

    Each adapter knows how to:
    - Navigate the model's layer structure
    - Register forward hooks for activation capture
    - Extract attention patterns in a normalized format
    """

    @property
    @abstractmethod
    def architecture_name(self) -> str:
        """Return the canonical name of this architecture."""
        pass

    @abstractmethod
    def get_layers(self, model: Any) -> List[Any]:
        """Get the list of transformer layers from the model.

        Args:
            model: The transformer model instance

        Returns:
            List of layer modules
        """
        pass

    @abstractmethod
    def get_attention_module(self, layer: Any) -> Any:
        """Get the attention module from a layer.

        Args:
            layer: A single transformer layer

        Returns:
            The attention module
        """
        pass

    @abstractmethod
    def get_mlp_module(self, layer: Any) -> Any:
        """Get the MLP/FFN module from a layer.

        Args:
            layer: A single transformer layer

        Returns:
            The MLP module
        """
        pass

    def get_layer_norm_modules(self, layer: Any) -> Dict[str, Any]:
        """Get layer normalization modules from a layer.

        Args:
            layer: A single transformer layer

        Returns:
            Dict with keys like 'pre_attn', 'pre_mlp', 'post_attn', etc.
        """
        return {}

    @abstractmethod
    def get_num_heads(self, model: Any) -> int:
        """Get the number of attention heads.

        Args:
            model: The transformer model instance

        Returns:
            Number of attention heads
        """
        pass

    @abstractmethod
    def get_hidden_size(self, model: Any) -> int:
        """Get the hidden dimension size.

        Args:
            model: The transformer model instance

        Returns:
            Hidden size
        """
        pass

    @abstractmethod
    def get_num_layers(self, model: Any) -> int:
        """Get the number of transformer layers.

        Args:
            model: The transformer model instance

        Returns:
            Number of layers
        """
        pass

    def register_residual_hook(
        self,
        model: Any,
        layer_idx: int,
        callback: Callable[[np.ndarray], None]
    ) -> Callable[[], None]:
        """Register a hook to capture residual stream activations.

        Args:
            model: The transformer model instance
            layer_idx: Layer index to hook
            callback: Function called with activations as numpy array

        Returns:
            A function that removes the hook when called
        """
        layers = self.get_layers(model)
        layer = layers[layer_idx]

        def hook(module, input, output):
            # Handle different output formats
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            if hasattr(hidden_states, 'detach'):
                arr = hidden_states.detach().cpu().numpy()
            else:
                arr = np.array(hidden_states)
            callback(arr)

        handle = layer.register_forward_hook(hook)
        return handle.remove

    def register_attention_hook(
        self,
        model: Any,
        layer_idx: int,
        callback: Callable[[np.ndarray], None]
    ) -> Callable[[], None]:
        """Register a hook to capture attention output.

        Args:
            model: The transformer model instance
            layer_idx: Layer index to hook
            callback: Function called with attention output as numpy array

        Returns:
            A function that removes the hook when called
        """
        layers = self.get_layers(model)
        attn = self.get_attention_module(layers[layer_idx])

        def hook(module, input, output):
            if isinstance(output, tuple):
                attn_out = output[0]
            else:
                attn_out = output

            if hasattr(attn_out, 'detach'):
                arr = attn_out.detach().cpu().numpy()
            else:
                arr = np.array(attn_out)
            callback(arr)

        handle = attn.register_forward_hook(hook)
        return handle.remove

    def register_mlp_hook(
        self,
        model: Any,
        layer_idx: int,
        callback: Callable[[np.ndarray], None]
    ) -> Callable[[], None]:
        """Register a hook to capture MLP output.

        Args:
            model: The transformer model instance
            layer_idx: Layer index to hook
            callback: Function called with MLP output as numpy array

        Returns:
            A function that removes the hook when called
        """
        layers = self.get_layers(model)
        mlp = self.get_mlp_module(layers[layer_idx])

        def hook(module, input, output):
            if isinstance(output, tuple):
                mlp_out = output[0]
            else:
                mlp_out = output

            if hasattr(mlp_out, 'detach'):
                arr = mlp_out.detach().cpu().numpy()
            else:
                arr = np.array(mlp_out)
            callback(arr)

        handle = mlp.register_forward_hook(hook)
        return handle.remove

    def extract_attention_pattern(
        self,
        model: Any,
        layer_idx: int,
        attention_output: Tuple[Any, ...]
    ) -> Optional[np.ndarray]:
        """Extract attention weights from attention module output.

        Args:
            model: The transformer model instance
            layer_idx: Layer index
            attention_output: Output tuple from attention forward pass

        Returns:
            Attention weights as numpy array [batch, heads, seq_q, seq_k]
            or None if not available
        """
        # Default implementation: look for attention weights in output tuple
        if len(attention_output) > 1 and attention_output[1] is not None:
            attn_weights = attention_output[1]
            if hasattr(attn_weights, 'detach'):
                return attn_weights.detach().cpu().numpy()
            return np.array(attn_weights)
        return None

    def supports_attention_output(self) -> bool:
        """Check if this architecture supports output_attentions.

        Returns:
            True if the model can output attention weights
        """
        return True

    def get_model_config(self, model: Any) -> Dict[str, Any]:
        """Extract model configuration.

        Args:
            model: The transformer model instance

        Returns:
            Dict with num_layers, num_heads, hidden_size, etc.
        """
        return {
            "architecture": self.architecture_name,
            "num_layers": self.get_num_layers(model),
            "num_heads": self.get_num_heads(model),
            "hidden_size": self.get_hidden_size(model),
        }
