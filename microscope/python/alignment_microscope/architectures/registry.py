"""Adapter registry and factory."""

from typing import Any, Dict, Optional, Type
import warnings

from .base import ArchitectureAdapter
from .detection import Architecture, detect_architecture


class AdapterRegistry:
    """Registry for architecture adapters.

    Provides adapter lookup and fallback chain for unknown architectures.
    """

    _adapters: Dict[Architecture, Type[ArchitectureAdapter]] = {}
    _fallback_chain = [
        Architecture.LLAMA,  # Most common structure
        Architecture.GPT2,   # Alternative common structure
    ]

    @classmethod
    def register(cls, architecture: Architecture):
        """Decorator to register an adapter class.

        Usage:
            @AdapterRegistry.register(Architecture.LLAMA)
            class LlamaAdapter(ArchitectureAdapter):
                ...
        """
        def decorator(adapter_class: Type[ArchitectureAdapter]):
            cls._adapters[architecture] = adapter_class
            return adapter_class
        return decorator

    @classmethod
    def get(cls, architecture: Architecture) -> Optional[Type[ArchitectureAdapter]]:
        """Get adapter class for an architecture.

        Args:
            architecture: Architecture enum value

        Returns:
            Adapter class or None if not found
        """
        return cls._adapters.get(architecture)

    @classmethod
    def get_for_model(
        cls,
        model: Any,
        fallback: bool = True
    ) -> Optional[ArchitectureAdapter]:
        """Get an adapter instance for a model.

        Args:
            model: Transformer model instance
            fallback: Whether to try fallback adapters for unknown architectures

        Returns:
            Adapter instance or None if no suitable adapter found
        """
        arch = detect_architecture(model)

        # Try direct lookup
        adapter_class = cls._adapters.get(arch)
        if adapter_class is not None:
            return adapter_class()

        if arch == Architecture.UNKNOWN and fallback:
            # Try fallback chain
            for fallback_arch in cls._fallback_chain:
                fallback_class = cls._adapters.get(fallback_arch)
                if fallback_class is not None:
                    # Verify the adapter works with this model
                    adapter = fallback_class()
                    if cls._verify_adapter(adapter, model):
                        warnings.warn(
                            f"Using {fallback_arch.name} adapter as fallback for "
                            f"unknown architecture. Some features may not work correctly.",
                            UserWarning
                        )
                        return adapter

        return None

    @classmethod
    def _verify_adapter(cls, adapter: ArchitectureAdapter, model: Any) -> bool:
        """Verify that an adapter can work with a model.

        Args:
            adapter: Adapter instance
            model: Transformer model instance

        Returns:
            True if adapter appears compatible
        """
        try:
            # Try basic operations
            layers = adapter.get_layers(model)
            if not layers:
                return False

            # Try to get attention module from first layer
            attn = adapter.get_attention_module(layers[0])
            if attn is None:
                return False

            return True
        except (AttributeError, IndexError, TypeError):
            return False

    @classmethod
    def list_architectures(cls) -> list:
        """List all registered architectures.

        Returns:
            List of Architecture enum values with registered adapters
        """
        return list(cls._adapters.keys())


def get_adapter(model: Any, fallback: bool = True) -> Optional[ArchitectureAdapter]:
    """Convenience function to get an adapter for a model.

    Args:
        model: Transformer model instance
        fallback: Whether to try fallback adapters

    Returns:
        Adapter instance or None
    """
    return AdapterRegistry.get_for_model(model, fallback=fallback)
