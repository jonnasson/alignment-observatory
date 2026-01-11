"""Architecture adapters for different transformer model families.

This package provides a unified interface for extracting activations and
attention patterns from various transformer architectures.
"""

from .base import ArchitectureAdapter
from .detection import Architecture, detect_architecture
from .registry import AdapterRegistry, get_adapter

# Lazy imports for specific adapters
from .llama import LlamaAdapter
from .gpt2 import GPT2Adapter
from .mistral import MistralAdapter
from .qwen import QwenAdapter
from .gemma import GemmaAdapter

__all__ = [
    "ArchitectureAdapter",
    "Architecture",
    "detect_architecture",
    "AdapterRegistry",
    "get_adapter",
    "LlamaAdapter",
    "GPT2Adapter",
    "MistralAdapter",
    "QwenAdapter",
    "GemmaAdapter",
]
