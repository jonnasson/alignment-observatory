"""Type stubs for architecture adapters package."""

from typing import List

from .base import ArchitectureAdapter as ArchitectureAdapter
from .detection import Architecture as Architecture
from .detection import detect_architecture as detect_architecture
from .registry import AdapterRegistry as AdapterRegistry
from .registry import get_adapter as get_adapter

# Concrete adapters
class LlamaAdapter(ArchitectureAdapter): ...
class GPT2Adapter(ArchitectureAdapter): ...
class MistralAdapter(ArchitectureAdapter): ...
class QwenAdapter(ArchitectureAdapter): ...
class GemmaAdapter(ArchitectureAdapter): ...

__all__: List[str]
