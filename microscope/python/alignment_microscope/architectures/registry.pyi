"""Type stubs for adapter registry."""

from typing import Any, Dict, List, Optional, Type

from .base import ArchitectureAdapter
from .detection import Architecture

class AdapterRegistry:
    """Registry for architecture adapters."""

    _adapters: Dict[Architecture, Type[ArchitectureAdapter]]
    _fallback_chain: List[Architecture]

    @classmethod
    def register(
        cls, architecture: Architecture
    ) -> Any: ...  # Returns decorator
    @classmethod
    def get(
        cls, architecture: Architecture
    ) -> Optional[Type[ArchitectureAdapter]]: ...
    @classmethod
    def get_for_model(
        cls, model: Any, fallback: bool = True
    ) -> Optional[ArchitectureAdapter]: ...
    @classmethod
    def list_architectures(cls) -> List[Architecture]: ...

def get_adapter(
    model: Any, fallback: bool = True
) -> Optional[ArchitectureAdapter]: ...
