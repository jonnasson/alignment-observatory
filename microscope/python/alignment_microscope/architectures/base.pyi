"""Type stubs for base architecture adapter."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

ArrayFloat32 = npt.NDArray[np.float32]

class ArchitectureAdapter(ABC):
    """Abstract base class for architecture-specific adapters."""

    @property
    @abstractmethod
    def architecture_name(self) -> str: ...
    @abstractmethod
    def get_layers(self, model: Any) -> List[Any]: ...
    @abstractmethod
    def get_attention_module(self, layer: Any) -> Any: ...
    @abstractmethod
    def get_mlp_module(self, layer: Any) -> Any: ...
    def get_layer_norm_modules(self, layer: Any) -> Dict[str, Any]: ...
    @abstractmethod
    def get_num_heads(self, model: Any) -> int: ...
    @abstractmethod
    def get_hidden_size(self, model: Any) -> int: ...
    @abstractmethod
    def get_num_layers(self, model: Any) -> int: ...
    def register_residual_hook(
        self,
        model: Any,
        layer_idx: int,
        callback: Callable[[ArrayFloat32], None],
    ) -> Callable[[], None]: ...
    def register_attention_hook(
        self,
        model: Any,
        layer_idx: int,
        callback: Callable[[ArrayFloat32], None],
    ) -> Callable[[], None]: ...
    def register_mlp_hook(
        self,
        model: Any,
        layer_idx: int,
        callback: Callable[[ArrayFloat32], None],
    ) -> Callable[[], None]: ...
    def extract_attention_pattern(
        self,
        model: Any,
        layer_idx: int,
        attention_output: Tuple[Any, ...],
    ) -> Optional[ArrayFloat32]: ...
    def supports_attention_output(self) -> bool: ...
    def get_model_config(self, model: Any) -> Dict[str, Any]: ...
