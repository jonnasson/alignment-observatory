"""Type stubs for alignment_microscope package."""

from __future__ import annotations

from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt

__version__: str
__author__: str

# Type aliases
ArrayFloat32 = npt.NDArray[np.float32]
ArrayFloat64 = npt.NDArray[np.float64]

class ActivationTrace:
    """Container for activations captured during a forward pass."""

    def __init__(self, rust_trace: Optional[Any] = None) -> None: ...
    def add_activation(
        self, layer: int, component: str, data: Any  # torch.Tensor
    ) -> None: ...
    def add_attention_pattern(self, layer: int, pattern: Any) -> None: ...
    def get(self, layer: int, component: str) -> Optional[ArrayFloat32]: ...
    def residual(self, layer: int) -> Optional[ArrayFloat32]: ...
    def attention_out(self, layer: int) -> Optional[ArrayFloat32]: ...
    def mlp_out(self, layer: int) -> Optional[ArrayFloat32]: ...
    def attention(self, layer: int) -> Optional[ArrayFloat32]: ...
    @property
    def layers(self) -> List[int]: ...
    def token_norms(
        self, layer: int, component: str = "residual"
    ) -> Optional[ArrayFloat32]: ...
    @property
    def activations(self) -> Dict[str, ArrayFloat32]: ...
    @property
    def attention_patterns(self) -> Dict[int, ArrayFloat32]: ...
    @property
    def input_tokens(self) -> List[int]: ...
    def to_dict(self) -> Dict[str, Any]: ...

class AttentionPattern:
    """Wrapper for attention patterns with analysis methods."""

    layer: int
    pattern: ArrayFloat32  # Shape: [batch, heads, seq_q, seq_k]

    def __init__(self, layer: int, pattern: ArrayFloat32) -> None: ...
    @property
    def num_heads(self) -> int: ...
    @property
    def seq_len(self) -> int: ...
    def head(self, head_idx: int, batch: int = 0) -> ArrayFloat32: ...
    def entropy(self) -> ArrayFloat32: ...
    def top_attended(self, k: int = 5) -> npt.NDArray[np.int64]: ...

class Circuit:
    """Represents a computational circuit in the model."""

    name: str
    description: str
    behavior: str
    nodes: List[Tuple[int, str, Optional[int]]]  # (layer, component, head)
    edges: List[Tuple[Tuple[int, str, Optional[int]], Tuple[int, str, Optional[int]], float]]

    def __init__(
        self, name: str, description: str = "", behavior: str = ""
    ) -> None: ...
    def add_node(
        self, layer: int, component: str, head: Optional[int] = None
    ) -> None: ...
    def add_edge(
        self,
        from_node: Tuple[int, str, Optional[int]],
        to_node: Tuple[int, str, Optional[int]],
        importance: float = 1.0,
    ) -> None: ...
    def minimal(self, threshold: float = 0.5) -> Circuit: ...
    def to_dot(self) -> str: ...

class Microscope:
    """Main interface for the interpretability toolkit."""

    architecture: str
    num_layers: int
    num_heads: int
    hidden_size: int

    def __init__(
        self,
        architecture: str = "llama",
        num_layers: int = 32,
        num_heads: int = 32,
        hidden_size: int = 4096,
    ) -> None: ...
    @classmethod
    def for_model(cls, model: Any) -> Microscope: ...
    @classmethod
    def for_llama(
        cls,
        num_layers: int = 32,
        num_heads: int = 32,
        hidden_size: int = 4096,
    ) -> Microscope: ...
    @contextmanager
    def trace(
        self, input_tokens: Optional[List[int]] = None
    ) -> Iterator[ActivationTrace]: ...
    def classify_heads(
        self, pattern: Union[ArrayFloat32, AttentionPattern]
    ) -> List[str]: ...
    def discover_circuit(
        self,
        behavior: str,
        clean_trace: ActivationTrace,
        corrupt_trace: ActivationTrace,
        metric_fn: Optional[Callable[[ActivationTrace], float]] = None,
    ) -> Circuit: ...

def create_microscope(
    model: Optional[Any] = None, **kwargs: Any
) -> Microscope: ...

# Re-exports from submodules
from alignment_microscope.sae import (
    SAEConfig as SAEConfig,
    SAEFeatures as SAEFeatures,
    SAEWrapper as SAEWrapper,
    SAEAnalyzer as SAEAnalyzer,
)
from alignment_microscope.streaming import (
    StreamingConfig as StreamingConfig,
    StreamingTrace as StreamingTrace,
    StreamingMicroscope as StreamingMicroscope,
    MemoryEstimator as MemoryEstimator,
)

__all__: List[str]
