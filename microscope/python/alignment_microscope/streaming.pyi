"""Type stubs for streaming activation capture module."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

ArrayFloat32 = npt.NDArray[np.float32]

@dataclass
class StreamingConfig:
    """Configuration for streaming activation capture."""

    storage_dir: str = "/tmp/alignment_microscope"
    memory_limit_gb: float = 4.0
    capture_layers: List[int] = field(default_factory=list)
    capture_components: List[str] = field(
        default_factory=lambda: ["residual", "attn_out", "mlp_out"]
    )
    use_mmap: bool = True
    ring_buffer_size: int = 1000

    @classmethod
    def for_large_model(
        cls, storage_dir: str = "/tmp/alignment_microscope"
    ) -> StreamingConfig: ...
    @classmethod
    def selective(
        cls, layers: List[int], storage_dir: str = "/tmp/alignment_microscope"
    ) -> StreamingConfig: ...

@dataclass
class ChunkMetadata:
    """Metadata for a stored activation chunk."""

    layer: int
    component: str
    shape: Tuple[int, ...]
    dtype: str
    offset: int
    size_bytes: int
    token_range: Tuple[int, int]

class ActivationStorage:
    """Backend for storing activations to disk."""

    config: StreamingConfig
    storage_dir: Any  # Path

    def __init__(self, config: StreamingConfig) -> None: ...
    def should_capture(self, layer: int, component: str) -> bool: ...
    def store(
        self,
        layer: int,
        component: str,
        data: ArrayFloat32,
        token_range: Tuple[int, int] = (0, 0),
    ) -> bool: ...
    def load(
        self, layer: int, component: str, chunk_idx: int = 0
    ) -> Optional[ArrayFloat32]: ...
    def iter_chunks(
        self, layer: int, component: str
    ) -> Iterator[Tuple[int, ArrayFloat32]]: ...
    def get_metadata(
        self, layer: int, component: str
    ) -> List[ChunkMetadata]: ...
    def available_layers(self) -> List[int]: ...
    def total_size_bytes(self) -> int: ...
    def flush(self) -> None: ...
    def cleanup(self) -> None: ...

class RingBuffer:
    """Ring buffer for streaming activation analysis."""

    capacity: int
    layer: int
    component: str

    def __init__(
        self, capacity: int, layer: int, component: str
    ) -> None: ...
    def push(self, data: ArrayFloat32) -> None: ...
    def recent(self, n: int = 1) -> List[ArrayFloat32]: ...
    def all(self) -> List[ArrayFloat32]: ...
    def __len__(self) -> int: ...
    def is_empty(self) -> bool: ...
    def clear(self) -> None: ...

class StreamingTrace:
    """Activation trace backed by disk storage."""

    def __init__(self, storage: ActivationStorage) -> None: ...
    def add_activation(
        self,
        layer: int,
        component: str,
        data: ArrayFloat32,
        token_range: Tuple[int, int] = (0, 0),
    ) -> None: ...
    def enable_ring_buffer(
        self, layer: int, component: str, capacity: int
    ) -> None: ...
    def get(
        self, layer: int, component: str, chunk_idx: int = 0
    ) -> Optional[ArrayFloat32]: ...
    def get_recent(
        self, layer: int, component: str, n: int = 1
    ) -> List[ArrayFloat32]: ...
    def iter_layer(
        self, layer: int, component: str = "residual"
    ) -> Iterator[Tuple[int, ArrayFloat32]]: ...
    def residual(
        self, layer: int, chunk_idx: int = 0
    ) -> Optional[ArrayFloat32]: ...
    def attention_out(
        self, layer: int, chunk_idx: int = 0
    ) -> Optional[ArrayFloat32]: ...
    def mlp_out(
        self, layer: int, chunk_idx: int = 0
    ) -> Optional[ArrayFloat32]: ...
    @property
    def layers(self) -> List[int]: ...
    @property
    def input_tokens(self) -> List[int]: ...
    def flush(self) -> None: ...
    def cleanup(self) -> None: ...

class StreamingMicroscope:
    """Microscope with streaming activation capture."""

    config: StreamingConfig
    architecture: str
    num_layers: int
    num_heads: int
    hidden_size: int

    def __init__(
        self,
        config: Optional[StreamingConfig] = None,
        architecture: str = "llama",
        num_layers: int = 32,
        num_heads: int = 32,
        hidden_size: int = 4096,
    ) -> None: ...
    @classmethod
    def for_model(
        cls, model: Any, config: Optional[StreamingConfig] = None
    ) -> StreamingMicroscope: ...
    @contextmanager
    def trace(
        self,
        input_tokens: Optional[List[int]] = None,
        cleanup_on_exit: bool = False,
    ) -> Generator[StreamingTrace, None, None]: ...

class MemoryEstimator:
    """Utilities for estimating memory requirements."""

    @staticmethod
    def estimate_full_capture(
        num_layers: int,
        hidden_size: int,
        batch_size: int = 1,
        seq_len: int = 1024,
    ) -> int: ...
    @staticmethod
    def suggest_strategy(
        num_layers: int,
        hidden_size: int,
        memory_limit_gb: float = 4.0,
    ) -> str: ...
    @staticmethod
    def key_layers(num_layers: int) -> List[int]: ...
