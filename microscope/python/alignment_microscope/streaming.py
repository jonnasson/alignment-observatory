"""Streaming activation capture for large models.

This module provides memory-efficient streaming capture:

- StreamingConfig: Configuration for streaming capture
- StreamingTrace: Lazy-loading activation trace backed by disk storage
- StreamingMicroscope: Microscope with streaming support
- MemoryEstimator: Utilities for estimating memory requirements
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, Union
import json
import os
import shutil
import tempfile

import numpy as np


@dataclass
class StreamingConfig:
    """Configuration for streaming activation capture.

    Attributes:
        storage_dir: Directory for storing activation data
        memory_limit_gb: Maximum memory usage in GB
        capture_layers: Layers to capture (empty = all)
        capture_components: Components to capture
        use_mmap: Whether to use memory-mapped files
        ring_buffer_size: Size of ring buffer for sliding window
    """

    storage_dir: str = "/tmp/alignment_microscope"
    memory_limit_gb: float = 4.0
    capture_layers: List[int] = field(default_factory=list)
    capture_components: List[str] = field(
        default_factory=lambda: ["residual", "attn_out", "mlp_out"]
    )
    use_mmap: bool = True
    ring_buffer_size: int = 1000

    @classmethod
    def for_large_model(cls, storage_dir: str = "/tmp/alignment_microscope") -> "StreamingConfig":
        """Create config optimized for large models (70B+).

        Only captures residual stream to minimize memory usage.
        """
        return cls(
            storage_dir=storage_dir,
            memory_limit_gb=8.0,
            capture_components=["residual"],
            ring_buffer_size=100,
        )

    @classmethod
    def selective(
        cls,
        layers: List[int],
        storage_dir: str = "/tmp/alignment_microscope"
    ) -> "StreamingConfig":
        """Create config that only captures specific layers."""
        return cls(
            storage_dir=storage_dir,
            capture_layers=layers,
        )


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
    """Backend for storing activations to disk.

    Supports memory-mapped file access for efficient reading.
    """

    def __init__(self, config: StreamingConfig):
        """Initialize storage.

        Args:
            config: Streaming configuration
        """
        self.config = config
        self.storage_dir = Path(config.storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._metadata: Dict[str, List[ChunkMetadata]] = {}
        self._offsets: Dict[str, int] = {}
        self._files: Dict[str, Any] = {}

        # Load existing metadata if present
        self._load_metadata()

    def _metadata_path(self) -> Path:
        """Get path to metadata file."""
        return self.storage_dir / "metadata.json"

    def _file_path(self, layer: int, component: str) -> Path:
        """Get file path for layer/component."""
        return self.storage_dir / f"layer_{layer}_{component}.bin"

    def _key(self, layer: int, component: str) -> str:
        """Get key for layer/component."""
        return f"{layer}_{component}"

    def _load_metadata(self):
        """Load metadata from disk."""
        meta_path = self._metadata_path()
        if meta_path.exists():
            with open(meta_path) as f:
                data = json.load(f)
                for key, chunks in data.get("chunks", {}).items():
                    self._metadata[key] = [
                        ChunkMetadata(**{**c, "shape": tuple(c["shape"])})
                        for c in chunks
                    ]
                self._offsets = data.get("offsets", {})

    def _save_metadata(self):
        """Save metadata to disk."""
        data = {
            "chunks": {
                k: [
                    {**vars(c), "shape": list(c.shape)}
                    for c in chunks
                ]
                for k, chunks in self._metadata.items()
            },
            "offsets": self._offsets,
        }
        with open(self._metadata_path(), "w") as f:
            json.dump(data, f)

    def should_capture(self, layer: int, component: str) -> bool:
        """Check if we should capture this layer/component."""
        if self.config.capture_layers and layer not in self.config.capture_layers:
            return False
        if component not in self.config.capture_components:
            return False
        return True

    def store(
        self,
        layer: int,
        component: str,
        data: np.ndarray,
        token_range: Tuple[int, int] = (0, 0),
    ) -> bool:
        """Store an activation chunk.

        Args:
            layer: Layer index
            component: Component type
            data: Activation array [batch, seq, hidden]
            token_range: Token indices covered

        Returns:
            True if stored, False if skipped
        """
        if not self.should_capture(layer, component):
            return False

        key = self._key(layer, component)
        path = self._file_path(layer, component)

        # Get current offset
        offset = self._offsets.get(key, 0)

        # Write data
        data_f32 = data.astype(np.float32)
        with open(path, "ab") as f:
            data_f32.tofile(f)

        # Create metadata
        meta = ChunkMetadata(
            layer=layer,
            component=component,
            shape=data_f32.shape,
            dtype="f32",
            offset=offset,
            size_bytes=data_f32.nbytes,
            token_range=token_range,
        )

        # Update index
        if key not in self._metadata:
            self._metadata[key] = []
        self._metadata[key].append(meta)
        self._offsets[key] = offset + data_f32.nbytes

        return True

    def load(
        self,
        layer: int,
        component: str,
        chunk_idx: int = 0,
    ) -> Optional[np.ndarray]:
        """Load an activation chunk.

        Args:
            layer: Layer index
            component: Component type
            chunk_idx: Chunk index to load

        Returns:
            Activation array or None if not found
        """
        key = self._key(layer, component)

        if key not in self._metadata:
            return None

        chunks = self._metadata[key]
        if chunk_idx >= len(chunks):
            return None

        meta = chunks[chunk_idx]
        path = self._file_path(layer, component)

        if not path.exists():
            return None

        if self.config.use_mmap:
            # Memory-mapped access
            mmap = np.memmap(path, dtype=np.float32, mode="r")
            offset_idx = meta.offset // 4  # f32 = 4 bytes
            count = meta.size_bytes // 4
            data = np.array(mmap[offset_idx:offset_idx + count])
        else:
            # Direct read
            with open(path, "rb") as f:
                f.seek(meta.offset)
                data = np.fromfile(f, dtype=np.float32, count=meta.size_bytes // 4)

        return data.reshape(meta.shape)

    def iter_chunks(
        self,
        layer: int,
        component: str,
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """Iterate over all chunks for a layer/component.

        Yields:
            Tuples of (chunk_idx, activation_array)
        """
        key = self._key(layer, component)

        if key not in self._metadata:
            return

        for idx in range(len(self._metadata[key])):
            data = self.load(layer, component, idx)
            if data is not None:
                yield idx, data

    def get_metadata(self, layer: int, component: str) -> List[ChunkMetadata]:
        """Get metadata for a layer/component."""
        key = self._key(layer, component)
        return self._metadata.get(key, [])

    def available_layers(self) -> List[int]:
        """Get list of available layers."""
        layers = set()
        for key in self._metadata.keys():
            layer = int(key.split("_")[0])
            layers.add(layer)
        return sorted(layers)

    def total_size_bytes(self) -> int:
        """Get total stored size in bytes."""
        return sum(
            meta.size_bytes
            for chunks in self._metadata.values()
            for meta in chunks
        )

    def flush(self):
        """Flush metadata to disk."""
        self._save_metadata()

    def cleanup(self):
        """Remove all stored data."""
        self._metadata.clear()
        self._offsets.clear()

        # Remove files
        for f in self.storage_dir.glob("*.bin"):
            f.unlink()

        # Remove metadata
        meta_path = self._metadata_path()
        if meta_path.exists():
            meta_path.unlink()


class RingBuffer:
    """Ring buffer for streaming activation analysis."""

    def __init__(self, capacity: int, layer: int, component: str):
        """Initialize ring buffer.

        Args:
            capacity: Maximum number of items
            layer: Layer this buffer is for
            component: Component this buffer is for
        """
        self.capacity = capacity
        self.layer = layer
        self.component = component
        self._buffer: List[Optional[np.ndarray]] = [None] * capacity
        self._write_pos = 0
        self._count = 0

    def push(self, data: np.ndarray):
        """Push an activation to the buffer."""
        self._buffer[self._write_pos] = data.copy()
        self._write_pos = (self._write_pos + 1) % self.capacity
        self._count = min(self._count + 1, self.capacity)

    def recent(self, n: int = 1) -> List[np.ndarray]:
        """Get the most recent n activations."""
        n = min(n, self._count)
        result = []

        for i in range(n):
            idx = (self._write_pos - 1 - i) % self.capacity
            if self._buffer[idx] is not None:
                result.append(self._buffer[idx])

        return result

    def all(self) -> List[np.ndarray]:
        """Get all activations in order (oldest first)."""
        if self._count == 0:
            return []

        if self._count < self.capacity:
            start = 0
        else:
            start = self._write_pos

        result = []
        for i in range(self._count):
            idx = (start + i) % self.capacity
            if self._buffer[idx] is not None:
                result.append(self._buffer[idx])

        return result

    def __len__(self) -> int:
        return self._count

    def is_empty(self) -> bool:
        return self._count == 0

    def clear(self):
        """Clear the buffer."""
        self._buffer = [None] * self.capacity
        self._write_pos = 0
        self._count = 0


class StreamingTrace:
    """Activation trace backed by disk storage.

    Supports lazy loading of activations to minimize memory usage.
    """

    def __init__(self, storage: ActivationStorage):
        """Initialize streaming trace.

        Args:
            storage: Backend storage
        """
        self._storage = storage
        self._ring_buffers: Dict[str, RingBuffer] = {}
        self._input_tokens: List[int] = []

    def add_activation(
        self,
        layer: int,
        component: str,
        data: np.ndarray,
        token_range: Tuple[int, int] = (0, 0),
    ):
        """Add an activation.

        Args:
            layer: Layer index
            component: Component type
            data: Activation array
            token_range: Token indices covered
        """
        # Store to disk
        self._storage.store(layer, component, data, token_range)

        # Also add to ring buffer if configured
        key = f"{layer}_{component}"
        if key in self._ring_buffers:
            self._ring_buffers[key].push(data)

    def enable_ring_buffer(self, layer: int, component: str, capacity: int):
        """Enable ring buffer for a layer/component.

        Args:
            layer: Layer index
            component: Component type
            capacity: Buffer capacity
        """
        key = f"{layer}_{component}"
        self._ring_buffers[key] = RingBuffer(capacity, layer, component)

    def get(
        self,
        layer: int,
        component: str,
        chunk_idx: int = 0,
    ) -> Optional[np.ndarray]:
        """Get activation (lazy load from disk).

        Args:
            layer: Layer index
            component: Component type
            chunk_idx: Chunk index

        Returns:
            Activation array or None
        """
        return self._storage.load(layer, component, chunk_idx)

    def get_recent(
        self,
        layer: int,
        component: str,
        n: int = 1,
    ) -> List[np.ndarray]:
        """Get recent activations from ring buffer.

        Args:
            layer: Layer index
            component: Component type
            n: Number of recent items

        Returns:
            List of activations (most recent first)
        """
        key = f"{layer}_{component}"
        buffer = self._ring_buffers.get(key)
        if buffer is None:
            return []
        return buffer.recent(n)

    def iter_layer(
        self,
        layer: int,
        component: str = "residual",
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """Iterate over all chunks for a layer.

        Yields:
            Tuples of (chunk_idx, activation)
        """
        yield from self._storage.iter_chunks(layer, component)

    def residual(self, layer: int, chunk_idx: int = 0) -> Optional[np.ndarray]:
        """Get residual stream activation."""
        return self.get(layer, "residual", chunk_idx)

    def attention_out(self, layer: int, chunk_idx: int = 0) -> Optional[np.ndarray]:
        """Get attention output."""
        return self.get(layer, "attn_out", chunk_idx)

    def mlp_out(self, layer: int, chunk_idx: int = 0) -> Optional[np.ndarray]:
        """Get MLP output."""
        return self.get(layer, "mlp_out", chunk_idx)

    @property
    def layers(self) -> List[int]:
        """Get available layers."""
        return self._storage.available_layers()

    @property
    def input_tokens(self) -> List[int]:
        """Get input tokens."""
        return self._input_tokens.copy()

    def flush(self):
        """Flush to disk."""
        self._storage.flush()

    def cleanup(self):
        """Clean up all stored data."""
        self._storage.cleanup()
        self._ring_buffers.clear()


class StreamingMicroscope:
    """Microscope with streaming activation capture.

    Use this for large models that don't fit activations in memory.
    """

    def __init__(
        self,
        config: Optional[StreamingConfig] = None,
        architecture: str = "llama",
        num_layers: int = 32,
        num_heads: int = 32,
        hidden_size: int = 4096,
    ):
        """Initialize streaming microscope.

        Args:
            config: Streaming configuration
            architecture: Model architecture
            num_layers: Number of layers
            num_heads: Number of attention heads
            hidden_size: Hidden dimension size
        """
        self.config = config or StreamingConfig()
        self.architecture = architecture
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self._storage: Optional[ActivationStorage] = None
        self._current_trace: Optional[StreamingTrace] = None
        self._model: Any = None
        self._hooks: List[Any] = []

    @classmethod
    def for_model(
        cls,
        model: Any,
        config: Optional[StreamingConfig] = None,
    ) -> "StreamingMicroscope":
        """Create microscope for a specific model.

        Args:
            model: HuggingFace model
            config: Streaming configuration

        Returns:
            Configured StreamingMicroscope
        """
        model_config = model.config

        # Detect architecture
        arch = model_config.model_type.lower()

        # Get dimensions
        num_layers = getattr(model_config, "num_hidden_layers", 32)
        num_heads = getattr(model_config, "num_attention_heads", 32)
        hidden_size = getattr(model_config, "hidden_size", 4096)

        # Create config if not provided
        if config is None:
            strategy = MemoryEstimator.suggest_strategy(
                num_layers, hidden_size
            )
            if strategy == "streaming":
                config = StreamingConfig.for_large_model()
            elif strategy == "selective":
                key_layers = MemoryEstimator.key_layers(num_layers)
                config = StreamingConfig.selective(key_layers)
            else:
                config = StreamingConfig()

        scope = cls(
            config=config,
            architecture=arch,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=hidden_size,
        )
        scope._model = model
        return scope

    def _register_hooks(self) -> None:
        """Register hooks on the model."""
        if self._model is None or self._current_trace is None:
            return

        import torch

        def make_hook(layer_idx: int, component: str):
            def hook(module, input, output):
                if self._current_trace is not None:
                    if isinstance(output, tuple):
                        out = output[0]
                    else:
                        out = output
                    arr = out.detach().cpu().numpy()
                    self._current_trace.add_activation(layer_idx, component, arr)
            return hook

        # Get layers based on architecture
        if hasattr(self._model, "model") and hasattr(self._model.model, "layers"):
            layers = self._model.model.layers
        elif hasattr(self._model, "transformer") and hasattr(self._model.transformer, "h"):
            layers = self._model.transformer.h
        else:
            return

        for i, layer in enumerate(layers):
            if self._storage and not self._storage.should_capture(i, "residual"):
                continue

            # Hook layer output
            handle = layer.register_forward_hook(make_hook(i, "residual"))
            self._hooks.append(handle)

    def _remove_hooks(self) -> None:
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    @contextmanager
    def trace(
        self,
        input_tokens: Optional[List[int]] = None,
        cleanup_on_exit: bool = False,
    ) -> Generator[StreamingTrace, None, None]:
        """Context manager for streaming trace.

        Args:
            input_tokens: Input token IDs
            cleanup_on_exit: Whether to clean up storage on exit

        Yields:
            StreamingTrace instance
        """
        self._storage = ActivationStorage(self.config)
        self._current_trace = StreamingTrace(self._storage)

        if input_tokens:
            self._current_trace._input_tokens = input_tokens

        self._register_hooks()

        try:
            yield self._current_trace
        finally:
            self._remove_hooks()
            self._current_trace.flush()

            if cleanup_on_exit:
                self._current_trace.cleanup()

            self._current_trace = None
            self._storage = None


class MemoryEstimator:
    """Utilities for estimating memory requirements."""

    @staticmethod
    def estimate_full_capture(
        num_layers: int,
        hidden_size: int,
        batch_size: int = 1,
        seq_len: int = 1024,
    ) -> int:
        """Estimate memory for capturing all activations.

        Args:
            num_layers: Number of layers
            hidden_size: Hidden dimension
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Estimated bytes needed
        """
        # Per component: batch * seq * hidden * 4 (f32)
        per_component = batch_size * seq_len * hidden_size * 4
        per_layer = per_component * 3  # residual, attn_out, mlp_out
        return num_layers * per_layer

    @staticmethod
    def suggest_strategy(
        num_layers: int,
        hidden_size: int,
        memory_limit_gb: float = 4.0,
    ) -> str:
        """Suggest capture strategy based on model size.

        Args:
            num_layers: Number of layers
            hidden_size: Hidden dimension
            memory_limit_gb: Memory limit in GB

        Returns:
            Strategy: "in_memory", "selective", or "streaming"
        """
        memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        full_mem = MemoryEstimator.estimate_full_capture(num_layers, hidden_size)

        if full_mem < memory_limit_bytes:
            return "in_memory"
        elif full_mem < memory_limit_bytes * 4:
            return "selective"
        else:
            return "streaming"

    @staticmethod
    def key_layers(num_layers: int) -> List[int]:
        """Get key layers to capture.

        Returns first, middle, and last layers.
        """
        layers = [0, 1]

        mid = num_layers // 2
        layers.extend([mid - 1, mid, mid + 1])

        layers.extend([num_layers - 2, num_layers - 1])

        return [l for l in layers if 0 <= l < num_layers]
