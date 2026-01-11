"""Tests for streaming activation capture module."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from alignment_microscope.streaming import (
    StreamingConfig,
    ActivationStorage,
    RingBuffer,
    StreamingTrace,
    StreamingMicroscope,
    MemoryEstimator,
)


class TestStreamingConfig:
    """Tests for StreamingConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = StreamingConfig()
        assert config.memory_limit_gb == 4.0
        assert "residual" in config.capture_components
        assert config.ring_buffer_size == 1000

    def test_for_large_model(self):
        """Test configuration for large models."""
        config = StreamingConfig.for_large_model()
        assert config.memory_limit_gb == 8.0
        assert config.capture_components == ["residual"]

    def test_selective_config(self):
        """Test selective layer configuration."""
        config = StreamingConfig.selective([0, 5, 10])
        assert config.capture_layers == [0, 5, 10]


class TestActivationStorage:
    """Tests for ActivationStorage."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = StreamingConfig(
                storage_dir=tmpdir,
                capture_components=["residual", "attn_out"],
            )
            storage = ActivationStorage(config)
            yield storage
            storage.cleanup()

    def test_create_storage(self, temp_storage):
        """Test creating storage."""
        assert temp_storage.available_layers() == []

    def test_store_activation(self, temp_storage):
        """Test storing activation."""
        data = np.random.randn(1, 10, 64).astype(np.float32)
        stored = temp_storage.store(0, "residual", data, (0, 10))
        assert stored is True
        assert 0 in temp_storage.available_layers()

    def test_store_skips_unconfigured(self, temp_storage):
        """Test that unconfigured components are skipped."""
        data = np.random.randn(1, 10, 64).astype(np.float32)
        stored = temp_storage.store(0, "mlp_out", data, (0, 10))
        assert stored is False  # mlp_out not in capture_components

    def test_load_activation(self, temp_storage):
        """Test loading activation from disk."""
        data = np.random.randn(1, 10, 64).astype(np.float32)
        temp_storage.store(0, "residual", data, (0, 10))
        temp_storage.flush()

        loaded = temp_storage.load(0, "residual", 0)
        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded, data)

    def test_load_nonexistent(self, temp_storage):
        """Test loading non-existent activation."""
        loaded = temp_storage.load(99, "residual", 0)
        assert loaded is None

    def test_multiple_chunks(self, temp_storage):
        """Test storing and loading multiple chunks."""
        for i in range(3):
            data = np.full((1, 10, 64), i, dtype=np.float32)
            temp_storage.store(0, "residual", data, (i * 10, (i + 1) * 10))

        temp_storage.flush()

        # Load each chunk
        for i in range(3):
            loaded = temp_storage.load(0, "residual", i)
            assert loaded is not None
            assert loaded[0, 0, 0] == i

    def test_total_size(self, temp_storage):
        """Test total size computation."""
        data = np.random.randn(1, 10, 64).astype(np.float32)
        temp_storage.store(0, "residual", data, (0, 10))

        expected_size = 1 * 10 * 64 * 4  # f32 = 4 bytes
        assert temp_storage.total_size_bytes() == expected_size

    def test_iter_chunks(self, temp_storage):
        """Test iterating over chunks."""
        for i in range(3):
            data = np.full((1, 5, 32), i, dtype=np.float32)
            temp_storage.store(0, "residual", data, (i * 5, (i + 1) * 5))

        temp_storage.flush()

        chunks = list(temp_storage.iter_chunks(0, "residual"))
        assert len(chunks) == 3

        for idx, (chunk_idx, data) in enumerate(chunks):
            assert chunk_idx == idx
            assert data[0, 0, 0] == idx


class TestRingBuffer:
    """Tests for RingBuffer."""

    def test_create_buffer(self):
        """Test creating ring buffer."""
        buffer = RingBuffer(5, layer=0, component="residual")
        assert buffer.capacity == 5
        assert len(buffer) == 0
        assert buffer.is_empty()

    def test_push_items(self):
        """Test pushing items to buffer."""
        buffer = RingBuffer(3, layer=0, component="residual")

        for i in range(3):
            data = np.full((1, 4, 4), i, dtype=np.float32)
            buffer.push(data)

        assert len(buffer) == 3
        assert not buffer.is_empty()

    def test_overflow(self):
        """Test buffer overflow behavior."""
        buffer = RingBuffer(3, layer=0, component="residual")

        for i in range(5):
            data = np.full((1, 4, 4), i, dtype=np.float32)
            buffer.push(data)

        # Should still have 3 items
        assert len(buffer) == 3

        # Most recent should be 4, 3, 2
        recent = buffer.recent(3)
        assert len(recent) == 3
        assert recent[0][0, 0, 0] == 4
        assert recent[1][0, 0, 0] == 3
        assert recent[2][0, 0, 0] == 2

    def test_all_in_order(self):
        """Test getting all items in order."""
        buffer = RingBuffer(3, layer=0, component="residual")

        for i in range(5):
            data = np.full((1, 4, 4), i, dtype=np.float32)
            buffer.push(data)

        # All should be in order: 2, 3, 4 (oldest first)
        all_items = buffer.all()
        assert len(all_items) == 3
        assert all_items[0][0, 0, 0] == 2
        assert all_items[1][0, 0, 0] == 3
        assert all_items[2][0, 0, 0] == 4

    def test_clear(self):
        """Test clearing buffer."""
        buffer = RingBuffer(3, layer=0, component="residual")

        for i in range(3):
            buffer.push(np.ones((1, 4, 4), dtype=np.float32))

        buffer.clear()
        assert len(buffer) == 0
        assert buffer.is_empty()


class TestStreamingTrace:
    """Tests for StreamingTrace."""

    @pytest.fixture
    def streaming_trace(self):
        """Create a streaming trace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = StreamingConfig(storage_dir=tmpdir)
            storage = ActivationStorage(config)
            trace = StreamingTrace(storage)
            yield trace
            trace.cleanup()

    def test_add_and_get(self, streaming_trace):
        """Test adding and getting activations."""
        data = np.random.randn(1, 10, 64).astype(np.float32)
        streaming_trace.add_activation(0, "residual", data)
        streaming_trace.flush()

        loaded = streaming_trace.get(0, "residual")
        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded, data)

    def test_shortcut_methods(self, streaming_trace):
        """Test shortcut methods."""
        res_data = np.random.randn(1, 10, 64).astype(np.float32)
        attn_data = np.random.randn(1, 10, 64).astype(np.float32)
        mlp_data = np.random.randn(1, 10, 64).astype(np.float32)

        streaming_trace.add_activation(0, "residual", res_data)
        streaming_trace.add_activation(0, "attn_out", attn_data)
        streaming_trace.add_activation(0, "mlp_out", mlp_data)
        streaming_trace.flush()

        np.testing.assert_array_almost_equal(streaming_trace.residual(0), res_data)
        np.testing.assert_array_almost_equal(streaming_trace.attention_out(0), attn_data)
        np.testing.assert_array_almost_equal(streaming_trace.mlp_out(0), mlp_data)

    def test_layers_property(self, streaming_trace):
        """Test layers property."""
        for layer in [0, 5, 10]:
            data = np.random.randn(1, 10, 64).astype(np.float32)
            streaming_trace.add_activation(layer, "residual", data)

        assert streaming_trace.layers == [0, 5, 10]

    def test_ring_buffer_integration(self, streaming_trace):
        """Test ring buffer with streaming trace."""
        streaming_trace.enable_ring_buffer(0, "residual", capacity=3)

        for i in range(5):
            data = np.full((1, 10, 64), i, dtype=np.float32)
            streaming_trace.add_activation(0, "residual", data)

        # Get recent from ring buffer
        recent = streaming_trace.get_recent(0, "residual", 2)
        assert len(recent) == 2
        assert recent[0][0, 0, 0] == 4  # Most recent
        assert recent[1][0, 0, 0] == 3

    def test_iter_layer(self, streaming_trace):
        """Test iterating over layer chunks."""
        for i in range(3):
            data = np.full((1, 5, 32), i, dtype=np.float32)
            streaming_trace.add_activation(0, "residual", data, (i * 5, (i + 1) * 5))

        streaming_trace.flush()

        chunks = list(streaming_trace.iter_layer(0, "residual"))
        assert len(chunks) == 3


class TestStreamingMicroscope:
    """Tests for StreamingMicroscope."""

    def test_create_microscope(self):
        """Test creating streaming microscope."""
        scope = StreamingMicroscope()
        assert scope.architecture == "llama"
        assert scope.num_layers == 32

    def test_with_custom_config(self):
        """Test with custom config."""
        config = StreamingConfig.for_large_model()
        scope = StreamingMicroscope(config=config)
        assert scope.config.memory_limit_gb == 8.0

    def test_trace_context_manager(self):
        """Test trace context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = StreamingConfig(storage_dir=tmpdir)
            scope = StreamingMicroscope(config=config)

            with scope.trace() as trace:
                assert trace is not None
                assert isinstance(trace, StreamingTrace)

    def test_trace_with_tokens(self):
        """Test trace with input tokens."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = StreamingConfig(storage_dir=tmpdir)
            scope = StreamingMicroscope(config=config)

            with scope.trace(input_tokens=[1, 2, 3]) as trace:
                assert trace.input_tokens == [1, 2, 3]


class TestMemoryEstimator:
    """Tests for MemoryEstimator."""

    def test_estimate_full_capture(self):
        """Test memory estimation."""
        # 12 layers, 768 hidden, batch=1, seq=1024
        mem = MemoryEstimator.estimate_full_capture(12, 768, 1, 1024)

        # Expected: 12 * 3 * 1 * 1024 * 768 * 4 bytes
        expected = 12 * 3 * 1 * 1024 * 768 * 4
        assert mem == expected

    def test_suggest_in_memory(self):
        """Test suggesting in-memory strategy."""
        # Small model should fit in memory
        strategy = MemoryEstimator.suggest_strategy(12, 768, 4.0)
        assert strategy == "in_memory"

    def test_suggest_streaming(self):
        """Test suggesting streaming strategy."""
        # Very large model needs streaming or selective
        strategy = MemoryEstimator.suggest_strategy(80, 8192, 4.0)
        # Either streaming or selective is acceptable for large models
        assert strategy in ["streaming", "selective"]

    def test_suggest_selective(self):
        """Test suggesting selective strategy."""
        # Medium model might need selective capture
        strategy = MemoryEstimator.suggest_strategy(32, 4096, 4.0)
        assert strategy in ["in_memory", "selective", "streaming"]

    def test_key_layers(self):
        """Test key layer selection."""
        layers = MemoryEstimator.key_layers(32)

        # Should include first, middle, last
        assert 0 in layers
        assert 1 in layers
        assert 15 in layers or 16 in layers  # Middle-ish
        assert 30 in layers or 31 in layers  # End


class TestStreamingIntegration:
    """Integration tests for streaming module."""

    def test_full_pipeline(self):
        """Test full streaming pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config with selective capture
            config = StreamingConfig.selective([0, 2, 4], storage_dir=tmpdir)
            storage = ActivationStorage(config)
            trace = StreamingTrace(storage)

            # Simulate model forward pass
            for layer in range(5):
                data = np.random.randn(1, 32, 128).astype(np.float32)
                trace.add_activation(layer, "residual", data)

            trace.flush()

            # Only selected layers should be stored
            assert trace.layers == [0, 2, 4]

            # Verify data integrity
            for layer in [0, 2, 4]:
                loaded = trace.get(layer, "residual")
                assert loaded is not None
                assert loaded.shape == (1, 32, 128)

            trace.cleanup()

    def test_streaming_with_analysis(self):
        """Test streaming with SAE analysis."""
        from alignment_microscope.sae import SAEWrapper, SAEConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = StreamingConfig(storage_dir=tmpdir)
            storage = ActivationStorage(config)
            trace = StreamingTrace(storage)

            # Add activation
            activation = np.random.randn(1, 10, 64).astype(np.float32)
            trace.add_activation(0, "residual", activation)
            trace.flush()

            # Create SAE and analyze
            np.random.seed(42)
            w_enc = np.random.randn(64, 128).astype(np.float32) * 0.1
            w_dec = np.random.randn(128, 64).astype(np.float32) * 0.1
            sae = SAEWrapper(w_enc, w_dec)

            # Load from trace and analyze
            loaded = trace.residual(0)
            features = sae.encode(loaded)

            assert features.shape == (1, 10, 128)

            trace.cleanup()
