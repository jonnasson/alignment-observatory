"""Tests for Sparse Autoencoder (SAE) module."""

import pytest
import numpy as np

from alignment_microscope.sae import (
    SAEConfig,
    SAEFeatures,
    SAEWrapper,
    SAEAnalyzer,
)


class TestSAEConfig:
    """Tests for SAEConfig."""

    def test_create_config(self):
        """Test creating an SAE config."""
        config = SAEConfig(d_in=768, d_sae=16384)
        assert config.d_in == 768
        assert config.d_sae == 16384
        assert config.activation == "relu"

    def test_config_with_topk(self):
        """Test config with topk activation."""
        config = SAEConfig(d_in=512, d_sae=8192, activation="topk", k=32)
        assert config.activation == "topk"
        assert config.k == 32

    def test_config_with_hook_point(self):
        """Test config with hook point."""
        config = SAEConfig(
            d_in=768,
            d_sae=16384,
            hook_point="blocks.0.hook_resid_post",
            layer=0,
        )
        assert config.hook_point == "blocks.0.hook_resid_post"
        assert config.layer == 0


class TestSAEFeatures:
    """Tests for SAEFeatures."""

    def test_create_features(self):
        """Test creating SAE features."""
        activations = np.random.randn(10, 256).astype(np.float32)
        activations = np.maximum(activations, 0)  # ReLU-like
        features = SAEFeatures(activations)
        assert features.shape == (10, 256)
        assert features.d_sae == 256

    def test_sparsity(self):
        """Test sparsity computation."""
        # Create sparse activations
        activations = np.zeros((10, 100), dtype=np.float32)
        activations[0, 0] = 1.0  # Only one active
        features = SAEFeatures(activations)
        assert features.sparsity > 0.99  # 999/1000 are zero

    def test_active_features(self):
        """Test finding active features."""
        activations = np.zeros((5, 10), dtype=np.float32)
        activations[0, 0] = 1.0
        activations[0, 3] = 0.5
        activations[1, 5] = 2.0

        features = SAEFeatures(activations)
        active = features.active_features(threshold=0.0)

        assert len(active) == 5  # 5 positions
        assert 0 in active[0]
        assert 3 in active[0]
        assert 5 in active[1]

    def test_top_k_features(self):
        """Test getting top-k features."""
        activations = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ], dtype=np.float32)

        features = SAEFeatures(activations)
        top3 = features.top_k_features(k=3)

        assert len(top3) == 2
        # First position: top 3 are indices 4, 3, 2
        assert top3[0][0][0] == 4
        assert top3[0][1][0] == 3
        assert top3[0][2][0] == 2
        # Second position: top 3 are indices 0, 1, 2
        assert top3[1][0][0] == 0
        assert top3[1][1][0] == 1
        assert top3[1][2][0] == 2

    def test_feature_frequency(self):
        """Test feature frequency computation."""
        activations = np.array([
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
        ], dtype=np.float32)

        features = SAEFeatures(activations)
        freq = features.feature_frequency()

        # Feature 0 active in 4/4 positions
        assert freq[0] == 1.0
        # Feature 1 active in 1/4 positions
        assert freq[1] == 0.25
        # Feature 2 active in 2/4 positions
        assert freq[2] == 0.5

    def test_to_dict(self):
        """Test export to dictionary."""
        activations = np.ones((2, 4), dtype=np.float32)
        features = SAEFeatures(activations)
        d = features.to_dict()

        assert "activations" in d
        assert "shape" in d
        assert "sparsity" in d
        assert d["shape"] == [2, 4]


class TestSAEWrapper:
    """Tests for SAEWrapper."""

    @pytest.fixture
    def simple_sae(self):
        """Create a simple SAE for testing."""
        d_in = 64
        d_sae = 128

        # Simple random weights
        np.random.seed(42)
        w_enc = np.random.randn(d_in, d_sae).astype(np.float32) * 0.1
        w_dec = np.random.randn(d_sae, d_in).astype(np.float32) * 0.1
        b_enc = np.zeros(d_sae, dtype=np.float32)
        b_dec = np.zeros(d_in, dtype=np.float32)

        config = SAEConfig(d_in=d_in, d_sae=d_sae)
        return SAEWrapper(w_enc, w_dec, b_enc, b_dec, config)

    def test_create_sae(self, simple_sae):
        """Test creating an SAE wrapper."""
        assert simple_sae.d_in == 64
        assert simple_sae.d_sae == 128

    def test_encode(self, simple_sae):
        """Test encoding activations."""
        activations = np.random.randn(4, 64).astype(np.float32)
        features = simple_sae.encode(activations)

        assert features.shape == (4, 128)
        # ReLU should produce non-negative values
        assert np.all(features.activations >= 0)

    def test_encode_batch_seq(self, simple_sae):
        """Test encoding with batch and sequence dimensions."""
        activations = np.random.randn(2, 8, 64).astype(np.float32)
        features = simple_sae.encode(activations)

        assert features.shape == (2, 8, 128)

    def test_decode(self, simple_sae):
        """Test decoding features back to activations."""
        activations = np.random.randn(4, 64).astype(np.float32)
        features = simple_sae.encode(activations)
        reconstructed = simple_sae.decode(features)

        assert reconstructed.shape == activations.shape

    def test_reconstruction_error(self, simple_sae):
        """Test reconstruction error computation."""
        activations = np.random.randn(4, 64).astype(np.float32)
        error = simple_sae.reconstruction_error(activations)

        assert error >= 0  # MSE is non-negative

    def test_topk_activation(self):
        """Test SAE with top-k activation."""
        d_in = 32
        d_sae = 64
        k = 5

        np.random.seed(42)
        w_enc = np.random.randn(d_in, d_sae).astype(np.float32) * 0.1
        w_dec = np.random.randn(d_sae, d_in).astype(np.float32) * 0.1

        config = SAEConfig(d_in=d_in, d_sae=d_sae, activation="topk", k=k)
        sae = SAEWrapper(w_enc, w_dec, config=config)

        activations = np.random.randn(2, d_in).astype(np.float32)
        features = sae.encode(activations)

        # Each position should have at most k active features
        for pos in range(features.shape[0]):
            active = np.sum(features.activations[pos] > 0)
            assert active <= k

    def test_to_dict(self, simple_sae):
        """Test export to dictionary."""
        d = simple_sae.to_dict()

        assert "w_enc" in d
        assert "w_dec" in d
        assert "config" in d
        assert d["config"]["d_in"] == 64
        assert d["config"]["d_sae"] == 128

    def test_from_dict(self, simple_sae):
        """Test loading from dictionary."""
        d = simple_sae.to_dict()
        loaded = SAEWrapper.from_dict(d)

        assert loaded.d_in == simple_sae.d_in
        assert loaded.d_sae == simple_sae.d_sae
        np.testing.assert_array_almost_equal(loaded.w_enc, simple_sae.w_enc)


class TestSAEAnalyzer:
    """Tests for SAEAnalyzer."""

    @pytest.fixture
    def analyzer_with_sae(self):
        """Create an analyzer with a registered SAE."""
        analyzer = SAEAnalyzer()

        d_in = 32
        d_sae = 64
        np.random.seed(42)
        w_enc = np.random.randn(d_in, d_sae).astype(np.float32) * 0.1
        w_dec = np.random.randn(d_sae, d_in).astype(np.float32) * 0.1

        config = SAEConfig(d_in=d_in, d_sae=d_sae)
        sae = SAEWrapper(w_enc, w_dec, config=config)

        analyzer.register_sae("layer_0", sae)
        return analyzer

    def test_register_sae(self, analyzer_with_sae):
        """Test registering an SAE."""
        sae = analyzer_with_sae.get_sae("layer_0")
        assert sae is not None
        assert sae.d_sae == 64

    def test_get_nonexistent_sae(self, analyzer_with_sae):
        """Test getting non-existent SAE returns None."""
        assert analyzer_with_sae.get_sae("layer_99") is None

    def test_analyze_activations(self, analyzer_with_sae):
        """Test analyzing activations."""
        activations = {
            "layer_0": np.random.randn(4, 32).astype(np.float32),
        }

        results = analyzer_with_sae.analyze_activations(activations)

        assert "layer_0" in results
        assert results["layer_0"].shape == (4, 64)

    def test_find_behavior_features(self, analyzer_with_sae):
        """Test finding behavior-specific features."""
        # Create clean features with feature 0 active
        clean_arr = np.zeros((4, 64), dtype=np.float32)
        clean_arr[:, 0] = 1.0
        clean_features = SAEFeatures(clean_arr)

        # Create corrupt features with feature 1 active
        corrupt_arr = np.zeros((4, 64), dtype=np.float32)
        corrupt_arr[:, 1] = 1.0
        corrupt_features = SAEFeatures(corrupt_arr)

        diff = analyzer_with_sae.find_behavior_features(
            clean_features, corrupt_features, threshold=0.5
        )

        assert "activated" in diff
        assert "deactivated" in diff
        assert 0 in diff["activated"]  # More active in clean
        assert 1 in diff["deactivated"]  # Less active in clean

    def test_feature_coactivation(self, analyzer_with_sae):
        """Test feature co-activation matrix."""
        # Features 0 and 1 always co-activate
        arr = np.zeros((10, 64), dtype=np.float32)
        for i in range(10):
            arr[i, 0] = 1.0
            arr[i, 1] = 0.5

        features = SAEFeatures(arr)
        coact = analyzer_with_sae.feature_coactivation(features, top_k=5)

        assert coact.shape == (64, 64)
        # Features 0 and 1 should have high co-activation
        assert coact[0, 1] > 0
        assert coact[1, 0] > 0


class TestSAEIntegration:
    """Integration tests for SAE module."""

    def test_full_encode_decode_pipeline(self):
        """Test full encode-decode pipeline."""
        # Create SAE
        d_in = 128
        d_sae = 512

        np.random.seed(42)
        # Initialize weights with orthogonal initialization for better reconstruction
        w_enc = np.random.randn(d_in, d_sae).astype(np.float32) * 0.1
        # Make decoder roughly inverse of encoder
        w_dec = w_enc.T.copy()

        sae = SAEWrapper(w_enc, w_dec)

        # Create activations
        activations = np.random.randn(8, 16, d_in).astype(np.float32)

        # Encode
        features = sae.encode(activations)

        # Decode
        reconstructed = sae.decode(features)

        # Should have same shape
        assert reconstructed.shape == activations.shape

        # Reconstruction error should be finite
        error = sae.reconstruction_error(activations, features)
        assert np.isfinite(error)

    def test_analyzer_with_trace(self):
        """Test analyzer with simulated trace."""
        from alignment_microscope import ActivationTrace

        # Create trace with simulated activations
        trace = ActivationTrace()

        # Add activations
        import torch
        layer0_act = torch.randn(1, 10, 64)
        layer1_act = torch.randn(1, 10, 64)
        trace.add_activation(0, "residual", layer0_act)
        trace.add_activation(1, "residual", layer1_act)

        # Create analyzer with SAEs
        analyzer = SAEAnalyzer()

        for layer in [0, 1]:
            np.random.seed(42 + layer)
            w_enc = np.random.randn(64, 128).astype(np.float32) * 0.1
            w_dec = np.random.randn(128, 64).astype(np.float32) * 0.1
            config = SAEConfig(d_in=64, d_sae=128, layer=layer)
            sae = SAEWrapper(w_enc, w_dec, config=config)
            analyzer.register_sae(f"layer_{layer}", sae)

        # Analyze
        activations = {
            f"layer_{layer}": trace.get(layer, "residual")
            for layer in trace.layers
        }

        results = analyzer.analyze_activations(activations)

        assert "layer_0" in results
        assert "layer_1" in results
        assert results["layer_0"].shape == (1, 10, 128)
