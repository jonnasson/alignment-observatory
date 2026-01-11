"""Sparse Autoencoder (SAE) integration module.

This module provides integration with Sparse Autoencoders for interpretability:

- SAEWrapper: Unified interface for SAE encoding/decoding
- SAEFeatures: Container for SAE feature activations
- SAEAnalyzer: Analysis tools for SAE features
- SAELens integration for loading pre-trained SAEs
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class SAEConfig:
    """Configuration for a Sparse Autoencoder."""

    d_in: int
    """Input dimension (model hidden size)."""

    d_sae: int
    """SAE feature dimension."""

    activation: str = "relu"
    """Activation function: 'relu', 'topk', 'jumprelu'."""

    k: Optional[int] = None
    """Top-k value for topk activation."""

    hook_point: str = ""
    """Hook point name (e.g., 'blocks.0.hook_resid_post')."""

    layer: int = 0
    """Layer index this SAE is associated with."""


class SAEFeatures:
    """Container for SAE feature activations.

    Provides utilities for analyzing which features are active,
    computing sparsity, and finding top-k features.
    """

    def __init__(
        self,
        activations: np.ndarray,
        config: Optional[SAEConfig] = None,
    ):
        """Initialize SAE features.

        Args:
            activations: Feature activations, shape [batch, seq_len, d_sae] or [positions, d_sae]
            config: Optional SAE configuration
        """
        self._activations = activations
        self._config = config
        self._compute_stats()

    def _compute_stats(self):
        """Compute sparsity statistics."""
        total = self._activations.size
        zeros = np.sum(self._activations == 0)
        self._sparsity = zeros / total if total > 0 else 0.0

        # Mean active features per position
        if self._activations.ndim >= 2:
            active_per_pos = np.sum(self._activations > 0, axis=-1)
            self._mean_active = float(np.mean(active_per_pos))
        else:
            self._mean_active = float(np.sum(self._activations > 0))

    @property
    def activations(self) -> np.ndarray:
        """Get raw activations array."""
        return self._activations

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get shape of activations."""
        return self._activations.shape

    @property
    def d_sae(self) -> int:
        """Get SAE feature dimension."""
        return self._activations.shape[-1]

    @property
    def sparsity(self) -> float:
        """Get sparsity ratio (fraction of zero activations)."""
        return self._sparsity

    @property
    def mean_active_features(self) -> float:
        """Get mean number of active features per position."""
        return self._mean_active

    def active_features(self, threshold: float = 0.0) -> List[np.ndarray]:
        """Get indices of active features per position.

        Args:
            threshold: Minimum activation value to be considered active

        Returns:
            List of arrays, each containing active feature indices for a position
        """
        # Reshape to 2D: [positions, d_sae]
        flat = self._activations.reshape(-1, self.d_sae)
        return [np.where(row > threshold)[0] for row in flat]

    def top_k_features(self, k: int = 10) -> List[List[Tuple[int, float]]]:
        """Get top-k features per position with their activation values.

        Args:
            k: Number of top features to return

        Returns:
            List of lists of (feature_idx, activation_value) tuples
        """
        flat = self._activations.reshape(-1, self.d_sae)
        results = []

        for row in flat:
            # Get indices that would sort descending
            sorted_indices = np.argsort(row)[::-1][:k]
            top_k = [(int(idx), float(row[idx])) for idx in sorted_indices if row[idx] > 0]
            results.append(top_k)

        return results

    def feature_frequency(self, threshold: float = 0.0) -> np.ndarray:
        """Compute how often each feature activates across positions.

        Args:
            threshold: Minimum activation to count as active

        Returns:
            Array of shape [d_sae] with activation frequencies
        """
        flat = self._activations.reshape(-1, self.d_sae)
        num_positions = flat.shape[0]
        active_counts = np.sum(flat > threshold, axis=0)
        return active_counts / num_positions

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return {
            "activations": self._activations.tolist(),
            "shape": list(self.shape),
            "sparsity": self._sparsity,
            "mean_active_features": self._mean_active,
        }


class SAEWrapper:
    """Unified interface for Sparse Autoencoders.

    Supports loading SAEs from SAELens or custom weights.
    Provides encode/decode operations and feature analysis.
    """

    def __init__(
        self,
        w_enc: np.ndarray,
        w_dec: np.ndarray,
        b_enc: Optional[np.ndarray] = None,
        b_dec: Optional[np.ndarray] = None,
        config: Optional[SAEConfig] = None,
    ):
        """Initialize SAE wrapper with weights.

        Args:
            w_enc: Encoder weights [d_in, d_sae]
            w_dec: Decoder weights [d_sae, d_in]
            b_enc: Encoder bias [d_sae]
            b_dec: Decoder bias [d_in]
            config: SAE configuration
        """
        self.w_enc = w_enc.astype(np.float32)
        self.w_dec = w_dec.astype(np.float32)
        self.b_enc = b_enc.astype(np.float32) if b_enc is not None else None
        self.b_dec = b_dec.astype(np.float32) if b_dec is not None else None

        if config is None:
            config = SAEConfig(
                d_in=w_enc.shape[0],
                d_sae=w_enc.shape[1],
            )
        self.config = config

    @classmethod
    def from_saelens(
        cls,
        model_name: str,
        hook_point: str,
        layer: Optional[int] = None,
        device: str = "cpu",
    ) -> "SAEWrapper":
        """Load a pre-trained SAE from SAELens.

        Args:
            model_name: Model name in SAELens format (e.g., 'gpt2-small')
            hook_point: Hook point (e.g., 'blocks.0.hook_resid_post')
            layer: Optional layer index to extract from hook_point
            device: Device to load to ('cpu' or 'cuda')

        Returns:
            SAEWrapper instance

        Raises:
            ImportError: If SAELens is not installed
            ValueError: If SAE not found
        """
        try:
            from sae_lens import SAE
        except ImportError:
            raise ImportError(
                "SAELens is required for loading pre-trained SAEs. "
                "Install with: pip install sae-lens"
            )

        # Load SAE from SAELens
        sae = SAE.from_pretrained(
            release=model_name,
            sae_id=hook_point,
            device=device,
        )[0]  # Returns (sae, cfg, sparsity)

        # Extract weights
        w_enc = sae.W_enc.detach().cpu().numpy()
        w_dec = sae.W_dec.detach().cpu().numpy()
        b_enc = sae.b_enc.detach().cpu().numpy() if hasattr(sae, 'b_enc') else None
        b_dec = sae.b_dec.detach().cpu().numpy() if hasattr(sae, 'b_dec') else None

        # Determine layer from hook_point if not provided
        if layer is None and "blocks." in hook_point:
            layer = int(hook_point.split("blocks.")[1].split(".")[0])

        config = SAEConfig(
            d_in=w_enc.shape[0],
            d_sae=w_enc.shape[1],
            activation="relu",  # SAELens default
            hook_point=hook_point,
            layer=layer or 0,
        )

        return cls(w_enc, w_dec, b_enc, b_dec, config)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SAEWrapper":
        """Load SAE from dictionary.

        Args:
            data: Dictionary with weights and config

        Returns:
            SAEWrapper instance
        """
        w_enc = np.array(data["w_enc"])
        w_dec = np.array(data["w_dec"])
        b_enc = np.array(data["b_enc"]) if "b_enc" in data else None
        b_dec = np.array(data["b_dec"]) if "b_dec" in data else None

        config = None
        if "config" in data:
            config = SAEConfig(**data["config"])

        return cls(w_enc, w_dec, b_enc, b_dec, config)

    @property
    def d_in(self) -> int:
        """Get input dimension."""
        return self.config.d_in

    @property
    def d_sae(self) -> int:
        """Get SAE feature dimension."""
        return self.config.d_sae

    def encode(self, activations: np.ndarray) -> SAEFeatures:
        """Encode activations to SAE features.

        Args:
            activations: Input activations [..., d_in]

        Returns:
            SAEFeatures container
        """
        original_shape = activations.shape
        # Flatten to 2D: [positions, d_in]
        flat = activations.reshape(-1, self.d_in).astype(np.float32)

        # Center by decoder bias
        if self.b_dec is not None:
            flat = flat - self.b_dec

        # Encode: x @ W_enc + b_enc
        features = flat @ self.w_enc
        if self.b_enc is not None:
            features = features + self.b_enc

        # Apply activation
        if self.config.activation == "relu":
            features = np.maximum(features, 0)
        elif self.config.activation == "topk" and self.config.k is not None:
            features = self._apply_topk(features, self.config.k)
        elif self.config.activation == "jumprelu":
            features = np.maximum(features, 0)  # Simplified JumpReLU

        # Reshape to match original batch/seq structure
        new_shape = list(original_shape[:-1]) + [self.d_sae]
        features = features.reshape(new_shape)

        return SAEFeatures(features, self.config)

    def _apply_topk(self, features: np.ndarray, k: int) -> np.ndarray:
        """Apply top-k sparsification.

        Args:
            features: Feature activations [positions, d_sae]
            k: Number of features to keep per position

        Returns:
            Sparsified features
        """
        result = np.zeros_like(features)

        for i in range(features.shape[0]):
            top_indices = np.argsort(features[i])[-k:]
            for idx in top_indices:
                if features[i, idx] > 0:
                    result[i, idx] = features[i, idx]

        return result

    def decode(self, features: Union[SAEFeatures, np.ndarray]) -> np.ndarray:
        """Decode SAE features back to activations.

        Args:
            features: SAE features [..., d_sae]

        Returns:
            Reconstructed activations [..., d_in]
        """
        if isinstance(features, SAEFeatures):
            feat_arr = features.activations
        else:
            feat_arr = features

        original_shape = feat_arr.shape
        flat = feat_arr.reshape(-1, self.d_sae).astype(np.float32)

        # Decode: features @ W_dec + b_dec
        reconstructed = flat @ self.w_dec
        if self.b_dec is not None:
            reconstructed = reconstructed + self.b_dec

        # Reshape
        new_shape = list(original_shape[:-1]) + [self.d_in]
        return reconstructed.reshape(new_shape)

    def reconstruction_error(
        self,
        activations: np.ndarray,
        features: Optional[SAEFeatures] = None,
    ) -> float:
        """Compute mean squared reconstruction error.

        Args:
            activations: Original activations
            features: Optional pre-computed features

        Returns:
            MSE reconstruction error
        """
        if features is None:
            features = self.encode(activations)

        reconstructed = self.decode(features)
        diff = activations - reconstructed
        return float(np.mean(diff ** 2))

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        result = {
            "w_enc": self.w_enc.tolist(),
            "w_dec": self.w_dec.tolist(),
            "config": {
                "d_in": self.config.d_in,
                "d_sae": self.config.d_sae,
                "activation": self.config.activation,
                "k": self.config.k,
                "hook_point": self.config.hook_point,
                "layer": self.config.layer,
            },
        }
        if self.b_enc is not None:
            result["b_enc"] = self.b_enc.tolist()
        if self.b_dec is not None:
            result["b_dec"] = self.b_dec.tolist()
        return result


class SAEAnalyzer:
    """Analyzer for SAE features across traces.

    Provides tools for:
    - Feature activation analysis
    - Cross-layer feature tracking
    - Behavior-specific feature identification
    """

    def __init__(self):
        """Initialize analyzer."""
        self._saes: Dict[str, SAEWrapper] = {}

    def register_sae(self, name: str, sae: SAEWrapper):
        """Register an SAE for analysis.

        Args:
            name: Unique name for this SAE (e.g., 'layer_0_resid')
            sae: SAE wrapper instance
        """
        self._saes[name] = sae

    def get_sae(self, name: str) -> Optional[SAEWrapper]:
        """Get a registered SAE.

        Args:
            name: SAE name

        Returns:
            SAE wrapper or None if not found
        """
        return self._saes.get(name)

    def analyze_activations(
        self,
        activations: Dict[str, np.ndarray],
    ) -> Dict[str, SAEFeatures]:
        """Analyze activations using registered SAEs.

        Args:
            activations: Dict mapping names to activation arrays

        Returns:
            Dict mapping names to SAEFeatures
        """
        results = {}

        for name, sae in self._saes.items():
            if name in activations:
                results[name] = sae.encode(activations[name])

        return results

    def find_behavior_features(
        self,
        clean_features: SAEFeatures,
        corrupt_features: SAEFeatures,
        threshold: float = 0.1,
    ) -> Dict[str, List[int]]:
        """Find features that differ between clean and corrupt traces.

        Args:
            clean_features: Features from clean run
            corrupt_features: Features from corrupted run
            threshold: Minimum difference to consider significant

        Returns:
            Dict with 'activated' and 'deactivated' feature lists
        """
        clean_arr = clean_features.activations.reshape(-1, clean_features.d_sae)
        corrupt_arr = corrupt_features.activations.reshape(-1, corrupt_features.d_sae)

        # Compare mean activation per feature
        clean_mean = np.mean(clean_arr, axis=0)
        corrupt_mean = np.mean(corrupt_arr, axis=0)
        diff = clean_mean - corrupt_mean

        activated = np.where(diff > threshold)[0].tolist()
        deactivated = np.where(diff < -threshold)[0].tolist()

        return {
            "activated": activated,  # More active in clean
            "deactivated": deactivated,  # Less active in clean
        }

    def feature_coactivation(
        self,
        features: SAEFeatures,
        top_k: int = 10,
    ) -> np.ndarray:
        """Compute feature co-activation matrix.

        Args:
            features: SAE features
            top_k: Consider only top-k features per position

        Returns:
            Co-activation matrix [d_sae, d_sae]
        """
        d_sae = features.d_sae
        coact = np.zeros((d_sae, d_sae), dtype=np.float32)

        top_features = features.top_k_features(top_k)

        for position_features in top_features:
            indices = [idx for idx, _ in position_features]
            for i in indices:
                for j in indices:
                    coact[i, j] += 1

        # Normalize
        num_positions = len(top_features)
        if num_positions > 0:
            coact /= num_positions

        return coact
