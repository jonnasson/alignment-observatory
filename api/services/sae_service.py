"""
Sparse Autoencoder (SAE) service.
"""

import time
from typing import Any

import numpy as np

from api.schemas import (
    EncodeRequest,
    EncodeResponse,
    FeatureActivation,
    LoadSAERequest,
    SAEConfig,
    SAEFeatures,
)
from api.schemas.common import TensorData
from api.schemas.sae import FeatureCoactivation, FeatureInfo


class SAEService:
    """Service for SAE operations."""

    def __init__(self) -> None:
        self._loaded_saes: dict[int, dict[str, Any]] = {}

    async def load_sae(self, request: LoadSAERequest) -> SAEConfig:
        """
        Load an SAE model.

        This is a placeholder. In production, this would load actual SAE weights
        from disk or a model registry.
        """
        # Create mock SAE config
        config = SAEConfig(
            name=request.sae_name or f"sae_layer_{request.layer}",
            hidden_size=768,  # GPT-2 small
            sae_size=32768,  # Typical SAE size
            layer=request.layer,
            hook_point="resid_post",
            l1_coefficient=0.001,
            trained_on="OpenWebText",
        )

        self._loaded_saes[request.layer] = {
            "config": config,
            "weights": None,  # Would store actual weights
        }

        return config

    async def encode(self, request: EncodeRequest) -> EncodeResponse:
        """
        Encode activations through the SAE.

        This is a placeholder that generates mock feature activations.
        In production, this would run the actual SAE encoder.
        """
        start_time = time.perf_counter()

        # TODO: Get actual activations from trace and encode

        # Mock data
        seq_len = 20  # Would get from trace
        sae_size = 32768
        top_k = request.top_k

        # Generate sparse activations (mock)
        num_active = int(sae_size * 0.01)  # 1% sparsity
        active_features = sorted(np.random.choice(sae_size, num_active, replace=False).tolist())

        # Generate top-k features per token
        top_features_per_token = []
        for pos in range(seq_len):
            features = []
            for _ in range(top_k):
                feat_idx = int(np.random.choice(active_features))
                features.append(FeatureActivation(
                    feature_idx=feat_idx,
                    activation=float(np.random.exponential(1.0)),
                    token_idx=pos,
                    token=f"token_{pos}",
                ))
            features.sort(key=lambda x: x.activation, reverse=True)
            top_features_per_token.append(features[:top_k])

        # Create sparse activation tensor (mock)
        activations = TensorData(
            data=[0.0] * (seq_len * sae_size),  # Would be actual sparse values
            shape=[seq_len, sae_size],
            dtype="float32",
        )

        config = SAEConfig(
            name=f"sae_layer_{request.layer}",
            hidden_size=768,
            sae_size=sae_size,
            layer=request.layer,
            hook_point="resid_post",
        )

        features = SAEFeatures(
            config=config,
            activations=activations,
            top_features_per_token=top_features_per_token,
            active_features=active_features,
            sparsity=len(active_features) / sae_size,
        )

        computation_time = (time.perf_counter() - start_time) * 1000

        return EncodeResponse(
            trace_id=request.trace_id,
            layer=request.layer,
            features=features,
            reconstruction_loss=0.05,  # Mock value
            computation_time_ms=computation_time,
        )

    async def get_top_features(self, layer: int, top_k: int) -> list[FeatureInfo]:
        """Get information about top SAE features."""
        # Mock feature info
        features = []
        for i in range(top_k):
            features.append(FeatureInfo(
                feature_idx=i,
                max_activation=float(np.random.exponential(2.0)),
                mean_activation=float(np.random.exponential(0.5)),
                frequency=float(np.random.uniform(0.01, 0.1)),
                top_tokens=[
                    (f"token_{j}", float(np.random.exponential(1.0)))
                    for j in range(5)
                ],
                description=None,
            ))
        return features

    async def get_feature_info(self, layer: int, feature_idx: int) -> FeatureInfo:
        """Get information about a specific feature."""
        return FeatureInfo(
            feature_idx=feature_idx,
            max_activation=float(np.random.exponential(2.0)),
            mean_activation=float(np.random.exponential(0.5)),
            frequency=float(np.random.uniform(0.01, 0.1)),
            top_tokens=[
                (f"token_{j}", float(np.random.exponential(1.0)))
                for j in range(10)
            ],
            description=f"Feature {feature_idx} at layer {layer}",
        )

    async def get_coactivations(
        self,
        trace_id: str,
        layer: int,
        top_k: int,
    ) -> list[FeatureCoactivation]:
        """Get feature co-activation patterns."""
        # Mock co-activations
        coactivations = []
        for i in range(top_k):
            coactivations.append(FeatureCoactivation(
                feature_a=i,
                feature_b=i + 1,
                coactivation_count=int(np.random.randint(1, 100)),
                correlation=float(np.random.uniform(0.3, 0.9)),
            ))
        return coactivations

    async def list_available(self) -> list[dict]:
        """List available pre-trained SAEs."""
        return [
            {
                "name": "gpt2-small-resid-post",
                "model": "gpt2",
                "layers": list(range(12)),
                "sae_size": 32768,
                "source": "SAELens",
            },
            {
                "name": "gpt2-small-mlp-out",
                "model": "gpt2",
                "layers": list(range(12)),
                "sae_size": 32768,
                "source": "SAELens",
            },
        ]


def get_sae_service() -> SAEService:
    """Dependency injection for SAEService."""
    return SAEService()
