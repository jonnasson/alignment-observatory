"""
Sparse Autoencoder (SAE) schemas.
"""

from pydantic import BaseModel, Field

from api.schemas.common import TensorData


class SAEConfig(BaseModel):
    """Configuration for a Sparse Autoencoder."""

    name: str
    hidden_size: int = Field(description="Model hidden dimension")
    sae_size: int = Field(description="SAE dictionary size")
    layer: int = Field(description="Which layer this SAE is for")
    hook_point: str = Field(description="Hook point name, e.g., 'resid_post'")
    l1_coefficient: float | None = None
    trained_on: str | None = Field(default=None, description="Training data description")


class FeatureActivation(BaseModel):
    """Activation of a single SAE feature."""

    feature_idx: int
    activation: float
    token_idx: int
    token: str


class SAEFeatures(BaseModel):
    """SAE feature activations for a trace."""

    config: SAEConfig
    activations: TensorData = Field(description="[seq_len, sae_size] sparse activations")
    top_features_per_token: list[list[FeatureActivation]] = Field(
        description="Top-k features for each token position"
    )
    active_features: list[int] = Field(description="Indices of features that activated")
    sparsity: float = Field(description="Fraction of active features")


class FeatureInfo(BaseModel):
    """Information about a single SAE feature."""

    feature_idx: int
    max_activation: float
    mean_activation: float
    frequency: float = Field(description="Fraction of tokens where this feature activates")
    top_tokens: list[tuple[str, float]] = Field(
        description="Top activating tokens with their activation values"
    )
    description: str | None = Field(default=None, description="Human-interpretable description")


class FeatureCoactivation(BaseModel):
    """Co-activation pattern between features."""

    feature_a: int
    feature_b: int
    coactivation_count: int
    correlation: float


class LoadSAERequest(BaseModel):
    """Request to load an SAE."""

    sae_path: str | None = Field(default=None, description="Path to SAE weights")
    sae_name: str | None = Field(
        default=None, description="Name of pre-configured SAE to load"
    )
    layer: int = Field(description="Layer this SAE is for")


class EncodeRequest(BaseModel):
    """Request to encode activations through SAE."""

    trace_id: str
    layer: int
    top_k: int = Field(default=10, ge=1, le=100, description="Top features per token")
    threshold: float = Field(
        default=0.0, ge=0, description="Min activation to include"
    )


class EncodeResponse(BaseModel):
    """Response containing SAE encoding."""

    trace_id: str
    layer: int
    features: SAEFeatures
    reconstruction_loss: float | None = Field(
        default=None, description="MSE between original and reconstructed activations"
    )
    computation_time_ms: float
