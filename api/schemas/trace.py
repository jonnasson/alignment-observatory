"""
Trace-related schemas for activation tracing and attention analysis.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from api.schemas.common import TensorData, TensorStats


class CreateTraceRequest(BaseModel):
    """Request to create a new trace from text input."""

    text: str = Field(min_length=1, max_length=10000, description="Input text to trace")
    model_name: str | None = Field(
        default=None, description="Model name (uses default if not specified)"
    )
    include_attention: bool = Field(default=True, description="Include attention patterns")
    include_activations: bool = Field(default=True, description="Include activations")
    layers: list[int] | None = Field(
        default=None, description="Specific layers to trace (all if not specified)"
    )


class LoadTraceRequest(BaseModel):
    """Request to load a pre-computed trace from disk."""

    path: str = Field(description="Path to the saved trace file")


class TraceMetadata(BaseModel):
    """Metadata about a trace."""

    model_name: str
    num_layers: int
    num_heads: int
    hidden_size: int
    seq_length: int
    vocab_size: int


class TraceInfo(BaseModel):
    """Information about a created/loaded trace."""

    trace_id: str
    created_at: datetime
    input_text: str
    tokens: list[str]
    token_ids: list[int]
    metadata: TraceMetadata
    has_attention: bool
    has_activations: bool
    layers_available: list[int]


class HeadClassification(BaseModel):
    """Classification of an attention head's behavior."""

    category: Literal[
        "induction",
        "previous_token",
        "duplicate_token",
        "positional",
        "semantic",
        "mixed",
        "unknown",
    ]
    confidence: float = Field(ge=0, le=1)
    description: str | None = None


class HeadAnalysis(BaseModel):
    """Analysis results for a single attention head."""

    layer: int
    head: int
    classification: HeadClassification
    entropy: float
    max_attention: float
    sparsity: float = Field(description="Fraction of near-zero attention weights")


class AttentionPattern(BaseModel):
    """Attention pattern data for visualization."""

    layer: int
    head: int | None = Field(default=None, description="None means all heads")
    pattern: TensorData = Field(description="Attention weights [heads?, seq_q, seq_k]")
    tokens: list[str]
    stats: TensorStats


class AttentionRequest(BaseModel):
    """Request for attention pattern data."""

    layer: int
    head: int | None = Field(default=None, description="Specific head or all heads")
    aggregate: Literal["none", "mean", "max"] = Field(
        default="none", description="How to aggregate across heads"
    )


class AttentionResponse(BaseModel):
    """Response containing attention data."""

    trace_id: str
    pattern: AttentionPattern
    analysis: list[HeadAnalysis] | None = None


class LayerActivation(BaseModel):
    """Activation data for a single layer."""

    layer: int
    component: Literal["residual", "attention_out", "mlp_out", "mlp_hidden"]
    activations: TensorData
    stats: TensorStats


class ActivationRequest(BaseModel):
    """Request for activation data."""

    layer: int
    component: Literal["residual", "attention_out", "mlp_out", "mlp_hidden"] = Field(
        default="residual"
    )
    token_indices: list[int] | None = Field(
        default=None, description="Specific tokens to return"
    )


class ActivationResponse(BaseModel):
    """Response containing activation data."""

    trace_id: str
    activation: LayerActivation
    token_norms: list[float] | None = Field(
        default=None, description="L2 norms per token position"
    )
