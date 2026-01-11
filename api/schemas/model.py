"""
Model management schemas.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Information about an available model."""

    name: str
    display_name: str
    num_layers: int
    num_heads: int
    hidden_size: int
    vocab_size: int
    max_seq_length: int
    parameter_count: int
    is_loaded: bool = False
    loaded_at: datetime | None = None


class ModelsListResponse(BaseModel):
    """Response listing available models."""

    models: list[ModelInfo]
    current_model: str | None = Field(
        default=None, description="Currently loaded model name"
    )


class MemoryEstimate(BaseModel):
    """Memory requirements estimate for a model."""

    model_name: str
    parameters_mb: float
    activations_mb_per_token: float
    estimated_total_mb: float
    gpu_available_mb: float | None = None
    fits_in_memory: bool


class LoadModelRequest(BaseModel):
    """Request to load a model."""

    model_name: str = Field(description="Model name or HuggingFace path")
    device: str | None = Field(
        default=None, description="Device to load on (auto-detect if not specified)"
    )
    dtype: str | None = Field(
        default=None, description="Data type (float32, float16, bfloat16)"
    )
    force_reload: bool = Field(
        default=False, description="Reload even if already loaded"
    )


class UnloadModelRequest(BaseModel):
    """Request to unload current model."""

    clear_cache: bool = Field(default=True, description="Clear model cache")
