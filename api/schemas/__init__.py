"""
Pydantic schemas for API request/response validation.
"""

from api.schemas.circuit import (
    Circuit,
    CircuitDiscoveryParams,
    CircuitDiscoveryRequest,
    CircuitDiscoveryResponse,
    CircuitEdge,
    CircuitNode,
)
from api.schemas.common import (
    ApiError,
    ApiResponse,
    PaginatedResponse,
    TensorData,
    TensorStats,
)
from api.schemas.ioi import (
    DetectIOIRequest,
    DetectIOIResponse,
    IOICircuit,
    IOIHead,
    IOISentence,
    ParseIOISentenceRequest,
)
from api.schemas.model import (
    LoadModelRequest,
    MemoryEstimate,
    ModelInfo,
    ModelsListResponse,
)
from api.schemas.sae import (
    EncodeRequest,
    EncodeResponse,
    FeatureActivation,
    LoadSAERequest,
    SAEConfig,
    SAEFeatures,
)
from api.schemas.trace import (
    ActivationRequest,
    ActivationResponse,
    AttentionPattern,
    AttentionRequest,
    AttentionResponse,
    CreateTraceRequest,
    HeadAnalysis,
    LayerActivation,
    LoadTraceRequest,
    TraceInfo,
    TraceMetadata,
)

__all__ = [
    # Common
    "ApiError",
    "ApiResponse",
    "PaginatedResponse",
    "TensorData",
    "TensorStats",
    # Trace
    "CreateTraceRequest",
    "LoadTraceRequest",
    "TraceInfo",
    "TraceMetadata",
    "AttentionRequest",
    "AttentionResponse",
    "AttentionPattern",
    "HeadAnalysis",
    "ActivationRequest",
    "ActivationResponse",
    "LayerActivation",
    # Circuit
    "Circuit",
    "CircuitNode",
    "CircuitEdge",
    "CircuitDiscoveryParams",
    "CircuitDiscoveryRequest",
    "CircuitDiscoveryResponse",
    # IOI
    "IOISentence",
    "IOIHead",
    "IOICircuit",
    "ParseIOISentenceRequest",
    "DetectIOIRequest",
    "DetectIOIResponse",
    # SAE
    "SAEConfig",
    "SAEFeatures",
    "FeatureActivation",
    "LoadSAERequest",
    "EncodeRequest",
    "EncodeResponse",
    # Model
    "ModelInfo",
    "ModelsListResponse",
    "LoadModelRequest",
    "MemoryEstimate",
]
