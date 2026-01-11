"""
Common schema definitions shared across the API.
"""

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class TensorStats(BaseModel):
    """Statistics for a tensor."""

    min: float
    max: float
    mean: float
    std: float
    shape: list[int]
    dtype: str


class TensorData(BaseModel):
    """Serialized tensor data with metadata."""

    data: list[Any] = Field(description="Flattened tensor data")
    shape: list[int] = Field(description="Original tensor shape")
    dtype: str = Field(default="float32", description="Data type")
    stats: TensorStats | None = None


class ApiError(BaseModel):
    """API error response."""

    code: str = Field(description="Error code for programmatic handling")
    message: str = Field(description="Human-readable error message")
    details: dict[str, Any] | None = Field(
        default=None, description="Additional error context"
    )


class ApiResponse(BaseModel, Generic[T]):
    """Standard API response wrapper."""

    success: bool = True
    data: T | None = None
    error: ApiError | None = None


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response for list endpoints."""

    items: list[T]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool


class ErrorCode:
    """Error code constants."""

    TRACE_NOT_FOUND = "TRACE_NOT_FOUND"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    INVALID_INPUT = "INVALID_INPUT"
    LAYER_OUT_OF_RANGE = "LAYER_OUT_OF_RANGE"
    HEAD_OUT_OF_RANGE = "HEAD_OUT_OF_RANGE"
    SAE_NOT_LOADED = "SAE_NOT_LOADED"
    CIRCUIT_DISCOVERY_FAILED = "CIRCUIT_DISCOVERY_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    WEBSOCKET_ERROR = "WEBSOCKET_ERROR"


class WSMessageType:
    """WebSocket message type constants."""

    # Client -> Server
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    CREATE_TRACE = "create_trace"
    CANCEL = "cancel"

    # Server -> Client
    PROGRESS = "progress"
    ACTIVATION_DATA = "activation_data"
    ATTENTION_DATA = "attention_data"
    COMPLETE = "complete"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class ProgressData(BaseModel):
    """Progress update for long-running operations."""

    operation: str
    current: int
    total: int
    message: str | None = None
    percent: float = Field(ge=0, le=100)


class WSServerMessage(BaseModel):
    """WebSocket message from server to client."""

    type: str
    trace_id: str | None = None
    data: dict[str, Any] | None = None
    error: ApiError | None = None
