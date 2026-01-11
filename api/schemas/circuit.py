"""
Circuit discovery schemas.
"""

from typing import Literal

from pydantic import BaseModel, Field


class CircuitNode(BaseModel):
    """A node in a circuit graph."""

    id: str = Field(description="Unique node identifier")
    layer: int
    component: Literal["attention", "mlp", "residual", "embed", "unembed"]
    head: int | None = Field(default=None, description="Head index for attention nodes")
    label: str | None = None
    importance: float = Field(default=0.0, ge=0, le=1)


class CircuitEdge(BaseModel):
    """An edge in a circuit graph."""

    source: str = Field(description="Source node ID")
    target: str = Field(description="Target node ID")
    importance: float = Field(ge=0, le=1)
    edge_type: Literal["direct", "residual", "virtual"] = "direct"


class Circuit(BaseModel):
    """A discovered circuit."""

    name: str
    description: str | None = None
    nodes: list[CircuitNode]
    edges: list[CircuitEdge]
    total_importance: float
    num_layers: int
    num_components: int


class CircuitDiscoveryParams(BaseModel):
    """Parameters for circuit discovery."""

    method: Literal["activation_patching", "edge_attribution", "causal_tracing"] = (
        Field(default="activation_patching")
    )
    threshold: float = Field(
        default=0.01, ge=0, le=1, description="Importance threshold for inclusion"
    )
    max_nodes: int = Field(default=50, ge=1, le=200)
    include_mlp: bool = Field(default=True)
    include_attention: bool = Field(default=True)
    aggregate_heads: bool = Field(
        default=False, description="Aggregate attention heads per layer"
    )


class CircuitDiscoveryRequest(BaseModel):
    """Request to discover a circuit."""

    trace_id: str = Field(description="ID of the trace to analyze")
    target_token_idx: int = Field(description="Token position to explain")
    params: CircuitDiscoveryParams = Field(default_factory=CircuitDiscoveryParams)
    clean_input: str | None = Field(
        default=None, description="Clean input for activation patching"
    )


class CircuitDiscoveryResponse(BaseModel):
    """Response containing discovered circuit."""

    trace_id: str
    target_token: str
    target_token_idx: int
    circuit: Circuit
    method_used: str
    computation_time_ms: float
