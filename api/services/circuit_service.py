"""
Circuit discovery service.
"""

import time
from typing import Any

from api.schemas import (
    Circuit,
    CircuitDiscoveryRequest,
    CircuitDiscoveryResponse,
    CircuitEdge,
    CircuitNode,
)


class CircuitService:
    """Service for circuit discovery operations."""

    def __init__(self) -> None:
        pass

    async def discover_circuit(
        self,
        request: CircuitDiscoveryRequest,
    ) -> CircuitDiscoveryResponse:
        """
        Discover a circuit for a given trace.

        This is a placeholder implementation that returns a mock circuit.
        In production, this would use activation patching or edge attribution
        from the alignment_microscope package.
        """
        start_time = time.perf_counter()

        # TODO: Implement actual circuit discovery using:
        # - Activation patching
        # - Edge attribution
        # - Causal tracing

        # For now, return a mock circuit structure
        nodes = [
            CircuitNode(
                id="embed",
                layer=0,
                component="embed",
                label="Embedding",
                importance=0.8,
            ),
            CircuitNode(
                id="attn_0_1",
                layer=0,
                component="attention",
                head=1,
                label="L0H1",
                importance=0.6,
            ),
            CircuitNode(
                id="mlp_0",
                layer=0,
                component="mlp",
                label="MLP 0",
                importance=0.4,
            ),
            CircuitNode(
                id="attn_1_0",
                layer=1,
                component="attention",
                head=0,
                label="L1H0",
                importance=0.7,
            ),
            CircuitNode(
                id="unembed",
                layer=2,
                component="unembed",
                label="Unembedding",
                importance=0.9,
            ),
        ]

        edges = [
            CircuitEdge(source="embed", target="attn_0_1", importance=0.7),
            CircuitEdge(source="attn_0_1", target="mlp_0", importance=0.5),
            CircuitEdge(source="mlp_0", target="attn_1_0", importance=0.6),
            CircuitEdge(source="attn_1_0", target="unembed", importance=0.8),
            CircuitEdge(source="embed", target="unembed", importance=0.3, edge_type="residual"),
        ]

        circuit = Circuit(
            name=f"Circuit for token {request.target_token_idx}",
            description="Discovered circuit explaining model prediction",
            nodes=nodes,
            edges=edges,
            total_importance=0.85,
            num_layers=2,
            num_components=5,
        )

        computation_time = (time.perf_counter() - start_time) * 1000

        return CircuitDiscoveryResponse(
            trace_id=request.trace_id,
            target_token=f"[token_{request.target_token_idx}]",
            target_token_idx=request.target_token_idx,
            circuit=circuit,
            method_used=request.params.method,
            computation_time_ms=computation_time,
        )


def get_circuit_service() -> CircuitService:
    """Dependency injection for CircuitService."""
    return CircuitService()
