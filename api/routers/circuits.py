"""
Circuit discovery endpoints.
"""

from fastapi import APIRouter, Depends, status

from api.schemas import (
    ApiResponse,
    CircuitDiscoveryRequest,
    CircuitDiscoveryResponse,
)
from api.services.circuit_service import CircuitService, get_circuit_service

router = APIRouter()


@router.post(
    "/discover",
    response_model=ApiResponse[CircuitDiscoveryResponse],
    status_code=status.HTTP_200_OK,
    summary="Discover a circuit",
    description="Discover the circuit responsible for a specific prediction using activation patching or edge attribution.",
)
async def discover_circuit(
    request: CircuitDiscoveryRequest,
    circuit_service: CircuitService = Depends(get_circuit_service),
) -> ApiResponse[CircuitDiscoveryResponse]:
    """Discover a circuit for a given trace and target token."""
    response = await circuit_service.discover_circuit(request)
    return ApiResponse(data=response)


@router.get(
    "/methods",
    response_model=ApiResponse[list[dict]],
    summary="List available discovery methods",
)
async def list_methods() -> ApiResponse[list[dict]]:
    """List available circuit discovery methods."""
    methods = [
        {
            "name": "activation_patching",
            "display_name": "Activation Patching",
            "description": "Patch activations from clean to corrupted input to measure importance",
            "requires_clean_input": True,
        },
        {
            "name": "edge_attribution",
            "display_name": "Edge Attribution",
            "description": "Compute attribution scores for edges between components",
            "requires_clean_input": False,
        },
        {
            "name": "causal_tracing",
            "display_name": "Causal Tracing",
            "description": "Restore activations in corrupted input to find causal components",
            "requires_clean_input": True,
        },
    ]
    return ApiResponse(data=methods)
