"""
Trace management endpoints.

Handles creating traces from live inference and loading pre-computed traces.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status

from api.schemas import (
    ActivationRequest,
    ActivationResponse,
    ApiResponse,
    AttentionRequest,
    AttentionResponse,
    CreateTraceRequest,
    LoadTraceRequest,
    TraceInfo,
)
from api.schemas.common import ErrorCode
from api.services.trace_service import TraceService, get_trace_service, get_trace_service_with_model

router = APIRouter()


@router.post(
    "",
    response_model=ApiResponse[TraceInfo],
    status_code=status.HTTP_201_CREATED,
    summary="Create a new trace",
    description="Run inference on input text and capture activations/attention patterns.",
)
async def create_trace(
    request: CreateTraceRequest,
    http_request: Request,
) -> ApiResponse[TraceInfo]:
    """Create a new trace from text input via live inference."""
    trace_service = get_trace_service_with_model(http_request)
    try:
        trace_info = await trace_service.create_trace(request)
        return ApiResponse(data=trace_info)
    except RuntimeError as e:
        if "model not loaded" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "MODEL_NOT_LOADED",
                    "message": "No model is currently loaded. Please load a model first using the /api/v1/models/load endpoint.",
                },
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "INFERENCE_ERROR", "message": str(e)},
        )


@router.post(
    "/load",
    response_model=ApiResponse[TraceInfo],
    summary="Load a pre-computed trace",
    description="Load a trace that was previously saved to disk.",
)
async def load_trace(
    request: LoadTraceRequest,
    trace_service: TraceService = Depends(get_trace_service),
) -> ApiResponse[TraceInfo]:
    """Load a pre-computed trace from disk."""
    trace_info = await trace_service.load_trace(request)
    return ApiResponse(data=trace_info)


@router.get(
    "/{trace_id}",
    response_model=ApiResponse[TraceInfo],
    summary="Get trace information",
)
async def get_trace(
    trace_id: str,
    trace_service: TraceService = Depends(get_trace_service),
) -> ApiResponse[TraceInfo]:
    """Get information about a specific trace."""
    trace_info = await trace_service.get_trace_info(trace_id)
    if trace_info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": ErrorCode.TRACE_NOT_FOUND, "message": f"Trace {trace_id} not found"},
        )
    return ApiResponse(data=trace_info)


@router.delete(
    "/{trace_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a trace",
)
async def delete_trace(
    trace_id: str,
    trace_service: TraceService = Depends(get_trace_service),
) -> None:
    """Delete a trace from memory."""
    await trace_service.delete_trace(trace_id)


@router.get(
    "/{trace_id}/attention/{layer}",
    response_model=ApiResponse[AttentionResponse],
    summary="Get attention patterns",
)
async def get_attention(
    trace_id: str,
    layer: int,
    head: int | None = None,
    aggregate: str = "none",
    trace_service: TraceService = Depends(get_trace_service),
) -> ApiResponse[AttentionResponse]:
    """Get attention patterns for a specific layer and optionally head."""
    request = AttentionRequest(layer=layer, head=head, aggregate=aggregate)  # type: ignore
    response = await trace_service.get_attention(trace_id, request)
    return ApiResponse(data=response)


@router.get(
    "/{trace_id}/activations/{layer}/{component}",
    response_model=ApiResponse[ActivationResponse],
    summary="Get activations",
)
async def get_activations(
    trace_id: str,
    layer: int,
    component: str,
    token_indices: str | None = None,
    trace_service: TraceService = Depends(get_trace_service),
) -> ApiResponse[ActivationResponse]:
    """Get activations for a specific layer and component."""
    indices = None
    if token_indices:
        indices = [int(i) for i in token_indices.split(",")]

    request = ActivationRequest(layer=layer, component=component, token_indices=indices)  # type: ignore
    response = await trace_service.get_activations(trace_id, request)
    return ApiResponse(data=response)


@router.get(
    "",
    response_model=ApiResponse[list[TraceInfo]],
    summary="List all traces",
)
async def list_traces(
    trace_service: TraceService = Depends(get_trace_service),
) -> ApiResponse[list[TraceInfo]]:
    """List all traces currently in memory."""
    traces = await trace_service.list_traces()
    return ApiResponse(data=traces)
