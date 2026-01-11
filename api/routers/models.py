"""
Model management endpoints.
"""

from fastapi import APIRouter, Depends, Request, status

from api.schemas import (
    ApiResponse,
    LoadModelRequest,
    MemoryEstimate,
    ModelInfo,
    ModelsListResponse,
)
from api.services.model_manager import ModelManager

router = APIRouter()


def get_model_manager(request: Request) -> ModelManager:
    """Get model manager from app state."""
    return request.app.state.model_manager


@router.get(
    "",
    response_model=ApiResponse[ModelsListResponse],
    summary="List available models",
)
async def list_models(
    model_manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse[ModelsListResponse]:
    """List all available models and their status."""
    response = await model_manager.list_models()
    return ApiResponse(data=response)


@router.post(
    "/load",
    response_model=ApiResponse[ModelInfo],
    status_code=status.HTTP_200_OK,
    summary="Load a model",
)
async def load_model(
    request: LoadModelRequest,
    model_manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse[ModelInfo]:
    """Load a model into memory for inference."""
    model_info = await model_manager.load_model(request)
    return ApiResponse(data=model_info)


@router.post(
    "/unload",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Unload current model",
)
async def unload_model(
    model_manager: ModelManager = Depends(get_model_manager),
) -> None:
    """Unload the currently loaded model."""
    await model_manager.unload_model()


@router.get(
    "/current",
    response_model=ApiResponse[ModelInfo | None],
    summary="Get current model",
)
async def get_current_model(
    model_manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse[ModelInfo | None]:
    """Get information about the currently loaded model."""
    model_info = await model_manager.get_current_model()
    return ApiResponse(data=model_info)


@router.get(
    "/estimate/{model_name}",
    response_model=ApiResponse[MemoryEstimate],
    summary="Estimate memory requirements",
)
async def estimate_memory(
    model_name: str,
    model_manager: ModelManager = Depends(get_model_manager),
) -> ApiResponse[MemoryEstimate]:
    """Estimate memory requirements for a model."""
    estimate = await model_manager.estimate_memory(model_name)
    return ApiResponse(data=estimate)
