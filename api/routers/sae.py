"""
Sparse Autoencoder (SAE) endpoints.
"""

from fastapi import APIRouter, Depends, status

from api.schemas import (
    ApiResponse,
    EncodeRequest,
    EncodeResponse,
    LoadSAERequest,
    SAEConfig,
)
from api.schemas.sae import FeatureCoactivation, FeatureInfo
from api.services.sae_service import SAEService, get_sae_service

router = APIRouter()


@router.post(
    "/load",
    response_model=ApiResponse[SAEConfig],
    status_code=status.HTTP_200_OK,
    summary="Load an SAE",
    description="Load a Sparse Autoencoder for a specific layer.",
)
async def load_sae(
    request: LoadSAERequest,
    sae_service: SAEService = Depends(get_sae_service),
) -> ApiResponse[SAEConfig]:
    """Load an SAE model."""
    config = await sae_service.load_sae(request)
    return ApiResponse(data=config)


@router.post(
    "/encode",
    response_model=ApiResponse[EncodeResponse],
    summary="Encode activations",
    description="Encode activations through the SAE to get feature activations.",
)
async def encode_activations(
    request: EncodeRequest,
    sae_service: SAEService = Depends(get_sae_service),
) -> ApiResponse[EncodeResponse]:
    """Encode activations through the SAE."""
    response = await sae_service.encode(request)
    return ApiResponse(data=response)


@router.get(
    "/features/{layer}",
    response_model=ApiResponse[list[FeatureInfo]],
    summary="Get feature information",
)
async def get_features(
    layer: int,
    top_k: int = 100,
    sae_service: SAEService = Depends(get_sae_service),
) -> ApiResponse[list[FeatureInfo]]:
    """Get information about top SAE features."""
    features = await sae_service.get_top_features(layer, top_k)
    return ApiResponse(data=features)


@router.get(
    "/features/{layer}/{feature_idx}",
    response_model=ApiResponse[FeatureInfo],
    summary="Get single feature info",
)
async def get_feature(
    layer: int,
    feature_idx: int,
    sae_service: SAEService = Depends(get_sae_service),
) -> ApiResponse[FeatureInfo]:
    """Get information about a specific SAE feature."""
    feature = await sae_service.get_feature_info(layer, feature_idx)
    return ApiResponse(data=feature)


@router.get(
    "/coactivations/{layer}",
    response_model=ApiResponse[list[FeatureCoactivation]],
    summary="Get feature co-activations",
)
async def get_coactivations(
    layer: int,
    trace_id: str,
    top_k: int = 50,
    sae_service: SAEService = Depends(get_sae_service),
) -> ApiResponse[list[FeatureCoactivation]]:
    """Get feature co-activation patterns for a trace."""
    coactivations = await sae_service.get_coactivations(trace_id, layer, top_k)
    return ApiResponse(data=coactivations)


@router.get(
    "/available",
    response_model=ApiResponse[list[dict]],
    summary="List available SAEs",
)
async def list_available_saes(
    sae_service: SAEService = Depends(get_sae_service),
) -> ApiResponse[list[dict]]:
    """List available pre-trained SAEs."""
    saes = await sae_service.list_available()
    return ApiResponse(data=saes)
