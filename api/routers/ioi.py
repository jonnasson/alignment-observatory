"""
Indirect Object Identification (IOI) circuit detection endpoints.
"""

from fastapi import APIRouter, Depends, status

from api.schemas import (
    ApiResponse,
    DetectIOIRequest,
    DetectIOIResponse,
    IOISentence,
    ParseIOISentenceRequest,
)
from api.schemas.ioi import KnownIOIHeads
from api.services.ioi_service import IOIService, get_ioi_service

router = APIRouter()


@router.post(
    "/parse",
    response_model=ApiResponse[IOISentence],
    summary="Parse an IOI sentence",
    description="Parse a sentence to identify subject, indirect object, and key positions.",
)
async def parse_sentence(
    request: ParseIOISentenceRequest,
    ioi_service: IOIService = Depends(get_ioi_service),
) -> ApiResponse[IOISentence]:
    """Parse a sentence to identify IOI-relevant components."""
    sentence = await ioi_service.parse_sentence(request.text)
    return ApiResponse(data=sentence)


@router.post(
    "/detect",
    response_model=ApiResponse[DetectIOIResponse],
    status_code=status.HTTP_200_OK,
    summary="Detect IOI circuit",
    description="Analyze a trace to detect the IOI circuit components.",
)
async def detect_ioi_circuit(
    request: DetectIOIRequest,
    ioi_service: IOIService = Depends(get_ioi_service),
) -> ApiResponse[DetectIOIResponse]:
    """Detect IOI circuit in a trace."""
    response = await ioi_service.detect_circuit(request)
    return ApiResponse(data=response)


@router.get(
    "/known-heads",
    response_model=ApiResponse[KnownIOIHeads],
    summary="Get known IOI heads",
    description="Get the known IOI heads for GPT-2 Small for reference.",
)
async def get_known_heads() -> ApiResponse[KnownIOIHeads]:
    """Get known IOI heads from the literature."""
    return ApiResponse(data=KnownIOIHeads())


@router.get(
    "/templates",
    response_model=ApiResponse[list[dict]],
    summary="Get IOI sentence templates",
)
async def get_templates() -> ApiResponse[list[dict]]:
    """Get example IOI sentence templates."""
    templates = [
        {
            "name": "ABBA",
            "template": "When {A} and {B} went to the {place}, {B} gave a {object} to",
            "expected_completion": "{A}",
            "example": "When Mary and John went to the store, John gave a drink to",
        },
        {
            "name": "BABA",
            "template": "When {B} and {A} went to the {place}, {B} gave a {object} to",
            "expected_completion": "{A}",
            "example": "When John and Mary went to the store, John gave a drink to",
        },
        {
            "name": "Simple",
            "template": "{A} gave the {object} to {B}. {B} gave the {object} to",
            "expected_completion": "{A}",
            "example": "Mary gave the ball to John. John gave the ball to",
        },
    ]
    return ApiResponse(data=templates)
