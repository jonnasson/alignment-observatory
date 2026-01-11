"""
Indirect Object Identification (IOI) circuit schemas.
"""

from typing import Literal

from pydantic import BaseModel, Field


class IOISentence(BaseModel):
    """Parsed IOI sentence with identified roles."""

    text: str
    subject_name: str
    indirect_object_name: str
    subject_token_idx: int
    indirect_object_token_idx: int
    final_token_idx: int
    template: str | None = Field(
        default=None, description="Template pattern if recognized"
    )


class IOIHead(BaseModel):
    """An attention head involved in IOI."""

    layer: int
    head: int
    role: Literal[
        "name_mover",
        "negative_name_mover",
        "s_inhibition",
        "duplicate_token",
        "induction",
        "backup_name_mover",
    ]
    importance: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)


class IOICircuit(BaseModel):
    """Complete IOI circuit analysis."""

    sentence: IOISentence
    name_mover_heads: list[IOIHead]
    negative_name_mover_heads: list[IOIHead]
    s_inhibition_heads: list[IOIHead]
    duplicate_token_heads: list[IOIHead]
    induction_heads: list[IOIHead]
    backup_name_mover_heads: list[IOIHead]
    validity_score: float = Field(ge=0, le=1, description="How well circuit matches expected pattern")
    prediction_correct: bool
    predicted_token: str
    correct_token: str


class IOIDetectionConfig(BaseModel):
    """Configuration for IOI detection."""

    use_patching: bool = Field(default=True, description="Use activation patching")
    head_threshold: float = Field(
        default=0.05, ge=0, le=1, description="Min importance to include head"
    )
    validate_against_known: bool = Field(
        default=True, description="Compare against known GPT-2 IOI heads"
    )


class ParseIOISentenceRequest(BaseModel):
    """Request to parse an IOI-format sentence."""

    text: str = Field(
        description="Sentence in IOI format, e.g., 'When Mary and John went to the store, John gave a drink to'"
    )


class DetectIOIRequest(BaseModel):
    """Request to detect IOI circuit."""

    trace_id: str
    sentence: IOISentence | None = Field(
        default=None, description="Pre-parsed sentence, or will be auto-detected"
    )
    config: IOIDetectionConfig = Field(default_factory=IOIDetectionConfig)


class DetectIOIResponse(BaseModel):
    """Response containing IOI circuit analysis."""

    trace_id: str
    circuit: IOICircuit
    known_heads_matched: int | None = Field(
        default=None, description="Number of known IOI heads found"
    )
    computation_time_ms: float


class KnownIOIHeads(BaseModel):
    """Known IOI heads for GPT-2 Small (for validation)."""

    name_movers: list[tuple[int, int]] = [
        (9, 9),
        (10, 0),
        (9, 6),
    ]
    negative_name_movers: list[tuple[int, int]] = [
        (10, 7),
        (11, 10),
    ]
    s_inhibition: list[tuple[int, int]] = [
        (7, 3),
        (7, 9),
        (8, 6),
        (8, 10),
    ]
    duplicate_token: list[tuple[int, int]] = [
        (0, 1),
        (0, 10),
        (3, 0),
    ]
    induction: list[tuple[int, int]] = [
        (5, 5),
        (5, 8),
        (5, 9),
        (6, 9),
    ]
