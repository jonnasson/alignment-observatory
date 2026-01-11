"""
Indirect Object Identification (IOI) circuit detection service.
"""

import re
import time
from typing import Any

from api.schemas import (
    DetectIOIRequest,
    DetectIOIResponse,
    IOICircuit,
    IOIHead,
    IOISentence,
)


class IOIService:
    """Service for IOI circuit detection."""

    def __init__(self) -> None:
        # Known IOI heads for GPT-2 Small
        self._known_heads = {
            "name_movers": [(9, 9), (10, 0), (9, 6)],
            "negative_name_movers": [(10, 7), (11, 10)],
            "s_inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
            "duplicate_token": [(0, 1), (0, 10), (3, 0)],
            "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
        }

    async def parse_sentence(self, text: str) -> IOISentence:
        """
        Parse a sentence to identify IOI components.

        Looks for patterns like:
        - "When {A} and {B} went to the {place}, {B} gave a {object} to"
        - "{A} gave the {object} to {B}. {B} gave the {object} to"
        """
        # Common patterns
        patterns = [
            # ABBA pattern
            r"When\s+(\w+)\s+and\s+(\w+)\s+went\s+to\s+the\s+\w+,\s+\2\s+gave\s+(?:a|the)\s+\w+\s+to",
            # BABA pattern
            r"When\s+(\w+)\s+and\s+(\w+)\s+went\s+to\s+the\s+\w+,\s+\1\s+gave\s+(?:a|the)\s+\w+\s+to",
            # Simple pattern
            r"(\w+)\s+gave\s+(?:a|the)\s+\w+\s+to\s+(\w+)\.\s+\2\s+gave\s+(?:a|the)\s+\w+\s+to",
        ]

        template = None
        subject_name = ""
        indirect_object_name = ""

        for i, pattern in enumerate(patterns):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if i == 0:  # ABBA
                    subject_name = match.group(1)
                    indirect_object_name = match.group(2)
                    template = "ABBA"
                elif i == 1:  # BABA
                    indirect_object_name = match.group(1)
                    subject_name = match.group(2)
                    template = "BABA"
                else:  # Simple
                    subject_name = match.group(1)
                    indirect_object_name = match.group(2)
                    template = "Simple"
                break

        # Find token positions (simplified - would need actual tokenization)
        words = text.split()
        subject_idx = -1
        io_idx = -1

        for i, word in enumerate(words):
            word_clean = word.strip(",.;:")
            if word_clean == subject_name and subject_idx == -1:
                subject_idx = i
            elif word_clean == indirect_object_name and io_idx == -1:
                io_idx = i

        return IOISentence(
            text=text,
            subject_name=subject_name,
            indirect_object_name=indirect_object_name,
            subject_token_idx=subject_idx,
            indirect_object_token_idx=io_idx,
            final_token_idx=len(words) - 1,
            template=template,
        )

    async def detect_circuit(
        self,
        request: DetectIOIRequest,
    ) -> DetectIOIResponse:
        """
        Detect IOI circuit in a trace.

        This is a placeholder that returns known IOI heads.
        In production, this would use activation patching to verify
        which heads are actually active for the specific input.
        """
        start_time = time.perf_counter()

        # Parse sentence if not provided
        sentence = request.sentence
        if sentence is None:
            # Would need to get text from trace
            sentence = IOISentence(
                text="",
                subject_name="",
                indirect_object_name="",
                subject_token_idx=0,
                indirect_object_token_idx=1,
                final_token_idx=10,
            )

        # Build IOI circuit from known heads
        # TODO: Actually verify these via activation patching

        def make_heads(role: str, heads: list[tuple[int, int]]) -> list[IOIHead]:
            return [
                IOIHead(
                    layer=layer,
                    head=head,
                    role=role,  # type: ignore
                    importance=0.8,  # Would be computed via patching
                    confidence=0.9,
                )
                for layer, head in heads
            ]

        circuit = IOICircuit(
            sentence=sentence,
            name_mover_heads=make_heads("name_mover", self._known_heads["name_movers"]),
            negative_name_mover_heads=make_heads("negative_name_mover", self._known_heads["negative_name_movers"]),
            s_inhibition_heads=make_heads("s_inhibition", self._known_heads["s_inhibition"]),
            duplicate_token_heads=make_heads("duplicate_token", self._known_heads["duplicate_token"]),
            induction_heads=make_heads("induction", self._known_heads["induction"]),
            backup_name_mover_heads=[],
            validity_score=0.85,
            prediction_correct=True,
            predicted_token=sentence.subject_name,
            correct_token=sentence.subject_name,
        )

        computation_time = (time.perf_counter() - start_time) * 1000

        # Count matched known heads
        total_known = sum(len(v) for v in self._known_heads.values())

        return DetectIOIResponse(
            trace_id=request.trace_id,
            circuit=circuit,
            known_heads_matched=total_known if request.config.validate_against_known else None,
            computation_time_ms=computation_time,
        )


def get_ioi_service() -> IOIService:
    """Dependency injection for IOIService."""
    return IOIService()
