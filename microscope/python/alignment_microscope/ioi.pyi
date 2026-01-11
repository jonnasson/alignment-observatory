"""Type stubs for IOI (Indirect Object Identification) module."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

@dataclass
class IOISentence:
    tokens: List[int]
    token_strings: List[str]
    subject_positions: List[int]
    io_position: int
    subject2_position: int
    end_position: int
    correct_answer: str
    distractor: str

    @classmethod
    def parse(
        cls,
        text: str,
        tokenizer: Any,
        subject_name: str,
        io_name: str,
    ) -> IOISentence: ...
    @classmethod
    def from_positions(
        cls,
        tokens: List[int],
        token_strings: List[str],
        subject_positions: List[int],
        io_position: int,
        subject2_position: int,
        end_position: int,
        correct_answer: str,
        distractor: str,
    ) -> IOISentence: ...

@dataclass
class IOIHead:
    layer: int
    head: int
    component_type: str
    score: float
    metrics: Dict[str, float]

@dataclass
class IOICircuit:
    name_mover_heads: List[IOIHead]
    s_inhibition_heads: List[IOIHead]
    duplicate_token_heads: List[IOIHead]
    previous_token_heads: List[IOIHead]
    backup_name_mover_heads: List[IOIHead]
    validity_score: float
    sentence: IOISentence

    def to_dot(self) -> str: ...
    def validate_against_known(
        self, model_type: str = "gpt2"
    ) -> IOIValidationResult: ...

@dataclass
class IOIValidationResult:
    precision: float
    recall: float
    f1_score: float
    per_component_metrics: Dict[str, Tuple[float, float, float]]
    false_positives: List[Tuple[int, int]]
    false_negatives: List[Tuple[int, int]]

@dataclass
class IOIDetectionConfig:
    name_mover_threshold: float = 0.3
    s_inhibition_threshold: float = 0.2
    top_k_heads: int = 5
    layer_ranges: Dict[str, Tuple[int, int]] = ...

class KnownIOIHeads:
    @staticmethod
    def name_movers_gpt2() -> List[Tuple[int, int]]: ...
    @staticmethod
    def backup_name_movers_gpt2() -> List[Tuple[int, int]]: ...
    @staticmethod
    def s_inhibition_gpt2() -> List[Tuple[int, int]]: ...
    @staticmethod
    def duplicate_token_gpt2() -> List[Tuple[int, int]]: ...
    @staticmethod
    def all_gpt2() -> Dict[str, List[Tuple[int, int]]]: ...

class IOIDetector:
    microscope: Any
    config: IOIDetectionConfig

    def __init__(
        self,
        microscope: Any,
        config: Optional[IOIDetectionConfig] = None,
    ) -> None: ...
    def detect(
        self,
        model: Any,
        sentence: IOISentence,
        clean_prompt: str,
        corrupt_prompt: str,
    ) -> IOICircuit: ...
    def detect_from_attention(
        self,
        attention_patterns: Dict[int, npt.NDArray[np.float32]],
        sentence: IOISentence,
    ) -> IOICircuit: ...
    @staticmethod
    def compute_logit_diff(
        logits: npt.NDArray[np.float32],
        io_token_id: int,
        s_token_id: int,
    ) -> float: ...
