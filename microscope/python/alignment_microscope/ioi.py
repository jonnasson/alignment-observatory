"""IOI (Indirect Object Identification) Circuit Detection module.

This module provides tools for detecting the IOI circuit in transformer models,
based on the paper "Interpretability in the Wild" (Wang et al., 2022).

The IOI task involves sentences like:
  "When John and Mary went to the store, Mary gave a bottle of milk to"
where the model should predict "John" (the indirect object) rather than "Mary" (the subject).

Example:
    from alignment_microscope.ioi import IOIDetector, IOISentence

    detector = IOIDetector(microscope)
    sentence = IOISentence.parse(
        "When John and Mary went to the store, Mary gave to",
        tokenizer,
        subject_name="Mary",
        io_name="John",
    )
    result = detector.detect(model, sentence, clean_prompt, corrupt_prompt)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class IOISentence:
    """Parsed IOI sentence with token role annotations."""

    tokens: List[int]
    token_strings: List[str]
    subject_positions: List[int]
    io_position: int
    subject2_position: int  # Second occurrence of subject
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
    ) -> "IOISentence":
        """Parse a sentence and identify IOI token positions.

        Args:
            text: The IOI prompt text
            tokenizer: HuggingFace tokenizer
            subject_name: The subject name (distractor)
            io_name: The indirect object name (correct answer)

        Returns:
            IOISentence with token positions identified
        """
        # Tokenize the text
        encoding = tokenizer(text, return_tensors="pt")
        token_ids = encoding["input_ids"][0].tolist()
        token_strings = [tokenizer.decode([t]) for t in token_ids]

        # Find positions
        subject_positions = []
        io_position = -1
        subject2_position = -1

        for i, token_str in enumerate(token_strings):
            token_clean = token_str.strip().lower()
            subject_clean = subject_name.strip().lower()
            io_clean = io_name.strip().lower()

            # Check for subject name
            if subject_clean in token_clean or token_clean in subject_clean:
                subject_positions.append(i)

            # Check for IO name
            if io_clean in token_clean or token_clean in io_clean:
                if io_position == -1:
                    io_position = i

        # Subject2 is the last occurrence of subject
        if len(subject_positions) >= 2:
            subject2_position = subject_positions[-1]
        elif len(subject_positions) == 1:
            subject2_position = subject_positions[0]

        # End position is the last token
        end_position = len(token_ids) - 1

        return cls(
            tokens=token_ids,
            token_strings=token_strings,
            subject_positions=subject_positions,
            io_position=io_position,
            subject2_position=subject2_position,
            end_position=end_position,
            correct_answer=io_name,
            distractor=subject_name,
        )

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
    ) -> "IOISentence":
        """Create an IOISentence with manually specified positions."""
        return cls(
            tokens=tokens,
            token_strings=token_strings,
            subject_positions=subject_positions,
            io_position=io_position,
            subject2_position=subject2_position,
            end_position=end_position,
            correct_answer=correct_answer,
            distractor=distractor,
        )


@dataclass
class IOIHead:
    """A detected IOI circuit component head."""

    layer: int
    head: int
    component_type: str  # "name_mover", "s_inhibition", etc.
    score: float
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class IOICircuit:
    """Complete IOI circuit detection result."""

    name_mover_heads: List[IOIHead]
    s_inhibition_heads: List[IOIHead]
    duplicate_token_heads: List[IOIHead]
    previous_token_heads: List[IOIHead]
    backup_name_mover_heads: List[IOIHead]
    validity_score: float
    sentence: IOISentence

    def to_dot(self) -> str:
        """Export to Graphviz DOT format with IOI-specific styling."""
        lines = [
            "digraph IOICircuit {",
            "  rankdir=TB;",
            "  node [shape=box];",
            "",
            "  // Name Mover heads (red)",
        ]

        # Name movers in red
        for head in self.name_mover_heads:
            label = f"L{head.layer}H{head.head}"
            lines.append(f'  "{label}" [label="{label}\\nNM", color=red, style=filled, fillcolor=lightpink];')

        lines.append("")
        lines.append("  // S-Inhibition heads (blue)")

        # S-Inhibition in blue
        for head in self.s_inhibition_heads:
            label = f"L{head.layer}H{head.head}"
            lines.append(f'  "{label}" [label="{label}\\nSI", color=blue, style=filled, fillcolor=lightblue];')

        lines.append("")
        lines.append("  // Duplicate Token heads (green)")

        # Duplicate Token in green
        for head in self.duplicate_token_heads:
            label = f"L{head.layer}H{head.head}"
            lines.append(f'  "{label}" [label="{label}\\nDT", color=green, style=filled, fillcolor=lightgreen];')

        lines.append("")
        lines.append("  // Edges")

        # Connect DT -> SI -> NM
        for dt_head in self.duplicate_token_heads:
            for si_head in self.s_inhibition_heads:
                if si_head.layer > dt_head.layer:
                    from_label = f"L{dt_head.layer}H{dt_head.head}"
                    to_label = f"L{si_head.layer}H{si_head.head}"
                    lines.append(f'  "{from_label}" -> "{to_label}";')

        for si_head in self.s_inhibition_heads:
            for nm_head in self.name_mover_heads:
                if nm_head.layer > si_head.layer:
                    from_label = f"L{si_head.layer}H{si_head.head}"
                    to_label = f"L{nm_head.layer}H{nm_head.head}"
                    lines.append(f'  "{from_label}" -> "{to_label}";')

        lines.append("}")
        return "\n".join(lines)

    def validate_against_known(self, model_type: str = "gpt2") -> "IOIValidationResult":
        """Compare detected heads against known IOI heads from the paper.

        Args:
            model_type: Model type ("gpt2" supported)

        Returns:
            Validation result with precision/recall metrics
        """
        if model_type != "gpt2":
            raise ValueError(f"Validation only supported for gpt2, got {model_type}")

        known = KnownIOIHeads.all_gpt2()

        results = {}
        for component_type in ["name_mover", "s_inhibition", "duplicate_token"]:
            known_heads = set(known.get(component_type, []))

            if component_type == "name_mover":
                detected_heads = {(h.layer, h.head) for h in self.name_mover_heads}
            elif component_type == "s_inhibition":
                detected_heads = {(h.layer, h.head) for h in self.s_inhibition_heads}
            elif component_type == "duplicate_token":
                detected_heads = {(h.layer, h.head) for h in self.duplicate_token_heads}
            else:
                detected_heads = set()

            true_positives = len(known_heads & detected_heads)
            precision = true_positives / len(detected_heads) if detected_heads else 0.0
            recall = true_positives / len(known_heads) if known_heads else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            results[component_type] = (precision, recall, f1)

        # Overall metrics
        all_detected = (
            {(h.layer, h.head) for h in self.name_mover_heads}
            | {(h.layer, h.head) for h in self.s_inhibition_heads}
            | {(h.layer, h.head) for h in self.duplicate_token_heads}
        )
        all_known = (
            set(known.get("name_mover", []))
            | set(known.get("s_inhibition", []))
            | set(known.get("duplicate_token", []))
        )

        overall_tp = len(all_known & all_detected)
        overall_precision = overall_tp / len(all_detected) if all_detected else 0.0
        overall_recall = overall_tp / len(all_known) if all_known else 0.0
        overall_f1 = (
            2 * overall_precision * overall_recall / (overall_precision + overall_recall)
            if (overall_precision + overall_recall) > 0
            else 0.0
        )

        return IOIValidationResult(
            precision=overall_precision,
            recall=overall_recall,
            f1_score=overall_f1,
            per_component_metrics=results,
            false_positives=list(all_detected - all_known),
            false_negatives=list(all_known - all_detected),
        )


@dataclass
class IOIValidationResult:
    """Result of validating detected heads against known IOI heads."""

    precision: float
    recall: float
    f1_score: float
    per_component_metrics: Dict[str, Tuple[float, float, float]]  # (precision, recall, f1)
    false_positives: List[Tuple[int, int]]
    false_negatives: List[Tuple[int, int]]


@dataclass
class IOIDetectionConfig:
    """Configuration for IOI detection."""

    name_mover_threshold: float = 0.3
    s_inhibition_threshold: float = 0.2
    top_k_heads: int = 5
    layer_ranges: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "duplicate_token": (0, 3),
        "previous_token": (0, 3),
        "induction": (4, 7),
        "s_inhibition": (6, 9),
        "name_mover": (9, 12),
        "backup_name_mover": (9, 12),
    })


class KnownIOIHeads:
    """Known IOI heads from the original paper (GPT-2 small)."""

    @staticmethod
    def name_movers_gpt2() -> List[Tuple[int, int]]:
        """Known Name Mover heads for GPT-2 small."""
        return [(9, 9), (10, 0), (9, 6)]

    @staticmethod
    def backup_name_movers_gpt2() -> List[Tuple[int, int]]:
        """Known Backup Name Mover heads for GPT-2 small."""
        return [(10, 10), (10, 6), (10, 2), (11, 2), (9, 7), (10, 1)]

    @staticmethod
    def s_inhibition_gpt2() -> List[Tuple[int, int]]:
        """Known S-Inhibition heads for GPT-2 small."""
        return [(7, 3), (7, 9), (8, 6), (8, 10)]

    @staticmethod
    def duplicate_token_gpt2() -> List[Tuple[int, int]]:
        """Known Duplicate Token heads for GPT-2 small."""
        return [(0, 1), (0, 10), (3, 0)]

    @staticmethod
    def all_gpt2() -> Dict[str, List[Tuple[int, int]]]:
        """All known heads for GPT-2 small."""
        return {
            "name_mover": KnownIOIHeads.name_movers_gpt2(),
            "backup_name_mover": KnownIOIHeads.backup_name_movers_gpt2(),
            "s_inhibition": KnownIOIHeads.s_inhibition_gpt2(),
            "duplicate_token": KnownIOIHeads.duplicate_token_gpt2(),
        }


class IOIDetector:
    """Detect the IOI (Indirect Object Identification) circuit.

    Based on "Interpretability in the Wild" (Wang et al., 2022).

    Example:
        from alignment_microscope import Microscope
        from alignment_microscope.ioi import IOIDetector, IOISentence

        scope = Microscope.for_model(model)
        detector = IOIDetector(scope)

        sentence = IOISentence.parse(
            "When John and Mary went to the store, Mary gave to",
            tokenizer,
            subject_name="Mary",
            io_name="John"
        )

        result = detector.detect(
            model,
            sentence,
            clean_prompt="When John and Mary went to the store, Mary gave to",
            corrupt_prompt="When Mary and John went to the store, Mary gave to",
        )

        print(f"Found {len(result.name_mover_heads)} name mover heads")
        print(result.to_dot())
    """

    def __init__(
        self,
        microscope: Any,  # Microscope
        config: Optional[IOIDetectionConfig] = None,
    ):
        """Initialize the IOI detector.

        Args:
            microscope: Microscope instance for tracing
            config: Detection configuration
        """
        self.microscope = microscope
        self.config = config or IOIDetectionConfig()

    def detect(
        self,
        model: Any,
        sentence: IOISentence,
        clean_prompt: str,
        corrupt_prompt: str,
    ) -> IOICircuit:
        """Detect the IOI circuit using activation patching.

        Args:
            model: The model to analyze
            sentence: Parsed IOI sentence
            clean_prompt: Prompt where model predicts correct name
            corrupt_prompt: Prompt where names are swapped

        Returns:
            IOICircuit with detected components
        """
        import torch

        # Get tokenizer from model if available
        tokenizer = getattr(model, "tokenizer", None)

        # Run model on clean prompt to get attention patterns
        with self.microscope.trace() as trace:
            clean_input = tokenizer(clean_prompt, return_tensors="pt") if tokenizer else None
            if clean_input is not None:
                with torch.no_grad():
                    model(clean_input["input_ids"], output_attentions=True)

        # Extract attention patterns from trace
        attention_patterns = trace.attention_patterns

        # Detect each component type
        name_mover_heads = self._find_name_mover_heads(attention_patterns, sentence)
        s_inhibition_heads = self._find_s_inhibition_heads(attention_patterns, sentence)
        duplicate_token_heads = self._find_duplicate_token_heads(attention_patterns, sentence)
        previous_token_heads = self._find_previous_token_heads(attention_patterns)
        backup_name_mover_heads = self._find_backup_name_mover_heads(attention_patterns, sentence)

        # Compute validity score
        validity_score = self._compute_validity_score(
            name_mover_heads,
            s_inhibition_heads,
            duplicate_token_heads,
        )

        return IOICircuit(
            name_mover_heads=name_mover_heads,
            s_inhibition_heads=s_inhibition_heads,
            duplicate_token_heads=duplicate_token_heads,
            previous_token_heads=previous_token_heads,
            backup_name_mover_heads=backup_name_mover_heads,
            validity_score=validity_score,
            sentence=sentence,
        )

    def detect_from_attention(
        self,
        attention_patterns: Dict[int, np.ndarray],
        sentence: IOISentence,
    ) -> IOICircuit:
        """Detect IOI circuit from pre-computed attention patterns.

        Args:
            attention_patterns: Dict mapping layer -> attention array [batch, heads, seq, seq]
            sentence: Parsed IOI sentence

        Returns:
            IOICircuit with detected components
        """
        name_mover_heads = self._find_name_mover_heads(attention_patterns, sentence)
        s_inhibition_heads = self._find_s_inhibition_heads(attention_patterns, sentence)
        duplicate_token_heads = self._find_duplicate_token_heads(attention_patterns, sentence)
        previous_token_heads = self._find_previous_token_heads(attention_patterns)
        backup_name_mover_heads = self._find_backup_name_mover_heads(attention_patterns, sentence)

        validity_score = self._compute_validity_score(
            name_mover_heads,
            s_inhibition_heads,
            duplicate_token_heads,
        )

        return IOICircuit(
            name_mover_heads=name_mover_heads,
            s_inhibition_heads=s_inhibition_heads,
            duplicate_token_heads=duplicate_token_heads,
            previous_token_heads=previous_token_heads,
            backup_name_mover_heads=backup_name_mover_heads,
            validity_score=validity_score,
            sentence=sentence,
        )

    def _find_name_mover_heads(
        self,
        attention_patterns: Dict[int, np.ndarray],
        sentence: IOISentence,
    ) -> List[IOIHead]:
        """Find Name Mover heads by attention pattern analysis."""
        heads = []
        io_pos = sentence.io_position
        end_pos = sentence.end_position

        min_layer, max_layer = self.config.layer_ranges.get("name_mover", (9, 12))

        for layer, pattern in attention_patterns.items():
            if layer < min_layer or layer >= max_layer:
                continue

            num_heads = pattern.shape[1]
            for head in range(num_heads):
                head_pattern = pattern[0, head]  # [seq, seq]

                # Name Mover: high attention from END to IO position
                if end_pos < head_pattern.shape[0] and io_pos < head_pattern.shape[1]:
                    end_to_io_attention = float(head_pattern[end_pos, io_pos])
                else:
                    end_to_io_attention = 0.0

                if end_to_io_attention > self.config.name_mover_threshold:
                    heads.append(IOIHead(
                        layer=layer,
                        head=head,
                        component_type="name_mover",
                        score=end_to_io_attention,
                        metrics={"end_to_io_attention": end_to_io_attention},
                    ))

        # Sort by score and take top-k
        heads.sort(key=lambda h: h.score, reverse=True)
        return heads[:self.config.top_k_heads]

    def _find_s_inhibition_heads(
        self,
        attention_patterns: Dict[int, np.ndarray],
        sentence: IOISentence,
    ) -> List[IOIHead]:
        """Find S-Inhibition heads."""
        heads = []
        s2_pos = sentence.subject2_position
        end_pos = sentence.end_position

        min_layer, max_layer = self.config.layer_ranges.get("s_inhibition", (6, 9))

        for layer, pattern in attention_patterns.items():
            if layer < min_layer or layer >= max_layer:
                continue

            num_heads = pattern.shape[1]
            for head in range(num_heads):
                head_pattern = pattern[0, head]

                # S-Inhibition: attend from END to S2 position
                if end_pos < head_pattern.shape[0] and s2_pos < head_pattern.shape[1]:
                    end_to_s2_attention = float(head_pattern[end_pos, s2_pos])
                else:
                    end_to_s2_attention = 0.0

                if end_to_s2_attention > self.config.s_inhibition_threshold:
                    heads.append(IOIHead(
                        layer=layer,
                        head=head,
                        component_type="s_inhibition",
                        score=end_to_s2_attention,
                        metrics={"end_to_s2_attention": end_to_s2_attention},
                    ))

        heads.sort(key=lambda h: h.score, reverse=True)
        return heads[:self.config.top_k_heads]

    def _find_duplicate_token_heads(
        self,
        attention_patterns: Dict[int, np.ndarray],
        sentence: IOISentence,
    ) -> List[IOIHead]:
        """Find Duplicate Token heads."""
        heads = []

        if not sentence.subject_positions:
            return heads

        s1_pos = sentence.subject_positions[0]
        s2_pos = sentence.subject2_position

        min_layer, max_layer = self.config.layer_ranges.get("duplicate_token", (0, 3))

        for layer, pattern in attention_patterns.items():
            if layer < min_layer or layer >= max_layer:
                continue

            num_heads = pattern.shape[1]
            for head in range(num_heads):
                head_pattern = pattern[0, head]

                # Duplicate Token: high attention from S2 to S1
                if s2_pos < head_pattern.shape[0] and s1_pos < head_pattern.shape[1]:
                    s2_to_s1_attention = float(head_pattern[s2_pos, s1_pos])
                else:
                    s2_to_s1_attention = 0.0

                if s2_to_s1_attention > 0.2:
                    heads.append(IOIHead(
                        layer=layer,
                        head=head,
                        component_type="duplicate_token",
                        score=s2_to_s1_attention,
                        metrics={"s2_to_s1_attention": s2_to_s1_attention},
                    ))

        heads.sort(key=lambda h: h.score, reverse=True)
        return heads[:self.config.top_k_heads]

    def _find_previous_token_heads(
        self,
        attention_patterns: Dict[int, np.ndarray],
    ) -> List[IOIHead]:
        """Find Previous Token heads."""
        heads = []

        min_layer, max_layer = self.config.layer_ranges.get("previous_token", (0, 3))

        for layer, pattern in attention_patterns.items():
            if layer < min_layer or layer >= max_layer:
                continue

            num_heads = pattern.shape[1]
            seq_len = pattern.shape[2]

            for head in range(num_heads):
                head_pattern = pattern[0, head]

                # Check for previous token pattern
                prev_token_score = 0.0
                for i in range(1, seq_len):
                    prev_token_score += head_pattern[i, i - 1]
                prev_token_score /= max(seq_len - 1, 1)

                if prev_token_score > 0.5:
                    heads.append(IOIHead(
                        layer=layer,
                        head=head,
                        component_type="previous_token",
                        score=prev_token_score,
                        metrics={"prev_token_score": prev_token_score},
                    ))

        heads.sort(key=lambda h: h.score, reverse=True)
        return heads[:self.config.top_k_heads]

    def _find_backup_name_mover_heads(
        self,
        attention_patterns: Dict[int, np.ndarray],
        sentence: IOISentence,
    ) -> List[IOIHead]:
        """Find Backup Name Mover heads."""
        heads = []
        io_pos = sentence.io_position
        end_pos = sentence.end_position

        min_layer, max_layer = self.config.layer_ranges.get("backup_name_mover", (9, 12))
        backup_threshold = self.config.name_mover_threshold * 0.7

        for layer, pattern in attention_patterns.items():
            if layer < min_layer or layer >= max_layer:
                continue

            num_heads = pattern.shape[1]
            for head in range(num_heads):
                head_pattern = pattern[0, head]

                if end_pos < head_pattern.shape[0] and io_pos < head_pattern.shape[1]:
                    end_to_io_attention = float(head_pattern[end_pos, io_pos])
                else:
                    end_to_io_attention = 0.0

                # Backup name movers have moderate attention
                if backup_threshold < end_to_io_attention < self.config.name_mover_threshold:
                    heads.append(IOIHead(
                        layer=layer,
                        head=head,
                        component_type="backup_name_mover",
                        score=end_to_io_attention,
                        metrics={"end_to_io_attention": end_to_io_attention},
                    ))

        heads.sort(key=lambda h: h.score, reverse=True)
        return heads[:self.config.top_k_heads]

    def _compute_validity_score(
        self,
        name_mover_heads: List[IOIHead],
        s_inhibition_heads: List[IOIHead],
        duplicate_token_heads: List[IOIHead],
    ) -> float:
        """Compute validity score for the detected circuit."""
        score = 0.0

        if name_mover_heads:
            score += 0.4
        if s_inhibition_heads:
            score += 0.3
        if duplicate_token_heads:
            score += 0.3

        # Bonus for strong scores
        if name_mover_heads:
            score += name_mover_heads[0].score * 0.1
        if s_inhibition_heads:
            score += s_inhibition_heads[0].score * 0.1

        return min(score, 1.0)

    @staticmethod
    def compute_logit_diff(
        logits: np.ndarray,
        io_token_id: int,
        s_token_id: int,
    ) -> float:
        """Compute logit difference between IO and S tokens.

        Args:
            logits: Model output logits [seq, vocab] or [batch, seq, vocab]
            io_token_id: Token ID of the indirect object name
            s_token_id: Token ID of the subject name

        Returns:
            Logit difference (IO - S)
        """
        if logits.ndim == 3:
            # [batch, seq, vocab] -> use last position
            last_logits = logits[0, -1]
        else:
            # [seq, vocab] -> use last position
            last_logits = logits[-1]

        io_logit = float(last_logits[io_token_id])
        s_logit = float(last_logits[s_token_id])

        return io_logit - s_logit
