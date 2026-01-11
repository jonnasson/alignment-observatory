"""Tests for IOI (Indirect Object Identification) circuit detection."""

import numpy as np
import pytest

from alignment_microscope.ioi import (
    IOIDetector,
    IOISentence,
    IOICircuit,
    IOIHead,
    IOIDetectionConfig,
    IOIValidationResult,
    KnownIOIHeads,
)


class TestIOISentence:
    """Tests for IOISentence parsing."""

    def test_from_positions(self):
        """Test creating IOISentence with manual positions."""
        sentence = IOISentence.from_positions(
            tokens=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            token_strings=["When", "John", "and", "Mary", "went", "to", "store", "Mary", "gave", "to"],
            subject_positions=[3, 7],  # Mary positions
            io_position=1,  # John position
            subject2_position=7,  # Second Mary
            end_position=9,  # "to"
            correct_answer="John",
            distractor="Mary",
        )

        assert sentence.io_position == 1
        assert sentence.subject2_position == 7
        assert sentence.end_position == 9
        assert sentence.correct_answer == "John"
        assert sentence.distractor == "Mary"
        assert len(sentence.subject_positions) == 2

    def test_sentence_positions_valid(self):
        """Test that position accessors work correctly."""
        sentence = IOISentence.from_positions(
            tokens=list(range(15)),
            token_strings=["t"] * 15,
            subject_positions=[3, 10],
            io_position=5,
            subject2_position=10,
            end_position=14,
            correct_answer="John",
            distractor="Mary",
        )

        assert sentence.io_position == 5
        assert sentence.subject2_position == 10
        assert sentence.end_position == 14


class TestIOIHead:
    """Tests for IOIHead data class."""

    def test_ioi_head_creation(self):
        """Test creating an IOI head."""
        head = IOIHead(
            layer=9,
            head=9,
            component_type="name_mover",
            score=0.85,
            metrics={"end_to_io_attention": 0.85},
        )

        assert head.layer == 9
        assert head.head == 9
        assert head.component_type == "name_mover"
        assert head.score == 0.85
        assert "end_to_io_attention" in head.metrics


class TestIOIDetectionConfig:
    """Tests for IOI detection configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = IOIDetectionConfig()

        assert config.name_mover_threshold == 0.3
        assert config.s_inhibition_threshold == 0.2
        assert config.top_k_heads == 5
        assert "name_mover" in config.layer_ranges
        assert "s_inhibition" in config.layer_ranges

    def test_custom_config(self):
        """Test custom configuration."""
        config = IOIDetectionConfig(
            name_mover_threshold=0.5,
            top_k_heads=10,
        )

        assert config.name_mover_threshold == 0.5
        assert config.top_k_heads == 10


class TestKnownIOIHeads:
    """Tests for known IOI heads database."""

    def test_name_movers_gpt2(self):
        """Test known name mover heads for GPT-2."""
        heads = KnownIOIHeads.name_movers_gpt2()

        assert len(heads) == 3
        assert (9, 9) in heads
        assert (10, 0) in heads
        assert (9, 6) in heads

    def test_s_inhibition_gpt2(self):
        """Test known S-inhibition heads for GPT-2."""
        heads = KnownIOIHeads.s_inhibition_gpt2()

        assert len(heads) == 4
        assert (7, 3) in heads
        assert (8, 6) in heads

    def test_duplicate_token_gpt2(self):
        """Test known duplicate token heads for GPT-2."""
        heads = KnownIOIHeads.duplicate_token_gpt2()

        assert len(heads) == 3
        assert (0, 1) in heads

    def test_all_gpt2(self):
        """Test getting all known heads."""
        all_heads = KnownIOIHeads.all_gpt2()

        assert "name_mover" in all_heads
        assert "s_inhibition" in all_heads
        assert "duplicate_token" in all_heads
        assert "backup_name_mover" in all_heads


class TestIOIDetector:
    """Tests for IOI circuit detector."""

    @pytest.fixture
    def mock_sentence(self):
        """Create a mock IOI sentence."""
        return IOISentence.from_positions(
            tokens=list(range(15)),
            token_strings=["t"] * 15,
            subject_positions=[3, 10],
            io_position=5,
            subject2_position=10,
            end_position=14,
            correct_answer="John",
            distractor="Mary",
        )

    @pytest.fixture
    def mock_attention_patterns(self):
        """Create mock attention patterns with IOI-like structure."""
        patterns = {}

        # Create attention patterns for layers 0-11
        for layer in range(12):
            # Shape: [batch=1, heads=12, seq=15, seq=15]
            pattern = np.random.rand(1, 12, 15, 15).astype(np.float32)

            # Normalize to sum to 1 along last axis (valid attention)
            pattern = pattern / pattern.sum(axis=-1, keepdims=True)

            # Add specific patterns for IOI
            if layer in [9, 10]:
                # Name mover: high attention from end (14) to IO (5)
                pattern[0, 9, 14, 5] = 0.6  # Head 9 in layer 9
                pattern[0, 0, 14, 5] = 0.5  # Head 0 in layer 10

            if layer in [7, 8]:
                # S-inhibition: attention from end (14) to S2 (10)
                pattern[0, 3, 14, 10] = 0.4
                pattern[0, 9, 14, 10] = 0.35

            if layer in [0, 1]:
                # Duplicate token: attention from S2 (10) to S1 (3)
                pattern[0, 1, 10, 3] = 0.5
                pattern[0, 10, 10, 3] = 0.4

            patterns[layer] = pattern

        return patterns

    def test_detector_creation(self):
        """Test creating an IOI detector."""
        # Mock microscope
        class MockMicroscope:
            pass

        detector = IOIDetector(MockMicroscope())
        assert detector.config is not None

    def test_detect_from_attention_finds_name_movers(
        self, mock_sentence, mock_attention_patterns
    ):
        """Test that detection finds name mover heads."""
        class MockMicroscope:
            pass

        detector = IOIDetector(MockMicroscope())
        result = detector.detect_from_attention(mock_attention_patterns, mock_sentence)

        # Should find name mover heads
        assert len(result.name_mover_heads) > 0

        # Check that high-scoring heads were found
        nm_layers = {h.layer for h in result.name_mover_heads}
        assert 9 in nm_layers or 10 in nm_layers

    def test_detect_from_attention_finds_s_inhibition(
        self, mock_sentence, mock_attention_patterns
    ):
        """Test that detection finds S-inhibition heads."""
        class MockMicroscope:
            pass

        detector = IOIDetector(MockMicroscope())
        result = detector.detect_from_attention(mock_attention_patterns, mock_sentence)

        # Should find S-inhibition heads
        assert len(result.s_inhibition_heads) > 0

    def test_detect_from_attention_finds_duplicate_token(
        self, mock_sentence, mock_attention_patterns
    ):
        """Test that detection finds duplicate token heads."""
        class MockMicroscope:
            pass

        detector = IOIDetector(MockMicroscope())
        result = detector.detect_from_attention(mock_attention_patterns, mock_sentence)

        # Should find duplicate token heads
        assert len(result.duplicate_token_heads) > 0

    def test_validity_score_computation(
        self, mock_sentence, mock_attention_patterns
    ):
        """Test that validity score is computed correctly."""
        class MockMicroscope:
            pass

        detector = IOIDetector(MockMicroscope())
        result = detector.detect_from_attention(mock_attention_patterns, mock_sentence)

        # Validity score should be between 0 and 1
        assert 0 <= result.validity_score <= 1

        # Should have positive validity if we found components
        if result.name_mover_heads or result.s_inhibition_heads:
            assert result.validity_score > 0


class TestIOICircuit:
    """Tests for IOI circuit results."""

    @pytest.fixture
    def mock_circuit(self):
        """Create a mock IOI circuit result."""
        sentence = IOISentence.from_positions(
            tokens=list(range(15)),
            token_strings=["t"] * 15,
            subject_positions=[3, 10],
            io_position=5,
            subject2_position=10,
            end_position=14,
            correct_answer="John",
            distractor="Mary",
        )

        return IOICircuit(
            name_mover_heads=[
                IOIHead(9, 9, "name_mover", 0.6, {}),
                IOIHead(10, 0, "name_mover", 0.5, {}),
            ],
            s_inhibition_heads=[
                IOIHead(7, 3, "s_inhibition", 0.4, {}),
                IOIHead(8, 6, "s_inhibition", 0.35, {}),
            ],
            duplicate_token_heads=[
                IOIHead(0, 1, "duplicate_token", 0.5, {}),
            ],
            previous_token_heads=[],
            backup_name_mover_heads=[],
            validity_score=0.85,
            sentence=sentence,
        )

    def test_to_dot(self, mock_circuit):
        """Test DOT export."""
        dot = mock_circuit.to_dot()

        assert "digraph IOICircuit" in dot
        assert "L9H9" in dot  # Name mover
        assert "L7H3" in dot  # S-inhibition
        assert "L0H1" in dot  # Duplicate token

    def test_validate_against_known(self, mock_circuit):
        """Test validation against known heads."""
        result = mock_circuit.validate_against_known("gpt2")

        assert isinstance(result, IOIValidationResult)
        assert 0 <= result.precision <= 1
        assert 0 <= result.recall <= 1
        assert 0 <= result.f1_score <= 1

        # Should have some true positives since we used known heads
        assert "name_mover" in result.per_component_metrics

    def test_validate_against_known_invalid_model(self, mock_circuit):
        """Test validation fails for unsupported models."""
        with pytest.raises(ValueError):
            mock_circuit.validate_against_known("llama")


class TestIOILogitDiff:
    """Tests for logit difference computation."""

    def test_compute_logit_diff_2d(self):
        """Test logit diff computation with 2D logits."""
        # [seq=10, vocab=100]
        logits = np.random.randn(10, 100).astype(np.float32)

        # Set specific logits at the last position
        logits[-1, 50] = 5.0  # IO token
        logits[-1, 60] = 3.0  # S token

        diff = IOIDetector.compute_logit_diff(logits, io_token_id=50, s_token_id=60)

        assert diff == pytest.approx(2.0, abs=0.01)

    def test_compute_logit_diff_3d(self):
        """Test logit diff computation with 3D logits."""
        # [batch=1, seq=10, vocab=100]
        logits = np.random.randn(1, 10, 100).astype(np.float32)

        # Set specific logits at the last position
        logits[0, -1, 50] = 4.0  # IO token
        logits[0, -1, 60] = 1.0  # S token

        diff = IOIDetector.compute_logit_diff(logits, io_token_id=50, s_token_id=60)

        assert diff == pytest.approx(3.0, abs=0.01)


class TestIOICircuitEdgeCases:
    """Edge case tests for IOI detection."""

    def test_empty_attention_patterns(self):
        """Test detection with empty attention patterns."""
        class MockMicroscope:
            pass

        sentence = IOISentence.from_positions(
            tokens=list(range(15)),
            token_strings=["t"] * 15,
            subject_positions=[3, 10],
            io_position=5,
            subject2_position=10,
            end_position=14,
            correct_answer="John",
            distractor="Mary",
        )

        detector = IOIDetector(MockMicroscope())
        result = detector.detect_from_attention({}, sentence)

        # Should return empty lists
        assert len(result.name_mover_heads) == 0
        assert len(result.s_inhibition_heads) == 0
        assert len(result.duplicate_token_heads) == 0
        assert result.validity_score == 0.0

    def test_single_layer_attention(self):
        """Test detection with only one layer."""
        class MockMicroscope:
            pass

        sentence = IOISentence.from_positions(
            tokens=list(range(10)),
            token_strings=["t"] * 10,
            subject_positions=[2, 6],
            io_position=4,
            subject2_position=6,
            end_position=9,
            correct_answer="John",
            distractor="Mary",
        )

        # Only layer 9 (name mover layer)
        patterns = {
            9: np.random.rand(1, 12, 10, 10).astype(np.float32),
        }
        patterns[9][0, 5, 9, 4] = 0.7  # High attention to IO

        detector = IOIDetector(MockMicroscope())
        result = detector.detect_from_attention(patterns, sentence)

        # Should find name mover but not others
        assert len(result.name_mover_heads) > 0

    def test_invalid_positions(self):
        """Test detection handles invalid positions gracefully."""
        class MockMicroscope:
            pass

        sentence = IOISentence.from_positions(
            tokens=list(range(5)),
            token_strings=["t"] * 5,
            subject_positions=[1, 2],
            io_position=100,  # Out of bounds
            subject2_position=200,  # Out of bounds
            end_position=300,  # Out of bounds
            correct_answer="John",
            distractor="Mary",
        )

        patterns = {
            9: np.random.rand(1, 12, 5, 5).astype(np.float32),
        }

        detector = IOIDetector(MockMicroscope())
        result = detector.detect_from_attention(patterns, sentence)

        # Should not crash, just return empty
        assert isinstance(result, IOICircuit)
