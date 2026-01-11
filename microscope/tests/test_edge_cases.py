"""Edge case tests for the Alignment Microscope."""

import numpy as np
import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from alignment_microscope.ioi import (
    IOIDetector,
    IOISentence,
    IOICircuit,
    IOIHead,
    IOIDetectionConfig,
    KnownIOIHeads,
)


class TestNumericalStability:
    """Tests for numerical edge cases."""

    def test_nan_in_attention_patterns(self):
        """Test handling of NaN values in attention patterns."""
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

        # Create patterns with NaN values
        patterns = {
            9: np.full((1, 12, 10, 10), np.nan, dtype=np.float32),
        }

        detector = IOIDetector(MockMicroscope())
        # Should not crash, should return empty results
        result = detector.detect_from_attention(patterns, sentence)
        assert isinstance(result, IOICircuit)

    def test_inf_in_attention_patterns(self):
        """Test handling of Inf values in attention patterns."""
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

        # Create patterns with Inf values
        patterns = {
            9: np.full((1, 12, 10, 10), np.inf, dtype=np.float32),
        }

        detector = IOIDetector(MockMicroscope())
        result = detector.detect_from_attention(patterns, sentence)
        assert isinstance(result, IOICircuit)

    def test_very_small_attention_values(self):
        """Test handling of very small (denormalized) float values."""
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

        # Create patterns with very small values (near denormalized floats)
        patterns = {
            9: np.full((1, 12, 10, 10), 1e-45, dtype=np.float32),
        }

        detector = IOIDetector(MockMicroscope())
        result = detector.detect_from_attention(patterns, sentence)
        assert isinstance(result, IOICircuit)
        # With such small values, no heads should be detected
        assert len(result.name_mover_heads) == 0

    def test_logit_diff_with_extreme_values(self):
        """Test logit diff computation with extreme values."""
        # Test with very large logits
        logits = np.zeros((10, 100), dtype=np.float32)
        logits[-1, 50] = 1e10
        logits[-1, 60] = 1e10 - 1

        diff = IOIDetector.compute_logit_diff(logits, io_token_id=50, s_token_id=60)
        assert np.isfinite(diff)
        assert diff == pytest.approx(1.0, abs=0.01)

    def test_logit_diff_with_negative_values(self):
        """Test logit diff with negative logit values."""
        logits = np.zeros((10, 100), dtype=np.float32)
        logits[-1, 50] = -3.0  # IO token
        logits[-1, 60] = -5.0  # S token

        diff = IOIDetector.compute_logit_diff(logits, io_token_id=50, s_token_id=60)
        assert diff == pytest.approx(2.0, abs=0.01)

    def test_mixed_precision_patterns(self):
        """Test with mixed precision attention patterns."""
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

        # Create float32 patterns
        patterns_f32 = {
            9: np.random.rand(1, 12, 10, 10).astype(np.float32),
        }

        detector = IOIDetector(MockMicroscope())
        result = detector.detect_from_attention(patterns_f32, sentence)
        assert isinstance(result, IOICircuit)


class TestInvalidInputs:
    """Tests for invalid input handling."""

    def test_empty_tokens(self):
        """Test with empty token list."""
        with pytest.raises((ValueError, IndexError)):
            IOISentence.from_positions(
                tokens=[],
                token_strings=[],
                subject_positions=[],
                io_position=0,
                subject2_position=0,
                end_position=0,
                correct_answer="John",
                distractor="Mary",
            )

    def test_mismatched_token_lengths(self):
        """Test with mismatched tokens and token_strings lengths."""
        # This should either raise an error or handle gracefully
        sentence = IOISentence.from_positions(
            tokens=[1, 2, 3],
            token_strings=["a", "b"],  # Length mismatch
            subject_positions=[0],
            io_position=1,
            subject2_position=2,
            end_position=2,
            correct_answer="John",
            distractor="Mary",
        )
        # Should still create object (may have inconsistent state)
        assert sentence is not None

    def test_negative_positions(self):
        """Test with negative position values."""
        sentence = IOISentence.from_positions(
            tokens=list(range(10)),
            token_strings=["t"] * 10,
            subject_positions=[-1, 5],  # Negative position
            io_position=3,
            subject2_position=5,
            end_position=9,
            correct_answer="John",
            distractor="Mary",
        )
        # Should handle gracefully
        assert sentence is not None

    def test_out_of_bounds_positions(self):
        """Test with positions exceeding token list length."""
        sentence = IOISentence.from_positions(
            tokens=list(range(5)),
            token_strings=["t"] * 5,
            subject_positions=[10],  # Out of bounds
            io_position=100,  # Out of bounds
            subject2_position=200,  # Out of bounds
            end_position=300,  # Out of bounds
            correct_answer="John",
            distractor="Mary",
        )

        class MockMicroscope:
            pass

        patterns = {
            9: np.random.rand(1, 12, 5, 5).astype(np.float32),
        }

        detector = IOIDetector(MockMicroscope())
        result = detector.detect_from_attention(patterns, sentence)
        # Should not crash, just return empty/zero validity
        assert isinstance(result, IOICircuit)

    def test_wrong_attention_shape(self):
        """Test with incorrectly shaped attention patterns."""
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

        # Create patterns with wrong shape (3D instead of 4D)
        patterns = {
            9: np.random.rand(12, 10, 10).astype(np.float32),
        }

        detector = IOIDetector(MockMicroscope())
        # Should handle gracefully or raise appropriate error
        try:
            result = detector.detect_from_attention(patterns, sentence)
            # If it doesn't raise, result should be valid
            assert isinstance(result, IOICircuit)
        except (ValueError, IndexError, KeyError):
            pass  # Expected for wrong shape

    def test_zero_sequence_length(self):
        """Test with zero-length sequences in attention patterns."""
        class MockMicroscope:
            pass

        sentence = IOISentence.from_positions(
            tokens=[0],
            token_strings=["t"],
            subject_positions=[0],
            io_position=0,
            subject2_position=0,
            end_position=0,
            correct_answer="John",
            distractor="Mary",
        )

        patterns = {
            9: np.zeros((1, 12, 0, 0), dtype=np.float32),
        }

        detector = IOIDetector(MockMicroscope())
        result = detector.detect_from_attention(patterns, sentence)
        assert isinstance(result, IOICircuit)


class TestBoundaryConditions:
    """Tests for boundary conditions."""

    def test_single_token_sequence(self):
        """Test with single token sequence."""
        sentence = IOISentence.from_positions(
            tokens=[42],
            token_strings=["John"],
            subject_positions=[0],
            io_position=0,
            subject2_position=0,
            end_position=0,
            correct_answer="John",
            distractor="Mary",
        )

        assert sentence.io_position == 0
        assert sentence.end_position == 0

    def test_single_layer_model(self):
        """Test with a model having only one layer."""
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

        # Only layer 0
        patterns = {
            0: np.random.rand(1, 12, 10, 10).astype(np.float32),
        }

        detector = IOIDetector(MockMicroscope())
        result = detector.detect_from_attention(patterns, sentence)
        assert isinstance(result, IOICircuit)

    def test_many_layers(self):
        """Test with many layers (100+)."""
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

        # 100 layers
        patterns = {
            i: np.random.rand(1, 8, 10, 10).astype(np.float32)
            for i in range(100)
        }

        detector = IOIDetector(MockMicroscope())
        result = detector.detect_from_attention(patterns, sentence)
        assert isinstance(result, IOICircuit)

    def test_large_head_count(self):
        """Test with many attention heads (128 heads)."""
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

        patterns = {
            9: np.random.rand(1, 128, 10, 10).astype(np.float32),
        }

        detector = IOIDetector(MockMicroscope())
        result = detector.detect_from_attention(patterns, sentence)
        assert isinstance(result, IOICircuit)

    def test_long_sequence(self):
        """Test with very long sequence (2048 tokens)."""
        class MockMicroscope:
            pass

        seq_len = 2048
        sentence = IOISentence.from_positions(
            tokens=list(range(seq_len)),
            token_strings=["t"] * seq_len,
            subject_positions=[100, 500],
            io_position=200,
            subject2_position=500,
            end_position=seq_len - 1,
            correct_answer="John",
            distractor="Mary",
        )

        # Use smaller pattern for memory efficiency
        patterns = {
            9: np.random.rand(1, 12, seq_len, seq_len).astype(np.float32) * 0.01,
        }
        patterns[9][0, 0, seq_len - 1, 200] = 0.8  # Name mover pattern

        detector = IOIDetector(MockMicroscope())
        result = detector.detect_from_attention(patterns, sentence)
        assert isinstance(result, IOICircuit)


class TestConfigurationEdgeCases:
    """Tests for configuration edge cases."""

    def test_zero_threshold(self):
        """Test with zero thresholds (everything passes)."""
        config = IOIDetectionConfig(
            name_mover_threshold=0.0,
            s_inhibition_threshold=0.0,
            top_k_heads=100,
        )

        assert config.name_mover_threshold == 0.0

    def test_threshold_of_one(self):
        """Test with threshold of 1.0 (nothing passes)."""
        config = IOIDetectionConfig(
            name_mover_threshold=1.0,
            s_inhibition_threshold=1.0,
        )

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

        patterns = {
            9: np.ones((1, 12, 10, 10), dtype=np.float32) * 0.5,
        }

        detector = IOIDetector(MockMicroscope(), config=config)
        result = detector.detect_from_attention(patterns, sentence)
        # With threshold of 1.0, no heads should pass
        assert len(result.name_mover_heads) == 0

    def test_top_k_zero(self):
        """Test with top_k of 0."""
        config = IOIDetectionConfig(top_k_heads=0)

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

        patterns = {
            9: np.ones((1, 12, 10, 10), dtype=np.float32),
        }

        detector = IOIDetector(MockMicroscope(), config=config)
        result = detector.detect_from_attention(patterns, sentence)
        # Should return empty lists with top_k=0
        assert len(result.name_mover_heads) == 0

    def test_empty_layer_ranges(self):
        """Test with empty layer ranges configuration."""
        config = IOIDetectionConfig()
        config.layer_ranges = {}  # Empty layer ranges

        assert config.layer_ranges == {}


class TestIOIHeadEdgeCases:
    """Tests for IOIHead edge cases."""

    def test_head_with_no_metrics(self):
        """Test IOIHead creation with no metrics."""
        head = IOIHead(
            layer=0,
            head=0,
            component_type="name_mover",
            score=0.5,
            metrics={},
        )
        assert head.metrics == {}

    def test_head_with_many_metrics(self):
        """Test IOIHead with many metrics."""
        metrics = {f"metric_{i}": float(i) for i in range(100)}
        head = IOIHead(
            layer=5,
            head=10,
            component_type="s_inhibition",
            score=0.8,
            metrics=metrics,
        )
        assert len(head.metrics) == 100

    def test_head_equality(self):
        """Test IOIHead comparison (same layer/head should be distinguishable)."""
        head1 = IOIHead(layer=9, head=9, component_type="name_mover", score=0.8, metrics={})
        head2 = IOIHead(layer=9, head=9, component_type="name_mover", score=0.9, metrics={})
        # Different scores but same location
        assert head1.layer == head2.layer
        assert head1.head == head2.head


class TestValidationEdgeCases:
    """Tests for validation against known heads."""

    def test_validation_empty_circuit(self):
        """Test validation with empty detected circuit."""
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

        circuit = IOICircuit(
            name_mover_heads=[],
            s_inhibition_heads=[],
            duplicate_token_heads=[],
            previous_token_heads=[],
            backup_name_mover_heads=[],
            validity_score=0.0,
            sentence=sentence,
        )

        result = circuit.validate_against_known("gpt2")
        # Should handle empty circuit gracefully
        assert result.precision == 0.0 or result.recall == 0.0

    def test_validation_perfect_match(self):
        """Test validation with perfect match to known heads."""
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

        # Use exact known heads
        known_nm = KnownIOIHeads.name_movers_gpt2()
        known_si = KnownIOIHeads.s_inhibition_gpt2()
        known_dt = KnownIOIHeads.duplicate_token_gpt2()

        circuit = IOICircuit(
            name_mover_heads=[
                IOIHead(l, h, "name_mover", 0.8, {}) for l, h in known_nm
            ],
            s_inhibition_heads=[
                IOIHead(l, h, "s_inhibition", 0.7, {}) for l, h in known_si
            ],
            duplicate_token_heads=[
                IOIHead(l, h, "duplicate_token", 0.6, {}) for l, h in known_dt
            ],
            previous_token_heads=[],
            backup_name_mover_heads=[],
            validity_score=1.0,
            sentence=sentence,
        )

        result = circuit.validate_against_known("gpt2")
        # Should have perfect or near-perfect scores
        assert result.recall >= 0.8


class TestDOTExport:
    """Tests for DOT graph export."""

    def test_dot_empty_circuit(self):
        """Test DOT export with empty circuit."""
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

        circuit = IOICircuit(
            name_mover_heads=[],
            s_inhibition_heads=[],
            duplicate_token_heads=[],
            previous_token_heads=[],
            backup_name_mover_heads=[],
            validity_score=0.0,
            sentence=sentence,
        )

        dot = circuit.to_dot()
        assert "digraph IOICircuit" in dot

    def test_dot_special_characters(self):
        """Test DOT export handles special characters."""
        sentence = IOISentence.from_positions(
            tokens=list(range(15)),
            token_strings=["t"] * 15,
            subject_positions=[3, 10],
            io_position=5,
            subject2_position=10,
            end_position=14,
            correct_answer='John "the" O\'Brien',  # Special chars
            distractor='Mary <>&',  # More special chars
        )

        circuit = IOICircuit(
            name_mover_heads=[IOIHead(9, 9, "name_mover", 0.8, {})],
            s_inhibition_heads=[],
            duplicate_token_heads=[],
            previous_token_heads=[],
            backup_name_mover_heads=[],
            validity_score=0.5,
            sentence=sentence,
        )

        dot = circuit.to_dot()
        # Should not crash and produce valid DOT
        assert "digraph" in dot


class TestConcurrency:
    """Tests for thread safety (where applicable)."""

    def test_multiple_detectors_parallel(self):
        """Test running multiple detectors in parallel."""
        class MockMicroscope:
            pass

        def run_detection():
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

            patterns = {
                9: np.random.rand(1, 12, 10, 10).astype(np.float32),
            }

            detector = IOIDetector(MockMicroscope())
            return detector.detect_from_attention(patterns, sentence)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_detection) for _ in range(10)]
            results = [f.result() for f in futures]

        assert len(results) == 10
        assert all(isinstance(r, IOICircuit) for r in results)

    def test_shared_config_parallel(self):
        """Test using shared config from multiple threads."""
        config = IOIDetectionConfig()

        def read_config():
            for _ in range(100):
                _ = config.name_mover_threshold
                _ = config.s_inhibition_threshold
            return True

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(read_config) for _ in range(10)]
            results = [f.result() for f in futures]

        assert all(results)


class TestMemoryEdgeCases:
    """Tests for memory handling."""

    def test_large_batch_size(self):
        """Test with large batch size."""
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

        # Large batch size
        patterns = {
            9: np.random.rand(32, 12, 10, 10).astype(np.float32),
        }

        detector = IOIDetector(MockMicroscope())
        result = detector.detect_from_attention(patterns, sentence)
        assert isinstance(result, IOICircuit)

    def test_repeated_detection(self):
        """Test repeated detection doesn't leak memory."""
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

        detector = IOIDetector(MockMicroscope())

        for _ in range(100):
            patterns = {
                9: np.random.rand(1, 12, 10, 10).astype(np.float32),
            }
            result = detector.detect_from_attention(patterns, sentence)
            del patterns
            del result

        # If we get here without error, test passes
        assert True
