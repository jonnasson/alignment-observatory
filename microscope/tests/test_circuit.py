"""Tests for Circuit class."""

import pytest
from alignment_microscope import Circuit


class TestCircuit:
    """Unit tests for Circuit."""

    def test_create_circuit(self):
        """Test creating an empty circuit."""
        circuit = Circuit(name="Test", description="Test circuit", behavior="test")
        assert circuit.name == "Test"
        assert len(circuit.nodes) == 0
        assert len(circuit.edges) == 0

    def test_add_node(self):
        """Test adding nodes to circuit."""
        circuit = Circuit("Test", "", "")
        circuit.add_node(0, "attention", head=3)
        circuit.add_node(1, "mlp")

        assert len(circuit.nodes) == 2

    def test_add_edge(self, sample_circuit):
        """Test adding edges creates nodes automatically."""
        assert len(sample_circuit.edges) == 2
        assert len(sample_circuit.nodes) >= 2

    def test_minimal_circuit(self, sample_circuit):
        """Test extracting minimal circuit above threshold."""
        minimal = sample_circuit.minimal(threshold=0.7)

        # Only the 0.8 edge should remain
        assert len(minimal.edges) == 1
        edges = minimal.edges
        assert edges[0][2] == 0.8  # importance

    def test_minimal_circuit_empty(self, sample_circuit):
        """Test minimal circuit with high threshold returns empty."""
        minimal = sample_circuit.minimal(threshold=0.99)
        assert len(minimal.edges) == 0

    def test_to_dot(self, sample_circuit):
        """Test DOT format export."""
        dot = sample_circuit.to_dot()

        assert "digraph Circuit" in dot
        assert "L0H3" in dot  # Node label for attention head
        assert "L1H5" in dot
        assert "L2" in dot  # Layer 2
        assert "->" in dot  # Edge

    def test_to_dot_valid_syntax(self, sample_circuit):
        """Test that DOT output has valid syntax structure."""
        dot = sample_circuit.to_dot()

        # Check structure
        assert dot.startswith("digraph Circuit {")
        assert dot.strip().endswith("}")
        assert "rankdir=TB" in dot
        assert "node [shape=box]" in dot

    def test_edges_property(self, sample_circuit):
        """Test edges property returns correct format."""
        edges = sample_circuit.edges
        assert len(edges) == 2
        # Each edge is (from_tuple, to_tuple, importance)
        assert all(len(e) == 3 for e in edges)
        assert all(isinstance(e[2], float) for e in edges)

    def test_total_importance(self, sample_circuit):
        """Test total importance calculation."""
        total = sum(e[2] for e in sample_circuit.edges)
        expected = 0.8 + 0.6  # Sum of edge importances
        assert abs(total - expected) < 0.01


class TestCircuitEdgeCases:
    """Edge case tests for Circuit."""

    def test_empty_circuit_dot(self):
        """Test DOT output for empty circuit."""
        circuit = Circuit("Empty", "", "")
        dot = circuit.to_dot()

        assert "digraph Circuit" in dot
        # Should still be valid DOT
        assert dot.strip().endswith("}")

    def test_single_node_circuit(self):
        """Test circuit with single node."""
        circuit = Circuit("Single", "", "")
        circuit.add_node(0, "attention", head=0)

        dot = circuit.to_dot()
        assert "L0H0" in dot

    def test_edge_importance_bounds(self):
        """Test edges with various importance values."""
        circuit = Circuit("Test", "", "")

        # Zero importance
        circuit.add_edge((0, "attention", 0), (1, "attention", 0), 0.0)
        # Full importance
        circuit.add_edge((1, "attention", 0), (2, "attention", 0), 1.0)

        assert len(circuit.edges) == 2

    def test_deep_circuit(self):
        """Test circuit with many layers."""
        circuit = Circuit("Deep", "A deep circuit", "deep_behavior")

        for i in range(100):
            circuit.add_node(i, "mlp")
            if i > 0:
                circuit.add_edge((i - 1, "mlp", None), (i, "mlp", None), 0.5)

        assert len(circuit.nodes) == 100
        assert len(circuit.edges) == 99

    def test_wide_circuit(self):
        """Test circuit with many parallel heads."""
        circuit = Circuit("Wide", "A wide circuit", "wide_behavior")

        # Add many heads at same layer
        for head in range(64):
            circuit.add_node(0, "attention", head=head)

        assert len(circuit.nodes) == 64
