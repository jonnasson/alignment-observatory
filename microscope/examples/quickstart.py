#!/usr/bin/env python3
"""
Alignment Microscope - Quick Start Example

This script demonstrates the core capabilities of the Alignment Microscope:
1. Tracing activations through a model
2. Analyzing attention patterns
3. Discovering computational circuits
4. Performing causal interventions

Run with: python quickstart.py
"""

import numpy as np
from typing import Optional

# Check if we can use real models
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Note: transformers not installed. Using synthetic data for demonstration.")

from alignment_microscope import (
    Microscope,
    ActivationTrace,
    AttentionPattern,
    Circuit,
)


def create_synthetic_trace(num_layers: int = 32) -> ActivationTrace:
    """Create a synthetic trace for demonstration without a real model."""
    trace = ActivationTrace()
    
    seq_len = 10
    hidden_size = 4096
    num_heads = 32
    
    for layer in range(num_layers):
        # Residual stream
        residual = np.random.randn(1, seq_len, hidden_size).astype(np.float32)
        trace._activations[f"{layer}_residual"] = residual
        
        # Attention output
        attn_out = np.random.randn(1, seq_len, hidden_size).astype(np.float32)
        trace._activations[f"{layer}_attn_out"] = attn_out
        
        # MLP output
        mlp_out = np.random.randn(1, seq_len, hidden_size).astype(np.float32)
        trace._activations[f"{layer}_mlp_out"] = mlp_out
        
        # Attention pattern - create a realistic pattern
        pattern = np.zeros((1, num_heads, seq_len, seq_len), dtype=np.float32)
        
        # Some heads attend to previous token
        for h in range(0, num_heads, 4):
            for i in range(seq_len):
                if i > 0:
                    pattern[0, h, i, i-1] = 0.8
                    pattern[0, h, i, :i-1] = 0.2 / max(i-1, 1)
                else:
                    pattern[0, h, i, i] = 1.0
        
        # Some heads attend to BOS
        for h in range(1, num_heads, 4):
            pattern[0, h, :, 0] = 0.9
            for i in range(seq_len):
                pattern[0, h, i, 1:] = 0.1 / max(seq_len-1, 1)
        
        # Some heads are more uniform
        for h in range(2, num_heads, 4):
            for i in range(seq_len):
                pattern[0, h, i, :i+1] = 1.0 / (i+1)
        
        # Remaining heads have mixed patterns
        for h in range(3, num_heads, 4):
            pattern[0, h] = np.random.softmax(
                np.random.randn(seq_len, seq_len), axis=-1
            ).astype(np.float32) if hasattr(np.random, 'softmax') else (
                np.exp(np.random.randn(seq_len, seq_len)) / 
                np.exp(np.random.randn(seq_len, seq_len)).sum(axis=-1, keepdims=True)
            ).astype(np.float32)
        
        trace._attention_patterns[layer] = pattern
    
    return trace


def demo_activation_tracing():
    """Demonstrate activation tracing."""
    print("\n" + "="*60)
    print("DEMO 1: Activation Tracing")
    print("="*60)
    
    scope = Microscope.for_llama(num_layers=32, num_heads=32, hidden_size=4096)
    
    # Create a synthetic trace (in real use, this comes from model inference)
    trace = create_synthetic_trace(num_layers=32)
    
    print(f"\nCaptured activations for {len(trace.layers)} layers")
    print(f"Layers: {trace.layers[:5]}... (showing first 5)")
    
    # Analyze residual stream norms
    print("\nResidual stream L2 norms per position:")
    for layer in [0, 15, 31]:
        norms = trace.token_norms(layer, "residual")
        if norms is not None:
            print(f"  Layer {layer:2d}: mean={norms.mean():.3f}, std={norms.std():.3f}")
    
    # Look at activation statistics
    print("\nActivation statistics:")
    for layer in [0, 15, 31]:
        residual = trace.residual(layer)
        if residual is not None:
            print(f"  Layer {layer:2d}: shape={residual.shape}, "
                  f"mean={residual.mean():.3f}, std={residual.std():.3f}")


def demo_attention_analysis():
    """Demonstrate attention pattern analysis."""
    print("\n" + "="*60)
    print("DEMO 2: Attention Analysis")
    print("="*60)
    
    scope = Microscope.for_llama(num_layers=32, num_heads=32, hidden_size=4096)
    trace = create_synthetic_trace(num_layers=32)
    
    # Analyze attention patterns
    print("\nAttention head classification by layer:")
    
    for layer in [0, 5, 15, 31]:
        pattern_data = trace.attention(layer)
        if pattern_data is not None:
            head_types = scope.classify_heads(pattern_data)
            
            # Count each type
            type_counts = {}
            for t in head_types:
                type_counts[t] = type_counts.get(t, 0) + 1
            
            print(f"  Layer {layer:2d}: {type_counts}")
    
    # Compute entropy for layer 15
    print("\nAttention entropy (higher = more uniform):")
    pattern_data = trace.attention(15)
    if pattern_data is not None:
        pattern = AttentionPattern(15, pattern_data)
        entropy = pattern.entropy()
        
        # Show entropy statistics per head
        for head in range(min(8, pattern.num_heads)):
            head_entropy = entropy[0, head].mean()
            print(f"  Head {head}: entropy={head_entropy:.3f}")


def demo_circuit_discovery():
    """Demonstrate circuit discovery."""
    print("\n" + "="*60)
    print("DEMO 3: Circuit Discovery")
    print("="*60)
    
    scope = Microscope.for_llama()
    
    # Create clean and corrupt traces
    clean_trace = create_synthetic_trace(num_layers=32)
    corrupt_trace = create_synthetic_trace(num_layers=32)
    
    # Make the corrupt trace different in specific layers
    # (simulating a behavioral change)
    for layer in [5, 10, 15, 20]:
        key = f"{layer}_attn_out"
        if key in corrupt_trace._activations:
            corrupt_trace._activations[key] *= 0.5  # Reduce activation magnitude
    
    # Discover circuit
    circuit = scope.discover_circuit(
        behavior="demonstration_circuit",
        clean_trace=clean_trace,
        corrupt_trace=corrupt_trace,
    )
    
    print(f"\nDiscovered circuit: {circuit.name}")
    print(f"Number of nodes: {len(circuit.nodes)}")
    print(f"Number of edges: {len(circuit.edges)}")
    
    print("\nNodes in circuit:")
    for layer, component, head in sorted(circuit.nodes)[:10]:
        head_str = f" (head {head})" if head is not None else ""
        print(f"  Layer {layer:2d} - {component}{head_str}")
    
    # Generate DOT visualization
    print("\nDOT format for Graphviz:")
    print("-" * 40)
    dot = circuit.to_dot()
    # Print first few lines
    for line in dot.split("\n")[:15]:
        print(line)
    if len(dot.split("\n")) > 15:
        print("  ... (truncated)")


def demo_manual_circuit():
    """Demonstrate manually creating a circuit (e.g., from research papers)."""
    print("\n" + "="*60)
    print("DEMO 4: Manual Circuit Definition")
    print("="*60)
    
    # Recreate a known circuit (simplified Indirect Object Identification)
    circuit = Circuit(
        name="IOI Circuit (Simplified)",
        description="Circuit for Indirect Object Identification task",
        behavior="identify_indirect_object"
    )
    
    # Add nodes based on the paper
    # Previous token heads (early layers)
    circuit.add_node(0, "attention", head=3)
    circuit.add_node(1, "attention", head=5)
    
    # Duplicate token heads
    circuit.add_node(3, "attention", head=0)
    
    # S-inhibition heads
    circuit.add_node(7, "attention", head=3)
    circuit.add_node(8, "attention", head=6)
    
    # Name mover heads (late layers)
    circuit.add_node(9, "attention", head=9)
    circuit.add_node(10, "attention", head=0)
    
    # Add edges with importance scores
    circuit.add_edge((0, "attention", 3), (3, "attention", 0), 0.7)
    circuit.add_edge((1, "attention", 5), (3, "attention", 0), 0.6)
    circuit.add_edge((3, "attention", 0), (7, "attention", 3), 0.8)
    circuit.add_edge((3, "attention", 0), (8, "attention", 6), 0.7)
    circuit.add_edge((7, "attention", 3), (9, "attention", 9), 0.9)
    circuit.add_edge((8, "attention", 6), (9, "attention", 9), 0.85)
    circuit.add_edge((9, "attention", 9), (10, "attention", 0), 0.95)
    
    print(f"\nCircuit: {circuit.name}")
    print(f"Nodes: {len(circuit.nodes)}")
    print(f"Edges: {len(circuit.edges)}")
    
    # Get minimal circuit
    minimal = circuit.minimal(threshold=0.8)
    print(f"\nMinimal circuit (threshold 0.8):")
    print(f"  Edges: {len(minimal.edges)}")
    
    print("\nFull DOT output:")
    print(circuit.to_dot())


def main():
    """Run all demonstrations."""
    print("="*60)
    print("Alignment Microscope - Demonstration")
    print("="*60)
    print("\nThis demo shows the core capabilities of the interpretability toolkit.")
    
    if HAS_TRANSFORMERS:
        print("\n✓ PyTorch and Transformers are available")
    else:
        print("\n⚠ Using synthetic data (install torch & transformers for real models)")
    
    # Run demos
    demo_activation_tracing()
    demo_attention_analysis()
    demo_circuit_discovery()
    demo_manual_circuit()
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Install with: pip install alignment-microscope")
    print("2. Use with real models from HuggingFace")
    print("3. Discover circuits in your own models")
    print("4. Contribute at: github.com/alignment-observatory/microscope")


if __name__ == "__main__":
    main()
