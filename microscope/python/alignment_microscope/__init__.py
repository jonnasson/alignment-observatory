"""
Alignment Microscope - Interpretability toolkit for transformer models

This library provides tools for understanding AI model internals,
with a focus on AI alignment research.

Example usage:

    from alignment_microscope import Microscope
    import torch
    from transformers import AutoModelForCausalLM

    # Load your model
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

    # Create microscope
    scope = Microscope.for_model(model)

    # Trace activations during inference
    with scope.trace() as trace:
        outputs = model.generate(input_ids, max_length=100)

    # Analyze attention patterns
    for layer in range(model.config.num_hidden_layers):
        pattern = trace.attention(layer)
        print(f"Layer {layer}: {scope.classify_heads(pattern)}")

    # Discover circuits
    circuit = scope.discover_circuit("induction")
    print(circuit.to_dot())  # Visualize in Graphviz
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Alignment Observatory"

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
import numpy as np

# Import Rust core (will be available after maturin build)
try:
    from alignment_microscope._core import (
        Microscope as _RustMicroscope,
        ActivationTrace as _RustTrace,
        AttentionPattern as _RustAttentionPattern,
        Circuit as _RustCircuit,
        AttentionAnalyzer as _RustAttentionAnalyzer,
        CircuitDiscoverer as _RustCircuitDiscoverer,
        InterventionEngine as _RustInterventionEngine,
    )
    _HAS_RUST_CORE = True
except ImportError:
    _HAS_RUST_CORE = False
    _RustMicroscope = None

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel


class ActivationTrace:
    """
    Container for activations captured during a forward pass.
    
    Provides access to:
    - Residual stream activations at each layer
    - Attention outputs
    - MLP outputs
    - Attention patterns (queries attending to keys)
    """
    
    def __init__(self, rust_trace: Optional[Any] = None):
        self._rust_trace = rust_trace
        self._activations: Dict[str, np.ndarray] = {}
        self._attention_patterns: Dict[int, np.ndarray] = {}
        self._input_tokens: List[int] = []
    
    def add_activation(
        self,
        layer: int,
        component: str,
        data: "torch.Tensor"
    ) -> None:
        """Add an activation to the trace."""
        key = f"{layer}_{component}"
        self._activations[key] = data.detach().cpu().numpy()
    
    def add_attention_pattern(self, layer: int, pattern: "torch.Tensor") -> None:
        """Add an attention pattern."""
        self._attention_patterns[layer] = pattern.detach().cpu().numpy()
    
    def get(self, layer: int, component: str) -> Optional[np.ndarray]:
        """Get activation for a specific layer and component."""
        key = f"{layer}_{component}"
        return self._activations.get(key)
    
    def residual(self, layer: int) -> Optional[np.ndarray]:
        """Get residual stream at a layer."""
        return self.get(layer, "residual")
    
    def attention_out(self, layer: int) -> Optional[np.ndarray]:
        """Get attention output at a layer."""
        return self.get(layer, "attn_out")
    
    def mlp_out(self, layer: int) -> Optional[np.ndarray]:
        """Get MLP output at a layer."""
        return self.get(layer, "mlp_out")
    
    def attention(self, layer: int) -> Optional[np.ndarray]:
        """Get attention pattern at a layer. Shape: [batch, heads, seq, seq]"""
        return self._attention_patterns.get(layer)
    
    @property
    def layers(self) -> List[int]:
        """Get list of layers with captured activations."""
        layers = set()
        for key in self._activations.keys():
            layer = int(key.split("_")[0])
            layers.add(layer)
        return sorted(layers)
    
    def token_norms(self, layer: int, component: str = "residual") -> Optional[np.ndarray]:
        """Compute L2 norm of activations per token position."""
        act = self.get(layer, component)
        if act is None:
            return None
        # Shape: [batch, seq, hidden] -> [batch, seq]
        return np.linalg.norm(act, axis=-1)
    
    @property
    def activations(self) -> Dict[str, np.ndarray]:
        """Get all activations as a dictionary (copy)."""
        return self._activations.copy()

    @property
    def attention_patterns(self) -> Dict[int, np.ndarray]:
        """Get all attention patterns by layer (copy)."""
        return self._attention_patterns.copy()

    @property
    def input_tokens(self) -> List[int]:
        """Get input tokens for this trace (copy)."""
        return self._input_tokens.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Export trace to a dictionary."""
        return {
            "activations": {k: v.tolist() for k, v in self._activations.items()},
            "attention_patterns": {k: v.tolist() for k, v in self._attention_patterns.items()},
            "input_tokens": self._input_tokens,
        }


class AttentionPattern:
    """
    Wrapper for attention patterns with analysis methods.
    """
    
    def __init__(self, layer: int, pattern: np.ndarray):
        self.layer = layer
        self.pattern = pattern  # Shape: [batch, heads, seq_q, seq_k]
    
    @property
    def num_heads(self) -> int:
        return self.pattern.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.pattern.shape[2]
    
    def head(self, head_idx: int, batch: int = 0) -> np.ndarray:
        """Get attention pattern for a specific head. Shape: [seq_q, seq_k]"""
        return self.pattern[batch, head_idx]
    
    def entropy(self) -> np.ndarray:
        """
        Compute entropy of attention distributions.
        Higher entropy = more uniform attention.
        Shape: [batch, heads, seq_q]
        """
        # Clip to avoid log(0)
        p = np.clip(self.pattern, 1e-10, 1.0)
        return -np.sum(p * np.log(p), axis=-1)
    
    def top_attended(self, k: int = 5) -> np.ndarray:
        """Get indices of top-k attended positions for each query."""
        return np.argsort(self.pattern, axis=-1)[..., -k:][:, :, :, ::-1]


class Circuit:
    """
    Represents a computational circuit in the model.
    
    A circuit is a subgraph of the model's computation graph
    that implements a specific behavior.
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        behavior: str = ""
    ):
        self.name = name
        self.description = description
        self.behavior = behavior
        self.nodes: List[Tuple[int, str, Optional[int]]] = []  # (layer, component, head)
        self.edges: List[Tuple[Tuple, Tuple, float]] = []  # (from, to, importance)
    
    def add_node(
        self,
        layer: int,
        component: str,
        head: Optional[int] = None
    ) -> None:
        """Add a node to the circuit."""
        node = (layer, component, head)
        if node not in self.nodes:
            self.nodes.append(node)
    
    def add_edge(
        self,
        from_node: Tuple[int, str, Optional[int]],
        to_node: Tuple[int, str, Optional[int]],
        importance: float = 1.0
    ) -> None:
        """Add an edge between nodes."""
        self.add_node(*from_node)
        self.add_node(*to_node)
        self.edges.append((from_node, to_node, importance))
    
    def minimal(self, threshold: float = 0.5) -> "Circuit":
        """Return a minimal circuit with edges above the threshold."""
        minimal = Circuit(self.name, self.description, self.behavior)
        for from_node, to_node, importance in self.edges:
            if importance >= threshold:
                minimal.add_edge(from_node, to_node, importance)
        return minimal
    
    def to_dot(self) -> str:
        """Export circuit to DOT format for Graphviz visualization."""
        lines = ["digraph Circuit {", "  rankdir=TB;", "  node [shape=box];", ""]
        
        # Add nodes
        for layer, component, head in self.nodes:
            if head is not None:
                label = f"L{layer}H{head}"
            else:
                label = f"L{layer}{component[:3].upper()}"
            lines.append(f'  "{label}" [label="{label}"];')
        
        lines.append("")
        
        # Add edges
        for (fl, fc, fh), (tl, tc, th), importance in self.edges:
            from_label = f"L{fl}H{fh}" if fh is not None else f"L{fl}{fc[:3].upper()}"
            to_label = f"L{tl}H{th}" if th is not None else f"L{tl}{tc[:3].upper()}"
            width = 1.0 + importance * 3.0
            lines.append(
                f'  "{from_label}" -> "{to_label}" '
                f'[penwidth={width:.1f}, label="{importance:.2f}"];'
            )
        
        lines.append("}")
        return "\n".join(lines)


class Microscope:
    """
    Main interface for the interpretability toolkit.
    
    The Microscope attaches hooks to a model to capture activations
    and provides methods for analysis.
    
    Example:
        scope = Microscope.for_model(model)
        
        with scope.trace() as trace:
            model(input_ids)
        
        # Analyze the trace
        print(trace.residual(0).shape)
    """
    
    def __init__(
        self,
        architecture: str = "llama",
        num_layers: int = 32,
        num_heads: int = 32,
        hidden_size: int = 4096,
    ):
        self.architecture = architecture
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self._model: Optional["PreTrainedModel"] = None
        self._hooks: List[Any] = []
        self._current_trace: Optional[ActivationTrace] = None
        
        # Use Rust core if available
        if _HAS_RUST_CORE:
            self._rust = _RustMicroscope(
                architecture, num_layers, num_heads, hidden_size
            )
        else:
            self._rust = None
    
    @classmethod
    def for_model(cls, model: "PreTrainedModel") -> "Microscope":
        """
        Create a Microscope configured for a specific model.
        
        Automatically detects architecture and dimensions.
        """
        config = model.config
        
        # Detect architecture
        arch = config.model_type.lower()
        
        # Get dimensions (handle different config naming conventions)
        num_layers = getattr(config, "num_hidden_layers", 32)
        num_heads = getattr(config, "num_attention_heads", 32)
        hidden_size = getattr(config, "hidden_size", 4096)
        
        scope = cls(
            architecture=arch,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=hidden_size,
        )
        scope._model = model
        return scope
    
    @classmethod
    def for_llama(
        cls,
        num_layers: int = 32,
        num_heads: int = 32,
        hidden_size: int = 4096,
    ) -> "Microscope":
        """Create a Microscope configured for Llama models."""
        return cls("llama", num_layers, num_heads, hidden_size)
    
    def _register_hooks(self) -> None:
        """Register forward hooks on the model."""
        if self._model is None:
            return
        
        import torch
        
        def make_hook(layer_idx: int, component: str):
            def hook(module, input, output):
                if self._current_trace is not None:
                    if isinstance(output, tuple):
                        # Attention returns (output, attn_weights, ...)
                        out = output[0]
                        if len(output) > 1 and output[1] is not None:
                            self._current_trace.add_attention_pattern(
                                layer_idx, output[1]
                            )
                    else:
                        out = output
                    self._current_trace.add_activation(layer_idx, component, out)
            return hook
        
        # Register hooks based on architecture
        if hasattr(self._model, "model") and hasattr(self._model.model, "layers"):
            # Llama-style architecture
            for i, layer in enumerate(self._model.model.layers):
                # Attention output
                if hasattr(layer, "self_attn"):
                    handle = layer.self_attn.register_forward_hook(
                        make_hook(i, "attn_out")
                    )
                    self._hooks.append(handle)
                
                # MLP output
                if hasattr(layer, "mlp"):
                    handle = layer.mlp.register_forward_hook(
                        make_hook(i, "mlp_out")
                    )
                    self._hooks.append(handle)
    
    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    @contextmanager
    def trace(self, input_tokens: Optional[List[int]] = None):
        """
        Context manager for tracing activations.
        
        Example:
            with scope.trace() as trace:
                model(input_ids)
            
            print(trace.residual(0).shape)
        """
        self._current_trace = ActivationTrace()
        if input_tokens:
            self._current_trace._input_tokens = input_tokens
        
        self._register_hooks()
        
        try:
            yield self._current_trace
        finally:
            self._remove_hooks()
            self._current_trace = None
    
    def classify_heads(
        self,
        pattern: Union[np.ndarray, AttentionPattern],
    ) -> List[str]:
        """
        Classify attention heads by their pattern type.
        
        Returns a list of head types like:
        - "previous_token": Attends to previous position
        - "bos": Attends to beginning of sequence
        - "uniform": Uniform attention distribution
        - "induction": Copies patterns from earlier in sequence
        - "other": Unclassified
        """
        if isinstance(pattern, np.ndarray):
            pattern = AttentionPattern(0, pattern)
        
        classifications = []
        for head in range(pattern.num_heads):
            head_pattern = pattern.head(head)
            seq_len = head_pattern.shape[0]
            
            # Check for previous token pattern
            prev_score = 0.0
            for i in range(1, seq_len):
                prev_score += head_pattern[i, i - 1]
            prev_score /= max(seq_len - 1, 1)
            
            # Check for BOS attention
            bos_score = np.mean(head_pattern[:, 0])
            
            # Check for uniform
            entropy = pattern.entropy()[0, head].mean()
            max_entropy = np.log(seq_len)
            uniformity = entropy / max_entropy if max_entropy > 0 else 0
            
            if prev_score > 0.5:
                classifications.append("previous_token")
            elif bos_score > 0.5:
                classifications.append("bos")
            elif uniformity > 0.8:
                classifications.append("uniform")
            else:
                classifications.append("other")
        
        return classifications
    
    def discover_circuit(
        self,
        behavior: str,
        clean_trace: ActivationTrace,
        corrupt_trace: ActivationTrace,
        metric_fn: Optional[Callable[[ActivationTrace], float]] = None,
    ) -> Circuit:
        """
        Discover the circuit responsible for a behavior.
        
        Uses activation patching to identify important components.
        
        Args:
            behavior: Name of the behavior to discover
            clean_trace: Trace from the "clean" (correct behavior) run
            corrupt_trace: Trace from the "corrupt" (wrong behavior) run
            metric_fn: Function to measure the behavior (optional)
        
        Returns:
            Circuit object with the discovered components
        """
        circuit = Circuit(behavior, f"Auto-discovered circuit for {behavior}")
        
        # Compare activations between clean and corrupt
        for layer in clean_trace.layers:
            for component in ["residual", "attn_out", "mlp_out"]:
                clean_act = clean_trace.get(layer, component)
                corrupt_act = corrupt_trace.get(layer, component)
                
                if clean_act is None or corrupt_act is None:
                    continue
                
                # Compute difference
                diff = np.linalg.norm(clean_act - corrupt_act)
                normalized_diff = diff / np.linalg.norm(clean_act)
                
                if normalized_diff > 0.1:  # Threshold for significance
                    circuit.add_node(layer, component)
        
        # Add edges between consecutive significant layers
        sorted_nodes = sorted(circuit.nodes, key=lambda x: x[0])
        for i in range(len(sorted_nodes) - 1):
            from_node = sorted_nodes[i]
            to_node = sorted_nodes[i + 1]
            circuit.add_edge(from_node, to_node, importance=0.5)
        
        return circuit


# Convenience functions
def create_microscope(
    model: Optional["PreTrainedModel"] = None,
    **kwargs
) -> Microscope:
    """
    Create a Microscope, optionally for a specific model.
    
    Args:
        model: A HuggingFace model to analyze (optional)
        **kwargs: Additional configuration options
    
    Returns:
        Configured Microscope instance
    """
    if model is not None:
        return Microscope.for_model(model)
    return Microscope(**kwargs)


# Import SAE module
from alignment_microscope.sae import (
    SAEConfig,
    SAEFeatures,
    SAEWrapper,
    SAEAnalyzer,
)

# Import streaming module
from alignment_microscope.streaming import (
    StreamingConfig,
    StreamingTrace,
    StreamingMicroscope,
    MemoryEstimator,
)

# Import IOI module
from alignment_microscope.ioi import (
    IOIDetector,
    IOISentence,
    IOICircuit,
    IOIHead,
    IOIDetectionConfig,
    IOIValidationResult,
    KnownIOIHeads,
)

# Export public API
__all__ = [
    # Core classes
    "Microscope",
    "ActivationTrace",
    "AttentionPattern",
    "Circuit",
    "create_microscope",
    # SAE classes
    "SAEConfig",
    "SAEFeatures",
    "SAEWrapper",
    "SAEAnalyzer",
    # Streaming classes
    "StreamingConfig",
    "StreamingTrace",
    "StreamingMicroscope",
    "MemoryEstimator",
    # IOI classes
    "IOIDetector",
    "IOISentence",
    "IOICircuit",
    "IOIHead",
    "IOIDetectionConfig",
    "IOIValidationResult",
    "KnownIOIHeads",
    # Metadata
    "__version__",
]
