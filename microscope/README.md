# Alignment Microscope 🔬

**High-performance interpretability toolkit for transformer models**

Part of the [Alignment Observatory](https://github.com/jonnasson/alignment-observatory) project.

## Overview

Alignment Microscope helps you understand what happens inside neural networks, with a focus on AI alignment research. It enables:

- **Activation Tracing**: Track how information flows through model layers
- **Attention Analysis**: Visualize and classify attention head behaviors
- **Circuit Discovery**: Automatically identify computational circuits
- **IOI Circuit Detection**: Detect Indirect Object Identification circuits (Wang et al. 2022)
- **Causal Interventions**: Understand which components cause specific outputs
- **Full Type Support**: IDE autocomplete with comprehensive `.pyi` type stubs

## Installation

```bash
pip install alignment-microscope
```

For development:

```bash
git clone https://github.com/jonnasson/alignment-observatory
cd alignment-observatory/microscope
pip install maturin
maturin develop
```

## Quick Start

```python
from alignment_microscope import Microscope
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Create microscope
scope = Microscope.for_model(model)

# Trace activations
inputs = tokenizer("The capital of France is", return_tensors="pt")

with scope.trace() as trace:
    outputs = model(**inputs)

# Analyze attention patterns
for layer in range(model.config.num_hidden_layers):
    pattern = trace.attention(layer)
    if pattern is not None:
        head_types = scope.classify_heads(pattern)
        print(f"Layer {layer}: {head_types}")

# Get residual stream norms (useful for detecting anomalies)
for layer in [0, 15, 31]:
    norms = trace.token_norms(layer)
    print(f"Layer {layer} norms: {norms}")
```

## Circuit Discovery

Find the circuit responsible for a specific behavior:

```python
from alignment_microscope import Microscope, Circuit

# Run clean and corrupt inputs
with scope.trace() as clean_trace:
    model(clean_input_ids)

with scope.trace() as corrupt_trace:
    model(corrupt_input_ids)

# Discover circuit
circuit = scope.discover_circuit(
    behavior="indirect_object_identification",
    clean_trace=clean_trace,
    corrupt_trace=corrupt_trace,
)

# Visualize with Graphviz
print(circuit.to_dot())
```

## IOI Circuit Detection

Detect Indirect Object Identification circuits based on [Wang et al. 2022](https://arxiv.org/abs/2211.00593):

```python
from alignment_microscope import IOIDetector, IOISentence, KnownIOIHeads

# Create IOI sentence with token roles
sentence = IOISentence.from_positions(
    tokens=token_ids,
    token_strings=["When", "John", "and", "Mary", "went", "to", "store", ",", "Mary", "gave", "to"],
    subject_positions=[3, 8],  # "Mary" positions
    io_position=1,             # "John" position
    subject2_position=8,       # Second "Mary"
    end_position=10,           # "to" (prediction position)
    correct_answer="John",
    distractor="Mary",
)

# Detect IOI circuit components
detector = IOIDetector(microscope)
circuit = detector.detect_from_attention(attention_patterns, sentence)

# Examine detected heads
print(f"Name Mover heads: {[(h.layer, h.head) for h in circuit.name_mover_heads]}")
print(f"S-Inhibition heads: {[(h.layer, h.head) for h in circuit.s_inhibition_heads]}")
print(f"Duplicate Token heads: {[(h.layer, h.head) for h in circuit.duplicate_token_heads]}")
print(f"Circuit validity: {circuit.validity_score:.2f}")

# Validate against known GPT-2 heads
validation = circuit.validate_against_known("gpt2")
print(f"Precision: {validation.precision:.2f}, Recall: {validation.recall:.2f}")

# Export circuit visualization
print(circuit.to_dot())  # Graphviz DOT format
```

**IOI Circuit Components:**

- **Name Mover Heads** (L9-11): Move IO name to final position
- **S-Inhibition Heads** (L7-8): Inhibit subject from being copied
- **Duplicate Token Heads** (L0-2): Detect repeated tokens
- **Previous Token Heads** (L0-2): Track token sequences
- **Backup Name Mover Heads** (L9-11): Redundant name movers

**Known GPT-2 Heads:**

```python
known = KnownIOIHeads.all_gpt2()
# Returns: {
#   "name_mover": [(9,9), (10,0), (9,6)],
#   "s_inhibition": [(7,3), (7,9), (8,6), (8,10)],
#   "duplicate_token": [(0,1), (0,10), (3,0)],
#   ...
# }
```

## Attention Analysis

Classify attention heads and find induction heads:

```python
from alignment_microscope import AttentionPattern

# Get attention pattern
pattern = AttentionPattern(layer=5, pattern=trace.attention(5))

# Compute entropy (higher = more uniform attention)
entropy = pattern.entropy()

# Find top attended positions
top_k = pattern.top_attended(k=5)

# Classify head types
types = scope.classify_heads(pattern)
# Returns: ["previous_token", "bos", "induction", ...]
```

## Architecture

```
alignment-microscope/
├── src/                    # Rust core (high performance)
│   ├── lib.rs             # Main entry point, error types
│   ├── activation.rs      # Activation tracing
│   ├── attention.rs       # Attention analysis (+ IOI head types)
│   ├── circuit.rs         # Circuit discovery (+ IOI detection)
│   ├── hooks.rs           # Model hooks
│   ├── intervention.rs    # Causal interventions
│   ├── python.rs          # Python bindings
│   ├── sae.rs             # Sparse autoencoder support
│   └── streaming.rs       # Memory-efficient streaming
├── python/                 # Python package
│   └── alignment_microscope/
│       ├── __init__.py    # High-level Python API
│       ├── ioi.py         # IOI circuit detection
│       ├── sae.py         # SAE integration
│       ├── streaming.py   # Streaming capture
│       ├── *.pyi          # Type stubs for IDE support
│       └── architectures/ # Multi-model support
├── tests/                  # 200+ unit tests
│   ├── test_ioi.py        # IOI circuit tests
│   ├── test_edge_cases.py # Edge case coverage
│   └── ...
└── examples/              # Usage examples
```

## Performance

The Rust core provides:

- Zero-copy analysis where possible
- Parallel computation across layers/heads
- Memory-efficient processing for large models (70B+)
- Streaming support for real-time interpretation

## Roadmap

Part of the [Alignment Observatory](https://github.com/jonnasson/alignment-observatory) project:

| Component        | Status      | Description                            |
| ---------------- | ----------- | -------------------------------------- |
| **Microscope**   | ✅ Complete | Interpretability engine (this package) |
| **Dashboard**    | 80%         | WebGL visualization (Vue + Three.js)   |
| **Benchmarks**   | Planned     | Alignment evaluation suite             |
| **Red Team**     | Planned     | Adversarial testing framework          |
| **Guardrails**   | Planned     | Runtime monitoring                     |
| **Construction** | Planned     | Aligned-by-design training             |

See [ROADMAP.md](../ROADMAP.md) for detailed progress.

## Contributing

We welcome contributions!

Key areas where help is needed:

- **Dashboard polish** - Integration testing, E2E tests
- Additional circuit discovery algorithms
- Performance benchmarking and optimization
- Documentation and examples

## License

Apache 2.0

## Citation

If you use this in research, please cite:

```bibtex
@software{alignment_microscope,
  title = {Alignment Microscope: Interpretability Toolkit for Transformer Models},
  author = {Alignment Observatory},
  year = {2026},
  url = {https://github.com/jonnasson/alignment-observatory}
}
```

## Related Projects

- [TransformerLens](https://github.com/neelnanda-io/TransformerLens)
- [nnsight](https://github.com/ndif-team/nnsight)
- [Baukit](https://github.com/davidbau/baukit)
- [Anthropic's Circuits Work](https://transformer-circuits.pub/)
