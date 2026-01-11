# Alignment Observatory - Project Roadmap

> **Mission:** Build the definitive open-source toolkit for AI alignment research.

Last Updated: January 2026

---

## Project Status Overview

| Component | Status | Progress |
|-----------|--------|----------|
| Microscope (Interpretability) | **Complete** | ██████████ 100% |
| Dashboard (Visualization) | **Next Up** | ░░░░░░░░░░ 0% |
| Benchmarks | Not Started | ░░░░░░░░░░ 0% |
| Red Team | Not Started | ░░░░░░░░░░ 0% |
| Guardrails | Not Started | ░░░░░░░░░░ 0% |
| Construction | Not Started | ░░░░░░░░░░ 0% |

---

## Year 1: Alignment Microscope (Current Focus)

### Completed ✅

#### Rust Core (`microscope/src/`)
- [x] **lib.rs** - Main entry point, configuration, error types
- [x] **activation.rs** - Activation tracing and analysis
  - `Activation` struct for storing captured activations
  - `ActivationTrace` for complete forward pass traces
  - `ActivationTracer` for managing capture during inference
  - `ActivationPatcher` for comparing clean/corrupt runs
  - Token norm computation, statistics, top-k dimensions
- [x] **attention.rs** - Attention pattern analysis
  - `AttentionPattern` for storing attention weights
  - Entropy computation per head
  - Head type classification (previous_token, BOS, uniform, induction, **name_mover, s_inhibition, backup_name_mover**)
  - `AttentionAnalyzer` for cross-layer analysis
  - Induction head detection
  - Attention flow computation
  - Robust NaN handling with `total_cmp()` for sorting
- [x] **circuit.rs** - Circuit discovery
  - `CircuitNode` and `CircuitEdge` types
  - `Circuit` container with DOT export for Graphviz
  - `CircuitDiscoverer` with automatic pattern detection
  - Minimal circuit extraction (threshold-based pruning)
  - Support for known patterns (induction circuits)
  - **IOI Circuit Detection** (Wang et al. 2022):
    - `IOITokenRole` enum (Subject, IndirectObject, SubjectRepeat, EndPosition)
    - `IOISentence` for token role annotations
    - `IOIDetectionConfig` with thresholds and layer ranges
    - `IOIHead` and `IOICircuitResult` types
    - `find_ioi_circuit()`, `find_name_mover_heads()`, `find_s_inhibition_heads()`
    - `find_duplicate_token_heads()`, `find_previous_token_heads()`
    - `KnownIOIHeads` with GPT-2 ground truth for validation
- [x] **hooks.rs** - Model hook system
  - `HookRegistry` for managing hooks by name and hook point
  - Standard hook points (embed, attn_out, mlp_out, residual, etc.)
  - `HookBuilder` for common hook configurations
  - Zero ablation, mean ablation, patching hooks
- [x] **intervention.rs** - Causal intervention tools
  - `Intervention` types (zero/mean ablation, patching, noise, scaling)
  - Position masking for targeted interventions
  - `InterventionResult` for measuring effects
  - `InterventionEngine` for orchestrating experiments
  - Layer importance ranking
- [x] **python.rs** - PyO3 Python bindings
  - `PyMicroscope` wrapper class
  - `PyActivationTrace` with numpy integration
  - `PyAttentionPattern` with analysis methods
  - `PyCircuit` with DOT export
  - `PyAttentionAnalyzer` and `PyCircuitDiscoverer`
  - `PyInterventionEngine` for Python-based experiments
- [x] **sae.rs** - Sparse Autoencoder support
  - `SAEConfig`, `SAEWeights`, `SAEFeatures` types
  - Encode/decode operations with ReLU, TopK, JumpReLU activations
  - Feature frequency and co-activation analysis
  - `FeatureAnalyzer` for pattern-specific feature detection
- [x] **streaming.rs** - Streaming activation capture
  - `StreamingConfig` for memory management
  - `ActivationStorage` with memory-mapped file support
  - `ActivationRingBuffer` for sliding window analysis
  - `MemoryEstimator` for capture strategy recommendations

#### Python Package (`microscope/python/`)
- [x] **alignment_microscope/__init__.py** - High-level Python API
  - `Microscope` class with HuggingFace integration
  - `ActivationTrace` with numpy-based storage and public accessors
  - `AttentionPattern` with entropy and top-k methods
  - `Circuit` class with manual construction support
  - Context manager for tracing (`with scope.trace()`)
  - Head classification methods
  - Circuit discovery from trace comparison
  - IOI module exports (IOIDetector, IOISentence, IOICircuit, etc.)
- [x] **alignment_microscope/ioi.py** - IOI Circuit Detection (~600 lines)
  - `IOISentence` with `parse()` and `from_positions()` class methods
  - `IOIHead`, `IOICircuit`, `IOIValidationResult`, `IOIDetectionConfig` dataclasses
  - `KnownIOIHeads` with GPT-2 ground truth heads for validation
  - `IOIDetector` with `detect()` and `detect_from_attention()` methods
  - `compute_logit_diff()` static method for IOI metric
  - DOT export for circuit visualization
  - Validation against known heads with precision/recall/F1 metrics
- [x] **alignment_microscope/sae.py** - SAE integration
  - `SAEConfig`, `SAEFeatures`, `SAEWrapper` classes
  - SAELens integration for pre-trained SAE loading
  - `SAEAnalyzer` for cross-trace feature analysis
  - Top-k features, sparsity metrics, feature frequency
- [x] **alignment_microscope/streaming.py** - Streaming capture
  - `StreamingConfig` for configuring capture strategy
  - `StreamingTrace` with lazy disk-backed loading
  - `StreamingMicroscope` for large model analysis
  - `RingBuffer` for efficient sliding window
  - `MemoryEstimator` with strategy recommendations
- [x] **alignment_microscope/architectures/** - Multi-architecture support
  - `ArchitectureAdapter` base class with unified interface
  - `Architecture` enum with auto-detection
  - `AdapterRegistry` with fallback chain
  - Adapters: `LlamaAdapter`, `GPT2Adapter`, `MistralAdapter`, `QwenAdapter`, `GemmaAdapter`

#### Type Stubs (`microscope/python/`)
- [x] **py.typed** - PEP 561 marker for type checking
- [x] **__init__.pyi** - Main API type stubs
- [x] **_core.pyi** - Rust binding type stubs
- [x] **sae.pyi** - SAE module type stubs
- [x] **streaming.pyi** - Streaming module type stubs
- [x] **ioi.pyi** - IOI module type stubs
- [x] **architectures/__init__.pyi** - Architecture exports
- [x] **architectures/base.pyi** - ArchitectureAdapter type stubs
- [x] **architectures/detection.pyi** - Architecture enum type stubs
- [x] **architectures/registry.pyi** - AdapterRegistry type stubs

#### Testing (`microscope/tests/`)
- [x] **conftest.py** - Pytest fixtures for all tests
- [x] **test_activation.py** - 16 tests for ActivationTrace
- [x] **test_attention.py** - 12 tests for AttentionPattern
- [x] **test_circuit.py** - 14 tests for Circuit
- [x] **test_microscope.py** - 12 tests for Microscope
- [x] **test_architectures.py** - 26 tests for architecture adapters
- [x] **test_sae.py** - 24 tests for SAE module
- [x] **test_streaming.py** - 32 tests for streaming module
- [x] **test_integration.py** - 9 GPT-2 integration tests
- [x] **test_ioi.py** - 25+ tests for IOI circuit detection (~350 lines)
  - `TestIOISentence` - sentence parsing and positions
  - `TestIOIHead` - head data structure
  - `TestIOIDetectionConfig` - configuration options
  - `TestKnownIOIHeads` - GPT-2 ground truth validation
  - `TestIOIDetector` - detection algorithms
  - `TestIOICircuit` - circuit representation and DOT export
  - `TestIOILogitDiff` - logit difference computation
  - `TestIOICircuitEdgeCases` - boundary conditions
- [x] **test_edge_cases.py** - 35+ edge case tests (~400 lines)
  - `TestNumericalStability` - NaN, Inf, underflow, mixed precision
  - `TestInvalidInputs` - empty tokens, mismatched lengths, out-of-bounds
  - `TestBoundaryConditions` - single token, many layers, long sequences
  - `TestConfigurationEdgeCases` - zero/one thresholds, empty ranges
  - `TestIOIHeadEdgeCases` - metrics, equality
  - `TestValidationEdgeCases` - empty circuits, perfect matches
  - `TestDOTExport` - special characters, empty graphs
  - `TestConcurrency` - parallel detection, shared config
  - `TestMemoryEdgeCases` - large batches, repeated detection

**Total: 200+ unit tests passing**

#### Project Configuration
- [x] **Cargo.toml** - Rust dependencies configured
  - ndarray, rayon for computation
  - PyO3, numpy for Python bindings
  - serde for serialization
  - tokio for async support
- [x] **pyproject.toml** - Python packaging with maturin
  - Optional dependencies for viz, transformers
  - Development dependencies for testing
- [x] **README.md** - Documentation with examples
- [x] **examples/quickstart.py** - Demonstration script

#### Build Validation ✅
- [x] **Build and test** - `maturin develop` builds successfully
- [x] **Quickstart validation** - `examples/quickstart.py` runs correctly
- [x] **Module imports** - All Python modules import cleanly

### Recently Completed ✅

- [x] **IOI Circuit Detection** - Full implementation based on Wang et al. 2022
- [x] **Known circuit validation** - GPT-2 ground truth heads for validation
- [x] **Python type stubs** - 10+ `.pyi` files for full IDE support
- [x] **Error handling hardening** - Replaced unsafe unwraps with robust patterns
- [x] **Edge case testing** - 35+ tests for numerical stability, concurrency, memory

### Next Up 📋

#### Month 1-2: Core Polish ✅ COMPLETE
- [x] Add support for more model architectures (Mistral, Qwen, Gemma)
- [x] Implement sparse autoencoders (SAE) integration
- [x] Add streaming activation capture for large models
- [x] Create unit tests for all Python modules
- [x] Add Python type stubs for IDE support
- [x] IOI circuit detection and validation
- [ ] Performance benchmarking and optimization (optional)

#### Month 3-4: Visualization Dashboard
- [ ] Set up VueJS project in `dashboard/`
- [ ] WebGL-based attention visualization (using Three.js or PixiJS)
- [ ] Interactive circuit explorer with graph layout
- [ ] Activation heatmaps with zoom/pan
- [ ] Real-time tracing UI with WebSocket updates
- [ ] Feature activation browser for SAE analysis

#### Month 5-6: Advanced Features
- [ ] Concept probing (find where concepts are encoded)
- [ ] Automated circuit discovery improvements
- [ ] Logit lens implementation
- [ ] Residual stream decomposition
- [ ] Memory-efficient analysis for 70B+ models (streaming is ready)
- [ ] Activation patching experiments UI

### Year 1 Deliverables Checklist
- [x] Core interpretability engine (Rust + Python)
- [x] Support for Llama, Mistral, GPT, Qwen, Gemma architectures
- [x] SAE integration with SAELens compatibility
- [x] Streaming support for large models
- [x] Comprehensive test suite (200+ tests)
- [x] IOI circuit detection (Wang et al. 2022)
- [x] Python type stubs for IDE support
- [x] Robust error handling throughout
- [ ] Web dashboard for visualization (Month 3-4)
- [ ] Paper on novel circuit discovery methods

---

## Year 2: Alignment Benchmarks (Future)

### Planned Components
```
alignment-observatory/
├── benchmarks/
│   ├── deception_detection/
│   ├── goal_stability/
│   ├── corrigibility/
│   ├── value_consistency/
│   ├── power_seeking/
│   └── specification_gaming/
```

### Key Features
- 1000+ curated alignment test cases
- Adversarial benchmark generation
- Multi-turn evaluation
- Public leaderboard
- HuggingFace integration

---

## Year 3: Adversarial Red-Teaming (Future)

### Planned Components
```
alignment-observatory/
├── red_team/
│   ├── jailbreak_generator/
│   ├── goal_hijacking/
│   ├── capability_elicitation/
│   ├── deceptive_alignment/
│   └── multi_agent_failures/
```

---

## Year 4: Runtime Guardrails (Future)

### Planned Components
```
alignment-observatory/
├── guardrails/
│   ├── runtime_monitor/
│   ├── anomaly_detector/
│   ├── circuit_breaker/
│   ├── audit_logger/
│   └── human_escalation/
```

---

## Year 5: Aligned by Construction (Future)

### Planned Components
```
alignment-observatory/
├── construction/
│   ├── value_specification/
│   ├── verified_training/
│   ├── constitutional_engine/
│   └── proof_of_alignment/
```

---

## File Structure (Current)

```
alignment-observatory/
├── ROADMAP.md                 # This file
├── microscope/
│   ├── Cargo.toml             # Rust config
│   ├── pyproject.toml         # Python config
│   ├── README.md              # Documentation
│   ├── src/
│   │   ├── lib.rs             # ~170 lines (+ error types)
│   │   ├── activation.rs      # ~380 lines (+ try_as_array, safe locks)
│   │   ├── attention.rs       # ~360 lines (+ IOI head types, total_cmp)
│   │   ├── circuit.rs         # ~1100 lines (+ full IOI detection)
│   │   ├── hooks.rs           # ~350 lines (+ safe lock handling)
│   │   ├── intervention.rs    # ~420 lines (+ total_cmp sorting)
│   │   ├── python.rs          # ~350 lines
│   │   ├── sae.rs             # ~440 lines (+ safe array handling)
│   │   └── streaming.rs       # ~540 lines (+ safe file handling)
│   ├── python/
│   │   └── alignment_microscope/
│   │       ├── __init__.py    # ~600 lines (+ IOI exports)
│   │       ├── __init__.pyi   # Type stubs
│   │       ├── _core.pyi      # Rust binding stubs
│   │       ├── ioi.py         # ~600 lines (NEW - IOI detection)
│   │       ├── ioi.pyi        # Type stubs
│   │       ├── sae.py         # ~380 lines
│   │       ├── sae.pyi        # Type stubs
│   │       ├── streaming.py   # ~450 lines
│   │       ├── streaming.pyi  # Type stubs
│   │       ├── py.typed       # PEP 561 marker
│   │       └── architectures/
│   │           ├── __init__.py
│   │           ├── __init__.pyi
│   │           ├── base.py    # ~180 lines
│   │           ├── base.pyi
│   │           ├── detection.py # ~180 lines
│   │           ├── detection.pyi
│   │           ├── registry.py  # ~100 lines
│   │           ├── registry.pyi
│   │           ├── llama.py     # ~80 lines
│   │           ├── gpt2.py      # ~120 lines
│   │           ├── mistral.py   # ~90 lines
│   │           ├── qwen.py      # ~95 lines
│   │           └── gemma.py     # ~100 lines
│   ├── tests/
│   │   ├── conftest.py        # ~150 lines
│   │   ├── test_activation.py # ~120 lines
│   │   ├── test_attention.py  # ~120 lines
│   │   ├── test_circuit.py    # ~140 lines
│   │   ├── test_microscope.py # ~135 lines
│   │   ├── test_architectures.py # ~230 lines
│   │   ├── test_sae.py        # ~280 lines
│   │   ├── test_streaming.py  # ~360 lines
│   │   ├── test_integration.py # ~150 lines
│   │   ├── test_ioi.py        # ~350 lines (NEW - IOI tests)
│   │   └── test_edge_cases.py # ~400 lines (NEW - edge cases)
│   └── examples/
│       └── quickstart.py      # ~200 lines
├── benchmarks/                 # Year 2 (empty)
├── dashboard/                  # Year 1 (VueJS - next up)
└── docs/                       # (empty)
```

**Total code written:** ~9,000+ lines (Rust + Python + Tests + Type Stubs)
- Rust core: ~4,100 lines
- Python package: ~2,600 lines
- Type stubs: ~500 lines
- Tests: ~2,400 lines

---

## Quick Start for New Sessions

```bash
# Navigate to project
cd ~/alignment-observatory/microscope

# Build the Rust library with Python bindings
maturin develop

# Run the example
python examples/quickstart.py

# Run tests
pytest tests/ -v -m "not integration"

# Or use interactively
python -c "from alignment_microscope import Microscope, SAEWrapper, StreamingMicroscope; print('Ready!')"
```

---

## Contributing

Priority areas for contribution:
1. **Dashboard development** - VueJS + WebGL visualization
2. **Circuit discovery algorithms** - Improve automatic detection
3. **Documentation** - Examples, tutorials, API docs
4. **Integration tests** - Real model testing
5. **Performance** - Profiling and optimization

---

## References

- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) - Inspiration for API design
- [nnsight](https://github.com/ndif-team/nnsight) - Intervention patterns
- [Anthropic Circuits](https://transformer-circuits.pub/) - Circuit discovery methodology
- [IOI Paper](https://arxiv.org/abs/2211.00593) - Indirect Object Identification circuit
- [SAELens](https://github.com/jbloomAus/SAELens) - Sparse Autoencoder training and analysis
