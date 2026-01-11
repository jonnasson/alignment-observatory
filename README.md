# Alignment Observatory

**The definitive open-source toolkit for AI alignment research.**

Alignment Observatory is a comprehensive platform for understanding, testing, and ensuring AI systems behave as intended. It combines high-performance interpretability tools, visualization dashboards, and alignment benchmarks in a unified research environment.

## Project Status

| Component | Description | Status |
|-----------|-------------|--------|
| **Microscope** | Interpretability engine (Rust + Python) | Complete |
| **API** | FastAPI backend for the dashboard | Complete |
| **Dashboard** | WebGL visualization interface | In Progress |
| **Benchmarks** | Alignment evaluation suite | Planned (Year 2) |

## Repository Structure

```
alignment-observatory/
├── microscope/          # High-performance interpretability toolkit
│   ├── src/             # Rust core (~4,100 lines)
│   ├── python/          # Python bindings (~2,600 lines)
│   └── tests/           # 200+ unit tests
│
├── api/                 # FastAPI backend
│   ├── routers/         # REST endpoints
│   ├── services/        # Business logic
│   ├── schemas/         # Pydantic models
│   └── websockets/      # Real-time streaming
│
├── dashboard/           # Vue.js visualization frontend
│   ├── src/components/  # Vue components
│   ├── src/stores/      # Pinia state management
│   └── src/visualization/ # WebGL/Three.js rendering
│
├── benchmarks/          # Alignment benchmarks (Year 2)
├── docs/                # Documentation
├── ROADMAP.md           # Detailed project roadmap
└── docker-compose.yml   # Container orchestration
```

## Quick Start

### Microscope (Interpretability Toolkit)

```bash
cd microscope
pip install maturin
maturin develop

# Run example
python examples/quickstart.py
```

```python
from alignment_microscope import Microscope
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Create microscope and trace
scope = Microscope.for_model(model)
inputs = tokenizer("The capital of France is", return_tensors="pt")

with scope.trace() as trace:
    outputs = model(**inputs)

# Analyze attention patterns
for layer in range(model.config.n_layer):
    pattern = trace.attention(layer)
    head_types = scope.classify_heads(pattern)
    print(f"Layer {layer}: {head_types}")
```

### Full Stack (API + Dashboard)

```bash
# Using Docker
docker-compose up

# Or manually:
# Terminal 1: API
cd api && pip install -r requirements.txt && uvicorn main:app --reload

# Terminal 2: Dashboard
cd dashboard && npm install && npm run dev
```

Access the dashboard at http://localhost:5173

## Key Features

### Microscope
- **Activation Tracing**: Track information flow through model layers
- **Attention Analysis**: Visualize and classify attention head behaviors
- **Circuit Discovery**: Automatically identify computational circuits
- **IOI Detection**: Detect Indirect Object Identification circuits (Wang et al. 2022)
- **SAE Integration**: Sparse autoencoder analysis with SAELens compatibility
- **Streaming**: Memory-efficient analysis for 70B+ models
- **Multi-Architecture**: Supports Llama, GPT-2, Mistral, Qwen, Gemma

### Dashboard (In Progress)
- WebGL-based attention flow visualization
- Interactive circuit explorer with graph layout
- Activation heatmaps with zoom/pan
- Real-time tracing via WebSocket
- SAE feature browser

## 5-Year Roadmap

| Year | Focus | Description |
|------|-------|-------------|
| **1** | Interpretability | Microscope toolkit + Dashboard (Current) |
| **2** | Benchmarks | 1000+ alignment test cases, adversarial generation |
| **3** | Red-Teaming | Jailbreak detection, capability elicitation |
| **4** | Guardrails | Runtime monitoring, anomaly detection |
| **5** | Construction | Aligned-by-construction training methods |

See [ROADMAP.md](ROADMAP.md) for detailed progress and plans.

## Tech Stack

- **Rust**: High-performance core computation
- **Python**: PyO3 bindings, ML integration (PyTorch, Transformers)
- **TypeScript/Vue 3**: Modern reactive frontend
- **Three.js/WebGL**: 3D attention visualization
- **Cytoscape.js**: Graph layout for circuits
- **FastAPI**: REST + WebSocket backend
- **Docker**: Container orchestration

## Contributing

We welcome contributions! Priority areas:

1. **Dashboard Development** - Vue.js + WebGL visualization
2. **Circuit Discovery** - New detection algorithms
3. **Documentation** - Examples and tutorials
4. **Testing** - Integration tests with real models

See individual component READMEs for setup instructions:
- [microscope/README.md](microscope/README.md)
- [dashboard/README.md](dashboard/README.md)

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{alignment_observatory,
  title = {Alignment Observatory: Open-Source Toolkit for AI Alignment Research},
  author = {Alignment Observatory Contributors},
  year = {2026},
  url = {https://github.com/jonnasson/alignment-observatory}
}
```

## Related Work

- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) - Mechanistic interpretability
- [nnsight](https://github.com/ndif-team/nnsight) - Neural network intervention
- [SAELens](https://github.com/jbloomAus/SAELens) - Sparse autoencoder training
- [Anthropic Circuits](https://transformer-circuits.pub/) - Circuit discovery research
