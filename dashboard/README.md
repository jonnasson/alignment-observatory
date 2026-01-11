# Alignment Observatory Dashboard

**Interactive visualization interface for AI interpretability research.**

Part of the [Alignment Observatory](../README.md) project.

## Overview

The dashboard provides real-time visualization of neural network internals, enabling researchers to explore attention patterns, discover circuits, and analyze model behavior through an intuitive web interface.

## Features

### Attention Visualization
- **3D Attention Flow**: WebGL-powered visualization of attention patterns across layers
- **Head Classification**: Automatic identification of attention head types (induction, previous-token, BOS, etc.)
- **Interactive Exploration**: Zoom, pan, and filter attention matrices

### Circuit Explorer
- **Graph Layout**: Cytoscape.js-powered circuit visualization
- **IOI Detection**: Visualize Indirect Object Identification circuits
- **Path Tracing**: Follow information flow through the network

### Activation Analysis
- **Heatmaps**: Layer-by-layer activation visualization
- **Token Norms**: Track residual stream magnitude
- **Anomaly Detection**: Highlight unusual activation patterns

### SAE Feature Browser
- **Feature Activation**: Explore sparse autoencoder features
- **Top-k Features**: Identify most active features per token
- **Co-activation**: Discover feature relationships

### Real-time Tracing
- **WebSocket Streaming**: Live updates during model inference
- **Progress Indicators**: Track analysis progress
- **Cached Results**: Efficient re-analysis of previous traces

## Tech Stack

- **Vue 3** - Reactive component framework
- **TypeScript** - Type-safe JavaScript
- **Vite** - Fast build tooling
- **Three.js** - WebGL 3D rendering
- **Cytoscape.js** - Graph visualization
- **D3.js** - Data visualization
- **Pinia** - State management
- **Tailwind CSS** - Utility-first styling
- **Axios** - HTTP client

## Development Setup

### Prerequisites
- Node.js 18+
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The dashboard will be available at http://localhost:5173

### Build for Production

```bash
npm run build
```

Output will be in the `dist/` directory.

### Linting

```bash
npm run lint
```

## Project Structure

```
dashboard/
├── src/
│   ├── App.vue              # Root component
│   ├── main.ts              # Entry point
│   │
│   ├── views/               # Page components
│   │
│   ├── components/          # Reusable components
│   │   ├── attention/       # Attention visualization
│   │   ├── attention3d/     # 3D attention rendering
│   │   ├── circuit/         # Circuit explorer
│   │   ├── activation/      # Activation heatmaps
│   │   ├── sae/             # SAE feature browser
│   │   ├── ioi/             # IOI circuit viewer
│   │   ├── three/           # Three.js utilities
│   │   ├── common/          # Shared UI components
│   │   └── layout/          # Layout components
│   │
│   ├── stores/              # Pinia state management
│   │   ├── trace.store.ts   # Trace data management
│   │   ├── model.store.ts   # Model state
│   │   ├── circuit.store.ts # Circuit discovery state
│   │   └── ui.store.ts      # UI preferences
│   │
│   ├── composables/         # Vue 3 composition utilities
│   │   ├── useThreeScene.ts # 3D scene setup
│   │   ├── useAttentionViz.ts
│   │   ├── useCircuitGraph.ts
│   │   └── ...
│   │
│   ├── services/            # API clients
│   │   └── api.ts           # REST/WebSocket client
│   │
│   ├── types/               # TypeScript definitions
│   │
│   ├── visualization/       # Advanced rendering
│   │   ├── core/            # Rendering engines
│   │   ├── scenes/          # Scene configurations
│   │   └── shaders/         # GLSL shaders
│   │
│   └── router/              # Vue Router config
│
├── public/                  # Static assets
├── index.html               # HTML template
├── package.json             # Dependencies
├── vite.config.ts           # Vite configuration
├── tsconfig.json            # TypeScript config
└── Dockerfile               # Container image
```

## API Integration

The dashboard connects to the FastAPI backend at `http://localhost:8000` (configurable via `VITE_API_URL`).

### Key Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/models` | List available models |
| `POST /api/traces` | Create new trace |
| `GET /api/traces/{id}` | Get trace details |
| `POST /api/circuits/discover` | Discover circuits |
| `WS /ws/trace/{id}` | Real-time trace updates |

## Docker

```bash
# Build image
docker build -t alignment-observatory-dashboard .

# Run container
docker run -p 5173:5173 alignment-observatory-dashboard
```

Or use docker-compose from the root directory:

```bash
docker-compose up dashboard
```

## License

Apache 2.0 - See [LICENSE](../LICENSE) for details.
