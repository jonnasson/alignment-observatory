"""
Trace management service.

Handles creating, loading, and querying activation traces.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from starlette.requests import Request

import numpy as np

from api.config import get_settings
from api.schemas import (
    ActivationRequest,
    ActivationResponse,
    AttentionPattern,
    AttentionRequest,
    AttentionResponse,
    CreateTraceRequest,
    HeadAnalysis,
    LayerActivation,
    LoadTraceRequest,
    TraceInfo,
    TraceMetadata,
)
from api.schemas.common import TensorData, TensorStats
from api.schemas.trace import HeadClassification
from api.services.microscope_service import MicroscopeService

settings = get_settings()


class TraceStore:
    """In-memory store for traces."""

    def __init__(self, max_size: int = 100) -> None:
        self._traces: dict[str, dict[str, Any]] = {}
        self._max_size = max_size

    def add(self, trace_id: str, data: dict[str, Any]) -> None:
        """Add a trace to the store."""
        if len(self._traces) >= self._max_size:
            # Remove oldest trace
            oldest = next(iter(self._traces))
            del self._traces[oldest]
        self._traces[trace_id] = data

    def get(self, trace_id: str) -> dict[str, Any] | None:
        """Get a trace by ID."""
        return self._traces.get(trace_id)

    def delete(self, trace_id: str) -> bool:
        """Delete a trace by ID."""
        if trace_id in self._traces:
            del self._traces[trace_id]
            return True
        return False

    def list_all(self) -> list[str]:
        """List all trace IDs."""
        return list(self._traces.keys())


# Global trace store
_trace_store = TraceStore(max_size=settings.trace_cache_max_size)


class TraceService:
    """Service for managing activation traces."""

    def __init__(self, microscope_service: MicroscopeService | None = None) -> None:
        self._microscope = microscope_service
        self._store = _trace_store

    async def create_trace(self, request: CreateTraceRequest) -> TraceInfo:
        """Create a new trace via live inference."""
        if self._microscope is None:
            raise RuntimeError("MicroscopeService not available - model not loaded")

        # Run inference with caching
        cache_data = await self._microscope.run_with_cache(
            text=request.text,
            layers=request.layers,
            include_attention=request.include_attention,
            include_activations=request.include_activations,
        )

        # Generate trace ID
        trace_id = str(uuid.uuid4())[:8]

        # Store trace data
        trace_data = {
            "trace_id": trace_id,
            "created_at": datetime.utcnow(),
            "input_text": request.text,
            "tokens": cache_data["tokens"],
            "token_ids": cache_data["token_ids"],
            "attention": cache_data.get("attention", {}),
            "activations": cache_data.get("activations", {}),
            "logits": cache_data.get("logits"),
            "model_name": request.model_name or settings.default_model,
        }

        self._store.add(trace_id, trace_data)

        return self._build_trace_info(trace_data)

    async def load_trace(self, request: LoadTraceRequest) -> TraceInfo:
        """Load a pre-computed trace from disk."""
        path = Path(request.path)
        if not path.exists():
            raise FileNotFoundError(f"Trace file not found: {path}")

        # Load trace data (assumes npz format)
        loaded = np.load(path, allow_pickle=True)

        trace_id = str(uuid.uuid4())[:8]
        trace_data = {
            "trace_id": trace_id,
            "created_at": datetime.utcnow(),
            "input_text": str(loaded.get("input_text", "")),
            "tokens": list(loaded.get("tokens", [])),
            "token_ids": list(loaded.get("token_ids", [])),
            "attention": dict(loaded.get("attention", {}).item()) if "attention" in loaded else {},
            "activations": dict(loaded.get("activations", {}).item()) if "activations" in loaded else {},
            "model_name": str(loaded.get("model_name", "unknown")),
        }

        self._store.add(trace_id, trace_data)
        return self._build_trace_info(trace_data)

    async def get_trace_info(self, trace_id: str) -> TraceInfo | None:
        """Get information about a trace."""
        trace_data = self._store.get(trace_id)
        if trace_data is None:
            return None
        return self._build_trace_info(trace_data)

    async def delete_trace(self, trace_id: str) -> bool:
        """Delete a trace."""
        return self._store.delete(trace_id)

    async def list_traces(self) -> list[TraceInfo]:
        """List all traces."""
        traces = []
        for trace_id in self._store.list_all():
            trace_data = self._store.get(trace_id)
            if trace_data:
                traces.append(self._build_trace_info(trace_data))
        return traces

    async def get_attention(
        self,
        trace_id: str,
        request: AttentionRequest,
    ) -> AttentionResponse:
        """Get attention patterns for a trace."""
        trace_data = self._store.get(trace_id)
        if trace_data is None:
            raise ValueError(f"Trace {trace_id} not found")

        attention = trace_data.get("attention", {})
        if request.layer not in attention:
            raise ValueError(f"Layer {request.layer} not found in trace")

        pattern = attention[request.layer]  # Shape: [num_heads, seq_q, seq_k]

        # Apply head selection or aggregation
        if request.head is not None:
            pattern = pattern[request.head]  # Shape: [seq_q, seq_k]
        elif request.aggregate == "mean":
            pattern = pattern.mean(axis=0)
        elif request.aggregate == "max":
            pattern = pattern.max(axis=0)

        # Build response
        stats = TensorStats(
            min=float(pattern.min()),
            max=float(pattern.max()),
            mean=float(pattern.mean()),
            std=float(pattern.std()),
            shape=list(pattern.shape),
            dtype=str(pattern.dtype),
        )

        tensor_data = TensorData(
            data=pattern.flatten().tolist(),
            shape=list(pattern.shape),
            dtype=str(pattern.dtype),
            stats=stats,
        )

        attention_pattern = AttentionPattern(
            layer=request.layer,
            head=request.head,
            pattern=tensor_data,
            tokens=trace_data["tokens"],
            stats=stats,
        )

        # Analyze heads if returning all heads
        analysis = None
        if request.head is None and request.aggregate == "none":
            analysis = await self._analyze_heads(
                attention[request.layer],
                trace_data["tokens"],
                request.layer,
            )

        return AttentionResponse(
            trace_id=trace_id,
            pattern=attention_pattern,
            analysis=analysis,
        )

    async def get_activations(
        self,
        trace_id: str,
        request: ActivationRequest,
    ) -> ActivationResponse:
        """Get activations for a trace."""
        trace_data = self._store.get(trace_id)
        if trace_data is None:
            raise ValueError(f"Trace {trace_id} not found")

        activations = trace_data.get("activations", {})
        if request.layer not in activations:
            raise ValueError(f"Layer {request.layer} not found in trace")

        layer_acts = activations[request.layer]
        if request.component not in layer_acts:
            raise ValueError(f"Component {request.component} not found in layer {request.layer}")

        act_data = layer_acts[request.component]  # Shape: [seq_len, hidden_size]

        # Apply token selection
        if request.token_indices:
            act_data = act_data[request.token_indices]

        # Compute token norms
        token_norms = np.linalg.norm(act_data, axis=-1).tolist()

        # Build response
        stats = TensorStats(
            min=float(act_data.min()),
            max=float(act_data.max()),
            mean=float(act_data.mean()),
            std=float(act_data.std()),
            shape=list(act_data.shape),
            dtype=str(act_data.dtype),
        )

        tensor_data = TensorData(
            data=act_data.flatten().tolist(),
            shape=list(act_data.shape),
            dtype=str(act_data.dtype),
            stats=stats,
        )

        layer_activation = LayerActivation(
            layer=request.layer,
            component=request.component,
            activations=tensor_data,
            stats=stats,
        )

        return ActivationResponse(
            trace_id=trace_id,
            activation=layer_activation,
            token_norms=token_norms,
        )

    def _build_trace_info(self, trace_data: dict[str, Any]) -> TraceInfo:
        """Build TraceInfo from trace data."""
        # Determine model config
        num_layers = len(trace_data.get("attention", {})) or len(trace_data.get("activations", {}))
        attention = trace_data.get("attention", {})
        num_heads = 0
        if attention:
            first_layer = next(iter(attention.values()))
            num_heads = first_layer.shape[0] if len(first_layer.shape) >= 3 else 0

        activations = trace_data.get("activations", {})
        hidden_size = 0
        if activations:
            first_layer = next(iter(activations.values()))
            if "residual" in first_layer:
                hidden_size = first_layer["residual"].shape[-1]

        metadata = TraceMetadata(
            model_name=trace_data.get("model_name", "unknown"),
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=hidden_size,
            seq_length=len(trace_data.get("tokens", [])),
            vocab_size=50257,  # Default for GPT-2
        )

        return TraceInfo(
            trace_id=trace_data["trace_id"],
            created_at=trace_data["created_at"],
            input_text=trace_data["input_text"],
            tokens=trace_data["tokens"],
            token_ids=trace_data["token_ids"],
            metadata=metadata,
            has_attention=bool(trace_data.get("attention")),
            has_activations=bool(trace_data.get("activations")),
            layers_available=list(trace_data.get("attention", {}).keys()) or list(trace_data.get("activations", {}).keys()),
        )

    async def _analyze_heads(
        self,
        attention: np.ndarray,
        tokens: list[str],
        layer: int,
    ) -> list[HeadAnalysis]:
        """Analyze all attention heads in a layer."""
        analyses = []
        num_heads = attention.shape[0]

        for head in range(num_heads):
            pattern = attention[head]

            # Compute statistics
            entropy = -np.sum(pattern * np.log(pattern + 1e-10), axis=-1).mean()
            max_att = float(pattern.max())
            sparsity = float(np.mean(pattern < 0.01))

            # Simple classification
            seq_len = pattern.shape[0]
            if seq_len > 1:
                prev_token_score = np.mean([
                    pattern[i, i - 1] if i > 0 else 0
                    for i in range(seq_len)
                ])
            else:
                prev_token_score = 0.0

            if prev_token_score > 0.5:
                category = "previous_token"
                confidence = float(prev_token_score)
            elif entropy < 1.0:
                category = "semantic"
                confidence = 1.0 - float(entropy) / 3.0
            else:
                category = "mixed"
                confidence = 0.5

            classification = HeadClassification(
                category=category,  # type: ignore
                confidence=min(1.0, max(0.0, confidence)),
            )

            analyses.append(HeadAnalysis(
                layer=layer,
                head=head,
                classification=classification,
                entropy=float(entropy),
                max_attention=max_att,
                sparsity=sparsity,
            ))

        return analyses


def get_trace_service() -> TraceService:
    """Dependency injection for TraceService (without model - for list/read operations)."""
    return TraceService()


def get_trace_service_with_model(request: "Request") -> TraceService:
    """Dependency injection for TraceService with model access.

    Args:
        request: FastAPI Request object to access app state.

    Returns:
        TraceService with MicroscopeService if model is loaded, otherwise without.
    """
    from starlette.requests import Request as StarletteRequest

    model_manager = getattr(request.app.state, "model_manager", None)
    if model_manager is None or model_manager.model is None:
        # Return service without microscope - caller should handle this
        return TraceService()

    microscope = MicroscopeService(
        model=model_manager.model,
        tokenizer=model_manager.tokenizer,
        device=model_manager.device,
    )
    return TraceService(microscope_service=microscope)
