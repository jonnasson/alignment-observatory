"""
Alignment Observatory API - FastAPI Application Entry Point

Provides REST and WebSocket endpoints for the interpretability dashboard.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import get_settings
from api.routers import circuits, ioi, models, sae, traces
from api.services.model_manager import ModelManager
from api.websockets.trace_ws import router as ws_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown."""
    # Startup
    app.state.model_manager = ModelManager()

    # Ensure cache directories exist
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    settings.trace_cache_dir.mkdir(parents=True, exist_ok=True)

    yield

    # Shutdown
    await app.state.model_manager.cleanup()


app = FastAPI(
    title=settings.app_name,
    description="API for Alignment Observatory interpretability dashboard",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(traces.router, prefix="/api/v1/traces", tags=["traces"])
app.include_router(circuits.router, prefix="/api/v1/circuits", tags=["circuits"])
app.include_router(ioi.router, prefix="/api/v1/ioi", tags=["ioi"])
app.include_router(sae.router, prefix="/api/v1/sae", tags=["sae"])
app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
app.include_router(ws_router, prefix="/ws", tags=["websocket"])


@app.get("/api/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "environment": settings.environment,
    }


@app.get("/api/v1/config")
async def get_config() -> dict:
    """Get public configuration."""
    return {
        "default_model": settings.default_model,
        "max_sequence_length": settings.max_sequence_length,
        "available_features": [
            "attention_analysis",
            "activation_tracing",
            "circuit_discovery",
            "sae_encoding",
            "ioi_detection",
        ],
    }
