"""
Model lifecycle management service.

Handles loading, unloading, and managing transformer models.
"""

import gc
from datetime import datetime
from typing import Any

import torch

from api.config import get_settings
from api.schemas import LoadModelRequest, MemoryEstimate, ModelInfo, ModelsListResponse

settings = get_settings()


# Known model configurations
KNOWN_MODELS: dict[str, dict[str, Any]] = {
    "gpt2": {
        "display_name": "GPT-2 Small",
        "num_layers": 12,
        "num_heads": 12,
        "hidden_size": 768,
        "vocab_size": 50257,
        "max_seq_length": 1024,
        "parameter_count": 124_000_000,
    },
    "gpt2-medium": {
        "display_name": "GPT-2 Medium",
        "num_layers": 24,
        "num_heads": 16,
        "hidden_size": 1024,
        "vocab_size": 50257,
        "max_seq_length": 1024,
        "parameter_count": 355_000_000,
    },
    "gpt2-large": {
        "display_name": "GPT-2 Large",
        "num_layers": 36,
        "num_heads": 20,
        "hidden_size": 1280,
        "vocab_size": 50257,
        "max_seq_length": 1024,
        "parameter_count": 774_000_000,
    },
    "gpt2-xl": {
        "display_name": "GPT-2 XL",
        "num_layers": 48,
        "num_heads": 25,
        "hidden_size": 1600,
        "vocab_size": 50257,
        "max_seq_length": 1024,
        "parameter_count": 1_500_000_000,
    },
}


class ModelManager:
    """Manages model loading and lifecycle."""

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._current_model_name: str | None = None
        self._loaded_at: datetime | None = None
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"

    async def list_models(self) -> ModelsListResponse:
        """List all available models."""
        models = []
        for name, config in KNOWN_MODELS.items():
            models.append(
                ModelInfo(
                    name=name,
                    display_name=config["display_name"],
                    num_layers=config["num_layers"],
                    num_heads=config["num_heads"],
                    hidden_size=config["hidden_size"],
                    vocab_size=config["vocab_size"],
                    max_seq_length=config["max_seq_length"],
                    parameter_count=config["parameter_count"],
                    is_loaded=(name == self._current_model_name),
                    loaded_at=self._loaded_at if name == self._current_model_name else None,
                )
            )
        return ModelsListResponse(models=models, current_model=self._current_model_name)

    async def load_model(self, request: LoadModelRequest) -> ModelInfo:
        """Load a model into memory."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = request.model_name

        # Check if already loaded
        if model_name == self._current_model_name and not request.force_reload:
            return await self.get_current_model()  # type: ignore

        # Unload existing model
        await self.unload_model()

        # Determine device and dtype
        device = request.device or self._device
        dtype = None
        if request.dtype:
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            dtype = dtype_map.get(request.dtype, torch.float32)

        # Load model and tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
        )

        if device == "cpu":
            self._model = self._model.to(device)

        self._model.eval()
        self._current_model_name = model_name
        self._loaded_at = datetime.utcnow()

        return await self.get_current_model()  # type: ignore

    async def unload_model(self) -> None:
        """Unload the current model from memory."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        self._current_model_name = None
        self._loaded_at = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def get_current_model(self) -> ModelInfo | None:
        """Get info about the currently loaded model."""
        if self._current_model_name is None:
            return None

        config = KNOWN_MODELS.get(self._current_model_name, {})
        return ModelInfo(
            name=self._current_model_name,
            display_name=config.get("display_name", self._current_model_name),
            num_layers=config.get("num_layers", 0),
            num_heads=config.get("num_heads", 0),
            hidden_size=config.get("hidden_size", 0),
            vocab_size=config.get("vocab_size", 0),
            max_seq_length=config.get("max_seq_length", 0),
            parameter_count=config.get("parameter_count", 0),
            is_loaded=True,
            loaded_at=self._loaded_at,
        )

    async def estimate_memory(self, model_name: str) -> MemoryEstimate:
        """Estimate memory requirements for a model."""
        config = KNOWN_MODELS.get(model_name)
        if config is None:
            # Default estimates for unknown models
            params_mb = 500.0
            act_per_token = 1.0
        else:
            # ~4 bytes per parameter in float32
            params_mb = config["parameter_count"] * 4 / (1024 * 1024)
            # Rough estimate for activations
            act_per_token = config["hidden_size"] * config["num_layers"] * 4 / (1024 * 1024)

        gpu_available = None
        if torch.cuda.is_available():
            gpu_available = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)

        total_estimate = params_mb + (act_per_token * 512)  # Assume 512 tokens

        return MemoryEstimate(
            model_name=model_name,
            parameters_mb=params_mb,
            activations_mb_per_token=act_per_token,
            estimated_total_mb=total_estimate,
            gpu_available_mb=gpu_available,
            fits_in_memory=gpu_available is None or total_estimate < gpu_available * 0.9,
        )

    @property
    def model(self) -> Any:
        """Get the loaded model."""
        return self._model

    @property
    def tokenizer(self) -> Any:
        """Get the loaded tokenizer."""
        return self._tokenizer

    @property
    def device(self) -> str:
        """Get the device."""
        return self._device

    async def cleanup(self) -> None:
        """Cleanup on shutdown."""
        await self.unload_model()
