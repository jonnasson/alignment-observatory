"""
Service wrapper for alignment_microscope functionality.

This service provides a clean interface to the microscope package's
activation tracing, attention analysis, and circuit discovery features.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer

from api.config import get_settings

settings = get_settings()


class MicroscopeService:
    """
    Wrapper service for alignment_microscope functionality.

    Provides methods for:
    - Running forward passes with activation caching
    - Extracting attention patterns
    - Computing activation statistics
    """

    def __init__(self, model: Any, tokenizer: Any, device: str = "cpu") -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._hooked_model: "HookedTransformer | None" = None

    async def create_hooked_model(self) -> "HookedTransformer":
        """Create a TransformerLens HookedTransformer for interpretability."""
        if self._hooked_model is not None:
            return self._hooked_model

        from transformer_lens import HookedTransformer

        # Load directly with TransformerLens
        model_name = getattr(self._model.config, "_name_or_path", "gpt2")
        self._hooked_model = HookedTransformer.from_pretrained(
            model_name,
            device=self._device,
        )
        return self._hooked_model

    async def run_with_cache(
        self,
        text: str,
        layers: list[int] | None = None,
        include_attention: bool = True,
        include_activations: bool = True,
    ) -> dict[str, Any]:
        """
        Run a forward pass and cache activations.

        Returns a dictionary containing:
        - tokens: List of token strings
        - token_ids: List of token IDs
        - attention: Dict of attention patterns per layer (if requested)
        - activations: Dict of activations per layer/component (if requested)
        - logits: Output logits
        """
        hooked_model = await self.create_hooked_model()

        # Tokenize
        tokens = hooked_model.to_tokens(text)
        str_tokens = hooked_model.to_str_tokens(text)

        # Determine which activations to cache
        names_filter = []
        if include_attention:
            names_filter.append(lambda name: "attn.hook_pattern" in name)
        if include_activations:
            names_filter.extend([
                lambda name: "hook_resid_post" in name,
                lambda name: "hook_attn_out" in name,
                lambda name: "hook_mlp_out" in name,
            ])

        # Run with cache
        logits, cache = hooked_model.run_with_cache(
            tokens,
            names_filter=lambda name: any(f(name) for f in names_filter) if names_filter else True,
        )

        # Extract results
        result: dict[str, Any] = {
            "tokens": list(str_tokens),
            "token_ids": tokens[0].tolist(),
            "logits": logits[0].detach().cpu().numpy(),
        }

        # Extract attention patterns
        if include_attention:
            attention = {}
            num_layers = hooked_model.cfg.n_layers
            for layer in range(num_layers):
                if layers is not None and layer not in layers:
                    continue
                key = f"blocks.{layer}.attn.hook_pattern"
                if key in cache:
                    attention[layer] = cache[key][0].detach().cpu().numpy()
            result["attention"] = attention

        # Extract activations
        if include_activations:
            activations: dict[int, dict[str, np.ndarray]] = {}
            num_layers = hooked_model.cfg.n_layers
            for layer in range(num_layers):
                if layers is not None and layer not in layers:
                    continue
                layer_acts: dict[str, np.ndarray] = {}

                resid_key = f"blocks.{layer}.hook_resid_post"
                if resid_key in cache:
                    layer_acts["residual"] = cache[resid_key][0].detach().cpu().numpy()

                attn_key = f"blocks.{layer}.hook_attn_out"
                if attn_key in cache:
                    layer_acts["attention_out"] = cache[attn_key][0].detach().cpu().numpy()

                mlp_key = f"blocks.{layer}.hook_mlp_out"
                if mlp_key in cache:
                    layer_acts["mlp_out"] = cache[mlp_key][0].detach().cpu().numpy()

                if layer_acts:
                    activations[layer] = layer_acts
            result["activations"] = activations

        return result

    async def get_attention_pattern(
        self,
        cache_data: dict[str, Any],
        layer: int,
        head: int | None = None,
    ) -> np.ndarray:
        """Extract attention pattern from cached data."""
        attention = cache_data.get("attention", {})
        if layer not in attention:
            raise ValueError(f"Layer {layer} not in cached attention data")

        pattern = attention[layer]  # Shape: [num_heads, seq_q, seq_k]

        if head is not None:
            pattern = pattern[head]  # Shape: [seq_q, seq_k]

        return pattern

    async def classify_attention_head(
        self,
        pattern: np.ndarray,
        tokens: list[str],
    ) -> dict[str, Any]:
        """
        Classify an attention head based on its pattern.

        Categories:
        - induction: Attends to tokens that follow similar tokens
        - previous_token: Primarily attends to previous token
        - duplicate_token: Attends to duplicate tokens
        - positional: Strong positional pattern
        - semantic: Content-based attention
        - mixed: Combination of patterns
        """
        seq_len = pattern.shape[0]

        # Compute statistics
        entropy = -np.sum(pattern * np.log(pattern + 1e-10), axis=-1).mean()
        max_attention = pattern.max()

        # Check for previous token pattern
        if seq_len > 1:
            prev_token_score = np.mean([
                pattern[i, i - 1] if i > 0 else 0
                for i in range(seq_len)
            ])
        else:
            prev_token_score = 0.0

        # Check for diagonal (self-attention) pattern
        diag_score = np.mean(np.diag(pattern))

        # Classify based on scores
        if prev_token_score > 0.5:
            category = "previous_token"
            confidence = float(prev_token_score)
        elif diag_score > 0.5:
            category = "positional"
            confidence = float(diag_score)
        elif entropy < 1.0:
            category = "semantic"
            confidence = 1.0 - float(entropy)
        else:
            category = "mixed"
            confidence = 0.5

        return {
            "category": category,
            "confidence": confidence,
            "entropy": float(entropy),
            "max_attention": float(max_attention),
            "sparsity": float(np.mean(pattern < 0.01)),
        }

    def compute_tensor_stats(self, tensor: np.ndarray) -> dict[str, Any]:
        """Compute statistics for a tensor."""
        return {
            "min": float(tensor.min()),
            "max": float(tensor.max()),
            "mean": float(tensor.mean()),
            "std": float(tensor.std()),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
