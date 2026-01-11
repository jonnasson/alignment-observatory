"""Architecture detection utilities."""

from enum import Enum, auto
from typing import Any, Optional


class Architecture(Enum):
    """Supported transformer architectures."""

    LLAMA = auto()
    MISTRAL = auto()
    QWEN = auto()
    GEMMA = auto()
    GPT2 = auto()
    GPTJ = auto()
    GPTNEO = auto()
    FALCON = auto()
    PHI = auto()
    UNKNOWN = auto()


# Mapping from model_type strings to Architecture enum
_MODEL_TYPE_MAP = {
    # Llama family
    "llama": Architecture.LLAMA,
    "llama2": Architecture.LLAMA,
    "llama3": Architecture.LLAMA,
    "code_llama": Architecture.LLAMA,
    # Mistral family
    "mistral": Architecture.MISTRAL,
    "mixtral": Architecture.MISTRAL,
    # Qwen family
    "qwen": Architecture.QWEN,
    "qwen2": Architecture.QWEN,
    # Gemma family
    "gemma": Architecture.GEMMA,
    "gemma2": Architecture.GEMMA,
    # GPT-2 family
    "gpt2": Architecture.GPT2,
    "gpt_2": Architecture.GPT2,
    # GPT-J family
    "gptj": Architecture.GPTJ,
    "gpt-j": Architecture.GPTJ,
    # GPT-Neo family
    "gpt_neo": Architecture.GPTNEO,
    "gpt-neo": Architecture.GPTNEO,
    "gpt_neox": Architecture.GPTNEO,
    # Falcon family
    "falcon": Architecture.FALCON,
    "refinedweb": Architecture.FALCON,
    # Phi family
    "phi": Architecture.PHI,
    "phi-2": Architecture.PHI,
    "phi-3": Architecture.PHI,
}

# Mapping from class name patterns to Architecture enum
_CLASS_NAME_MAP = {
    "llama": Architecture.LLAMA,
    "mistral": Architecture.MISTRAL,
    "qwen": Architecture.QWEN,
    "gemma": Architecture.GEMMA,
    "gpt2": Architecture.GPT2,
    "gptj": Architecture.GPTJ,
    "gptneo": Architecture.GPTNEO,
    "falcon": Architecture.FALCON,
    "phi": Architecture.PHI,
}


def detect_architecture(model: Any) -> Architecture:
    """Detect the architecture type of a transformer model.

    Detection strategy:
    1. Check model.config.model_type if available
    2. Check model class name for known patterns
    3. Probe for architecture-specific attributes
    4. Return UNKNOWN if detection fails

    Args:
        model: A transformer model instance

    Returns:
        Architecture enum value
    """
    # Strategy 1: Check config.model_type
    config = getattr(model, "config", None)
    if config is not None:
        model_type = getattr(config, "model_type", None)
        if model_type is not None:
            model_type_lower = model_type.lower().replace("-", "_")
            if model_type_lower in _MODEL_TYPE_MAP:
                return _MODEL_TYPE_MAP[model_type_lower]

    # Strategy 2: Check class name
    class_name = type(model).__name__.lower()
    for pattern, arch in _CLASS_NAME_MAP.items():
        if pattern in class_name:
            return arch

    # Strategy 3: Probe for architecture-specific attributes
    detected = _probe_architecture_attributes(model)
    if detected is not None:
        return detected

    return Architecture.UNKNOWN


def _probe_architecture_attributes(model: Any) -> Optional[Architecture]:
    """Probe model structure to detect architecture.

    Args:
        model: A transformer model instance

    Returns:
        Detected Architecture or None
    """
    # Check for GPT-2 structure
    if hasattr(model, "transformer"):
        transformer = model.transformer
        if hasattr(transformer, "h"):
            # GPT-2 style: model.transformer.h
            return Architecture.GPT2
        if hasattr(transformer, "blocks"):
            # GPT-J style: model.transformer.blocks
            return Architecture.GPTJ

    # Check for Llama-style structure
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "layers"):
            # Could be Llama, Mistral, Qwen, or Gemma
            # Try to narrow down based on layer structure
            layers = inner.layers
            if len(layers) > 0:
                layer = layers[0]
                # Check for Gemma-specific attributes
                if hasattr(layer, "input_layernorm") and hasattr(layer, "post_attention_layernorm"):
                    # Check attention type
                    attn = getattr(layer, "self_attn", None)
                    if attn is not None:
                        if hasattr(attn, "rotary_emb"):
                            # Most Llama-style models have rotary embeddings
                            # Default to Llama as most common
                            return Architecture.LLAMA

    return None


def architecture_to_string(arch: Architecture) -> str:
    """Convert Architecture enum to string.

    Args:
        arch: Architecture enum value

    Returns:
        Lowercase string name
    """
    return arch.name.lower()


def string_to_architecture(name: str) -> Architecture:
    """Convert string to Architecture enum.

    Args:
        name: Architecture name string

    Returns:
        Architecture enum value

    Raises:
        ValueError: If name is not recognized
    """
    name_upper = name.upper().replace("-", "_")
    try:
        return Architecture[name_upper]
    except KeyError:
        # Try model_type mapping
        name_lower = name.lower().replace("-", "_")
        if name_lower in _MODEL_TYPE_MAP:
            return _MODEL_TYPE_MAP[name_lower]
        raise ValueError(f"Unknown architecture: {name}")
