"""Type stubs for SAE (Sparse Autoencoder) module."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

ArrayFloat32 = npt.NDArray[np.float32]

@dataclass
class SAEConfig:
    """Configuration for a Sparse Autoencoder."""

    d_in: int
    d_sae: int
    activation: str = "relu"
    k: Optional[int] = None
    hook_point: str = ""
    layer: int = 0

class SAEFeatures:
    """Container for SAE feature activations."""

    def __init__(
        self,
        activations: ArrayFloat32,
        config: Optional[SAEConfig] = None,
    ) -> None: ...
    @property
    def activations(self) -> ArrayFloat32: ...
    @property
    def shape(self) -> Tuple[int, ...]: ...
    @property
    def d_sae(self) -> int: ...
    @property
    def sparsity(self) -> float: ...
    @property
    def mean_active_features(self) -> float: ...
    def active_features(
        self, threshold: float = 0.0
    ) -> List[npt.NDArray[np.intp]]: ...
    def top_k_features(
        self, k: int = 10
    ) -> List[List[Tuple[int, float]]]: ...
    def feature_frequency(
        self, threshold: float = 0.0
    ) -> ArrayFloat32: ...
    def to_dict(self) -> Dict[str, Any]: ...

class SAEWrapper:
    """Unified interface for Sparse Autoencoders."""

    w_enc: ArrayFloat32
    w_dec: ArrayFloat32
    b_enc: Optional[ArrayFloat32]
    b_dec: Optional[ArrayFloat32]
    config: SAEConfig

    def __init__(
        self,
        w_enc: ArrayFloat32,
        w_dec: ArrayFloat32,
        b_enc: Optional[ArrayFloat32] = None,
        b_dec: Optional[ArrayFloat32] = None,
        config: Optional[SAEConfig] = None,
    ) -> None: ...
    @classmethod
    def from_saelens(
        cls,
        model_name: str,
        hook_point: str,
        layer: Optional[int] = None,
        device: str = "cpu",
    ) -> SAEWrapper: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SAEWrapper: ...
    @property
    def d_in(self) -> int: ...
    @property
    def d_sae(self) -> int: ...
    def encode(self, activations: ArrayFloat32) -> SAEFeatures: ...
    def decode(
        self, features: Union[SAEFeatures, ArrayFloat32]
    ) -> ArrayFloat32: ...
    def reconstruction_error(
        self,
        activations: ArrayFloat32,
        features: Optional[SAEFeatures] = None,
    ) -> float: ...
    def to_dict(self) -> Dict[str, Any]: ...

class SAEAnalyzer:
    """Analyzer for SAE features across traces."""

    def __init__(self) -> None: ...
    def register_sae(self, name: str, sae: SAEWrapper) -> None: ...
    def get_sae(self, name: str) -> Optional[SAEWrapper]: ...
    def analyze_activations(
        self, activations: Dict[str, ArrayFloat32]
    ) -> Dict[str, SAEFeatures]: ...
    def find_behavior_features(
        self,
        clean_features: SAEFeatures,
        corrupt_features: SAEFeatures,
        threshold: float = 0.1,
    ) -> Dict[str, List[int]]: ...
    def feature_coactivation(
        self, features: SAEFeatures, top_k: int = 10
    ) -> ArrayFloat32: ...
