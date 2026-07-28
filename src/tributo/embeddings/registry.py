"""Embedding model registry.

Pre-configured lightweight models with validated pooling and normalization
settings. Users reference models by short name instead of raw HF model IDs.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Literal

from tributo.exceptions import JobConfigurationError
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="beta")
@dataclass(frozen=True)
class ModelSpec:
    """Immutable specification for a supported embedding model.

    Attributes:
        name: Short reference name used in CLI and APIs.
        hf_model_id: HuggingFace model ID for download/export.
        dim: Output embedding dimension.
        pooling: Pooling strategy applied to hidden states.
        normalize: Whether to apply L2 normalization.
        max_length: Maximum token sequence length.
        onnx_opset: ONNX opset version for export.
    """

    name: str
    hf_model_id: str
    dim: int
    pooling: Literal["cls", "mean"] = "cls"
    normalize: bool = True
    max_length: int = 512
    onnx_opset: int = 14


#: Built-in registry of validated models.
_REGISTRY: dict[str, ModelSpec] = {
    "bge-small-zh": ModelSpec(
        name="bge-small-zh-v1.5",
        hf_model_id="BAAI/bge-small-zh-v1.5",
        dim=512,
        pooling="cls",
        normalize=True,
    ),
}
_REGISTRY_LOCK = threading.Lock()

#: Module-level list of built-in ModelSpecs, exported for entry_points discovery.
#: Third-party plugins are appended by auto-discovery below.
model_specs: list[ModelSpec] = list(_REGISTRY.values())


@PublicAPI(stability="beta")
def get_spec(name: str) -> ModelSpec:
    """Retrieve a model specification by short name.

    Args:
        name: Short model name, e.g. ``"bge-small-zh"``.

    Returns:
        The matching ``ModelSpec``.

    Raises:
        JobConfigurationError: If the model name is not registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise JobConfigurationError(f"Unknown model '{name}'. Available: {available}")
    return _REGISTRY[name]


@PublicAPI(stability="beta")
def list_models() -> list[str]:
    """Return a sorted list of registered short model names."""
    return sorted(_REGISTRY)


@PublicAPI(stability="beta")
def register(spec: ModelSpec) -> None:
    """Register a custom model specification.

    Args:
        spec: Model specification to register.

    Raises:
        JobConfigurationError: If the short name is already registered.
    """
    with _REGISTRY_LOCK:
        if spec.name in _REGISTRY:
            raise JobConfigurationError(f"Model '{spec.name}' is already registered")
        _REGISTRY[spec.name] = spec
    logger.info("Registered embedding model: %s", spec.name)


# Auto-discover third-party model plugins via entry_points
from tributo.plugin import discover_model_plugins  # noqa: E402

for _ep_spec in discover_model_plugins():
    try:
        register(_ep_spec)
    except JobConfigurationError:
        logger.debug(
            "Model %r from plugin already registered; skipping.", _ep_spec.name
        )
