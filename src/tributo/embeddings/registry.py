"""Embedding model registry.

Pre-configured lightweight models with validated pooling and normalization
settings. Users reference models by short name instead of raw HF model IDs.

Delegates to the generic ``Registry`` base class in ``_common/registry.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from tributo._common.registry import PluginAwareRegistry, Registry
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


def _discover_model_plugins() -> list[tuple[str, ModelSpec]]:
    """Lazy entry_points discovery helper."""
    from tributo.plugin import discover_model_plugins  # noqa: E402

    result: list[tuple[str, ModelSpec]] = []
    for ep_spec in discover_model_plugins():
        result.append((ep_spec.name, ep_spec))
    return result


_registry: Registry[str, ModelSpec] = PluginAwareRegistry(
    name="model",
    discover=_discover_model_plugins,
)

# Register built-in models.
_registry.register(
    "bge-small-zh",
    ModelSpec(
        name="bge-small-zh-v1.5",
        hf_model_id="BAAI/bge-small-zh-v1.5",
        dim=512,
        pooling="cls",
        normalize=True,
    ),
)

#: Module-level list of built-in ModelSpecs, exported for entry_points discovery.
#: Populated from ``_registry._store`` directly to avoid triggering lazy plugin
#: discovery at import time.  Callers needing the full list including plugins
#: should use ``list_models()`` instead.
model_specs: list[ModelSpec] = [
    _registry._store["bge-small-zh"],
]


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
    return _registry.get(name)


@PublicAPI(stability="beta")
def list_models() -> list[str]:
    """Return a sorted list of registered short model names."""
    return _registry.list()


@PublicAPI(stability="beta")
def register(spec: ModelSpec) -> None:
    """Register a custom model specification.

    Args:
        spec: Model specification to register.

    Raises:
        JobConfigurationError: If the short name is already registered.
    """
    _registry.register(spec.name, spec)
    logger.info("Registered embedding model: %s", spec.name)
