"""Ray Serve deployment management for text embedding service.

Provides start/stop/status wrappers around ``serve.run`` / ``serve.delete``.
"""

from __future__ import annotations

import logging
from typing import Any

from tributo._common.serve_utils import (
    delete_serve_app,
    deploy_serve_app,
    get_serve_app_status,
)
from tributo.embeddings.serve_deployment import TextEmbeddingService
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)

DEFAULT_APP_NAME = "tributo-embed"
DEFAULT_ROUTE_PREFIX = "/embed"


@PublicAPI(stability="beta")
def start_embedding_serving(
    model_path: str,
    *,
    model_name: str | None = None,
    app_name: str = DEFAULT_APP_NAME,
    route_prefix: str = DEFAULT_ROUTE_PREFIX,
    num_replicas: int = 1,
    ray_address: str | None = None,
) -> str:
    """Start the text embedding HTTP service.

    Args:
        model_path: Directory containing exported ONNX model and tokenizer.
        model_name: Optional short registered model name.
        app_name: Serve application name.
        route_prefix: HTTP route prefix, default ``/embed``.
        num_replicas: Number of serve replicas.
        ray_address: Ray cluster address, None for local default.

    Returns:
        Application name.
    """
    return deploy_serve_app(
        TextEmbeddingService,
        app_name=app_name,
        route_prefix=route_prefix,
        num_replicas=num_replicas,
        ray_address=ray_address,
        model_path=model_path,
        model_name=model_name,
    )


@PublicAPI(stability="beta")
def stop_embedding_serving(
    app_name: str = DEFAULT_APP_NAME,
    *,
    ray_address: str | None = None,
) -> bool:
    """Stop the embedding serve application.

    Args:
        app_name: Application name.
        ray_address: Ray cluster address, None for local default.

    Returns:
        True if stopped successfully.
    """
    return delete_serve_app(app_name, ray_address=ray_address)


@PublicAPI(stability="beta")
def get_embedding_serving_status(
    app_name: str = DEFAULT_APP_NAME,
    *,
    ray_address: str | None = None,
) -> dict[str, Any]:
    """Query embedding serve application status.

    Args:
        app_name: Application name.
        ray_address: Ray cluster address, None for local default.

    Returns:
        Status dictionary.
    """
    return get_serve_app_status(app_name, ray_address=ray_address)
