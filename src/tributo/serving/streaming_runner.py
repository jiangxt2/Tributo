"""Streaming inference service lifecycle management.

Wraps ``start`` / ``stop`` / ``status``, reusing the common Serve deployment
logic provided by ``_common.serve_utils``.
"""

from __future__ import annotations

import logging
from typing import Any

from tributo._common.serve_utils import (
    delete_serve_app,
    deploy_serve_app,
    get_serve_app_status,
)
from tributo.serving.streaming_deployment import LLMStreamingService
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)

DEFAULT_APP_NAME = "tributo-streaming"
DEFAULT_ROUTE_PREFIX = "/v1/chat/completions"


@PublicAPI(stability="beta")
def start_streaming_serving(
    model_path: str,
    tokenizer_path: str,
    *,
    app_name: str = DEFAULT_APP_NAME,
    route_prefix: str = DEFAULT_ROUTE_PREFIX,
    num_replicas: int = 1,
    max_tokens: int = 512,
    max_workers: int = 4,
    ray_address: str | None = None,
) -> str:
    """Start streaming inference service.

    Args:
        model_path: Model file path.
        tokenizer_path: Tokenizer file path.
        app_name: Serve Application name.
        route_prefix: HTTP route prefix.
        num_replicas: Number of replicas.
        max_tokens: Default maximum number of tokens to generate.
        max_workers: Inference thread pool size.
        ray_address: Ray cluster address.

    Returns:
        Deployed Serve Application name.
    """
    return deploy_serve_app(
        LLMStreamingService,
        app_name=app_name,
        route_prefix=route_prefix,
        num_replicas=num_replicas,
        ray_address=ray_address,
        user_config={},
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        max_tokens=max_tokens,
        max_workers=max_workers,
    )


@PublicAPI(stability="beta")
def stop_streaming_serving(
    app_name: str = DEFAULT_APP_NAME,
    *,
    ray_address: str | None = None,
) -> bool:
    """Stop streaming inference service.

    Args:
        app_name: Application name.
        ray_address: Ray cluster address.

    Returns:
        Whether the stop was successful.
    """
    return delete_serve_app(app_name, ray_address=ray_address)


@PublicAPI(stability="beta")
def get_streaming_serving_status(
    app_name: str = DEFAULT_APP_NAME,
    *,
    ray_address: str | None = None,
) -> dict[str, Any]:
    """Query streaming inference service status.

    Args:
        app_name: Application name.
        ray_address: Ray cluster address.

    Returns:
        Status dictionary.
    """
    return get_serve_app_status(app_name, ray_address=ray_address)
