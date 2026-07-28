"""Ray Serve deployment management utilities.

Wraps ``serve.run`` / ``serve.delete``, providing simple start/stop/query interfaces.
"""

from __future__ import annotations

import logging
from typing import Any

from tributo._common.serve_utils import (
    delete_serve_app,
    deploy_serve_app,
    get_serve_app_status,
)
from tributo.serving.model_deployment import ONNXModel
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)

DEFAULT_APP_NAME = "tributo-onnx"
DEFAULT_ROUTE_PREFIX = "/predict"


@PublicAPI(stability="beta")
def start_serving(
    model_path: str,
    *,
    app_name: str = DEFAULT_APP_NAME,
    route_prefix: str = DEFAULT_ROUTE_PREFIX,
    num_replicas: int = 1,
    ray_address: str | None = None,
) -> str:
    """Start ONNX inference service.

    Deploys an HTTP service via Ray Serve, loading the specified ONNX model into memory,
    exposing a ``POST {route_prefix}`` inference endpoint.

    Args:
        model_path: ONNX model file path.
        app_name: Serve Application name, used for subsequent stop/status queries.
        route_prefix: HTTP route prefix, default ``/predict``.
        num_replicas: Number of replicas; >1 enables Ray Serve auto load balancing.
        ray_address: Ray cluster address; uses local default when None.

    Returns:
        Deployed Serve Application name.

    Example:
        >>> start_serving("/workspace/onnx/test_completes.onnx")
        'tributo-onnx'
    """
    return deploy_serve_app(
        ONNXModel,
        app_name=app_name,
        route_prefix=route_prefix,
        num_replicas=num_replicas,
        ray_address=ray_address,
        model_path=model_path,
    )


@PublicAPI(stability="beta")
def stop_serving(
    app_name: str = DEFAULT_APP_NAME,
    *,
    ray_address: str | None = None,
) -> bool:
    """Stop the specified Serve Application.

    Args:
        app_name: Application name.
        ray_address: Ray cluster address; uses local default when None.

    Returns:
        Whether the stop was successful.
    """
    return delete_serve_app(app_name, ray_address=ray_address)


@PublicAPI(stability="beta")
def get_serving_status(
    app_name: str = DEFAULT_APP_NAME,
    *,
    ray_address: str | None = None,
) -> dict[str, Any]:
    """Query Serve Application status.

    Args:
        app_name: Application name.
        ray_address: Ray cluster address; uses local default when None.

    Returns:
        Status dictionary containing ``running``, ``route``, ``deployments`` and other fields.
    """
    return get_serve_app_status(app_name, ray_address=ray_address)
