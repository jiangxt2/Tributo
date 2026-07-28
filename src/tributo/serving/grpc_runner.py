"""gRPC inference service lifecycle management.

Provides start, stop, and status query functions for gRPC Serve Application.
"""

from __future__ import annotations

import logging
from typing import Any

from tributo._common.serve_utils import (
    delete_serve_app,
    deploy_serve_app,
    get_serve_app_status,
)
from tributo.serving.grpc_deployment import gRPCInferenceService
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)

DEFAULT_APP_NAME = "tributo-grpc"
DEFAULT_GRPC_PORT = 8001


@PublicAPI(stability="beta")
def start_grpc_serving(
    model_path: str,
    *,
    app_name: str = DEFAULT_APP_NAME,
    grpc_port: int = DEFAULT_GRPC_PORT,
    num_replicas: int = 1,
    ray_address: str | None = None,
    enable_http: bool = True,
    runtime_env: dict[str, Any] | None = None,
) -> str:
    """Start gRPC inference service.

    Args:
        model_path: ONNX model file path.
        app_name: Serve Application name.
        grpc_port: gRPC port.
        num_replicas: Number of replicas.
        ray_address: Ray cluster address.
        enable_http: Whether to also start HTTP proxy. Can be set to False for gRPC-only scenarios.
        runtime_env: Ray runtime_env, used to inject code into worker nodes.

    Returns:
        Deployed Serve Application name.
    """
    return deploy_serve_app(
        gRPCInferenceService,
        app_name=app_name,
        route_prefix=None,  # gRPC does not use route_prefix
        num_replicas=num_replicas,
        ray_address=ray_address,
        grpc_port=grpc_port,
        enable_http=enable_http,
        runtime_env=runtime_env,
        model_path=model_path,
    )


@PublicAPI(stability="beta")
def stop_grpc_serving(
    app_name: str = DEFAULT_APP_NAME,
    *,
    ray_address: str | None = None,
) -> bool:
    """Stop gRPC inference service.

    Args:
        app_name: Serve Application name.
        ray_address: Ray cluster address.

    Returns:
        Whether the stop was successful.
    """
    return delete_serve_app(app_name, ray_address=ray_address)


@PublicAPI(stability="beta")
def get_grpc_serving_status(
    app_name: str = DEFAULT_APP_NAME,
    *,
    ray_address: str | None = None,
) -> dict[str, Any]:
    """Query gRPC inference service status.

    Args:
        app_name: Serve Application name.
        ray_address: Ray cluster address.

    Returns:
        Status information dictionary.
    """
    return get_serve_app_status(app_name, ray_address=ray_address)
