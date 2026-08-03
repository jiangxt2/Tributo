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
    model_path: str | None = None,
    *,
    bundle_uri: str | None = None,
    role: str = "inference",
    unsafe: bool = False,
    storage_profile: str | None = None,
    app_name: str = DEFAULT_APP_NAME,
    grpc_port: int = DEFAULT_GRPC_PORT,
    num_replicas: int = 1,
    ray_address: str | None = None,
    enable_http: bool = True,
    runtime_env: dict[str, Any] | None = None,
) -> str:
    """Start gRPC inference service.

    The stable model entry is a published ``bundle_uri``; a raw
    ``model_path`` remains as a compatibility adapter.

    Args:
        model_path: ONNX model file path (legacy compat adapter).
        bundle_uri: Published bundle URI (stable serving entry point).
        role: Artifact role to serve; defaults to ``inference``.
        unsafe: Permit loading bundles without typed signatures or
            flavors that are not safe.
        storage_profile: Storage profile name for S3 bundles.
        app_name: Serve Application name.
        grpc_port: gRPC port.
        num_replicas: Number of replicas.
        ray_address: Ray cluster address.
        enable_http: Whether to also start HTTP proxy. Can be set to False for gRPC-only scenarios.
        runtime_env: Ray runtime_env, used to inject code into worker nodes.

    Returns:
        Deployed Serve Application name.
    """
    if (model_path is None) == (bundle_uri is None):
        raise ValueError(
            "exactly one of 'model_path' (legacy) or 'bundle_uri' must be provided"
        )
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
        bundle_uri=bundle_uri,
        role=role,
        unsafe=unsafe,
        storage_profile=storage_profile,
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
