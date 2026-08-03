"""Ray Serve lifecycle management utilities.

Shared by ``serving/serve_runner.py`` and ``embeddings/serve_runner.py``
to eliminate duplicated start/stop/status logic.
"""

from __future__ import annotations

import logging
from typing import Any

import ray
from ray import serve

logger = logging.getLogger(__name__)


DEFAULT_GRPC_SERVICER_FUNCTION = (
    "tributo.serving.proto.inference_pb2_grpc.add_InferenceServiceServicer_to_server"
)


def deploy_serve_app(
    deployment_cls: type,
    *,
    app_name: str,
    route_prefix: str | None,
    num_replicas: int = 1,
    ray_address: str | None = None,
    grpc_port: int | None = None,
    enable_http: bool = True,
    grpc_servicer_functions: list[str] | None = None,
    user_config: Any = None,
    runtime_env: dict[str, Any] | None = None,
    **bind_kwargs: Any,
) -> str:
    """Deploy a Ray Serve application.

    Handles ``ray.init``, ``serve.start``, deployment creation, and
    ``serve.run`` in one call. Business modules only need to provide
    the deployment class and its bind arguments.

    Supports both HTTP and gRPC protocols. When ``grpc_port`` is
    provided, gRPC protocol is enabled. HTTP can be disabled by setting
    ``enable_http=False`` for gRPC-only deployments.

    Args:
        deployment_cls: The Serve Deployment class (e.g. ``ONNXModel``).
        app_name: Serve Application name for subsequent stop/status queries.
        route_prefix: HTTP route prefix. ``None`` for gRPC-only deployments.
        num_replicas: Number of replicas.
        ray_address: Ray cluster address; ``None`` for local default.
        grpc_port: gRPC port. If provided, enables gRPC protocol.
        enable_http: Whether to start the HTTP proxy. Defaults to ``True``.
        grpc_servicer_functions: Import paths of gRPC servicer functions.
            Required for gRPC deployments. Defaults to Tributo's inference
            service servicer when ``grpc_port`` is provided.
        user_config: Optional user config passed to deployment ``reconfigure``.
        runtime_env: Optional Ray runtime_env for the deployment. Use
            ``{"py_modules": [...]}`` to inject code into worker nodes.
        **bind_kwargs: Keyword arguments passed to ``deployment_cls.bind()``.

    Returns:
        The deployed application name.
    """
    if not ray.is_initialized():
        ray.init(address=ray_address, ignore_reinit_error=True)

    # Start HTTP (optional) + gRPC (if specified)
    if enable_http:
        http_options = {"host": "0.0.0.0", "port": 8000}
    else:
        http_options = {"location": "NoServer"}

    if grpc_port:
        grpc_options = {
            "port": grpc_port,
            "grpc_servicer_functions": grpc_servicer_functions
            or [DEFAULT_GRPC_SERVICER_FUNCTION],
        }
    else:
        grpc_options = None

    serve.start(http_options=http_options, grpc_options=grpc_options)

    # Check if deployment_cls is already a Deployment object (decorated by @serve.deployment)
    if hasattr(deployment_cls, "bind"):
        # Already a Deployment object, use directly
        deployment = deployment_cls
    else:
        # Regular class, needs decoration
        ray_actor_options = {}
        if runtime_env:
            ray_actor_options["runtime_env"] = runtime_env
        deployment = serve.deployment(
            num_replicas=num_replicas,
            user_config=user_config,
            ray_actor_options=ray_actor_options or None,
        )(deployment_cls)

    # gRPC proxy relies on ROUTE_TABLE to discover applications, but deployments
    # with route_prefix=None are not registered in ROUTE_TABLE (Ray Serve design
    # limitation). gRPC deployments need a non-None route_prefix for the proxy
    # to route requests correctly.
    if grpc_port and route_prefix is None:
        route_prefix = f"/{app_name}"

    serve.run(
        deployment.bind(**bind_kwargs),
        name=app_name,
        route_prefix=route_prefix,
    )

    if grpc_port and enable_http:
        protocol = "HTTP+gRPC"
    elif grpc_port:
        protocol = "gRPC"
    else:
        protocol = "HTTP"
    logger.info(
        "Serve app '%s' started (%s) at route '%s'",
        app_name,
        protocol,
        route_prefix,
    )
    return app_name


def delete_serve_app(app_name: str, *, ray_address: str | None = None) -> bool:
    """Stop a running Serve application.

    Args:
        app_name: Application name to stop.
        ray_address: Ray cluster address; ``None`` for local default.

    Returns:
        True if stopped successfully, False otherwise.
    """
    if not ray.is_initialized():
        ray.init(address=ray_address, ignore_reinit_error=True)

    try:
        serve.delete(app_name)
        logger.info("Serve app '%s' stopped.", app_name)
        return True
    except Exception as e:
        logger.warning("Failed to stop serve app '%s': %s", app_name, e)
        return False


def get_serve_app_status(
    app_name: str, *, ray_address: str | None = None
) -> dict[str, Any]:
    """Query a Serve application's status.

    Args:
        app_name: Application name to query.
        ray_address: Ray cluster address; ``None`` for local default.

    Returns:
        Status dictionary with keys: ``running``, ``app_name``,
        ``route``, ``status``, ``deployments``.
    """
    if not ray.is_initialized():
        ray.init(address=ray_address, ignore_reinit_error=True)

    status = serve.status()
    apps = status.applications if hasattr(status, "applications") else {}
    app_info = (
        apps.get(app_name) if hasattr(apps, "get") else getattr(apps, app_name, None)
    )

    if app_info is None:
        return {
            "running": False,
            "app_name": app_name,
            "route": None,
            "status": "NOT_FOUND",
            "deployments": [],
        }

    return {
        "running": True,
        "app_name": app_name,
        "route": getattr(app_info, "route_prefix", None),
        "status": getattr(app_info, "status", "UNKNOWN"),
        "deployments": list(getattr(app_info, "deployments", {}).keys()),
    }
