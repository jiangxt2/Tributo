"""Command-line interface for Tributo.

Provides CLI commands for submitting and managing Ray jobs.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import click

from tributo import JobConfig
from tributo._common import DEFAULT_DASHBOARD_URL, configure_logging
from tributo.exceptions import (
    JobConfigurationError,
    PostPublishCallbackError,
    TributoError,
)
from tributo.job import TributoClient
from tributo.vector_index.cli import vector as vector_group

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(package_name="tributo")
def main():
    """Tributo: Unified framework for submitting Ray Jobs."""
    configure_logging(log_format="text")


@main.command()
@click.option(
    "--address",
    required=True,
    help="Ray cluster address (e.g., http://127.0.0.1:8265)",
)
@click.option("--entrypoint", required=True, help="Command to run for the job")
@click.option("--config", type=click.Path(exists=True), help="Path to config JSON file")
@click.option("--num-cpus", type=float, help="Number of CPUs to allocate")
@click.option("--num-gpus", type=float, help="Number of GPUs to allocate")
@click.option("--memory", type=int, help="Memory to allocate for entrypoint (in bytes)")
def submit(
    address: str,
    entrypoint: str,
    config: Optional[str],
    num_cpus: Optional[float],
    num_gpus: Optional[float],
    memory: Optional[int],
):
    """Submit a job to the Ray cluster."""
    try:
        # Load config from file if provided
        config_dict: dict[str, Any] = {}
        if config:
            from pathlib import Path

            if Path(config).suffix.lower() in {".yaml", ".yml"}:
                raise ValueError("YAML config is no longer supported; please use JSON.")
            with open(config, encoding="utf-8") as f:
                config_dict = json.load(f)

        # Override with CLI options
        config_dict["entrypoint"] = entrypoint
        if num_cpus is not None:
            config_dict["num_cpus"] = num_cpus
        if num_gpus is not None:
            config_dict["num_gpus"] = num_gpus
        if memory is not None:
            config_dict["memory"] = memory

        job_config = JobConfig(**config_dict)
        client = TributoClient(address)
        job_id = client.submit(
            entrypoint=job_config.entrypoint,
            runtime_env=job_config.runtime_env,
            metadata=job_config.metadata,
            submission_id=job_config.submission_id,
            entrypoint_num_cpus=job_config.num_cpus,
            entrypoint_num_gpus=job_config.num_gpus,
            entrypoint_memory=job_config.memory,
        )

        click.echo(f"Job submitted successfully: {job_id}")
    except TributoError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--address",
    required=True,
    help="Ray cluster address (e.g., http://127.0.0.1:8265)",
)
@click.argument("job_id")
def status(address: str, job_id: str):
    """Get the status of a submitted job."""
    try:
        client = TributoClient(address)
        job_status = client.get_status(job_id)

        click.echo(f"Job {job_id} status: {job_status}")
    except TributoError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--address",
    required=True,
    help="Ray cluster address (e.g., http://127.0.0.1:8265)",
)
@click.argument("job_id")
def logs(address: str, job_id: str):
    """Get logs for a submitted job."""
    try:
        client = TributoClient(address)
        job_logs = client.get_logs(job_id)

        click.echo(job_logs)
    except TributoError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


@main.command("explain")
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to an explainability request JSON file",
)
@click.option(
    "--address",
    default=DEFAULT_DASHBOARD_URL,
    show_default=True,
    help="Ray Dashboard address",
)
def explain(config: str, address: str):
    """Submit a distributed batch explainability operation."""
    try:
        from tributo.explainability.job_runner import submit_explainability_job

        job_id = submit_explainability_job(config, dashboard_url=address)
        click.echo(f"Explainability job submitted: {job_id}")
    except TributoError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Unexpected error: {exc}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--address",
    required=True,
    help="Ray cluster address (e.g., http://127.0.0.1:8265)",
)
@click.argument("job_id")
def stop(address: str, job_id: str):
    """Stop a running job."""
    try:
        client = TributoClient(address)
        result = client.stop_job(job_id)

        if result:
            click.echo(f"Job {job_id} stopped successfully")
        else:
            click.echo(f"Failed to stop job {job_id}")
    except TributoError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--source",
    required=True,
    help="Model source: ray://<checkpoint-path>, hf://<model-id>, or local path",
)
@click.option(
    "--targets",
    required=True,
    help="Comma-separated export targets (e.g. 'onnx,safetensors')",
)
@click.option(
    "--output",
    required=True,
    help="Bundle output URI (local path or s3://bucket/prefix)",
)
@click.option(
    "--alias",
    default=None,
    help="Optional alias name for the published bundle",
)
@click.option(
    "--storage-profile",
    default=None,
    help="Storage profile name for S3 credentials",
)
@click.option(
    "--request-id",
    default=None,
    help="Idempotency key (same request_id → same bundle_id)",
)
@click.option(
    "--role",
    "roles",
    multiple=True,
    help="Role assignments: name=target (repeatable)",
)
@click.option(
    "--hook",
    "hooks",
    multiple=True,
    help=(
        "Post-publish HookBinding as JSON (repeatable), for example "
        '\'{"hook_id":"mlflow-log-artifacts-v1","options":{...}}\''
    ),
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Output result as JSON",
)
def export(
    source: str,
    targets: str,
    output: str,
    alias: str | None,
    storage_profile: str | None,
    request_id: str | None,
    roles: tuple[str, ...],
    hooks: tuple[str, ...],
    json_output: bool,
):
    """Export a trained model to one or more formats as a bundle."""
    try:
        from tributo.exporting.models import (
            AliasConfig,
            BundleOutputConfig,
            ExportTarget,
            HookBinding,
        )
        from tributo.exporting.service import BundleExportService

        # Parse targets.
        target_list = [
            ExportTarget(name=fmt.strip(), format=fmt.strip())
            for fmt in targets.split(",")
            if fmt.strip()
        ]
        if not target_list:
            raise click.BadParameter("At least one target format is required")

        # Parse roles.
        roles_dict: dict[str, str] = {}
        for r in roles:
            if "=" not in r:
                raise click.BadParameter(f"Role must be 'name=target', got {r!r}")
            name, target = r.split("=", 1)
            roles_dict[name.strip()] = target.strip()

        hook_bindings: list[HookBinding] = []
        for raw_hook in hooks:
            try:
                hook_bindings.append(HookBinding.model_validate_json(raw_hook))
            except Exception as exc:
                raise click.BadParameter(
                    "Hook must be a valid HookBinding JSON object "
                    f"({type(exc).__name__})",
                    param_hint="--hook",
                ) from exc

        # Build config.
        config = BundleOutputConfig(
            bundle_uri=output,
            targets=target_list,
            request_id=request_id,
            storage_profile=storage_profile,
            roles=roles_dict,
            hooks=tuple(hook_bindings),
            alias=AliasConfig(name=alias) if alias else None,
        )

        # Resolve source provider by scheme.
        source_uri = source
        if source.startswith("ray://"):
            source_uri = source[6:]  # strip ray:// prefix
        elif source.startswith("hf://"):
            source_uri = source[5:]  # strip hf:// prefix

        from tributo.exporting.protocols import ExportSourceProvider

        provider: ExportSourceProvider
        if source.startswith("hf://"):
            from tributo.integrations.sources.huggingface import (
                HuggingFaceSourceProvider,
            )

            provider = HuggingFaceSourceProvider()
        else:
            # Ray checkpoints and local checkpoint paths use the
            # RayXGBoostSourceProvider (DNN/PU providers are selected by
            # the trainer-type registry in the training path).
            from tributo.integrations.sources.ray_xgboost import (
                RayXGBoostSourceProvider,
            )

            provider = RayXGBoostSourceProvider()
        service = BundleExportService()

        with provider.open_source(source_uri) as export_source:
            result = service.export_bundle(
                source=export_source,
                config=config,
                tributo_version=_get_tributo_version(),
            )

        if json_output:
            click.echo(result.model_dump_json(indent=2))
        else:
            click.echo(f"Bundle exported: {result.canonical_uri}")
            click.echo(f"  Bundle ID:   {result.bundle_id}")
            click.echo(f"  Manifest:    {result.manifest_uri}")
            click.echo(f"  Status:      {result.status}")
            if result.alias_uri:
                click.echo(f"  Alias:       {result.alias_uri} ({result.alias_status})")
            for receipt in result.hook_receipts:
                click.echo(f"  Hook:        {receipt.hook_id} ({receipt.status.value})")

    except PostPublishCallbackError as e:
        click.echo(
            "Bundle committed, but a required post-publish Hook failed.", err=True
        )
        if e.bundle_result is not None:
            click.echo(f"  Bundle:      {e.bundle_result.canonical_uri}", err=True)
            click.echo(f"  Manifest:    {e.bundle_result.manifest_uri}", err=True)
            for receipt in e.receipts:
                click.echo(
                    f"  Hook:        {receipt.hook_id} ({receipt.status.value})",
                    err=True,
                )
        sys.exit(1)
    except Exception as e:
        click.echo(f"Export failed: {e}", err=True)
        sys.exit(1)


@main.command("inspect")
@click.argument("uri")
@click.option(
    "--storage-profile",
    default=None,
    help="Storage profile name for S3 credentials",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Output the manifest as JSON",
)
def inspect(uri: str, storage_profile: str | None, json_output: bool):
    """Inspect a published bundle manifest.

    URI can be a local path, ``file://``, ``s3://`` manifest/bundle URI,
    or an ``s3://`` alias pointer.
    """
    try:
        from tributo.exporting.bundle_reader import BundleReader

        reader = BundleReader()
        manifest = reader.read_manifest(uri, storage_profile=storage_profile)

        if json_output:
            click.echo(manifest.model_dump_json(indent=2))
            return

        click.echo(f"Bundle ID:     {manifest.bundle_id}")
        click.echo(f"Status:        {manifest.status}")
        click.echo(f"Created:       {manifest.created_at.isoformat()}")
        click.echo(f"Canonical URI: {manifest.canonical_uri}")
        click.echo("Roles:")
        for name, target in (manifest.roles or {}).items():
            click.echo(f"  {name} → {target}")
        click.echo("Artifacts:")
        for a in manifest.artifacts:
            click.echo(
                f"  {a.name}  [{a.format}/{a.flavor_id}]  digest {a.tree_digest[:12]}…"
            )
    except Exception as e:
        click.echo(f"Inspect failed: {e}", err=True)
        sys.exit(1)


@main.command("export-gc")
@click.argument("bundle_uri")
@click.option(
    "--storage-profile",
    default=None,
    help="Storage profile name for S3 credentials",
)
@click.option(
    "--orphan-ttl",
    default=3600,
    type=int,
    help="Minimum age (seconds) before an orphan prefix is collectable",
)
@click.option(
    "--yes",
    "confirmed",
    is_flag=True,
    default=False,
    help="Actually delete orphans (default is a dry run)",
)
def export_gc(
    bundle_uri: str,
    storage_profile: str | None,
    orphan_ttl: int,
    confirmed: bool,
):
    """Collect orphaned bundle prefixes on S3 (dry-run by default).

    BUNDLE_URI must be the exact store root used for publication. Only
    prefixes that look like bundle IDs, have no manifest, are older than
    --orphan-ttl, and are not lease-protected are deleted.
    """
    try:
        from tributo.exporting.gc import BundleGarbageCollector

        collector = BundleGarbageCollector()
        result = collector.collect(
            bundle_uri,
            storage_profile=storage_profile,
            orphan_ttl_seconds=orphan_ttl,
            dry_run=not confirmed,
        )
        click.echo(f"Scanned:       {result['scanned']}")
        click.echo(f"Orphans found: {result['orphans_found']}")
        click.echo(f"Deleted:       {result['deleted']}")
        for err in result["errors"]:
            click.echo(f"  error: {err}", err=True)
        if not confirmed:
            click.echo("Dry-run — pass --yes to delete.")
    except Exception as e:
        click.echo(f"GC failed: {e}", err=True)
        sys.exit(1)


def _get_tributo_version() -> str:
    """Get the current tributo version string."""
    from importlib.metadata import PackageNotFoundError

    try:
        from importlib.metadata import version

        return version("tributo")
    except PackageNotFoundError:
        return "0.0.0"


@main.group()
def serve():
    """Manage ONNX inference service."""
    pass


@serve.command("start")
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    help="Path to ONNX model file (legacy entry)",
)
@click.option(
    "--bundle-uri",
    help="Published bundle URI (stable serving entry point)",
)
@click.option(
    "--role",
    default="inference",
    show_default=True,
    help="Artifact role to serve from the bundle",
)
@click.option(
    "--unsafe",
    is_flag=True,
    default=False,
    help="Permit bundles without typed signatures or flavors that are "
    "not safe (compat-only)",
)
@click.option(
    "--storage-profile",
    default=None,
    help="Storage profile name for S3 bundles",
)
@click.option(
    "--app-name",
    default="tributo-onnx",
    help="Serve application name",
)
@click.option(
    "--route-prefix",
    default="/predict",
    help="HTTP route prefix",
)
@click.option(
    "--num-replicas",
    default=1,
    type=int,
    help="Number of serve replicas",
)
@click.option(
    "--ray-address",
    default=None,
    help="Ray cluster address (e.g., ray://127.0.0.1:10001)",
)
def serve_start(
    model_path: str | None,
    bundle_uri: str | None,
    role: str,
    unsafe: bool,
    storage_profile: str | None,
    app_name: str,
    route_prefix: str,
    num_replicas: int,
    ray_address: str | None,
):
    """Start ONNX inference service.

    Exactly one of ``--model-path`` (legacy) or ``--bundle-uri`` must be
    provided.
    """
    if (model_path is None) == (bundle_uri is None):
        raise click.ClickException(
            "exactly one of --model-path (legacy) or --bundle-uri must be provided"
        )
    from tributo.serving import start_serving

    try:
        start_serving(
            model_path=model_path,
            bundle_uri=bundle_uri,
            role=role,
            unsafe=unsafe,
            storage_profile=storage_profile,
            app_name=app_name,
            route_prefix=route_prefix,
            num_replicas=num_replicas,
            ray_address=ray_address,
        )
        click.echo(f"Serve app '{app_name}' started at {route_prefix}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@serve.command("stop")
@click.option(
    "--app-name",
    default="tributo-onnx",
    help="Serve application name",
)
@click.option(
    "--ray-address",
    default=None,
    help="Ray cluster address (e.g., ray://127.0.0.1:10001)",
)
def serve_stop(app_name: str, ray_address: str | None):
    """Stop ONNX inference service."""
    from tributo.serving import stop_serving

    try:
        stopped = stop_serving(app_name, ray_address=ray_address)
        if stopped:
            click.echo(f"Serve app '{app_name}' stopped.")
        else:
            click.echo(f"Failed to stop serve app '{app_name}'.")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@serve.command("status")
@click.option(
    "--app-name",
    default="tributo-onnx",
    help="Serve application name",
)
@click.option(
    "--ray-address",
    default=None,
    help="Ray cluster address (e.g., ray://127.0.0.1:10001)",
)
def serve_status(app_name: str, ray_address: str | None):
    """Get ONNX inference service status."""
    from tributo.serving import get_serving_status

    try:
        status = get_serving_status(app_name, ray_address=ray_address)
        click.echo(f"App: {status['app_name']}")
        click.echo(f"Running: {status['running']}")
        click.echo(f"Route: {status['route']}")
        click.echo(f"Status: {status['status']}")
        click.echo(f"Deployments: {', '.join(status['deployments'])}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@serve.group(name="streaming")
def serve_streaming_group():
    """Manage LLM streaming inference service."""
    pass


@serve_streaming_group.command("start")
@click.option(
    "--model-path",
    required=True,
    help="Path to LLM model directory",
)
@click.option(
    "--tokenizer-path",
    required=True,
    help="Path to tokenizer directory",
)
@click.option(
    "--app-name",
    default="tributo-streaming",
    help="Serve application name",
)
@click.option(
    "--route-prefix",
    default="/v1/chat/completions",
    help="HTTP route prefix",
)
@click.option(
    "--num-replicas",
    default=1,
    type=int,
    help="Number of serve replicas",
)
@click.option(
    "--max-tokens",
    default=512,
    type=int,
    help="Default max tokens to generate",
)
@click.option(
    "--max-workers",
    default=4,
    type=int,
    help="Inference thread pool size",
)
@click.option(
    "--ray-address",
    default=None,
    help="Ray cluster address (e.g., ray://127.0.0.1:10001)",
)
def serve_streaming_start(
    model_path: str,
    tokenizer_path: str,
    app_name: str,
    route_prefix: str,
    num_replicas: int,
    max_tokens: int,
    max_workers: int,
    ray_address: str | None,
):
    """Start LLM streaming inference service."""
    from tributo.serving.streaming_runner import start_streaming_serving

    try:
        start_streaming_serving(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            app_name=app_name,
            route_prefix=route_prefix,
            num_replicas=num_replicas,
            max_tokens=max_tokens,
            max_workers=max_workers,
            ray_address=ray_address,
        )
        click.echo(f"Streaming serve app '{app_name}' started at {route_prefix}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@serve_streaming_group.command("stop")
@click.option(
    "--app-name",
    default="tributo-streaming",
    help="Serve application name",
)
@click.option(
    "--ray-address",
    default=None,
    help="Ray cluster address (e.g., ray://127.0.0.1:10001)",
)
def serve_streaming_stop(app_name: str, ray_address: str | None):
    """Stop LLM streaming inference service."""
    from tributo.serving.streaming_runner import stop_streaming_serving

    try:
        stopped = stop_streaming_serving(app_name, ray_address=ray_address)
        if stopped:
            click.echo(f"Streaming serve app '{app_name}' stopped.")
        else:
            click.echo(f"Failed to stop streaming serve app '{app_name}'.")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@serve_streaming_group.command("status")
@click.option(
    "--app-name",
    default="tributo-streaming",
    help="Serve application name",
)
@click.option(
    "--ray-address",
    default=None,
    help="Ray cluster address (e.g., ray://127.0.0.1:10001)",
)
def serve_streaming_status(app_name: str, ray_address: str | None):
    """Get LLM streaming inference service status."""
    from tributo.serving.streaming_runner import get_streaming_serving_status

    try:
        status = get_streaming_serving_status(app_name, ray_address=ray_address)
        click.echo(f"App: {status['app_name']}")
        click.echo(f"Running: {status['running']}")
        click.echo(f"Route: {status['route']}")
        click.echo(f"Status: {status['status']}")
        click.echo(f"Deployments: {', '.join(status['deployments'])}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@serve.group()
def grpc():
    """gRPC inference service management."""
    pass


@grpc.command("start")
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    help="Path to ONNX model file (legacy entry)",
)
@click.option(
    "--bundle-uri",
    help="Published bundle URI (stable serving entry point)",
)
@click.option(
    "--role",
    default="inference",
    show_default=True,
    help="Artifact role to serve from the bundle",
)
@click.option(
    "--unsafe",
    is_flag=True,
    default=False,
    help="Permit bundles without typed signatures or flavors that are "
    "not safe (compat-only)",
)
@click.option(
    "--storage-profile",
    default=None,
    help="Storage profile name for S3 bundles",
)
@click.option(
    "--app-name",
    default="tributo-grpc",
    help="Serve application name",
)
@click.option(
    "--grpc-port",
    default=8001,
    type=int,
    help="gRPC port",
)
@click.option(
    "--num-replicas",
    default=1,
    type=int,
    help="Number of serve replicas",
)
@click.option(
    "--ray-address",
    default=None,
    help="Ray cluster address (e.g., ray://127.0.0.1:10001)",
)
@click.option(
    "--enable-http/--disable-http",
    default=True,
    help="Whether to also start the HTTP proxy (default: enabled)",
)
def grpc_start(
    model_path: str | None,
    bundle_uri: str | None,
    role: str,
    unsafe: bool,
    storage_profile: str | None,
    app_name: str,
    grpc_port: int,
    num_replicas: int,
    ray_address: Optional[str],
    enable_http: bool,
):
    """Start gRPC inference service.

    Exactly one of ``--model-path`` (legacy) or ``--bundle-uri`` must be
    provided.
    """
    if (model_path is None) == (bundle_uri is None):
        raise click.ClickException(
            "exactly one of --model-path (legacy) or --bundle-uri must be provided"
        )
    from tributo.serving.grpc_runner import start_grpc_serving

    try:
        start_grpc_serving(
            model_path=model_path,
            bundle_uri=bundle_uri,
            role=role,
            unsafe=unsafe,
            storage_profile=storage_profile,
            app_name=app_name,
            grpc_port=grpc_port,
            num_replicas=num_replicas,
            ray_address=ray_address,
            enable_http=enable_http,
        )
        protocol = "HTTP+gRPC" if enable_http else "gRPC"
        click.echo(
            f"{protocol} serve app '{app_name}' started on gRPC port {grpc_port}"
        )
    except TributoError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


@grpc.command("stop")
@click.option(
    "--app-name",
    default="tributo-grpc",
    help="Serve application name",
)
@click.option(
    "--ray-address",
    default=None,
    help="Ray cluster address (e.g., ray://127.0.0.1:10001)",
)
def grpc_stop(app_name: str, ray_address: Optional[str]):
    """Stop gRPC inference service."""
    from tributo.serving.grpc_runner import stop_grpc_serving

    try:
        stopped = stop_grpc_serving(app_name, ray_address=ray_address)
        if stopped:
            click.echo(f"gRPC serve app '{app_name}' stopped.")
        else:
            click.echo(f"Failed to stop gRPC serve app '{app_name}'.")
    except TributoError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


@grpc.command("status")
@click.option(
    "--app-name",
    default="tributo-grpc",
    help="Serve application name",
)
@click.option(
    "--ray-address",
    default=None,
    help="Ray cluster address (e.g., ray://127.0.0.1:10001)",
)
def grpc_status(app_name: str, ray_address: Optional[str]):
    """Get gRPC inference service status."""
    from tributo.serving.grpc_runner import get_grpc_serving_status

    try:
        status = get_grpc_serving_status(app_name, ray_address=ray_address)
        click.echo(f"App: {status['app_name']}")
        click.echo(f"Running: {status['running']}")
        click.echo("Protocol: gRPC")
        click.echo(f"Status: {status['status']}")
        click.echo(f"Deployments: {', '.join(status['deployments'])}")
    except TributoError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


# ── Algo ────────────────────────────────────────────────────────────────────────


@main.group()
def algo():
    """Algorithm catalog: list, inspect, and validate algorithm configurations."""
    pass


@algo.command("list")
@click.option("--family", type=str, default=None, help="Problem family filter")
@click.option("--problem-type", type=str, default=None, help="Problem type filter")
@click.option("--modality", type=str, default=None, help="Data modality filter")
@click.option("--tag", type=str, default=None, help="Tag filter")
@click.option("--extras", type=str, default=None, help="Extras group filter")
@click.option(
    "--deprecated/--no-deprecated", default=False, help="Include deprecated algorithms"
)
@click.option("--json", "json_output", is_flag=True, default=False, help="JSON output")
def algo_list(
    family: str | None,
    problem_type: str | None,
    modality: str | None,
    tag: str | None,
    extras: str | None,
    deprecated: bool,
    json_output: bool,
):
    """List available algorithms."""
    from tributo.training.algorithm_spec import (
        PROBLEM_FAMILY_MAP,
        ProblemFamily,
        ProblemType,
    )
    from tributo.training.catalog import get_algorithm_catalog
    from tributo.training.support_snapshot import (
        build_algorithm_support_snapshot,
        snapshot_json_objects,
    )

    catalog = get_algorithm_catalog()

    try:
        pf = ProblemFamily(family) if family else None
    except ValueError as e:
        raise click.BadParameter(
            f"'{family}' is not a valid problem family. "
            f"Choices: {[pf.value for pf in ProblemFamily]}",
            param_hint="--family",
        ) from e
    try:
        pt = ProblemType(problem_type) if problem_type else None
    except ValueError as e:
        raise click.BadParameter(
            f"'{problem_type}' is not a valid problem type. "
            f"Choices: {[pt.value for pt in ProblemType]}",
            param_hint="--problem-type",
        ) from e

    records = catalog.list_records(
        problem_family=pf,
        problem_type=pt,
        modality=modality,
        tag=tag,
        extras_group=extras,
        include_deprecated=deprecated,
    )

    if json_output:
        result = snapshot_json_objects(build_algorithm_support_snapshot(records))
        click.echo(json.dumps(result, indent=2))
    else:
        if not records:
            click.echo("No algorithms found.")
            return
        header = (
            f"{'NAME':<20} {'FAMILY':<25} {'MODALITY':<12} {'GPU REQ':<8} "
            f"{'STATUS':<10} {'STABILITY':<9} {'AVAILABLE':<10} {'COMPAT':<8} "
            f"{'TESTED':<7} {'SUPPORTED':<9}"
        )
        click.echo(header)
        click.echo("-" * len(header))
        for record in records:
            spec = record.spec
            name = record.name
            families = sorted(
                {
                    pf.value
                    for pf in ProblemFamily
                    if spec is not None
                    and any(pt in spec.problem_types for pt in PROBLEM_FAMILY_MAP[pf])
                }
            )
            family_str = ", ".join(families) if families else "-"
            modality_str = (
                ", ".join(spec.data_modality) if spec and spec.data_modality else "-"
            )
            gpu_required = (
                "-"
                if spec is None
                else ("Y" if spec.resource_hints.gpu_required else "N")
            )
            status = spec.status.value if spec is not None else "-"
            click.echo(
                f"{name:<20} {family_str:<25} "
                f"{modality_str:<12} {gpu_required:<8} {status:<10} "
                f"{record.stability:<9} "
                f"{'Y' if record.available else 'N':<10} "
                f"{'Y' if record.compatibility_only else 'N':<8} "
                f"{'Y' if record.tested else 'N':<7} "
                f"{'Y' if record.supported else 'N':<9}"
            )


@algo.command("info")
@click.argument("name")
def algo_info(name: str):
    """Show detailed information about an algorithm."""
    from tributo.training.catalog import get_algorithm_catalog

    catalog = get_algorithm_catalog()
    try:
        record = catalog.get_record(name)
    except TributoError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    spec = record.spec
    click.echo(f"Name:           {record.name}")
    click.echo(f"Available:      {record.available}")
    click.echo(f"Compatibility:  {record.compatibility_only}")
    click.echo(f"Tested:         {record.tested}")
    click.echo(f"Supported:      {record.supported}")
    click.echo(f"Stability:      {record.stability}")
    click.echo(f"Implementations: {list(record.implementation_ids) or '-'}")
    click.echo(f"Topologies:     {list(record.runtime_topologies) or '-'}")
    click.echo(f"Distribution:   {list(record.distribution_strategies) or '-'}")
    click.echo(f"Profiles:       {list(record.execution_profiles) or '-'}")
    click.echo(f"Validated:      {list(record.validated_execution_profiles) or '-'}")
    click.echo(f"Input Views:    {list(record.input_views) or '-'}")
    click.echo(f"Limitations:    {list(record.limitations) or '-'}")
    if spec is None:
        return
    click.echo(f"Version:        {spec.version}")
    click.echo(f"Status:         {spec.status.value}")
    click.echo(f"Problem Types:  {[pt.value for pt in spec.problem_types] or '-'}")
    click.echo(f"Data Modality:  {list(spec.data_modality) or '-'}")
    click.echo(f"Tags:           {list(spec.tags) or '-'}")
    click.echo(f"GPU Required:   {spec.resource_hints.gpu_required}")
    click.echo(f"Min Memory GB:  {spec.resource_hints.min_memory_gb}")
    click.echo(f"Min CPUs:       {spec.resource_hints.min_cpus}")
    click.echo(f"Extras Group:   {spec.extras_group or '-'}")
    click.echo(f"Supported Tasks:{list(spec.supported_tasks)}")
    click.echo(f"Execution Kind: {spec.execution_kind.value}")
    click.echo(f"Capabilities:   {[cap.value for cap in spec.capabilities] or '-'}")
    click.echo(
        f"Config Model:   {spec.config_model.__name__ if spec.config_model else '-'}"
    )
    click.echo(f"Data Loading:   {spec.data_loading.value}")
    if spec.deprecated_since:
        click.echo(
            f"Deprecated:     since {spec.deprecated_since} → {spec.replacement}"
        )


@algo.command("config-schema")
@click.argument("name")
def algo_config_schema(name: str):
    """Print the JSON Schema for an algorithm's config model."""
    from tributo.training.catalog import get_algorithm_catalog

    catalog = get_algorithm_catalog()
    try:
        schema = catalog.get_config_schema(name)
    except TributoError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    click.echo(json.dumps(schema, indent=2))


@algo.command("validate")
@click.option("--algo", "algo_name", required=True, help="Algorithm name")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to config JSON file",
)
def algo_validate(algo_name: str, config_path: str):
    """Validate a training config against an algorithm's config model."""
    from tributo.training.catalog import get_algorithm_catalog
    from tributo.training.config import build_effective_config

    catalog = get_algorithm_catalog()
    try:
        spec = catalog.get_spec(algo_name)
    except TributoError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if spec.config_model is None:
        click.echo(
            f"Validation unavailable: algorithm '{algo_name}' "
            f"does not declare a config_model."
        )
        sys.exit(2)

    with open(config_path, encoding="utf-8") as f:
        user_config = json.load(f)

    try:
        build_effective_config(spec, user_config, datasets_supplied=False)
    except TributoError as e:
        click.echo(f"Validation FAILED:\n{e}", err=True)
        sys.exit(1)

    click.echo(f"Config is valid for algorithm '{algo_name}'.")


@algo.command("run")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the distributed algorithm execution JSON file",
)
def algo_run(config_path: str) -> None:
    """Run one formal algorithm through local[*] or an attached KubeRay job."""
    from pydantic import ValidationError

    from tributo._common.immutable import deep_thaw
    from tributo.algorithms.api import (
        AlgorithmRequest,
        ExecutionRequest,
        InputBinding,
        WorkerResources,
    )
    from tributo.algorithms.composition import build_algorithm_dispatcher
    from tributo.algorithms.core import LocalRuntimeOptions, RayRuntimeManager
    from tributo.algorithms.spi import InputExecutionContext, InputResolutionContext
    from tributo.config import AlgorithmExecutionConfig
    from tributo.integrations.algorithm_inputs import (
        INGESTION_RESOLVER_ID,
        IngestionInputInvocation,
    )

    path = Path(config_path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        raise click.ClickException(
            "YAML config is not supported; use one JSON execution envelope"
        )
    try:
        raw_config = json.loads(path.read_text(encoding="utf-8"))
        execution_config = AlgorithmExecutionConfig.model_validate(raw_config)
    except ValidationError as exc:
        diagnostics = "; ".join(
            f"{'.'.join(str(item) for item in error['loc'])}: {error['msg']}"
            for error in exc.errors(include_url=False, include_input=False)
        )
        raise click.ClickException(
            f"Distributed algorithm execution config is invalid: {diagnostics}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(
            f"Unable to read distributed algorithm JSON: {type(exc).__name__}"
        ) from exc

    request_key = "tributo.cli-input"
    invocation = IngestionInputInvocation(request=execution_config.input.ingestion)
    values = {request_key: invocation}
    binding = InputBinding(
        name="train",
        resolver_id=INGESTION_RESOLVER_ID,
        reference=request_key,
        feature_names=tuple(execution_config.input.features),
        label_name=execution_config.input.label,
        sample_weight_name=execution_config.input.sample_weight,
    )
    resources = execution_config.resources_per_worker
    local_runtime = execution_config.local_runtime
    manager = RayRuntimeManager(
        default_local_options=(
            LocalRuntimeOptions(
                num_cpus=local_runtime.num_cpus,
                num_gpus=local_runtime.num_gpus,
            )
            if local_runtime is not None
            else None
        )
    )
    request = ExecutionRequest(
        algorithm_request=AlgorithmRequest(
            algorithm=execution_config.algorithm,
            operation=execution_config.operation,
            input_binding=binding,
            algorithm_config=execution_config.algorithm_config,
            implementation_id=execution_config.implementation_id,
        ),
        profile=execution_config.profile,
        worker_count=execution_config.worker_count,
        resources_per_worker=(
            WorkerResources(
                num_cpus=resources.num_cpus,
                num_gpus=resources.num_gpus,
                custom=resources.custom,
            )
            if resources is not None
            else None
        ),
        resume_from=execution_config.resume_from,
    )
    dispatcher = build_algorithm_dispatcher(runtime_manager=manager)
    try:
        result = dispatcher.execute(
            request,
            InputExecutionContext(values=values),
            resolution_context=InputResolutionContext(values=values),
        )
    except TributoError as exc:
        raise click.ClickException(
            f"Distributed algorithm execution failed: {type(exc).__name__}: {exc}"
        ) from exc
    receipt = result.execution_receipt
    if receipt is None:
        raise click.ClickException(
            "Formal distributed execution completed without an ExecutionReceipt"
        )
    click.echo(
        json.dumps(
            {
                "run_id": result.run_id,
                "plan_id": result.plan_id,
                "status": result.execution.status,
                "metrics": deep_thaw(result.execution.metrics),
                "outputs": deep_thaw(result.execution.outputs),
                "execution_receipt": receipt.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )


# ── Tune ────────────────────────────────────────────────────────────────────────


@main.group()
def tune():
    """Hyperparameter tuning with Ray Tune."""
    pass


@tune.command("run")
@click.option(
    "--trainer",
    required=True,
    help="Trainer name (e.g., xgboost)",
)
@click.option(
    "--space",
    required=True,
    type=click.Path(exists=True),
    help="Search space JSON file path",
)
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True),
    help="Training config JSON file path",
)
@click.option(
    "--output",
    required=True,
    help="Output path (local or S3)",
)
@click.option(
    "--metric",
    default="loss",
    help="Optimization metric name",
)
@click.option(
    "--mode",
    default="min",
    type=click.Choice(["min", "max"]),
    help="Optimization direction",
)
@click.option(
    "--num-samples",
    default=1,
    type=int,
    help="Number of samples per configuration",
)
@click.option(
    "--search-alg",
    default="random",
    type=click.Choice(["random", "bayesopt"]),
    help="Search algorithm",
)
@click.option(
    "--scheduler",
    default="fifo",
    type=click.Choice(["fifo", "asha", "hyperband"]),
    help="Trial scheduler",
)
@click.option(
    "--max-concurrent",
    default=None,
    type=int,
    help="Max concurrent trials",
)
@click.option(
    "--time-budget",
    default=None,
    type=float,
    help="Global time budget in seconds",
)
@click.option(
    "--fail-fast",
    is_flag=True,
    default=False,
    help="Stop experiment on first trial failure",
)
@click.option(
    "--experiment-name",
    default="tributo-tune",
    help="Experiment name for Ray Tune",
)
@click.option(
    "--local",
    is_flag=True,
    default=False,
    help="Run a single local trial without a Ray cluster (dev/debug mode)",
)
def tune_run(
    trainer: str,
    space: str,
    config: str,
    output: str,
    metric: str,
    mode: str,
    num_samples: int,
    search_alg: str,
    scheduler: str,
    max_concurrent: int | None,
    time_budget: float | None,
    fail_fast: bool,
    experiment_name: str,
    local: bool,
):
    """Run hyperparameter search with Ray Tune (or local trial with --local)."""
    try:
        from tributo.training.registry import get_execution_registry, get_trainer

        # Parse config
        if Path(config).suffix.lower() in {".yaml", ".yml"}:
            raise ValueError(
                "YAML training config is no longer supported; please use JSON."
            )
        with open(config, encoding="utf-8") as f:
            training_cfg = json.load(f) or {}
        if not isinstance(training_cfg, dict):
            raise JobConfigurationError("Training config must be a JSON mapping")

        trainer_spec = get_trainer(trainer)
        distribution_spec = None
        try:
            from tributo.algorithms.api import AlgorithmOperation

            registration = get_execution_registry().resolve(
                algorithm=trainer,
                operation=AlgorithmOperation.FIT,
                implementation_id=None,
            )
            distribution_spec = registration.distribution_spec
        except TributoError:
            # Programmatic Beta-only Trainer registrations remain compatibility
            # paths and cannot claim formal distributed resource semantics.
            pass

        # Prep order: parse → merge → validate → search check → data → runner
        from tributo.training.config import (
            apply_dot_overrides,
            build_effective_config,
            resolve_data_source,
        )
        from tributo.training.tune_space import (
            parse_search_space,
            resolve_local_overrides,
            validate_search_targets,
            warn_search_space_conflicts,
        )

        space_spec = parse_search_space(space)
        effective = build_effective_config(
            trainer_spec, training_cfg, datasets_supplied=False
        )
        warn_search_space_conflicts(training_cfg, space_spec)
        validate_search_targets(effective, space_spec)

        if local:
            # ── local mode: single trial, no Ray ──────────────────────────
            from tributo.training.local_runner import run_local_trial

            overrides = resolve_local_overrides(space_spec, effective)
            local_config = apply_dot_overrides(effective, overrides)
            # Re-validate with dot overrides applied.
            from tributo.training.config import (
                validate_and_normalize_config,
                validate_execution_config,
            )

            local_config = validate_and_normalize_config(trainer_spec, local_config)
            validate_execution_config(
                trainer_spec, local_config, datasets_supplied=False
            )

            click.echo(f"Running local trial (trainer={trainer})...")
            result = run_local_trial(
                trainer_spec=trainer_spec,
                effective_config=local_config,
                output_path=output,
            )
            if result.get("status") == "succeeded":
                click.echo(
                    f"Local trial completed in {result['duration_sec']}s → "
                    f"{result.get('model_path', output)}"
                )
            else:
                click.echo(
                    f"Local trial failed after {result['duration_sec']}s", err=True
                )
                sys.exit(1)
            return

        # ── distributed mode: Ray Tune ───────────────────────────────────
        from tributo.training.data_loader import load_ray_dataset_from_source
        from tributo.training.tune_config import TuneSearchConfig
        from tributo.training.tune_runner import TuneRunner, extract_best_params

        tune_config = TuneSearchConfig(
            metric=metric,
            mode=mode,
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent,
            time_budget_s=time_budget,
            search_alg=search_alg,
            scheduler=scheduler,
            fail_fast=fail_fast,
        )

        # Load dataset from canonical source (DRIVER mode only).
        datasets: dict[str, Any] = {}
        if trainer_spec.data_loading.value != "canonical_trainer":
            source = resolve_data_source(trainer_spec, effective)
            train_ds = load_ray_dataset_from_source(source)
            datasets = {"train": train_ds}

        runner = TuneRunner(
            trainer_spec,
            tune_config,
            space_spec,
            effective_config=effective,
            distribution_spec=distribution_spec,
        )
        result_grid = runner.run(
            datasets=datasets,
            output_path=output,
            experiment_name=experiment_name,
        )

        # Output best result
        best_params = extract_best_params(result_grid, metric, mode)
        click.echo(f"Best parameters: {best_params}")

    except TributoError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


# ── Registry ───────────────────────────────────────────────────────────────────


@main.group()
@click.option(
    "--tracking-uri",
    default=None,
    help="MLflow tracking server URI (e.g., http://127.0.0.1:8050)",
)
@click.pass_context
def registry(ctx: click.Context, tracking_uri: str | None):
    """Model registry management (MLflow)."""
    ctx.ensure_object(dict)
    ctx.obj["tracking_uri"] = tracking_uri


def _handle_registry_error(error: Exception) -> None:
    """Unified handler for common exceptions in registry commands."""
    if isinstance(error, ImportError):
        click.echo(
            "Error: mlflow is required for registry commands. "
            "Install with: pip install tributo[registry]",
            err=True,
        )
    elif isinstance(error, ValueError):
        click.echo(f"Error: {error}", err=True)
    else:
        click.echo(f"Unexpected error: {error}", err=True)
    sys.exit(1)


@registry.command("register")
@click.option("--name", required=True, help="Model name")
@click.option(
    "--uri",
    required=True,
    help="Model URI (e.g., runs:/<run_id>/model)",
)
@click.option("--description", default=None, help="Model description")
@click.pass_context
def registry_register(ctx: click.Context, name: str, uri: str, description: str | None):
    """Register a model to Model Registry."""
    try:
        from tributo.registry import ModelRegistry

        reg = ModelRegistry(tracking_uri=ctx.obj.get("tracking_uri"))
        mv = reg.register_model(model_uri=uri, name=name, description=description)
        click.echo(f"Registered: {mv.name} v{mv.version}")
    except TributoError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        _handle_registry_error(e)


@registry.command("get")
@click.option("--name", required=True, help="Model name")
@click.option("--version", type=int, default=None, help="Version number")
@click.option(
    "--stage",
    default=None,
    help="Model stage (Staging/Production/Archived)",
)
@click.pass_context
def registry_get(ctx: click.Context, name: str, version: int | None, stage: str | None):
    """Get model version info."""
    try:
        from tributo.registry import ModelRegistry

        reg = ModelRegistry(tracking_uri=ctx.obj.get("tracking_uri"))
        mv = reg.get_model(name=name, version=version, stage=stage)
        click.echo(mv.model_dump_json(indent=2))
    except TributoError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        _handle_registry_error(e)


@registry.command("list")
@click.pass_context
def registry_list(ctx: click.Context):
    """List all registered models."""
    try:
        from tributo.registry import ModelRegistry

        reg = ModelRegistry(tracking_uri=ctx.obj.get("tracking_uri"))
        models = reg.list_models()
        for m in models:
            click.echo(m)
    except TributoError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        _handle_registry_error(e)


@registry.command("transition")
@click.option("--name", required=True, help="Model name")
@click.option("--version", required=True, type=int, help="Version number")
@click.option(
    "--stage",
    required=True,
    help="Target stage (Staging/Production/Archived)",
)
@click.pass_context
def registry_transition(ctx: click.Context, name: str, version: int, stage: str):
    """Transition model stage."""
    try:
        from tributo.registry import ModelRegistry

        reg = ModelRegistry(tracking_uri=ctx.obj.get("tracking_uri"))
        reg.transition_stage(name=name, version=version, stage=stage)
        click.echo(f"Transitioned {name} v{version} to {stage}")
    except TributoError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        _handle_registry_error(e)


@registry.command("delete-version")
@click.option("--name", required=True, help="Model name")
@click.option(
    "--version",
    required=True,
    type=int,
    help="Version number to delete",
)
@click.pass_context
def registry_delete_version(ctx: click.Context, name: str, version: int):
    """Delete a specific model version."""
    try:
        from tributo.registry import ModelRegistry

        reg = ModelRegistry(tracking_uri=ctx.obj.get("tracking_uri"))
        reg.delete_model_version(name=name, version=version)
        click.echo(f"Deleted {name} v{version}")
    except TributoError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        _handle_registry_error(e)


@registry.command("delete-model")
@click.option("--name", required=True, help="Model name")
@click.pass_context
def registry_delete_model(ctx: click.Context, name: str):
    """Delete a registered model and all its versions."""
    try:
        from tributo.registry import ModelRegistry

        reg = ModelRegistry(tracking_uri=ctx.obj.get("tracking_uri"))
        reg.delete_model(name=name)
        click.echo(f"Deleted model '{name}'")
    except TributoError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        _handle_registry_error(e)


main.add_command(vector_group)


if __name__ == "__main__":
    main()
