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
from tributo.exceptions import JobConfigurationError, TributoError
from tributo.job import TributoClient

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
    json_output: bool,
):
    """Export a trained model to one or more formats as a bundle."""
    try:
        from tributo.exporting.models import (
            AliasConfig,
            BundleOutputConfig,
            ExportTarget,
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

        # Build config.
        config = BundleOutputConfig(
            bundle_uri=output,
            targets=target_list,
            request_id=request_id,
            storage_profile=storage_profile,
            roles=roles_dict,
            alias=AliasConfig(name=alias) if alias else None,
        )

        # Resolve source provider by scheme.
        source_uri = source
        if source.startswith("ray://"):
            source_uri = source[6:]  # strip ray:// prefix
        elif source.startswith("hf://"):
            source_uri = source[5:]  # strip hf:// prefix

        from tributo.exporting.protocols import SourceProvider

        provider: SourceProvider
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
                provider=provider,
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

    Only prefixes that look like bundle IDs, have no manifest, are older
    than --orphan-ttl, and are not lease-protected are deleted.
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
    try:
        from importlib.metadata import version

        return version("tributo")
    except Exception:
        return "0.0.0"


@main.group()
def serve():
    """Manage ONNX inference service."""
    pass


@serve.command("start")
@click.option(
    "--model-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to ONNX model file",
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
    model_path: str,
    app_name: str,
    route_prefix: str,
    num_replicas: int,
    ray_address: str | None,
):
    """Start ONNX inference service."""
    from tributo.serving import start_serving

    try:
        start_serving(
            model_path=model_path,
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
    required=True,
    type=click.Path(exists=True),
    help="Path to ONNX model file",
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
    model_path: str,
    app_name: str,
    grpc_port: int,
    num_replicas: int,
    ray_address: Optional[str],
    enable_http: bool,
):
    """Start gRPC inference service."""
    from tributo.serving.grpc_runner import start_grpc_serving

    try:
        start_grpc_serving(
            model_path=model_path,
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


@main.group()
def embed():
    """Batch text embedding jobs."""
    pass


@embed.command("batch")
@click.option("--input", "input_path", required=True, help="S3 input Parquet path")
@click.option("--output", "output_path", required=True, help="S3 output path")
@click.option("--model", "model_name", default="bge-small-zh", help="Model name")
@click.option("--text-column", default="text", help="Text column name")
@click.option("--batch-size", default=64, type=int, help="Inference batch size")
@click.option("--concurrency", default=4, type=int, help="Number of actors")
@click.option(
    "--address",
    default=DEFAULT_DASHBOARD_URL,
    help="Ray Dashboard address",
)
def embed_batch(
    input_path: str,
    output_path: str,
    model_name: str,
    text_column: str,
    batch_size: int,
    concurrency: int,
    address: str,
):
    """Submit a batch embedding job to the Ray cluster."""
    from tributo.embeddings.job_runner import submit_embedding_job

    try:
        job_id = submit_embedding_job(
            s3_input_path=input_path,
            s3_output_path=output_path,
            model_name=model_name,
            text_column=text_column,
            batch_size=batch_size,
            concurrency=concurrency,
            dashboard_url=address,
        )
        click.echo(f"Embedding job submitted: {job_id}")
    except TributoError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


@embed.command("export")
@click.option("--model", "model_name", default="bge-small-zh", help="Model name")
@click.option("--output-dir", required=True, help="Local output directory")
def embed_export(model_name: str, output_dir: str):
    """Export a registered model to ONNX + tokenizer."""
    try:
        from tributo.embeddings.exporter import export_model
        from tributo.embeddings.registry import get_spec

        spec = get_spec(model_name)
        export_model(spec.hf_model_id, Path(output_dir))
        click.echo(f"Model exported to {output_dir}")
    except TributoError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


@embed.command("list")
def embed_list():
    """List registered embedding models."""
    from tributo.embeddings.registry import list_models

    for name in list_models():
        click.echo(name)


@embed.group(name="serve")
def embed_serve_group():
    """Manage text embedding inference service."""
    pass


@embed_serve_group.command("start")
@click.option(
    "--model-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to exported model directory (contains model.onnx + tokenizer)",
)
@click.option("--model-name", default=None, help="Short registered model name")
@click.option(
    "--app-name",
    default="tributo-embed",
    help="Serve application name",
)
@click.option(
    "--route-prefix",
    default="/embed",
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
def embed_serve_start(
    model_path: str,
    model_name: str | None,
    app_name: str,
    route_prefix: str,
    num_replicas: int,
    ray_address: str | None,
):
    """Start text embedding HTTP service."""
    from tributo.embeddings.serve_runner import start_embedding_serving

    try:
        start_embedding_serving(
            model_path=model_path,
            model_name=model_name,
            app_name=app_name,
            route_prefix=route_prefix,
            num_replicas=num_replicas,
            ray_address=ray_address,
        )
        click.echo(f"Embedding serve app '{app_name}' started at {route_prefix}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@embed_serve_group.command("stop")
@click.option(
    "--app-name",
    default="tributo-embed",
    help="Serve application name",
)
def embed_serve_stop(app_name: str):
    """Stop text embedding service."""
    from tributo.embeddings.serve_runner import stop_embedding_serving

    try:
        stopped = stop_embedding_serving(app_name)
        if stopped:
            click.echo(f"Embedding serve app '{app_name}' stopped.")
        else:
            click.echo(f"Failed to stop embedding serve app '{app_name}'.")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@embed_serve_group.command("status")
@click.option(
    "--app-name",
    default="tributo-embed",
    help="Serve application name",
)
def embed_serve_status(app_name: str):
    """Get text embedding service status."""
    from tributo.embeddings.serve_runner import get_embedding_serving_status

    try:
        status = get_embedding_serving_status(app_name)
        click.echo(f"App: {status['app_name']}")
        click.echo(f"Running: {status['running']}")
        click.echo(f"Route: {status['route']}")
        click.echo(f"Status: {status['status']}")
        click.echo(f"Deployments: {', '.join(status['deployments'])}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
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

    names = catalog.list(
        problem_family=pf,
        problem_type=pt,
        modality=modality,
        tag=tag,
        extras_group=extras,
        include_deprecated=deprecated,
    )

    if json_output:
        result = []
        for name in names:
            spec = catalog.get_spec(name)
            result.append(
                {
                    "name": name,
                    "problem_types": [pt.value for pt in spec.problem_types],
                    "data_modality": list(spec.data_modality),
                    "tags": list(spec.tags),
                    "gpu_required": spec.resource_hints.gpu_required,
                    "status": spec.status.value,
                    "extras_group": spec.extras_group,
                }
            )
        click.echo(json.dumps(result, indent=2))
    else:
        if not names:
            click.echo("No algorithms found.")
            return
        header = f"{'NAME':<20} {'FAMILY':<25} {'MODALITY':<12} {'GPU REQ':<8} {'STATUS':<10}"
        click.echo(header)
        click.echo("-" * len(header))
        for name in names:
            spec = catalog.get_spec(name)
            families = sorted(
                {
                    pf.value
                    for pf in ProblemFamily
                    if any(pt in spec.problem_types for pt in PROBLEM_FAMILY_MAP[pf])
                }
            )
            family_str = ", ".join(families) if families else "-"
            modality_str = ", ".join(spec.data_modality) if spec.data_modality else "-"
            click.echo(
                f"{name:<20} {family_str:<25} {modality_str:<12} "
                f"{'Y' if spec.resource_hints.gpu_required else 'N':<8} "
                f"{spec.status.value:<10}"
            )


@algo.command("info")
@click.argument("name")
def algo_info(name: str):
    """Show detailed information about an algorithm."""
    from tributo.training.catalog import get_algorithm_catalog

    catalog = get_algorithm_catalog()
    try:
        spec = catalog.get_spec(name)
    except TributoError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"Name:           {spec.name}")
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
        from tributo.training.registry import get_trainer

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

        # Phase 3 prep order: parse → merge → validate → search check → data → runner
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
            trainer_spec, tune_config, space_spec, effective_config=effective
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


if __name__ == "__main__":
    main()
