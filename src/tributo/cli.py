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


# ── Tune ────────────────────────────────────────────────────────────────────────


@main.group()
def tune():
    """Hyperparameter tuning with Ray Tune."""
    pass


def _detect_search_space_conflicts(
    training_cfg: dict[str, Any],
    search_space: dict[str, Any],
) -> list[str]:
    """Detect keys defined in both training config and search space.

    Search space values override fixed config values in TuneRunner, so warn
    the user to avoid surprising behavior.
    """
    conflicts: list[str] = []
    for section in ("model", "training", "ray", "output"):
        section_cfg = training_cfg.get(section)
        if not isinstance(section_cfg, dict):
            continue
        for key in section_cfg:
            if key in search_space:
                conflicts.append(f"{section}.{key}")
    return conflicts


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

        if local:
            # ── local mode: single trial, no Ray ──────────────────────────
            from tributo.training.local_runner import run_local_trial
            from tributo.training.tune_space import parse_search_space

            search_space = parse_search_space(space)

            click.echo(f"Running local trial (trainer={trainer})...")
            result = run_local_trial(
                trainer_spec=trainer_spec,
                training_config=training_cfg,
                output_path=output,
                search_space=search_space,
            )
            if result["status"] == "success":
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
        from tributo.training.data_loader import load_ray_dataset_from_config
        from tributo.training.tune_config import TuneSearchConfig
        from tributo.training.tune_runner import TuneRunner, extract_best_params
        from tributo.training.tune_space import parse_search_space

        search_space = parse_search_space(space)

        conflicts = _detect_search_space_conflicts(training_cfg, search_space)
        if conflicts:
            logger.warning(
                "The following keys are defined in both training config and search space; "
                "search space values will override: %s",
                conflicts,
            )

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

        # Load dataset
        data_cfg = training_cfg.get("data")
        if not data_cfg:
            raise JobConfigurationError("Training config must contain 'data' section")
        train_ds = load_ray_dataset_from_config(data_cfg)
        datasets = {"train": train_ds}

        # Run tuning
        runner = TuneRunner(trainer_spec, tune_config, search_space)
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
