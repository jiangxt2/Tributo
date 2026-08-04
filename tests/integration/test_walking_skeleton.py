"""Required O1 walking skeleton: Parquet → XGBoost → Bundle → Serving."""

from __future__ import annotations

import json
import socket
import sys
import uuid
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.support.object_storage import S3Service

pytestmark = [
    pytest.mark.integration,
    pytest.mark.minio_compat,
    pytest.mark.tributo_walking_skeleton,
    # Ray 2.55.1 leaks /dev/null file handles from its internal process
    # spawn (gcs/raylet/monitor); the garbage-collected handles surface as
    # unraisable ResourceWarnings that the project-wide
    # filterwarnings=error would turn into failures. This test cannot fix
    # Ray internals, so suppress that warning class here only.
    pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning"),
    pytest.mark.skipif(
        sys.platform == "darwin",
        reason="Ray Data local runtime is supported by the Linux CI gate, not macOS",
    ),
]


def _available_port() -> int:
    """Return an unused loopback port for the local Ray Serve HTTP proxy."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture()
def local_ray_runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Own the two-CPU Ray runtime used by the required walking skeleton."""
    import ray
    from ray._private import ray_constants

    if ray.is_initialized():
        raise RuntimeError(
            "walking skeleton requires an uninitialized local Ray runtime"
        )

    # CI launches pytest under `uv run`. Ray 2.55 then propagates the uv
    # runtime environment to workers by default, so raylet rebuilds the
    # project env inside the uploaded working dir, fails, and falls back to
    # the system interpreter (no ray module): worker startup repeatedly
    # fails and Ray Data tasks never schedule. Mirror the conftest.py
    # ray_local_runtime fixture and disable the propagation.
    previous_uv_runtime_env = ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV
    ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
    monkeypatch.setenv("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    try:
        ray.init(include_dashboard=False, num_cpus=2)
        print("[ws] ray initialized", flush=True)
        yield
    finally:
        if ray.is_initialized():
            ray.shutdown()
        ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = previous_uv_runtime_env


@pytest.fixture()
def minio_service(monkeypatch: pytest.MonkeyPatch) -> Iterator[S3Service]:
    """Own a required MinIO service without converting outages to skips."""
    service = S3Service.start_minio()
    print("[ws] minio started", flush=True)
    monkeypatch.setenv("S3_ENDPOINT", service.endpoint)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "minioadmin")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    try:
        yield service
    finally:
        service.close()


@pytest.fixture()
def minio_bucket(
    minio_service: S3Service,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[str]:
    """Create and clean an isolated MinIO bucket for the walking skeleton."""

    from tributo._common.storage import get_boto3_client

    monkeypatch.setenv(
        "TRIBUTO_STORAGE_PROFILE_TEST",
        json.dumps({"path_style": True}),
    )
    client = get_boto3_client(path_style=True)
    bucket = f"tributo-walking-skeleton-{uuid.uuid4().hex[:12]}"
    client.create_bucket(Bucket=bucket)
    print(f"[ws] bucket created {bucket}", flush=True)
    try:
        yield bucket
    finally:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket):
            objects = page.get("Contents", [])
            if objects:
                client.delete_objects(
                    Bucket=bucket,
                    Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
                )
        client.delete_bucket(Bucket=bucket)


def _bundle_config(bundle_uri: str) -> Any:
    """Build the single required ONNX inference role for the bundle."""
    from tributo.exporting.models import BundleOutputConfig, ExportTarget

    return BundleOutputConfig(
        bundle_uri=bundle_uri,
        storage_profile="test",
        targets=[
            ExportTarget(
                name="xgboost-onnx",
                format="onnx",
                # XGBoostONNXOptions locks opset to 12 (onnxmltools
                # constraint); 18 is the TorchONNXOptions value used by the
                # DNN/PU vertical slices.
                options={"opset": 12},
            )
        ],
        roles={"inference": "xgboost-onnx"},
    )


def _training_config(source: dict[str, Any], storage_path: Path) -> dict[str, Any]:
    """Return a bounded, deterministic XGBoost training configuration."""
    return {
        "data": {
            "source": source,
            "label_col": "label",
            "feature_columns": ["feature_a", "feature_b"],
        },
        "model": {
            "objective": "binary:logistic",
            "max_depth": 2,
            "eta": 0.3,
        },
        "training": {
            "num_rounds": 3,
            "val_size": 0.0,
            "test_size": 0.0,
            "max_rows_per_worker": 16,
            "seed": 42,
        },
        "ray": {
            "num_workers": 1,
            "use_gpu": False,
            "storage_path": str(storage_path),
            "max_failures": 0,
        },
    }


def test_walking_skeleton(
    local_ray_runtime: None,
    minio_bucket: str,
    tmp_path: Path,
) -> None:
    """Run the required Parquet → training → Bundle → batch/HTTP path."""
    import httpx
    import ray
    from ray import serve

    from tributo.exporting.bundle_reader import BundleReader
    from tributo.inference.batch_predictor import XGBoostONNXPredictor
    from tributo.serving.model_deployment import ONNXModel
    from tributo.training.data_loader import load_ray_dataset_from_source
    from tributo.training.xgboost_trainer import XGBoostTrainerImpl

    rows = [
        {"feature_a": 0.0, "feature_b": 0.0, "label": 0},
        {"feature_a": 0.0, "feature_b": 1.0, "label": 1},
        {"feature_a": 1.0, "feature_b": 0.0, "label": 1},
        {"feature_a": 1.0, "feature_b": 1.0, "label": 0},
        {"feature_a": 0.2, "feature_b": 0.1, "label": 0},
        {"feature_a": 0.1, "feature_b": 0.9, "label": 1},
        {"feature_a": 0.8, "feature_b": 0.2, "label": 1},
        {"feature_a": 0.9, "feature_b": 0.8, "label": 0},
    ]
    parquet_path = tmp_path / "training-data"
    ray.data.from_items(rows).write_parquet(str(parquet_path))
    print("[ws] parquet written", flush=True)

    source = {
        "provider": "tributo.parquet",
        "uri": str(parquet_path),
        "options": {},
    }
    dataset = load_ray_dataset_from_source(source)
    print("[ws] dataset loaded", flush=True)
    trainer = XGBoostTrainerImpl(
        datasets={"train": dataset},
        config=_training_config(source, tmp_path / "ray-results"),
    )
    print("[ws] trainer built", flush=True)

    bundle_uri = f"s3://{minio_bucket}/models"
    print("[ws] trainer.run start", flush=True)
    summary = trainer.run(bundle_config=_bundle_config(bundle_uri))
    print(f"[ws] trainer.run done: {summary['status']}", flush=True)

    assert summary["status"] == "succeeded"
    assert summary["canonical_uri"].startswith(bundle_uri)

    reader = BundleReader()
    manifest = reader.read_manifest(summary["canonical_uri"], storage_profile="test")
    print("[ws] manifest read", flush=True)
    assert manifest.schema_version == 1
    assert manifest.input_signature is not None
    assert manifest.output_signature is not None
    assert manifest.roles["inference"] == "xgboost-onnx"

    predictor = XGBoostONNXPredictor(
        bundle_uri=summary["canonical_uri"],
        predictor_config={"return_probs": False},
        storage_profile="test",
    )
    print("[ws] predictor built", flush=True)
    try:
        predictions = predictor(
            {
                "feature_a": np.asarray([0.0, 1.0], dtype=np.float32),
                "feature_b": np.asarray([1.0, 0.0], dtype=np.float32),
            }
        )
        print("[ws] predictor done", flush=True)
        assert predictions["prediction"].shape == (2,)
    finally:
        predictor.close()

    port = _available_port()
    app_name = "tributo-walking-skeleton"
    serve_started = False
    try:
        try:
            print("[ws] serve.start", flush=True)
            serve.start(
                http_options={"host": "127.0.0.1", "port": port},
            )
            print("[ws] serve started", flush=True)
        except BaseException:
            # Serve may have started its controller before a proxy/deployment
            # error is raised; preserve the original failure after cleanup.
            with suppress(Exception):
                serve.shutdown()
            raise
        serve_started = True
        deployment = serve.deployment(
            name="tributo-walking-skeleton-model",
            num_replicas=1,
        )(ONNXModel)
        print("[ws] serve.run start", flush=True)
        serve.run(
            deployment.bind(
                bundle_uri=summary["canonical_uri"],
                role="inference",
                storage_profile="test",
            ),
            name=app_name,
            route_prefix="/predict",
        )
        print("[ws] serve running", flush=True)

        print("[ws] http request", flush=True)
        response = httpx.post(
            f"http://127.0.0.1:{port}/predict",
            json={
                "inputs": [
                    {
                        "name": "float_input",
                        "shape": [2, 2],
                        "datatype": "float32",
                        "data": [0.0, 1.0, 1.0, 0.0],
                    }
                ],
                "return_probs": False,
            },
            timeout=30.0,
        )
        print("[ws] http done", flush=True)
        assert response.status_code == 200, response.text
        payload = response.json()
        assert len(payload["predictions"]) == 2
        assert payload["bundle_id"] == manifest.bundle_id
    finally:
        if serve_started:
            serve.shutdown()
