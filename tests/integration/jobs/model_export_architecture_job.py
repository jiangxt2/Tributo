"""Ray Job for the required Parquet → Trainer → Bundle → Batch → Serve path."""

from __future__ import annotations

import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any


def _delete_bucket(client: Any, bucket: str) -> None:
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        objects = page.get("Contents", [])
        if objects:
            client.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [{"Key": item["Key"]} for item in objects]},
            )
    client.delete_bucket(Bucket=bucket)


def main() -> int:
    """Run the complete model-export architecture on Docker-owned services."""
    # Ray Train may remove its temporary storage directory before deferred
    # framework imports finish. Keep Python reflection independent of that
    # lifecycle-owned directory.
    os.chdir("/tmp")

    import importlib.metadata

    import httpx
    import mlflow
    import ray
    from ray import serve
    from ray.data import DataContext

    from tributo._common.storage import get_boto3_client
    from tributo.exporting.bundle_reader import BundleReader
    from tributo.exporting.models import BundleOutputConfig, HookBinding
    from tributo.inference.batch_predictor import XGBoostONNXPredictor
    from tributo.serving.model_deployment import ONNXModel
    from tributo.training.xgboost_trainer import XGBoostTrainerImpl

    DataContext.get_current().enable_progress_bars = False

    execution_suffix = uuid.uuid4().hex
    root = Path("/workspace/tributo-work") / f"model-export-{execution_suffix}"
    root.mkdir(parents=True)
    bucket = f"tributo-model-export-job-{uuid.uuid4().hex[:12]}"
    client = get_boto3_client(path_style=True)
    client.create_bucket(Bucket=bucket)
    mlflow_client = mlflow.MlflowClient()
    model_versions_before = {
        (version.name, version.version)
        for version in mlflow_client.search_model_versions()
    }
    experiment_name = f"tributo-golden-path-{execution_suffix}"
    serve_started = False
    result_payload: dict[str, Any] | None = None
    try:
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
        parquet_path = root / "parquet"
        ray.data.from_items(rows).write_parquet(str(parquet_path))
        dataset = ray.data.read_parquet(str(parquet_path))

        source = {
            "provider": "tributo.parquet",
            "uri": str(parquet_path),
            "options": {},
        }
        trainer = XGBoostTrainerImpl(
            datasets={"train": dataset},
            config={
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
                    "storage_path": str(root / "ray-results"),
                    "max_failures": 0,
                },
            },
        )

        config = BundleOutputConfig(
            bundle_uri=f"s3://{bucket}/models",
            storage_profile="test",
            hooks=(
                HookBinding(
                    hook_id="mlflow-log-artifacts-v1",
                    required=True,
                    options={
                        "tracking_uri": mlflow.get_tracking_uri(),
                        "experiment_name": experiment_name,
                        "run_name": "golden-path",
                    },
                ),
            ),
        )
        summary = trainer.run(bundle_config=config)
        assert summary["training_status"] == "succeeded"
        assert summary["bundle_status"] == "succeeded"
        assert summary["hook_status"] == "succeeded"

        reader = BundleReader(cache_dir=root / "cache")
        manifest = reader.read_manifest(summary["bundle_uri"], storage_profile="test")
        assert manifest.roles == {"inference": "onnx-model"}
        formats = {artifact.name: artifact.format for artifact in manifest.artifacts}
        assert formats == {
            "native": "ubj",
            "onnx-model": "onnx",
        }
        assert all(artifact.artifact_kind == "model" for artifact in manifest.artifacts)

        prediction_dataset = ray.data.from_items(
            [
                {"feature_a": 0.0, "feature_b": 1.0},
                {"feature_a": 1.0, "feature_b": 0.0},
            ]
        )
        prediction_rows = prediction_dataset.map_batches(
            XGBoostONNXPredictor,
            fn_constructor_kwargs={
                "bundle_uri": summary["bundle_uri"],
                "predictor_config": {"return_probs": False},
                "storage_profile": "test",
            },
            batch_format="numpy",
        ).take_all()
        assert len(prediction_rows) == 2

        serve.start(http_options={"host": "0.0.0.0", "port": 8000})
        serve_started = True
        deployment = serve.deployment(
            name=f"tributo-golden-model-{execution_suffix[:8]}",
            num_replicas=1,
        )(ONNXModel)
        serve.run(
            deployment.bind(
                bundle_uri=summary["bundle_uri"],
                role="inference",
                storage_profile="test",
            ),
            name=f"tributo-golden-{execution_suffix[:8]}",
            route_prefix="/predict",
        )
        response = httpx.post(
            "http://127.0.0.1:8000/predict",
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
        assert response.status_code == 200, response.text
        http_payload = response.json()
        assert len(http_payload["predictions"]) == 2
        assert http_payload["bundle_id"] == manifest.bundle_id

        experiment = mlflow_client.get_experiment_by_name(experiment_name)
        assert experiment is not None
        runs = mlflow_client.search_runs([experiment.experiment_id])
        assert len(runs) == 1
        assert runs[0].data.tags["tributo.bundle_uri"] == summary["bundle_uri"]
        assert (
            runs[0].data.tags["tributo.manifest_sha256"] == summary["manifest_sha256"]
        )
        model_versions_after = {
            (version.name, version.version)
            for version in mlflow_client.search_model_versions()
        }
        assert model_versions_after == model_versions_before

        expected_versions = {
            "boto3": "BOTO3_VERSION",
            "botocore": "BOTOCORE_VERSION",
            "ray": "RAY_VERSION",
            "mlflow": "MLFLOW_VERSION",
            "xgboost": "XGBOOST_VERSION",
            "onnx": "ONNX_VERSION",
            "onnxruntime": "ONNXRUNTIME_VERSION",
            "onnxmltools": "ONNXMLTOOLS_VERSION",
            "torch": "TORCH_VERSION",
            "transformers": "TRANSFORMERS_VERSION",
            "pyarrow": "PYARROW_VERSION",
            "pandas": "PANDAS_VERSION",
        }
        observed_versions = {
            package: importlib.metadata.version(package)
            for package in expected_versions
        }
        for package, environment_key in expected_versions.items():
            assert observed_versions[package] == os.environ[environment_key]
        observed_python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        assert observed_python_version == os.environ["PYTHON_VERSION"]

        alive_nodes = [node for node in ray.nodes() if node["Alive"]]
        assert len(alive_nodes) >= 2

        result_payload = {
            "status": summary["status"],
            "bundle_id": manifest.bundle_id,
            "execution_id": summary["execution_id"],
            "manifest_sha256": summary["manifest_sha256"],
            "artifact_kinds": sorted(
                {artifact.artifact_kind for artifact in manifest.artifacts}
            ),
            "formats": formats,
            "batch_rows": len(prediction_rows),
            "http_rows": len(http_payload["predictions"]),
            "mlflow_runs": len(runs),
            "model_versions_created": 0,
            "python_version": observed_python_version,
            "versions": observed_versions,
            "alive_nodes": len(alive_nodes),
        }
    finally:
        if serve_started:
            serve.shutdown()
        experiment = mlflow_client.get_experiment_by_name(experiment_name)
        if experiment is not None:
            for run in mlflow_client.search_runs([experiment.experiment_id]):
                mlflow_client.delete_run(run.info.run_id)
            mlflow_client.delete_experiment(experiment.experiment_id)
        _delete_bucket(client, bucket)
        shutil.rmtree(root, ignore_errors=True)

    assert result_payload is not None
    print("RESULT: " + json.dumps(result_payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
