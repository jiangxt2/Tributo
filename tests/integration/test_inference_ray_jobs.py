"""True Docker Ray Jobs integration tests for batch inference.

Required infrastructure is the repository's Ray 2.55.1 Docker cluster.  The
host and containers share one bounded test directory so workers can read the
same Parquet input and immutable Bundle and the host can inspect sink output.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import pytest
from ray.job_submission import JobStatus, JobSubmissionClient

from tests.serving.bundle_fixtures import build_test_bundle
from tests.support.mlflow_service import MLflowService
from tests.support.object_storage import (
    MINIO_ACCESS_KEY_ID,
    MINIO_SECRET_ACCESS_KEY,
    S3Service,
)
from tributo._common import DEFAULT_DASHBOARD_URL, build_runtime_env
from tributo._common.storage import get_boto3_client
from tributo._common.submission_id import generate_submission_id
from tributo.data import IngestionRequest
from tributo.data.source_config import ParquetSourceConfig
from tributo.exporting.bundle_reader import BundleReader
from tributo.exporting.models import BundleRef
from tributo.inference.api import resolve_inference
from tributo.inference.contracts import (
    ArtifactModelReference,
    BundleModelReference,
    InferenceRequest,
    InputBindingSpec,
    OutputBindingSpec,
    ParquetResultSinkRequest,
    RayExecutionPolicy,
    RegistryModelReference,
    TensorInputBinding,
    TensorOutputBinding,
)
from tributo.inference.job_runner import (
    submit_inference_request,
    submit_resolved_inference,
    wait_for_job,
)
from tributo.inference.post_training import (
    PostTrainingInferenceAction,
    submit_post_training_inference,
)
from tributo.integrations.model_importers._bundle import republish_verified_bundle

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class _Assets:
    host_root: Path
    container_root: str
    host_bundle: Path
    container_bundle: str
    manifest_sha256: str
    bundle_id: str


@dataclass(frozen=True)
class _S3Assets:
    service: S3Service
    bucket: str
    model_bundle: BundleRef
    host_profile: str
    cluster_profile: str


@pytest.fixture(scope="module")
def minio_assets(inference_assets: _Assets) -> Iterator[_S3Assets]:
    service = S3Service.start_minio()
    host_profile = _storage_profile(service.endpoint)
    cluster_endpoint = service.endpoint.replace("127.0.0.1", "host.docker.internal")
    cluster_profile = _storage_profile(cluster_endpoint)
    profile_names = ("source_domain", "model_domain", "sink_domain")
    saved = {
        _profile_env(name): os.environ.get(_profile_env(name)) for name in profile_names
    }
    for name in profile_names:
        os.environ[_profile_env(name)] = host_profile

    client = get_boto3_client(
        endpoint=service.endpoint,
        access_key_id=MINIO_ACCESS_KEY_ID,
        secret_access_key=MINIO_SECRET_ACCESS_KEY,
        region="us-east-1",
        use_ssl=False,
        path_style=True,
    )
    bucket = f"tributo-inference-it-{uuid.uuid4().hex[:12]}"
    client.create_bucket(Bucket=bucket)
    client.upload_file(
        str(inference_assets.host_root / "input" / "part-0.parquet"),
        bucket,
        "input/part-0.parquet",
    )
    original = BundleReader().read_manifest(str(inference_assets.host_bundle))
    model_bundle = republish_verified_bundle(
        source_bundle_uri=str(inference_assets.host_bundle),
        destination_uri=f"s3://{bucket}/models",
        destination_storage_profile="model_domain",
        source_info=original.source_info,
    )
    try:
        yield _S3Assets(
            service=service,
            bucket=bucket,
            model_bundle=model_bundle,
            host_profile=host_profile,
            cluster_profile=cluster_profile,
        )
    finally:
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket):
            objects = page.get("Contents", [])
            if objects:
                client.delete_objects(
                    Bucket=bucket,
                    Delete={"Objects": [{"Key": item["Key"]} for item in objects]},
                )
        client.delete_bucket(Bucket=bucket)
        service.close()
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture(scope="module")
def mlflow_service() -> Iterator[MLflowService]:
    service = MLflowService.start()
    try:
        yield service
    finally:
        service.close()


@pytest.fixture(scope="module")
def inference_assets() -> Iterator[_Assets]:
    shared_host = Path(
        os.environ.get(
            "TRIBUTO_RAY_SHARED_WORKSPACE_HOST",
            "/Users/jiangxintong/Docker/ray-cluster/workspace",
        )
    )
    shared_container = os.environ.get(
        "TRIBUTO_RAY_SHARED_WORKSPACE_CONTAINER", "/workspace"
    ).rstrip("/")
    if not shared_host.is_dir():
        raise RuntimeError(f"Ray shared host workspace does not exist: {shared_host}")

    with tempfile.TemporaryDirectory(
        prefix="tributo-inference-it-", dir=shared_host
    ) as raw_root:
        host_root = Path(raw_root)
        container_root = f"{shared_container}/{host_root.name}"
        input_dir = host_root / "input"
        input_dir.mkdir()
        pq.write_table(
            pa.table(
                {
                    "entity_id": pa.array([101, 102, 103, 104], type=pa.int64()),
                    "feature_a": pa.array([1.0, -1.0, 2.0, -2.0], type=pa.float32()),
                    "feature_b": pa.array([0.5, 0.5, -0.25, -0.25], type=pa.float32()),
                }
            ),
            input_dir / "part-0.parquet",
        )

        onnx_path = host_root / "model.onnx"
        _write_test_onnx(onnx_path)
        host_bundle = build_test_bundle(host_root, onnx_path=str(onnx_path))
        manifest_path = host_bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["canonical_uri"] = f"{container_root}/bundle"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        manifest_bytes = manifest_path.read_bytes()
        yield _Assets(
            host_root=host_root,
            container_root=container_root,
            host_bundle=host_bundle,
            container_bundle=f"{container_root}/bundle",
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            bundle_id=manifest["bundle_id"],
        )


def _input_binding() -> InputBindingSpec:
    return InputBindingSpec(
        tensors=(
            TensorInputBinding(
                tensor_name="float_input",
                columns=("feature_a", "feature_b"),
                dtype="float32",
            ),
        ),
        passthrough_columns=("entity_id",),
    )


def _classifier_output_binding() -> OutputBindingSpec:
    return OutputBindingSpec(
        tensors=(
            TensorOutputBinding(
                tensor_name="label",
                column="prediction",
                semantic="label",
                squeeze_singleton=True,
            ),
            TensorOutputBinding(
                tensor_name="probabilities",
                column="score",
                semantic="probability",
            ),
        )
    )


def _ray_input(
    source: ParquetSourceConfig, *, storage_profile: str | None = None
) -> IngestionRequest:
    return IngestionRequest(
        source=source,
        engine="ray",
        storage_profile=storage_profile,
    )


def _request(assets: _Assets, output_name: str) -> InferenceRequest:
    return InferenceRequest(
        model=BundleModelReference(
            uri=str(assets.host_bundle),
            expected_manifest_sha256=assets.manifest_sha256,
        ),
        input=_ray_input(
            ParquetSourceConfig(path=f"{assets.container_root}/input/part-0.parquet")
        ),
        input_binding=_input_binding(),
        output_binding=_classifier_output_binding(),
        result_sink=ParquetResultSinkRequest(
            uri=f"{assets.container_root}/{output_name}"
        ),
        execution=RayExecutionPolicy(batch_size=2, concurrency=2),
        run_id=f"inference-it-{output_name}",
    )


def test_standalone_request_runs_through_ray_jobs_and_parquet_sink(
    inference_assets: _Assets,
) -> None:
    job_id = submit_inference_request(
        _request(inference_assets, "standalone-output"),
        dashboard_url=DEFAULT_DASHBOARD_URL,
    )
    client = JobSubmissionClient(DEFAULT_DASHBOARD_URL)
    result = wait_for_job(client, job_id, timeout=240, poll_interval=2)

    assert result["status"] == JobStatus.SUCCEEDED, result["logs"]
    table = pads.dataset(
        inference_assets.host_root / "standalone-output", format="parquet"
    ).to_table()
    assert table.num_rows == 4
    assert table.column_names == ["entity_id", "prediction", "score"]
    assert table["entity_id"].to_pylist() == [101, 102, 103, 104]


def test_empty_input_completes_without_synthetic_count_or_rows(
    inference_assets: _Assets,
) -> None:
    input_dir = inference_assets.host_root / "empty-input"
    input_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "entity_id": pa.array([], type=pa.int64()),
                "feature_a": pa.array([], type=pa.float32()),
                "feature_b": pa.array([], type=pa.float32()),
            }
        ),
        input_dir / "part-0.parquet",
    )
    output_name = "empty-output"
    request = _request(inference_assets, output_name).model_copy(
        update={
            "input": _ray_input(
                ParquetSourceConfig(
                    path=f"{inference_assets.container_root}/empty-input/part-0.parquet"
                )
            ),
            "run_id": "inference-it-empty-input",
        }
    )

    job_id = submit_inference_request(request, dashboard_url=DEFAULT_DASHBOARD_URL)
    client = JobSubmissionClient(DEFAULT_DASHBOARD_URL)
    result = wait_for_job(client, job_id, timeout=240, poll_interval=2)

    assert result["status"] == JobStatus.SUCCEEDED, result["logs"]
    output = inference_assets.host_root / output_name
    parquet_files = tuple(output.rglob("*.parquet")) if output.exists() else ()
    if parquet_files:
        assert pads.dataset(output, format="parquet").to_table().num_rows == 0


def test_nan_failure_is_classified_as_materialization(
    inference_assets: _Assets,
) -> None:
    input_dir = inference_assets.host_root / "nan-input"
    input_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "entity_id": pa.array([201], type=pa.int64()),
                "feature_a": pa.array([np.nan], type=pa.float32()),
                "feature_b": pa.array([1.0], type=pa.float32()),
            }
        ),
        input_dir / "part-0.parquet",
    )
    request = _request(inference_assets, "nan-output").model_copy(
        update={
            "input": _ray_input(
                ParquetSourceConfig(
                    path=(f"{inference_assets.container_root}/nan-input/part-0.parquet")
                )
            ),
            "run_id": "inference-it-nan-materialization",
        }
    )

    job_id = submit_inference_request(request, dashboard_url=DEFAULT_DASHBOARD_URL)
    client = JobSubmissionClient(DEFAULT_DASHBOARD_URL)
    result = wait_for_job(client, job_id, timeout=240, poll_interval=2)

    assert result["status"] == JobStatus.FAILED
    assert "materialization" in result["logs"]


def test_published_bundle_triggers_inline_post_training_inference(
    inference_assets: _Assets,
) -> None:
    output_uri = f"{inference_assets.container_root}/post-training-output"
    parent_run_id = "training-run-it"
    runtime_env = build_runtime_env(
        env_vars={
            "INPUT_URI": f"{inference_assets.container_root}/input/part-0.parquet",
            "OUTPUT_URI": output_uri,
            "BUNDLE_URI": inference_assets.container_bundle,
            "BUNDLE_ID": inference_assets.bundle_id,
            "MANIFEST_SHA256": inference_assets.manifest_sha256,
            "PARENT_RUN_ID": parent_run_id,
        }
    )
    client = JobSubmissionClient(DEFAULT_DASHBOARD_URL)
    job_id = client.submit_job(
        entrypoint="python tests/integration/jobs/post_training_inference_job.py",
        runtime_env=runtime_env,
        submission_id=generate_submission_id(
            "infer-post-training", inference_assets.manifest_sha256
        ),
    )
    result = wait_for_job(client, job_id, timeout=240, poll_interval=2)

    assert result["status"] == JobStatus.SUCCEEDED, result["logs"]
    summary = _result_from_logs(result["logs"])
    assert summary == {
        "status": "succeeded",
        "parent_run_id": parent_run_id,
        "rows": 4,
        "columns": ["entity_id", "prediction", "score"],
    }


def test_published_bundle_triggers_detached_post_training_ray_job(
    inference_assets: _Assets,
) -> None:
    output_name = "post-training-detached-output"
    action = PostTrainingInferenceAction(
        input=_ray_input(
            ParquetSourceConfig(
                path=f"{inference_assets.container_root}/input/part-0.parquet"
            )
        ),
        input_binding=_input_binding(),
        output_binding=_classifier_output_binding(),
        result_sink=ParquetResultSinkRequest(
            uri=f"{inference_assets.container_root}/{output_name}"
        ),
        execution=RayExecutionPolicy(batch_size=2, concurrency=2),
        mode="detached",
    )
    job_id = submit_post_training_inference(
        action,
        BundleRef(
            canonical_uri=str(inference_assets.host_bundle),
            bundle_id=inference_assets.bundle_id,
            manifest_sha256=inference_assets.manifest_sha256,
        ),
        parent_run_id="training-run-detached-it",
        dashboard_url=DEFAULT_DASHBOARD_URL,
    )
    client = JobSubmissionClient(DEFAULT_DASHBOARD_URL)
    result = wait_for_job(client, job_id, timeout=240, poll_interval=2)

    assert result["status"] == JobStatus.SUCCEEDED, result["logs"]
    table = pads.dataset(
        inference_assets.host_root / output_name, format="parquet"
    ).to_table()
    assert table.num_rows == 4
    assert table.column_names == ["entity_id", "prediction", "score"]


def test_source_model_and_sink_use_independent_minio_profiles(
    inference_assets: _Assets,
    minio_assets: _S3Assets,
) -> None:
    request = InferenceRequest(
        model=BundleModelReference.from_bundle_ref(
            minio_assets.model_bundle,
            storage_profile="model_domain",
        ),
        input=_ray_input(
            ParquetSourceConfig(
                path=f"s3://{minio_assets.bucket}/input/part-0.parquet"
            ),
            storage_profile="source_domain",
        ),
        input_binding=_input_binding(),
        output_binding=_classifier_output_binding(),
        result_sink=ParquetResultSinkRequest(
            uri=f"s3://{minio_assets.bucket}/results/profile-isolation",
            storage_profile="sink_domain",
        ),
        execution=RayExecutionPolicy(batch_size=2, concurrency=2),
        run_id="inference-it-profile-isolation",
    )
    job_id = submit_inference_request(
        request,
        dashboard_url=DEFAULT_DASHBOARD_URL,
        env_vars=_cluster_storage_env(minio_assets),
    )
    client = JobSubmissionClient(DEFAULT_DASHBOARD_URL)
    result = wait_for_job(client, job_id, timeout=240, poll_interval=2)

    assert result["status"] == JobStatus.SUCCEEDED, result["logs"]
    assert MINIO_SECRET_ACCESS_KEY not in result["logs"]
    output_dir = inference_assets.host_root / "profile-isolation-download"
    _download_parquet_prefix(
        minio_assets,
        prefix="results/profile-isolation/",
        destination=output_dir,
    )
    table = pads.dataset(output_dir, format="parquet").to_table()
    assert table.num_rows == 4
    assert table.column_names == ["entity_id", "prediction", "score"]


def test_external_xgboost_artifact_is_normalized_before_ray_job(
    inference_assets: _Assets,
    minio_assets: _S3Assets,
) -> None:
    import xgboost

    artifact = inference_assets.host_root / "external-model.ubj"
    features = np.asarray(
        [[1.0, 0.5], [-1.0, 0.5], [2.0, -0.25], [-2.0, -0.25]],
        dtype=np.float32,
    )
    booster = xgboost.train(
        {
            "objective": "binary:logistic",
            "max_depth": 1,
            "eta": 1.0,
            "nthread": 1,
            "seed": 7,
        },
        xgboost.DMatrix(features, label=np.asarray([1, 0, 1, 0])),
        num_boost_round=3,
    )
    booster.save_model(artifact)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    request = InferenceRequest(
        model=ArtifactModelReference(
            provider_id="tributo.artifact",
            uri=str(artifact),
            format_id="xgboost",
            flavor_id="xgboost-native-v1",
            architecture_id="xgboost",
            expected_sha256=digest,
            import_bundle_uri=f"s3://{minio_assets.bucket}/models/external-xgboost",
            import_storage_profile="model_domain",
            options={
                "variant": "ubj",
                "input_fields": [
                    {
                        "name": "float_input",
                        "dtype": "float32",
                        "shape": ["batch", 2],
                    }
                ],
                "output_fields": [
                    {"name": "label", "dtype": "int64", "shape": ["batch"]},
                    {
                        "name": "probabilities",
                        "dtype": "float32",
                        "shape": ["batch", 2],
                    },
                ],
            },
        ),
        input=_ray_input(
            ParquetSourceConfig(
                path=f"{inference_assets.container_root}/input/part-0.parquet"
            )
        ),
        input_binding=_input_binding(),
        output_binding=_classifier_output_binding(),
        result_sink=ParquetResultSinkRequest(
            uri=f"{inference_assets.container_root}/external-xgboost-output"
        ),
        execution=RayExecutionPolicy(batch_size=2, concurrency=2),
        run_id="inference-it-external-xgboost",
    )

    job_id = submit_inference_request(
        request,
        dashboard_url=DEFAULT_DASHBOARD_URL,
        env_vars=_cluster_storage_env(minio_assets),
    )
    client = JobSubmissionClient(DEFAULT_DASHBOARD_URL)
    result = wait_for_job(client, job_id, timeout=240, poll_interval=2)

    assert result["status"] == JobStatus.SUCCEEDED, result["logs"]
    table = pads.dataset(
        inference_assets.host_root / "external-xgboost-output", format="parquet"
    ).to_table()
    assert table.num_rows == 4
    assert table.column_names == ["entity_id", "prediction", "score"]
    assert all(len(probabilities) == 2 for probabilities in table["score"].to_pylist())


def test_external_onnx_artifact_is_normalized_before_ray_job(
    inference_assets: _Assets,
    minio_assets: _S3Assets,
) -> None:
    artifact = inference_assets.host_root / "model.onnx"
    request = InferenceRequest(
        model=ArtifactModelReference(
            provider_id="tributo.artifact",
            uri=str(artifact),
            format_id="onnx",
            flavor_id="onnx-runtime-v1",
            expected_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            import_bundle_uri=f"s3://{minio_assets.bucket}/models/external-onnx",
            import_storage_profile="model_domain",
            options={
                "variant": "onnx",
                "input_fields": [
                    {
                        "name": "float_input",
                        "dtype": "float32",
                        "shape": ["batch", 2],
                    }
                ],
                "output_fields": [
                    {"name": "label", "dtype": "int64", "shape": ["batch", 1]},
                    {
                        "name": "probabilities",
                        "dtype": "float32",
                        "shape": ["batch", 2],
                    },
                ],
            },
        ),
        input=_ray_input(
            ParquetSourceConfig(
                path=f"{inference_assets.container_root}/input/part-0.parquet"
            )
        ),
        input_binding=_input_binding(),
        output_binding=_classifier_output_binding(),
        result_sink=ParquetResultSinkRequest(
            uri=f"{inference_assets.container_root}/external-onnx-output"
        ),
        execution=RayExecutionPolicy(batch_size=2, concurrency=2),
        run_id="inference-it-external-onnx",
    )

    job_id = submit_inference_request(
        request,
        dashboard_url=DEFAULT_DASHBOARD_URL,
        env_vars=_cluster_storage_env(minio_assets),
    )
    client = JobSubmissionClient(DEFAULT_DASHBOARD_URL)
    result = wait_for_job(client, job_id, timeout=240, poll_interval=2)

    assert result["status"] == JobStatus.SUCCEEDED, result["logs"]
    table = pads.dataset(
        inference_assets.host_root / "external-onnx-output", format="parquet"
    ).to_table()
    assert table.num_rows == 4
    assert table.column_names == ["entity_id", "prediction", "score"]


def test_mlflow_alias_is_frozen_before_ray_job_and_numeric_version_runs(
    inference_assets: _Assets,
    minio_assets: _S3Assets,
    mlflow_service: MLflowService,
) -> None:
    model_name, first_version, second_version, client = _register_mlflow_models(
        mlflow_service.tracking_uri
    )
    client.set_registered_model_alias(model_name, "champion", first_version)
    alias_request = _mlflow_request(
        inference_assets,
        minio_assets,
        tracking_uri=mlflow_service.tracking_uri,
        model_name=model_name,
        alias="champion",
        output_name="mlflow-alias-output",
    )

    frozen_plan = resolve_inference(alias_request)
    assert f"version={first_version}" in frozen_plan.model.source_provenance
    client.set_registered_model_alias(model_name, "champion", second_version)
    alias_job = submit_resolved_inference(
        frozen_plan,
        dashboard_url=DEFAULT_DASHBOARD_URL,
        env_vars=_cluster_storage_env(minio_assets),
    )
    ray_client = JobSubmissionClient(DEFAULT_DASHBOARD_URL)
    alias_result = wait_for_job(ray_client, alias_job, timeout=240, poll_interval=2)

    assert alias_result["status"] == JobStatus.SUCCEEDED, alias_result["logs"]
    alias_table = pads.dataset(
        inference_assets.host_root / "mlflow-alias-output", format="parquet"
    ).to_table()
    assert alias_table["entity_id"].to_pylist() == [101, 102, 103, 104]
    assert alias_table["score"].to_pylist()[0] == pytest.approx([2.0, 1.5])

    numeric_request = _mlflow_request(
        inference_assets,
        minio_assets,
        tracking_uri=mlflow_service.tracking_uri,
        model_name=model_name,
        version=second_version,
        output_name="mlflow-version-output",
    )
    numeric_job = submit_inference_request(
        numeric_request,
        dashboard_url=DEFAULT_DASHBOARD_URL,
        env_vars=_cluster_storage_env(minio_assets),
    )
    numeric_result = wait_for_job(ray_client, numeric_job, timeout=240, poll_interval=2)

    assert numeric_result["status"] == JobStatus.SUCCEEDED, numeric_result["logs"]
    numeric_table = pads.dataset(
        inference_assets.host_root / "mlflow-version-output", format="parquet"
    ).to_table()
    assert numeric_table["score"].to_pylist()[0] == pytest.approx([3.0, 2.5])


def _mlflow_request(
    assets: _Assets,
    s3: _S3Assets,
    *,
    tracking_uri: str,
    model_name: str,
    output_name: str,
    version: str | None = None,
    alias: str | None = None,
) -> InferenceRequest:
    return InferenceRequest(
        model=RegistryModelReference(
            provider_id="mlflow.v2",
            model_name=model_name,
            version=version,
            alias=alias,
            import_bundle_uri=f"s3://{s3.bucket}/models/mlflow",
            import_storage_profile="model_domain",
            options={"tracking_uri": tracking_uri},
        ),
        input=_ray_input(
            ParquetSourceConfig(path=f"{assets.container_root}/input/part-0.parquet")
        ),
        input_binding=_input_binding(),
        output_binding=OutputBindingSpec(
            tensors=(
                TensorOutputBinding(
                    tensor_name="score",
                    column="score",
                    semantic="score",
                ),
            )
        ),
        result_sink=ParquetResultSinkRequest(
            uri=f"{assets.container_root}/{output_name}"
        ),
        execution=RayExecutionPolicy(batch_size=2, concurrency=2),
        run_id=f"inference-it-{output_name}",
    )


def _register_mlflow_models(tracking_uri: str):
    import mlflow
    import onnx
    from mlflow import MlflowClient
    from mlflow.models import Model, ModelSignature
    from mlflow.types.schema import Schema, TensorSpec

    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = f"tributo-inference-{uuid.uuid4().hex}"
    mlflow.set_experiment(experiment_name)
    model_name = f"tributo-inference-{uuid.uuid4().hex}"
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    client.create_registered_model(model_name)
    signature = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype("float32"), (-1, 2), name="float_input")]),
        outputs=Schema([TensorSpec(np.dtype("float32"), (-1, 2), name="score")]),
    )
    versions: list[str] = []
    for offset in (1.0, 2.0):
        with tempfile.TemporaryDirectory(prefix="tributo-mlflow-model-") as raw:
            model_dir = Path(raw) / "model"
            model_dir.mkdir()
            onnx.save_model(
                _offset_onnx_model(offset),
                str(model_dir / "model.onnx"),
                save_as_external_data=False,
            )
            metadata = Model(signature=signature)
            metadata.add_flavor(
                "onnx",
                onnx_version=onnx.__version__,
                data="model.onnx",
            )
            metadata.save(str(model_dir / "MLmodel"))
            with mlflow.start_run() as run:
                mlflow.log_artifacts(str(model_dir), artifact_path="model")
                source = f"runs:/{run.info.run_id}/model"
                model_version = client.create_model_version(
                    model_name,
                    source,
                    run.info.run_id,
                )
        versions.append(
            _wait_model_version(client, model_name, str(model_version.version))
        )
    return model_name, versions[0], versions[1], client


def _wait_model_version(client, name: str, version: str) -> str:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        current = client.get_model_version(name, version)
        if str(current.status).upper() == "READY":
            return str(current.version)
        if str(current.status).upper() == "FAILED_REGISTRATION":
            raise RuntimeError(f"MLflow model version {name}/{version} failed")
        time.sleep(0.25)
    raise TimeoutError(f"MLflow model version {name}/{version} did not become READY")


def _offset_onnx_model(offset: float):
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Constant",
                [],
                ["offset"],
                value=helper.make_tensor("offset", TensorProto.FLOAT, [1], [offset]),
            ),
            helper.make_node("Add", ["float_input", "offset"], ["score"]),
        ],
        name=f"tributo-mlflow-offset-{offset}",
        inputs=[
            helper.make_tensor_value_info("float_input", TensorProto.FLOAT, [None, 2])
        ],
        outputs=[helper.make_tensor_value_info("score", TensorProto.FLOAT, [None, 2])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 11)],
        producer_name="tributo-mlflow-it",
    )
    onnx.checker.check_model(model)
    return model


def _storage_profile(endpoint: str) -> str:
    return json.dumps(
        {
            "endpoint": endpoint,
            "region": "us-east-1",
            "access_key_id": MINIO_ACCESS_KEY_ID,
            "secret_access_key": MINIO_SECRET_ACCESS_KEY,
            "use_ssl": False,
            "path_style": True,
        }
    )


def _profile_env(profile: str) -> str:
    return f"TRIBUTO_STORAGE_PROFILE_{profile.upper()}"


def _cluster_storage_env(assets: _S3Assets) -> dict[str, str]:
    return {
        _profile_env(profile): assets.cluster_profile
        for profile in ("source_domain", "model_domain", "sink_domain")
    }


def _download_parquet_prefix(
    assets: _S3Assets, *, prefix: str, destination: Path
) -> None:
    client = get_boto3_client(
        endpoint=assets.service.endpoint,
        access_key_id=MINIO_ACCESS_KEY_ID,
        secret_access_key=MINIO_SECRET_ACCESS_KEY,
        region="us-east-1",
        use_ssl=False,
        path_style=True,
    )
    destination.mkdir(parents=True, exist_ok=True)
    keys = [
        item["Key"]
        for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=assets.bucket, Prefix=prefix
        )
        for item in page.get("Contents", [])
        if item["Key"].endswith(".parquet")
    ]
    if not keys:
        raise AssertionError(
            f"No Parquet objects found under s3://{assets.bucket}/{prefix}"
        )
    for index, key in enumerate(keys):
        client.download_file(
            assets.bucket, key, str(destination / f"part-{index}.parquet")
        )


def _result_from_logs(logs: str) -> dict:
    for line in logs.splitlines():
        if line.startswith("RESULT: "):
            return json.loads(line.removeprefix("RESULT: "))
    raise AssertionError(f"RESULT line not found in job logs:\n{logs}")


def _write_test_onnx(path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "ReduceSum", ["float_input"], ["sum"], axes=[1], keepdims=1
            ),
            helper.make_node(
                "Constant",
                [],
                ["zero"],
                value=helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0]),
            ),
            helper.make_node("Greater", ["sum", "zero"], ["positive"]),
            helper.make_node("Cast", ["positive"], ["label"], to=TensorProto.INT64),
            helper.make_node("Identity", ["float_input"], ["probabilities"]),
        ],
        name="tributo-inference-it",
        inputs=[
            helper.make_tensor_value_info("float_input", TensorProto.FLOAT, [None, 2])
        ],
        outputs=[
            helper.make_tensor_value_info("label", TensorProto.INT64, [None, 1]),
            helper.make_tensor_value_info(
                "probabilities", TensorProto.FLOAT, [None, 2]
            ),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 11)],
        producer_name="tributo-inference-it",
    )
    onnx.checker.check_model(model)
    path.write_bytes(model.SerializeToString())
