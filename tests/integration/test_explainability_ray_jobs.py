"""Docker Ray Jobs IT for the SHAP explainability adapter."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from ray.job_submission import JobStatus, JobSubmissionClient

from tributo._common.submission_id import generate_submission_id
from tributo.data import IngestionRequest
from tributo.data.source_config import ParquetSourceConfig
from tributo.explainability.contracts import (
    ExplainabilityConfig,
    ExplainabilityRequest,
    ReferenceBinding,
    ResultPolicy,
)
from tributo.exporting.bundle_reader import BundleReader
from tributo.exporting.manifest import (
    ExportManifestV2,
    ManifestExecution,
    ManifestSourceInfo,
)
from tributo.exporting.models import (
    ArtifactFile,
    CheckpointField,
    ExportCheckpointV1,
    LogicalArtifact,
    ProducerInfo,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def require_isolated_explainability_compose() -> None:
    if os.environ.get("TRIBUTO_DOCKER_EXPLAINABILITY_IT") != "1":
        pytest.fail(
            "Explainability IT must run through scripts/run_explainability_it.sh"
        )


def _assert_persisted_succeeded_operation(root: Path, operation_id: str) -> None:
    from tributo.integrations.storage.json_operation_store import JsonFileOperationStore

    records = JsonFileOperationStore(root / "operations").list_explainability(
        operation_id
    )
    assert records
    statuses = {record.status for record in records}
    assert "running" in statuses
    assert "succeeded" in statuses


def _read_persisted_receipt(root: Path, operation_id: str) -> tuple[dict, Path]:
    from urllib.parse import unquote, urlsplit

    from tributo.integrations.storage.json_operation_store import JsonFileOperationStore

    record = JsonFileOperationStore(root / "operations").get_explainability(
        operation_id
    )
    assert record is not None
    assert record.receipt_uri is not None
    parsed = urlsplit(record.receipt_uri)
    receipt_path = Path(
        unquote(parsed.path if parsed.scheme == "file" else record.receipt_uri)
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    result_path = Path(receipt["result_uri"])
    assert "attempts" in result_path.parts
    return receipt, result_path


def test_onnx_model_agnostic_shap_ray_job_writes_long_parquet_and_receipt(
    tmp_path: Path,
) -> None:
    """Run the real ONNX + SHAP approximate path in a Ray actor."""
    from tests.serving.bundle_fixtures import build_test_bundle, make_dummy_onnx

    shared_root = Path(
        os.environ.get("TRIBUTO_RAY_SHARED_WORKSPACE_CONTAINER", "/workspace")
    )
    root = shared_root / f"tributo-explainability-{uuid.uuid4().hex[:12]}"
    root.mkdir(parents=True)
    try:
        input_dir = root / "input"
        input_dir.mkdir()
        pq.write_table(
            pa.table(
                {
                    "entity_id": pa.array([101, 102, 103, 104], type=pa.int64()),
                    "feature_a": pa.array([0.0, 0.0, 1.0, 1.0], type=pa.float32()),
                    "feature_b": pa.array([0.0, 1.0, 0.0, 1.0], type=pa.float32()),
                }
            ),
            input_dir / "part-0.parquet",
        )
        reference_path = root / "reference.npy"
        np.save(reference_path, np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
        reference_digest = hashlib.sha256(reference_path.read_bytes()).hexdigest()

        onnx_path = Path(make_dummy_onnx(root))
        bundle = build_test_bundle(root, onnx_path=str(onnx_path))
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = ExplainabilityConfig(
            enabled=True,
            backend="model_agnostic",
            output_target="probability",
            allow_approximate=True,
            reference=ReferenceBinding(
                uri=str(reference_path), digest=reference_digest
            ),
        )
        manifest["schema_version"] = 2
        manifest["canonical_uri"] = str(bundle)
        manifest["explainability"] = config.to_descriptor(
            required_artifacts=("model",)
        ).model_dump(mode="json")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        result_dir = root / "explanations"
        request = ExplainabilityRequest(
            bundle_uri=str(bundle),
            input=IngestionRequest(
                source=ParquetSourceConfig(path=str(input_dir)),
                engine="ray",
            ),
            model_role="inference",
            feature_columns=("feature_a", "feature_b"),
            input_id_column="entity_id",
            backend="model_agnostic",
            output_target="probability",
            allow_approximate=True,
            reference=ReferenceBinding(
                uri=str(reference_path), digest=reference_digest
            ),
            result_uri=str(result_dir),
            operation_store_uri=str(root / "operations"),
            result_policy=ResultPolicy(
                access_scope="project",
                privacy_level="restricted",
                retention_seconds=3600,
            ),
            request_id=f"request-{uuid.uuid4().hex}",
        )
        config_path = root / "request.json"
        config_path.write_text(request.model_dump_json(), encoding="utf-8")

        client = JobSubmissionClient("http://ray-head:8265")
        submission_id = generate_submission_id("explain-it", request.request_id)
        job_id = client.submit_job(
            entrypoint=(
                f"python -m tributo.explainability.batch_job --config {config_path}"
            ),
            submission_id=submission_id,
        )
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            status = client.get_job_status(job_id)
            if status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED}:
                break
            time.sleep(2)
        assert client.get_job_status(job_id) == JobStatus.SUCCEEDED, (
            client.get_job_logs(job_id)
        )

        receipt, result_path = _read_persisted_receipt(root, request.request_id)
        assert receipt["status"] == "succeeded"
        assert receipt["backend"] == "model_agnostic"
        assert receipt["exactness"] == "approximate"
        assert receipt["output_target"] == "probability"
        assert receipt["reference_digest"] == reference_digest
        assert receipt["result_access_scope"] == "project"
        assert receipt["result_retention_seconds"] == 3600
        assert receipt["explanation_rows"] > 0
        result_files = sorted(result_path.glob("*.parquet"))
        assert result_files
        result_table = pa.concat_tables([pq.read_table(path) for path in result_files])
        assert {"input_id", "feature_id", "contribution", "model_digest"}.issubset(
            result_table.column_names
        )
        assert result_table.num_rows == receipt["explanation_rows"]
        assert "model_output" in result_table.column_names
        assert all(
            value is not None
            for value in result_table.column("model_output").to_pylist()
        )
        _assert_persisted_succeeded_operation(root, request.request_id)
    finally:
        import shutil

        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize("objective", ["binary:logistic", "multi:softprob"])
def test_xgboost_ubj_tree_shap_ray_job_writes_exact_attributions(
    objective: str,
) -> None:
    """Run the native XGBoost TreeSHAP path in a Ray actor."""
    import xgboost

    shared_root = Path(
        os.environ.get("TRIBUTO_RAY_SHARED_WORKSPACE_CONTAINER", "/workspace")
    )
    root = shared_root / f"tributo-explainability-tree-{uuid.uuid4().hex[:12]}"
    root.mkdir(parents=True)
    try:
        X = np.asarray(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
            dtype=np.float32,
        )
        y = np.asarray(
            [0, 1, 1, 1] if objective == "binary:logistic" else [0, 1, 2, 1],
            dtype=np.float32,
        )
        training_params = {
            "objective": objective,
            "max_depth": 2,
            "eta": 0.5,
        }
        class_count = 2
        if objective == "multi:softprob":
            class_count = 3
            training_params["num_class"] = class_count
        booster = xgboost.train(
            training_params,
            xgboost.DMatrix(X, label=y, feature_names=["feature_a", "feature_b"]),
            num_boost_round=4,
        )
        expected_margins = booster.predict(
            xgboost.DMatrix(X, feature_names=["feature_a", "feature_b"]),
            output_margin=True,
            strict_shape=True,
        )
        model_bytes = bytes(booster.save_raw(raw_format="ubj"))
        artifact_dir = root / "bundle" / "artifacts" / "native"
        artifact_dir.mkdir(parents=True)
        model_path = artifact_dir / "model.ubj"
        model_path.write_bytes(model_bytes)
        artifact_file = ArtifactFile(
            relative_path="model.ubj",
            sha256=hashlib.sha256(model_bytes).hexdigest(),
            size_bytes=len(model_bytes),
            role="model",
        )
        artifact = LogicalArtifact(
            name="native",
            format="ubj",
            flavor_id="xgboost-native-v1",
            files=(artifact_file,),
            entrypoint="model.ubj",
            tree_digest=LogicalArtifact.compute_tree_digest((artifact_file,)),
            producer=ProducerInfo(exporter_id="xgboost-ubj-v1"),
        )
        config = ExplainabilityConfig(
            enabled=True,
            backend="tree",
            model_role="explainability_model",
        )
        checkpoint_contract = ExportCheckpointV1(
            trainer_type="xgboost",
            architecture_id="xgboost",
            input_schema=(
                CheckpointField(
                    name="float_input",
                    dtype="float32",
                    shape=("batch", X.shape[1]),
                ),
            ),
            output_schema=(
                CheckpointField(name="label", dtype="int64", shape=("batch",)),
                CheckpointField(
                    name="probabilities",
                    dtype="float32",
                    shape=("batch", class_count),
                ),
            ),
            task_type="classification",
            framework="xgboost",
            framework_version=xgboost.__version__,
            preprocessing={"type": "none"},
            checkpoint_format_version=1,
        )
        input_signature, output_signature = checkpoint_contract.to_manifest_signatures()
        manifest = ExportManifestV2(
            bundle_id="bundle-tree-it",
            status="succeeded",
            canonical_uri=str(root / "bundle"),
            tributo_version="1.0.0",
            source_info=ManifestSourceInfo(
                source_kind="xgboost_result",
                framework=checkpoint_contract.framework,
                framework_version=checkpoint_contract.framework_version,
                architecture_id=checkpoint_contract.architecture_id,
                task_type=checkpoint_contract.task_type,
            ),
            input_signature=input_signature,
            output_signature=output_signature,
            artifacts=(artifact,),
            roles={"explainability_model": "native"},
            execution=ManifestExecution(execution_id="exec-tree-it"),
            explainability=config.to_descriptor(required_artifacts=("native",)),
        )
        (root / "bundle" / "manifest.json").write_bytes(manifest.canonical_json())

        input_dir = root / "input"
        input_dir.mkdir()
        pq.write_table(
            pa.table(
                {
                    "entity_id": pa.array([201, 202, 203, 204], type=pa.int64()),
                    "feature_a": pa.array(X[:, 0], type=pa.float32()),
                    "feature_b": pa.array(X[:, 1], type=pa.float32()),
                }
            ),
            input_dir / "part-0.parquet",
        )
        result_dir = root / "tree-explanations"
        output_selection = "predicted" if objective == "multi:softprob" else "all"
        request = ExplainabilityRequest(
            bundle_uri=str(root / "bundle"),
            input=IngestionRequest(
                source=ParquetSourceConfig(path=str(input_dir)),
                engine="ray",
            ),
            feature_columns=("feature_a", "feature_b"),
            input_id_column="entity_id",
            backend="tree",
            output_selection=output_selection,
            result_uri=str(result_dir),
            operation_store_uri=str(root / "operations"),
            request_id=f"tree-request-{uuid.uuid4().hex}",
        )
        config_path = root / "tree-request.json"
        config_path.write_text(request.model_dump_json(), encoding="utf-8")

        client = JobSubmissionClient("http://ray-head:8265")
        job_id = client.submit_job(
            entrypoint=(
                f"python -m tributo.explainability.batch_job --config {config_path}"
            ),
            submission_id=generate_submission_id("explain-tree-it", request.request_id),
        )
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            status = client.get_job_status(job_id)
            if status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED}:
                break
            time.sleep(2)
        assert client.get_job_status(job_id) == JobStatus.SUCCEEDED, (
            client.get_job_logs(job_id)
        )

        receipt, result_path = _read_persisted_receipt(root, request.request_id)
        assert receipt["status"] == "succeeded"
        assert receipt["backend"] == "tree"
        assert receipt["exactness"] == "exact"
        assert receipt["output_selection"] == output_selection
        assert receipt["explanation_rows"] == len(X) * X.shape[1]
        result_files = sorted(result_path.glob("*.parquet"))
        result_table = pa.concat_tables([pq.read_table(path) for path in result_files])
        assert result_table.num_rows == receipt["explanation_rows"]
        assert set(result_table.column("feature_id").to_pylist()) <= {
            "feature_a",
            "feature_b",
        }
        result_rows = result_table.to_pylist()
        expected_output_indexes = (
            np.argmax(expected_margins, axis=1)
            if objective == "multi:softprob"
            else np.zeros(len(X), dtype=np.int64)
        )
        for row_index, (input_id, output_index) in enumerate(
            zip((201, 202, 203, 204), expected_output_indexes, strict=True)
        ):
            selected = [row for row in result_rows if row["input_id"] == str(input_id)]
            assert len(selected) == X.shape[1]
            assert {row["output_id"] for row in selected} == {f"output_{output_index}"}
            assert len({row["base_value"] for row in selected}) == 1
            assert len({row["model_output"] for row in selected}) == 1
            reconstructed = (
                sum(row["contribution"] for row in selected) + selected[0]["base_value"]
            )
            assert reconstructed == pytest.approx(selected[0]["model_output"])
            assert selected[0]["model_output"] == pytest.approx(
                expected_margins[row_index, output_index]
            )
        _assert_persisted_succeeded_operation(root, request.request_id)
    finally:
        import shutil

        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize("source_kind", ["dnn_result", "pu_result"])
def test_dnn_and_pu_onnx_bundles_use_preprocessor_and_feature_map_in_ray_job(
    source_kind: str,
) -> None:
    """Exercise the real DNN/PU ONNX exporter sidecars and SHAP worker path."""
    import torch

    from tributo.explainability.contracts import (
        ExplainabilityConfig,
        ReferenceBinding,
    )
    from tributo.exporting.models import BundleOutputConfig, ExportSource, ExportTarget
    from tributo.exporting.service import BundleExportService

    shared_root = Path(
        os.environ.get("TRIBUTO_RAY_SHARED_WORKSPACE_CONTAINER", "/workspace")
    )
    root = shared_root / f"tributo-explainability-{source_kind}-{uuid.uuid4().hex[:12]}"
    root.mkdir(parents=True)
    try:

        class TwoInputModel(torch.nn.Module):
            def forward(
                self, feature_a: torch.Tensor, feature_b: torch.Tensor
            ) -> torch.Tensor:
                return torch.sigmoid(feature_a + 2.0 * feature_b)

        model = TwoInputModel().eval()
        reference_path = root / "reference.npy"
        np.save(
            reference_path,
            np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        )
        reference_digest = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        preprocessing_state = {
            "features": [
                {"name": "feature_a", "type": "dense", "dimension": 1, "norm": "none"},
                {"name": "feature_b", "type": "dense", "dimension": 1, "norm": "none"},
            ],
            "label_encoders": {},
            "norm_params": {},
        }
        source = ExportSource(
            source_kind=source_kind,
            model_object=model,
            architecture_id="dnn" if source_kind == "dnn_result" else "pu",
            model_config_data={
                "architecture_id": "dnn-family",
                "output": "probability",
            },
            feature_schema={"feature_names": ["feature_a", "feature_b"]},
            preprocessing_state=preprocessing_state,
            sample_inputs={
                "feature_a": torch.zeros(1),
                "feature_b": torch.zeros(1),
            },
            checkpoint_contract=ExportCheckpointV1(
                trainer_type="dnn" if source_kind == "dnn_result" else "pu",
                architecture_id="dnn-family",
                input_schema=(
                    CheckpointField(
                        name="feature_a", dtype="float32", shape=("batch",)
                    ),
                    CheckpointField(
                        name="feature_b", dtype="float32", shape=("batch",)
                    ),
                ),
                output_schema=(
                    CheckpointField(name="output", dtype="float32", shape=("batch",)),
                ),
                task_type="classification",
                framework="pytorch",
                framework_version=torch.__version__,
                preprocessing={"artifact": "preprocessor.json"},
                required_artifacts=(
                    "model.onnx",
                    "model_config.json",
                    "preprocessor.json",
                ),
            ),
        )
        config = BundleOutputConfig(
            bundle_uri=str(root / "bundle"),
            targets=[
                ExportTarget(name="model", format="onnx", exporter_id="torch-onnx-v1")
            ],
            roles={"inference": "model"},
            explainability=ExplainabilityConfig(
                enabled=True,
                backend="model_agnostic",
                allow_approximate=True,
                reference=ReferenceBinding(
                    uri=str(reference_path), digest=reference_digest, rows=2
                ),
            ),
        )
        result = BundleExportService().export_bundle(source=source, config=config)
        manifest = ExportManifestV2(
            **json.loads((Path(result.canonical_uri) / "manifest.json").read_text())
        )
        assert manifest.explainability is not None
        with BundleReader().open_artifact(
            result.canonical_uri, role="inference"
        ) as artifact:
            assert artifact.path_for("preprocessor.json").is_file()
            assert artifact.path_for("feature_map.json").is_file()

        input_dir = root / "input"
        input_dir.mkdir()
        pq.write_table(
            pa.table(
                {
                    "entity_id": pa.array([301, 302, 303, 304], type=pa.int64()),
                    "feature_a": pa.array([0.0, 0.0, 1.0, 1.0], type=pa.float32()),
                    "feature_b": pa.array([0.0, 1.0, 0.0, 1.0], type=pa.float32()),
                }
            ),
            input_dir / "part-0.parquet",
        )
        result_dir = root / "explanations"
        request = ExplainabilityRequest(
            bundle_uri=result.canonical_uri,
            input=IngestionRequest(
                source=ParquetSourceConfig(path=str(input_dir)), engine="ray"
            ),
            feature_columns=("feature_a", "feature_b"),
            input_id_column="entity_id",
            backend="model_agnostic",
            allow_approximate=True,
            reference=ReferenceBinding(
                uri=str(reference_path), digest=reference_digest, rows=2
            ),
            result_uri=str(result_dir),
            operation_store_uri=str(root / "operations"),
            request_id=f"{source_kind}-request-{uuid.uuid4().hex}",
        )
        config_path = root / "request.json"
        config_path.write_text(request.model_dump_json(), encoding="utf-8")
        client = JobSubmissionClient("http://ray-head:8265")
        job_id = client.submit_job(
            entrypoint=(
                f"python -m tributo.explainability.batch_job --config {config_path}"
            ),
            submission_id=generate_submission_id(source_kind, request.request_id),
        )
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            status = client.get_job_status(job_id)
            if status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED}:
                break
            time.sleep(2)
        assert client.get_job_status(job_id) == JobStatus.SUCCEEDED, (
            client.get_job_logs(job_id)
        )
        receipt, result_path = _read_persisted_receipt(root, request.request_id)
        assert receipt["status"] == "succeeded"
        assert receipt["backend"] == "model_agnostic"
        assert receipt["feature_map_digest"]
        assert receipt["reference_rows"] == 2
        table = pa.concat_tables(
            [pq.read_table(path) for path in result_path.glob("*.parquet")]
        )
        assert table.num_rows == receipt["explanation_rows"]
        assert set(table.column("feature_id").to_pylist()) == {"feature_a", "feature_b"}
        _assert_persisted_succeeded_operation(root, request.request_id)
    finally:
        import shutil

        shutil.rmtree(root, ignore_errors=True)
