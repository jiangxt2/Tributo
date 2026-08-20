"""Unit tests for explainability contracts and policy gates."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tributo.data import IngestionRequest
from tributo.data.source_config import ParquetSourceConfig
from tributo.explainability.contracts import (
    ExplainabilityConfig,
    ExplainabilityRequest,
    ReferenceBinding,
    ResultPolicy,
)
from tributo.explainability.planner import ExplainabilityPlanner
from tributo.exporting.assembler import BundleAssembler
from tributo.exporting.manifest import ManifestSourceInfo
from tributo.exporting.models import (
    ArtifactFile,
    ArtifactRef,
    BundleOutputConfig,
    ExportExecutionResult,
    ExportSource,
    ExportTarget,
    LogicalArtifact,
    NodeResult,
    ProducerInfo,
)
from tributo.exporting.service import BundleExportService


def _request(**updates: object) -> ExplainabilityRequest:
    values: dict[str, object] = {
        "bundle_uri": "/models/bundle",
        "input": IngestionRequest(
            source=ParquetSourceConfig(path="/data/input.parquet"),
            engine="ray",
        ),
        "feature_columns": ("feature_a", "feature_b"),
        "result_uri": "/data/explanations",
        "request_id": "request-1",
    }
    values.update(updates)
    return ExplainabilityRequest(**values)


def test_disabled_config_is_default_and_serializable() -> None:
    config = ExplainabilityConfig()
    assert config.enabled is False
    assert config.model_dump()["backend"] == "auto"


def test_request_model_role_is_resolved_from_bundle_descriptor_by_default() -> None:
    request = _request()
    assert request.model_role is None


def test_output_selection_defaults_to_all_and_is_serializable() -> None:
    request = _request()
    assert request.output_selection == "all"
    assert request.model_dump(mode="json")["output_selection"] == "all"

    predicted = _request(output_selection="predicted")
    assert predicted.output_selection == "predicted"

    with pytest.raises(ValidationError, match="output_selection"):
        _request(output_selection="unsupported")


def test_operation_store_uri_is_local_and_credential_free() -> None:
    request = _request(operation_store_uri="file:///tmp/tributo-operations")
    assert request.operation_store_uri == "file:///tmp/tributo-operations"
    with pytest.raises(ValidationError, match="operation_store_uri"):
        _request(operation_store_uri="s3://bucket/operations")


def test_force_resume_is_explicit_and_serializable() -> None:
    request = _request(force_resume=True)
    assert request.force_resume is True
    assert request.model_dump(mode="json")["force_resume"] is True


def test_model_agnostic_requires_explicit_approximation() -> None:
    with pytest.raises(ValidationError, match="allow_approximate"):
        ExplainabilityConfig(enabled=True, backend="model_agnostic")
    with pytest.raises(ValidationError, match="allow_approximate"):
        _request(backend="model_agnostic")


def test_reference_binding_rejects_credentials_and_query() -> None:
    with pytest.raises(ValidationError, match="credentials"):
        ReferenceBinding(uri="s3://user:secret@bucket/reference.npy")
    with pytest.raises(ValidationError, match="query"):
        ReferenceBinding(uri="s3://bucket/reference.npy?signature=secret")


def test_request_rejects_top_k_larger_than_feature_limit() -> None:
    with pytest.raises(ValidationError, match="top_k"):
        _request(limits={"top_k": 4, "max_features": 2})


def test_tree_log_loss_requires_reference_and_label_column() -> None:
    reference = ReferenceBinding(uri="/data/reference.npy")
    with pytest.raises(ValidationError, match="label_column"):
        ExplainabilityConfig(
            enabled=True,
            backend="tree",
            output_target="log_loss",
            reference=reference,
        )
    with pytest.raises(ValidationError, match="label_column"):
        _request(
            backend="tree",
            output_target="log_loss",
            reference=reference,
        )


def test_auto_tree_outputs_require_reference_before_backend_selection() -> None:
    with pytest.raises(ValidationError, match="reference binding"):
        ExplainabilityConfig(
            enabled=True,
            backend="auto",
            output_target="probability",
        )


def test_label_column_is_not_an_explanation_feature() -> None:
    with pytest.raises(ValidationError, match="feature_columns"):
        _request(label_column="feature_a")


def test_tree_bundle_config_adds_a_required_xgboost_companion_target() -> None:
    config = BundleOutputConfig(
        bundle_uri="/models/bundle",
        targets=[ExportTarget(name="onnx-model", format="onnx")],
        roles={"inference": "onnx-model"},
        explainability=ExplainabilityConfig(enabled=True, backend="tree"),
    )
    assert [target.name for target in config.targets or []] == [
        "onnx-model",
        "explainability-model",
    ]
    assert config.roles["explainability_model"] == "explainability-model"


def test_tree_bundle_config_reuses_existing_native_target_as_companion() -> None:
    config = BundleOutputConfig(
        bundle_uri="/models/bundle",
        targets=[
            ExportTarget(name="onnx-model", format="onnx"),
            ExportTarget(name="native", format="ubj"),
        ],
        roles={"inference": "onnx-model"},
        explainability=ExplainabilityConfig(enabled=True, backend="tree"),
    )
    assert config.roles["explainability_model"] == "native"
    assert [target.name for target in config.targets or []] == [
        "onnx-model",
        "native",
    ]


def test_auto_bundle_config_does_not_inject_native_target_without_algorithm_context() -> (
    None
):
    config = BundleOutputConfig(
        bundle_uri="/models/bundle",
        targets=[ExportTarget(name="onnx-model", format="onnx")],
        roles={"inference": "onnx-model"},
        explainability=ExplainabilityConfig(
            enabled=True,
            backend="auto",
            allow_approximate=True,
        ),
    )
    assert [target.name for target in config.targets or []] == ["onnx-model"]


def test_xgboost_auto_bundle_config_adds_ubj_companion_at_service_boundary() -> None:
    config = BundleOutputConfig(
        bundle_uri="/models/bundle",
        targets=[ExportTarget(name="onnx-model", format="onnx")],
        roles={"inference": "onnx-model"},
        explainability=ExplainabilityConfig(
            enabled=True,
            backend="auto",
            allow_approximate=True,
        ),
    )
    prepared = BundleExportService._prepare_explainability_config(
        config, ExportSource(source_kind="xgboost_result")
    )
    assert [target.format for target in prepared.targets or []] == ["onnx", "ubj"]
    assert prepared.roles["explainability_model"] == "explainability-model"


def test_descriptor_can_bind_the_actual_companion_model_role() -> None:
    config = ExplainabilityConfig(enabled=True, backend="tree")
    descriptor = config.to_descriptor(
        model_roles=("explainability_model",),
        required_artifacts=("native",),
    )
    assert descriptor is not None
    assert descriptor.model_roles == ("explainability_model",)


def test_assembler_writes_companion_role_into_v2_descriptor(tmp_path) -> None:
    artifact_file = ArtifactFile(
        relative_path="model.ubj",
        sha256="a" * 64,
        size_bytes=1,
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
    execution = ExportExecutionResult(
        execution_id="execution-1",
        status="succeeded",
        node_results=(
            NodeResult(
                node_id="native",
                target_name="native",
                status="succeeded",
                required=True,
                publish=True,
                artifact_ref=ArtifactRef(
                    node_id="native",
                    artifact_name="native",
                    tree_digest=artifact.tree_digest,
                ),
            ),
        ),
        staged_artifacts={"native": artifact},
    )
    staged = BundleAssembler().assemble(
        execution=execution,
        staging_root=tmp_path,
        bundle_uri=str(tmp_path / "bundles"),
        bundle_id="bundle-1",
        execution_id="execution-1",
        tributo_version="1.0.0",
        source_info=ManifestSourceInfo(source_kind="xgboost_result"),
        roles={"inference": "native", "explainability_model": "native"},
        explainability=ExplainabilityConfig(enabled=True, backend="tree"),
    )
    assert staged.manifest.schema_version == 2
    assert staged.manifest.explainability is not None
    assert staged.manifest.explainability.model_roles == ("explainability_model",)


def test_legacy_output_cannot_enable_explainability() -> None:
    from tributo.training.xgboost_trainer import OutputConfig

    with pytest.raises(ValidationError, match="requires output.bundle_uri"):
        OutputConfig(
            onnx_path="/models/model.onnx",
            explainability={"enabled": True, "backend": "tree"},
        )


def test_result_policy_is_explicit_and_serializable() -> None:
    request = _request(
        result_policy=ResultPolicy(
            access_scope="project",
            privacy_level="sensitive",
            allow_sensitive_features=True,
            retention_seconds=3600,
        )
    )
    assert request.result_policy.retention_seconds == 3600
    assert request.model_dump(mode="json")["result_policy"]["access_scope"] == "project"


def test_planner_preflights_known_explanation_byte_budget() -> None:
    request = _request(limits={"max_explanation_bytes": 100})
    with pytest.raises(ValueError, match="estimated explanation output"):
        ExplainabilityPlanner.preflight_limits(
            request,
            input_rows=10,
            output_count=2,
        )


def test_planner_uses_explicit_multiclass_output_bound_and_selection() -> None:
    all_outputs = ExplainabilityPlanner.preflight_limits(
        _request(),
        input_rows=10,
        output_count=10,
    )
    predicted = ExplainabilityPlanner.preflight_limits(
        _request(output_selection="predicted"),
        input_rows=10,
        output_count=10,
    )

    assert all_outputs["estimated_output_count"] == 10
    assert all_outputs["estimated_output_rows"] == 200
    assert predicted["estimated_output_count"] == 1
    assert predicted["estimated_output_rows"] == 20


def test_planner_rejects_non_positive_output_count() -> None:
    with pytest.raises(ValueError, match="output_count"):
        ExplainabilityPlanner.preflight_limits(
            _request(),
            input_rows=10,
            output_count=0,
        )
