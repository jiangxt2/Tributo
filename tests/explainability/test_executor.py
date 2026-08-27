"""Unit tests for ONNX explainability input binding."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tributo.data import IngestionRequest
from tributo.data.source_config import ParquetSourceConfig
from tributo.explainability import executor as executor_module
from tributo.explainability.contracts import (
    ExplainabilityDescriptor,
    ExplainabilityRequest,
    ReferenceBinding,
)
from tributo.explainability.executor import (
    _attempt_result_uri,
    _LeaseHeartbeat,
    _load_reference,
    _make_receipt,
    _operation_idempotency_key,
    _operation_store_for_request,
    _schema_signature,
    run_batch_explainability,
)
from tributo.explainability.protocols import ExplainabilityModelBinding
from tributo.exporting.models import ArtifactFile, LogicalArtifact, ProducerInfo
from tributo.exporting.records import InMemoryOperationStore
from tributo.integrations.model_runtimes.explainability import (
    _output_count_upper_bound as _explanation_output_count_upper_bound,
)
from tributo.integrations.model_runtimes.explainability import (
    build_onnx_inputs as _build_onnx_inputs,
)
from tributo.integrations.model_runtimes.explainability import (
    manifest_role_digest as _manifest_role_digest,
)
from tributo.integrations.model_runtimes.explainability import (
    select_onnx_output as _select_onnx_output,
)
from tributo.integrations.model_runtimes.explainability import (
    selected_model_role as _selected_model_role,
)
from tributo.integrations.model_runtimes.explainability import (
    validate_explainability_request as _validate_request_against_descriptor,
)
from tributo.integrations.storage.json_operation_store import JsonFileOperationStore
from tributo.training.features.column_types import DenseFeat, NormMethod
from tributo.training.features.transformer import FeatureTransformer


class _UnusedSessionFactory:
    factory_id = "test.explainability-session-v1"

    def create(self, reference_provider):
        raise AssertionError(reference_provider)


def test_explainability_worker_does_not_import_concrete_model_loaders() -> None:
    path = Path(__file__).parents[2] / "src/tributo/explainability/executor.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "tributo.exporting.runtime" not in imports
    assert "tributo.data.persistence" not in imports
    assert "tributo.integrations.sinks.parquet" not in imports
    assert "manifest_bytes" not in source
    assert "xgboost.Booster" not in source
    assert ".load_model(" not in source


def test_build_onnx_inputs_applies_dnn_preprocessor_and_named_inputs() -> None:
    transformer = FeatureTransformer(
        [
            DenseFeat("age", norm=NormMethod.STANDARD),
            DenseFeat("income", norm=NormMethod.MINMAX),
        ]
    )
    transformer.fit(
        {
            "age": np.asarray([10.0, 20.0], dtype=np.float32),
            "income": np.asarray([100.0, 200.0], dtype=np.float32),
        }
    )

    inputs = _build_onnx_inputs(
        np.asarray([[20.0, 150.0], [10.0, 200.0]], dtype=np.float32),
        input_names=("age", "income"),
        input_dtypes=(np.dtype("float32"), np.dtype("float32")),
        input_shapes=((None,), (None,)),
        feature_names=("age", "income"),
        preprocessor=transformer,
        feature_view="raw",
    )

    np.testing.assert_allclose(inputs["age"], [1.0, -1.0])
    np.testing.assert_allclose(inputs["income"], [0.5, 1.0])


def test_build_onnx_inputs_binds_generic_multi_input_by_declared_names() -> None:
    inputs = _build_onnx_inputs(
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        input_names=("left", "right"),
        input_dtypes=(np.dtype("float32"), np.dtype("float32")),
        input_shapes=((None,), (None,)),
        feature_names=("right", "left"),
        preprocessor=None,
        feature_view="model_input",
    )

    np.testing.assert_allclose(inputs["left"], [2.0, 4.0])
    np.testing.assert_allclose(inputs["right"], [1.0, 3.0])


def test_manifest_role_digest_reads_file_level_preprocessor_role() -> None:
    preprocessor = ArtifactFile(
        relative_path="preprocessor.json",
        sha256="a" * 64,
        size_bytes=1,
        role="preprocessor",
    )
    model = ArtifactFile(
        relative_path="model.onnx",
        sha256="b" * 64,
        size_bytes=1,
        role="model",
    )
    artifact = LogicalArtifact(
        name="model",
        format="onnx",
        flavor_id="onnx-runtime-v1",
        files=(model, preprocessor),
        entrypoint="model.onnx",
        tree_digest=LogicalArtifact.compute_tree_digest((model, preprocessor)),
        producer=ProducerInfo(exporter_id="test"),
    )

    class _Manifest:
        roles: dict[str, str] = {}
        artifacts = (artifact,)

    assert _manifest_role_digest(_Manifest(), "preprocessor") == "a" * 64


def test_load_reference_supports_npy_and_verifies_digest(tmp_path) -> None:
    reference_path = tmp_path / "reference.npy"
    np.save(reference_path, np.asarray([[1.0], [2.0]], dtype=np.float32))
    digest = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    request = ExplainabilityRequest(
        bundle_uri="/models/bundle",
        input=IngestionRequest(
            source=ParquetSourceConfig(path="/data/input.parquet"),
            engine="ray",
        ),
        backend="model_agnostic",
        allow_approximate=True,
        reference=ReferenceBinding(uri=str(reference_path), digest=digest),
        result_uri="/data/explanations",
        request_id="request-reference",
    )

    np.testing.assert_array_equal(_load_reference(request), [[1.0], [2.0]])


def test_tree_descriptor_resolves_explainability_role_when_request_omits_role() -> None:
    request = ExplainabilityRequest(
        bundle_uri="/models/bundle",
        input=IngestionRequest(
            source=ParquetSourceConfig(path="/data/input.parquet"), engine="ray"
        ),
        backend="tree",
        result_uri="/data/explanations",
        request_id="request-tree-default-role",
    )
    manifest = SimpleNamespace(
        roles={"inference": "onnx", "explainability_model": "native"},
        explainability=ExplainabilityDescriptor(
            adapter_id="shap-v1",
            backend="tree",
            exactness="exact",
            model_roles=("explainability_model",),
            feature_view="raw",
            output_target="model_output",
            reference_policy="optional",
        ),
    )
    assert _selected_model_role(manifest, request) == "explainability_model"
    _validate_request_against_descriptor(manifest, request)


@pytest.mark.parametrize(
    ("task_type", "field_name", "shape", "expected"),
    [
        ("classification", "probabilities", ("batch", 10), 10),
        ("classification", "probabilities", ("batch", 2), 2),
        ("regression", "prediction", ("batch", 1), 1),
    ],
)
def test_native_output_bound_comes_from_typed_manifest_signature(
    task_type: str,
    field_name: str,
    shape: tuple[str | int, ...],
    expected: int,
) -> None:
    request = ExplainabilityRequest(
        bundle_uri="/models/bundle",
        input=IngestionRequest(
            source=ParquetSourceConfig(path="/data/input.parquet"), engine="ray"
        ),
        backend="tree",
        result_uri="/data/explanations",
        request_id="request-output-bound",
    )
    artifact = SimpleNamespace(name="native", flavor_id="external-tree-v1")
    manifest = SimpleNamespace(
        roles={"explainability_model": "native"},
        artifacts=(artifact,),
        source_info=SimpleNamespace(task_type=task_type),
        output_signature=SimpleNamespace(
            output_fields=(SimpleNamespace(name=field_name, shape=shape),)
        ),
    )

    assert (
        _explanation_output_count_upper_bound(manifest, request, native=True)
        == expected
    )


def test_native_output_bound_requires_a_fixed_typed_signature() -> None:
    request = ExplainabilityRequest(
        bundle_uri="/models/bundle",
        input=IngestionRequest(
            source=ParquetSourceConfig(path="/data/input.parquet"), engine="ray"
        ),
        backend="tree",
        result_uri="/data/explanations",
        request_id="request-missing-output-bound",
    )
    manifest = SimpleNamespace(
        roles={"explainability_model": "native"},
        artifacts=(SimpleNamespace(name="native", flavor_id="external-tree-v1"),),
        source_info=SimpleNamespace(task_type="classification"),
        output_signature=SimpleNamespace(output_fields=()),
    )

    with pytest.raises(ValueError, match="typed probability or prediction"):
        _explanation_output_count_upper_bound(manifest, request, native=True)


def test_descriptorless_bundle_is_rejected_before_worker_loading() -> None:
    request = ExplainabilityRequest(
        bundle_uri="/models/bundle",
        input=IngestionRequest(
            source=ParquetSourceConfig(path="/data/input.parquet"), engine="ray"
        ),
        result_uri="/data/explanations",
        request_id="request-descriptorless",
    )
    with pytest.raises(
        ValueError, match="does not declare an explainability descriptor"
    ):
        _validate_request_against_descriptor(SimpleNamespace(roles={}), request)


def test_request_operation_store_uri_is_consumed_by_executor_boundary(tmp_path) -> None:
    request = ExplainabilityRequest(
        bundle_uri="/models/bundle",
        input=IngestionRequest(
            source=ParquetSourceConfig(path="/data/input.parquet"), engine="ray"
        ),
        operation_store_uri=str(tmp_path / "operations"),
        result_uri="/data/explanations",
        request_id="request-store-uri",
    )
    store = _operation_store_for_request(request)
    assert isinstance(store, JsonFileOperationStore)


@pytest.mark.parametrize(
    ("base_uri", "expected_prefix"),
    [
        ("/data/explanations", "/data/explanations/attempts/"),
        ("file:///data/explanations", "file:///data/explanations/attempts/"),
        ("s3://bucket/explanations", "s3://bucket/explanations/attempts/"),
    ],
)
def test_attempt_result_uri_isolated_by_lease_token(
    base_uri: str, expected_prefix: str
) -> None:
    first = _attempt_result_uri(base_uri, "lease-1")
    second = _attempt_result_uri(base_uri, "lease-2")

    assert first == expected_prefix + "lease-1"
    assert second == expected_prefix + "lease-2"
    assert first != second


def test_lease_renewal_failure_message_includes_tuning_hint() -> None:
    heartbeat = _LeaseHeartbeat(
        InMemoryOperationStore(),
        operation_id="operation-lease-message",
        idempotency_key="a" * 64,
        lease_token="lease-token",
        lease_seconds=17,
    )
    heartbeat._error = ValueError("lease expired")

    with pytest.raises(RuntimeError, match="operation_lease_seconds"):
        heartbeat.raise_if_failed()


def test_executor_writes_every_attempt_to_its_isolated_result_uri(monkeypatch) -> None:
    request = ExplainabilityRequest(
        bundle_uri="/models/bundle",
        input=IngestionRequest(
            source=ParquetSourceConfig(path="/data/input.parquet"), engine="ray"
        ),
        result_uri="/data/explanations",
        operation_id="operation-attempt-isolation",
        request_id="request-attempt-isolation",
    )
    descriptor = ExplainabilityDescriptor(
        adapter_id="shap-v1",
        backend="tree",
        exactness="exact",
        model_roles=("explainability_model",),
        feature_view="raw",
        output_target="model_output",
        reference_policy="optional",
    )
    model_binding = ExplainabilityModelBinding(
        bundle_id="bundle-attempt-isolation",
        bundle_digest="a" * 64,
        manifest_sha256="c" * 64,
        model_role="explainability_model",
        model_digest="d" * 64,
        preprocessor_digest=None,
        feature_map_digest=None,
        descriptor=descriptor,
        backend="tree",
        exactness="exact",
        output_count_upper_bound=1,
        session_factory=_UnusedSessionFactory(),
    )
    opened = SimpleNamespace(
        dataset=SimpleNamespace(
            count=lambda: 1,
            map_batches=lambda *args, **kwargs: args[0],
        ),
        close=lambda: None,
    )
    captured: dict[str, str] = {}

    class Resolver:
        def describe(self, _request):
            return object()

        def open(self, _selection):
            return opened

    monkeypatch.setattr(
        executor_module,
        "_safe_reference_digest",
        lambda _request, _provider: None,
    )

    class ResultStore:
        provider_id = "test-results-v1"

        def materialize(
            self,
            dataset,
            *,
            uri,
            storage_profile,
            max_bytes,
            run_id,
            plan_digest,
        ):
            del dataset, storage_profile, max_bytes, run_id, plan_digest
            captured["sink_uri"] = uri
            return SimpleNamespace(digest="b" * 64, total_bytes=10, rows=1)

        def write_receipt(self, uri, receipt, *, storage_profile):
            del receipt, storage_profile
            captured["receipt_write_uri"] = uri

        def read_receipt(self, uri, *, storage_profile):
            del uri, storage_profile
            return None

        def cleanup(self, uri, *, storage_profile):
            del uri, storage_profile

    class Models:
        provider_id = "test-models-v1"

        def resolve(self, _request):
            return model_binding

    def fake_make_receipt(**kwargs):
        captured["receipt_payload_result_uri"] = kwargs["result_uri"]
        return SimpleNamespace(input_rows=1, explanation_rows=1)

    monkeypatch.setattr(executor_module, "_make_receipt", fake_make_receipt)
    store = InMemoryOperationStore()
    run_batch_explainability(
        request,
        input_resolver=Resolver(),
        operation_store=store,
        model_provider=Models(),
        result_store=ResultStore(),
    )

    record = store.get_explainability(request.operation_id)
    assert record is not None
    assert captured["sink_uri"] == record.result_uri
    assert captured["receipt_payload_result_uri"] == record.result_uri
    assert captured["receipt_write_uri"] == record.result_uri
    assert record.receipt_uri == record.result_uri + "/receipt.json"
    assert record.result_uri.startswith("/data/explanations/attempts/")
    assert record.result_uri != request.result_uri


def test_receipt_and_idempotency_record_output_selection() -> None:
    request = ExplainabilityRequest(
        bundle_uri="/models/bundle",
        input=IngestionRequest(
            source=ParquetSourceConfig(path="/data/input.parquet"), engine="ray"
        ),
        backend="tree",
        output_selection="predicted",
        result_uri="/data/explanations",
        request_id="request-output-selection",
    )
    model_binding = ExplainabilityModelBinding(
        bundle_id="bundle-output-selection",
        bundle_digest="a" * 64,
        manifest_sha256="d" * 64,
        model_role="explainability_model",
        model_digest="b" * 64,
        preprocessor_digest=None,
        feature_map_digest=None,
        descriptor=ExplainabilityDescriptor(
            adapter_id="shap-v1",
            backend="tree",
            exactness="exact",
            model_roles=("explainability_model",),
            feature_view="raw",
            output_target="model_output",
            reference_policy="optional",
        ),
        backend="tree",
        exactness="exact",
        output_count_upper_bound=1,
        session_factory=_UnusedSessionFactory(),
    )
    receipt = _make_receipt(
        model_binding=model_binding,
        request=request,
        operation_id="operation-output-selection",
        bundle_digest="a" * 64,
        selected_backend="tree",
        exactness="exact",
        input_rows=1,
        explanation_rows=2,
        result_digest="c" * 64,
        result_bytes=128,
        result_uri="/data/explanations/attempts/lease",
        status="succeeded",
        reference_provider=SimpleNamespace(),
    )
    assert receipt.output_selection == "predicted"

    all_key = _operation_idempotency_key(
        request.model_copy(update={"output_selection": "all"}),
        bundle_digest="a" * 64,
        reference_provider=SimpleNamespace(),
    )
    predicted_key = _operation_idempotency_key(
        request,
        bundle_digest="a" * 64,
        reference_provider=SimpleNamespace(),
    )
    assert all_key != predicted_key


def test_schema_signature_is_derived_from_attribution_contract() -> None:
    signature = _schema_signature()
    assert len(signature) == 64


def test_select_onnx_output_requires_explicit_semantics_for_multi_output() -> None:
    outputs = {
        "label": np.asarray([0, 1]),
        "probabilities": np.asarray([[0.8, 0.2], [0.1, 0.9]]),
    }
    selected = _select_onnx_output(
        outputs,
        ("label", "probabilities"),
        output_target="probability",
    )
    np.testing.assert_allclose(selected, outputs["probabilities"])

    with pytest.raises(ValueError, match="ambiguous"):
        _select_onnx_output(
            outputs,
            ("label", "probabilities"),
            output_target="model_output",
        )
