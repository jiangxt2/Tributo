"""Unit tests for ONNX explainability input binding."""

from __future__ import annotations

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
    _build_onnx_inputs,
    _LeaseHeartbeat,
    _load_reference,
    _manifest_role_digest,
    _operation_store_for_request,
    _resolve_xgboost_feature_names,
    _schema_signature,
    _select_onnx_output,
    _selected_model_role,
    _validate_request_against_descriptor,
    run_batch_explainability,
)
from tributo.exporting.bundle_reader import BundleReader
from tributo.exporting.models import ArtifactFile, LogicalArtifact, ProducerInfo
from tributo.exporting.records import InMemoryOperationStore
from tributo.integrations.storage.json_operation_store import JsonFileOperationStore
from tributo.training.features.column_types import DenseFeat, NormMethod
from tributo.training.features.transformer import FeatureTransformer


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
    manifest = SimpleNamespace(bundle_id="bundle-attempt-isolation")
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
        BundleReader,
        "read_manifest_with_bytes",
        lambda self, *args, **kwargs: (manifest, b"manifest"),
    )
    monkeypatch.setattr(executor_module, "_bundle_digest", lambda _manifest: "a" * 64)
    monkeypatch.setattr(
        executor_module,
        "_validate_request_against_descriptor",
        lambda _manifest, _request: None,
    )
    monkeypatch.setattr(
        executor_module,
        "_selected_backend",
        lambda _manifest, _request: ("tree", "exact"),
    )
    monkeypatch.setattr(
        executor_module,
        "_safe_reference_digest",
        lambda _request, _provider: None,
    )
    monkeypatch.setattr(
        executor_module,
        "_result_stats",
        lambda _uri: ("b" * 64, 10, 1),
    )

    def fake_sink_write(self, dataset, sink_request, *, run_id, plan_digest):
        del self, dataset, run_id, plan_digest
        captured["sink_uri"] = sink_request.uri

    monkeypatch.setattr(executor_module.ParquetResultSink, "write", fake_sink_write)

    def fake_make_receipt(**kwargs):
        captured["receipt_payload_result_uri"] = kwargs["result_uri"]
        return SimpleNamespace(input_rows=1, explanation_rows=1)

    monkeypatch.setattr(executor_module, "_make_receipt", fake_make_receipt)
    monkeypatch.setattr(
        executor_module,
        "_write_receipt",
        lambda uri, receipt: captured.__setitem__("receipt_write_uri", uri),
    )

    store = InMemoryOperationStore()
    run_batch_explainability(
        request,
        input_resolver=Resolver(),
        operation_store=store,
    )

    record = store.get_explainability(request.operation_id)
    assert record is not None
    assert captured["sink_uri"] == record.result_uri
    assert captured["receipt_payload_result_uri"] == record.result_uri
    assert captured["receipt_write_uri"] == record.result_uri
    assert record.receipt_uri == record.result_uri + "/receipt.json"
    assert record.result_uri.startswith("/data/explanations/attempts/")
    assert record.result_uri != request.result_uri


def test_xgboost_feature_order_is_checked_without_sidecar() -> None:
    class FakeBooster:
        feature_names = ["feature_a", "feature_b"]

        @staticmethod
        def num_features() -> int:
            return 2

    class ResolvedArtifact:
        @staticmethod
        def path_for(name: str) -> Path:
            assert name == "feature_names.json"
            return Path("/does/not/exist")

    request = ExplainabilityRequest(
        bundle_uri="/models/bundle",
        input=IngestionRequest(
            source=ParquetSourceConfig(path="/data/input.parquet"), engine="ray"
        ),
        feature_columns=("feature_b", "feature_a"),
        result_uri="/data/explanations",
        request_id="request-feature-order",
    )
    with pytest.raises(ValueError, match="feature order"):
        _resolve_xgboost_feature_names(ResolvedArtifact(), FakeBooster(), request)


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
