"""Tests for Bundle-backed explainability model providers."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from tests.serving.bundle_fixtures import build_test_bundle, make_dummy_onnx
from tributo.data import IngestionRequest
from tributo.data.source_config import ParquetSourceConfig
from tributo.explainability.contracts import ExplainabilityRequest, ReferenceBinding
from tributo.explainability.protocols import ExplainabilityModelSessionFactory
from tributo.explainability.reference import FileReferenceProvider
from tributo.integrations.model_runtimes.explainability import (
    BundleExplainabilityModelProvider,
)


def test_onnx_provider_builds_prediction_capability_without_format_branching(
    tmp_path: Path,
) -> None:
    onnx_path = make_dummy_onnx(tmp_path)
    bundle = build_test_bundle(
        tmp_path,
        onnx_path=onnx_path,
        input_field_shape=("batch", 2),
        output_field_shapes={
            "label": ("batch",),
            "probabilities": ("batch", 2),
        },
    )
    reference_path = tmp_path / "reference.npy"
    reference = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    np.save(reference_path, reference)
    reference_digest = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    request = ExplainabilityRequest(
        bundle_uri=str(bundle),
        input=IngestionRequest(
            source=ParquetSourceConfig(path="/data/input.parquet"),
            engine="ray",
        ),
        feature_columns=("feature_a", "feature_b"),
        backend="model_agnostic",
        output_target="probabilities",
        allow_approximate=True,
        reference=ReferenceBinding(uri=str(reference_path), digest=reference_digest),
        result_uri=str(tmp_path / "results"),
        request_id="model-provider-test",
    )
    manifest_path = bundle / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["schema_version"] = 2
    manifest_payload["explainability"] = {
        "adapter_id": "shap-v1",
        "backend": "model_agnostic",
        "exactness": "approximate",
        "model_roles": ["inference"],
        "required_artifacts": ["model"],
        "feature_view": "raw",
        "output_target": "probabilities",
        "reference_policy": "required",
        "reference_digest": reference_digest,
    }
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    provider = BundleExplainabilityModelProvider()
    binding = provider.resolve(request)
    assert binding.bundle_id == "bundle-e3-test"
    assert binding.backend == "model_agnostic"
    assert binding.output_count_upper_bound == 2

    factory = pickle.loads(pickle.dumps(binding.session_factory))
    assert isinstance(factory, ExplainabilityModelSessionFactory)
    session = factory.create(FileReferenceProvider())
    try:
        context = session.context
        assert context.native_attribution_id is None
        assert context.predict is not None
        assert session.output_count_upper_bound == 2
        probabilities = context.predict(reference)
        assert probabilities.shape == (2, 2)
    finally:
        session.close()
