"""Conformance harness for Bundle model flavors and their named runtime."""

from __future__ import annotations

import numpy as np
import pytest

from tributo.exceptions import ModelLoadError, UnsupportedArtifactFormat
from tributo.exporting.runtime import (
    FLAVOR_SUPPORT_MATRIX,
    SECURITY_MODE_SAFE,
    BundleModel,
    BundleModelFlavor,
)


class _FakeModel:
    input_names = ("x",)
    output_names = ("score",)
    input_dtypes = ("float32",)
    output_dtypes = ("float32",)
    input_shapes = ((None, 2),)
    output_shapes = ((None, 1),)

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {"score": inputs["x"].sum(axis=1, keepdims=True)}


class _FakeFlavor:
    api_version = 1
    flavor_id = "conformance-safe-v1"
    security_mode = SECURITY_MODE_SAFE
    signature_required = True
    required_dependencies: tuple[str, ...] = ()

    def load(self, artifact, *, role, unsafe=False, architecture_id=None):
        del artifact, role, unsafe, architecture_id
        return _FakeModel()


def assert_flavor_conformance(flavor_cls: type[BundleModelFlavor]) -> None:
    assert flavor_cls.api_version == 1
    assert flavor_cls.flavor_id
    assert flavor_cls.security_mode in {
        "safe",
        "pickle",
        "remote-code",
        "unknown-executable",
    }
    model = flavor_cls().load(object(), role="inference")
    assert isinstance(model, BundleModel)
    assert len(model.input_names) == len(model.input_dtypes) == len(model.input_shapes)
    assert (
        len(model.output_names) == len(model.output_dtypes) == len(model.output_shapes)
    )
    inputs = {"x": np.array([[1.0, 2.0]], dtype=np.float32)}
    outputs = model.predict(inputs)
    repeated = model.predict(inputs)
    assert tuple(outputs) == model.output_names
    assert tuple(repeated) == model.output_names
    np.testing.assert_array_equal(outputs["score"], [[3.0]])
    np.testing.assert_array_equal(repeated["score"], outputs["score"])


def test_fake_flavor_runs_full_conformance_suite() -> None:
    assert_flavor_conformance(_FakeFlavor)


def test_explicitly_unsupported_flavor_has_structured_failure() -> None:
    class _UnsupportedFlavor(_FakeFlavor):
        flavor_id = "unsupported-conformance-v1"

        def load(self, artifact, *, role, unsafe=False, architecture_id=None):
            del artifact, role, unsafe, architecture_id
            raise UnsupportedArtifactFormat("unsupported conformance flavor")

    with pytest.raises(UnsupportedArtifactFormat, match="unsupported"):
        _UnsupportedFlavor().load(object(), role="inference")


def test_flavor_load_failure_is_not_misclassified_as_unsupported() -> None:
    class _FailingFlavor(_FakeFlavor):
        flavor_id = "failing-conformance-v1"

        def load(self, artifact, *, role, unsafe=False, architecture_id=None):
            del artifact, role, unsafe, architecture_id
            raise ModelLoadError("classified model load failure")

    with pytest.raises(ModelLoadError, match="classified"):
        _FailingFlavor().load(object(), role="inference")


def test_first_party_flavor_metadata_reuses_capability_matrix() -> None:
    from tributo.integrations.flavors.onnx_runtime import ONNXRuntimeFlavor
    from tributo.integrations.flavors.xgboost_native import XGBoostNativeFlavor

    entries = {entry.flavor_id: entry for entry in FLAVOR_SUPPORT_MATRIX}
    for flavor in (ONNXRuntimeFlavor, XGBoostNativeFlavor):
        entry = entries[flavor.flavor_id]
        assert flavor.api_version == 1
        assert entry.loader == f"{flavor.__module__}:{flavor.__qualname__}"
        assert set(flavor.required_dependencies).issubset(entry.dependencies)
        assert entry.batch_inference_capable is True
        assert entry.online_serveable is True


def test_plugin_capabilities_are_the_execution_source_of_truth() -> None:
    from tributo.exporting.capabilities import get_default_capability_registry

    capabilities = get_default_capability_registry()
    onnx = capabilities.for_flavor("onnx-runtime-v1")
    native = capabilities.for_flavor("xgboost-native-v1")

    assert onnx.operations == ("prediction.batch", "prediction.online")
    assert onnx.conditional_operations == ()
    assert "attribution.tree-shap" not in onnx.operations
    assert native.operations == (
        "prediction.batch",
        "prediction.online",
    )
    assert native.conditional_operations == ("attribution.tree-shap",)
