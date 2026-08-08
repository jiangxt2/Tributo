"""Conformance harness for Bundle model flavors and their named runtime."""

from __future__ import annotations

import numpy as np

from tributo.exporting.runtime import (
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
    assert tuple(outputs) == model.output_names
    np.testing.assert_array_equal(outputs["score"], [[3.0]])


def test_fake_flavor_runs_full_conformance_suite() -> None:
    assert_flavor_conformance(_FakeFlavor)
