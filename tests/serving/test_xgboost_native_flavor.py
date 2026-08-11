"""Unit tests for the native XGBoost Bundle flavor."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from tributo.exporting.models import (
    ArtifactFile,
    LogicalArtifact,
    ProducerInfo,
    ResolvedArtifact,
)
from tributo.integrations.flavors.xgboost_native import XGBoostNativeFlavor


class _Booster:
    objective = "binary:logistic"
    predictions = np.array([0.2, 0.8], dtype=np.float32)
    feature_names = ["a", "b"]

    def load_model(self, path: str) -> None:
        assert Path(path).is_file()

    def num_features(self) -> int:
        return 2

    def save_config(self) -> str:
        return json.dumps(
            {
                "learner": {
                    "objective": {"name": self.objective},
                    "learner_model_param": {"num_class": "0"},
                }
            }
        )

    def predict(self, matrix, output_margin=False):
        del matrix, output_margin
        return self.predictions


class _DMatrix:
    def __init__(self, values, feature_names=None) -> None:
        self.values = values
        self.feature_names = feature_names


def _artifact(tmp_path: Path, format_id: str = "ubj") -> ResolvedArtifact:
    extension = "ubj" if format_id == "ubj" else "json"
    path = tmp_path / f"model.{extension}"
    path.write_bytes(b"native-model")
    file = ArtifactFile(
        relative_path=f"model.{extension}",
        sha256="a" * 64,
        size_bytes=path.stat().st_size,
        role="model",
    )
    descriptor = LogicalArtifact(
        name="model",
        format=format_id,
        flavor_id="xgboost-native-v1",
        files=(file,),
        entrypoint=f"model.{extension}",
        tree_digest=LogicalArtifact.compute_tree_digest((file,)),
        producer=ProducerInfo(exporter_id="test"),
    )
    return ResolvedArtifact(descriptor, tmp_path)


def _fake_xgboost(monkeypatch) -> None:
    module = ModuleType("xgboost")
    module.Booster = _Booster
    module.DMatrix = _DMatrix
    monkeypatch.setitem(sys.modules, "xgboost", module)


@pytest.mark.parametrize("format_id", ("ubj", "xgboost-json"))
def test_binary_native_formats_expose_named_label_and_probability_outputs(
    tmp_path: Path, monkeypatch, format_id: str
) -> None:
    _fake_xgboost(monkeypatch)
    model = XGBoostNativeFlavor().load(_artifact(tmp_path, format_id), role="inference")

    outputs = model.predict(
        {"float_input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}
    )

    assert model.input_names == ("float_input",)
    assert model.output_names == ("label", "probabilities")
    np.testing.assert_array_equal(outputs["label"], [0, 1])
    np.testing.assert_allclose(outputs["probabilities"], [[0.8, 0.2], [0.2, 0.8]])


def test_regression_native_model_has_two_dimensional_prediction(
    tmp_path: Path, monkeypatch
) -> None:
    _Booster.objective = "reg:squarederror"
    _Booster.predictions = np.array([1.5, 2.5], dtype=np.float32)
    try:
        _fake_xgboost(monkeypatch)
        model = XGBoostNativeFlavor().load(_artifact(tmp_path), role="inference")
        output = model.predict(
            {"float_input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}
        )

        assert model.output_names == ("prediction",)
        assert output["prediction"].shape == (2, 1)
    finally:
        _Booster.objective = "binary:logistic"
        _Booster.predictions = np.array([0.2, 0.8], dtype=np.float32)
