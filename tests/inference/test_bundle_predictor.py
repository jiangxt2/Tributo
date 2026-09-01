"""Tests for named table/tensor binding in the Bundle Ray actor."""

from __future__ import annotations

import numpy as np
import pytest

from tributo.exporting.models import BundleRef
from tributo.inference.bundle_predictor import BundleBatchPredictor
from tributo.inference.contracts import (
    InputBindingSpec,
    OutputBindingSpec,
    ResolvedModelSelection,
    TensorInputBinding,
    TensorOutputBinding,
)


class _Runtime:
    def __init__(self) -> None:
        self.inputs: list[dict[str, np.ndarray]] = []
        self.close_calls = 0

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self.inputs.append(inputs)
        rows = next(iter(inputs.values())).shape[0]
        return {
            "label": np.arange(rows, dtype=np.int64),
            "probabilities": np.tile(
                np.array([[0.25, 0.75]], dtype=np.float32), (rows, 1)
            ),
        }

    def close(self) -> None:
        self.close_calls += 1


class _Factory:
    factory_id = "test.prediction-v1"

    def __init__(self, runtime: _Runtime) -> None:
        self.runtime = runtime

    def create(self) -> _Runtime:
        return self.runtime


class _Provider:
    provider_id = "test.model-runtime-v1"

    def __init__(self, runtime: _Runtime) -> None:
        self.runtime = runtime
        self.models: list[object] = []

    def prediction_factory(self, model: object) -> _Factory:
        self.models.append(model)
        return _Factory(self.runtime)


def _selection() -> ResolvedModelSelection:
    return ResolvedModelSelection(
        bundle_ref=BundleRef(
            canonical_uri="/models/bundle",
            bundle_id="bundle-1",
            manifest_sha256="a" * 64,
        ),
        role="inference",
        flavor_id="onnx-runtime-v1",
        storage_profile="model-domain",
        source_provenance="tributo-bundle",
    )


def _bindings(
    *, preserve_features: bool = False, nan_policy: str = "error"
) -> tuple[InputBindingSpec, OutputBindingSpec]:
    return (
        InputBindingSpec(
            tensors=(
                TensorInputBinding(
                    tensor_name="float_input",
                    columns=("feature_b", "feature_a"),
                    dtype="float32",
                ),
            ),
            passthrough_columns=("entity_id",),
            nan_policy=nan_policy,
        ),
        OutputBindingSpec(
            tensors=(
                TensorOutputBinding(
                    tensor_name="probabilities",
                    column="score",
                    semantic="probability",
                ),
                TensorOutputBinding(
                    tensor_name="label",
                    column="prediction",
                    semantic="label",
                ),
            ),
            preserve_features=preserve_features,
        ),
    )


class TestBundleBatchPredictor:
    def _predictor(
        self,
        runtime: _Runtime,
        *,
        preserve_features: bool = False,
        nan_policy: str = "error",
    ) -> BundleBatchPredictor:
        inputs, outputs = _bindings(
            preserve_features=preserve_features, nan_policy=nan_policy
        )
        provider = _Provider(runtime)
        predictor = BundleBatchPredictor(
            _selection(),
            inputs,
            outputs,
            kernel_provider=provider,
        )
        assert provider.models == [_selection()]
        return predictor

    def test_named_input_order_and_named_output_selection(self) -> None:
        runtime = _Runtime()
        predictor = self._predictor(runtime)

        result = predictor(
            {
                "feature_a": np.array([1.0, 2.0]),
                "feature_b": np.array([10.0, 20.0]),
                "entity_id": np.array([101, 102]),
            }
        )

        np.testing.assert_array_equal(
            runtime.inputs[0]["float_input"],
            np.array([[10.0, 1.0], [20.0, 2.0]], dtype=np.float32),
        )
        assert list(result) == ["entity_id", "score", "prediction"]
        np.testing.assert_array_equal(result["prediction"], [0, 1])
        np.testing.assert_allclose(result["score"], [[0.25, 0.75], [0.25, 0.75]])

    def test_preserve_features_is_explicit(self) -> None:
        runtime = _Runtime()
        predictor = self._predictor(runtime, preserve_features=True)

        result = predictor(
            {
                "feature_a": np.array([1.0]),
                "feature_b": np.array([2.0]),
                "entity_id": np.array([7]),
            }
        )

        assert list(result)[:3] == ["entity_id", "feature_b", "feature_a"]

    def test_missing_column_fails_before_runtime_call(self) -> None:
        runtime = _Runtime()
        predictor = self._predictor(runtime)

        with pytest.raises(KeyError, match="feature_b"):
            predictor(
                {
                    "feature_a": np.array([1.0]),
                    "entity_id": np.array([7]),
                }
            )

        assert runtime.inputs == []

    def test_null_policy_is_fail_closed(self) -> None:
        runtime = _Runtime()
        predictor = self._predictor(runtime)

        with pytest.raises(ValueError, match="NULL values"):
            predictor(
                {
                    "feature_a": np.array([1.0], dtype=object),
                    "feature_b": np.array([None], dtype=object),
                    "entity_id": np.array([7]),
                }
            )

    def test_masked_null_is_rejected_before_numpy_drops_mask(self) -> None:
        runtime = _Runtime()
        predictor = self._predictor(runtime)

        with pytest.raises(ValueError, match="NULL values"):
            predictor(
                {
                    "feature_a": np.array([1.0]),
                    "feature_b": np.ma.array([2.0], mask=[True]),
                    "entity_id": np.array([7]),
                }
            )

        assert runtime.inputs == []

    def test_nan_is_rejected_by_default(self) -> None:
        runtime = _Runtime()
        predictor = self._predictor(runtime)

        with pytest.raises(ValueError, match="nan_policy='error'"):
            predictor(
                {
                    "feature_a": np.array([np.nan]),
                    "feature_b": np.array([2.0]),
                    "entity_id": np.array([7]),
                }
            )

        assert runtime.inputs == []

    def test_nan_can_be_explicitly_allowed_for_model_missing_semantics(self) -> None:
        runtime = _Runtime()
        predictor = self._predictor(runtime, nan_policy="allow")

        predictor(
            {
                "feature_a": np.array([np.nan]),
                "feature_b": np.array([2.0]),
                "entity_id": np.array([7]),
            }
        )

        assert np.isnan(runtime.inputs[0]["float_input"][0, 1])

    @pytest.mark.parametrize("value", [np.nan, np.inf])
    def test_non_finite_values_cannot_be_silently_cast_to_integer(
        self, value: float
    ) -> None:
        runtime = _Runtime()
        inputs = InputBindingSpec(
            tensors=(
                TensorInputBinding(
                    tensor_name="float_input",
                    columns=("feature",),
                    dtype="int64",
                ),
            ),
            nan_policy="allow",
        )
        outputs = OutputBindingSpec(
            tensors=(
                TensorOutputBinding(
                    tensor_name="label", column="label", semantic="label"
                ),
            )
        )
        predictor = BundleBatchPredictor(
            _selection(), inputs, outputs, kernel_provider=_Provider(runtime)
        )

        with pytest.raises(ValueError, match="cannot be safely converted"):
            predictor({"feature": np.array([value])})

        assert runtime.inputs == []

    @pytest.mark.parametrize(
        "entity_id, expected_message",
        [
            (np.array([None], dtype=object), "NULL values"),
            (np.array([np.nan]), "nan_policy='error'"),
        ],
    )
    def test_passthrough_columns_apply_input_value_policies(
        self, entity_id: np.ndarray, expected_message: str
    ) -> None:
        runtime = _Runtime()
        predictor = self._predictor(runtime)

        with pytest.raises(ValueError, match=expected_message):
            predictor(
                {
                    "feature_a": np.array([1.0]),
                    "feature_b": np.array([2.0]),
                    "entity_id": entity_id,
                }
            )

        assert runtime.inputs == []

    def test_output_collision_is_rejected(self) -> None:
        runtime = _Runtime()
        inputs, _ = _bindings()
        outputs = OutputBindingSpec(
            tensors=(
                TensorOutputBinding(
                    tensor_name="label", column="entity_id", semantic="label"
                ),
            )
        )
        predictor = BundleBatchPredictor(
            _selection(), inputs, outputs, kernel_provider=_Provider(runtime)
        )

        with pytest.raises(ValueError, match="collides"):
            predictor(
                {
                    "feature_a": np.array([1.0]),
                    "feature_b": np.array([2.0]),
                    "entity_id": np.array([7]),
                }
            )

    def test_close_is_idempotently_delegated(self) -> None:
        runtime = _Runtime()
        predictor = self._predictor(runtime)

        predictor.close()
        predictor.close()

        assert runtime.close_calls == 1


def test_single_vector_column_is_not_restacked() -> None:
    runtime = _Runtime()
    inputs = InputBindingSpec(
        tensors=(
            TensorInputBinding(
                tensor_name="float_input", columns=("embedding",), dtype="float32"
            ),
        )
    )
    outputs = OutputBindingSpec(
        tensors=(
            TensorOutputBinding(tensor_name="label", column="label", semantic="label"),
        )
    )
    predictor = BundleBatchPredictor(
        _selection(), inputs, outputs, kernel_provider=_Provider(runtime)
    )

    predictor({"embedding": np.array([[1.0, 2.0], [3.0, 4.0]])})

    assert runtime.inputs[0]["float_input"].shape == (2, 2)


def test_single_scalar_column_preserves_batch_rank() -> None:
    runtime = _Runtime()
    inputs = InputBindingSpec(
        tensors=(
            TensorInputBinding(
                tensor_name="float_input",
                columns=("feature",),
                dtype="float32",
                single_column_mode="scalar",
            ),
        )
    )
    outputs = OutputBindingSpec(
        tensors=(
            TensorOutputBinding(tensor_name="label", column="label", semantic="label"),
        )
    )
    predictor = BundleBatchPredictor(
        _selection(), inputs, outputs, kernel_provider=_Provider(runtime)
    )

    predictor({"feature": np.array([1.0, 2.0])})

    assert runtime.inputs[0]["float_input"].shape == (2,)


def test_single_scalar_column_rejects_vector_valued_rows() -> None:
    runtime = _Runtime()
    inputs = InputBindingSpec(
        tensors=(
            TensorInputBinding(
                tensor_name="float_input",
                columns=("feature",),
                dtype="float32",
                single_column_mode="scalar",
            ),
        )
    )
    outputs = OutputBindingSpec(
        tensors=(
            TensorOutputBinding(tensor_name="label", column="label", semantic="label"),
        )
    )
    predictor = BundleBatchPredictor(
        _selection(), inputs, outputs, kernel_provider=_Provider(runtime)
    )

    with pytest.raises(ValueError, match="must be a one-dimensional batch column"):
        predictor({"feature": np.array([[1.0], [2.0]])})


def test_plugin_kernel_cannot_change_batch_row_count_without_passthrough() -> None:
    class WrongRowRuntime(_Runtime):
        def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            self.inputs.append(inputs)
            return {"label": np.array([1], dtype=np.int64)}

    runtime = WrongRowRuntime()
    inputs = InputBindingSpec(
        tensors=(
            TensorInputBinding(
                tensor_name="float_input",
                columns=("feature",),
                dtype="float32",
            ),
        )
    )
    outputs = OutputBindingSpec(
        tensors=(
            TensorOutputBinding(
                tensor_name="label",
                column="prediction",
                semantic="label",
            ),
        )
    )
    predictor = BundleBatchPredictor(
        _selection(), inputs, outputs, kernel_provider=_Provider(runtime)
    )

    with pytest.raises(ValueError, match="changed batch row count from 2 to 1"):
        predictor({"feature": np.array([1.0, 2.0])})


def test_plugin_kernel_cannot_return_scalar_output() -> None:
    class ScalarRuntime(_Runtime):
        def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            self.inputs.append(inputs)
            return {"label": np.asarray(1, dtype=np.int64)}

    runtime = ScalarRuntime()
    inputs = InputBindingSpec(
        tensors=(TensorInputBinding(tensor_name="float_input", columns=("feature",)),)
    )
    outputs = OutputBindingSpec(
        tensors=(
            TensorOutputBinding(
                tensor_name="label",
                column="prediction",
                semantic="label",
            ),
        )
    )
    predictor = BundleBatchPredictor(
        _selection(), inputs, outputs, kernel_provider=_Provider(runtime)
    )

    with pytest.raises(ValueError, match="must preserve the batch dimension"):
        predictor({"feature": np.array([1.0])})
