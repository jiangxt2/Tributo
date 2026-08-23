"""Unit tests for the dependency-neutral SHAP adapter boundary."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tributo.explainability import shap as shap_module
from tributo.explainability.protocols import ExplainableModelContext, PreparedExplainer
from tributo.explainability.shap import ShapAdapter, _NativeTreeExplainer

from .test_contracts import _request


class _Explanation:
    values = np.asarray([[[0.1], [1.0]], [[0.1], [2.0]]])
    data = np.asarray([[10.0, 20.0], [30.0, 40.0]])
    base_values = np.asarray([0.5, 0.6])


class _DynamicExplanation:
    values = np.asarray([[[0.1], [1.0]], [[0.1], [2.0]]])
    data = np.asarray([[10.0, 20.0], [30.0, 40.0]])

    def base_values(self, label: float) -> float:
        return 0.5 if label == 0 else 0.6


class _ArrayDynamicExplanation:
    values = np.asarray([[[0.1], [1.0]], [[0.1], [2.0]]])
    data = np.asarray([[10.0, 20.0], [30.0, 40.0]])

    def __init__(self) -> None:
        self.base_values = np.asarray(
            [self._base_value, self._base_value], dtype=object
        )

    @staticmethod
    def _base_value(label: float) -> float:
        return 0.5 if label == 0 else 0.6


class _MultiOutputExplanation:
    values = np.asarray(
        [
            [[10.0, 0.1], [0.2, 5.0]],
            [[4.0, 0.1], [0.2, 3.0]],
        ]
    )
    data = np.asarray([[10.0, 20.0], [30.0, 40.0]])
    base_values = np.asarray([[-10.0, 0.0], [0.0, -3.0]])
    model_outputs = values.sum(axis=1) + base_values


def test_shap_long_rows_use_top_k_and_preserve_provenance() -> None:
    request = _request(limits={"top_k": 1})
    prepared = PreparedExplainer(
        backend="tree",
        exactness="exact",
        explain=lambda batch, **_: _Explanation(),
        feature_names=("feature_a", "feature_b"),
        predict=lambda batch: np.asarray([1.6, 2.7]),
    )
    rows = ShapAdapter().explain_batch(
        prepared,
        np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        input_ids=("row-1", "row-2"),
        model_digest="a" * 64,
        request=request,
    )
    assert len(rows) == 2
    assert {row.feature_name for row in rows} == {"feature_b"}
    assert all(row.model_digest == "a" * 64 for row in rows)
    assert [row.model_output for row in rows] == [1.6, 2.7]
    assert all(row.feature_value is None for row in rows)


def test_predicted_selection_preserves_class_id_and_ranks_selected_output() -> None:
    request = _request(output_selection="predicted", limits={"top_k": 1})
    prepared = PreparedExplainer(
        backend="tree",
        exactness="exact",
        explain=lambda batch, **_: _MultiOutputExplanation(),
        feature_names=("feature_a", "feature_b"),
    )

    rows = ShapAdapter().explain_batch(
        prepared,
        np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        input_ids=("row-1", "row-2"),
        model_digest="a" * 64,
        request=request,
    )

    assert [(row.feature_name, row.output_id) for row in rows] == [
        ("feature_b", "output_1"),
        ("feature_a", "output_0"),
    ]


def test_binary_predicted_selection_is_the_single_output() -> None:
    request = _request(output_selection="predicted")
    prepared = PreparedExplainer(
        backend="tree",
        exactness="exact",
        explain=lambda batch, **_: _Explanation(),
        feature_names=("feature_a", "feature_b"),
        predict=lambda batch: np.asarray([1.6, 2.7]),
    )

    rows = ShapAdapter().explain_batch(
        prepared,
        np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        input_ids=("row-1", "row-2"),
        model_digest="a" * 64,
        request=request,
    )

    assert len(rows) == 4
    assert {row.output_id for row in rows} == {"output_0"}


def test_tree_log_loss_requires_labels_at_adapter_boundary() -> None:
    request = _request(
        backend="tree",
        output_target="log_loss",
        label_column="label",
        reference={"uri": "/data/reference.npy"},
    )
    prepared = PreparedExplainer(
        backend="tree",
        exactness="exact",
        explain=lambda batch, **kwargs: _Explanation(),
        feature_names=("feature_a", "feature_b"),
    )
    with pytest.raises(ValueError, match="requires labels"):
        ShapAdapter().explain_batch(
            prepared,
            np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
            input_ids=("row-1", "row-2"),
            model_digest="a" * 64,
            request=request,
        )


def test_tree_log_loss_materialises_dynamic_base_values() -> None:
    request = _request(
        backend="tree",
        output_target="log_loss",
        label_column="label",
        reference={"uri": "/data/reference.npy"},
    )
    prepared = PreparedExplainer(
        backend="tree",
        exactness="exact",
        explain=lambda batch, **kwargs: _DynamicExplanation(),
        feature_names=("feature_a", "feature_b"),
    )

    rows = ShapAdapter().explain_batch(
        prepared,
        np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        input_ids=("row-1", "row-2"),
        model_digest="a" * 64,
        request=request,
        labels=np.asarray([0, 1]),
    )

    assert [row.base_value for row in rows[::2]] == [0.5, 0.6]
    assert [row.model_output for row in rows[::2]] == [1.6, 2.7]


def test_tree_log_loss_materialises_array_dynamic_base_values() -> None:
    request = _request(
        backend="tree",
        output_target="log_loss",
        label_column="label",
        reference={"uri": "/data/reference.npy"},
    )
    prepared = PreparedExplainer(
        backend="tree",
        exactness="exact",
        explain=lambda batch, **kwargs: _ArrayDynamicExplanation(),
        feature_names=("feature_a", "feature_b"),
    )

    rows = ShapAdapter().explain_batch(
        prepared,
        np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        input_ids=("row-1", "row-2"),
        model_digest="a" * 64,
        request=request,
        labels=np.asarray([0, 1]),
    )

    assert [row.base_value for row in rows[::2]] == [0.5, 0.6]
    assert [row.model_output for row in rows[::2]] == [1.6, 2.7]


def test_sensitive_feature_values_are_opt_in() -> None:
    request = _request(result_policy={"allow_sensitive_features": True})
    prepared = PreparedExplainer(
        backend="tree",
        exactness="exact",
        explain=lambda batch, **_: _Explanation(),
        feature_names=("feature_a", "feature_b"),
    )
    rows = ShapAdapter().explain_batch(
        prepared,
        np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        input_ids=("row-1", "row-2"),
        model_digest="a" * 64,
        request=request,
    )
    assert rows[0].feature_value == 10.0


def test_summary_preserves_v1_rows_and_filters_exactness() -> None:
    request = _request()
    prepared_exact = PreparedExplainer(
        backend="tree",
        exactness="exact",
        explain=lambda batch, **_: _Explanation(),
        feature_names=("feature_a", "feature_b"),
    )
    exact_rows = ShapAdapter().explain_batch(
        prepared_exact,
        np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        input_ids=("row-1", "row-2"),
        model_digest="a" * 64,
        request=request,
    )
    approximate_rows = tuple(
        row.model_copy(update={"exactness": "approximate"}) for row in exact_rows
    )
    adapter = ShapAdapter()
    assert adapter.summarize(exact_rows) == exact_rows
    assert (
        adapter.summarize(exact_rows + approximate_rows, exactness="exact")
        == exact_rows
    )
    with pytest.raises(ValueError, match="exactness"):
        adapter.summarize(exact_rows, exactness="unknown")


def test_tree_support_rejects_unsupported_objective_before_optional_import() -> None:
    context = ExplainableModelContext(
        bundle_uri="/models/bundle",
        model_role="inference",
        artifact_name="native",
        artifact_format="ubj",
        flavor_id="xgboost-native-v1",
        native_attribution_id="xgboost-tree-shap-v1",
        artifact_path=None,
        objective="rank:pairwise",
    )
    decision = ShapAdapter.supports(context, _request(backend="tree"))
    assert decision.supported is False
    assert "Unsupported XGBoost objective" in decision.reason


def test_tree_support_rejects_unknown_output_target() -> None:
    context = ExplainableModelContext(
        bundle_uri="/models/bundle",
        model_role="inference",
        artifact_name="native",
        artifact_format="ubj",
        flavor_id="xgboost-native-v1",
        native_attribution_id="xgboost-tree-shap-v1",
        artifact_path=None,
        objective="binary:logistic",
    )
    decision = ShapAdapter.supports(
        context,
        _request(backend="tree", output_target="unknown_target"),
    )
    assert decision.supported is False
    assert "output_target" in decision.reason


def test_tree_support_rejects_predicted_selection_outside_native_classification() -> (
    None
):
    regression = ExplainableModelContext(
        bundle_uri="/models/bundle",
        model_role="inference",
        artifact_name="native",
        artifact_format="ubj",
        flavor_id="xgboost-native-v1",
        native_attribution_id="xgboost-tree-shap-v1",
        artifact_path=None,
        objective="reg:squarederror",
    )
    regression_decision = ShapAdapter.supports(
        regression,
        _request(backend="tree", output_selection="predicted"),
    )
    assert regression_decision.supported is False
    assert "classification objective" in regression_decision.reason

    classification = replace(regression, objective="binary:logistic")
    probability_decision = ShapAdapter.supports(
        classification,
        _request(
            backend="tree",
            output_target="probability",
            output_selection="predicted",
            reference={"uri": "/reference.npy"},
        ),
    )
    assert probability_decision.supported is False
    assert "native XGBoost raw" in probability_decision.reason


def test_native_prepare_does_not_load_shap(monkeypatch) -> None:
    pytest.importorskip("xgboost")

    class FakeBooster:
        feature_names = ["feature_a", "feature_b"]

        @staticmethod
        def save_config():
            return '{"learner":{"gradient_booster":{"name":"gbtree"}}}'

    context = ExplainableModelContext(
        bundle_uri="/models/bundle",
        model_role="inference",
        artifact_name="native",
        artifact_format="ubj",
        flavor_id="xgboost-native-v1",
        native_attribution_id="xgboost-tree-shap-v1",
        artifact_path=None,
        model_object=FakeBooster(),
        feature_names=("feature_a", "feature_b"),
        objective="binary:logistic",
    )

    def fail_if_loaded():
        raise AssertionError("native XGBoost TreeSHAP must not load SHAP")

    monkeypatch.setattr(shap_module, "_require_shap", fail_if_loaded)
    prepared = ShapAdapter().prepare(context, _request(backend="tree"))
    assert isinstance(prepared.explain, _NativeTreeExplainer)


def test_native_prepare_rejects_non_tree_booster() -> None:
    pytest.importorskip("xgboost")

    class FakeBooster:
        feature_names = ["feature_a", "feature_b"]

        @staticmethod
        def save_config():
            return '{"learner":{"gradient_booster":{"name":"gblinear"}}}'

    context = ExplainableModelContext(
        bundle_uri="/models/bundle",
        model_role="inference",
        artifact_name="native",
        artifact_format="ubj",
        flavor_id="xgboost-native-v1",
        native_attribution_id="xgboost-tree-shap-v1",
        artifact_path=None,
        model_object=FakeBooster(),
        feature_names=("feature_a", "feature_b"),
        objective="binary:logistic",
    )

    with pytest.raises(ValueError, match="gbtree or dart"):
        ShapAdapter().prepare(context, _request(backend="tree"))


def test_native_tree_shap_rejects_non_strict_contribution_shape() -> None:
    pytest.importorskip("xgboost")

    class FakeBooster:
        feature_types = None

        @staticmethod
        def predict(matrix, **kwargs):
            rows = matrix.num_row()
            if kwargs.get("pred_contribs"):
                return np.zeros((rows, 3), dtype=np.float32)
            return np.zeros((rows, 1), dtype=np.float32)

    explainer = _NativeTreeExplainer(
        FakeBooster(),
        feature_names=("feature_a", "feature_b"),
        objective="binary:logistic",
    )
    with pytest.raises(ValueError, match="strict shape contract"):
        explainer(np.asarray([[0.0, 1.0]], dtype=np.float32))


def test_real_xgboost_native_tree_shap_supports_regression() -> None:
    xgboost = pytest.importorskip("xgboost")
    X = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.asarray([0.0, 1.0, 1.0, 2.0])
    matrix = xgboost.DMatrix(
        X,
        label=y,
        feature_names=["feature_a", "feature_b"],
    )
    booster = xgboost.train(
        {"objective": "reg:squarederror", "max_depth": 2, "eta": 0.5},
        matrix,
        num_boost_round=4,
    )
    context = ExplainableModelContext(
        bundle_uri="/models/bundle",
        model_role="native",
        artifact_name="native",
        artifact_format="ubj",
        flavor_id="xgboost-native-v1",
        native_attribution_id="xgboost-tree-shap-v1",
        artifact_path=None,
        model_object=booster,
        feature_names=("feature_a", "feature_b"),
        objective="reg:squarederror",
    )
    request = _request(backend="tree", output_target="raw_margin")

    rows = ShapAdapter().explain_batch(
        ShapAdapter().prepare(context, request),
        X,
        input_ids=("0", "1", "2", "3"),
        model_digest="a" * 64,
        request=request,
    )

    assert len(rows) == len(X) * X.shape[1]
    assert {row.output_id for row in rows} == {"output_0"}
    expected = booster.predict(matrix, output_margin=True, strict_shape=True)
    for row_index in range(len(X)):
        selected = rows[row_index * X.shape[1] : (row_index + 1) * X.shape[1]]
        reconstructed = sum(row.contribution for row in selected) + float(
            selected[0].base_value
        )
        assert reconstructed == pytest.approx(float(expected[row_index, 0]))


def test_real_xgboost_native_tree_shap_selects_multiclass_output() -> None:
    xgboost = pytest.importorskip("xgboost")
    X = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.float32)
    matrix = xgboost.DMatrix(
        X,
        label=y,
        feature_names=["feature_a", "feature_b"],
    )
    booster = xgboost.train(
        {
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": 2,
            "eta": 0.5,
        },
        matrix,
        num_boost_round=6,
    )
    context = ExplainableModelContext(
        bundle_uri="/models/bundle",
        model_role="native",
        artifact_name="native",
        artifact_format="ubj",
        flavor_id="xgboost-native-v1",
        native_attribution_id="xgboost-tree-shap-v1",
        artifact_path=None,
        model_object=booster,
        feature_names=("feature_a", "feature_b"),
        objective="multi:softprob",
    )
    all_request = _request(backend="tree")
    predicted_request = _request(
        backend="tree",
        output_selection="predicted",
    )
    adapter = ShapAdapter()
    all_rows = adapter.explain_batch(
        adapter.prepare(context, all_request),
        X,
        input_ids=tuple(str(index) for index in range(len(X))),
        model_digest="a" * 64,
        request=all_request,
    )
    predicted_rows = adapter.explain_batch(
        adapter.prepare(context, predicted_request),
        X,
        input_ids=tuple(str(index) for index in range(len(X))),
        model_digest="a" * 64,
        request=predicted_request,
    )

    assert len(all_rows) == len(X) * X.shape[1] * 3
    assert len(predicted_rows) == len(X) * X.shape[1]
    margins = booster.predict(matrix, output_margin=True, strict_shape=True)
    expected_outputs = np.argmax(margins, axis=1)
    for row_index, output_index in enumerate(expected_outputs):
        selected = predicted_rows[row_index * X.shape[1] : (row_index + 1) * X.shape[1]]
        assert {row.output_id for row in selected} == {f"output_{output_index}"}


def test_real_xgboost_tree_shap_checks_raw_and_probability_outputs() -> None:
    xgboost = pytest.importorskip("xgboost")
    pytest.importorskip("shap")
    X = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    y = np.asarray([0, 1, 1, 1], dtype=np.float32)
    booster = xgboost.train(
        {"objective": "binary:logistic", "max_depth": 2, "eta": 0.5},
        xgboost.DMatrix(X, label=y, feature_names=["feature_a", "feature_b"]),
        num_boost_round=4,
    )
    context = ExplainableModelContext(
        bundle_uri="/models/bundle",
        model_role="native",
        artifact_name="native",
        artifact_format="ubj",
        flavor_id="xgboost-native-v1",
        native_attribution_id="xgboost-tree-shap-v1",
        artifact_path=None,
        model_object=booster,
        feature_names=("feature_a", "feature_b"),
        objective="binary:logistic",
    )
    raw_request = _request(backend="tree", output_target="model_output")
    prepared = ShapAdapter().prepare(context, raw_request)
    raw_rows = ShapAdapter().explain_batch(
        prepared,
        X,
        input_ids=("0", "1", "2", "3"),
        model_digest="a" * 64,
        request=raw_request,
    )
    assert raw_rows
    assert all(row.output_target == "model_output" for row in raw_rows)

    probability_request = _request(
        backend="tree",
        output_target="probability",
        reference={"uri": "/reference.npy"},
    )
    probability_context = replace(context, metadata={"reference_data": X})
    probability_prepared = ShapAdapter().prepare(
        probability_context, probability_request
    )
    probability_rows = ShapAdapter().explain_batch(
        probability_prepared,
        X,
        input_ids=("0", "1", "2", "3"),
        model_digest="a" * 64,
        request=probability_request,
    )
    assert probability_rows
    assert all(row.output_target == "probability" for row in probability_rows)

    log_loss_request = _request(
        backend="tree",
        output_target="log_loss",
        label_column="label",
        reference={"uri": "/reference.npy"},
    )
    log_loss_prepared = ShapAdapter().prepare(probability_context, log_loss_request)
    log_loss_rows = ShapAdapter().explain_batch(
        log_loss_prepared,
        X,
        input_ids=("0", "1", "2", "3"),
        model_digest="a" * 64,
        request=log_loss_request,
        labels=y,
    )
    assert log_loss_rows
    assert all(row.output_target == "log_loss" for row in log_loss_rows)
