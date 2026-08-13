"""Unit tests for the dependency-neutral SHAP adapter boundary."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tributo.explainability.protocols import ExplainableModelContext, PreparedExplainer
from tributo.explainability.shap import ShapAdapter

from .test_contracts import _request


class _Explanation:
    values = np.asarray([[[0.1], [1.0]], [[0.1], [2.0]]])
    data = np.asarray([[10.0, 20.0], [30.0, 40.0]])
    base_values = np.asarray([0.5, 0.6])


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
        artifact_path=None,
        objective="binary:logistic",
    )
    decision = ShapAdapter.supports(
        context,
        _request(backend="tree", output_target="unknown_target"),
    )
    assert decision.supported is False
    assert "output_target" in decision.reason


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
