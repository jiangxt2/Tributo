"""Unit tests for the dependency-neutral SHAP adapter boundary."""

from __future__ import annotations

import numpy as np
import pytest

from tributo.explainability.protocols import (
    ExplainableModelContext,
    PreparedExplainer,
    SupportDecision,
)
from tributo.explainability.shap import ShapAdapter

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


def test_native_attribution_is_delegated_to_the_loaded_wheel_model() -> None:
    prepared = PreparedExplainer(
        backend="tree",
        exactness="exact",
        explain=lambda batch, **_: _Explanation(),
        feature_names=("feature_a", "feature_b"),
    )

    class NativeModel:
        native_attribution_id = "external-tree-attribution-v1"
        native_model_object = object()
        native_feature_names = ("feature_a", "feature_b")
        native_objective = "external-objective"

        def native_attribution_support(self, request):
            del request
            return SupportDecision(
                supported=True,
                backend="tree",
                exactness="exact",
            )

        def prepare_native_attribution(
            self,
            request,
            *,
            feature_names,
            reference_data,
        ):
            del request, feature_names, reference_data
            return prepared

    model = NativeModel()
    context = ExplainableModelContext(
        bundle_uri="/bundle",
        model_role="explainability_model",
        artifact_name="native",
        artifact_format="external",
        flavor_id="external-tree-v1",
        artifact_path=None,
        model_object=model,
        feature_names=model.native_feature_names,
        native_attribution_id=model.native_attribution_id,
        preprocessor_digest="a" * 64,
    )
    request = _request(backend="tree")
    adapter = ShapAdapter()
    assert adapter.supports(context, request).supported is True
    resolved = adapter.prepare(context, request)
    assert resolved.explain is prepared.explain
    assert resolved.preprocessor_digest == "a" * 64
