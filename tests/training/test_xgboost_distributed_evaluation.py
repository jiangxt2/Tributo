from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tributo.training.xgboost_distributed_evaluation import (
    EvaluationConfig,
    batch_partial,
    evaluate_dataset,
    exact_rank_metrics,
    finalize_partial,
    merge_partials,
    merge_rank_contributions,
)
from tributo.training.xgboost_trainer import XGBoostTrainerImpl


def test_random_split_uses_row_level_shuffle_before_partitioning() -> None:
    from tributo.training.xgboost_evaluator import split_dataset

    dataset = MagicMock()
    shuffled = MagicMock()
    dataset.random_shuffle.return_value = shuffled
    shuffled.split_proportionately.return_value = ("train", "test")

    assert split_dataset(dataset, val_size=0, test_size=0.2, seed=17) == (
        "train",
        None,
        "test",
    )
    dataset.random_shuffle.assert_called_once_with(seed=17)
    dataset.randomize_block_order.assert_not_called()


def test_regression_partials_produce_exact_global_rmse_mae_and_r2() -> None:
    config = EvaluationConfig()
    combined = None
    for truth, prediction in (
        (np.array([1.0, 2.0]), np.array([1.5, 1.5])),
        (np.array([3.0, 4.0]), np.array([2.5, 4.5])),
    ):
        combined = merge_partials(
            combined,
            batch_partial(
                truth,
                prediction,
                "reg:squarederror",
                num_class=None,
                config=config,
            ),
        )

    assert combined is not None
    metrics = finalize_partial(combined, config)
    assert metrics["eval_test_rows"] == 4
    assert metrics["eval_rmse"] == pytest.approx(0.5)
    assert metrics["eval_mae"] == pytest.approx(0.5)
    assert metrics["eval_r2"] == pytest.approx(0.8)


def test_multiclass_softmax_1d_has_global_macro_keys_and_full_matrix() -> None:
    config = EvaluationConfig()
    combined = None
    for truth, prediction in (
        (np.array([0, 1]), np.array([0, 2])),
        (np.array([2, 0]), np.array([2, 1])),
    ):
        combined = merge_partials(
            combined,
            batch_partial(
                truth,
                prediction,
                "multi:softmax",
                num_class=3,
                config=config,
            ),
        )

    assert combined is not None
    metrics = finalize_partial(combined, config)
    assert metrics["eval_test_rows"] == 4
    assert set(metrics) >= {
        "eval_precision_macro",
        "eval_recall_macro",
        "eval_f1_macro",
        "eval_cm",
        "eval_cm_labels",
    }
    assert metrics["eval_cm"] == [[1, 1, 0], [0, 0, 1], [0, 0, 1]]
    assert metrics["eval_cm_labels"] == ["0", "1", "2"]
    assert "eval_auc" not in metrics


def test_production_evaluation_streams_bounded_partials_without_row_collect() -> None:
    partials = MagicMock()
    partials.iter_rows.return_value = iter(
        [
            {
                "partial": json.dumps(
                    {
                        "task": "regression",
                        "rows": 2,
                        "sum_abs_error": 1.0,
                        "sum_squared_error": 0.5,
                        "sum_y": 3.0,
                        "sum_y_squared": 5.0,
                    }
                )
            },
            {
                "partial": json.dumps(
                    {
                        "task": "regression",
                        "rows": 2,
                        "sum_abs_error": 1.0,
                        "sum_squared_error": 0.5,
                        "sum_y": 7.0,
                        "sum_y_squared": 25.0,
                    }
                )
            },
        ]
    )
    predictions = MagicMock()
    predictions.map_batches.return_value = partials
    lazy_predictions = MagicMock()
    lazy_predictions.materialize.return_value = predictions
    dataset = MagicMock()
    dataset.map_batches.return_value = lazy_predictions

    with patch(
        "tributo.training.xgboost_distributed_evaluation.checkpoint_model_artifact",
        return_value=object(),
    ):
        metrics = evaluate_dataset(
            dataset,
            MagicMock(),
            label_col="label",
            objective="reg:squarederror",
            num_class=None,
            config=EvaluationConfig(),
        )

    assert metrics["eval_test_rows"] == 4
    assert metrics["eval_rmse"] == pytest.approx(0.5)
    dataset.to_pandas.assert_not_called()
    dataset.take_all.assert_not_called()
    predictions.to_pandas.assert_not_called()
    predictions.take_all.assert_not_called()
    partials.to_pandas.assert_not_called()
    partials.take_all.assert_not_called()


def test_exact_rank_production_path_collects_only_partition_summaries() -> None:
    summary_rows = [
        {
            "partition_id": "high",
            "class_index": 0,
            "first_score": 0.501,
            "positive_total": 1,
            "negative_total": 0,
        },
        {
            "partition_id": "low",
            "class_index": 0,
            "first_score": 0.500,
            "positive_total": 0,
            "negative_total": 1,
        },
    ]
    summary_dataset = MagicMock()
    summary_dataset.iter_rows.return_value = iter(summary_rows)
    contributions = MagicMock()
    contributions.iter_rows.return_value = iter(
        [
            {
                "class_index": 0,
                "first_score": 0.501,
                "auc": 1.0,
                "ap": 1.0,
                "curve": "[[1.0,1.0]]",
            }
        ]
    )
    packed = MagicMock()
    packed.select_columns.return_value = summary_dataset
    packed.map_batches.return_value = contributions
    packed_lazy = MagicMock()
    packed_lazy.materialize.return_value = packed
    sorted_groups = MagicMock()
    sorted_groups.map_batches.return_value = packed_lazy
    aggregated = MagicMock()
    aggregated.sort.return_value = sorted_groups
    grouped = MagicMock()
    grouped.aggregate.return_value = aggregated
    score_rows = MagicMock()
    score_rows.groupby.return_value = grouped
    predictions = MagicMock()
    predictions.map_batches.return_value = score_rows

    metrics = exact_rank_metrics(
        predictions,
        objective="binary:logistic",
        num_class=None,
        partial={"task": "binary", "rows": 2, "tn": 1, "fp": 0, "fn": 0, "tp": 1},
        config=EvaluationConfig(),
    )

    assert metrics["eval_auc"] == 1.0
    assert metrics["eval_avg_precision"] == 1.0
    predictions.to_pandas.assert_not_called()
    predictions.take_all.assert_not_called()
    packed.take_all.assert_not_called()
    summary_dataset.take_all.assert_not_called()


def test_undefined_rank_metrics_are_omitted_and_missing_multiclass_is_excluded() -> (
    None
):
    binary = merge_rank_contributions(
        [{"class_index": 0, "auc": 0.0, "ap": 0.0, "first_score": 1.0}],
        {0: (2, 0)},
        multiclass=False,
        include_curve=True,
    )
    multiclass = merge_rank_contributions(
        [
            {"class_index": 0, "auc": 0.8, "ap": 0.0, "first_score": 1.0},
            {"class_index": 1, "auc": 0.6, "ap": 0.0, "first_score": 1.0},
            {"class_index": 2, "auc": 0.0, "ap": 0.0, "first_score": 1.0},
        ],
        {0: (2, 2), 1: (1, 3), 2: (0, 4)},
        multiclass=True,
        include_curve=False,
    )

    assert binary == {}
    assert multiclass["eval_auc"] == pytest.approx(0.7)


def test_cross_partition_tie_contributions_do_not_add_artificial_rank_gain() -> None:
    metrics = merge_rank_contributions(
        [
            {"class_index": 0, "auc": 0.25, "ap": 0.25, "first_score": 0.5},
            {"class_index": 0, "auc": 0.25, "ap": 0.25, "first_score": 0.5},
        ],
        {0: (1, 1)},
        multiclass=False,
        include_curve=False,
    )

    assert metrics["eval_auc"] == pytest.approx(0.5)


def test_training_loop_overrides_rank_local_counts_with_global_dataset_counts() -> None:
    progress = MagicMock()
    splits: dict[str, MagicMock] = {}
    for name, count in {"train": 160, "val": 20, "test": 20}.items():
        dataset = MagicMock(name=name)
        dataset.count.return_value = count
        splits[name] = dataset
    result = SimpleNamespace(
        metrics={"row_count_train": 80, "eval_test_rows": 3},
        checkpoint=MagicMock(),
    )
    ray_trainer = MagicMock()
    ray_trainer.fit.return_value = result
    trainer = XGBoostTrainerImpl(
        datasets=splits,
        config={"ray": {"num_workers": 2}},
        _progress=progress,
    )

    with (
        patch(
            "tributo.training.xgboost_trainer._build_trainer",
            return_value=ray_trainer,
        ),
        patch(
            "tributo.training.xgboost_distributed_evaluation.evaluate_dataset",
            return_value={"eval_test_rows": 20, "eval_auc": 0.8},
        ),
    ):
        returned = trainer.training_loop()

    assert returned.metrics["row_count_train"] == 160
    assert returned.metrics["row_count_val"] == 20
    assert returned.metrics["row_count_test"] == 20
    assert returned.metrics["eval_test_rows"] == 20
