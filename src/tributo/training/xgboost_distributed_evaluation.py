"""Bounded, exact evaluation over a distributed XGBoost test Dataset."""

from __future__ import annotations

import json
import math
from typing import Any, cast
from uuid import uuid4

from pydantic import Field

from tributo._common.config import StrictConfigModel
from tributo.exceptions import JobConfigurationError


class EvaluationConfig(StrictConfigModel):
    """Final evaluation controls; row-level predictions stay in Ray Data."""

    enabled: bool = True
    roc_curve: bool = True
    threshold_analysis: bool = True
    confusion_matrix: bool = True
    feature_importance: bool = True
    batch_size: int = Field(default=4096, ge=1)


_THRESHOLDS = tuple(index / 20 for index in range(1, 20))


def _binary_confusion(truth: Any, predicted: Any) -> tuple[int, int, int, int]:
    import numpy as np

    actual = np.asarray(truth).astype(int)
    inferred = np.asarray(predicted).astype(int)
    return (
        int(((actual == 0) & (inferred == 0)).sum()),
        int(((actual == 0) & (inferred == 1)).sum()),
        int(((actual == 1) & (inferred == 0)).sum()),
        int(((actual == 1) & (inferred == 1)).sum()),
    )


def batch_partial(
    y_true: Any,
    y_prediction: Any,
    objective: str,
    *,
    num_class: int | None,
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Reduce one prediction batch to fixed-size sufficient statistics."""
    import numpy as np

    truth = np.asarray(y_true)
    prediction = np.asarray(y_prediction)
    rows = int(len(truth))
    if objective.startswith("reg:"):
        residual = truth - prediction
        return {
            "task": "regression",
            "rows": rows,
            "sum_abs_error": float(np.abs(residual).sum()),
            "sum_squared_error": float((residual**2).sum()),
            "sum_y": float(truth.sum()),
            "sum_y_squared": float((truth**2).sum()),
        }
    if objective.startswith("multi:"):
        probabilities = prediction if prediction.ndim == 2 else None
        predicted = (
            probabilities.argmax(axis=1)
            if probabilities is not None
            else prediction.astype(int)
        )
        classes = num_class or (
            probabilities.shape[1]
            if probabilities is not None
            else int(max(truth.max(initial=0), predicted.max(initial=0))) + 1
        )
        matrix = np.zeros((classes, classes), dtype=int)
        for actual, inferred in zip(
            truth.astype(int), predicted.astype(int), strict=True
        ):
            if actual < 0 or inferred < 0 or actual >= classes or inferred >= classes:
                raise JobConfigurationError(
                    "Multiclass label/prediction falls outside configured num_class"
                )
            matrix[actual, inferred] += 1
        return {
            "task": "multiclass",
            "rows": rows,
            "num_class": classes,
            "confusion": matrix.ravel().tolist(),
        }
    probability = prediction.astype(float)
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = _binary_confusion(truth, predicted)
    partial: dict[str, Any] = {
        "task": "binary",
        "rows": rows,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    if config.threshold_analysis:
        partial["threshold_counts"] = [
            list(_binary_confusion(truth, probability >= threshold))
            for threshold in _THRESHOLDS
        ]
    return partial


def _sum_nested(left: Any, right: Any) -> Any:
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise JobConfigurationError(
                "Incompatible distributed evaluation statistics"
            )
        return [_sum_nested(a, b) for a, b in zip(left, right, strict=True)]
    return left + right


def merge_partials(
    combined: dict[str, Any] | None, partial: dict[str, Any]
) -> dict[str, Any]:
    if combined is None:
        return dict(partial)
    if combined["task"] != partial["task"] or combined.get("num_class") != partial.get(
        "num_class"
    ):
        raise JobConfigurationError("Incompatible distributed evaluation statistics")
    for key, value in partial.items():
        if key not in {"task", "num_class"}:
            combined[key] = _sum_nested(combined[key], value)
    return combined


def finalize_partial(
    partial: dict[str, Any], config: EvaluationConfig
) -> dict[str, Any]:
    """Convert global sufficient statistics to exact scalar metrics."""
    import numpy as np

    rows = int(partial["rows"])
    if not rows:
        raise JobConfigurationError("Test split produced no evaluation rows")
    report: dict[str, Any] = {"eval_test_rows": rows}
    if partial["task"] == "regression":
        sum_y = float(partial["sum_y"])
        squared_error = float(partial["sum_squared_error"])
        denominator = float(partial["sum_y_squared"]) - sum_y * sum_y / rows
        report.update(
            {
                "eval_rmse": math.sqrt(squared_error / rows),
                "eval_mae": float(partial["sum_abs_error"]) / rows,
                "eval_r2": 1.0 - squared_error / denominator if denominator else 0.0,
            }
        )
        return report
    if partial["task"] == "multiclass":
        classes = int(partial["num_class"])
        matrix = np.asarray(partial["confusion"], dtype=int).reshape(classes, classes)
        facts: list[tuple[float, float, float]] = []
        for index in range(classes):
            tp = int(matrix[index, index])
            fp = int(matrix[:, index].sum() - tp)
            fn = int(matrix[index, :].sum() - tp)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            facts.append((precision, recall, f1))
        report.update(
            {
                "eval_precision_macro": float(np.mean([x[0] for x in facts])),
                "eval_recall_macro": float(np.mean([x[1] for x in facts])),
                "eval_f1_macro": float(np.mean([x[2] for x in facts])),
            }
        )
        if config.confusion_matrix:
            report["eval_cm"] = matrix.tolist()
            report["eval_cm_labels"] = [str(index) for index in range(classes)]
        return report
    tn, fp, fn, tp = (int(partial[key]) for key in ("tn", "fp", "fn", "tp"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    report.update({"eval_precision": precision, "eval_recall": recall, "eval_f1": f1})
    if config.confusion_matrix:
        report.update(
            {
                "eval_cm_tp": tp,
                "eval_cm_fp": fp,
                "eval_cm_fn": fn,
                "eval_cm_tn": tn,
            }
        )
    if config.threshold_analysis:
        report.update(
            {
                "eval_thr_thresholds": list(_THRESHOLDS),
                "eval_thr_precision": [],
                "eval_thr_recall": [],
                "eval_thr_f1": [],
                "eval_thr_predicted_positive": [],
            }
        )
        for counts in partial["threshold_counts"]:
            _, threshold_fp, threshold_fn, threshold_tp = map(int, counts)
            p = (
                threshold_tp / (threshold_tp + threshold_fp)
                if threshold_tp + threshold_fp
                else 0.0
            )
            r = (
                threshold_tp / (threshold_tp + threshold_fn)
                if threshold_tp + threshold_fn
                else 0.0
            )
            score = 2 * p * r / (p + r) if p + r else 0.0
            report["eval_thr_precision"].append(p)
            report["eval_thr_recall"].append(r)
            report["eval_thr_f1"].append(score)
            report["eval_thr_predicted_positive"].append(threshold_tp + threshold_fp)
    return report


def predict_batch(
    batch: Any,
    *,
    model_artifact: Any,
    label_col: str,
    objective: str,
    num_class: int | None,
) -> Any:
    """Ray task: predict a batch without returning row data to the Driver."""
    del objective, num_class
    import numpy as np
    import pandas as pd
    import ray
    import xgboost

    if isinstance(model_artifact, ray.ObjectRef):
        model_artifact = ray.get(model_artifact)
    booster = xgboost.Booster()
    booster.load_model(bytearray(model_artifact))
    output: dict[str, Any] = {"y_true": batch[label_col].to_numpy()}
    prediction = np.asarray(
        booster.predict(xgboost.DMatrix(batch.drop(columns=[label_col])))
    )
    if prediction.ndim == 1:
        output["prediction"] = prediction
    else:
        for index in range(prediction.shape[1]):
            output[f"prediction_{index}"] = prediction[:, index]
    return pd.DataFrame(output)


def prediction_partial_batch(
    batch: Any,
    *,
    objective: str,
    num_class: int | None,
    evaluation: dict[str, Any],
) -> Any:
    probability_columns = sorted(
        (column for column in batch if column.startswith("prediction_")),
        key=lambda column: int(column.rsplit("_", 1)[1]),
    )
    prediction = (
        batch[probability_columns].to_numpy()
        if probability_columns
        else batch["prediction"].to_numpy()
    )
    partial = batch_partial(
        batch["y_true"].to_numpy(),
        prediction,
        objective,
        num_class=num_class,
        config=EvaluationConfig.model_validate(evaluation),
    )
    return {"partial": [json.dumps(partial, separators=(",", ":"))]}


def score_rows_batch(batch: Any, *, objective: str, num_class: int | None) -> Any:
    import numpy as np
    import pandas as pd

    truth = np.asarray(batch["y_true"]).astype(int)
    if objective.startswith("multi:"):
        columns = sorted(
            (column for column in batch if column.startswith("prediction_")),
            key=lambda column: int(column.rsplit("_", 1)[1]),
        )
        if not columns:  # multi:softmax has class ids, so no rank metrics.
            return pd.DataFrame(
                {
                    "class_index": np.array([], dtype=int),
                    "score": np.array([], dtype=float),
                    "positive": np.array([], dtype=int),
                    "negative": np.array([], dtype=int),
                }
            )
        rows = []
        for index in range(num_class or len(columns)):
            positive = (truth == index).astype(int)
            rows.append(
                pd.DataFrame(
                    {
                        "class_index": index,
                        "score": batch[f"prediction_{index}"].to_numpy(),
                        "positive": positive,
                        "negative": 1 - positive,
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)
    positive = (truth == 1).astype(int)
    return pd.DataFrame(
        {
            "class_index": 0,
            "score": batch["prediction"].to_numpy(),
            "positive": positive,
            "negative": 1 - positive,
        }
    )


def pack_score_groups_batch(batch: Any) -> Any:
    import pandas as pd

    columns = [
        "partition_id",
        "class_index",
        "first_score",
        "positive_total",
        "negative_total",
        "scores",
        "positives",
        "negatives",
    ]
    rows = []
    for class_index, group in batch.groupby("class_index", sort=False):
        rows.append(
            {
                "partition_id": uuid4().hex,
                "class_index": int(class_index),
                "first_score": float(group["score"].iloc[0]),
                "positive_total": int(group["positive"].sum()),
                "negative_total": int(group["negative"].sum()),
                "scores": group["score"].astype(float).tolist(),
                "positives": group["positive"].astype(int).tolist(),
                "negatives": group["negative"].astype(int).tolist(),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def rank_prefixes(summaries: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    prefixes: dict[str, tuple[int, int]] = {}
    running: dict[int, tuple[int, int]] = {}
    for row in sorted(
        summaries,
        key=lambda value: (
            int(value["class_index"]),
            -float(value["first_score"]),
        ),
    ):
        class_index = int(row["class_index"])
        positive, negative = running.get(class_index, (0, 0))
        prefixes[str(row["partition_id"])] = (positive, negative)
        running[class_index] = (
            positive + int(row["positive_total"]),
            negative + int(row["negative_total"]),
        )
    return prefixes


def _bounded_curve_append(
    points: list[tuple[float, float]], point: tuple[float, float], maximum: int = 256
) -> None:
    points.append(point)
    if len(points) > maximum * 2:
        points[:] = [points[0], *points[2::2]]


def rank_contribution_batch(
    batch: Any,
    *,
    prefixes: dict[str, tuple[int, int]],
    totals: dict[int, tuple[int, int]],
    include_curve: bool,
) -> Any:
    import pandas as pd

    rows = []
    for _, partition in batch.iterrows():
        partition_id = str(partition["partition_id"])
        class_index = int(partition["class_index"])
        tp, fp = prefixes[partition_id]
        total_positive, total_negative = totals[class_index]
        auc = ap = 0.0
        curve: list[tuple[float, float]] = []
        for positive, negative in zip(
            partition["positives"], partition["negatives"], strict=True
        ):
            previous_tpr = tp / total_positive if total_positive else 0.0
            previous_fpr = fp / total_negative if total_negative else 0.0
            tp += int(positive)
            fp += int(negative)
            tpr = tp / total_positive if total_positive else 0.0
            fpr = fp / total_negative if total_negative else 0.0
            auc += (fpr - previous_fpr) * (tpr + previous_tpr) / 2.0
            if total_positive and tp + fp:
                ap += int(positive) / total_positive * tp / (tp + fp)
            if include_curve:
                _bounded_curve_append(curve, (fpr, tpr))
        rows.append(
            {
                "class_index": class_index,
                "first_score": float(partition["first_score"]),
                "auc": auc,
                "ap": ap,
                "curve": json.dumps(curve, separators=(",", ":")),
            }
        )
    return pd.DataFrame(rows)


def merge_rank_contributions(
    rows: Any,
    totals: dict[int, tuple[int, int]],
    *,
    multiclass: bool,
    include_curve: bool,
) -> dict[str, Any]:
    contributions = list(rows)
    auc = dict.fromkeys(totals, 0.0)
    average_precision = dict.fromkeys(totals, 0.0)
    for row in contributions:
        class_index = int(row["class_index"])
        auc[class_index] += float(row["auc"])
        average_precision[class_index] += float(row["ap"])
    defined_classes = {
        class_index
        for class_index, (positives, negatives) in totals.items()
        if positives > 0 and negatives > 0
    }
    if multiclass:
        defined_auc = [auc[index] for index in defined_classes]
        return {"eval_auc": sum(defined_auc) / len(defined_auc)} if defined_auc else {}
    result: dict[str, Any] = {}
    if 0 in defined_classes:
        result.update(
            {
                "eval_auc": auc.get(0, 0.0),
                "eval_avg_precision": average_precision.get(0, 0.0),
            }
        )
    if include_curve and 0 in defined_classes:
        curve = [(0.0, 0.0)]
        for row in sorted(contributions, key=lambda item: -float(item["first_score"])):
            raw = row["curve"]
            curve.extend(json.loads(raw) if isinstance(raw, str) else raw)
        positives, negatives = totals.get(0, (0, 0))
        final = (1.0 if negatives else 0.0, 1.0 if positives else 0.0)
        if curve[-1] != final:
            curve.append(final)
        # Curve output is bounded independently from exact scalar calculation.
        if len(curve) > 256:
            step = math.ceil(len(curve) / 256)
            curve = [*curve[::step], curve[-1]]
        result["eval_roc_fpr"] = [float(point[0]) for point in curve]
        result["eval_roc_tpr"] = [float(point[1]) for point in curve]
    return result


def exact_rank_metrics(
    predictions: Any,
    *,
    objective: str,
    num_class: int | None,
    partial: dict[str, Any],
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Compute exact AUC/AP via global sort with O(partitions*classes) Driver state."""
    from ray.data.aggregate import Sum

    if objective.startswith("reg:") or objective == "multi:softmax":
        return {}
    score_rows = predictions.map_batches(
        cast(Any, score_rows_batch),
        batch_format="pandas",
        batch_size=config.batch_size,
        fn_kwargs={"objective": objective, "num_class": num_class},
    )
    grouped = (
        score_rows.groupby(["class_index", "score"])
        .aggregate(
            Sum("positive", alias_name="positive"),
            Sum("negative", alias_name="negative"),
        )
        .sort(["class_index", "score"], descending=[False, True])
    )
    if objective.startswith("multi:"):
        matrix = partial["confusion"]
        classes = int(partial["num_class"])
        totals = {
            index: (
                sum(matrix[index * classes : (index + 1) * classes]),
                int(partial["rows"])
                - sum(matrix[index * classes : (index + 1) * classes]),
            )
            for index in range(classes)
        }
    else:
        totals = {
            0: (
                int(partial["tp"]) + int(partial["fn"]),
                int(partial["tn"]) + int(partial["fp"]),
            )
        }
    packed = grouped.map_batches(
        cast(Any, pack_score_groups_batch),
        batch_format="pandas",
        batch_size=None,
    ).materialize()
    summary_columns = [
        "partition_id",
        "class_index",
        "first_score",
        "positive_total",
        "negative_total",
    ]
    summaries = list(packed.select_columns(summary_columns).iter_rows())
    contributions = packed.map_batches(
        cast(Any, rank_contribution_batch),
        batch_format="pandas",
        batch_size=None,
        fn_kwargs={
            "prefixes": rank_prefixes(summaries),
            "totals": totals,
            "include_curve": config.roc_curve and not objective.startswith("multi:"),
        },
    )
    return merge_rank_contributions(
        contributions.iter_rows(),
        totals,
        multiclass=objective.startswith("multi:"),
        include_curve=config.roc_curve and not objective.startswith("multi:"),
    )


def checkpoint_model_artifact(checkpoint: Any) -> Any:
    """Place immutable booster bytes once in Ray's distributed object store."""
    import ray
    from ray.train.xgboost import XGBoostCheckpoint

    typed = XGBoostCheckpoint(checkpoint.path, filesystem=checkpoint.filesystem)
    return ray.put(bytes(typed.get_model().save_raw()))


def evaluate_dataset(
    dataset: Any,
    checkpoint: Any,
    *,
    label_col: str,
    objective: str,
    num_class: int | None,
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Evaluate the entire lazy Dataset without collecting row-level data."""
    model_artifact = checkpoint_model_artifact(checkpoint)
    predictions = dataset.map_batches(
        cast(Any, predict_batch),
        batch_format="pandas",
        batch_size=config.batch_size,
        fn_kwargs={
            "model_artifact": model_artifact,
            "label_col": label_col,
            "objective": objective,
            "num_class": num_class,
        },
    ).materialize()
    partials = predictions.map_batches(
        cast(Any, prediction_partial_batch),
        batch_format="pandas",
        batch_size=config.batch_size,
        fn_kwargs={
            "objective": objective,
            "num_class": num_class,
            "evaluation": config.model_dump(),
        },
    )
    combined: dict[str, Any] | None = None
    for row in partials.iter_rows():
        raw = row["partial"]
        combined = merge_partials(
            combined, json.loads(raw) if isinstance(raw, str) else raw
        )
    if combined is None:
        raise JobConfigurationError("Test split produced no evaluation statistics")
    result = finalize_partial(combined, config)
    result.update(
        exact_rank_metrics(
            predictions,
            objective=objective,
            num_class=num_class,
            partial=combined,
            config=config,
        )
    )
    return result
