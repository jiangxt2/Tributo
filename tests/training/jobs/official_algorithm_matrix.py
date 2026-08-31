"""Immutable category and identity matrix for the official algorithm Gate."""

from __future__ import annotations

from types import MappingProxyType

CATEGORY_ENTRY_POINTS: dict[str, tuple[str, ...]] = {
    "classical": (
        "extra_trees.joblib",
        "extra_trees.native",
        "isolation_forest.parallel_ensemble",
        "kmeans.iterative",
        "kmeans_minibatch.iterative",
        "linear_regression.iterative",
        "logistic_regression.iterative",
        "multinomial_nb",
        "pca.map_reduce",
        "random_forest.joblib",
        "random_forest.native",
        "sgd_classifier.iterative",
        "sgd_regressor.iterative",
    ),
    "boosting": (
        "catboost.parallel_ensemble",
        "lightgbm.framework_native",
        "xgboost.framework_native",
    ),
    "torch": (
        "dnn.recipe_v2",
        "gru_classifier.recipe_v2",
        "lstm_classifier.recipe_v2",
        "pu.recipe_v2",
        "tabular_autoencoder",
        "temporal_conv_classifier",
    ),
    "recsys_multistage_nlp": (
        "jagged_embedding_recommender",
        "pretrain_finetune_classifier",
        "teacher_student_distillation",
        "token_transformer_classifier",
        "two_tower_recommender",
    ),
    "causal": (
        "difference_in_means_ate",
        "doubly_robust_ate",
        "dowhy_linear_refutation",
        "gcm_root_cause",
        "linear_dml_ate",
        "linear_iv_ate",
        "pc_stability_discovery",
        "x_learner.framework_native",
    ),
    "graph": (
        "graphsage_node_classifier",
        "rgcn_node_classifier",
    ),
}

ALL_ENTRY_POINTS = frozenset(
    entry_point
    for entry_points in CATEGORY_ENTRY_POINTS.values()
    for entry_point in entry_points
)


def parse_entry_point_selection(value: str) -> frozenset[str] | None:
    """Parse an optional exact Gate selection without silently widening it."""
    if not value:
        return None
    entries = tuple(item.strip() for item in value.split(",") if item.strip())
    if not entries or len(set(entries)) != len(entries):
        raise ValueError("official algorithm Gate entry points must be unique names")
    unknown = sorted(set(entries) - ALL_ENTRY_POINTS)
    if unknown:
        raise ValueError(f"unknown official algorithm Gate entry points: {unknown}")
    return frozenset(entries)


def entry_points_for_gate(category: str, selection: str) -> frozenset[str]:
    """Return the exact expected Entry Points for one category Ray Job."""
    if category == "all":
        category_entry_points = ALL_ENTRY_POINTS
    elif category in CATEGORY_ENTRY_POINTS:
        category_entry_points = frozenset(CATEGORY_ENTRY_POINTS[category])
    else:
        raise ValueError(f"unknown official algorithm Gate category: {category!r}")
    selected = parse_entry_point_selection(selection)
    return (
        category_entry_points
        if selected is None
        else category_entry_points.intersection(selected)
    )


_IMPLEMENTATION_ENTRY_POINTS = MappingProxyType(
    {
        "tributo.official.random_forest.joblib": "random_forest.joblib",
        "tributo.official.random_forest.native_ensemble": "random_forest.native",
        "tributo.official.extra_trees.joblib": "extra_trees.joblib",
        "tributo.official.extra_trees.native_ensemble": "extra_trees.native",
        "tributo.official.boosting.xgboost": "xgboost.framework_native",
        "tributo.official.boosting.lightgbm": "lightgbm.framework_native",
        "tributo.official.catboost.parallel_ensemble": ("catboost.parallel_ensemble"),
        "tributo.official.tabular_torch.dnn": "dnn.recipe_v2",
        "tributo.official.tabular_torch.pu": "pu.recipe_v2",
        "tributo.official.causal_xlearner.xgboost": ("x_learner.framework_native"),
    }
)

_ALGORITHM_ENTRY_POINTS = MappingProxyType(
    {
        "difference_in_means_ate": "difference_in_means_ate",
        "dnn": "dnn.recipe_v2",
        "doubly_robust_ate": "doubly_robust_ate",
        "dowhy_linear_refutation": "dowhy_linear_refutation",
        "gcm_root_cause": "gcm_root_cause",
        "graphsage_node_classifier": "graphsage_node_classifier",
        "gru_classifier": "gru_classifier.recipe_v2",
        "isolation_forest": "isolation_forest.parallel_ensemble",
        "jagged_embedding_recommender": "jagged_embedding_recommender",
        "kmeans": "kmeans.iterative",
        "kmeans_minibatch": "kmeans_minibatch.iterative",
        "lightgbm": "lightgbm.framework_native",
        "linear_dml_ate": "linear_dml_ate",
        "linear_iv_ate": "linear_iv_ate",
        "linear_regression": "linear_regression.iterative",
        "logistic_regression": "logistic_regression.iterative",
        "lstm_classifier": "lstm_classifier.recipe_v2",
        "multinomial_nb": "multinomial_nb",
        "pc_stability_discovery": "pc_stability_discovery",
        "pca": "pca.map_reduce",
        "pretrain_finetune_classifier": "pretrain_finetune_classifier",
        "pu": "pu.recipe_v2",
        "rgcn_node_classifier": "rgcn_node_classifier",
        "sgd_classifier": "sgd_classifier.iterative",
        "sgd_regressor": "sgd_regressor.iterative",
        "tabular_autoencoder": "tabular_autoencoder",
        "teacher_student_distillation": "teacher_student_distillation",
        "temporal_conv_classifier": "temporal_conv_classifier",
        "token_transformer_classifier": "token_transformer_classifier",
        "two_tower_recommender": "two_tower_recommender",
        "x_learner": "x_learner.framework_native",
        "xgboost": "xgboost.framework_native",
        "catboost": "catboost.parallel_ensemble",
    }
)


def entry_point_for(algorithm: str, implementation_id: str | None) -> str:
    """Resolve one execution record to its installed Entry Point identity."""
    if implementation_id is not None:
        resolved = _IMPLEMENTATION_ENTRY_POINTS.get(implementation_id)
        if resolved is not None:
            return resolved
    try:
        return _ALGORITHM_ENTRY_POINTS[algorithm]
    except KeyError as exc:
        raise KeyError(
            f"official Gate has no Entry Point identity for {algorithm!r}/"
            f"{implementation_id!r}"
        ) from exc


def category_for_entry_point(entry_point: str) -> str:
    """Return the single category that owns an Entry Point."""
    matches = [
        category
        for category, entry_points in CATEGORY_ENTRY_POINTS.items()
        if entry_point in entry_points
    ]
    if len(matches) != 1:
        raise KeyError(f"Entry Point {entry_point!r} has category matches {matches}")
    return matches[0]


__all__ = [
    "ALL_ENTRY_POINTS",
    "CATEGORY_ENTRY_POINTS",
    "category_for_entry_point",
    "entry_point_for",
    "entry_points_for_gate",
    "parse_entry_point_selection",
]
