"""Immutable category and identity matrix for the official algorithm Gate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class OfficialAlgorithmIdentity:
    """Immutable installed identity expected for one official Entry Point."""

    distribution: str
    algorithm_id: str
    implementation_id: str


OFFICIAL_ALGORITHM_IDENTITIES: Mapping[str, OfficialAlgorithmIdentity] = (
    MappingProxyType(
        {
            "catboost.parallel_ensemble": OfficialAlgorithmIdentity(
                "tributo-algorithms-catboost",
                "catboost",
                "tributo.official.catboost.parallel_ensemble",
            ),
            "difference_in_means_ate": OfficialAlgorithmIdentity(
                "tributo-algorithms-causal-core",
                "difference_in_means_ate",
                "tributo.official.causal.difference_in_means",
            ),
            "dnn.recipe_v2": OfficialAlgorithmIdentity(
                "tributo-algorithms-tabular-torch",
                "dnn",
                "tributo.official.tabular_torch.dnn",
            ),
            "doubly_robust_ate": OfficialAlgorithmIdentity(
                "tributo-algorithms-causal-dr",
                "doubly_robust_ate",
                "tributo.official.causal_dr.aipw",
            ),
            "dowhy_linear_refutation": OfficialAlgorithmIdentity(
                "tributo-algorithms-causal-dowhy",
                "dowhy_linear_refutation",
                "tributo.official.causal_dowhy.linear_refutation",
            ),
            "extra_trees.joblib": OfficialAlgorithmIdentity(
                "tributo-algorithms-classical",
                "extra_trees",
                "tributo.official.extra_trees.joblib",
            ),
            "extra_trees.native": OfficialAlgorithmIdentity(
                "tributo-algorithms-classical",
                "extra_trees",
                "tributo.official.extra_trees.native_ensemble",
            ),
            "gcm_root_cause": OfficialAlgorithmIdentity(
                "tributo-algorithms-causal-dowhy",
                "gcm_root_cause",
                "tributo.official.causal_dowhy.gcm_root_cause",
            ),
            "graphsage_node_classifier": OfficialAlgorithmIdentity(
                "tributo-algorithms-graph-pyg",
                "graphsage_node_classifier",
                "tributo.official.graph_pyg.graphsage",
            ),
            "gru_classifier.recipe_v2": OfficialAlgorithmIdentity(
                "tributo-algorithms-timeseries",
                "gru_classifier",
                "tributo.official.timeseries.gru.recipe_v2",
            ),
            "isolation_forest.parallel_ensemble": OfficialAlgorithmIdentity(
                "tributo-algorithms-classical",
                "isolation_forest",
                "tributo.official.classical.isolation_forest.parallel_ensemble",
            ),
            "jagged_embedding_recommender": OfficialAlgorithmIdentity(
                "tributo-algorithms-recsys-torch",
                "jagged_embedding_recommender",
                "tributo.official.recsys_torch.jagged_embedding",
            ),
            "kmeans.iterative": OfficialAlgorithmIdentity(
                "tributo-algorithms-classical",
                "kmeans",
                "tributo.official.classical.kmeans.iterative",
            ),
            "kmeans_minibatch.iterative": OfficialAlgorithmIdentity(
                "tributo-algorithms-classical",
                "kmeans_minibatch",
                "tributo.official.classical.kmeans_minibatch.iterative",
            ),
            "lightgbm.framework_native": OfficialAlgorithmIdentity(
                "tributo-algorithms-boosting",
                "lightgbm",
                "tributo.official.boosting.lightgbm",
            ),
            "linear_dml_ate": OfficialAlgorithmIdentity(
                "tributo-algorithms-causal-core",
                "linear_dml_ate",
                "tributo.official.causal.linear_dml",
            ),
            "linear_iv_ate": OfficialAlgorithmIdentity(
                "tributo-algorithms-causal-core",
                "linear_iv_ate",
                "tributo.official.causal.linear_iv",
            ),
            "linear_regression.iterative": OfficialAlgorithmIdentity(
                "tributo-algorithms-classical",
                "linear_regression",
                "tributo.official.linear_regression.squared_l2",
            ),
            "logistic_regression.iterative": OfficialAlgorithmIdentity(
                "tributo-algorithms-classical",
                "logistic_regression",
                "tributo.official.logistic_regression.binary_l2",
            ),
            "lstm_classifier.recipe_v2": OfficialAlgorithmIdentity(
                "tributo-algorithms-timeseries",
                "lstm_classifier",
                "tributo.official.timeseries.lstm.recipe_v2",
            ),
            "multinomial_nb": OfficialAlgorithmIdentity(
                "tributo-algorithms-classical",
                "multinomial_nb",
                "tributo.official.multinomial_nb.map_reduce",
            ),
            "pc_stability_discovery": OfficialAlgorithmIdentity(
                "tributo-algorithms-causal-discovery",
                "pc_stability_discovery",
                "tributo.official.causal_discovery.pc_stability",
            ),
            "pca.map_reduce": OfficialAlgorithmIdentity(
                "tributo-algorithms-classical",
                "pca",
                "tributo.official.classical.pca.map_reduce",
            ),
            "pretrain_finetune_classifier": OfficialAlgorithmIdentity(
                "tributo-algorithms-multistage-torch",
                "pretrain_finetune_classifier",
                "tributo.official.multistage_torch.pretrain_finetune",
            ),
            "pu.recipe_v2": OfficialAlgorithmIdentity(
                "tributo-algorithms-tabular-torch",
                "pu",
                "tributo.official.tabular_torch.pu",
            ),
            "random_forest.joblib": OfficialAlgorithmIdentity(
                "tributo-algorithms-classical",
                "random_forest",
                "tributo.official.random_forest.joblib",
            ),
            "random_forest.native": OfficialAlgorithmIdentity(
                "tributo-algorithms-classical",
                "random_forest",
                "tributo.official.random_forest.native_ensemble",
            ),
            "rgcn_node_classifier": OfficialAlgorithmIdentity(
                "tributo-algorithms-graph-pyg",
                "rgcn_node_classifier",
                "tributo.official.graph_pyg.rgcn",
            ),
            "sgd_classifier.iterative": OfficialAlgorithmIdentity(
                "tributo-algorithms-classical",
                "sgd_classifier",
                "tributo.official.classical.sgd_classifier.iterative",
            ),
            "sgd_regressor.iterative": OfficialAlgorithmIdentity(
                "tributo-algorithms-classical",
                "sgd_regressor",
                "tributo.official.classical.sgd_regressor.iterative",
            ),
            "tabular_autoencoder": OfficialAlgorithmIdentity(
                "tributo-algorithms-representation",
                "tabular_autoencoder",
                "tributo.official.representation.tabular_autoencoder",
            ),
            "teacher_student_distillation": OfficialAlgorithmIdentity(
                "tributo-algorithms-multistage-torch",
                "teacher_student_distillation",
                "tributo.official.multistage_torch.distillation",
            ),
            "temporal_conv_classifier": OfficialAlgorithmIdentity(
                "tributo-algorithms-timeseries",
                "temporal_conv_classifier",
                "tributo.official.timeseries.temporal_conv",
            ),
            "token_transformer_classifier": OfficialAlgorithmIdentity(
                "tributo-algorithms-transformers-nlp",
                "token_transformer_classifier",
                "tributo.official.transformer.token_classifier",
            ),
            "two_tower_recommender": OfficialAlgorithmIdentity(
                "tributo-algorithms-recsys-torch",
                "two_tower_recommender",
                "tributo.official.recsys_torch.two_tower",
            ),
            "x_learner.framework_native": OfficialAlgorithmIdentity(
                "tributo-algorithms-causal-xlearner",
                "x_learner",
                "tributo.official.causal_xlearner.xgboost",
            ),
            "xgboost.framework_native": OfficialAlgorithmIdentity(
                "tributo-algorithms-boosting",
                "xgboost",
                "tributo.official.boosting.xgboost",
            ),
        }
    )
)


def _build_distribution_entry_points() -> Mapping[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for entry_point, identity in OFFICIAL_ALGORITHM_IDENTITIES.items():
        grouped.setdefault(identity.distribution, []).append(entry_point)
    return MappingProxyType(
        {
            distribution: tuple(sorted(entry_points))
            for distribution, entry_points in sorted(grouped.items())
        }
    )


DISTRIBUTION_ENTRY_POINTS = _build_distribution_entry_points()
ENTRY_POINT_DISTRIBUTIONS: Mapping[str, str] = MappingProxyType(
    {
        entry_point: identity.distribution
        for entry_point, identity in OFFICIAL_ALGORITHM_IDENTITIES.items()
    }
)

CATEGORY_ENTRY_POINTS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
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
)

ALL_ENTRY_POINTS = frozenset(ENTRY_POINT_DISTRIBUTIONS)


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


def _build_implementation_entry_points() -> Mapping[str, str]:
    entry_points: dict[str, str] = {}
    for entry_point, identity in OFFICIAL_ALGORITHM_IDENTITIES.items():
        if identity.implementation_id in entry_points:
            raise RuntimeError(
                "official implementation identity has duplicate Entry Points: "
                f"{identity.implementation_id!r}"
            )
        entry_points[identity.implementation_id] = entry_point
    return MappingProxyType(entry_points)


_IMPLEMENTATION_ENTRY_POINTS = _build_implementation_entry_points()


def _build_unambiguous_algorithm_entry_points() -> Mapping[str, str]:
    candidates: dict[str, list[str]] = {}
    for entry_point, identity in OFFICIAL_ALGORITHM_IDENTITIES.items():
        candidates.setdefault(identity.algorithm_id, []).append(entry_point)
    return MappingProxyType(
        {
            algorithm_id: entry_points[0]
            for algorithm_id, entry_points in candidates.items()
            if len(entry_points) == 1
        }
    )


_ALGORITHM_ENTRY_POINTS = _build_unambiguous_algorithm_entry_points()


def entry_point_for(algorithm: str, implementation_id: str | None) -> str:
    """Resolve one execution record to its installed Entry Point identity."""
    if implementation_id is not None:
        try:
            resolved = _IMPLEMENTATION_ENTRY_POINTS[implementation_id]
        except KeyError as exc:
            raise KeyError(
                f"official Gate has no Entry Point identity for {algorithm!r}/"
                f"{implementation_id!r}"
            ) from exc
        if OFFICIAL_ALGORITHM_IDENTITIES[resolved].algorithm_id != algorithm:
            raise KeyError(
                f"official Gate identity mismatch for {algorithm!r}/"
                f"{implementation_id!r}"
            )
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
    "DISTRIBUTION_ENTRY_POINTS",
    "ENTRY_POINT_DISTRIBUTIONS",
    "OFFICIAL_ALGORITHM_IDENTITIES",
    "OfficialAlgorithmIdentity",
    "category_for_entry_point",
    "entry_point_for",
    "entry_points_for_gate",
    "parse_entry_point_selection",
]
