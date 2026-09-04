"""Immutable category and identity matrix for the official algorithm Gate."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True)
class OfficialAlgorithmIdentity:
    """Immutable installed identity expected for one official Entry Point."""

    distribution: str
    algorithm_id: str
    implementation_id: str


def _load_official_algorithm_identities() -> Mapping[str, OfficialAlgorithmIdentity]:
    path = Path(__file__).with_name("official_algorithm_identities.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "official algorithm identity manifest is unavailable"
        ) from exc
    entries = payload.get("entry_points") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or not isinstance(entries, Mapping)
        or len(entries) != 37
    ):
        raise RuntimeError("official algorithm identity manifest is malformed")
    identities: dict[str, OfficialAlgorithmIdentity] = {}
    required = {"distribution", "algorithm_id", "implementation_id"}
    for entry_point, value in entries.items():
        if (
            not isinstance(entry_point, str)
            or not entry_point
            or not isinstance(value, Mapping)
            or set(value) != required
            or any(
                not isinstance(value[name], str) or not value[name] for name in required
            )
        ):
            raise RuntimeError(
                "official algorithm identity manifest entry is malformed"
            )
        identities[entry_point] = OfficialAlgorithmIdentity(
            distribution=value["distribution"],
            algorithm_id=value["algorithm_id"],
            implementation_id=value["implementation_id"],
        )
    return MappingProxyType(identities)


OFFICIAL_ALGORITHM_IDENTITIES = _load_official_algorithm_identities()


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
            "dnn",
            "gru_classifier",
            "lstm_classifier",
            "pu",
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
