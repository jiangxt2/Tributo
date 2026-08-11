"""Conformance tests for Registry-derived algorithm support snapshots."""

from __future__ import annotations

from tributo._common.registry import Registry
from tributo.training.algorithm_spec import (
    AlgorithmSpec,
    Capability,
    DataLoadingMode,
    ExecutionKind,
    ProblemType,
)
from tributo.training.catalog import AlgorithmCatalog
from tributo.training.support_snapshot import (
    build_algorithm_support_snapshot,
    snapshot_json_objects,
)


class FakeTrainer:
    """Minimal trainer identity for immutable catalog fixtures."""


def _catalog(*specs: AlgorithmSpec) -> AlgorithmCatalog:
    registry: Registry[str, AlgorithmSpec] = Registry(name="test algorithm")
    for spec in specs:
        registry.register(spec.name, spec)
    return AlgorithmCatalog(registry)


def test_snapshot_is_name_sorted_and_preserves_declared_value_order() -> None:
    catalog = _catalog(
        AlgorithmSpec(name="zeta", trainer_cls=FakeTrainer),
        AlgorithmSpec(
            name="alpha",
            trainer_cls=FakeTrainer,
            problem_types=(ProblemType.REGRESSION,),
            supported_tasks=("fit", "predict"),
            capabilities=(Capability.TUNABLE, Capability.EXPORTABLE),
            data_loading=DataLoadingMode.CANONICAL_DRIVER,
        ),
    )

    snapshot = build_algorithm_support_snapshot(catalog.list_specs())

    assert [record.name for record in snapshot] == ["alpha", "zeta"]
    assert snapshot[0].supported_tasks == ("fit", "predict")
    assert snapshot[0].capabilities == ("tunable", "exportable")


def test_snapshot_json_projection_contains_every_cli_support_field() -> None:
    catalog = _catalog(
        AlgorithmSpec(
            name="causal",
            trainer_cls=FakeTrainer,
            problem_types=(ProblemType.CAUSAL_EFFECT_ESTIMATION,),
            data_modality=("tabular",),
            tags=("causal",),
            execution_kind=ExecutionKind.ESTIMATE,
            supported_tasks=("estimate",),
            capabilities=(Capability.EXPORTABLE,),
            data_loading=DataLoadingMode.LEGACY_DRIVER,
            extras_group="causal",
        )
    )

    assert snapshot_json_objects(
        build_algorithm_support_snapshot(catalog.list_specs())
    ) == [
        {
            "name": "causal",
            "problem_types": ["causal_effect_estimation"],
            "data_modality": ["tabular"],
            "tags": ["causal"],
            "execution_kind": "estimate",
            "supported_tasks": ["estimate"],
            "capabilities": ["exportable"],
            "data_loading": "legacy_driver",
            "gpu_required": False,
            "status": "ready",
            "extras_group": "causal",
            "implementation_ids": [],
            "runtime_topologies": [],
            "input_views": [],
            "stability": "beta",
            "limitations": [],
            "available": True,
            "compatibility_only": False,
            "tested": False,
            "supported": False,
            "native_migration_complete": False,
        }
    ]


def test_unified_snapshot_does_not_upgrade_legacy_adapters_to_supported() -> None:
    from tributo.training.catalog import get_algorithm_catalog

    snapshot = build_algorithm_support_snapshot(get_algorithm_catalog().list_records())
    by_name = {record.name: record for record in snapshot}

    assert set(by_name) >= {"dnn", "pu", "xgboost"}
    assert all(by_name[name].available for name in ("dnn", "pu", "xgboost"))
    assert all(
        not by_name[name].compatibility_only for name in ("dnn", "pu", "xgboost")
    )
    assert all(not by_name[name].tested for name in ("dnn", "pu", "xgboost"))
    assert all(not by_name[name].supported for name in ("dnn", "pu", "xgboost"))
    assert all(by_name[name].stability == "beta" for name in ("dnn", "pu", "xgboost"))
    assert all(
        not by_name[name].native_migration_complete for name in ("dnn", "pu", "xgboost")
    )
    assert "canonical_trainer" in " ".join(by_name["pu"].limitations)
