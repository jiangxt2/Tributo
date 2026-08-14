"""Tests for AlgorithmCatalog — filtering, lifecycle, integrity validation."""

from __future__ import annotations

import pytest

from tributo._common.registry import Registry
from tributo.exceptions import JobConfigurationError
from tributo.training.algorithm_spec import (
    AlgorithmSpec,
    AlgorithmStatus,
    Capability,
    DataLoadingMode,
    ProblemFamily,
    ProblemType,
    ResourceHints,
)
from tributo.training.catalog import AlgorithmCatalog, get_algorithm_catalog


class FakeTrainer:
    pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _spec(
    name: str,
    *,
    problem_types: tuple[ProblemType, ...] = (),
    data_modality: tuple[str, ...] = ("tabular",),
    tags: tuple[str, ...] = (),
    extras_group: str | None = None,
    status: AlgorithmStatus = AlgorithmStatus.READY,
    deprecated_since: str | None = None,
    replacement: str | None = None,
    gpu_required: bool = False,
    data_loading: DataLoadingMode = DataLoadingMode.LEGACY_DRIVER,
    config_model: type | None = None,
) -> AlgorithmSpec:
    return AlgorithmSpec(
        name=name,
        trainer_cls=FakeTrainer,
        problem_types=problem_types,
        data_modality=data_modality,
        tags=tags,
        extras_group=extras_group,
        status=status,
        deprecated_since=deprecated_since,
        replacement=replacement,
        resource_hints=ResourceHints(gpu_required=gpu_required),
        data_loading=data_loading,
        config_model=config_model,
    )


def _make_registry(**specs: AlgorithmSpec) -> Registry[str, AlgorithmSpec]:
    r: Registry[str, AlgorithmSpec] = Registry(name="test")
    for s in specs.values():
        r.register(s.name, s)
    return r


def _catalog(**specs: AlgorithmSpec) -> AlgorithmCatalog:
    return AlgorithmCatalog(_make_registry(**specs))


# ---------------------------------------------------------------------------
# list() — filtering
# ---------------------------------------------------------------------------


class TestCatalogList:
    def test_generic_registry_records_are_beta_compatibility_only(self) -> None:
        record = _catalog(example=_spec("example")).get_record("example")

        assert record.stability == "beta"
        assert record.available is False
        assert record.compatibility_only is True
        assert record.implementation_ids == ()
        assert "no executable portable descriptor" in " ".join(record.limitations)

    def test_list_specs_reads_one_registry_snapshot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry = _make_registry(
            xgb=_spec("xgb"),
            dnn=_spec("dnn"),
        )
        original_snapshot = registry.snapshot
        snapshot_calls = 0

        def counting_snapshot() -> dict[str, AlgorithmSpec]:
            nonlocal snapshot_calls
            snapshot_calls += 1
            return original_snapshot()

        monkeypatch.setattr(registry, "snapshot", counting_snapshot)

        assert [spec.name for spec in AlgorithmCatalog(registry).list_specs()] == [
            "xgb",
            "dnn",
        ]
        assert snapshot_calls == 1

    def test_empty_registry(self) -> None:
        cat = _catalog()
        assert cat.list() == []

    def test_all_ready(self) -> None:
        cat = _catalog(
            xgb=_spec("xgb", problem_types=(ProblemType.BINARY_CLASSIFICATION,)),
            dnn=_spec("dnn", problem_types=(ProblemType.BINARY_CLASSIFICATION,)),
        )
        assert sorted(cat.list()) == ["dnn", "xgb"]

    def test_filter_by_problem_type(self) -> None:
        cat = _catalog(
            xgb=_spec("xgb", problem_types=(ProblemType.BINARY_CLASSIFICATION,)),
            reg=_spec("reg", problem_types=(ProblemType.REGRESSION,)),
        )
        assert cat.list(problem_type=ProblemType.REGRESSION) == ["reg"]

    def test_filter_by_problem_family(self) -> None:
        cat = _catalog(
            bin_cls=_spec(
                "bin_cls", problem_types=(ProblemType.BINARY_CLASSIFICATION,)
            ),
            multi_cls=_spec(
                "multi_cls", problem_types=(ProblemType.MULTI_CLASS_CLASSIFICATION,)
            ),
            reg=_spec("reg", problem_types=(ProblemType.REGRESSION,)),
        )
        result = cat.list(problem_family=ProblemFamily.CLASSIFICATION)
        assert sorted(result) == ["bin_cls", "multi_cls"]

    def test_filter_by_modality(self) -> None:
        cat = _catalog(
            tab=_spec("tab", data_modality=("tabular",)),
            graph=_spec("g", data_modality=("graph",)),
        )
        assert cat.list(modality="graph") == ["g"]

    def test_filter_by_tag(self) -> None:
        cat = _catalog(
            ctr=_spec("ctr", tags=("deep-learning", "ctr")),
            trad=_spec("trad", tags=("classical",)),
        )
        assert cat.list(tag="ctr") == ["ctr"]

    def test_filter_by_capabilities(self) -> None:
        tunable = AlgorithmSpec(
            name="tunable-alg",
            trainer_cls=FakeTrainer,
            capabilities=(Capability.TUNABLE, Capability.EXPORTABLE),
        )
        plain = AlgorithmSpec(name="plain-alg", trainer_cls=FakeTrainer)
        cat = _catalog(tunable=tunable, plain=plain)
        # Enum and plain-string forms both filter on the declared set.
        assert cat.list(capabilities=Capability.TUNABLE) == ["tunable-alg"]
        assert cat.list(capabilities="tunable") == ["tunable-alg"]
        assert cat.list(capabilities=(Capability.TUNABLE, Capability.EXPORTABLE)) == [
            "tunable-alg"
        ]

    def test_filter_by_extras_group(self) -> None:
        cat = _catalog(
            a=_spec("a", extras_group="training"),
            b=_spec("b", extras_group="identity"),
        )
        assert cat.list(extras_group="identity") == ["b"]

    def test_conjunction_all_filters(self) -> None:
        cat = _catalog(
            xgb=_spec(
                "xgb",
                problem_types=(ProblemType.BINARY_CLASSIFICATION,),
                data_modality=("tabular",),
                tags=("deep-learning",),
                extras_group="training",
            ),
            other=_spec(
                "other",
                problem_types=(ProblemType.REGRESSION,),
                data_modality=("tabular",),
                tags=("deep-learning",),
                extras_group="training",
            ),
        )
        result = cat.list(
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            modality="tabular",
            tag="deep-learning",
            extras_group="training",
        )
        assert result == ["xgb"]

    def test_deprecated_excluded_by_default(self) -> None:
        cat = _catalog(
            v1=_spec(
                "v1",
                status=AlgorithmStatus.DEPRECATED,
                deprecated_since="1.0",
                replacement="v2",
            ),
            v2=_spec("v2"),
        )
        assert cat.list() == ["v2"]

    def test_include_deprecated(self) -> None:
        cat = _catalog(
            v1=_spec(
                "v1",
                status=AlgorithmStatus.DEPRECATED,
                deprecated_since="1.0",
                replacement="v2",
            ),
            v2=_spec("v2"),
        )
        result = cat.list(include_deprecated=True)
        assert sorted(result) == ["v1", "v2"]


# ---------------------------------------------------------------------------
# get_spec() — lifecycle
# ---------------------------------------------------------------------------


class TestCatalogGetSpec:
    def test_known_algorithm(self) -> None:
        cat = _catalog(xgb=_spec("xgb"))
        spec = cat.get_spec("xgb")
        assert spec.name == "xgb"

    def test_unknown_algorithm_raises(self) -> None:
        cat = _catalog()
        with pytest.raises(JobConfigurationError, match="Unknown algorithm"):
            cat.get_spec("nonexistent")

    def test_deprecated_emits_future_warning(self) -> None:
        cat = _catalog(
            old=_spec(
                "old",
                status=AlgorithmStatus.DEPRECATED,
                deprecated_since="0.4.0",
                replacement="new",
            ),
            new=_spec("new"),
        )
        with pytest.warns(FutureWarning, match="deprecated since 0.4.0"):
            spec = cat.get_spec("old")
        assert spec.name == "old"

    def test_unknown_error_message_lists_available(self) -> None:
        cat = _catalog(xgb=_spec("xgb"), dnn=_spec("dnn"))
        with pytest.raises(JobConfigurationError, match="Available:"):
            cat.get_spec("unknown")


# ---------------------------------------------------------------------------
# supports_* convenience methods
# ---------------------------------------------------------------------------


class TestCatalogSupports:
    def test_supports_classification(self) -> None:
        cat = _catalog(
            xgb=_spec("xgb", problem_types=(ProblemType.BINARY_CLASSIFICATION,)),
        )
        assert cat.supports_classification("xgb") is True

    def test_does_not_support_classification(self) -> None:
        cat = _catalog(
            reg=_spec("reg", problem_types=(ProblemType.REGRESSION,)),
        )
        assert cat.supports_classification("reg") is False

    def test_supports_regression(self) -> None:
        cat = _catalog(
            reg=_spec("reg", problem_types=(ProblemType.REGRESSION,)),
        )
        assert cat.supports_regression("reg") is True

    def test_requires_gpu(self) -> None:
        cat = _catalog(
            deep=_spec("deep", gpu_required=True),
            classic=_spec("classic", gpu_required=False),
        )
        assert cat.requires_gpu("deep") is True
        assert cat.requires_gpu("classic") is False


# ---------------------------------------------------------------------------
# get_config_schema
# ---------------------------------------------------------------------------


class TestCatalogGetConfigSchema:
    def test_returns_json_schema(self) -> None:
        from tributo.training.xgboost_trainer import XGBoostTrainingConfig

        cat = _catalog(
            xgb=_spec("xgb", config_model=XGBoostTrainingConfig),
        )
        schema = cat.get_config_schema("xgb")
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "data" in schema["properties"]

    def test_no_config_model_raises(self) -> None:
        cat = _catalog(xgb=_spec("xgb"))
        with pytest.raises(
            JobConfigurationError, match="does not declare a config_model"
        ):
            cat.get_config_schema("xgb")


# ---------------------------------------------------------------------------
# validate_integrity — replacement graph
# ---------------------------------------------------------------------------


class TestValidateIntegrity:
    def test_all_ready_passes(self) -> None:
        cat = _catalog(a=_spec("a"), b=_spec("b"))
        cat.validate_integrity()  # does not raise

    def test_one_level_deprecation_passes(self) -> None:
        cat = _catalog(
            old=_spec(
                "old",
                status=AlgorithmStatus.DEPRECATED,
                deprecated_since="1.0",
                replacement="new",
            ),
            new=_spec("new"),
        )
        cat.validate_integrity()

    def test_chain_to_ready_passes(self) -> None:
        cat = _catalog(
            v1=_spec(
                "v1",
                status=AlgorithmStatus.DEPRECATED,
                deprecated_since="1.0",
                replacement="v2",
            ),
            v2=_spec(
                "v2",
                status=AlgorithmStatus.DEPRECATED,
                deprecated_since="2.0",
                replacement="v3",
            ),
            v3=_spec("v3"),
        )
        cat.validate_integrity()

    def test_missing_replacement_raises(self) -> None:
        cat = _catalog(
            old=_spec(
                "old",
                status=AlgorithmStatus.DEPRECATED,
                deprecated_since="1.0",
                replacement="nonexistent",
            ),
        )
        with pytest.raises(JobConfigurationError, match="not found in registry"):
            cat.validate_integrity()

    def test_direct_cycle_raises(self) -> None:
        cat = _catalog(
            a=_spec(
                "a",
                status=AlgorithmStatus.DEPRECATED,
                deprecated_since="1.0",
                replacement="b",
            ),
            b=_spec(
                "b",
                status=AlgorithmStatus.DEPRECATED,
                deprecated_since="1.0",
                replacement="a",
            ),
        )
        with pytest.raises(JobConfigurationError, match="cycle"):
            cat.validate_integrity()

    def test_indirect_cycle_raises(self) -> None:
        cat = _catalog(
            a=_spec(
                "a",
                status=AlgorithmStatus.DEPRECATED,
                deprecated_since="1.0",
                replacement="b",
            ),
            b=_spec(
                "b",
                status=AlgorithmStatus.DEPRECATED,
                deprecated_since="1.0",
                replacement="c",
            ),
            c=_spec(
                "c",
                status=AlgorithmStatus.DEPRECATED,
                deprecated_since="1.0",
                replacement="a",
            ),
        )
        with pytest.raises(JobConfigurationError, match="cycle"):
            cat.validate_integrity()

    def test_integrity_called_on_list(self) -> None:
        """list() also calls _validate_integrity internally."""
        cat = _catalog(
            old=_spec(
                "old",
                status=AlgorithmStatus.DEPRECATED,
                deprecated_since="1.0",
                replacement="missing",
            ),
        )
        with pytest.raises(JobConfigurationError):
            cat.list()

    def test_integrity_called_on_get_spec(self) -> None:
        """get_spec() also calls _validate_integrity."""
        cat = _catalog(
            a=_spec(
                "a",
                status=AlgorithmStatus.DEPRECATED,
                deprecated_since="1.0",
                replacement="missing",
            ),
        )
        with pytest.raises(JobConfigurationError):
            cat.get_spec("a")


# ---------------------------------------------------------------------------
# get_algorithm_catalog factory
# ---------------------------------------------------------------------------


class TestGetAlgorithmCatalog:
    def test_returns_catalog(self) -> None:
        cat = get_algorithm_catalog()
        assert isinstance(cat, AlgorithmCatalog)

    def test_includes_registered_trainers(self) -> None:
        cat = get_algorithm_catalog()
        names = cat.list()
        # First-party descriptors are published by the unified Registry bootstrap.
        assert "xgboost" in names

    def test_unified_records_expose_execution_and_support_state(self) -> None:
        record = get_algorithm_catalog().get_record("xgboost")

        assert record.available is True
        assert record.compatibility_only is False
        assert record.tested is True
        assert record.supported is True
        assert record.native_migration_complete is True
        assert record.implementation_ids == (
            "tributo.xgboost.framework_native",
            "tributo.xgboost.legacy_trainer",
        )
        assert record.runtime_topologies == (
            "framework_managed",
            "framework_native",
        )
        assert record.distribution_strategies == ("framework_native",)
        assert record.execution_profiles == ("kubernetes", "local")
        assert record.input_views == ("ray_data",)
        assert record.stability == "alpha"

    def test_explicit_compatibility_access_reuses_hydrated_spec(self) -> None:
        catalog = get_algorithm_catalog()

        assert catalog.get_spec("xgboost") is catalog.get_spec("xgboost")

    def test_programmatic_beta_registration_is_compatibility_only(self) -> None:
        from tributo.training.registry import _registry

        name = "tests.compatibility-only"
        _registry.register(name, _spec(name))
        try:
            record = get_algorithm_catalog().get_record(name)
            assert record.available is False
            assert record.compatibility_only is True
            assert record.implementation_ids == ()
            assert record.supported is False
        finally:
            _registry.unregister(name)


# ---------------------------------------------------------------------------
# ProblemFamily mapping is consistent with ProblemType
# ---------------------------------------------------------------------------


class TestProblemFamilyMapping:
    def test_classification_covers_all_three(self) -> None:
        from tributo.training.algorithm_spec import PROBLEM_FAMILY_MAP

        classification = PROBLEM_FAMILY_MAP[ProblemFamily.CLASSIFICATION]
        assert ProblemType.BINARY_CLASSIFICATION in classification
        assert ProblemType.MULTI_CLASS_CLASSIFICATION in classification
        assert ProblemType.MULTI_LABEL_CLASSIFICATION in classification

    def test_family_includes_multi_label(self) -> None:
        cat = _catalog(
            ml=_spec("ml", problem_types=(ProblemType.MULTI_LABEL_CLASSIFICATION,)),
        )
        assert "ml" in cat.list(problem_family=ProblemFamily.CLASSIFICATION)
