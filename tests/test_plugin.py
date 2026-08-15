"""Fail-closed tests for explicitly requested Hook plugins."""

from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, ConfigDict

import tests.conftest as pytest_config
import tributo.plugin as plugin
from tributo.exceptions import JobConfigurationError
from tributo.exporting.dispatch import InlineHookDispatcher
from tributo.exporting.models import HookBinding


class _Options(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: int


class _Hook:
    api_version: ClassVar[int] = 1
    hook_id: ClassVar[str] = "hook-v1"
    options_model: ClassVar[type[BaseModel]] = _Options

    def deliver(self, *args: Any) -> Any:
        return None

    def idempotency_key(self, *args: Any) -> str:
        return "key"


class _EntryPoint:
    name = "hook-v1"
    value = "tests:_Hook"

    def __init__(self, loaded: Any = _Hook) -> None:
        self.loaded = loaded

    def load(self) -> Any:
        if isinstance(self.loaded, Exception):
            raise self.loaded
        return self.loaded


class _TrainerEntryPoint:
    def __init__(self, name: str, value: str, loaded: Any) -> None:
        self.name = name
        self.value = value
        self.loaded = loaded
        self.load_calls = 0

    def load(self) -> Any:
        self.load_calls += 1
        if isinstance(self.loaded, Exception):
            raise self.loaded
        return self.loaded


def _set_entry_points(monkeypatch: pytest.MonkeyPatch, *eps: _EntryPoint) -> None:
    monkeypatch.setattr(plugin, "_iter_entry_points", lambda group: iter(eps))


def _set_trainer_entry_points(
    monkeypatch: pytest.MonkeyPatch,
    *eps: _TrainerEntryPoint,
) -> None:
    monkeypatch.setattr(plugin, "_iter_trainer_entry_points", lambda: iter(eps))


def test_resolves_only_exact_requested_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_entry_points(monkeypatch, _EntryPoint())
    assert plugin.resolve_hook_plugin("hook-v1") is _Hook


def test_unknown_disabled_and_load_failure_are_configuration_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_entry_points(monkeypatch)
    with pytest.raises(JobConfigurationError, match="Unknown hook_id"):
        plugin.resolve_hook_plugin("missing")

    _set_entry_points(monkeypatch, _EntryPoint())
    monkeypatch.setenv("TRIBUTO_PLUGINS", "another-hook")
    with pytest.raises(JobConfigurationError, match="disabled"):
        plugin.resolve_hook_plugin("hook-v1")

    monkeypatch.delenv("TRIBUTO_PLUGINS")
    _set_entry_points(monkeypatch, _EntryPoint(ImportError("secret-extra-path")))
    with pytest.raises(JobConfigurationError, match="Failed to load") as exc_info:
        plugin.resolve_hook_plugin("hook-v1")
    assert "secret-extra-path" not in str(exc_info.value)


def test_entry_point_name_and_api_version_are_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _WrongName(_Hook):
        hook_id = "different-v1"

    _set_entry_points(monkeypatch, _EntryPoint(_WrongName))
    with pytest.raises(JobConfigurationError, match="does not match"):
        plugin.resolve_hook_plugin("hook-v1")

    class _WrongVersion(_Hook):
        api_version = 2

    _set_entry_points(monkeypatch, _EntryPoint(_WrongVersion))
    with pytest.raises(JobConfigurationError, match="unsupported api_version"):
        plugin.resolve_hook_plugin("hook-v1")

    class _InvalidOptionsModel(_Hook):
        options_model = object

    _set_entry_points(monkeypatch, _EntryPoint(_InvalidOptionsModel))
    with pytest.raises(JobConfigurationError, match="does not implement"):
        plugin.resolve_hook_plugin("hook-v1")


def test_legacy_hook_error_describes_the_v1_migration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _LegacyHook:
        api_version = 1
        hook_id = "hook-v1"
        options_model = _Options

        def execute(self, *args: Any) -> Any:
            return None

    _set_entry_points(monkeypatch, _EntryPoint(_LegacyHook))

    with pytest.raises(JobConfigurationError) as exc_info:
        plugin.resolve_hook_plugin("hook-v1")

    message = str(exc_info.value)
    assert "hook-v1" in message
    assert "deliver(event, artifacts, options)" in message
    assert "idempotency_key(event, options)" in message
    assert "legacy execute(" in message


def test_dispatcher_validates_options_during_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tributo.exporting.dispatch.resolve_hook_plugin", lambda hook_id: _Hook
    )
    with pytest.raises(JobConfigurationError, match="Invalid options"):
        InlineHookDispatcher().preflight(
            (HookBinding(hook_id="hook-v1", options={"unknown": 1}),)
        )


def test_dispatcher_reports_constructor_failure_as_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ConstructorFailure(_Hook):
        def __init__(self) -> None:
            raise RuntimeError("secret-constructor-value")

    monkeypatch.setattr(
        "tributo.exporting.dispatch.resolve_hook_plugin",
        lambda hook_id: _ConstructorFailure,
    )
    with pytest.raises(JobConfigurationError, match="Failed to initialize") as exc_info:
        InlineHookDispatcher().preflight(
            (HookBinding(hook_id="hook-v1", options={"value": 1}),)
        )
    assert "secret-constructor-value" not in str(exc_info.value)


def test_mlflow_integration_selection_requires_explicit_opt_in() -> None:
    integrations_dir = pytest_config._TESTS_DIR / "integrations"
    mlflow_test = integrations_dir / "test_e2e_mlflow.py"

    assert not pytest_config._mlflow_integration_requested([str(integrations_dir)])
    assert pytest_config._mlflow_integration_requested([str(mlflow_test)])
    assert pytest_config._mlflow_integration_requested(["-m=integration"])
    assert pytest_config._mlflow_integration_requested(
        ["-m", "integration and not slow"]
    )
    assert not pytest_config._mlflow_integration_requested(
        [str(mlflow_test), "-m", "not integration"]
    )
    assert not pytest_config._mlflow_integration_requested(
        ["-m", "not (integration or slow)"]
    )


def test_trainer_descriptor_discovery_keeps_old_plugins_compatibility_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tributo.integrations.algorithm_runtimes.legacy_descriptors import (
        DNN_DESCRIPTOR,
    )

    descriptor_ep = _TrainerEntryPoint(
        "dnn",
        "package.descriptors:DNN_DESCRIPTOR",
        DNN_DESCRIPTOR,
    )
    legacy_ep = _TrainerEntryPoint(
        "legacy",
        "package.legacy_trainer",
        RuntimeError("must not load"),
    )
    _set_trainer_entry_points(monkeypatch, descriptor_ep, legacy_ep)
    diagnostics = []
    compatibility = []

    descriptors = plugin.discover_trainer_descriptors(
        diagnostics=diagnostics,
        compatibility_entry_points=compatibility,
    )

    assert descriptors == [DNN_DESCRIPTOR]
    assert compatibility == [legacy_ep]
    assert descriptor_ep.load_calls == 1
    assert legacy_ep.load_calls == 0
    assert diagnostics[0].entry_point_name == "legacy"
    assert "compatibility-only" in diagnostics[0].reason


def test_trainer_descriptor_import_failure_is_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failing = _TrainerEntryPoint(
        "broken",
        "package.descriptors:BROKEN",
        ImportError("optional-secret-path"),
    )
    _set_trainer_entry_points(monkeypatch, failing)
    diagnostics = []

    assert plugin.discover_trainer_descriptors(diagnostics=diagnostics) == []
    assert diagnostics[0].entry_point_name == "broken"
    assert diagnostics[0].error_type == "ImportError"
    assert "optional-secret-path" not in diagnostics[0].reason


def test_trainer_descriptor_identity_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tributo.integrations.algorithm_runtimes.legacy_descriptors import (
        DNN_DESCRIPTOR,
    )

    mismatched = _TrainerEntryPoint(
        "different-name",
        "package.descriptors:DNN_DESCRIPTOR",
        DNN_DESCRIPTOR,
    )
    _set_trainer_entry_points(monkeypatch, mismatched)
    diagnostics = []

    with pytest.raises(JobConfigurationError, match="descriptor identity 'dnn'"):
        plugin.discover_trainer_descriptors(diagnostics=diagnostics)

    assert diagnostics[0].entry_point_name == "different-name"
    assert "identity" in diagnostics[0].reason


def test_entry_points_are_sorted_by_distribution_name_and_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = [
        _TrainerEntryPoint("same", "package:z", object()),
        _TrainerEntryPoint("beta", "package:b", object()),
        _TrainerEntryPoint("same", "package:a", object()),
        _TrainerEntryPoint("alpha", "package:c", object()),
    ]
    distribution_names = ("zeta", "alpha", "zeta", "zeta")
    for entry, distribution_name in zip(
        entries,
        distribution_names,
        strict=True,
    ):
        entry.dist = type("Distribution", (), {"name": distribution_name})()
    monkeypatch.setattr(plugin, "entry_points", lambda **_kwargs: entries)

    ordered = list(plugin._iter_trainer_entry_points())

    assert [(item.dist.name, item.name, item.value) for item in ordered] == [
        ("alpha", "beta", "package:b"),
        ("zeta", "alpha", "package:c"),
        ("zeta", "same", "package:a"),
        ("zeta", "same", "package:z"),
    ]


def test_non_trainer_entry_points_keep_name_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = [
        _TrainerEntryPoint("zeta", "package:z", object()),
        _TrainerEntryPoint("alpha", "package:a", object()),
    ]
    monkeypatch.setattr(plugin, "entry_points", lambda **_kwargs: entries)

    ordered = list(plugin._iter_entry_points("tributo.exporters"))

    assert [item.name for item in ordered] == ["alpha", "zeta"]


def test_independent_distributed_algorithm_package_passes_conformance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_src = (
        Path(__file__).parent / "fixtures" / "distributed_algorithm_plugin" / "src"
    )
    monkeypatch.syspath_prepend(str(fixture_src))
    fixture = importlib.import_module("tributo_test_distributed_algorithm")
    entry_point = _TrainerEntryPoint(
        "third_party_mean_regressor",
        "tributo_test_distributed_algorithm:DESCRIPTOR",
        fixture.DESCRIPTOR,
    )
    _set_entry_points(monkeypatch, entry_point)
    real_version = importlib.metadata.version

    def package_version(name: str) -> str:
        if name == "tributo-test-distributed-algorithm":
            return "0.1.0"
        return real_version(name)

    monkeypatch.setattr(importlib.metadata, "version", package_version)
    diagnostics = []

    descriptors = plugin.discover_algorithm_descriptors(diagnostics=diagnostics)
    validated = plugin.validate_distributed_algorithm_descriptor(
        fixture.DESCRIPTOR,
        entry_point_name="third_party_mean_regressor",
    )

    assert descriptors == [fixture.DESCRIPTOR]
    assert validated is fixture.DESCRIPTOR
    assert diagnostics == []
    assert fixture.DESCRIPTOR.registration.is_default is True
    assert fixture.DESCRIPTOR.registration.distribution_spec is not None
    assert (
        fixture.DESCRIPTOR.registration.distribution_spec.strategy.value
        == "ray_map_reduce"
    )
    assert (
        fixture.DESCRIPTOR.registration.distribution_spec.result_policy.value
        == "fit_only"
    )
    assert fixture.DESCRIPTOR.registration.implementation.exporter_ref is None
    assert fixture.DESCRIPTOR.registration.implementation.flavor_id is None


def test_independent_fixture_single_and_multi_partition_results_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from tributo.algorithms.spi import AlgorithmExecutionContext

    fixture_src = (
        Path(__file__).parent / "fixtures" / "distributed_algorithm_plugin" / "src"
    )
    monkeypatch.syspath_prepend(str(fixture_src))
    fixture = importlib.import_module("tributo_test_distributed_algorithm")
    plan = SimpleNamespace(
        input_binding=SimpleNamespace(
            feature_names=("f0", "f1"),
            label_name="label",
        )
    )
    algorithm = fixture.ThirdPartyMeanRegressor(plan)
    batch = {
        "f0": np.asarray([1.0, 2.0, 3.0, 4.0]),
        "f1": np.asarray([2.0, 4.0, 6.0, 8.0]),
        "label": np.asarray([0.0, 1.0, 1.0, 2.0]),
    }
    context = AlgorithmExecutionContext(inputs={})

    single = algorithm.finalize_model(algorithm.map_partition((batch,), context))
    left = algorithm.map_partition(
        ({name: values[:2] for name, values in batch.items()},),
        context,
    )
    right = algorithm.map_partition(
        ({name: values[2:] for name, values in batch.items()},),
        context,
    )
    multi = algorithm.finalize_model(algorithm.merge_states(left, right))

    assert single == multi
    assert single.feature_means == (2.5, 5.0)
    assert single.target_mean == 1.0
    assert single.row_count == 4


def test_direct_conformance_accepts_every_first_party_formal_descriptor() -> None:
    from tributo.algorithms.builtin import (
        DNN_DESCRIPTOR,
        MULTINOMIAL_NB_DESCRIPTOR,
        PU_DESCRIPTOR,
        XGBOOST_DESCRIPTOR,
    )

    descriptors = (
        DNN_DESCRIPTOR,
        PU_DESCRIPTOR,
        XGBOOST_DESCRIPTOR,
        MULTINOMIAL_NB_DESCRIPTOR,
    )

    assert [
        plugin.validate_distributed_algorithm_descriptor(
            descriptor,
            entry_point_name=descriptor.name,
        )
        for descriptor in descriptors
    ] == list(descriptors)


def test_distributed_algorithm_discovery_rejects_wrong_strategy_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tributo.algorithms.api import QualifiedReference
    from tributo.algorithms.builtin import DNN_DESCRIPTOR

    invalid_implementation = replace(
        DNN_DESCRIPTOR.registration.implementation,
        implementation_ref=QualifiedReference.parse("tests.test_plugin:_Hook"),
    )
    invalid_descriptor = replace(
        DNN_DESCRIPTOR,
        registration=replace(
            DNN_DESCRIPTOR.registration,
            implementation=invalid_implementation,
        ),
    )
    _set_entry_points(
        monkeypatch,
        _TrainerEntryPoint(
            "dnn",
            "tests.test_plugin:INVALID_DESCRIPTOR",
            invalid_descriptor,
        ),
    )
    diagnostics = []

    with pytest.raises(TypeError, match="CollectiveAlgorithm"):
        plugin.validate_distributed_algorithm_descriptor(
            invalid_descriptor,
            entry_point_name="dnn",
        )
    assert plugin.discover_algorithm_descriptors(diagnostics=diagnostics) == []
    assert diagnostics[0].entry_point_name == "dnn"
    assert diagnostics[0].error_type == "TypeError"


def test_distributed_algorithm_discovery_rejects_package_version_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tributo.algorithms.builtin import DNN_DESCRIPTOR

    _set_entry_points(
        monkeypatch,
        _TrainerEntryPoint(
            "dnn",
            "tributo.algorithms.builtin.torch_collective:DNN_DESCRIPTOR",
            DNN_DESCRIPTOR,
        ),
    )
    real_version = importlib.metadata.version

    def package_version(name: str) -> str:
        return "1.0.1" if name == "tributo" else real_version(name)

    monkeypatch.setattr(importlib.metadata, "version", package_version)
    diagnostics = []

    assert plugin.discover_algorithm_descriptors(diagnostics=diagnostics) == []
    assert diagnostics[0].error_type == "ValueError"
