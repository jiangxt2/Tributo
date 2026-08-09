"""Fail-closed tests for explicitly requested Hook plugins."""

from __future__ import annotations

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


def _set_entry_points(monkeypatch: pytest.MonkeyPatch, *eps: _EntryPoint) -> None:
    monkeypatch.setattr(plugin, "_iter_entry_points", lambda group: iter(eps))


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
