"""Unit tests for explainability adapter discovery and isolation."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from tributo.explainability.protocols import ExplainerAdapter
from tributo.explainability.registry import ExplainerRegistry
from tributo.explainability.shap import ShapAdapter


def test_registry_registers_shap_without_importing_shap_runtime() -> None:
    registry = ExplainerRegistry()
    registry.register(ShapAdapter)
    assert registry.get("shap-v1") is ShapAdapter
    assert registry.list_all() == ("shap-v1",)


def test_registry_rejects_duplicate_adapter_id() -> None:
    registry = ExplainerRegistry()
    registry.register(ShapAdapter)
    try:
        registry.register(ShapAdapter)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate adapter registration must fail")


def test_disabled_config_does_not_import_shap() -> None:
    code = (
        "import sys; "
        "from tributo.explainability import ExplainabilityConfig; "
        "ExplainabilityConfig(); "
        "assert 'shap' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_registry_rejects_incomplete_adapter_contract() -> None:
    class IncompleteAdapter:
        api_version = 1
        adapter_id = "incomplete-v1"
        adapter_version = "1"

    registry = ExplainerRegistry()
    try:
        registry.register(cast(type[ExplainerAdapter], IncompleteAdapter))
    except ValueError as exc:
        assert "supports" in str(exc)
    else:
        raise AssertionError("incomplete adapter contract must fail closed")


def test_registry_entry_point_failure_is_diagnostic_and_non_global() -> None:
    registry = ExplainerRegistry()
    entry_point = SimpleNamespace(
        name="broken-explainer",
        load=lambda: (_ for _ in ()).throw(ImportError("optional dependency")),
    )
    with patch(
        "tributo.explainability.registry.entry_points",
        return_value=[entry_point],
    ):
        registry.discover_entry_points()
    assert registry.list_all() == ()
    assert "broken-explainer" in registry.diagnostics()[0]


def test_registry_fails_closed_on_duplicate_entry_point_ids() -> None:
    class DuplicateAdapter:
        api_version = 1
        adapter_id = "shap-v1"
        adapter_version = "1"

        @classmethod
        def supports(cls, context, request):
            del context, request

        def prepare(self, context, request):
            del context, request

        def explain_batch(self, prepared, batch, **kwargs):
            del prepared, batch, kwargs

        def summarize(self, attribution_batch):
            return attribution_batch

    registry = ExplainerRegistry()
    registry.register(ShapAdapter)
    entry_point = SimpleNamespace(name="duplicate", load=lambda: DuplicateAdapter)
    with patch(
        "tributo.explainability.registry.entry_points",
        return_value=[entry_point],
    ):
        with pytest.raises(ValueError, match="duplicate"):
            registry.discover_entry_points()
