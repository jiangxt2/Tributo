"""Public-API tests for ``tributo.exporting.export`` input contract.

Locks the documented contract: ``export()`` accepts only an
``ExportSource`` produced by an ``ExportSourceProvider`` — raw model
objects are rejected with ``TypeError``.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from tributo.exporting import ExportSpec, export


class TestExportInputContract:
    def test_export_rejects_raw_model_object(self) -> None:
        spec = ExportSpec(bundle_uri="/tmp/b")
        # Simulate a caller passing a runtime object of unknown type.
        raw_object: Any = object()

        with pytest.raises(TypeError, match="ExportSource"):
            export(raw_object, spec)


class TestSourceProviderAlias:
    """Deprecated ``SourceProvider`` name resolves with a warning (STABILITY.md)."""

    def test_deprecated_name_warns_and_resolves(self) -> None:
        protocols = importlib.import_module("tributo.exporting.protocols")

        with pytest.warns(DeprecationWarning, match="deprecated"):
            alias = protocols.SourceProvider

        assert alias is protocols.ExportSourceProvider

    def test_wildcard_import_warns_and_exposes_deprecated_name(self) -> None:
        # STABILITY.md promises import compatibility — including `import *`.
        ns: dict[str, Any] = {}
        with pytest.warns(DeprecationWarning, match="deprecated"):
            exec("from tributo.exporting.protocols import *", ns)

        assert ns["SourceProvider"] is ns["ExportSourceProvider"]
