"""Public-API tests for ``tributo.exporting.export`` input contract.

Locks the documented contract: ``export()`` accepts only an
``ExportSource`` produced by a ``SourceProvider`` — raw model objects are
rejected with ``TypeError``.
"""

from __future__ import annotations

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
