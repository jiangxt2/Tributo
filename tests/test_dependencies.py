"""Contract tests for the unified dependency layer (``_common.dependencies``).

Covers the six side-effect and error-path contracts plus the two version
scenarios (TOO_OLD / VERSION_UNKNOWN) from the Phase 0-b plan.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tributo._common.dependencies import (
    SAFETENSORS,
    TORCH,
    TRANSFORMERS,
    DependencySpec,
    DependencyState,
    DependencyStatus,
    DependencyUnavailableError,
    MissingOptionalDependency,
    probe_dependency,
    require_dependency,
)

# A stdlib module that is always importable, used as a stand-in "present"
# top-level package without touching third-party dependencies.
_PRESENT = "os"
_MISSING = "zzz_no_such_pkg"


def _spec(
    minimum_version: str | None = "1.0.0",
    extra: str | None = None,
) -> DependencySpec:
    return DependencySpec(_PRESENT, "zzz-fake-dist", minimum_version, extra)


class TestProbe:
    def test_uses_top_level_name_and_maps_parent_missing_to_missing(self) -> None:
        """find_spec must be called with the top-level name only; a parent
        missing error (dotted-name crash, PR #17) maps to MISSING."""
        spec = DependencySpec(_MISSING, _MISSING, "1.0.0")
        with patch(
            "tributo._common.dependencies.importlib.util.find_spec",
            side_effect=ModuleNotFoundError(
                f"No module named '{_MISSING}'", name=_MISSING
            ),
        ) as mock_find_spec:
            status = probe_dependency(spec)
        mock_find_spec.assert_called_once_with(_MISSING)
        assert status.state is DependencyState.MISSING

    def test_no_spec_maps_to_missing(self) -> None:
        with patch(
            "tributo._common.dependencies.importlib.util.find_spec",
            return_value=None,
        ):
            status = probe_dependency(_spec())
        assert status.state is DependencyState.MISSING

    def test_value_error_maps_to_missing(self) -> None:
        """Namespace modules whose ``__spec__`` is None raise ValueError."""
        with patch(
            "tributo._common.dependencies.importlib.util.find_spec",
            side_effect=ValueError("no spec"),
        ):
            status = probe_dependency(_spec())
        assert status.state is DependencyState.MISSING

    def test_available_without_minimum_version_skips_metadata(self) -> None:
        """Present top-level package + no version floor → AVAILABLE even
        when distribution metadata is absent."""
        status = probe_dependency(DependencySpec(_PRESENT, "zzz-not-real", None))
        assert status.state is DependencyState.AVAILABLE

    def test_probe_has_no_side_effects(self) -> None:
        """probe must never import the module; a missing distribution
        reports VERSION_UNKNOWN instead."""
        with patch(
            "tributo._common.dependencies.importlib.import_module",
            side_effect=AssertionError("probe must not import the module"),
        ):
            status = probe_dependency(_spec())
        assert status.state is DependencyState.VERSION_UNKNOWN

    def test_too_old_when_below_minimum(self) -> None:
        with patch(
            "tributo._common.dependencies.importlib.metadata.version",
            return_value="0.9.9",
        ):
            status = probe_dependency(_spec())
        assert status.state is DependencyState.TOO_OLD
        assert status.installed_version == "0.9.9"

    def test_shorter_installed_version_satisfies_floor(self) -> None:
        """``1.0`` must satisfy ``>=1.0.0`` — tuple comparison would
        misreport TOO_OLD when segment counts differ."""
        with patch(
            "tributo._common.dependencies.importlib.metadata.version",
            return_value="1.0",
        ):
            status = probe_dependency(_spec())
        assert status.state is DependencyState.AVAILABLE

    def test_real_missing_package_reports_missing(self) -> None:
        """probe on a genuinely uninstalled top-level package reports
        MISSING without raising (real find_spec, no mocking)."""
        spec = DependencySpec(_MISSING, _MISSING, "1.0.0")
        status = probe_dependency(spec)
        assert status.state is DependencyState.MISSING

    @pytest.mark.parametrize(
        ("installed", "expected_state"),
        [
            ("2.5.0rc1", DependencyState.TOO_OLD),
            ("2.5.0.dev1", DependencyState.TOO_OLD),
            ("2.5.0", DependencyState.AVAILABLE),
            ("2.5.0.post1", DependencyState.AVAILABLE),
            ("2.5.0+cpu", DependencyState.AVAILABLE),
        ],
    )
    def test_pep440_version_semantics(
        self, installed: str, expected_state: DependencyState
    ) -> None:
        """Version floors follow PEP 440 pre/post/dev/local ordering."""
        spec = _spec(minimum_version="2.5.0")
        with patch(
            "tributo._common.dependencies.importlib.metadata.version",
            return_value=installed,
        ):
            status = probe_dependency(spec)
        assert status.state is expected_state

    def test_version_unknown_when_metadata_missing_or_invalid(self) -> None:
        # Distribution metadata absent (fake dist name) → VERSION_UNKNOWN.
        assert probe_dependency(_spec()).state is DependencyState.VERSION_UNKNOWN
        # Invalid version string → VERSION_UNKNOWN.
        with patch(
            "tributo._common.dependencies.importlib.metadata.version",
            return_value="not-a-version",
        ):
            status = probe_dependency(_spec())
        assert status.state is DependencyState.VERSION_UNKNOWN


class TestRequire:
    def test_converts_top_level_module_not_found(self) -> None:
        """A ModuleNotFoundError naming the top-level import (package
        vanished between probe and import) becomes DependencyUnavailableError."""
        spec = DependencySpec(_PRESENT, "zzz-fake-dist")
        with patch(
            "tributo._common.dependencies.importlib.import_module",
            side_effect=ModuleNotFoundError(
                f"No module named '{_PRESENT}'", name=_PRESENT
            ),
        ):
            with pytest.raises(DependencyUnavailableError, match="is not installed"):
                require_dependency(spec)

    def test_passes_through_internal_import_error(self) -> None:
        """A ModuleNotFoundError for a transitive dependency (not the
        top-level name) propagates unchanged."""
        spec = DependencySpec(_PRESENT, "zzz-fake-dist")
        with patch(
            "tributo._common.dependencies.importlib.import_module",
            side_effect=ModuleNotFoundError("No module named 'numpy'", name="numpy"),
        ):
            with pytest.raises(ModuleNotFoundError, match="numpy"):
                require_dependency(spec)

    def test_returns_module_when_available(self) -> None:
        module = require_dependency(DependencySpec(_PRESENT, "zzz-fake-dist", None))
        assert module.__name__ == _PRESENT

    def test_missing_optional_dependency_hints_extra(self) -> None:
        spec = DependencySpec(_MISSING, _MISSING, "1.0.0", extra="s3")
        with patch(
            "tributo._common.dependencies.importlib.util.find_spec",
            return_value=None,
        ):
            with pytest.raises(
                MissingOptionalDependency, match=r"pip install tributo\[s3\]"
            ):
                require_dependency(spec)

    def test_missing_core_dependency_reports_broken_environment(self) -> None:
        spec = DependencySpec(_MISSING, _MISSING, None)
        with patch(
            "tributo._common.dependencies.importlib.util.find_spec",
            return_value=None,
        ):
            with pytest.raises(DependencyUnavailableError) as exc_info:
                require_dependency(spec)
        assert not isinstance(exc_info.value, MissingOptionalDependency)
        assert "environment" in str(exc_info.value)

    def test_too_old_raises(self) -> None:
        with patch(
            "tributo._common.dependencies.importlib.metadata.version",
            return_value="0.9.9",
        ):
            with pytest.raises(DependencyUnavailableError, match=r"0\.9\.9.*>=1\.0\.0"):
                require_dependency(_spec(minimum_version="1.0.0"))


class TestSupportsExactMissingSet:
    """supports() must report exactly the missing dependencies."""

    def test_hf_reason_matches_exact_missing_set(self) -> None:
        from tributo.exporting.models import SupportRequest
        from tributo.integrations.exporters.hf_onnx import HuggingFaceONNXExporter

        request = SupportRequest(source_kind="hf_model", upstream_formats=())
        with patch(
            "tributo.integrations.exporters.hf_onnx.probe_dependency",
            side_effect=[
                DependencyStatus(TRANSFORMERS, DependencyState.MISSING),
                DependencyStatus(TORCH, DependencyState.AVAILABLE),
            ],
        ):
            result = HuggingFaceONNXExporter.supports(request)
        assert result.reason == (
            "transformers>=4.40.0 required; install tributo[model-export-hf]"
        )
        assert result.missing_dependencies == ("transformers",)

    def test_hf_reason_points_both_missing_dependencies_to_combined_extra(
        self,
    ) -> None:
        from tributo.exporting.models import SupportRequest
        from tributo.integrations.exporters.hf_onnx import HuggingFaceONNXExporter

        request = SupportRequest(source_kind="hf_model", upstream_formats=())
        with patch(
            "tributo.integrations.exporters.hf_onnx.probe_dependency",
            side_effect=[
                DependencyStatus(TRANSFORMERS, DependencyState.MISSING),
                DependencyStatus(TORCH, DependencyState.MISSING),
            ],
        ):
            result = HuggingFaceONNXExporter.supports(request)
        assert result.reason == (
            "transformers>=4.40.0/torch>=2.5.0 required; "
            "install tributo[model-export-hf]"
        )
        assert result.missing_dependencies == ("transformers", "torch")

    def test_safetensors_reason_matches_exact_missing_set(self) -> None:
        from tributo.exporting.models import SupportRequest
        from tributo.integrations.exporters.torch_safetensors import (
            TorchSafetensorsExporter,
        )

        request = SupportRequest(source_kind="torch_module", upstream_formats=())
        with patch(
            "tributo.integrations.exporters.torch_safetensors.probe_dependency",
            side_effect=[
                DependencyStatus(SAFETENSORS, DependencyState.MISSING),
                DependencyStatus(TORCH, DependencyState.AVAILABLE),
            ],
        ):
            result = TorchSafetensorsExporter.supports(request)
        assert result.reason == "safetensors>=0.4.3 required"
        assert result.missing_dependencies == ("safetensors",)
