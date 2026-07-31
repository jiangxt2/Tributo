"""Plugin conformance test kit — reusable test helpers for plugin authors.

Plugin authors (third-party exporters, validators, source providers)
can extend these base test classes to verify that their implementation
conforms to the framework protocols.

Usage::

    from tributo.exporting.conftest import ExporterConformanceTest

    class TestMyExporter(ExporterConformanceTest):
        exporter_cls = MyCustomExporter

        def make_source(self):
            return ExportSource(source_kind="my_kind", model_object=...)

        def make_target(self):
            return ExportTarget(name="test", format="my_format")

        def make_context(self, tmp_path):
            return ExportContext(
                execution_id="test",
                node_id="test",
                artifact_dir=tmp_path,
            )
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from tributo.exporting.models import (
    ExportSource,
    PlannedTarget,
    SupportRequest,
    ValidatorBinding,
)
from tributo.util.annotations import PublicAPI

# ═══════════════════════════════════════════════════════════════════════════════
# ModelExporter conformance test
# ═══════════════════════════════════════════════════════════════════════════════


@PublicAPI(stability="beta")
class ExporterConformanceTest:
    """Base class for testing custom ``ModelExporter`` implementations.

    Subclasses must provide:
    - ``exporter_cls``: The exporter class under test.
    - ``make_source()``: Return an ``ExportSource`` the exporter accepts.
    - ``make_target()``: Return an ``ExportTarget`` for the exporter's format.
    - ``make_context(tmp_path)``: Return an ``ExportContext``.

    Tests verify:
    - Protocol conformance (all required ClassVars present).
    - ``supports()`` returns correct results for valid/invalid sources.
    - ``export()`` produces a valid ``ArtifactDraft`` with existing files.
    - ``mutates_source`` behaviour is consistent.
    """

    exporter_cls: type[Any]
    make_source: Any
    make_target: Any
    make_context: Any

    def test_classvar_declaration(self) -> None:
        """Verify all required ClassVars are present."""
        required = (
            "api_version",
            "exporter_id",
            "priority",
            "output_format",
            "options_model",
            "validator_bindings",
            "mutates_source",
            "upstream_requirements",
        )
        for attr in required:
            assert hasattr(self.exporter_cls, attr), (
                f"{self.exporter_cls.__name__} missing ClassVar {attr!r}"
            )

    def test_api_version(self) -> None:
        """Verify api_version == 1."""
        assert self.exporter_cls.api_version == 1

    def test_exporter_id_format(self) -> None:
        """Verify exporter_id is a non-empty string."""
        eid = self.exporter_cls.exporter_id
        assert isinstance(eid, str), "exporter_id must be str"
        assert len(eid) > 0, "exporter_id must not be empty"

    def test_priority_is_int(self) -> None:
        """Verify priority is an int."""
        assert isinstance(self.exporter_cls.priority, int)

    def test_output_format_is_str(self) -> None:
        """Verify output_format is a non-empty string."""
        fmt = self.exporter_cls.output_format
        assert isinstance(fmt, str) and fmt

    def test_options_model_is_pydantic(self) -> None:
        """Verify options_model is a pydantic BaseModel subclass."""
        from pydantic import BaseModel

        assert issubclass(self.exporter_cls.options_model, BaseModel)

    def test_validator_bindings_are_tuple(self) -> None:
        """Verify validator_bindings is a tuple of ValidatorBinding."""
        bindings = self.exporter_cls.validator_bindings
        assert isinstance(bindings, tuple)
        for b in bindings:
            assert isinstance(b, ValidatorBinding)

    def test_upstream_requirements_are_tuple(self) -> None:
        """Verify upstream_requirements is a tuple."""
        reqs = self.exporter_cls.upstream_requirements
        assert isinstance(reqs, tuple)

    def test_supports_with_matching_source(self) -> None:
        """supports() returns supported=True for matching source_kind."""
        source = self.make_source()
        request = SupportRequest(
            source_kind=source.source_kind,
            source_metadata=source.metadata,
        )
        result = self.exporter_cls.supports(request)
        assert result.supported, (
            f"supports() returned False for matching source: [{result.code}] {result.reason}"
        )

    def test_supports_rejects_unknown_source_kind(self) -> None:
        """supports() returns supported=False for unknown source_kind."""
        request = SupportRequest(source_kind="unknown_xyz_123")
        result = self.exporter_cls.supports(request)
        assert not result.supported

    def test_export_produces_valid_draft(self) -> None:
        """export() returns an ArtifactDraft with existing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source = self.make_source()
            target = self.make_target()
            context = self.make_context(tmp_path)

            # Build PlannedTarget.
            pt = PlannedTarget(
                target=target,
                exporter_id=self.exporter_cls.exporter_id,
                typed_options=self.exporter_cls.options_model().model_dump(),
                validator_bindings=(),
                implicit=False,
                publish=True,
            )

            exporter = self.exporter_cls()
            draft = exporter.export(context, source, {}, pt)

            # Verify draft structure.
            assert draft.name == target.name
            assert draft.format == self.exporter_cls.output_format
            assert len(draft.files) > 0

            # Verify all declared files exist.
            for df in draft.files:
                fp = tmp_path / df.relative_path
                assert fp.is_file(), (
                    f"Declared file {df.relative_path!r} does not exist"
                )
                assert fp.stat().st_size > 0, f"File {df.relative_path!r} is empty"

            # Verify entrypoint.
            assert draft.entrypoint in {f.relative_path for f in draft.files}


# ═══════════════════════════════════════════════════════════════════════════════
# SourceProvider conformance test
# ═══════════════════════════════════════════════════════════════════════════════


@PublicAPI(stability="beta")
class SourceProviderConformanceTest:
    """Base class for testing custom ``SourceProvider`` implementations.

    Subclasses must provide:
    - ``provider_cls``: The provider class under test.
    - ``make_result()``: Return a training result the provider accepts.
    """

    provider_cls: type[Any]
    make_result: Any

    def test_classvar_declaration(self) -> None:
        """Verify all required ClassVars are present."""
        required = ("api_version", "provider_id", "trainer_type", "priority")
        for attr in required:
            assert hasattr(self.provider_cls, attr)

    def test_api_version(self) -> None:
        """Verify api_version == 1."""
        assert self.provider_cls.api_version == 1

    def test_trainer_type_is_str(self) -> None:
        """Verify trainer_type is a non-empty string."""
        tt = self.provider_cls.trainer_type
        assert isinstance(tt, str) and tt

    def test_open_source_yields_export_source(self) -> None:
        """open_source() yields an ExportSource with correct source_kind."""
        provider = self.provider_cls()
        result = self.make_result()

        with provider.open_source(result) as source:
            assert isinstance(source, ExportSource)
            assert source.source_kind
            assert source.model_object is not None


# ═══════════════════════════════════════════════════════════════════════════════
# ExportValidator conformance test
# ═══════════════════════════════════════════════════════════════════════════════


@PublicAPI(stability="beta")
class ValidatorConformanceTest:
    """Base class for testing custom ``ExportValidator`` implementations.

    Subclasses must provide:
    - ``validator_cls``: The validator class under test.
    - ``make_source()``: Return an ExportSource.
    - ``make_artifact()``: Return a ResolvedArtifact (valid).
    """

    validator_cls: type[Any]
    make_source: Any
    make_artifact: Any

    def test_classvar_declaration(self) -> None:
        """Verify all required ClassVars are present."""
        required = ("api_version", "validator_id", "options_model")
        for attr in required:
            assert hasattr(self.validator_cls, attr)

    def test_api_version(self) -> None:
        """Verify api_version == 1."""
        assert self.validator_cls.api_version == 1

    def test_validate_returns_result(self) -> None:
        """validate() returns a ValidationResult."""
        validator = self.validator_cls()
        source = self.make_source()
        artifact = self.make_artifact()
        opts = self.validator_cls.options_model()

        from tributo.exporting.models import ValidationResult

        result = validator.validate(source, artifact, {}, opts)
        assert isinstance(result, ValidationResult)
        assert result.status in ("passed", "failed", "advisory_failed")
