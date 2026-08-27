"""A new format plugin must compose without any core format branch."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Mapping

import pytest
from pydantic import BaseModel, ConfigDict

from tributo.exporting.executor import ExportManager
from tributo.exporting.models import (
    ArtifactDraft,
    BundleOutputConfig,
    DraftFile,
    ExportContext,
    ExportSource,
    ExportTarget,
    PlannedTarget,
    ProducerInfo,
    ResolvedArtifact,
    SupportRequest,
    SupportResult,
    UpstreamRequirement,
    ValidatorBinding,
)
from tributo.exporting.planner import ExportPlanner
from tributo.exporting.registries import ExportRegistry, ValidatorRegistry
from tributo.plugin import discover_exporter_plugins


class _GGUFOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _GGUFExporter:
    api_version: ClassVar[int] = 2
    exporter_id: ClassVar[str] = "third-party-gguf-v1"
    priority: ClassVar[int] = 100
    output_format: ClassVar[str] = "gguf"
    output_flavor_id: ClassVar[str] = "gguf-file-v1"
    source_kinds: ClassVar[tuple[str, ...]] = ("synthetic_result",)
    options_model: ClassVar[type[BaseModel]] = _GGUFOptions
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = ()
    mutates_source: ClassVar[bool] = False
    upstream_requirements: ClassVar[tuple[UpstreamRequirement, ...]] = ()

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        return SupportResult(
            supported=request.source_kind == "synthetic_result",
            code="OK" if request.source_kind == "synthetic_result" else "UNSUPPORTED",
        )

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: Mapping[str, ResolvedArtifact],
        target: PlannedTarget,
    ) -> ArtifactDraft:
        del source, upstream
        (context.artifact_dir / "model.gguf").write_bytes(b"synthetic-gguf")
        return ArtifactDraft(
            name=target.target.name,
            format=self.output_format,
            flavor_id=self.output_flavor_id,
            files=(DraftFile(relative_path="model.gguf"),),
            entrypoint="model.gguf",
            producer=ProducerInfo(exporter_id=self.exporter_id),
        )


def test_third_party_format_discovers_plans_and_executes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _EntryPoint:
        name = _GGUFExporter.exporter_id
        value = "third_party.exporters:GGUFExporter"

        @staticmethod
        def load() -> type[_GGUFExporter]:
            return _GGUFExporter

    monkeypatch.setattr(
        "tributo.plugin._iter_entry_points",
        lambda group: [_EntryPoint()] if group == "tributo.exporters" else [],
    )
    registry = ExportRegistry()
    for exporter in discover_exporter_plugins():
        registry.register(exporter)

    config = BundleOutputConfig(
        bundle_uri=str(tmp_path / "bundle"),
        targets=[ExportTarget(name="portable", format="gguf")],
    )
    source = ExportSource(source_kind="synthetic_result")
    validators = ValidatorRegistry()
    plan = ExportPlanner(registry, validators).plan(config, source)
    result = ExportManager(registry, validators).execute(
        plan,
        source,
        tmp_path / "staging",
        "execution-1",
    )

    assert result.status == "succeeded"
    assert result.staged_artifacts["portable"].format == "gguf"
    assert result.staged_artifacts["portable"].flavor_id == "gguf-file-v1"


def test_exporter_discovery_rejects_incomplete_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _IncompleteExporter:
        api_version = 2
        exporter_id = "incomplete-v1"
        priority = 1
        output_format = "gguf"
        output_flavor_id = "gguf-file-v1"
        source_kinds = ("synthetic_result",)
        options_model = _GGUFOptions
        validator_bindings: tuple[ValidatorBinding, ...] = ()
        mutates_source = False

    class _EntryPoint:
        name = _IncompleteExporter.exporter_id
        value = "third_party.exporters:IncompleteExporter"

        @staticmethod
        def load() -> type[_IncompleteExporter]:
            return _IncompleteExporter

    monkeypatch.setattr(
        "tributo.plugin._iter_entry_points",
        lambda group: [_EntryPoint()] if group == "tributo.exporters" else [],
    )

    diagnostics = []
    assert discover_exporter_plugins(diagnostics) == []
    assert diagnostics
    assert "Missing ModelExporter v2 attributes" in diagnostics[0].reason


def test_exporter_discovery_reports_legacy_protocol_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _LegacyExporter(_GGUFExporter):
        api_version = 1
        exporter_id = "legacy-gguf-v1"

    class _EntryPoint:
        name = _LegacyExporter.exporter_id
        value = "third_party.exporters:LegacyExporter"

        @staticmethod
        def load() -> type[_LegacyExporter]:
            return _LegacyExporter

    monkeypatch.setattr(
        "tributo.plugin._iter_entry_points",
        lambda group: [_EntryPoint()] if group == "tributo.exporters" else [],
    )
    diagnostics = []

    assert discover_exporter_plugins(diagnostics) == []
    assert diagnostics
    assert diagnostics[0].reason == (
        "Unsupported ModelExporter api_version 1; expected 2"
    )


def test_legacy_options_module_reexports_plugin_owned_schema() -> None:
    from tributo.exporting import options as compatibility_options
    from tributo.integrations.exporters.options import TorchONNXOptions

    with pytest.warns(DeprecationWarning, match=r"integrations\.exporters\.options"):
        resolved = compatibility_options.__getattr__("TorchONNXOptions")

    assert resolved is TorchONNXOptions
