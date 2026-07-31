"""Provenance and runtime specification models.

Provides:

- ``ProvenanceRecord``: Full export lineage — who, what, when, how.
- ``RuntimeSpec``: Target runtime environment requirements for a bundle.
- ``ProvenanceBuilder``: Assembles provenance from export components.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tributo.util.annotations import PublicAPI


# ── ProvenanceRecord ────────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class ProvenanceRecord(BaseModel):
    """Complete provenance chain for a bundle export.

    Tracks the full lineage: which trainer, which source, which exporters,
    which validators, at what time, with what versions.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    export_id: str = ""
    bundle_id: str = ""

    # ── Source ──
    source_kind: str = ""
    source_fingerprint: str = ""
    framework: str | None = None
    framework_version: str | None = None
    architecture_id: str | None = None

    # ── Toolchain ──
    tributo_version: str = "0.0.0"
    python_version: str | None = None
    system_info: dict[str, str] = Field(default_factory=dict)

    # ── Export chain ──
    exporter_ids: tuple[str, ...] = ()
    validator_ids: tuple[str, ...] = ()
    flavors: tuple[str, ...] = ()

    # ── Timing ──
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: int = 0

    # ── Environment ──
    env_hash: str = ""
    entry_point_group: str | None = None


# ── RuntimeSpec ──────────────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class RuntimeSpec(BaseModel):
    """Target runtime environment requirements.

    Embedded in the bundle manifest so downstream systems (deployment,
    CI, model server) can validate compatibility before loading.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ── Runtime name ──
    runtime: str = Field(
        default="onnxruntime",
        description="Target runtime (onnxruntime, tensorrt, tvm, coreml).",
    )
    runtime_version: str | None = Field(
        default=None, description="Minimum runtime version (semver)."
    )

    # ── Hardware requirements ──
    cpu_arch: str | None = Field(
        default=None, description="CPU architecture (x86_64, arm64)."
    )
    gpu_required: bool = False
    gpu_min_compute_capability: str | None = None
    gpu_min_memory_mb: int = 0

    # ── Memory / disk ──
    min_ram_mb: int = 0
    estimated_model_size_mb: int = 0

    # ── Opset / format version ──
    opset_version: int | None = None
    format_version: str | None = None

    # ── Python deps ──
    python_min_version: str | None = None
    pip_requirements: tuple[str, ...] = ()
    conda_channels: tuple[str, ...] = ()


# ── ProvenanceBuilder ───────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class ProvenanceBuilder:
    """Assemble a ``ProvenanceRecord`` from export execution components.

    Usage::

        builder = ProvenanceBuilder()
        builder.with_source(source).with_exporters(planner).build()
    """

    def __init__(self) -> None:
        self._export_id: str = ""
        self._bundle_id: str = ""
        self._source_kind: str = ""
        self._source_fingerprint: str = ""
        self._framework: str | None = None
        self._framework_version: str | None = None
        self._architecture_id: str | None = None
        self._tributo_version: str = "0.0.0"
        self._exporter_ids: list[str] = []
        self._validator_ids: list[str] = []
        self._flavors: list[str] = []
        self._started_at: datetime | None = None
        self._duration_ms: int = 0

    def with_export_id(self, export_id: str) -> ProvenanceBuilder:
        self._export_id = export_id
        return self

    def with_bundle_id(self, bundle_id: str) -> ProvenanceBuilder:
        self._bundle_id = bundle_id
        return self

    def with_source(self, source: Any) -> ProvenanceBuilder:
        """Populate source fields from an ``ExportSource``."""
        from tributo.exporting.models import ExportSource

        if isinstance(source, ExportSource):
            self._source_kind = source.source_kind
            self._source_fingerprint = source.source_fingerprint
            self._architecture_id = source.architecture_id
            self._framework = source.metadata.get("framework")
            self._framework_version = source.metadata.get("framework_version")
        return self

    def with_exporters(self, exporter_ids: list[str]) -> ProvenanceBuilder:
        self._exporter_ids = list(exporter_ids)
        return self

    def with_validators(self, validator_ids: list[str]) -> ProvenanceBuilder:
        self._validator_ids = list(validator_ids)
        return self

    def with_flavors(self, flavors: list[str]) -> ProvenanceBuilder:
        self._flavors = list(flavors)
        return self

    def with_tributo_version(self, version: str) -> ProvenanceBuilder:
        self._tributo_version = version
        return self

    def with_timing(self, started_at: datetime, duration_ms: int) -> ProvenanceBuilder:
        self._started_at = started_at
        self._duration_ms = duration_ms
        return self

    def build(self) -> ProvenanceRecord:
        """Build the immutable ``ProvenanceRecord``."""
        import platform
        import sys

        return ProvenanceRecord(
            export_id=self._export_id,
            bundle_id=self._bundle_id,
            source_kind=self._source_kind,
            source_fingerprint=self._source_fingerprint,
            framework=self._framework,
            framework_version=self._framework_version,
            architecture_id=self._architecture_id,
            tributo_version=self._tributo_version,
            python_version=sys.version.split()[0],
            system_info={
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            exporter_ids=tuple(self._exporter_ids),
            validator_ids=tuple(self._validator_ids),
            flavors=tuple(self._flavors),
            started_at=self._started_at or datetime.now(timezone.utc),
            duration_ms=self._duration_ms,
            env_hash="",  # Computed by service layer.
        )
