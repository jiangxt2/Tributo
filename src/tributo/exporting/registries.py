"""Bundle export registries — discovery, registration, and candidate selection.

Five registries provide the plugin backbone:

- ``ExportRegistry`` — ModelExporter candidates.
- ``SourceProviderRegistry`` — Resolve provider by trainer_type.
- ``ValidatorRegistry`` — Validator chain members.
- ``FlavorRegistry`` — Runtime model loading by flavor_id.
- ``ModelFactoryRegistry`` — Safetensors architecture reconstruction.

All registries expose ``diagnostics()`` for inspecting plugin load issues.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from tributo.exceptions import JobConfigurationError
from tributo.exporting.formats import validate_format_id
from tributo.exporting.models import (
    ExportTarget,
    PlannedTarget,
    PluginLoadDiagnostic,
    SupportRequest,
    SupportResult,
    ValidatorBinding,
)
from tributo.exporting.protocols import (
    ExportSourceProvider,
    ExportValidator,
    ModelExporter,
    ModelFactory,
)
from tributo.util.annotations import DeveloperAPI, PublicAPI

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ExportRegistry
# ═══════════════════════════════════════════════════════════════════════════════


@PublicAPI(stability="beta")
class ExportRegistry:
    """Registry of ``ModelExporter`` classes keyed by ``exporter_id``.

    Supports ``list_candidates()`` for filtering by ``(source_kind, output_format)``
    and priority-based selection.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, type[ModelExporter]] = {}
        self._diagnostics: list[PluginLoadDiagnostic] = []

    # -- write --

    def register(self, exporter_cls: type[ModelExporter]) -> None:
        """Register *exporter_cls* by its ``exporter_id``."""
        api_version = getattr(exporter_cls, "api_version", None)
        entry_point_name = getattr(exporter_cls, "exporter_id", None) or getattr(
            exporter_cls, "__name__", str(exporter_cls)
        )
        if api_version != 2:
            logger.warning(
                "Ignoring exporter %r with unsupported api_version %r; expected 2",
                entry_point_name,
                api_version,
            )
            self._diagnostics.append(
                PluginLoadDiagnostic(
                    group="tributo.exporters",
                    entry_point_name=str(entry_point_name),
                    reason=(
                        "Unsupported ModelExporter api_version "
                        f"{api_version!r}; expected 2"
                    ),
                )
            )
            return
        eid = getattr(exporter_cls, "exporter_id", None)
        if not isinstance(eid, str) or not eid:
            logger.warning("Ignoring exporter with invalid exporter_id %r", eid)
            self._diagnostics.append(
                PluginLoadDiagnostic(
                    group="tributo.exporters",
                    entry_point_name=getattr(
                        exporter_cls, "__name__", str(exporter_cls)
                    ),
                    reason=f"Invalid exporter_id: {eid!r}",
                )
            )
            return
        output_format = getattr(exporter_cls, "output_format", None)
        if not isinstance(output_format, str):
            logger.warning(
                "Ignoring exporter %r with invalid output_format %r",
                eid,
                output_format,
            )
            self._diagnostics.append(
                PluginLoadDiagnostic(
                    group="tributo.exporters",
                    entry_point_name=eid,
                    reason=f"Invalid output_format: {output_format!r}",
                )
            )
            return
        try:
            validate_format_id(output_format)
        except ValueError as exc:
            logger.warning("Ignoring exporter %r: invalid output_format: %s", eid, exc)
            self._diagnostics.append(
                PluginLoadDiagnostic(
                    group="tributo.exporters",
                    entry_point_name=eid,
                    reason=f"Invalid output_format: {exc}",
                )
            )
            return
        output_flavor_id = getattr(exporter_cls, "output_flavor_id", None)
        if not isinstance(output_flavor_id, str) or not output_flavor_id:
            logger.warning(
                "Ignoring exporter %r with invalid output_flavor_id %r",
                eid,
                output_flavor_id,
            )
            self._diagnostics.append(
                PluginLoadDiagnostic(
                    group="tributo.exporters",
                    entry_point_name=eid,
                    reason=f"Invalid output_flavor_id: {output_flavor_id!r}",
                )
            )
            return
        if eid in self._by_id:
            self._diagnostics.append(
                PluginLoadDiagnostic(
                    group="tributo.exporters",
                    entry_point_name=eid,
                    reason=f"Duplicate exporter_id {eid!r} — all conflicts disabled",
                )
            )
            del self._by_id[eid]
            return
        self._by_id[eid] = exporter_cls

    def unregister(self, exporter_id: str) -> None:
        self._by_id.pop(exporter_id, None)

    # -- read --

    def get(self, exporter_id: str) -> type[ModelExporter]:
        if exporter_id not in self._by_id:
            raise JobConfigurationError(
                f"Unknown exporter {exporter_id!r}. Available: {sorted(self._by_id)}"
            )
        return self._by_id[exporter_id]

    def list_candidates(
        self,
        source_kind: str,
        output_format: str,
    ) -> list[type[ModelExporter]]:
        """Return candidates sorted by priority (highest first), then exporter_id.

        Does NOT call ``supports()`` — that is the Planner's job.
        """
        candidates = [
            c
            for c in self._by_id.values()
            if c.output_format == output_format
            and (not getattr(c, "source_kinds", ()) or source_kind in c.source_kinds)
        ]
        candidates.sort(key=lambda c: (-c.priority, c.exporter_id))
        return candidates

    def list_all(self) -> list[str]:
        return sorted(self._by_id)

    def contains(self, exporter_id: str) -> bool:
        return exporter_id in self._by_id

    def diagnostics(self) -> tuple[PluginLoadDiagnostic, ...]:
        return tuple(self._diagnostics)

    def record_diagnostic(self, diagnostic: PluginLoadDiagnostic) -> None:
        """Append a plugin-loading diagnostic from entry-point discovery."""
        self._diagnostics.append(diagnostic)


# ═══════════════════════════════════════════════════════════════════════════════
# SourceProviderRegistry
# ═══════════════════════════════════════════════════════════════════════════════


@PublicAPI(stability="beta")
class SourceProviderRegistry:
    """Registry of ``ExportSourceProvider`` classes keyed by ``provider_id``.

    ``resolve(trainer_type)`` returns the highest-priority provider for
    a given trainer type.  Ties raise an error.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, type[ExportSourceProvider]] = {}
        self._diagnostics: list[PluginLoadDiagnostic] = []

    def register(self, provider_cls: type[ExportSourceProvider]) -> None:
        pid = provider_cls.provider_id
        if not isinstance(pid, str) or not pid:
            self._diagnostics.append(
                PluginLoadDiagnostic(
                    group="tributo.source_providers",
                    entry_point_name=str(provider_cls),
                    reason=f"Invalid provider_id: {pid!r}",
                )
            )
            return
        if pid in self._by_id:
            self._diagnostics.append(
                PluginLoadDiagnostic(
                    group="tributo.source_providers",
                    entry_point_name=pid,
                    reason=f"Duplicate provider_id {pid!r}",
                )
            )
            del self._by_id[pid]
            return
        self._by_id[pid] = provider_cls

    def resolve(self, trainer_type: str) -> type[ExportSourceProvider]:
        candidates = [p for p in self._by_id.values() if p.trainer_type == trainer_type]
        if not candidates:
            raise JobConfigurationError(
                f"No ExportSourceProvider for trainer_type {trainer_type!r}. "
                f"Registered: {sorted(self._by_id)}"
            )
        candidates.sort(key=lambda c: (-c.priority, c.provider_id))
        top = candidates[0]
        # Check for ties
        if len(candidates) > 1 and candidates[1].priority == top.priority:
            tied = [c.provider_id for c in candidates if c.priority == top.priority]
            raise JobConfigurationError(
                f"Ambiguous ExportSourceProvider for {trainer_type!r}: {tied}. "
                "Specify provider_id explicitly."
            )
        return top

    def list_all(self) -> list[str]:
        return sorted(self._by_id)

    def diagnostics(self) -> tuple[PluginLoadDiagnostic, ...]:
        return tuple(self._diagnostics)

    def record_diagnostic(self, diagnostic: PluginLoadDiagnostic) -> None:
        """Append a plugin-loading diagnostic from entry-point discovery."""
        self._diagnostics.append(diagnostic)


# ═══════════════════════════════════════════════════════════════════════════════
# ValidatorRegistry
# ═══════════════════════════════════════════════════════════════════════════════


@PublicAPI(stability="beta")
class ValidatorRegistry:
    """Registry of ``ExportValidator`` classes keyed by ``validator_id``."""

    def __init__(self) -> None:
        self._by_id: dict[str, type[ExportValidator]] = {}
        self._diagnostics: list[PluginLoadDiagnostic] = []

    def register(self, validator_cls: type[ExportValidator]) -> None:
        vid = validator_cls.validator_id
        if not isinstance(vid, str) or not vid:
            return
        if vid in self._by_id:
            self._diagnostics.append(
                PluginLoadDiagnostic(
                    group="tributo.validators",
                    entry_point_name=vid,
                    reason=f"Duplicate validator_id {vid!r}",
                )
            )
            del self._by_id[vid]
            return
        self._by_id[vid] = validator_cls

    def get(self, validator_id: str) -> type[ExportValidator]:
        if validator_id not in self._by_id:
            raise JobConfigurationError(
                f"Unknown validator {validator_id!r}. Available: {sorted(self._by_id)}"
            )
        return self._by_id[validator_id]

    def list_all(self) -> list[str]:
        return sorted(self._by_id)

    def diagnostics(self) -> tuple[PluginLoadDiagnostic, ...]:
        return tuple(self._diagnostics)

    def record_diagnostic(self, diagnostic: PluginLoadDiagnostic) -> None:
        """Append a plugin-loading diagnostic from entry-point discovery."""
        self._diagnostics.append(diagnostic)


# ═══════════════════════════════════════════════════════════════════════════════
# FlavorRegistry
# ═══════════════════════════════════════════════════════════════════════════════


@PublicAPI(stability="beta")
class FlavorRegistry:
    """Registry of ``ModelFlavor`` classes keyed by ``flavor_id``.

    Flavor IDs are more specific than format strings: ``onnx-runtime-v1``
    and ``hf-onnx-v1`` share ``format="onnx"`` but differ in
    their loading behaviour.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, type[Any]] = {}
        self._diagnostics: list[PluginLoadDiagnostic] = []

    def register(self, flavor_cls: type[Any]) -> None:
        fid = getattr(flavor_cls, "flavor_id", None)
        if not isinstance(fid, str) or not fid:
            return
        if fid in self._by_id:
            self._diagnostics.append(
                PluginLoadDiagnostic(
                    group="tributo.model_flavors",
                    entry_point_name=fid,
                    reason=f"Duplicate flavor_id {fid!r}",
                )
            )
            del self._by_id[fid]
            return
        self._by_id[fid] = flavor_cls

    def get(self, flavor_id: str) -> type[Any]:
        if flavor_id not in self._by_id:
            raise JobConfigurationError(
                f"Unknown flavor {flavor_id!r}. Available: {sorted(self._by_id)}"
            )
        return self._by_id[flavor_id]

    def list_all(self) -> list[str]:
        return sorted(self._by_id)

    def diagnostics(self) -> tuple[PluginLoadDiagnostic, ...]:
        return tuple(self._diagnostics)


# ═══════════════════════════════════════════════════════════════════════════════
# ModelFactoryRegistry
# ═══════════════════════════════════════════════════════════════════════════════


@PublicAPI(stability="beta")
class ModelFactoryRegistry:
    """Registry of ``ModelFactory`` classes keyed by ``architecture_id``."""

    def __init__(self) -> None:
        self._by_id: dict[str, type[ModelFactory]] = {}
        self._diagnostics: list[PluginLoadDiagnostic] = []

    def register(self, factory_cls: type[ModelFactory]) -> None:
        aid = factory_cls.architecture_id
        if not isinstance(aid, str) or not aid:
            return
        if aid in self._by_id:
            self._diagnostics.append(
                PluginLoadDiagnostic(
                    group="tributo.model_factories",
                    entry_point_name=aid,
                    reason=f"Duplicate architecture_id {aid!r}",
                )
            )
            del self._by_id[aid]
            return
        self._by_id[aid] = factory_cls

    def get(self, architecture_id: str) -> type[ModelFactory]:
        if architecture_id not in self._by_id:
            raise JobConfigurationError(
                f"Unknown architecture {architecture_id!r}. "
                f"Available: {sorted(self._by_id)}"
            )
        return self._by_id[architecture_id]

    def list_all(self) -> list[str]:
        return sorted(self._by_id)

    def diagnostics(self) -> tuple[PluginLoadDiagnostic, ...]:
        return tuple(self._diagnostics)


def build_factory_registry() -> ModelFactoryRegistry:
    """Build a ``ModelFactoryRegistry`` loaded with entry-point plugins.

    Mirrors the flavor registry assembly in ``runtime._build_flavor_registry``:
    discovery happens here, once, so model reconstruction never consults
    an empty registry when third-party factories are installed.
    """
    from tributo.plugin import discover_model_factory_plugins

    registry = ModelFactoryRegistry()
    for cls in discover_model_factory_plugins():
        registry.register(cls)
    return registry


# ═══════════════════════════════════════════════════════════════════════════════
# Candidate selection helpers
# ═══════════════════════════════════════════════════════════════════════════════


@DeveloperAPI
def select_candidate(
    candidates: list[type[ModelExporter]],
    target: ExportTarget,
    request: SupportRequest,
    validator_registry: ValidatorRegistry,
) -> PlannedTarget:
    """Select the best candidate from *candidates* for *target*.

    Algorithm (deterministic, no side-effects):

    1. Filter: call ``candidate.supports(request)``.
    2. If *target.exporter_id* is set, use only that candidate.
    3. Pick highest-priority supported candidate; tied priorities raise.
    4. Build ``PlannedTarget`` with typed options.

    Raises:
        JobConfigurationError: No supported candidate or ambiguous choice.
    """
    if target.exporter_id is not None:
        exact = [c for c in candidates if c.exporter_id == target.exporter_id]
        if not exact:
            raise JobConfigurationError(
                f"Exporter {target.exporter_id!r} not found among candidates "
                f"for format={target.format!r}"
            )
        candidate = exact[0]
        result = candidate.supports(request)
        if not result.supported:
            raise JobConfigurationError(
                f"Exporter {target.exporter_id!r} does not support "
                f"{target.format!r}: [{result.code}] {result.reason}"
            )
        # Mirror the auto-select branch: a candidate whose required
        # validators cannot be resolved is unavailable (fail-fast).
        # Optional bindings are tolerated here — the executor logs and
        # continues for missing optional validators.
        for vb in candidate.validator_bindings:
            if vb.required:
                validator_registry.get(vb.validator_id)
    else:
        supported: list[tuple[type[ModelExporter], SupportResult]] = []
        candidate_failures: list[str] = []
        for c in candidates:
            result = c.supports(request)
            if not result.supported:
                candidate_failures.append(
                    f"  {c.exporter_id}: [{result.code}] {result.reason}"
                )
                continue
            # Candidate-level options validation: invalid options make this
            # candidate unavailable — collect the reason and keep evaluating
            # the remaining candidates (diagnosability over fail-fast).
            try:
                c.options_model(**target.options)
            except ValidationError as exc:
                candidate_failures.append(f"  {c.exporter_id}: options invalid: {exc}")
                continue
            # A candidate whose required validators cannot be resolved is
            # unavailable too — the registry may be missing its plugin.
            # Optional bindings are tolerated (executor logs and continues).
            try:
                for vb in c.validator_bindings:
                    if vb.required:
                        validator_registry.get(vb.validator_id)
            except JobConfigurationError as exc:
                candidate_failures.append(f"  {c.exporter_id}: {exc}")
                continue
            supported.append((c, result))

        if not supported:
            raise JobConfigurationError(
                f"No exporter supports {target.format!r} for "
                f"source_kind={request.source_kind!r}:\n"
                + "\n".join(candidate_failures)
            )

        supported.sort(key=lambda x: (-x[0].priority, x[0].exporter_id))
        top = supported[0]
        if len(supported) > 1 and supported[1][0].priority == top[0].priority:
            tied = [
                x[0].exporter_id for x in supported if x[0].priority == top[0].priority
            ]
            raise JobConfigurationError(
                f"Ambiguous exporter choice for {target.format!r}: {tied}. "
                "Set exporter_id to disambiguate."
            )

        candidate = top[0]

    # Reject validation overrides the selected exporter does not bind —
    # a mistyped validator_id must not be silently ignored.
    bound_ids = {vb.validator_id for vb in candidate.validator_bindings}
    for vid in target.validation:
        if vid not in bound_ids:
            raise JobConfigurationError(
                f"Target {target.name!r} validation override {vid!r} is not "
                f"bound by exporter {candidate.exporter_id!r} "
                f"(bound: {sorted(bound_ids)})"
            )

    # Build typed options
    typed_options = candidate.options_model(**target.options).model_dump()

    # Resolve validator bindings
    bindings: list[ValidatorBinding] = []
    for vb in candidate.validator_bindings:
        # Override defaults from target.validation if provided
        overrides = target.validation.get(vb.validator_id, {})
        merged = {**vb.default_options, **overrides}
        # Required bindings must resolve here (fail-fast); missing optional
        # validators are tolerated — the executor logs and continues.
        if vb.required:
            validator_cls = validator_registry.get(vb.validator_id)
            validator_cls.options_model(**merged)
        bindings.append(
            ValidatorBinding(
                validator_id=vb.validator_id,
                required=vb.required,
                default_options=merged,
            )
        )

    return PlannedTarget(
        target=target,
        exporter_id=candidate.exporter_id,
        typed_options=typed_options,
        validator_bindings=tuple(bindings),
        implicit=False,
        publish=True,
    )
