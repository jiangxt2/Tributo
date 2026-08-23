"""Capabilities derived from installed exporter and runtime plugins."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from tributo.exporting.formats import validate_format_id
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="beta")
@dataclass(frozen=True)
class ArtifactCapability:
    """Capabilities for one immutable artifact flavor.

    ``readable`` means BundleReader can materialize and integrity-check the
    artifact.  ``batch`` and ``serveable`` come only from an executable
    Flavor plugin and are never inferred from a filename or format.
    """

    flavor_id: str
    exporter_ids: tuple[str, ...]
    exportable: bool
    readable: bool
    batch: bool
    serveable: bool
    runtime_flavor_id: str | None = None
    format_ids: tuple[str, ...] = ()
    artifact_kind: str = "model"
    signature_required: bool = False
    security_mode: str = "unknown-executable"
    required_dependencies: tuple[str, ...] = ()
    operations: tuple[str, ...] = ()
    conditional_operations: tuple[str, ...] = ()


@PublicAPI(stability="beta")
class CapabilityRegistry:
    """Immutable registry derived from exporter and flavor descriptors."""

    def __init__(self, entries: tuple[ArtifactCapability, ...]) -> None:
        by_flavor: dict[str, ArtifactCapability] = {}
        by_exporter: dict[str, ArtifactCapability] = {}
        for entry in entries:
            if entry.flavor_id in by_flavor:
                raise ValueError(f"Duplicate capability flavor {entry.flavor_id!r}")
            by_flavor[entry.flavor_id] = entry
            for exporter_id in entry.exporter_ids:
                if exporter_id in by_exporter:
                    raise ValueError(f"Duplicate capability exporter {exporter_id!r}")
                by_exporter[exporter_id] = entry
        self._entries = entries
        self._by_flavor = by_flavor
        self._by_exporter = by_exporter

    @classmethod
    def from_plugins(
        cls,
        exporters: Iterable[type[Any]],
        flavors: Iterable[type[Any]] = (),
    ) -> CapabilityRegistry:
        """Build capabilities without a core-maintained format allowlist."""
        flavor_plugins: dict[str, type[Any]] = {}
        for flavor in flavors:
            flavor_id = getattr(flavor, "flavor_id", None)
            if not isinstance(flavor_id, str) or not flavor_id:
                raise ValueError(f"Flavor has invalid flavor_id: {flavor_id!r}")
            if flavor_id in flavor_plugins:
                raise ValueError(f"Duplicate runtime flavor {flavor_id!r}")
            flavor_plugins[flavor_id] = flavor

        by_flavor: dict[str, list[type[Any]]] = defaultdict(list)
        for exporter in exporters:
            exporter_id = getattr(exporter, "exporter_id", None)
            flavor_id = getattr(exporter, "output_flavor_id", None)
            output_format = getattr(exporter, "output_format", None)
            if not isinstance(exporter_id, str) or not exporter_id:
                raise ValueError(f"Exporter has invalid exporter_id: {exporter_id!r}")
            if not isinstance(flavor_id, str) or not flavor_id:
                raise ValueError(
                    f"Exporter {exporter_id!r} has invalid output_flavor_id: "
                    f"{flavor_id!r}"
                )
            if not isinstance(output_format, str):
                raise ValueError(
                    f"Exporter {exporter_id!r} has invalid output_format: "
                    f"{output_format!r}"
                )
            validate_format_id(output_format)
            by_flavor[flavor_id].append(exporter)

        entries: list[ArtifactCapability] = []
        for flavor_id in sorted(by_flavor):
            exporter_classes = by_flavor[flavor_id]
            exporter_ids = tuple(sorted(e.exporter_id for e in exporter_classes))
            format_ids = tuple(sorted({e.output_format for e in exporter_classes}))
            runtime_flavor = flavor_plugins.get(flavor_id)
            if runtime_flavor is not None:
                supported_formats = tuple(
                    getattr(runtime_flavor, "supported_formats", ())
                )
                unsupported = sorted(set(format_ids).difference(supported_formats))
                if unsupported:
                    raise ValueError(
                        f"Flavor {flavor_id!r} does not declare exporter formats "
                        f"{unsupported!r}"
                    )
            entries.append(
                ArtifactCapability(
                    flavor_id=flavor_id,
                    exporter_ids=exporter_ids,
                    exportable=True,
                    readable=True,
                    batch=bool(
                        runtime_flavor
                        and getattr(runtime_flavor, "batch_supported", False)
                    ),
                    serveable=bool(
                        runtime_flavor and getattr(runtime_flavor, "serveable", False)
                    ),
                    runtime_flavor_id=flavor_id if runtime_flavor else None,
                    format_ids=format_ids,
                    artifact_kind=str(
                        getattr(runtime_flavor, "artifact_kind", "model")
                    ),
                    signature_required=bool(
                        runtime_flavor
                        and getattr(runtime_flavor, "signature_required", False)
                    ),
                    security_mode=str(
                        getattr(
                            runtime_flavor,
                            "security_mode",
                            "unknown-executable",
                        )
                    ),
                    required_dependencies=tuple(
                        getattr(runtime_flavor, "required_dependencies", ())
                    ),
                    operations=_runtime_operations(runtime_flavor),
                    conditional_operations=_runtime_conditional_operations(
                        runtime_flavor
                    ),
                )
            )

        # Runtime-only flavors supplied by third-party or future plugins still
        # declare read/batch/serve capabilities; they are not exportable until
        # an exporter declares the same output_flavor_id.
        for flavor_id, flavor_cls in sorted(flavor_plugins.items()):
            if flavor_id in by_flavor:
                continue
            entries.append(
                ArtifactCapability(
                    flavor_id=flavor_id,
                    exporter_ids=(),
                    exportable=False,
                    readable=True,
                    batch=bool(getattr(flavor_cls, "batch_supported", False)),
                    serveable=bool(getattr(flavor_cls, "serveable", False)),
                    runtime_flavor_id=flavor_id,
                    format_ids=tuple(getattr(flavor_cls, "supported_formats", ())),
                    artifact_kind=str(getattr(flavor_cls, "artifact_kind", "model")),
                    signature_required=bool(
                        getattr(flavor_cls, "signature_required", False)
                    ),
                    security_mode=str(
                        getattr(flavor_cls, "security_mode", "unknown-executable")
                    ),
                    required_dependencies=tuple(
                        getattr(flavor_cls, "required_dependencies", ())
                    ),
                    operations=_runtime_operations(flavor_cls),
                    conditional_operations=_runtime_conditional_operations(flavor_cls),
                )
            )
        return cls(tuple(entries))

    def entries(self) -> tuple[ArtifactCapability, ...]:
        """Return all declarations in deterministic order."""
        return self._entries

    def for_flavor(self, flavor_id: str) -> ArtifactCapability:
        """Return a declared flavor or fail closed."""
        try:
            return self._by_flavor[flavor_id]
        except KeyError as exc:
            raise KeyError(f"Undeclared artifact flavor {flavor_id!r}") from exc

    def for_exporter(self, exporter_id: str) -> ArtifactCapability:
        """Return a declared exporter or fail closed."""
        try:
            return self._by_exporter[exporter_id]
        except KeyError as exc:
            raise KeyError(f"Undeclared exporter {exporter_id!r}") from exc


def _runtime_operations(runtime_flavor: type[Any] | None) -> tuple[str, ...]:
    if runtime_flavor is None:
        return ()
    declared = tuple(getattr(runtime_flavor, "operations", ()))
    if declared:
        return tuple(dict.fromkeys(str(operation) for operation in declared))
    operations: list[str] = []
    if getattr(runtime_flavor, "batch_supported", False):
        operations.append("prediction.batch")
    if getattr(runtime_flavor, "serveable", False):
        operations.append("prediction.online")
    return tuple(operations)


def _runtime_conditional_operations(
    runtime_flavor: type[Any] | None,
) -> tuple[str, ...]:
    if runtime_flavor is None:
        return ()
    declared = tuple(getattr(runtime_flavor, "conditional_operations", ()))
    return tuple(dict.fromkeys(str(operation) for operation in declared))


def _build_default_capability_registry() -> CapabilityRegistry:
    """Compose first-party and discovered descriptors without duplicate ids."""
    from tributo._bootstrap import (
        first_party_export_plugins,
        first_party_model_flavors,
    )
    from tributo.plugin import discover_exporter_plugins, discover_flavor_plugins

    builtin_exporters, _ = first_party_export_plugins()
    builtin_flavors = first_party_model_flavors()
    exporter_ids = {exporter.exporter_id for exporter in builtin_exporters}
    flavor_ids = {flavor.flavor_id for flavor in builtin_flavors}
    exporters = (
        *builtin_exporters,
        *(
            exporter
            for exporter in discover_exporter_plugins()
            if exporter.exporter_id not in exporter_ids
        ),
    )
    flavors = (
        *builtin_flavors,
        *(
            flavor
            for flavor in discover_flavor_plugins()
            if flavor.flavor_id not in flavor_ids
        ),
    )
    return CapabilityRegistry.from_plugins(exporters, flavors)


_DEFAULT_CAPABILITY_REGISTRY: CapabilityRegistry | None = None


@PublicAPI(stability="beta")
def get_default_capability_registry() -> CapabilityRegistry:
    """Return the lazily composed first-party capability registry.

    Importing :mod:`tributo.exporting` therefore remains free of concrete
    integration imports; the internal composition root runs only when this
    factory is called.
    """
    global _DEFAULT_CAPABILITY_REGISTRY
    if _DEFAULT_CAPABILITY_REGISTRY is None:
        _DEFAULT_CAPABILITY_REGISTRY = _build_default_capability_registry()
    return _DEFAULT_CAPABILITY_REGISTRY


__all__ = [
    "ArtifactCapability",
    "CapabilityRegistry",
    "get_default_capability_registry",
]
