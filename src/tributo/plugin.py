"""Third-party plugin discovery via ``importlib.metadata`` entry points.

Tributo discovers external plugins through standard Python entry points
declared in ``pyproject.toml`` under the ``[project.entry-points]`` table.

Discovery groups:
    ``tributo.trainers``
        New entries expose a lightweight ``LegacyTrainerDescriptor`` through
        an explicit attribute. Module-only entries with a ``trainer_spec``
        remain available only through the Beta compatibility API. Example::

            [project.entry-points."tributo.trainers"]
            my_algo = "my_package.descriptors:MY_ALGO_DESCRIPTOR"

    ``tributo.connectors``
        Each entry point must point to a ``DataConnector`` subclass.
        Example::

            [project.entry-points."tributo.connectors"]
            my_db = "my_package.connector:MyDBConnector"

    ``tributo.models``
        Each entry point must point to a module with a ``model_specs``
        attribute (a list of ``ModelSpec``).  Example::

            [project.entry-points."tributo.models"]
            my_models = "my_package.embedding_models"

Filtering:
    Set the ``TRIBUTO_PLUGINS`` environment variable to a comma-separated
    list of entry point names to load only those plugins.  If unset, all
    discovered plugins are loaded.
"""

from __future__ import annotations

import logging
import os
from importlib.metadata import entry_points
from typing import Any, TypeGuard, cast

from pydantic import BaseModel

from tributo.exceptions import JobConfigurationError
from tributo.exporting.models import PluginLoadDiagnostic
from tributo.exporting.protocols import (
    ExportSourceProvider,
    ExportValidator,
    ModelExporter,
    ModelFactory,
)

logger = logging.getLogger(__name__)


def _record_diagnostic(
    diagnostics: list[PluginLoadDiagnostic] | None,
    group: str,
    entry_point_name: str,
    reason: str,
    error_type: str | None = None,
) -> None:
    """Record a non-fatal plugin loading issue when a sink is provided.

    Sinks are the registries' diagnostics lists — the plan requires
    plugin import/type/api-version failures to be queryable via
    ``registry.diagnostics()``, not just logged.
    """
    if diagnostics is not None:
        diagnostics.append(
            PluginLoadDiagnostic(
                group=group,
                entry_point_name=entry_point_name,
                reason=reason,
                error_type=error_type,
            )
        )


def _get_enabled_plugins() -> set[str] | None:
    """Parse ``TRIBUTO_PLUGINS`` env var.  Returns ``None`` if unset (load all)."""
    raw = os.environ.get("TRIBUTO_PLUGINS", "").strip()
    if not raw:
        return None
    return {name.strip() for name in raw.split(",") if name.strip()}


def discover_trainer_descriptors(
    diagnostics: list[PluginLoadDiagnostic] | None = None,
    compatibility_entry_points: list[Any] | None = None,
) -> list[Any]:
    """Collect lightweight Trainer descriptors without importing old plugins.

    Entry points with an explicit attribute may expose a
    ``LegacyTrainerDescriptor``. Module-only legacy entry points are retained
    as compatibility-only metadata and are loaded only if the Beta
    ``get_trainer`` API explicitly requests them.
    """
    from tributo.integrations.algorithm_runtimes.legacy_descriptors import (
        LegacyTrainerDescriptor,
    )

    enabled = _get_enabled_plugins()
    descriptors: list[Any] = []
    for ep in _iter_trainer_entry_points():
        if enabled is not None and ep.name not in enabled:
            logger.debug("Skipping trainer plugin %r (not in TRIBUTO_PLUGINS)", ep.name)
            continue
        if ":" not in ep.value:
            if compatibility_entry_points is not None:
                compatibility_entry_points.append(ep)
            _record_diagnostic(
                diagnostics,
                "tributo.trainers",
                ep.name,
                "compatibility-only: entry point does not expose a lightweight descriptor",
            )
            continue
        try:
            descriptor = ep.load()
        except Exception as exc:
            logger.warning(
                "Failed to load trainer descriptor %r (%s)",
                ep.name,
                ep.value,
                exc_info=True,
            )
            _record_diagnostic(
                diagnostics,
                "tributo.trainers",
                ep.name,
                "Failed to load lightweight trainer descriptor",
                error_type=type(exc).__name__,
            )
            continue
        if not isinstance(descriptor, LegacyTrainerDescriptor):
            if compatibility_entry_points is not None:
                compatibility_entry_points.append(ep)
            _record_diagnostic(
                diagnostics,
                "tributo.trainers",
                ep.name,
                "compatibility-only: entry point does not export a LegacyTrainerDescriptor",
                error_type=type(descriptor).__name__,
            )
            continue
        if descriptor.name != ep.name:
            _record_diagnostic(
                diagnostics,
                "tributo.trainers",
                ep.name,
                "descriptor algorithm identity does not match the entry-point name",
            )
            raise JobConfigurationError(
                f"Trainer entry point {ep.name!r} exposes descriptor identity "
                f"{descriptor.name!r}"
            )
        descriptors.append(descriptor)
    return descriptors


def discover_connector_plugins() -> list[type[Any]]:
    """Discover third-party data connectors registered via entry_points.

    Each entry point is expected to point to a ``DataConnector`` subclass.
    """
    from tributo.data.base import DataConnector

    enabled = _get_enabled_plugins()
    classes: list[type[Any]] = []

    for ep in _iter_entry_points("tributo.connectors"):
        if enabled is not None and ep.name not in enabled:
            logger.debug(
                "Skipping connector plugin %r (not in TRIBUTO_PLUGINS)", ep.name
            )
            continue
        try:
            cls = ep.load()
        except Exception:
            logger.warning(
                "Failed to load connector plugin %r (%s)",
                ep.name,
                ep.value,
                exc_info=True,
            )
            continue

        if not (isinstance(cls, type) and issubclass(cls, DataConnector)):
            logger.warning(
                "Connector plugin %r is not a DataConnector subclass (got %r); skipping.",
                ep.name,
                cls,
            )
            continue
        classes.append(cls)
        logger.info("Discovered connector plugin %r (%s)", ep.name, ep.value)

    return classes


def discover_model_plugins() -> list[Any]:
    """Discover third-party embedding models registered via entry_points.

    Each entry point is expected to point to a module whose top-level
    ``model_specs`` attribute is a list of :class:`ModelSpec` instances.
    """
    from tributo.embeddings.registry import ModelSpec

    enabled = _get_enabled_plugins()
    specs: list[Any] = []

    for ep in _iter_entry_points("tributo.models"):
        if enabled is not None and ep.name not in enabled:
            logger.debug("Skipping model plugin %r (not in TRIBUTO_PLUGINS)", ep.name)
            continue
        try:
            mod = ep.load()
        except Exception:
            logger.warning(
                "Failed to load model plugin %r (%s)", ep.name, ep.value, exc_info=True
            )
            continue

        model_specs = getattr(mod, "model_specs", None)
        if not isinstance(model_specs, list):
            logger.warning(
                "Model plugin %r does not export a list of ModelSpec as "
                "%r.model_specs (got %r); skipping.",
                ep.name,
                ep.value,
                type(model_specs).__name__,
            )
            continue

        for spec in model_specs:
            if not isinstance(spec, ModelSpec):
                logger.warning(
                    "Model plugin %r contained non-ModelSpec item %r; skipping item.",
                    ep.name,
                    spec,
                )
                continue
            specs.append(spec)

        logger.info(
            "Discovered model plugin %r (%s) — %d model(s)",
            ep.name,
            ep.value,
            len(model_specs),
        )

    return specs


def _entry_point_distribution_name(entry_point: Any) -> str:
    distribution = getattr(entry_point, "dist", None)
    if distribution is None:
        return ""
    name = getattr(distribution, "name", None)
    if isinstance(name, str):
        return name.casefold()
    metadata = getattr(distribution, "metadata", None)
    if metadata is not None:
        candidate = metadata.get("Name", "")
        if isinstance(candidate, str):
            return candidate.casefold()
    return ""


def _iter_entry_points(group: str) -> Any:
    """Iterate over entry points using the established name ordering."""
    yield from sorted(entry_points(group=group), key=lambda ep: ep.name)


def _iter_trainer_entry_points() -> Any:
    """Iterate over Trainer entries in deterministic discovery order."""
    eps = sorted(
        entry_points(group="tributo.trainers"),
        key=lambda ep: (
            _entry_point_distribution_name(ep),
            ep.name,
            ep.value,
        ),
    )
    yield from eps


# ═══════════════════════════════════════════════════════════════════════════════
# Export plugin groups (PR 1 — contracts only)
# ═══════════════════════════════════════════════════════════════════════════════


def _validate_api_version(obj: Any, expected: int) -> bool:
    """Check that *obj* declares ``api_version == expected``."""
    av = getattr(obj, "api_version", None)
    return av == expected


def discover_exporter_plugins(
    diagnostics: list[PluginLoadDiagnostic] | None = None,
) -> list[Any]:
    """Discover third-party exporters registered as ``tributo.exporters``.

    Each entry point must point to a ``ModelExporter`` class with
    ``api_version == 1``.  When *diagnostics* (a list) is provided,
    import/type/api-version/name failures are appended to it so they can
    be queried via ``ExportRegistry.diagnostics()``.
    """
    enabled = _get_enabled_plugins()
    classes: list[Any] = []

    for ep in _iter_entry_points("tributo.exporters"):
        if enabled is not None and ep.name not in enabled:
            logger.debug(
                "Skipping exporter plugin %r (not in TRIBUTO_PLUGINS)", ep.name
            )
            continue
        try:
            cls = ep.load()
        except Exception as exc:
            logger.warning(
                "Failed to load exporter plugin %r (%s)",
                ep.name,
                ep.value,
                exc_info=True,
            )
            _record_diagnostic(
                diagnostics,
                "tributo.exporters",
                ep.name,
                f"Failed to load entry point: {exc}",
                error_type=type(exc).__name__,
            )
            continue

        if not (isinstance(cls, type) and _looks_like_exporter(cls)):
            logger.warning(
                "Exporter plugin %r is not a ModelExporter (got %r); skipping.",
                ep.name,
                cls,
            )
            _record_diagnostic(
                diagnostics,
                "tributo.exporters",
                ep.name,
                f"Not a ModelExporter class (got {cls!r})",
                error_type=type(cls).__name__,
            )
            continue
        if not _validate_api_version(cls, 1):
            logger.warning(
                "Exporter plugin %r has unsupported api_version (got %r, expected 1); skipping.",
                ep.name,
                getattr(cls, "api_version", None),
            )
            _record_diagnostic(
                diagnostics,
                "tributo.exporters",
                ep.name,
                f"Unsupported api_version {getattr(cls, 'api_version', None)!r}",
            )
            continue
        if ep.name != cls.exporter_id:
            logger.warning(
                "Exporter plugin entry-point name %r != exporter_id %r; skipping.",
                ep.name,
                cls.exporter_id,
            )
            _record_diagnostic(
                diagnostics,
                "tributo.exporters",
                ep.name,
                f"Entry-point name {ep.name!r} != exporter_id {cls.exporter_id!r}",
            )
            continue

        classes.append(cls)
        logger.info("Discovered exporter plugin %r (%s)", ep.name, ep.value)

    return classes


def _looks_like_exporter(cls: type) -> TypeGuard[type[ModelExporter]]:
    """Check structural conformance to ModelExporter without issubclass.

    Uses manual attribute checks because ``@runtime_checkable`` protocols
    with ClassVar members don't support ``issubclass()``.
    """
    required_attrs = (
        "api_version",
        "exporter_id",
        "priority",
        "output_format",
        "options_model",
        "validator_bindings",
        "mutates_source",
    )
    return all(hasattr(cls, a) for a in required_attrs)


def resolve_hook_plugin(hook_id: str) -> type[Any]:
    """Load and validate one explicitly configured hook plugin.

    Unlike general discovery, hook resolution is fail-closed: a configured
    side effect must never disappear because an entry point failed to import.
    """
    enabled = _get_enabled_plugins()
    if enabled is not None and hook_id not in enabled:
        raise JobConfigurationError(f"Hook {hook_id!r} is disabled by TRIBUTO_PLUGINS")

    matches = [ep for ep in _iter_entry_points("tributo.hooks") if ep.name == hook_id]
    if not matches:
        raise JobConfigurationError(f"Unknown hook_id {hook_id!r}")
    if len(matches) > 1:
        raise JobConfigurationError(
            f"Multiple entry points are registered for hook_id {hook_id!r}"
        )

    ep = matches[0]
    try:
        cls = ep.load()
    except Exception as exc:
        raise JobConfigurationError(
            f"Failed to load hook {hook_id!r} ({type(exc).__name__})"
        ) from exc

    if not isinstance(cls, type):
        raise JobConfigurationError(
            f"Hook entry point {hook_id!r} must resolve to a class implementing "
            "the PublicationHook v1 contract"
        )
    contract_issues = _hook_contract_issues(cls)
    if contract_issues:
        legacy_hint = (
            "; legacy execute(canonical_uri, manifest, options, "
            "local_bundle_dir) detected; migrate to "
            "deliver(event, artifacts, options)"
            if callable(getattr(cls, "execute", None))
            else ""
        )
        raise JobConfigurationError(
            f"Hook entry point {hook_id!r} does not implement the "
            "PublicationHook v1 contract; missing or invalid members: "
            f"{', '.join(contract_issues)}{legacy_hint}"
        )
    if not _validate_api_version(cls, 1):
        raise JobConfigurationError(
            f"Hook {hook_id!r} has unsupported api_version "
            f"{getattr(cls, 'api_version', None)!r}; expected 1"
        )
    resolved_hook_id = cast(Any, cls).hook_id
    if resolved_hook_id != ep.name:
        raise JobConfigurationError(
            f"Hook entry-point name {ep.name!r} does not match hook_id "
            f"{resolved_hook_id!r}"
        )
    return cls


def _hook_contract_issues(cls: type[Any]) -> list[str]:
    issues: list[str] = []
    if not hasattr(cls, "api_version"):
        issues.append("api_version")
    if not hasattr(cls, "hook_id"):
        issues.append("hook_id")
    options_model = getattr(cls, "options_model", None)
    if not (isinstance(options_model, type) and issubclass(options_model, BaseModel)):
        issues.append("options_model (BaseModel subclass)")
    if not callable(getattr(cls, "deliver", None)):
        issues.append("deliver(event, artifacts, options)")
    if not callable(getattr(cls, "idempotency_key", None)):
        issues.append("idempotency_key(event, options)")
    return issues


def _looks_like_source_provider(cls: type) -> TypeGuard[type[ExportSourceProvider]]:
    """Check structural conformance to ExportSourceProvider without issubclass."""
    required_attrs = ("api_version", "provider_id", "trainer_type", "priority")
    return all(hasattr(cls, a) for a in required_attrs)


def _looks_like_validator(cls: type) -> TypeGuard[type[ExportValidator]]:
    """Check structural conformance to ExportValidator without issubclass."""
    required_attrs = ("api_version", "validator_id", "options_model")
    return all(hasattr(cls, a) for a in required_attrs)


def _looks_like_flavor(cls: type) -> bool:
    """Check structural conformance to ModelFlavor without issubclass."""
    return hasattr(cls, "flavor_id") and hasattr(cls, "api_version")


def _looks_like_factory(cls: type) -> TypeGuard[type[ModelFactory]]:
    """Check structural conformance to ModelFactory without issubclass."""
    return hasattr(cls, "architecture_id") and hasattr(cls, "api_version")


def discover_source_provider_plugins(
    diagnostics: list[PluginLoadDiagnostic] | None = None,
) -> list[Any]:
    """Discover third-party source providers as ``tributo.source_providers``.

    When *diagnostics* (a list) is provided, load/type/api-version/name
    failures are appended for ``SourceProviderRegistry.diagnostics()``.
    """

    enabled = _get_enabled_plugins()
    classes: list[Any] = []

    for ep in _iter_entry_points("tributo.source_providers"):
        if enabled is not None and ep.name not in enabled:
            logger.debug("Skipping source provider plugin %r", ep.name)
            continue
        try:
            cls = ep.load()
        except Exception as exc:
            logger.warning(
                "Failed to load source provider plugin %r (%s)",
                ep.name,
                ep.value,
                exc_info=True,
            )
            _record_diagnostic(
                diagnostics,
                "tributo.source_providers",
                ep.name,
                f"Failed to load entry point: {exc}",
                error_type=type(exc).__name__,
            )
            continue

        if not (isinstance(cls, type) and _looks_like_source_provider(cls)):
            logger.warning(
                "Source provider plugin %r is not an ExportSourceProvider subclass (got %r); skipping.",
                ep.name,
                cls,
            )
            _record_diagnostic(
                diagnostics,
                "tributo.source_providers",
                ep.name,
                f"Not an ExportSourceProvider subclass (got {cls!r})",
                error_type=type(cls).__name__,
            )
            continue
        if not _validate_api_version(cls, 1):
            logger.warning(
                "Source provider plugin %r unsupported api_version (got %r); skipping.",
                ep.name,
                getattr(cls, "api_version", None),
            )
            _record_diagnostic(
                diagnostics,
                "tributo.source_providers",
                ep.name,
                f"Unsupported api_version {getattr(cls, 'api_version', None)!r}",
            )
            continue
        if ep.name != cls.provider_id:
            logger.warning(
                "Source provider entry-point name %r != provider_id %r; skipping.",
                ep.name,
                cls.provider_id,
            )
            _record_diagnostic(
                diagnostics,
                "tributo.source_providers",
                ep.name,
                f"Entry-point name {ep.name!r} != provider_id {cls.provider_id!r}",
            )
            continue

        classes.append(cls)
        logger.info("Discovered source provider plugin %r (%s)", ep.name, ep.value)

    return classes


def discover_validator_plugins(
    diagnostics: list[PluginLoadDiagnostic] | None = None,
) -> list[Any]:
    """Discover third-party validators as ``tributo.validators``.

    When *diagnostics* (a list) is provided, load/type/api-version/name
    failures are appended for ``ValidatorRegistry.diagnostics()``.
    """

    enabled = _get_enabled_plugins()
    classes: list[Any] = []

    for ep in _iter_entry_points("tributo.validators"):
        if enabled is not None and ep.name not in enabled:
            logger.debug("Skipping validator plugin %r", ep.name)
            continue
        try:
            cls = ep.load()
        except Exception as exc:
            logger.warning(
                "Failed to load validator plugin %r (%s)",
                ep.name,
                ep.value,
                exc_info=True,
            )
            _record_diagnostic(
                diagnostics,
                "tributo.validators",
                ep.name,
                f"Failed to load entry point: {exc}",
                error_type=type(exc).__name__,
            )
            continue

        if not (isinstance(cls, type) and _looks_like_validator(cls)):
            logger.warning(
                "Validator plugin %r is not an ExportValidator subclass (got %r); skipping.",
                ep.name,
                cls,
            )
            _record_diagnostic(
                diagnostics,
                "tributo.validators",
                ep.name,
                f"Not an ExportValidator subclass (got {cls!r})",
                error_type=type(cls).__name__,
            )
            continue
        if not _validate_api_version(cls, 1):
            logger.warning(
                "Validator plugin %r unsupported api_version (got %r); skipping.",
                ep.name,
                getattr(cls, "api_version", None),
            )
            _record_diagnostic(
                diagnostics,
                "tributo.validators",
                ep.name,
                f"Unsupported api_version {getattr(cls, 'api_version', None)!r}",
            )
            continue
        if ep.name != cls.validator_id:
            logger.warning(
                "Validator entry-point name %r != validator_id %r; skipping.",
                ep.name,
                cls.validator_id,
            )
            _record_diagnostic(
                diagnostics,
                "tributo.validators",
                ep.name,
                f"Entry-point name {ep.name!r} != validator_id {cls.validator_id!r}",
            )
            continue

        classes.append(cls)
        logger.info("Discovered validator plugin %r (%s)", ep.name, ep.value)

    return classes


def discover_flavor_plugins() -> list[Any]:
    """Discover third-party model flavors as ``tributo.model_flavors``.

    Each entry point must point to a ``ModelFlavor`` subclass with a
    ``flavor_id`` class variable and ``api_version == 1``.
    """

    enabled = _get_enabled_plugins()
    classes: list[Any] = []

    for ep in _iter_entry_points("tributo.model_flavors"):
        if enabled is not None and ep.name not in enabled:
            logger.debug("Skipping flavor plugin %r", ep.name)
            continue
        try:
            cls = ep.load()
        except Exception:
            logger.warning(
                "Failed to load flavor plugin %r (%s)", ep.name, ep.value, exc_info=True
            )
            continue

        if not (isinstance(cls, type) and _looks_like_flavor(cls)):
            logger.warning(
                "Flavor plugin %r is not a ModelFlavor subclass (got %r); skipping.",
                ep.name,
                cls,
            )
            continue
        if not _validate_api_version(cls, 1):
            logger.warning(
                "Flavor plugin %r unsupported api_version (got %r); skipping.",
                ep.name,
                getattr(cls, "api_version", None),
            )
            continue

        fid = getattr(cls, "flavor_id", None)
        if ep.name != fid:
            logger.warning(
                "Flavor entry-point name %r != flavor_id %r; skipping.",
                ep.name,
                fid,
            )
            continue

        classes.append(cls)
        logger.info("Discovered flavor plugin %r (%s)", ep.name, ep.value)

    return classes


def discover_model_factory_plugins() -> list[Any]:
    """Discover third-party model factories as ``tributo.model_factories``."""

    enabled = _get_enabled_plugins()
    classes: list[Any] = []

    for ep in _iter_entry_points("tributo.model_factories"):
        if enabled is not None and ep.name not in enabled:
            logger.debug("Skipping model factory plugin %r", ep.name)
            continue
        try:
            cls = ep.load()
        except Exception:
            logger.warning(
                "Failed to load model factory plugin %r (%s)",
                ep.name,
                ep.value,
                exc_info=True,
            )
            continue

        if not (isinstance(cls, type) and _looks_like_factory(cls)):
            logger.warning(
                "Model factory plugin %r is not a ModelFactory subclass (got %r); skipping.",
                ep.name,
                cls,
            )
            continue
        if not _validate_api_version(cls, 1):
            logger.warning(
                "Model factory plugin %r unsupported api_version (got %r); skipping.",
                ep.name,
                getattr(cls, "api_version", None),
            )
            continue
        if ep.name != cls.architecture_id:
            logger.warning(
                "Model factory entry-point name %r != architecture_id %r; skipping.",
                ep.name,
                cls.architecture_id,
            )
            continue

        classes.append(cls)
        logger.info("Discovered model factory plugin %r (%s)", ep.name, ep.value)

    return classes
