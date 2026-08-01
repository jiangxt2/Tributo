"""Third-party plugin discovery via ``importlib.metadata`` entry points.

Tributo discovers external plugins through standard Python entry points
declared in ``pyproject.toml`` under the ``[project.entry-points]`` table.

Discovery groups:
    ``tributo.trainers``
        Each entry point must point to a module with a ``trainer_spec``
        attribute of type ``TrainerSpec``.  Example::

            [project.entry-points."tributo.trainers"]
            my_algo = "my_package.trainer"

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
from typing import Any, TypeGuard

from tributo.exporting.models import PluginLoadDiagnostic
from tributo.exporting.protocols import (
    ExportValidator,
    ModelExporter,
    ModelFactory,
    SourceProvider,
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


def discover_trainer_plugins() -> list[Any]:
    """Discover third-party trainers registered via entry_points.

    Each entry point is expected to point to a module whose top-level
    ``trainer_spec`` attribute is a :class:`TrainerSpec` instance.
    """
    from tributo.training.base import TrainerSpec

    enabled = _get_enabled_plugins()
    specs: list[Any] = []

    for ep in _iter_entry_points("tributo.trainers"):
        if enabled is not None and ep.name not in enabled:
            logger.debug("Skipping trainer plugin %r (not in TRIBUTO_PLUGINS)", ep.name)
            continue
        try:
            mod = ep.load()
        except Exception:
            logger.warning(
                "Failed to load trainer plugin %r (%s)",
                ep.name,
                ep.value,
                exc_info=True,
            )
            continue

        spec = getattr(mod, "trainer_spec", None)
        if not isinstance(spec, TrainerSpec):
            logger.warning(
                "Trainer plugin %r does not export a TrainerSpec instance as "
                "%r.trainer_spec (got %r); skipping.",
                ep.name,
                ep.value,
                type(spec).__name__,
            )
            continue
        specs.append(spec)
        logger.info("Discovered trainer plugin %r (%s)", ep.name, ep.value)

    return specs


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


def _iter_entry_points(group: str) -> Any:
    """Iterate over entry points for *group*, sorted by name.

    Deterministic order makes plugin loading (and any resulting
    candidate ordering) reproducible across runs.
    """
    eps = sorted(entry_points(group=group), key=lambda ep: ep.name)
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


def _looks_like_source_provider(cls: type) -> TypeGuard[type[SourceProvider]]:
    """Check structural conformance to SourceProvider without issubclass."""
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
                "Source provider plugin %r is not a SourceProvider subclass (got %r); skipping.",
                ep.name,
                cls,
            )
            _record_diagnostic(
                diagnostics,
                "tributo.source_providers",
                ep.name,
                f"Not a SourceProvider subclass (got {cls!r})",
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
