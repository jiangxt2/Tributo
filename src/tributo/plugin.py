"""Third-party plugin discovery via ``importlib.metadata`` entry points.

Tributo discovers external plugins through standard Python entry points
declared in ``pyproject.toml`` under the ``[project.entry-points]`` table.

Discovery groups:
    ``tributo.algorithms``
        Entries expose one immutable ``DistributedAlgorithmDescriptor`` for a
        trusted, pre-installed package. Tributo does not install dependencies
        or manage plugin lifecycles.

    ``tributo.trainers``
        New entries expose a lightweight ``LegacyTrainerDescriptor`` through
        an explicit attribute. Module-only entries with a ``trainer_spec``
        remain available only through the Beta compatibility API. Example::

            [project.entry-points."tributo.trainers"]
            my_algo = "my_package.descriptors:MY_ALGO_DESCRIPTOR"

    ``tributo.bundle_repositories`` / ``tributo.bundle_alias_stores``
        Each entry point must point to a storage adapter class implementing
        the corresponding versioned repository protocol. Adapter constructors
        must accept the keyword argument ``storage_resolver``.

Filtering:
    Set the ``TRIBUTO_PLUGINS`` environment variable to a comma-separated
    list of entry point names to load only those plugins.  If unset, all
    discovered plugins are loaded.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from importlib.metadata import entry_points
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, TypeGuard, cast

from packaging.utils import canonicalize_name
from pydantic import BaseModel

from tributo.exceptions import JobConfigurationError
from tributo.exporting.models import PluginLoadDiagnostic
from tributo.exporting.protocols import (
    ExportSourceProvider,
    ExportValidator,
    ModelExporter,
    ModelFactory,
)
from tributo.util.annotations import PublicAPI

if TYPE_CHECKING:
    from tributo.algorithms.api import DistributedAlgorithmDescriptor

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


@dataclass(frozen=True)
class _PluginSelector:
    distribution: str | None = None
    group: str | None = None
    name: str | None = None


def _parse_plugin_selector(value: str) -> _PluginSelector:
    parts = value.split(":")
    if len(parts) == 1 and parts[0]:
        return _PluginSelector(name=parts[0])
    if len(parts) == 2 and parts[0] == "distribution" and parts[1]:
        return _PluginSelector(distribution=canonicalize_name(parts[1]))
    if len(parts) == 3 and parts[0] == "group" and all(parts[1:]):
        return _PluginSelector(group=parts[1], name=parts[2])
    if (
        len(parts) == 5
        and parts[0] == "distribution"
        and parts[2] == "group"
        and all((parts[1], parts[3], parts[4]))
    ):
        return _PluginSelector(
            distribution=canonicalize_name(parts[1]),
            group=parts[3],
            name=parts[4],
        )
    raise JobConfigurationError(
        "TRIBUTO_PLUGINS selectors must be a bare name, distribution:<name>, "
        "group:<group>:<name>, or "
        "distribution:<name>:group:<group>:<name>"
    )


def _get_enabled_plugins() -> tuple[_PluginSelector, ...] | None:
    """Parse ``TRIBUTO_PLUGINS`` into deterministic qualified selectors."""
    raw = os.environ.get("TRIBUTO_PLUGINS", "").strip()
    if not raw:
        return None
    selectors = tuple(
        _parse_plugin_selector(item.strip()) for item in raw.split(",") if item.strip()
    )
    return tuple(
        sorted(
            set(selectors),
            key=lambda item: (
                item.distribution or "",
                item.group or "",
                item.name or "",
            ),
        )
    )


def _entry_point_enabled(
    entry_point: Any,
    group: str,
    selectors: tuple[_PluginSelector, ...] | None,
) -> bool:
    if selectors is None:
        return True
    distribution = _entry_point_distribution_name(entry_point)
    return any(
        (selector.distribution is None or selector.distribution == distribution)
        and (selector.group is None or selector.group == group)
        and (selector.name is None or selector.name == entry_point.name)
        for selector in selectors
    )


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
        if not _entry_point_enabled(ep, "tributo.trainers", enabled):
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


@PublicAPI(stability="alpha")
def validate_distributed_algorithm_descriptor(
    descriptor: object,
    *,
    entry_point_name: str,
    entry_point_distribution_name: str | None = None,
    load_implementation: bool = True,
) -> DistributedAlgorithmDescriptor:
    """Validate one installed, trusted distributed algorithm descriptor."""
    import importlib
    import importlib.metadata

    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    from tributo.algorithms.api import (
        DistributedAlgorithmDescriptor,
        DistributionStrategy,
        InputDistribution,
    )
    from tributo.algorithms.api.models import FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS
    from tributo.algorithms.spi import (
        CollectiveAlgorithm,
        FrameworkNativeAlgorithm,
        IterativeOptimizationAlgorithm,
        JoblibEstimatorRecipe,
        MapReduceAlgorithm,
        ParallelEnsembleAlgorithm,
        TorchTrainingRecipe,
        TrainingRecipeV2,
    )

    if not isinstance(descriptor, DistributedAlgorithmDescriptor):
        raise TypeError("entry point does not export DistributedAlgorithmDescriptor")
    if descriptor.name != entry_point_name and not entry_point_name.startswith(
        f"{descriptor.name}."
    ):
        raise ValueError(
            "descriptor entry-point name must equal the algorithm identity or use "
            "the '<algorithm>.<implementation>' form"
        )
    owner_distribution = canonicalize_name(entry_point_distribution_name or "")
    if owner_distribution and owner_distribution != descriptor.package_name:
        raise ValueError(
            "descriptor package identity does not match entry-point distribution"
        )
    installed_package_version = importlib.metadata.version(descriptor.package_name)
    if Version(installed_package_version) != Version(descriptor.package_version):
        raise ValueError("descriptor package version does not match installed metadata")
    installed_tributo_version = importlib.metadata.version("tributo")
    if Version(installed_tributo_version) not in SpecifierSet(
        descriptor.tributo_version_spec
    ):
        raise ValueError(
            "installed Tributo version is outside descriptor compatibility"
        )

    registration = descriptor.registration
    distribution_spec = registration.distribution_spec
    if distribution_spec is None:
        raise TypeError("distributed descriptor lost its DistributionSpec")
    implementation_descriptor = registration.implementation
    expected_base: type = {
        DistributionStrategy.RAY_TRAIN_COLLECTIVE: CollectiveAlgorithm,
        DistributionStrategy.RAY_MAP_REDUCE: MapReduceAlgorithm,
        DistributionStrategy.FRAMEWORK_NATIVE: FrameworkNativeAlgorithm,
        DistributionStrategy.RAY_JOBLIB_ESTIMATOR: JoblibEstimatorRecipe,
        DistributionStrategy.RAY_PARALLEL_ENSEMBLE: ParallelEnsembleAlgorithm,
        DistributionStrategy.RAY_ITERATIVE_OPTIMIZATION: (
            IterativeOptimizationAlgorithm
        ),
        DistributionStrategy.RAY_TRAIN_RECIPE_V2: TrainingRecipeV2,
    }[distribution_spec.strategy]
    if distribution_spec.strategy is DistributionStrategy.RAY_TRAIN_COLLECTIVE and str(
        implementation_descriptor.executable_factory_ref
    ) == (
        "tributo.integrations.algorithm_runtimes.torch_recipe:"
        "create_torch_recipe_algorithm"
    ):
        expected_base = TorchTrainingRecipe
    contract = FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS[distribution_spec.strategy]
    if implementation_descriptor.execution_mode is not contract.execution_mode:
        raise ValueError("implementation execution mode conflicts with strategy")
    if implementation_descriptor.runtime_id != contract.runtime_id:
        raise ValueError("implementation runtime_id conflicts with strategy")
    worker_adapter = implementation_descriptor.worker_input_adapter_ref
    if (
        worker_adapter is None
        or str(worker_adapter) != contract.worker_input_adapter_ref
    ):
        raise ValueError("implementation Worker input adapter conflicts with strategy")
    compatibility = implementation_descriptor.input_compatibility
    if compatibility.distribution_policy != (contract.topology,):
        raise ValueError("implementation input topology conflicts with strategy")
    if worker_adapter not in compatibility.supported_explicit_adapters:
        raise ValueError(
            "implementation input compatibility omits its Worker input adapter"
        )
    required_capability = (
        "shardable"
        if contract.input_distribution is not InputDistribution.FULL_DATASET
        else "materializable"
    )
    if (
        "ray_data" not in compatibility.accepted_input_views
        or "tributo.ray_data" not in compatibility.accepted_ingestion_engines
        or required_capability not in compatibility.required_input_capabilities
    ):
        raise ValueError(
            "implementation input compatibility omits the required Ray Data contract: "
            f"{required_capability}"
        )

    if load_implementation:
        reference = implementation_descriptor.implementation_ref
        implementation: object = importlib.import_module(reference.module)
        for segment in reference.qualname.split("."):
            implementation = getattr(implementation, segment)
        if not isinstance(implementation, type) or not issubclass(
            implementation, expected_base
        ):
            raise TypeError(f"implementation must inherit {expected_base.__name__}")
    return descriptor


def discover_algorithm_descriptors(
    diagnostics: list[PluginLoadDiagnostic] | None = None,
) -> list[Any]:
    """Discover and validate constrained distributed algorithm descriptors."""

    enabled = _get_enabled_plugins()
    descriptors: list[Any] = []
    for ep in _iter_entry_points("tributo.algorithms"):
        if not _entry_point_enabled(ep, "tributo.algorithms", enabled):
            continue
        try:
            descriptor = validate_distributed_algorithm_descriptor(
                ep.load(),
                entry_point_name=ep.name,
                entry_point_distribution_name=_entry_point_distribution_name(ep),
                load_implementation=False,
            )
        except Exception as exc:
            _record_diagnostic(
                diagnostics,
                "tributo.algorithms",
                ep.name,
                "Distributed algorithm descriptor failed conformance validation",
                error_type=type(exc).__name__,
            )
            logger.warning(
                "Failed to load distributed algorithm descriptor %r (%s)",
                ep.name,
                ep.value,
                exc_info=True,
            )
            continue
        descriptors.append(descriptor)
    return descriptors


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
    """Iterate over entry points using deterministic owner/name ordering."""
    yield from sorted(
        entry_points(group=group),
        key=lambda ep: (
            _entry_point_distribution_name(ep),
            ep.name,
            ep.value,
        ),
    )


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
    ``api_version == 2``.  When *diagnostics* (a list) is provided,
    import/type/api-version/name failures are appended to it so they can
    be queried via ``ExportRegistry.diagnostics()``.
    """
    enabled = _get_enabled_plugins()
    classes: list[Any] = []

    for ep in _iter_entry_points("tributo.exporters"):
        if not _entry_point_enabled(ep, "tributo.exporters", enabled):
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

        if not isinstance(cls, type):
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
        if not _validate_api_version(cls, 2):
            logger.warning(
                "Exporter plugin %r has unsupported api_version "
                "(got %r, expected 2); skipping.",
                ep.name,
                getattr(cls, "api_version", None),
            )
            _record_diagnostic(
                diagnostics,
                "tributo.exporters",
                ep.name,
                "Unsupported ModelExporter api_version "
                f"{getattr(cls, 'api_version', None)!r}; expected 2",
            )
            continue
        missing_attrs = _missing_exporter_attributes(cls)
        if missing_attrs:
            logger.warning(
                "Exporter plugin %r is missing ModelExporter v2 attributes %s; "
                "skipping.",
                ep.name,
                missing_attrs,
            )
            _record_diagnostic(
                diagnostics,
                "tributo.exporters",
                ep.name,
                "Missing ModelExporter v2 attributes: " + ", ".join(missing_attrs),
            )
            continue
        exporter_cls = cast(type[ModelExporter], cls)
        if ep.name != exporter_cls.exporter_id:
            logger.warning(
                "Exporter plugin entry-point name %r != exporter_id %r; skipping.",
                ep.name,
                exporter_cls.exporter_id,
            )
            _record_diagnostic(
                diagnostics,
                "tributo.exporters",
                ep.name,
                "Entry-point name "
                f"{ep.name!r} != exporter_id {exporter_cls.exporter_id!r}",
            )
            continue

        classes.append(exporter_cls)
        logger.info("Discovered exporter plugin %r (%s)", ep.name, ep.value)

    return classes


def _missing_exporter_attributes(cls: type) -> tuple[str, ...]:
    """Return missing structural attributes for the ModelExporter v2 contract."""
    required_attrs = (
        "api_version",
        "exporter_id",
        "priority",
        "output_format",
        "output_flavor_id",
        "source_kinds",
        "options_model",
        "validator_bindings",
        "mutates_source",
        "upstream_requirements",
        "supports",
        "export",
    )
    return tuple(attr for attr in required_attrs if not hasattr(cls, attr))


def resolve_hook_plugin(hook_id: str) -> type[Any]:
    """Load and validate one explicitly configured hook plugin.

    Unlike general discovery, hook resolution is fail-closed: a configured
    side effect must never disappear because an entry point failed to import.
    """
    enabled = _get_enabled_plugins()
    all_matches = [
        ep for ep in _iter_entry_points("tributo.hooks") if ep.name == hook_id
    ]
    matches = [
        ep for ep in all_matches if _entry_point_enabled(ep, "tributo.hooks", enabled)
    ]
    if not matches:
        if all_matches:
            raise JobConfigurationError(
                f"Hook {hook_id!r} is disabled by TRIBUTO_PLUGINS"
            )
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
    required_attrs = (
        "api_version",
        "flavor_id",
        "supported_formats",
        "batch_supported",
        "serveable",
        "security_mode",
        "signature_required",
        "required_dependencies",
        "load",
    )
    return all(hasattr(cls, attr) for attr in required_attrs)


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
        if not _entry_point_enabled(ep, "tributo.source_providers", enabled):
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
        if not _entry_point_enabled(ep, "tributo.validators", enabled):
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
        if not _entry_point_enabled(ep, "tributo.model_flavors", enabled):
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
        if not _entry_point_enabled(ep, "tributo.model_factories", enabled):
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


def discover_bundle_repository_plugins() -> list[Any]:
    """Discover ``BundleRepository`` adapter classes with API version 1."""
    return _discover_storage_adapter_plugins(
        group="tributo.bundle_repositories",
        identity_attribute="repository_id",
        required_methods=("commit", "read_manifest", "materialize_artifact"),
    )


def discover_bundle_alias_store_plugins() -> list[Any]:
    """Discover ``BundleAliasStore`` adapter classes with API version 1."""
    return _discover_storage_adapter_plugins(
        group="tributo.bundle_alias_stores",
        identity_attribute="alias_store_id",
        required_methods=("is_alias_uri", "resolve", "update"),
    )


def _accepts_storage_resolver(cls: type[Any]) -> bool:
    """Return whether an adapter honors the API v1 constructor contract."""
    try:
        parameter = signature(cls).parameters.get("storage_resolver")
    except (TypeError, ValueError):
        return False
    return parameter is not None and parameter.kind in {
        Parameter.POSITIONAL_OR_KEYWORD,
        Parameter.KEYWORD_ONLY,
    }


def _discover_storage_adapter_plugins(
    *,
    group: str,
    identity_attribute: str,
    required_methods: tuple[str, ...],
) -> list[Any]:
    classes: list[Any] = []
    for ep in _iter_entry_points(group):
        try:
            cls = ep.load()
        except Exception:
            logger.warning(
                "Failed to load storage adapter %r (%s)",
                ep.name,
                ep.value,
                exc_info=True,
            )
            continue
        adapter_id = getattr(cls, identity_attribute, None)
        schemes = getattr(cls, "schemes", None)
        if not (
            isinstance(cls, type)
            and _validate_api_version(cls, 1)
            and isinstance(adapter_id, str)
            and isinstance(schemes, tuple)
            and bool(schemes)
            and all(isinstance(scheme, str) for scheme in schemes)
            and all(callable(getattr(cls, method, None)) for method in required_methods)
            and _accepts_storage_resolver(cls)
        ):
            logger.warning(
                "Storage adapter %r does not satisfy the API v1 contract; skipping.",
                ep.name,
            )
            continue
        if ep.name != adapter_id:
            logger.warning(
                "Storage adapter entry-point name %r != adapter id %r; skipping.",
                ep.name,
                adapter_id,
            )
            continue
        classes.append(cls)
        logger.info("Discovered storage adapter %r (%s)", ep.name, ep.value)
    return classes


# ═══════════════════════════════════════════════════════════════════════════════
# Broker plugins
# ═══════════════════════════════════════════════════════════════════════════════


def _broker_contract_issues(cls: Any) -> tuple[str, ...]:
    """Return structural API issues without instantiating a provider."""
    issues: list[str] = []
    if not isinstance(cls, type):
        return ("provider class",)
    if type(getattr(cls, "api_version", None)) is not int:
        issues.append("api_version")
    broker_id = getattr(cls, "broker_id", None)
    if not isinstance(broker_id, str) or not broker_id.strip():
        issues.append("broker_id")
    capabilities = getattr(cls, "capabilities", None)
    if not isinstance(capabilities, frozenset) or not all(
        isinstance(value, str) and bool(value.strip()) for value in capabilities
    ):
        issues.append("capabilities")
    if getattr(cls, "stability", None) not in {"alpha", "beta", "stable"}:
        issues.append("stability")
    for method in (
        "validate_config",
        "create_runtime",
    ):
        if not callable(getattr(cls, method, None)):
            issues.append(method)
    return tuple(issues)


def discover_broker_plugins(
    diagnostics: list[PluginLoadDiagnostic] | None = None,
) -> list[type[Any]]:
    """Discover broker provider classes from ``tributo.brokers``.

    Discovery is fail-open and never instantiates a provider.  In particular,
    it cannot create a Redis client or perform a network probe.  Explicit
    resolution is provided by :func:`resolve_broker_plugin` and is
    fail-closed.
    """
    from tributo.integrations.broker import BROKER_API_VERSION

    enabled = _get_enabled_plugins()
    classes: list[type[Any]] = []
    for ep in _iter_entry_points("tributo.brokers"):
        if not _entry_point_enabled(ep, "tributo.brokers", enabled):
            logger.debug("Skipping broker plugin %r", ep.name)
            continue
        try:
            cls = ep.load()
        except Exception as exc:
            logger.warning(
                "Failed to load broker plugin %r (%s; %s)",
                ep.name,
                ep.value,
                type(exc).__name__,
            )
            _record_diagnostic(
                diagnostics,
                "tributo.brokers",
                ep.name,
                f"Failed to load entry point ({type(exc).__name__})",
                error_type=type(exc).__name__,
            )
            continue

        issues = _broker_contract_issues(cls)
        if issues:
            logger.warning(
                "Broker plugin %r does not satisfy API v%d: %s",
                ep.name,
                BROKER_API_VERSION,
                ", ".join(issues),
            )
            _record_diagnostic(
                diagnostics,
                "tributo.brokers",
                ep.name,
                "Missing or invalid BrokerPlugin members: " + ", ".join(issues),
            )
            continue
        if cls.api_version != BROKER_API_VERSION:
            reason = (
                f"Unsupported BrokerPlugin api_version {cls.api_version!r}; "
                f"expected {BROKER_API_VERSION}"
            )
            _record_diagnostic(diagnostics, "tributo.brokers", ep.name, reason)
            logger.warning("Broker plugin %r: %s", ep.name, reason)
            continue
        if ep.name != cls.broker_id:
            reason = (
                f"Entry-point name {ep.name!r} does not match broker_id "
                f"{cls.broker_id!r}"
            )
            _record_diagnostic(diagnostics, "tributo.brokers", ep.name, reason)
            logger.warning("Broker plugin %r: %s", ep.name, reason)
            continue
        classes.append(cls)
        logger.info("Discovered broker plugin %r (%s)", ep.name, ep.value)
    return classes


def resolve_broker_plugin(broker_id: str) -> type[Any]:
    """Resolve one explicitly selected broker, using fail-closed semantics."""
    enabled = _get_enabled_plugins()
    all_matches = [
        ep for ep in _iter_entry_points("tributo.brokers") if ep.name == broker_id
    ]
    matches = [
        ep for ep in all_matches if _entry_point_enabled(ep, "tributo.brokers", enabled)
    ]
    if not matches:
        if all_matches:
            raise JobConfigurationError(
                f"Broker {broker_id!r} is disabled by TRIBUTO_PLUGINS"
            )
        raise JobConfigurationError(f"Unknown broker {broker_id!r}")
    if len(matches) > 1:
        raise JobConfigurationError(
            f"Multiple entry points are registered for broker {broker_id!r}"
        )

    ep = matches[0]
    try:
        cls = cast(type[Any], ep.load())
    except Exception as exc:
        raise JobConfigurationError(
            f"Failed to load broker {broker_id!r} ({type(exc).__name__})"
        ) from exc

    issues = _broker_contract_issues(cls)
    from tributo.integrations.broker import BROKER_API_VERSION

    if issues:
        raise JobConfigurationError(
            f"Broker {broker_id!r} does not implement the BrokerPlugin v"
            f"{BROKER_API_VERSION} contract: {', '.join(issues)}"
        )
    if cls.api_version != BROKER_API_VERSION:
        raise JobConfigurationError(
            f"Broker {broker_id!r} has unsupported api_version "
            f"{cls.api_version!r}; expected {BROKER_API_VERSION}"
        )
    if cls.broker_id != ep.name:
        raise JobConfigurationError(
            f"Broker entry-point name {ep.name!r} does not match broker_id "
            f"{cls.broker_id!r}"
        )
    return cast(type[Any], cls)
