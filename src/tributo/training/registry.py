"""Beta Trainer façade over the unified portable algorithm Registry."""

from __future__ import annotations

import importlib
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from packaging.utils import canonicalize_name

from tributo.algorithms.api import (
    AlgorithmRegistration,
    AlgorithmResolutionError,
    AlgorithmSupportEvidence,
    AlgorithmSupportEvidenceRegistry,
    DistributedAlgorithmDescriptor,
)
from tributo.algorithms.core import AlgorithmRegistrationRegistry
from tributo.exceptions import JobConfigurationError
from tributo.exporting.models import PluginLoadDiagnostic
from tributo.integrations.algorithm_runtimes.legacy_descriptors import (
    BUILTIN_LEGACY_DESCRIPTORS,
    LegacyTrainerDescriptor,
    build_legacy_spec,
)
from tributo.training.algorithm_spec import AlgorithmSpec, ExecutionKind
from tributo.training.base import BaseTrainer, TrainerSpec
from tributo.util.annotations import DeveloperAPI, PublicAPI


@DeveloperAPI
@dataclass(frozen=True)
class AlgorithmRegistryEntry:
    """One immutable Catalog projection from the unified composition root."""

    name: str
    spec: AlgorithmSpec | None
    registrations: tuple[AlgorithmRegistration, ...]
    stability: str
    available: bool
    compatibility_only: bool
    tested: bool
    supported: bool
    validated_execution_profiles: tuple[str, ...]
    native_migration_complete: bool
    limitations: tuple[str, ...]


def _validate_spec(spec: TrainerSpec) -> None:
    """Validate that a Beta TrainerSpec agrees with its Trainer hierarchy."""
    trainer_cls = spec.trainer_cls
    if not isinstance(trainer_cls, type):
        raise TypeError(
            f"Algorithm {spec.name!r}: trainer_cls must be a class, "
            f"got {type(trainer_cls).__name__!r}"
        )
    if not issubclass(trainer_cls, BaseTrainer):
        raise TypeError(
            f"Algorithm {spec.name!r}: trainer_cls {trainer_cls.__name__} "
            "must be a BaseTrainer subclass."
        )
    from tributo.training.causal_estimator import BaseCausalEstimator

    is_causal = issubclass(trainer_cls, BaseCausalEstimator)
    if spec.execution_kind == ExecutionKind.ESTIMATE and not is_causal:
        raise TypeError(
            f"Algorithm {spec.name!r}: execution_kind=ESTIMATE requires "
            f"a BaseCausalEstimator subclass, got {trainer_cls.__name__}."
        )
    if spec.execution_kind != ExecutionKind.ESTIMATE and is_causal:
        raise TypeError(
            f"Algorithm {spec.name!r}: execution_kind={spec.execution_kind} "
            f"cannot use BaseCausalEstimator subclass {trainer_cls.__name__}; "
            "causal estimators run the ESTIMATE lifecycle."
        )


def _load_reference(module: str, qualname: str) -> object:
    try:
        value: object = importlib.import_module(module)
        for segment in qualname.split("."):
            value = getattr(value, segment)
        return value
    except Exception as exc:
        raise JobConfigurationError(
            f"Failed to load explicitly requested Trainer compatibility reference "
            f"{module}:{qualname}"
        ) from exc


class TrainingAlgorithmRegistry:
    """Own executable facts and derive all legacy compatibility projections."""

    def __init__(
        self,
        *,
        support_evidence: AlgorithmSupportEvidenceRegistry | None = None,
        installed_wheel_digests: Mapping[str, str] | None = None,
    ) -> None:
        self._execution_registry = AlgorithmRegistrationRegistry()
        self._descriptors: dict[str, LegacyTrainerDescriptor] = {}
        self._distributed_descriptors: dict[str, DistributedAlgorithmDescriptor] = {}
        self._compatibility_entry_points: dict[str, Any] = {}
        self._hydrated_specs: dict[str, AlgorithmSpec] = {}
        self._diagnostics: tuple[PluginLoadDiagnostic, ...] = ()
        self._support_evidence = (
            support_evidence
            if support_evidence is not None
            else AlgorithmSupportEvidenceRegistry()
        )
        self._installed_wheel_digests = {
            canonicalize_name(name): digest
            for name, digest in (installed_wheel_digests or {}).items()
        }
        self._bootstrapped = False
        self._lock = threading.Lock()

    def _ensure_bootstrapped(self) -> None:
        if self._bootstrapped:
            return
        with self._lock:
            if self._bootstrapped:
                return
            from tributo.plugin import (
                discover_algorithm_descriptors,
                discover_trainer_descriptors,
            )

            diagnostics: list[PluginLoadDiagnostic] = []
            compatibility_entry_points: list[Any] = []
            discovered = discover_trainer_descriptors(
                diagnostics=diagnostics,
                compatibility_entry_points=compatibility_entry_points,
            )
            discovered_algorithms = discover_algorithm_descriptors(
                diagnostics=diagnostics
            )
            descriptors: dict[str, LegacyTrainerDescriptor] = {}
            for descriptor in (*BUILTIN_LEGACY_DESCRIPTORS, *discovered):
                existing = descriptors.get(descriptor.name)
                if existing is not None and existing != descriptor:
                    raise AlgorithmResolutionError(
                        f"algorithm {descriptor.name!r} has conflicting descriptors"
                    )
                descriptors[descriptor.name] = descriptor

            compatibility: dict[str, Any] = {}
            for entry_point in compatibility_entry_points:
                name = entry_point.name
                if name in descriptors or name in compatibility:
                    raise AlgorithmResolutionError(
                        f"algorithm plugin identity is duplicated: {name!r}"
                    )
                compatibility[name] = entry_point
            conflicts = sorted(
                set(self._execution_registry.compatibility_snapshot())
                & set(descriptors)
            )
            if conflicts:
                raise AlgorithmResolutionError(
                    f"legacy compatibility registrations conflict with descriptors: {conflicts}"
                )

            distributed_descriptors: dict[str, DistributedAlgorithmDescriptor] = {}
            for descriptor in discovered_algorithms:
                key = descriptor.registration.implementation.implementation_id
                distributed_existing = distributed_descriptors.get(key)
                if (
                    distributed_existing is not None
                    and distributed_existing != descriptor
                ):
                    raise AlgorithmResolutionError(
                        f"distributed implementation {key!r} has conflicting descriptors"
                    )
                distributed_descriptors[key] = descriptor

            self._execution_registry.register_many(
                (
                    *tuple(
                        descriptor.registration
                        for descriptor in sorted(
                            descriptors.values(), key=lambda item: item.name
                        )
                    ),
                    *tuple(
                        descriptor.registration
                        for descriptor in sorted(
                            distributed_descriptors.values(),
                            key=lambda item: (
                                item.name,
                                item.registration.implementation.implementation_id,
                            ),
                        )
                    ),
                )
            )
            self._descriptors = descriptors
            self._distributed_descriptors = distributed_descriptors
            self._compatibility_entry_points = compatibility
            self._diagnostics = tuple(diagnostics)
            self._bootstrapped = True

    def register(
        self,
        key: str | AlgorithmSpec,
        spec: AlgorithmSpec | None = None,
    ) -> None:
        """Register one Beta-only TrainerSpec compatibility fact."""
        resolved = spec
        if resolved is None:
            if not isinstance(key, AlgorithmSpec):
                raise TypeError("Trainer registry requires an AlgorithmSpec")
            resolved = key
            key = resolved.name
        if not isinstance(resolved, AlgorithmSpec) or key != resolved.name:
            raise TypeError("Trainer registry key must match AlgorithmSpec.name")
        self._ensure_bootstrapped()
        compatibility = self._execution_registry.compatibility_snapshot()
        with self._lock:
            available = sorted(
                {
                    *compatibility,
                    *self._descriptors,
                    *(item.name for item in self._distributed_descriptors.values()),
                    *self._compatibility_entry_points,
                }
            )
            if key in available:
                raise JobConfigurationError(
                    f"Trainer {key!r} already registered. Available: {available}"
                )
        self._execution_registry.register_compatibility(resolved)

    def unregister(self, key: str) -> None:
        """Remove one programmatic Beta compatibility registration."""
        with self._lock:
            self._hydrated_specs.pop(key, None)
        self._execution_registry.unregister_compatibility(key)

    def _legacy_spec(self, descriptor: LegacyTrainerDescriptor) -> AlgorithmSpec:
        trainer_cls = _load_reference(
            descriptor.trainer_ref.module,
            descriptor.trainer_ref.qualname,
        )
        config_model = _load_reference(
            descriptor.config_model_ref.module,
            descriptor.config_model_ref.qualname,
        )
        if not isinstance(trainer_cls, type) or not issubclass(
            trainer_cls, BaseTrainer
        ):
            raise JobConfigurationError(
                f"Algorithm {descriptor.name!r} Trainer reference is invalid"
            )
        if not isinstance(config_model, type):
            raise JobConfigurationError(
                f"Algorithm {descriptor.name!r} config model reference is invalid"
            )
        return build_legacy_spec(
            descriptor,
            trainer_cls=trainer_cls,
            config_model=config_model,
        )

    def _load_compatibility_plugin(self, name: str, entry_point: Any) -> AlgorithmSpec:
        try:
            module = entry_point.load()
        except Exception as exc:
            raise JobConfigurationError(
                f"Compatibility-only Trainer plugin {name!r} could not be loaded"
            ) from exc
        spec = getattr(module, "trainer_spec", module)
        if not isinstance(spec, AlgorithmSpec) or spec.name != name:
            raise JobConfigurationError(
                f"Compatibility-only Trainer plugin {name!r} does not expose a matching TrainerSpec"
            )
        _validate_spec(spec)
        return spec

    def get(self, key: str) -> AlgorithmSpec:
        """Return a hydrated Beta TrainerSpec on explicit compatibility access."""
        self._ensure_bootstrapped()
        spec = self._execution_registry.get_compatibility(key)
        compatibility = self._execution_registry.compatibility_snapshot()
        with self._lock:
            descriptor = self._descriptors.get(key)
            portable = next(
                (
                    item.registration.spec
                    for item in self._distributed_descriptors.values()
                    if item.name == key
                ),
                None,
            )
            entry_point = self._compatibility_entry_points.get(key)
            hydrated = self._hydrated_specs.get(key)
            available = sorted(
                {
                    *compatibility,
                    *self._descriptors,
                    *(item.name for item in self._distributed_descriptors.values()),
                    *self._compatibility_entry_points,
                }
            )
        if spec is not None:
            return spec
        if hydrated is not None:
            return hydrated
        if descriptor is not None:
            hydrated = self._legacy_spec(descriptor)
        elif portable is not None:
            hydrated = portable
        elif entry_point is not None:
            hydrated = self._load_compatibility_plugin(key, entry_point)
        else:
            raise JobConfigurationError(
                f"Unknown trainer: {key!r}. Available: {available}"
            )
        with self._lock:
            return self._hydrated_specs.setdefault(key, hydrated)

    def list(self) -> list[str]:
        """Return every executable or compatibility-only algorithm identity."""
        self._ensure_bootstrapped()
        compatibility = self._execution_registry.compatibility_snapshot()
        with self._lock:
            return sorted(
                {
                    *compatibility,
                    *self._descriptors,
                    *(item.name for item in self._distributed_descriptors.values()),
                    *self._compatibility_entry_points,
                }
            )

    def contains(self, key: str) -> bool:
        """Return whether any executable or compatibility-only entry exists."""
        return key in self.list()

    def snapshot(self) -> dict[str, AlgorithmSpec]:
        """Return canonical facts without loading Trainer or framework modules."""
        self._ensure_bootstrapped()
        return self._execution_registry.catalog_spec_snapshot()

    def record_snapshot(self) -> tuple[AlgorithmRegistryEntry, ...]:
        """Return the immutable Catalog projection from one composition state."""
        self._ensure_bootstrapped()
        registrations_by_name: dict[str, list[AlgorithmRegistration]] = {}
        registrations, compatibility_specs = self._execution_registry.catalog_snapshot()
        for registration in registrations:
            registrations_by_name.setdefault(registration.spec.name, []).append(
                registration
            )
        with self._lock:
            descriptors = dict(self._descriptors)
            distributed = tuple(self._distributed_descriptors.values())
            compatibility_names = tuple(self._compatibility_entry_points)
        entries: list[AlgorithmRegistryEntry] = []
        distributed_by_name: dict[str, list[DistributedAlgorithmDescriptor]] = {}
        for item in distributed:
            distributed_by_name.setdefault(item.name, []).append(item)

        def support(
            descriptors: tuple[DistributedAlgorithmDescriptor, ...],
        ) -> tuple[AlgorithmSupportEvidence, ...]:
            return tuple(
                evidence
                for descriptor in descriptors
                for evidence in self._support_evidence.resolve(
                    descriptor,
                    wheel_sha256=self._installed_wheel_digests.get(
                        canonicalize_name(descriptor.package_name)
                    ),
                )
            )

        for name, descriptor in descriptors.items():
            native = tuple(distributed_by_name.get(name, ()))
            evidence = support(native)
            entries.append(
                AlgorithmRegistryEntry(
                    name=name,
                    spec=descriptor.registration.spec,
                    registrations=tuple(registrations_by_name.get(name, ())),
                    stability=native[0].stability if native else descriptor.stability,
                    available=True,
                    compatibility_only=False,
                    tested=bool(evidence),
                    supported=bool(evidence),
                    validated_execution_profiles=tuple(
                        sorted({item.execution_profile.value for item in evidence})
                    ),
                    native_migration_complete=bool(native),
                    limitations=(
                        native[0].limitations if native else descriptor.limitations
                    ),
                )
            )
        legacy_names = set(descriptors)
        for name, grouped in sorted(distributed_by_name.items()):
            if name in legacy_names:
                continue
            distributed_descriptor = grouped[0]
            evidence = support(tuple(grouped))
            entries.append(
                AlgorithmRegistryEntry(
                    name=distributed_descriptor.name,
                    spec=distributed_descriptor.registration.spec,
                    registrations=tuple(
                        registrations_by_name.get(distributed_descriptor.name, ())
                    ),
                    stability=distributed_descriptor.stability,
                    available=True,
                    compatibility_only=False,
                    tested=bool(evidence),
                    supported=bool(evidence),
                    validated_execution_profiles=tuple(
                        sorted({item.execution_profile.value for item in evidence})
                    ),
                    native_migration_complete=True,
                    limitations=distributed_descriptor.limitations,
                )
            )
        for name, spec in compatibility_specs.items():
            entries.append(
                AlgorithmRegistryEntry(
                    name=name,
                    spec=spec,
                    registrations=(),
                    stability="beta",
                    available=False,
                    compatibility_only=True,
                    tested=False,
                    supported=False,
                    validated_execution_profiles=(),
                    native_migration_complete=False,
                    limitations=(
                        "Programmatic Beta Trainer registration has no portable descriptor.",
                    ),
                )
            )
        for name in compatibility_names:
            entries.append(
                AlgorithmRegistryEntry(
                    name=name,
                    spec=None,
                    registrations=(),
                    stability="beta",
                    available=False,
                    compatibility_only=True,
                    tested=False,
                    supported=False,
                    validated_execution_profiles=(),
                    native_migration_complete=False,
                    limitations=(
                        "Legacy Trainer entry point has no lightweight portable descriptor.",
                    ),
                )
            )
        return tuple(sorted(entries, key=lambda item: item.name))

    def execution_registry(self) -> AlgorithmRegistrationRegistry:
        """Return the sole executable Registration source for composition roots."""
        self._ensure_bootstrapped()
        return self._execution_registry

    def diagnostics(self) -> tuple[PluginLoadDiagnostic, ...]:
        """Return immutable diagnostics captured during descriptor discovery."""
        self._ensure_bootstrapped()
        return self._diagnostics


_registry = TrainingAlgorithmRegistry()


@PublicAPI(stability="beta")
def register(spec: TrainerSpec) -> None:
    """Register a Beta-only TrainerSpec compatibility entry."""
    _validate_spec(spec)
    _registry.register(spec.name, spec)


@PublicAPI(stability="beta")
def get_trainer(name: str) -> TrainerSpec:
    """Return a registered TrainerSpec by name."""
    return _registry.get(name)


@PublicAPI(stability="beta")
def list_trainers() -> list[str]:
    """Return executable and compatibility-only Trainer names."""
    return _registry.list()


@DeveloperAPI
def get_execution_registry() -> AlgorithmRegistrationRegistry:
    """Return the sole formal/compatibility execution Registry."""
    return _registry.execution_registry()


__all__ = [
    "get_execution_registry",
    "get_trainer",
    "list_trainers",
    "register",
]
