"""Thin engine-binding registry with controlled third-party discovery.

The registry owns metadata, dependency, and capability validation.  A binding
owns the four internal compilation stages documented by the ingestion ADR:
capability validation, transform classification, native lazy-plan creation,
and typed-handle wrapping.  Neither layer implements data reading itself.
"""

from __future__ import annotations

import importlib.metadata
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Iterator,
    Literal,
    Mapping,
    NoReturn,
    Protocol,
    runtime_checkable,
)

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from tributo.data.ingestion import (
    DaftDataFrameHandle,
    DistributionVersionEvidence,
    HandleOwnership,
    IngestionHandle,
    IngestionRuntimeContext,
    PhysicalSplitSummary,
    RayDataHandle,
    ReadHint,
    ReadOptions,
    TransformDecision,
    _raise_after_failed_compilation,
)
from tributo.data.refs import _credential_paths
from tributo.data.scan_plan import (
    CatalogTableRef,
    FileScan,
    LogicalScanPlan,
    ScanKind,
    SourceCapability,
    TableScan,
)
from tributo.data.transform_ir import Limit, TransformPipeline
from tributo.exceptions import (
    DataSourceError,
    EngineNotAvailableError,
    JobConfigurationError,
    TributoError,
)
from tributo.util.annotations import DeveloperAPI

_ENGINE_DISTRIBUTIONS: dict[str, str] = {
    "tributo.ray_data": "ray",
    "tributo.daft": "daft",
}

PushdownLevel = Literal["exact", "inexact", "none"]
BindingCompileStage = Literal[
    "validate_capabilities",
    "classify_transforms",
    "build_native_plan",
    "wrap_handle",
    "compile",
]
BindingFailureCategory = Literal[
    "engine_not_available",
    "invalid_configuration",
    "data_source",
    "unexpected",
]

_BINDING_COMPILE_STAGES = frozenset(
    {
        "validate_capabilities",
        "classify_transforms",
        "build_native_plan",
        "wrap_handle",
        "compile",
    }
)
_BINDING_FAILURE_CATEGORIES = frozenset(
    {
        "engine_not_available",
        "invalid_configuration",
        "data_source",
        "unexpected",
    }
)
_SAFE_EXCEPTION_TYPE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_SAFE_DIAGNOSTIC_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_MAX_BINDING_DIAGNOSTIC_LENGTH = 512


@DeveloperAPI
class BindingStageError(Exception):
    """Credential-free failure metadata crossing the binding trust boundary."""

    def __init__(
        self,
        *,
        stage: BindingCompileStage,
        category: BindingFailureCategory,
        exception_type: str,
        diagnostic_code: str | None = None,
        diagnostic: str | None = None,
    ) -> None:
        if stage not in _BINDING_COMPILE_STAGES:
            raise ValueError(f"Unknown binding compile stage {stage!r}")
        if category not in _BINDING_FAILURE_CATEGORIES:
            raise ValueError(f"Unknown binding failure category {category!r}")
        if _SAFE_EXCEPTION_TYPE_RE.fullmatch(exception_type) is None:
            exception_type = "Exception"
        if (diagnostic_code is None) != (diagnostic is None):
            raise ValueError(
                "Binding diagnostics require both diagnostic_code and diagnostic"
            )
        if (
            diagnostic_code is not None
            and _SAFE_DIAGNOSTIC_CODE_RE.fullmatch(diagnostic_code) is None
        ):
            raise ValueError("Binding diagnostic_code must be a safe identifier")
        if diagnostic is not None:
            if (
                not diagnostic
                or len(diagnostic) > _MAX_BINDING_DIAGNOSTIC_LENGTH
                or "\n" in diagnostic
                or "\r" in diagnostic
            ):
                raise ValueError("Binding diagnostic must be bounded single-line text")
            if _credential_paths({"diagnostic": diagnostic}):
                raise ValueError("Binding diagnostic must be credential-free")
        self.stage = stage
        self.category = category
        self.exception_type = exception_type
        self.diagnostic_code = diagnostic_code
        self.diagnostic = diagnostic
        message = f"{stage}:{category}:{exception_type}"
        if diagnostic_code is not None:
            message = f"{message} [{diagnostic_code}] {diagnostic}"
        super().__init__(message)

    @classmethod
    def from_exception(
        cls, stage: BindingCompileStage, exc: Exception
    ) -> "BindingStageError":
        if isinstance(exc, EngineNotAvailableError):
            category: BindingFailureCategory = "engine_not_available"
        elif isinstance(exc, JobConfigurationError):
            category = "invalid_configuration"
        elif isinstance(exc, DataSourceError):
            category = "data_source"
        else:
            category = "unexpected"
        return cls(
            stage=stage,
            category=category,
            exception_type=type(exc).__name__,
        )

    @classmethod
    def framework_diagnostic(
        cls,
        stage: BindingCompileStage,
        *,
        error_type: type[TributoError],
        diagnostic_code: str,
        diagnostic: str,
    ) -> "BindingStageError":
        """Build an explicit first-party diagnostic without copying an exception."""
        if issubclass(error_type, EngineNotAvailableError):
            category: BindingFailureCategory = "engine_not_available"
        elif issubclass(error_type, JobConfigurationError):
            category = "invalid_configuration"
        elif issubclass(error_type, DataSourceError):
            category = "data_source"
        else:
            raise ValueError(
                "Framework binding diagnostic requires a Tributo data error"
            )
        return cls(
            stage=stage,
            category=category,
            exception_type=error_type.__name__,
            diagnostic_code=diagnostic_code,
            diagnostic=diagnostic,
        )

    def without_diagnostic(self) -> "BindingStageError":
        """Return failure metadata safe for an untrusted Binding boundary."""
        return BindingStageError(
            stage=self.stage,
            category=self.category,
            exception_type=self.exception_type,
        )


@DeveloperAPI
@contextmanager
def binding_stage(stage: BindingCompileStage) -> Iterator[None]:
    """Normalize one binding phase without retaining native exception text."""
    failure: BindingStageError | None = None
    try:
        yield
    except BindingStageError:
        raise
    except Exception as exc:
        failure = BindingStageError.from_exception(stage, exc)
    if failure is not None:
        # Raise after leaving the native ``except`` block so ``__context__``
        # cannot retain the credential-bearing exception object.
        raise failure


@DeveloperAPI
@dataclass(frozen=True)
class BindingKey:
    """Unique identity for one physical Binding implementation."""

    engine_id: str
    scan_kind: ScanKind
    connector_id: str
    binding_id: str

    def __post_init__(self) -> None:
        if self.engine_id not in _ENGINE_DISTRIBUTIONS:
            raise ValueError(
                "BindingKey.engine_id must be tributo.ray_data or tributo.daft"
            )
        if not isinstance(self.scan_kind, ScanKind):
            raise ValueError("BindingKey.scan_kind must be a ScanKind")
        if (
            not isinstance(self.connector_id, str)
            or re.fullmatch(r"[A-Za-z0-9_.-]+", self.connector_id) is None
        ):
            raise ValueError("BindingKey.connector_id must be an identifier")
        if (
            not isinstance(self.binding_id, str)
            or re.fullmatch(r"[a-z][a-z0-9_.-]+", self.binding_id) is None
        ):
            raise ValueError("BindingKey.binding_id must be a namespaced identifier")


@DeveloperAPI
@dataclass(frozen=True)
class BindingPlanConstraints:
    """Declarative plan constraints used before loading a Binding factory."""

    filesystem_ids: frozenset[str] = frozenset()
    catalog_ids: frozenset[str] = frozenset()
    storage_format_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for field_name, values in (
            ("filesystem_ids", self.filesystem_ids),
            ("catalog_ids", self.catalog_ids),
            ("storage_format_ids", self.storage_format_ids),
        ):
            if not isinstance(values, frozenset) or any(
                not isinstance(item, str)
                or re.fullmatch(r"[A-Za-z0-9_.-]+", item) is None
                for item in values
            ):
                raise ValueError(
                    f"BindingPlanConstraints.{field_name} must be identifiers"
                )

    def matches(self, plan: LogicalScanPlan) -> bool:
        """Return whether the static logical plan is in this Binding scope."""
        if self.filesystem_ids and (
            not isinstance(plan, FileScan)
            or plan.filesystem_id not in self.filesystem_ids
        ):
            return False
        if self.catalog_ids and (
            not isinstance(plan, TableScan)
            or not isinstance(plan.table, CatalogTableRef)
            or plan.table.catalog_id not in self.catalog_ids
        ):
            return False
        if self.storage_format_ids and (
            not isinstance(plan, TableScan)
            or plan.storage_format_id not in self.storage_format_ids
        ):
            return False
        return True


@DeveloperAPI
@dataclass(frozen=True)
class BindingCompileRequest:
    """Credential-isolated request passed to an installed binding."""

    plan: LogicalScanPlan
    runtime_options: Mapping[str, Any] = field(repr=False)
    transforms: TransformPipeline
    read_options: ReadOptions
    source_ref: str
    runtime_context: IngestionRuntimeContext = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_options, Mapping):
            raise ValueError("BindingCompileRequest.runtime_options must be a mapping")
        if (
            not isinstance(self.source_ref, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.source_ref) is None
        ):
            raise ValueError("BindingCompileRequest.source_ref must be SHA-256 hex")
        if not isinstance(self.runtime_context, IngestionRuntimeContext):
            raise ValueError(
                "BindingCompileRequest.runtime_context must be IngestionRuntimeContext"
            )


@DeveloperAPI
@dataclass(frozen=True)
class BindingCompilation:
    """Native handle and plan metadata returned by a thin binding."""

    handle: IngestionHandle
    engine_version: str
    reader_api: str
    transport_id: str
    transform_decisions: tuple[TransformDecision, ...] = ()
    input_schema_fingerprint: str | None = None
    schema_fingerprint: str | None = None
    metadata_fetched: bool = False
    physical_splits: PhysicalSplitSummary = field(default_factory=PhysicalSplitSummary)
    diagnostics: tuple[str, ...] = ()
    ownership: HandleOwnership = HandleOwnership.OWNED
    close_callback: Callable[[], None] | None = field(
        default=None, repr=False, compare=False
    )
    cancel_callback: Callable[[], None] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.handle, (RayDataHandle, DaftDataFrameHandle)):
            raise ValueError("BindingCompilation.handle must be a typed native handle")
        if not isinstance(self.transform_decisions, tuple):
            raise ValueError("BindingCompilation.transform_decisions must be a tuple")
        if not isinstance(self.metadata_fetched, bool):
            raise ValueError("BindingCompilation.metadata_fetched must be a bool")
        if not self.reader_api:
            raise ValueError("BindingCompilation.reader_api must be non-empty")
        if not self.transport_id:
            raise ValueError("BindingCompilation.transport_id must be non-empty")
        if not isinstance(self.physical_splits, PhysicalSplitSummary):
            raise ValueError(
                "BindingCompilation.physical_splits must be PhysicalSplitSummary"
            )
        if not isinstance(self.diagnostics, tuple) or any(
            not isinstance(item, str) for item in self.diagnostics
        ):
            raise ValueError("BindingCompilation.diagnostics must be a tuple of str")
        leaked = _credential_paths(
            {
                "reader_api": self.reader_api,
                "transport_id": self.transport_id,
                "physical_splits": self.physical_splits.model_dump(mode="python"),
                "diagnostics": self.diagnostics,
            }
        )
        if leaked:
            raise ValueError("BindingCompilation metadata must be credential-free")
        if not isinstance(self.ownership, HandleOwnership):
            raise ValueError("BindingCompilation.ownership must be HandleOwnership")
        if self.close_callback is not None and not callable(self.close_callback):
            raise ValueError("BindingCompilation.close_callback must be callable")
        if self.cancel_callback is not None and not callable(self.cancel_callback):
            raise ValueError("BindingCompilation.cancel_callback must be callable")
        if self.ownership is not HandleOwnership.OWNED and (
            self.close_callback is not None or self.cancel_callback is not None
        ):
            raise ValueError(
                "Non-owned BindingCompilation cannot define lifecycle callbacks"
            )
        if not isinstance(self.engine_version, str):
            raise ValueError("BindingCompilation.engine_version must be valid")
        try:
            Version(self.engine_version)
        except InvalidVersion as exc:
            raise ValueError("BindingCompilation.engine_version must be valid") from exc
        for field_name, fingerprint in (
            ("input_schema_fingerprint", self.input_schema_fingerprint),
            ("schema_fingerprint", self.schema_fingerprint),
        ):
            if (
                fingerprint is not None
                and re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
            ):
                raise ValueError(f"BindingCompilation.{field_name} must be SHA-256 hex")


@runtime_checkable
@DeveloperAPI
class EngineBinding(Protocol):
    """Thin plan-to-native-reader delegation contract."""

    def compile(self, request: BindingCompileRequest) -> BindingCompilation:
        """Delegate *request* to an engine or third-party public API."""


BindingFactory = Callable[[], EngineBinding]


@DeveloperAPI
@dataclass(frozen=True)
class BindingDescriptor:
    """Metadata and factory exported by a built-in or third-party package."""

    key: BindingKey
    factory: BindingFactory = field(repr=False, compare=False)
    capabilities: frozenset[SourceCapability]
    distribution_name: str
    distribution_version: str
    engine_version_spec: str
    dependency_distributions: tuple[str, ...] = ()
    constraints: BindingPlanConstraints = field(default_factory=BindingPlanConstraints)
    supported_read_hints: frozenset[ReadHint] = frozenset()
    api_version: Literal[1] = 1
    capability_version: Literal[1] = 1
    install_hint: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.distribution_name, str)
            or not self.distribution_name
            or not isinstance(self.distribution_version, str)
            or not self.distribution_version
        ):
            raise ValueError("Binding distribution metadata must be non-empty")
        try:
            Version(self.distribution_version)
        except InvalidVersion as exc:
            raise ValueError("Binding distribution_version must be valid") from exc
        if (
            not isinstance(self.engine_version_spec, str)
            or not self.engine_version_spec
        ):
            raise ValueError("Binding engine_version_spec must be non-empty")
        try:
            SpecifierSet(self.engine_version_spec)
        except InvalidSpecifier as exc:
            raise ValueError("Binding engine_version_spec must be valid") from exc
        if not callable(self.factory):
            raise ValueError("Binding factory must be callable")
        if not isinstance(self.dependency_distributions, tuple) or any(
            not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None
            for name in self.dependency_distributions
        ):
            raise ValueError("Binding dependency distributions must be identifiers")
        if len(set(self.dependency_distributions)) != len(
            self.dependency_distributions
        ):
            raise ValueError("Binding dependency distributions must be unique")
        if not isinstance(self.capabilities, frozenset):
            raise ValueError("Binding capabilities must be a frozenset")
        if any(
            not isinstance(capability, SourceCapability)
            for capability in self.capabilities
        ):
            raise ValueError("Binding capabilities must be SourceCapability values")
        if not isinstance(self.constraints, BindingPlanConstraints):
            raise ValueError("Binding constraints must be BindingPlanConstraints")
        if not isinstance(self.supported_read_hints, frozenset) or any(
            not isinstance(hint, ReadHint) for hint in self.supported_read_hints
        ):
            raise ValueError(
                "Binding supported_read_hints must contain ReadHint values"
            )
        if self.api_version != 1:
            raise ValueError("Binding api_version must be 1")
        if self.capability_version != 1:
            raise ValueError("Binding capability_version must be 1")
        if self.install_hint is not None and _credential_paths(
            {"install_hint": self.install_hint}
        ):
            raise ValueError("Binding install_hint must be credential-free")


@dataclass(frozen=True)
class _BindingRequirement:
    key: BindingKey
    constraints: BindingPlanConstraints
    install_hint: str


@DeveloperAPI
class EngineBindings:
    """Thread-safe registry for binding metadata and factories."""

    def __init__(self) -> None:
        self._descriptors: dict[BindingKey, BindingDescriptor] = {}
        self._requirements: dict[BindingKey, _BindingRequirement] = {}
        self._lock = threading.RLock()

    def register(self, descriptor: BindingDescriptor) -> None:
        """Register one installed descriptor atomically; duplicates fail closed."""
        self._validate_installed_versions(descriptor)
        with self._lock:
            if descriptor.key in self._descriptors:
                raise JobConfigurationError(
                    f"Engine binding {descriptor.key!r} is already registered"
                )
            self._descriptors[descriptor.key] = descriptor

    def register_requirement(
        self,
        key: BindingKey,
        install_hint: str,
        *,
        constraints: BindingPlanConstraints | None = None,
    ) -> None:
        """Register a credential-free install hint for a known external binding."""
        if not install_hint.strip():
            raise ValueError("install_hint must be non-empty")
        if _credential_paths({"install_hint": install_hint}):
            raise ValueError("install_hint must be credential-free")
        with self._lock:
            if key in self._requirements:
                raise JobConfigurationError(
                    f"Engine binding requirement {key!r} is already registered"
                )
            self._requirements[key] = _BindingRequirement(
                key=key,
                constraints=constraints or BindingPlanConstraints(),
                install_hint=install_hint,
            )

    def resolve(self, key: BindingKey) -> BindingDescriptor:
        """Resolve an explicitly registered first- or third-party descriptor."""
        with self._lock:
            existing = self._descriptors.get(key)
            requirement = self._requirements.get(key)
        if existing is not None:
            return existing
        detail = f" Install with: {requirement.install_hint}." if requirement else ""
        raise EngineNotAvailableError(
            f"No installed binding for {key.engine_id}/{key.scan_kind.value}/"
            f"{key.connector_id}/{key.binding_id}.{detail}"
        )

    def _select(
        self,
        *,
        engine_id: str,
        plan: LogicalScanPlan,
        binding_id: str | None,
    ) -> BindingDescriptor:
        with self._lock:
            descriptors = tuple(self._descriptors.values())
            requirements = tuple(self._requirements.values())

        def matches_key(key: BindingKey) -> bool:
            return (
                key.engine_id == engine_id
                and key.scan_kind is plan.scan_kind
                and key.connector_id == plan.connector_id
                and (binding_id is None or key.binding_id == binding_id)
            )

        candidates = tuple(
            descriptor
            for descriptor in descriptors
            if matches_key(descriptor.key) and descriptor.constraints.matches(plan)
        )
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            candidate_ids = ", ".join(
                sorted(descriptor.key.binding_id for descriptor in candidates)
            )
            raise JobConfigurationError(
                "Multiple installed bindings match "
                f"{engine_id}/{plan.scan_kind.value}/{plan.connector_id}: "
                f"{candidate_ids}. Set IngestionRequest.binding_id explicitly."
            )

        matching_requirements = tuple(
            requirement
            for requirement in requirements
            if matches_key(requirement.key) and requirement.constraints.matches(plan)
        )
        if matching_requirements:
            hints = "; ".join(
                sorted(
                    f"{item.key.binding_id}: {item.install_hint}"
                    for item in matching_requirements
                )
            )
            raise EngineNotAvailableError(
                "No installed binding matches "
                f"{engine_id}/{plan.scan_kind.value}/{plan.connector_id}. "
                f"Available installation options: {hints}."
            )
        requested = f" binding_id={binding_id!r}" if binding_id is not None else ""
        raise EngineNotAvailableError(
            "No installed binding matches "
            f"{engine_id}/{plan.scan_kind.value}/{plan.connector_id}{requested}."
        )

    def describe(
        self,
        *,
        engine_id: str,
        plan: LogicalScanPlan,
        binding_id: str | None = None,
        read_options: ReadOptions | None = None,
    ) -> tuple[BindingDescriptor, frozenset[SourceCapability]]:
        """Resolve metadata and validate capabilities without compiling a plan."""
        descriptor = self._select(
            engine_id=engine_id,
            plan=plan,
            binding_id=binding_id,
        )
        key = descriptor.key
        required = plan.required_capabilities
        missing = required - descriptor.capabilities
        if missing:
            names = ", ".join(sorted(item.value for item in missing))
            raise JobConfigurationError(
                f"Binding {key!r} does not provide required capabilities: {names}"
            )
        requested_hints = (read_options or ReadOptions()).requested_hints()
        unsupported_hints = requested_hints - descriptor.supported_read_hints
        if unsupported_hints:
            names = ", ".join(sorted(item.value for item in unsupported_hints))
            raise JobConfigurationError(
                f"Binding {key!r} does not support read hint(s): {names}"
            )
        return descriptor, required

    def compile(
        self,
        *,
        engine_id: str,
        binding_id: str | None,
        plan: LogicalScanPlan,
        runtime_options: Mapping[str, Any],
        transforms: TransformPipeline,
        read_options: ReadOptions,
        source_ref: str,
        runtime_context: IngestionRuntimeContext,
    ) -> tuple[
        BindingCompilation,
        BindingDescriptor,
        tuple[DistributionVersionEvidence, ...],
    ]:
        """Resolve and invoke a thin binding, then prove installed versions."""
        descriptor, _ = self.describe(
            engine_id=engine_id,
            plan=plan,
            binding_id=binding_id,
            read_options=read_options,
        )
        key = descriptor.key
        factory_failure: BindingStageError | None = None
        try:
            binding = descriptor.factory()
        except Exception as exc:
            factory_failure = BindingStageError.from_exception("compile", exc)
        if factory_failure is not None:
            _raise_binding_stage_error(key, factory_failure)
        if not isinstance(binding, EngineBinding):
            raise JobConfigurationError(
                f"Binding factory for {key!r} did not return an EngineBinding"
            )
        failure: BindingStageError | None = None
        try:
            compilation = binding.compile(
                BindingCompileRequest(
                    plan=plan,
                    runtime_options=runtime_options,
                    transforms=transforms,
                    read_options=read_options,
                    source_ref=source_ref,
                    runtime_context=runtime_context,
                )
            )
        except BindingStageError as exc:
            failure = (
                exc
                if _is_first_party_binding(descriptor, binding)
                else exc.without_diagnostic()
            )
        except Exception as exc:
            failure = BindingStageError.from_exception("compile", exc)
        if failure is not None:
            # Raising outside the native ``except`` block prevents the public
            # error from retaining a hidden exception context with secrets.
            _raise_binding_stage_error(key, failure)

        if not isinstance(compilation, BindingCompilation):
            raise JobConfigurationError(
                f"Binding {key!r} returned {type(compilation).__name__}, "
                "expected BindingCompilation"
            )
        validation_failure: BindingStageError | None = None
        try:
            self._validate_compilation(compilation, descriptor, transforms)
            versions = self._collect_version_evidence(
                descriptor, compilation, runtime_context
            )
        except BindingStageError as exc:
            validation_failure = exc
        except Exception as exc:
            validation_failure = BindingStageError.from_exception("compile", exc)
        if validation_failure is not None:
            _fail_compilation(
                compilation,
                _binding_stage_exception(key, validation_failure),
            )
        return compilation, descriptor, versions

    @staticmethod
    def _validate_compilation(
        compilation: BindingCompilation,
        descriptor: BindingDescriptor,
        transforms: TransformPipeline,
    ) -> None:
        key = descriptor.key
        expected_handle = (
            RayDataHandle
            if key.engine_id == "tributo.ray_data"
            else DaftDataFrameHandle
        )
        if not isinstance(compilation.handle, expected_handle):
            raise BindingStageError.framework_diagnostic(
                "compile",
                error_type=JobConfigurationError,
                diagnostic_code="handle_engine_mismatch",
                diagnostic=(
                    f"Binding {key!r} returned {type(compilation.handle).__name__} "
                    f"for engine {key.engine_id!r}"
                ),
            )
        engine_distribution = _ENGINE_DISTRIBUTIONS[key.engine_id]
        engine_missing = False
        try:
            installed_version = importlib.metadata.version(engine_distribution)
        except importlib.metadata.PackageNotFoundError:
            engine_missing = True
        if engine_missing:
            raise BindingStageError.framework_diagnostic(
                "compile",
                error_type=EngineNotAvailableError,
                diagnostic_code="engine_distribution_missing",
                diagnostic=(
                    f"Engine distribution {engine_distribution!r} is no longer installed"
                ),
            )
        if Version(compilation.engine_version) != Version(installed_version):
            raise BindingStageError.framework_diagnostic(
                "compile",
                error_type=EngineNotAvailableError,
                diagnostic_code="engine_version_mismatch",
                diagnostic=(
                    f"Binding {key!r} reported engine version "
                    f"{compilation.engine_version}, installed {installed_version}"
                ),
            )
        if len(compilation.transform_decisions) != len(transforms.steps):
            raise BindingStageError.framework_diagnostic(
                "compile",
                error_type=JobConfigurationError,
                diagnostic_code="transform_decision_count_mismatch",
                diagnostic=(
                    f"Binding {key!r} classified "
                    f"{len(compilation.transform_decisions)} of "
                    f"{len(transforms.steps)} transforms"
                ),
            )
        for ordinal, (decision, transform) in enumerate(
            zip(compilation.transform_decisions, transforms.steps)
        ):
            if decision.ordinal != ordinal or decision.transform_type != transform.type:
                raise BindingStageError.framework_diagnostic(
                    "compile",
                    error_type=JobConfigurationError,
                    diagnostic_code="transform_decision_mismatch",
                    diagnostic=(
                        f"Binding {key!r} returned an inconsistent transform "
                        f"decision at ordinal {ordinal}"
                    ),
                )

    @staticmethod
    def _collect_version_evidence(
        descriptor: BindingDescriptor,
        compilation: BindingCompilation,
        context: IngestionRuntimeContext,
    ) -> tuple[DistributionVersionEvidence, ...]:
        names = tuple(
            dict.fromkeys(
                (
                    _ENGINE_DISTRIBUTIONS[descriptor.key.engine_id],
                    descriptor.distribution_name,
                    *descriptor.dependency_distributions,
                )
            )
        )
        driver_versions: dict[str, str] = {}
        for name in names:
            distribution_missing = False
            try:
                driver_versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                distribution_missing = True
            if distribution_missing:
                raise BindingStageError.framework_diagnostic(
                    "compile",
                    error_type=EngineNotAvailableError,
                    diagnostic_code="distribution_missing",
                    diagnostic=f"Distribution {name!r} is no longer installed",
                )
        worker_versions: Mapping[str, tuple[str, ...]] = {}
        if context.distribution_probe is not None:
            probe_failure: BindingStageError | None = None
            try:
                worker_versions = context.distribution_probe(names)
            except Exception as exc:
                probe_failure = BindingStageError.from_exception("compile", exc)
            if probe_failure is not None:
                raise probe_failure

        evidence: list[DistributionVersionEvidence] = []
        for name in names:
            workers = tuple(worker_versions.get(name, ()))
            if context.distribution_probe is not None and not workers:
                raise BindingStageError.framework_diagnostic(
                    "compile",
                    error_type=EngineNotAvailableError,
                    diagnostic_code="worker_version_evidence_missing",
                    diagnostic=(
                        f"Worker version evidence is missing for distribution {name!r}"
                    ),
                )
            try:
                driver_version = Version(driver_versions[name])
                parsed_workers = tuple(Version(item) for item in workers)
            except InvalidVersion:
                raise BindingStageError.framework_diagnostic(
                    "compile",
                    error_type=EngineNotAvailableError,
                    diagnostic_code="worker_version_invalid",
                    diagnostic=(
                        f"Driver or worker reported an invalid version for "
                        f"distribution {name!r}"
                    ),
                ) from None
            if any(item != driver_version for item in parsed_workers):
                raise BindingStageError.framework_diagnostic(
                    "compile",
                    error_type=EngineNotAvailableError,
                    diagnostic_code="worker_version_mismatch",
                    diagnostic=(
                        f"Driver and worker versions differ for distribution {name!r}"
                    ),
                )
            evidence.append(
                DistributionVersionEvidence(
                    distribution_name=name,
                    driver_version=driver_versions[name],
                    worker_versions=workers,
                    worker_validation_complete=context.distribution_probe is not None,
                )
            )
        return tuple(evidence)

    @staticmethod
    def _validate_installed_versions(descriptor: BindingDescriptor) -> None:
        engine_distribution = _ENGINE_DISTRIBUTIONS[descriptor.key.engine_id]
        try:
            installed_engine = importlib.metadata.version(engine_distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise EngineNotAvailableError(
                f"Engine distribution {engine_distribution!r} is not installed"
            ) from exc
        try:
            compatible = Version(installed_engine) in SpecifierSet(
                descriptor.engine_version_spec
            )
        except InvalidVersion as exc:
            raise EngineNotAvailableError(
                f"Installed {engine_distribution} version is invalid"
            ) from exc
        if not compatible:
            raise EngineNotAvailableError(
                f"Binding {descriptor.key!r} requires {engine_distribution}"
                f"{descriptor.engine_version_spec}, installed {installed_engine}"
            )

        try:
            installed_binding = importlib.metadata.version(descriptor.distribution_name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise EngineNotAvailableError(
                f"Binding distribution {descriptor.distribution_name!r} is not installed"
            ) from exc
        if Version(installed_binding) != Version(descriptor.distribution_version):
            raise EngineNotAvailableError(
                f"Binding {descriptor.key!r} declares distribution version "
                f"{descriptor.distribution_version}, installed {installed_binding}"
            )
        for dependency in descriptor.dependency_distributions:
            try:
                importlib.metadata.version(dependency)
            except importlib.metadata.PackageNotFoundError as exc:
                raise EngineNotAvailableError(
                    f"Binding {descriptor.key!r} requires distribution {dependency!r}"
                ) from exc


@DeveloperAPI
def classify_transform_decisions(
    pipeline: TransformPipeline,
    pushdown_levels: Mapping[int, PushdownLevel] | None = None,
) -> tuple[TransformDecision, ...]:
    """Classify every ordered Transform with fail-closed residual semantics."""
    levels = dict(pushdown_levels or {})
    unknown_ordinals = set(levels) - set(range(len(pipeline.steps)))
    if unknown_ordinals:
        raise JobConfigurationError(
            f"Pushdown classification contains unknown ordinals {sorted(unknown_ordinals)}"
        )

    decisions: list[TransformDecision] = []
    for ordinal, transform in enumerate(pipeline.steps):
        level = levels.get(ordinal, "none")
        if isinstance(transform, Limit) and level != "none":
            raise JobConfigurationError(
                "Limit must remain an ordered residual transform"
            )
        if level == "exact":
            compiled_result: Literal["pushed", "residual", "pushed_and_residual"] = (
                "pushed"
            )
            residual_required = False
        elif level == "inexact":
            compiled_result = "pushed_and_residual"
            residual_required = True
        elif level == "none":
            compiled_result = "residual"
            residual_required = True
        else:
            raise JobConfigurationError(
                f"Unknown pushdown level {level!r} for transform {ordinal}"
            )
        decisions.append(
            TransformDecision(
                ordinal=ordinal,
                transform_type=transform.type,
                pushdown_level=level,
                residual_required=residual_required,
                compiled_result=compiled_result,
            )
        )
    return tuple(decisions)


def _fail_compilation(compilation: BindingCompilation, error: TributoError) -> NoReturn:
    """Release an invalid owned compilation before raising a contract error."""
    _raise_after_failed_compilation(
        ownership=compilation.ownership,
        close_callback=compilation.close_callback,
        cancel_callback=compilation.cancel_callback,
        error=error,
    )


def _binding_stage_exception(key: BindingKey, error: BindingStageError) -> TributoError:
    """Build a credential-free public error without raising in a native handler."""
    message = (
        f"Engine binding {key!r} failed during {error.stage} "
        f"[{error.category}] with {error.exception_type}"
    )
    if error.diagnostic_code is not None:
        message = f"{message} [{error.diagnostic_code}] {error.diagnostic}"
    if error.category == "engine_not_available":
        return EngineNotAvailableError(message)
    if error.category == "invalid_configuration":
        return JobConfigurationError(message)
    return DataSourceError(message)


def _is_first_party_binding(
    descriptor: BindingDescriptor, binding: EngineBinding
) -> bool:
    """Return whether governance permits framework-authored diagnostics.

    This is a registration trust boundary, not package-origin authentication:
    descriptor metadata and Python module names are not security attestations.
    Credential scanning in ``BindingStageError`` remains the security boundary.
    """
    return descriptor.distribution_name == "tributo" and type(
        binding
    ).__module__.startswith("tributo.data.bindings.")


def _raise_binding_stage_error(key: BindingKey, error: BindingStageError) -> NoReturn:
    raise _binding_stage_exception(key, error) from None


__all__ = [
    "BindingCompilation",
    "BindingCompileRequest",
    "BindingDescriptor",
    "BindingKey",
    "BindingPlanConstraints",
    "BindingStageError",
    "EngineBinding",
    "EngineBindings",
    "binding_stage",
    "classify_transform_decisions",
]
