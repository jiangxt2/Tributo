"""Framework-neutral value objects for portable algorithm execution."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qsl, urlsplit

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name

from tributo._common.immutable import FrozenDict, deep_freeze, deep_thaw
from tributo.algorithms.api.distribution import (
    DistributionSpec,
    DistributionStrategy,
    ExecutionProfile,
    InputDistribution,
    ResultPolicy,
    StateCoordination,
)
from tributo.algorithms.api.errors import AlgorithmConfigurationError
from tributo.training.algorithm_spec import AlgorithmSpec
from tributo.util.annotations import DeveloperAPI, PublicAPI

if TYPE_CHECKING:
    from tributo.algorithms.api.execution import ExecutionReceipt

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NAMESPACED_ID = re.compile(r"^[a-z][a-z0-9_.-]*$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SENSITIVE_CONFIG_KEYS = frozenset(
    {
        "access_key",
        "access_key_id",
        "api_key",
        "apikey",
        "authorization",
        "client_secret",
        "cookie",
        "credential",
        "credentials",
        "password",
        "passwd",
        "private_key",
        "refresh_token",
        "secret",
        "secret_access_key",
        "secret_key",
        "token",
    }
)


def _require_namespaced_id(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _NAMESPACED_ID.fullmatch(value) is None:
        raise AlgorithmConfigurationError(
            f"{field_name} must be a namespaced lower-case identifier"
        )


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise AlgorithmConfigurationError(
            "portable algorithm metadata must be canonical JSON"
        ) from exc


def _credential_paths(value: object, prefix: str = "algorithm_config") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            path = f"{prefix}.{key}"
            if normalized in _SENSITIVE_CONFIG_KEYS:
                paths.append(path)
            paths.extend(_credential_paths(nested, path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            paths.extend(_credential_paths(nested, f"{prefix}[{index}]"))
    return paths


def _reference_contains_credentials(reference: str) -> bool:
    try:
        parsed = urlsplit(reference)
    except ValueError:
        return True
    if parsed.username is not None or parsed.password is not None:
        return True
    components = [parsed.query, parsed.fragment]
    if "?" in parsed.fragment:
        components.append(parsed.fragment.partition("?")[2])
    return any(
        key.strip().lower().replace("-", "_") in _SENSITIVE_CONFIG_KEYS
        for component in components
        for key, _ in parse_qsl(component, keep_blank_values=True)
    )


@DeveloperAPI
def canonical_digest(value: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 digest for a JSON-compatible mapping."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@PublicAPI(stability="alpha")
class AlgorithmOperation(str, Enum):
    """Bounded operations implemented by the first portable execution slice."""

    FIT = "fit"
    EVALUATE = "evaluate"
    PREDICT = "predict"
    TRANSFORM = "transform"


@PublicAPI(stability="alpha")
class ExecutionMode(str, Enum):
    """How an implementation is loaded inside a managed Ray Worker."""

    MANAGED_ESTIMATOR = "managed_estimator"
    CUSTOM_RAY_FUNCTION = "custom_ray_function"
    LEGACY_TRAINER = "legacy_trainer"
    COLLECTIVE = "collective"
    MAP_REDUCE = "map_reduce"
    FRAMEWORK_NATIVE = "framework_native"


@PublicAPI(stability="alpha")
class RuntimeTopology(str, Enum):
    """How one bounded invocation occupies Ray execution resources."""

    SINGLE_WORKER = "single_worker"
    FRAMEWORK_MANAGED = "framework_managed"
    DATA_PARALLEL = "data_parallel"
    RAY_TRAIN_COLLECTIVE = "ray_train_collective"
    FRAMEWORK_NATIVE = "framework_native"
    RAY_MAP_REDUCE = "ray_map_reduce"


@dataclass(frozen=True)
class FormalDistributedStrategyContract:
    """Canonical fields mechanically implied by a formal strategy."""

    execution_mode: ExecutionMode
    runtime_id: str
    topology: RuntimeTopology
    input_distribution: InputDistribution
    state_coordination: StateCoordination
    worker_input_adapter_ref: str


FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS: Mapping[
    DistributionStrategy, FormalDistributedStrategyContract
] = MappingProxyType(
    {
        DistributionStrategy.RAY_TRAIN_COLLECTIVE: FormalDistributedStrategyContract(
            execution_mode=ExecutionMode.COLLECTIVE,
            runtime_id="tributo.ray_train_collective",
            topology=RuntimeTopology.RAY_TRAIN_COLLECTIVE,
            input_distribution=InputDistribution.SHARDED,
            state_coordination=StateCoordination.ALL_REDUCE,
            worker_input_adapter_ref=(
                "tributo.integrations.algorithm_inputs.ingestion:"
                "prepare_ray_train_input"
            ),
        ),
        DistributionStrategy.RAY_MAP_REDUCE: FormalDistributedStrategyContract(
            execution_mode=ExecutionMode.MAP_REDUCE,
            runtime_id="tributo.ray_map_reduce",
            topology=RuntimeTopology.RAY_MAP_REDUCE,
            input_distribution=InputDistribution.SHARDED,
            state_coordination=StateCoordination.ASSOCIATIVE_REDUCE,
            worker_input_adapter_ref=(
                "tributo.integrations.algorithm_inputs.ingestion:"
                "prepare_ray_batch_input"
            ),
        ),
        DistributionStrategy.FRAMEWORK_NATIVE: FormalDistributedStrategyContract(
            execution_mode=ExecutionMode.FRAMEWORK_NATIVE,
            runtime_id="tributo.framework_native",
            topology=RuntimeTopology.FRAMEWORK_NATIVE,
            input_distribution=InputDistribution.FRAMEWORK_OWNED,
            state_coordination=StateCoordination.FRAMEWORK_NATIVE,
            worker_input_adapter_ref=(
                "tributo.integrations.algorithm_inputs.ingestion:"
                "prepare_ray_train_input"
            ),
        ),
    }
)


@PublicAPI(stability="alpha")
class FailureCategory(str, Enum):
    """Stable failure metadata; this does not replace the exception hierarchy."""

    VALIDATION = "validation"
    DEPENDENCY = "dependency"
    INPUT = "input"
    EXECUTION = "execution"
    CHECKPOINT = "checkpoint"
    CANCELLED = "cancelled"
    INTERNAL = "internal"


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class QualifiedReference:
    """A non-executable module and attribute reference stored in a plan."""

    module: str
    qualname: str

    def __post_init__(self) -> None:
        if not self.module or any(
            _IDENTIFIER.fullmatch(part) is None for part in self.module.split(".")
        ):
            raise AlgorithmConfigurationError(
                f"invalid module in qualified reference: {self.module!r}"
            )
        if not self.qualname or any(
            _IDENTIFIER.fullmatch(part) is None for part in self.qualname.split(".")
        ):
            raise AlgorithmConfigurationError(
                f"invalid attribute in qualified reference: {self.qualname!r}"
            )

    @classmethod
    def parse(cls, value: str) -> QualifiedReference:
        """Parse ``module:qualified.attribute`` without importing the module."""
        if not isinstance(value, str):
            raise AlgorithmConfigurationError(
                "qualified references must be module-qualified strings"
            )
        module, separator, qualname = value.partition(":")
        if not separator or ":" in qualname:
            raise AlgorithmConfigurationError(
                "qualified references must use 'module:attribute' syntax"
            )
        return cls(module=module, qualname=qualname)

    def __str__(self) -> str:
        return f"{self.module}:{self.qualname}"


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class BackendInputCompatibility:
    """Declare the input contract accepted by one implementation Backend."""

    accepted_input_views: tuple[str, ...]
    accepted_ingestion_engines: tuple[str, ...]
    required_input_capabilities: tuple[str, ...]
    supported_explicit_adapters: tuple[QualifiedReference, ...]
    distribution_policy: tuple[RuntimeTopology, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "accepted_input_views",
            "accepted_ingestion_engines",
            "required_input_capabilities",
        ):
            values = tuple(getattr(self, field_name))
            if any(not isinstance(value, str) or not value for value in values):
                raise AlgorithmConfigurationError(
                    f"{field_name} must contain non-empty strings"
                )
            for value in values:
                _require_namespaced_id(value, field_name)
            if len(set(values)) != len(values):
                raise AlgorithmConfigurationError(
                    f"{field_name} must not contain duplicates"
                )
            object.__setattr__(self, field_name, tuple(sorted(values)))
        if not self.accepted_input_views:
            raise AlgorithmConfigurationError(
                "accepted_input_views must declare at least one view"
            )
        if not self.accepted_ingestion_engines:
            raise AlgorithmConfigurationError(
                "accepted_ingestion_engines must declare at least one engine"
            )

        adapters = tuple(self.supported_explicit_adapters)
        if not adapters or any(
            not isinstance(adapter, QualifiedReference) for adapter in adapters
        ):
            raise AlgorithmConfigurationError(
                "supported_explicit_adapters must contain qualified references"
            )
        if len({str(adapter) for adapter in adapters}) != len(adapters):
            raise AlgorithmConfigurationError(
                "supported_explicit_adapters must not contain duplicates"
            )
        object.__setattr__(
            self,
            "supported_explicit_adapters",
            tuple(sorted(adapters, key=str)),
        )

        try:
            distribution_policy = tuple(
                RuntimeTopology(topology) for topology in self.distribution_policy
            )
        except (TypeError, ValueError) as exc:
            raise AlgorithmConfigurationError(
                f"distribution_policy contains an invalid topology: {exc}"
            ) from exc
        if not distribution_policy:
            raise AlgorithmConfigurationError(
                "distribution_policy must declare at least one topology"
            )
        if len(set(distribution_policy)) != len(distribution_policy):
            raise AlgorithmConfigurationError(
                "distribution_policy must not contain duplicates"
            )
        object.__setattr__(
            self,
            "distribution_policy",
            tuple(sorted(distribution_policy, key=lambda topology: topology.value)),
        )


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class ImplementationDescriptor:
    """Describe implementation code without importing or instantiating it."""

    implementation_id: str
    version: str
    execution_mode: ExecutionMode
    implementation_ref: QualifiedReference
    executable_factory_ref: QualifiedReference
    operations: tuple[AlgorithmOperation, ...]
    input_compatibility: BackendInputCompatibility
    distribution: str | None = None
    framework: str | None = None
    code_digest: str | None = None
    artifact_format: Literal["none", "trusted_pickle"] = "none"
    allowed_config_keys: tuple[str, ...] = ()
    runtime_id: str | None = None
    worker_input_adapter_ref: QualifiedReference | None = None
    exporter_ref: QualifiedReference | None = None
    flavor_id: str | None = None

    def __post_init__(self) -> None:
        _require_namespaced_id(self.implementation_id, "implementation_id")
        if not isinstance(self.input_compatibility, BackendInputCompatibility):
            raise AlgorithmConfigurationError(
                "input_compatibility must be a BackendInputCompatibility"
            )
        if not isinstance(self.version, str) or not self.version:
            raise AlgorithmConfigurationError("implementation version must be set")
        try:
            execution_mode = ExecutionMode(self.execution_mode)
            operations = tuple(
                AlgorithmOperation(operation) for operation in self.operations
            )
        except (TypeError, ValueError) as exc:
            raise AlgorithmConfigurationError(
                f"invalid implementation mode or operation: {exc}"
            ) from exc
        object.__setattr__(self, "execution_mode", execution_mode)
        object.__setattr__(self, "operations", operations)
        if not self.operations:
            raise AlgorithmConfigurationError(
                "an implementation must declare at least one operation"
            )
        if len(set(self.operations)) != len(self.operations):
            raise AlgorithmConfigurationError(
                "implementation operations must not contain duplicates"
            )
        if self.distribution is not None and (
            not isinstance(self.distribution, str) or not self.distribution
        ):
            raise AlgorithmConfigurationError("distribution must be non-empty")
        if self.distribution is not None:
            try:
                distribution_requirement = Requirement(self.distribution)
            except InvalidRequirement as exc:
                raise AlgorithmConfigurationError(
                    f"invalid implementation distribution: {self.distribution!r}"
                ) from exc
            if (
                distribution_requirement.extras
                or distribution_requirement.specifier
                or distribution_requirement.url is not None
                or distribution_requirement.marker is not None
            ):
                raise AlgorithmConfigurationError(
                    "implementation distribution must be a package name without "
                    "extras or version constraints"
                )
            object.__setattr__(
                self,
                "distribution",
                canonicalize_name(distribution_requirement.name),
            )
        if self.framework is not None and (
            not isinstance(self.framework, str) or not self.framework
        ):
            raise AlgorithmConfigurationError("framework must be non-empty")
        if self.code_digest is not None and (
            not isinstance(self.code_digest, str)
            or _DIGEST.fullmatch(self.code_digest) is None
        ):
            raise AlgorithmConfigurationError(
                "code_digest must be a lower-case SHA-256 digest"
            )
        if self.artifact_format not in {"none", "trusted_pickle"}:
            raise AlgorithmConfigurationError(
                "artifact_format must be 'none' or 'trusted_pickle'"
            )
        object.__setattr__(self, "allowed_config_keys", tuple(self.allowed_config_keys))
        if any(not isinstance(key, str) or not key for key in self.allowed_config_keys):
            raise AlgorithmConfigurationError(
                "allowed_config_keys must contain non-empty strings"
            )
        if len(set(self.allowed_config_keys)) != len(self.allowed_config_keys):
            raise AlgorithmConfigurationError(
                "allowed_config_keys must not contain duplicates"
            )
        object.__setattr__(
            self, "allowed_config_keys", tuple(sorted(self.allowed_config_keys))
        )
        if self.runtime_id is not None:
            _require_namespaced_id(self.runtime_id, "runtime_id")
        if self.flavor_id is not None:
            _require_namespaced_id(self.flavor_id, "flavor_id")
        for field_name in (
            "worker_input_adapter_ref",
            "exporter_ref",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, QualifiedReference):
                raise AlgorithmConfigurationError(
                    f"{field_name} must be a QualifiedReference when provided"
                )
        formal_mode = self.execution_mode in {
            ExecutionMode.COLLECTIVE,
            ExecutionMode.MAP_REDUCE,
            ExecutionMode.FRAMEWORK_NATIVE,
        }
        if formal_mode and (
            self.runtime_id is None or self.worker_input_adapter_ref is None
        ):
            raise AlgorithmConfigurationError(
                "formal distributed implementations must declare runtime_id and "
                "worker_input_adapter_ref"
            )


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class EnvironmentSpec:
    """Declarative Worker environment constraints."""

    environment_id: str
    python: str = ">=3.12,<3.14"
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_namespaced_id(self.environment_id, "environment_id")
        if not isinstance(self.python, str):
            raise AlgorithmConfigurationError(
                "Python version constraint must be a string"
            )
        try:
            SpecifierSet(self.python)
        except (InvalidSpecifier, TypeError) as exc:
            raise AlgorithmConfigurationError(
                f"invalid Python version constraint: {self.python!r}"
            ) from exc
        normalized: list[str] = []
        distributions: set[str] = set()
        for dependency in self.dependencies:
            try:
                requirement = Requirement(dependency)
            except (InvalidRequirement, TypeError) as exc:
                raise AlgorithmConfigurationError(
                    f"invalid dependency requirement: {dependency!r}"
                ) from exc
            if requirement.url is not None:
                raise AlgorithmConfigurationError(
                    "environment dependencies must use version constraints, not URLs"
                )
            distribution = canonicalize_name(requirement.name)
            if distribution in distributions:
                raise AlgorithmConfigurationError(
                    f"environment dependency is declared more than once: {distribution}"
                )
            distributions.add(distribution)
            rendered = str(requirement)
            normalized.append(distribution + rendered[len(requirement.name) :])
        object.__setattr__(self, "dependencies", tuple(sorted(normalized)))

    @property
    def current_python_version(self) -> str:
        """Return the current major.minor.micro version for diagnostics."""
        return ".".join(str(part) for part in sys.version_info[:3])


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class RuntimeBinding:
    """Describe the Ray runtime and Worker input adapter selected at planning."""

    runtime_id: str
    worker_input_adapter_ref: QualifiedReference
    topology: RuntimeTopology = RuntimeTopology.SINGLE_WORKER
    worker_count: int = 1
    framework_parallelism: int = 1
    result_reducer_ref: QualifiedReference | None = None
    num_cpus: float = 1.0
    num_gpus: float = 0.0
    custom_resources: Mapping[str, float] = field(default_factory=dict)
    max_retries: int = 0
    execution_profile: ExecutionProfile | None = None
    strategy: DistributionStrategy | None = None
    distribution_digest: str | None = None
    resume_from: str | None = None

    def __post_init__(self) -> None:
        _require_namespaced_id(self.runtime_id, "runtime_id")
        try:
            topology = RuntimeTopology(self.topology)
        except (TypeError, ValueError) as exc:
            raise AlgorithmConfigurationError(
                f"invalid runtime topology: {self.topology!r}"
            ) from exc
        object.__setattr__(self, "topology", topology)
        if (
            not isinstance(self.worker_count, int)
            or isinstance(self.worker_count, bool)
            or not isinstance(self.framework_parallelism, int)
            or isinstance(self.framework_parallelism, bool)
        ):
            raise AlgorithmConfigurationError(
                "runtime Worker count and framework parallelism must be integers"
            )
        if self.worker_count < 1 or self.framework_parallelism < 1:
            raise AlgorithmConfigurationError(
                "runtime Worker count and framework parallelism must be positive"
            )
        if topology is RuntimeTopology.SINGLE_WORKER and (
            self.worker_count != 1
            or self.framework_parallelism != 1
            or self.result_reducer_ref is not None
        ):
            raise AlgorithmConfigurationError(
                "single_worker requires one Worker, framework_parallelism=1, "
                "and no result reducer"
            )
        if topology is RuntimeTopology.FRAMEWORK_MANAGED and (
            self.worker_count != 1
            or self.framework_parallelism < 2
            or self.result_reducer_ref is not None
        ):
            raise AlgorithmConfigurationError(
                "framework_managed requires one outer Worker, framework_parallelism "
                "of at least two, and no result reducer"
            )
        if topology is RuntimeTopology.DATA_PARALLEL and (
            self.worker_count < 2
            or self.framework_parallelism != 1
            or self.result_reducer_ref is None
        ):
            raise AlgorithmConfigurationError(
                "data_parallel requires at least two Workers, "
                "framework_parallelism=1, and a result reducer"
            )
        if (
            not isinstance(self.num_cpus, (int, float))
            or isinstance(self.num_cpus, bool)
            or not isinstance(self.num_gpus, (int, float))
            or isinstance(self.num_gpus, bool)
        ):
            raise AlgorithmConfigurationError(
                "runtime CPU and GPU requirements must be numeric"
            )
        object.__setattr__(self, "num_cpus", float(self.num_cpus))
        object.__setattr__(self, "num_gpus", float(self.num_gpus))
        if (
            not math.isfinite(self.num_cpus)
            or not math.isfinite(self.num_gpus)
            or self.num_cpus < 0
            or self.num_gpus < 0
        ):
            raise AlgorithmConfigurationError(
                "runtime CPU and GPU requirements must be finite and non-negative"
            )
        custom_resources: dict[str, float] = {}
        for name, value in self.custom_resources.items():
            if not isinstance(name, str) or not name:
                raise AlgorithmConfigurationError(
                    "runtime custom resource names must be non-empty strings"
                )
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise AlgorithmConfigurationError(
                    f"runtime custom resource {name!r} must be finite and non-negative"
                )
            custom_resources[name] = float(value)
        object.__setattr__(self, "custom_resources", FrozenDict(custom_resources))
        if not isinstance(self.max_retries, int) or isinstance(self.max_retries, bool):
            raise AlgorithmConfigurationError("max_retries must be an integer")
        if self.max_retries < 0:
            raise AlgorithmConfigurationError("max_retries must be non-negative")
        if topology is RuntimeTopology.DATA_PARALLEL and self.max_retries != 0:
            raise AlgorithmConfigurationError(
                "the first data_parallel slice requires max_retries=0"
            )
        formal_topologies = {
            RuntimeTopology.RAY_TRAIN_COLLECTIVE,
            RuntimeTopology.FRAMEWORK_NATIVE,
            RuntimeTopology.RAY_MAP_REDUCE,
        }
        if topology in formal_topologies and (
            self.framework_parallelism != 1 or self.result_reducer_ref is not None
        ):
            raise AlgorithmConfigurationError(
                "formal distributed runtimes use explicit worker groups and may not "
                "use the legacy result reducer"
            )
        if (self.execution_profile is None) != (self.strategy is None):
            raise AlgorithmConfigurationError(
                "execution_profile and strategy must be resolved together"
            )
        if self.execution_profile is not None:
            try:
                profile = ExecutionProfile(self.execution_profile)
                strategy = DistributionStrategy(self.strategy)
            except (TypeError, ValueError) as exc:
                raise AlgorithmConfigurationError(
                    f"invalid resolved execution profile or strategy: {exc}"
                ) from exc
            object.__setattr__(self, "execution_profile", profile)
            object.__setattr__(self, "strategy", strategy)
            expected_topology = FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS[strategy].topology
            if topology is not expected_topology:
                raise AlgorithmConfigurationError(
                    "resolved RuntimeBinding topology conflicts with strategy"
                )
            if (
                not isinstance(self.distribution_digest, str)
                or _DIGEST.fullmatch(self.distribution_digest) is None
            ):
                raise AlgorithmConfigurationError(
                    "formal RuntimeBinding requires a DistributionSpec digest"
                )
        elif self.distribution_digest is not None:
            raise AlgorithmConfigurationError(
                "legacy RuntimeBinding must not carry a DistributionSpec digest"
            )
        if self.resume_from is not None and (
            not isinstance(self.resume_from, str) or not self.resume_from
        ):
            raise AlgorithmConfigurationError(
                "runtime resume_from must be a non-empty string when provided"
            )
        if self.execution_profile is None and self.resume_from is not None:
            raise AlgorithmConfigurationError(
                "legacy RuntimeBinding must not carry formal resume state"
            )


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class AlgorithmRegistration:
    """Associate one algorithm fact with one portable implementation."""

    spec: AlgorithmSpec
    implementation: ImplementationDescriptor
    environment: EnvironmentSpec
    runtime: RuntimeBinding | None = None
    distribution_spec: DistributionSpec | None = None
    is_default: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.is_default, bool):
            raise AlgorithmConfigurationError("is_default must be a boolean")
        if self.spec.trainer_cls is not None:
            raise AlgorithmConfigurationError(
                "portable registrations must not store trainer_cls"
            )
        if self.spec.config_model is not None:
            raise AlgorithmConfigurationError(
                "portable registrations must use config_contract_ref, not config_model"
            )
        if not self.spec.operations:
            raise AlgorithmConfigurationError(
                "portable AlgorithmSpec must declare target operations"
            )
        missing_contracts = [
            field_name
            for field_name in (
                "learning_paradigm",
                "model_family",
                "lifecycle_kind",
                "config_contract_ref",
                "input_contract_ref",
                "output_contract_ref",
            )
            if not getattr(self.spec, field_name)
        ]
        if not self.spec.data_modalities:
            missing_contracts.append("data_modalities")
        if not self.spec.allowed_execution_modes:
            missing_contracts.append("allowed_execution_modes")
        if missing_contracts:
            raise AlgorithmConfigurationError(
                "portable AlgorithmSpec is missing target contract field(s): "
                f"{sorted(missing_contracts)}"
            )
        try:
            spec_operations = tuple(
                AlgorithmOperation(operation) for operation in self.spec.operations
            )
        except (TypeError, ValueError) as exc:
            raise AlgorithmConfigurationError(
                f"portable AlgorithmSpec has an invalid operation: {exc}"
            ) from exc
        if set(self.implementation.operations) - set(spec_operations):
            raise AlgorithmConfigurationError(
                "implementation offers an operation absent from AlgorithmSpec"
            )
        if (
            self.spec.allowed_execution_modes
            and self.implementation.execution_mode.value
            not in self.spec.allowed_execution_modes
        ):
            raise AlgorithmConfigurationError(
                "implementation execution mode is not allowed by AlgorithmSpec"
            )
        if self.runtime is not None and self.distribution_spec is not None:
            raise AlgorithmConfigurationError(
                "registration must bind either a legacy runtime or a "
                "DistributionSpec, not both"
            )
        if self.runtime is None and self.distribution_spec is None:
            raise AlgorithmConfigurationError(
                "registration requires a legacy runtime or a DistributionSpec"
            )
        if self.runtime is not None and (
            self.runtime.topology is RuntimeTopology.FRAMEWORK_MANAGED
            and self.implementation.execution_mode
            not in {ExecutionMode.MANAGED_ESTIMATOR, ExecutionMode.LEGACY_TRAINER}
        ):
            raise AlgorithmConfigurationError(
                "framework_managed topology is supported only for managed estimators "
                "and the bounded legacy Trainer adapter"
            )
        if (
            self.runtime is not None
            and self.runtime.topology is RuntimeTopology.DATA_PARALLEL
            and self.implementation.execution_mode
            is not ExecutionMode.CUSTOM_RAY_FUNCTION
        ):
            raise AlgorithmConfigurationError(
                "the first data_parallel topology supports only Custom Ray Function"
            )
        if (
            self.runtime is not None
            and self.runtime.topology
            not in self.implementation.input_compatibility.distribution_policy
        ):
            raise AlgorithmConfigurationError(
                "runtime topology is not accepted by the implementation's "
                "distribution_policy"
            )
        if (
            self.runtime is not None
            and self.runtime.topology is RuntimeTopology.DATA_PARALLEL
        ):
            reducer_ref = self.runtime.result_reducer_ref
            if (
                reducer_ref is None
                or reducer_ref.module != self.implementation.implementation_ref.module
            ):
                raise AlgorithmConfigurationError(
                    "a data_parallel user reducer must be declared in the same module "
                    "as the user function"
                )
        if self.distribution_spec is not None:
            if not isinstance(self.distribution_spec, DistributionSpec):
                raise AlgorithmConfigurationError(
                    "distribution_spec must be a DistributionSpec"
                )
            expected_mode = FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS[
                self.distribution_spec.strategy
            ].execution_mode
            if self.implementation.execution_mode is not expected_mode:
                raise AlgorithmConfigurationError(
                    "implementation execution mode conflicts with DistributionSpec"
                )
            has_exporter = self.implementation.exporter_ref is not None
            has_flavor = self.implementation.flavor_id is not None
            if has_exporter != has_flavor:
                raise AlgorithmConfigurationError(
                    "formal distributed exporter_ref and flavor_id must be declared "
                    "together"
                )
            if (
                AlgorithmOperation.FIT in self.implementation.operations
                and self.distribution_spec.result_policy is ResultPolicy.BUNDLE_REQUIRED
                and not has_exporter
            ):
                raise AlgorithmConfigurationError(
                    "bundle_required formal fit must declare exporter_ref and flavor_id"
                )
        unknown_defaults = sorted(
            set(self.spec.default_config) - set(self.implementation.allowed_config_keys)
        )
        if unknown_defaults:
            raise AlgorithmConfigurationError(
                f"AlgorithmSpec defaults contain undeclared key(s): {unknown_defaults}"
            )
        leaked_defaults = _credential_paths(
            self.spec.default_config,
            prefix="default_config",
        )
        if leaked_defaults:
            raise AlgorithmConfigurationError(
                "AlgorithmSpec defaults must not contain credential values; "
                f"sensitive field(s): {sorted(leaked_defaults)}"
            )
        if self.implementation.distribution is not None:
            declared = {
                canonicalize_name(Requirement(requirement).name)
                for requirement in self.environment.dependencies
            }
            if canonicalize_name(self.implementation.distribution) not in declared:
                raise AlgorithmConfigurationError(
                    "implementation distribution must be constrained by "
                    "EnvironmentSpec.dependencies"
                )


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class InputBinding:
    """Bind one algorithm input role to an invocation-scoped resolver reference."""

    name: str
    resolver_id: str
    reference: str
    feature_names: tuple[str, ...]
    label_name: str | None = None
    sample_weight_name: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or not isinstance(self.reference, str)
            or not self.reference
        ):
            raise AlgorithmConfigurationError(
                "input binding name and reference must be non-empty"
            )
        if _reference_contains_credentials(self.reference):
            raise AlgorithmConfigurationError(
                "input binding reference must not contain credentials"
            )
        _require_namespaced_id(self.resolver_id, "resolver_id")
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        if not self.feature_names or any(
            not isinstance(name, str) or not name for name in self.feature_names
        ):
            raise AlgorithmConfigurationError(
                "input binding requires non-empty feature names"
            )
        if len(set(self.feature_names)) != len(self.feature_names):
            raise AlgorithmConfigurationError("feature names must be unique")
        if self.label_name is not None:
            if not isinstance(self.label_name, str) or not self.label_name:
                raise AlgorithmConfigurationError(
                    "label_name must be a non-empty string when set"
                )
            if self.label_name in self.feature_names:
                raise AlgorithmConfigurationError(
                    "label_name must not also be a feature name"
                )
        if self.sample_weight_name is not None:
            if (
                not isinstance(self.sample_weight_name, str)
                or not self.sample_weight_name
            ):
                raise AlgorithmConfigurationError(
                    "sample_weight_name must be a non-empty string when set"
                )
            if self.sample_weight_name in self.feature_names or (
                self.sample_weight_name == self.label_name
            ):
                raise AlgorithmConfigurationError(
                    "sample_weight_name must not also be a feature or label name"
                )
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version < 1
        ):
            raise AlgorithmConfigurationError("schema_version must be positive")

    def descriptor_payload(self) -> dict[str, Any]:
        """Return the credential-free fields used to detect binding drift."""
        return {
            "name": self.name,
            "resolver_id": self.resolver_id,
            "reference": self.reference,
            "feature_names": list(self.feature_names),
            "label_name": self.label_name,
            "sample_weight_name": self.sample_weight_name,
            "schema_version": self.schema_version,
        }


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class ResolvedInputDescriptor:
    """Credential-free planning result returned by an InputResolverPort."""

    resolver_id: str
    reference: str
    descriptor_version: int
    binding_digest: str
    engine_id: str
    view_kind: str
    input_capabilities: tuple[str, ...] = ()
    deferred_validations: tuple[str, ...] = ()
    resolver_payload: Mapping[str, Any] = field(default_factory=dict, repr=False)
    compatible_worker_input_adapter_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_namespaced_id(self.resolver_id, "resolver_id")
        if (
            not isinstance(self.reference, str)
            or not self.reference
            or not isinstance(self.engine_id, str)
            or not self.engine_id
            or not isinstance(self.view_kind, str)
            or not self.view_kind
        ):
            raise AlgorithmConfigurationError(
                "resolved input reference, engine_id, and view_kind must be non-empty"
            )
        if _reference_contains_credentials(self.reference):
            raise AlgorithmConfigurationError(
                "resolved input reference must not contain credentials"
            )
        _require_namespaced_id(self.engine_id, "engine_id")
        _require_namespaced_id(self.view_kind, "view_kind")
        if (
            not isinstance(self.descriptor_version, int)
            or isinstance(self.descriptor_version, bool)
            or self.descriptor_version < 1
        ):
            raise AlgorithmConfigurationError("descriptor_version must be positive")
        if (
            not isinstance(self.binding_digest, str)
            or _DIGEST.fullmatch(self.binding_digest) is None
        ):
            raise AlgorithmConfigurationError(
                "binding_digest must be a lower-case SHA-256 digest"
            )
        object.__setattr__(
            self, "deferred_validations", tuple(self.deferred_validations)
        )
        object.__setattr__(self, "input_capabilities", tuple(self.input_capabilities))
        if any(
            not isinstance(item, str) or not item for item in self.input_capabilities
        ) or len(set(self.input_capabilities)) != len(self.input_capabilities):
            raise AlgorithmConfigurationError(
                "input_capabilities must contain unique non-empty strings"
            )
        for capability in self.input_capabilities:
            _require_namespaced_id(capability, "input_capabilities")
        object.__setattr__(
            self, "input_capabilities", tuple(sorted(self.input_capabilities))
        )
        if any(
            not isinstance(item, str) or not item for item in self.deferred_validations
        ) or len(set(self.deferred_validations)) != len(self.deferred_validations):
            raise AlgorithmConfigurationError(
                "deferred_validations must contain unique non-empty strings"
            )
        try:
            payload = deep_freeze(self.resolver_payload)
        except TypeError as exc:
            raise AlgorithmConfigurationError(
                "resolver_payload must contain only portable JSON values"
            ) from exc
        leaked = _credential_paths(payload, prefix="resolver_payload")
        if leaked:
            raise AlgorithmConfigurationError(
                "resolved input descriptor must not contain credential values; "
                f"sensitive field(s): {sorted(leaked)}"
            )
        _canonical_json(deep_thaw(payload))
        object.__setattr__(self, "resolver_payload", payload)
        object.__setattr__(
            self,
            "compatible_worker_input_adapter_refs",
            tuple(self.compatible_worker_input_adapter_refs),
        )
        if len(set(self.compatible_worker_input_adapter_refs)) != len(
            self.compatible_worker_input_adapter_refs
        ):
            raise AlgorithmConfigurationError(
                "compatible Worker input adapter references must be unique"
            )
        for reference in self.compatible_worker_input_adapter_refs:
            QualifiedReference.parse(reference)


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class AlgorithmRequest:
    """A bounded algorithm request with framework-free configuration."""

    algorithm: str
    operation: AlgorithmOperation
    input_binding: InputBinding
    algorithm_config: Mapping[str, Any] = field(default_factory=dict)
    implementation_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.algorithm, str) or not self.algorithm:
            raise AlgorithmConfigurationError("algorithm must be non-empty")
        try:
            operation = AlgorithmOperation(self.operation)
        except (TypeError, ValueError) as exc:
            raise AlgorithmConfigurationError(
                f"invalid algorithm operation: {self.operation!r}"
            ) from exc
        object.__setattr__(self, "operation", operation)
        if self.implementation_id is not None:
            _require_namespaced_id(self.implementation_id, "implementation_id")
        leaked = _credential_paths(self.algorithm_config)
        if leaked:
            raise AlgorithmConfigurationError(
                "algorithm_config must use secret references instead of credential "
                f"values; sensitive field(s): {sorted(leaked)}"
            )
        try:
            frozen = deep_freeze(self.algorithm_config)
        except TypeError as exc:
            raise AlgorithmConfigurationError(
                "algorithm_config must contain only portable JSON values"
            ) from exc
        _canonical_json(deep_thaw(frozen))
        object.__setattr__(self, "algorithm_config", frozen)


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class AlgorithmResolution:
    """The unique implementation and runtime selected for one request."""

    algorithm: str
    algorithm_version: str
    implementation_id: str
    implementation_version: str
    execution_mode: ExecutionMode
    environment_id: str
    runtime_id: str
    requested_algorithm: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.algorithm, str)
            or not self.algorithm
            or not isinstance(self.algorithm_version, str)
            or not self.algorithm_version
        ):
            raise AlgorithmConfigurationError(
                "algorithm resolution requires an identity and version"
            )
        if (
            not isinstance(self.requested_algorithm, str)
            or not self.requested_algorithm
        ):
            raise AlgorithmConfigurationError(
                "requested algorithm identity must be non-empty"
            )
        _require_namespaced_id(self.implementation_id, "implementation_id")
        _require_namespaced_id(self.environment_id, "environment_id")
        _require_namespaced_id(self.runtime_id, "runtime_id")
        if (
            not isinstance(self.implementation_version, str)
            or not self.implementation_version
        ):
            raise AlgorithmConfigurationError(
                "implementation resolution version must be set"
            )
        try:
            execution_mode = ExecutionMode(self.execution_mode)
        except (TypeError, ValueError) as exc:
            raise AlgorithmConfigurationError(
                f"invalid resolved execution mode: {self.execution_mode!r}"
            ) from exc
        object.__setattr__(self, "execution_mode", execution_mode)


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class ResolvedAlgorithmPlan:
    """Immutable, deterministic, credential-free bounded execution plan."""

    format_version: int
    plan_id: str
    operation: AlgorithmOperation
    resolution: AlgorithmResolution
    implementation: ImplementationDescriptor
    environment: EnvironmentSpec
    runtime: RuntimeBinding
    input_binding: InputBinding
    input_descriptor: ResolvedInputDescriptor
    algorithm_config: Mapping[str, Any]
    config_digest: str
    distribution_spec: DistributionSpec | None = None

    def __post_init__(self) -> None:
        if self.format_version < 1:
            raise AlgorithmConfigurationError("plan format_version must be positive")
        if not isinstance(self.plan_id, str) or _DIGEST.fullmatch(self.plan_id) is None:
            raise AlgorithmConfigurationError(
                "plan_id must be a lower-case SHA-256 digest"
            )
        try:
            operation = AlgorithmOperation(self.operation)
        except (TypeError, ValueError) as exc:
            raise AlgorithmConfigurationError(
                f"invalid plan operation: {self.operation!r}"
            ) from exc
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "algorithm_config", deep_freeze(self.algorithm_config))
        if (
            not isinstance(self.config_digest, str)
            or _DIGEST.fullmatch(self.config_digest) is None
        ):
            raise AlgorithmConfigurationError(
                "config_digest must be a lower-case SHA-256 digest"
            )
        if self.config_digest != canonical_digest(self.algorithm_config):
            raise AlgorithmConfigurationError(
                "config_digest does not match algorithm_config"
            )
        leaked = _credential_paths(self.algorithm_config)
        if leaked:
            raise AlgorithmConfigurationError(
                "resolved plan contains sensitive algorithm configuration field(s): "
                f"{sorted(leaked)}"
            )
        if self.operation not in self.implementation.operations:
            raise AlgorithmConfigurationError(
                "plan operation is not offered by the resolved implementation"
            )
        if self.distribution_spec is not None:
            if self.runtime.strategy is not self.distribution_spec.strategy:
                raise AlgorithmConfigurationError(
                    "resolved runtime strategy conflicts with DistributionSpec"
                )
            if self.runtime.distribution_digest != self.distribution_spec.digest:
                raise AlgorithmConfigurationError(
                    "resolved runtime DistributionSpec digest is inconsistent"
                )
        expected_resolution = (
            self.implementation.implementation_id,
            self.implementation.version,
            self.implementation.execution_mode,
            self.environment.environment_id,
            self.runtime.runtime_id,
        )
        actual_resolution = (
            self.resolution.implementation_id,
            self.resolution.implementation_version,
            self.resolution.execution_mode,
            self.resolution.environment_id,
            self.resolution.runtime_id,
        )
        if actual_resolution != expected_resolution:
            raise AlgorithmConfigurationError(
                "plan resolution conflicts with implementation, environment, or runtime"
            )
        if (
            self.input_descriptor.resolver_id != self.input_binding.resolver_id
            or self.input_descriptor.reference != self.input_binding.reference
            or self.input_descriptor.binding_digest
            != canonical_digest(self.input_binding.descriptor_payload())
        ):
            raise AlgorithmConfigurationError(
                "plan input descriptor conflicts with its input binding"
            )
        unknown = sorted(
            set(self.algorithm_config) - set(self.implementation.allowed_config_keys)
        )
        if unknown:
            raise AlgorithmConfigurationError(
                f"plan configuration contains undeclared key(s): {unknown}"
            )

    def validate_integrity(self) -> None:
        """Reject a plan whose canonical fields no longer match its digest."""
        expected = canonical_digest(self.to_dict(include_plan_id=False))
        if self.plan_id != expected:
            raise AlgorithmConfigurationError(
                "resolved algorithm plan digest does not match its contents"
            )

    def to_dict(self, *, include_plan_id: bool = True) -> dict[str, Any]:
        """Return the canonical public plan projection."""
        payload: dict[str, Any] = {
            "format_version": self.format_version,
            "operation": self.operation.value,
            "resolution": {
                "requested_algorithm": self.resolution.requested_algorithm,
                "canonical_algorithm": self.resolution.algorithm,
                "algorithm": self.resolution.algorithm,
                "algorithm_version": self.resolution.algorithm_version,
                "implementation_id": self.resolution.implementation_id,
                "implementation_version": self.resolution.implementation_version,
                "execution_mode": self.resolution.execution_mode.value,
                "environment_id": self.resolution.environment_id,
                "runtime_id": self.resolution.runtime_id,
            },
            "implementation": {
                "implementation_id": self.implementation.implementation_id,
                "version": self.implementation.version,
                "execution_mode": self.implementation.execution_mode.value,
                "implementation_ref": str(self.implementation.implementation_ref),
                "executable_factory_ref": str(
                    self.implementation.executable_factory_ref
                ),
                "operations": [
                    operation.value for operation in self.implementation.operations
                ],
                "input_compatibility": {
                    "accepted_input_views": list(
                        self.implementation.input_compatibility.accepted_input_views
                    ),
                    "accepted_ingestion_engines": list(
                        self.implementation.input_compatibility.accepted_ingestion_engines
                    ),
                    "required_input_capabilities": list(
                        self.implementation.input_compatibility.required_input_capabilities
                    ),
                    "supported_explicit_adapters": [
                        str(reference)
                        for reference in self.implementation.input_compatibility.supported_explicit_adapters
                    ],
                    "distribution_policy": [
                        topology.value
                        for topology in self.implementation.input_compatibility.distribution_policy
                    ],
                },
                "distribution": self.implementation.distribution,
                "framework": self.implementation.framework,
                "code_digest": self.implementation.code_digest,
                "artifact_format": self.implementation.artifact_format,
                "allowed_config_keys": list(self.implementation.allowed_config_keys),
                "runtime_id": self.implementation.runtime_id,
                "worker_input_adapter_ref": (
                    str(self.implementation.worker_input_adapter_ref)
                    if self.implementation.worker_input_adapter_ref is not None
                    else None
                ),
                "exporter_ref": (
                    str(self.implementation.exporter_ref)
                    if self.implementation.exporter_ref is not None
                    else None
                ),
                "flavor_id": self.implementation.flavor_id,
            },
            "environment": {
                "environment_id": self.environment.environment_id,
                "python": self.environment.python,
                "dependencies": list(self.environment.dependencies),
            },
            "runtime": {
                "runtime_id": self.runtime.runtime_id,
                "worker_input_adapter_ref": str(self.runtime.worker_input_adapter_ref),
                "topology": self.runtime.topology.value,
                "worker_count": self.runtime.worker_count,
                "framework_parallelism": self.runtime.framework_parallelism,
                "result_reducer_ref": (
                    str(self.runtime.result_reducer_ref)
                    if self.runtime.result_reducer_ref is not None
                    else None
                ),
                "num_cpus": self.runtime.num_cpus,
                "num_gpus": self.runtime.num_gpus,
                "custom_resources": dict(sorted(self.runtime.custom_resources.items())),
                "max_retries": self.runtime.max_retries,
                "execution_profile": (
                    self.runtime.execution_profile.value
                    if self.runtime.execution_profile is not None
                    else None
                ),
                "strategy": (
                    self.runtime.strategy.value
                    if self.runtime.strategy is not None
                    else None
                ),
                "distribution_digest": self.runtime.distribution_digest,
                "resume_from": self.runtime.resume_from,
            },
            "distribution_spec": (
                self.distribution_spec.to_dict()
                if self.distribution_spec is not None
                else None
            ),
            "input_descriptor": {
                "resolver_id": self.input_descriptor.resolver_id,
                "reference": self.input_descriptor.reference,
                "descriptor_version": self.input_descriptor.descriptor_version,
                "binding_digest": self.input_descriptor.binding_digest,
                "engine_id": self.input_descriptor.engine_id,
                "view_kind": self.input_descriptor.view_kind,
                "input_capabilities": list(self.input_descriptor.input_capabilities),
                "deferred_validations": list(
                    self.input_descriptor.deferred_validations
                ),
                "resolver_payload": deep_thaw(self.input_descriptor.resolver_payload),
                "compatible_worker_input_adapter_refs": list(
                    self.input_descriptor.compatible_worker_input_adapter_refs
                ),
            },
            "input_binding": self.input_binding.descriptor_payload(),
            "algorithm_config": deep_thaw(self.algorithm_config),
            "config_digest": self.config_digest,
        }
        if include_plan_id:
            payload["plan_id"] = self.plan_id
        return payload

    def to_json(self) -> str:
        """Serialize the canonical plan with deterministic ordering."""
        return _canonical_json(self.to_dict())


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class ArtifactDraft:
    """A typed artifact payload emitted by a Worker execution."""

    name: str
    kind: str
    format: str
    payload: bytes = field(repr=False)
    sha256: str
    trusted: bool = False

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.name, self.kind, self.format)
        ):
            raise AlgorithmConfigurationError(
                "artifact name, kind, and format must be non-empty"
            )
        if not isinstance(self.payload, bytes):
            raise AlgorithmConfigurationError("artifact payload must be bytes")
        if not isinstance(self.trusted, bool):
            raise AlgorithmConfigurationError("artifact trusted flag must be boolean")
        if not isinstance(self.sha256, str) or _DIGEST.fullmatch(self.sha256) is None:
            raise AlgorithmConfigurationError(
                "artifact sha256 must be a lower-case SHA-256 digest"
            )
        if hashlib.sha256(self.payload).hexdigest() != self.sha256:
            raise AlgorithmConfigurationError("artifact payload digest mismatch")

    @classmethod
    def from_payload(
        cls,
        *,
        name: str,
        kind: str,
        format: str,
        payload: bytes,
        trusted: bool = False,
    ) -> ArtifactDraft:
        """Create an artifact and calculate its content digest."""
        return cls(
            name=name,
            kind=kind,
            format=format,
            payload=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
            trusted=trusted,
        )


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class AlgorithmExecutionResult:
    """Framework-neutral result produced by one Worker lifecycle."""

    status: Literal["succeeded", "failed"]
    metrics: Mapping[str, Any] = field(default_factory=dict)
    outputs: Mapping[str, Any] = field(default_factory=dict)
    artifacts: tuple[ArtifactDraft, ...] = ()
    failure_category: FailureCategory | None = None
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"succeeded", "failed"}:
            raise AlgorithmConfigurationError("invalid execution result status")
        try:
            metrics = deep_freeze(self.metrics)
            outputs = deep_freeze(self.outputs)
        except TypeError as exc:
            raise AlgorithmConfigurationError(
                "execution metrics and outputs must contain portable JSON values"
            ) from exc
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "outputs", outputs)
        _canonical_json(deep_thaw(metrics))
        _canonical_json(deep_thaw(outputs))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        if any(not isinstance(artifact, ArtifactDraft) for artifact in self.artifacts):
            raise AlgorithmConfigurationError(
                "execution artifacts must contain only ArtifactDraft values"
            )
        if self.failure_category is not None:
            try:
                failure_category = FailureCategory(self.failure_category)
            except (TypeError, ValueError) as exc:
                raise AlgorithmConfigurationError(
                    f"invalid failure category: {self.failure_category!r}"
                ) from exc
            object.__setattr__(self, "failure_category", failure_category)
        if self.status == "succeeded" and any(
            value is not None
            for value in (
                self.failure_category,
                self.error_type,
                self.error_message,
            )
        ):
            raise AlgorithmConfigurationError(
                "successful results must not carry failure metadata"
            )
        if self.status == "failed" and (
            self.failure_category is None
            or self.error_type is None
            or self.error_message is None
        ):
            raise AlgorithmConfigurationError(
                "failed results require category, type, and message"
            )


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class WorkerExecutionResult:
    """Execution result plus actual Worker dependency versions."""

    execution: AlgorithmExecutionResult
    actual_versions: Mapping[str, str]
    worker_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.execution, AlgorithmExecutionResult):
            raise AlgorithmConfigurationError(
                "Worker execution must contain an AlgorithmExecutionResult"
            )
        if any(
            not isinstance(name, str) or not isinstance(version, str)
            for name, version in self.actual_versions.items()
        ):
            raise AlgorithmConfigurationError(
                "actual Worker versions must map strings to strings"
            )
        object.__setattr__(self, "actual_versions", FrozenDict(self.actual_versions))
        try:
            worker_metadata = deep_freeze(self.worker_metadata)
        except TypeError as exc:
            raise AlgorithmConfigurationError(
                "Worker metadata must contain only portable JSON values"
            ) from exc
        _canonical_json(deep_thaw(worker_metadata))
        object.__setattr__(self, "worker_metadata", worker_metadata)


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class AlgorithmRunResult:
    """Driver result with run identity and input provenance."""

    run_id: str
    plan_id: str
    execution: AlgorithmExecutionResult
    actual_versions: Mapping[str, str]
    input_provenance: Mapping[str, Any]
    worker_metadata: Mapping[str, Any] = field(default_factory=dict)
    execution_receipt: ExecutionReceipt | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise AlgorithmConfigurationError("run_id must be non-empty")
        if _DIGEST.fullmatch(self.plan_id) is None:
            raise AlgorithmConfigurationError("invalid plan_id in run result")
        if not isinstance(self.execution, AlgorithmExecutionResult):
            raise AlgorithmConfigurationError(
                "algorithm run must contain an AlgorithmExecutionResult"
            )
        object.__setattr__(self, "actual_versions", FrozenDict(self.actual_versions))
        object.__setattr__(self, "input_provenance", deep_freeze(self.input_provenance))
        object.__setattr__(self, "worker_metadata", deep_freeze(self.worker_metadata))
        if self.execution_receipt is not None:
            from tributo.algorithms.api.execution import ExecutionReceipt

            if not isinstance(self.execution_receipt, ExecutionReceipt):
                raise AlgorithmConfigurationError(
                    "execution_receipt must be an ExecutionReceipt when provided"
                )


__all__ = [
    "AlgorithmExecutionResult",
    "AlgorithmOperation",
    "AlgorithmRegistration",
    "AlgorithmRequest",
    "AlgorithmResolution",
    "AlgorithmRunResult",
    "ArtifactDraft",
    "EnvironmentSpec",
    "ExecutionMode",
    "FailureCategory",
    "ImplementationDescriptor",
    "InputBinding",
    "QualifiedReference",
    "ResolvedAlgorithmPlan",
    "ResolvedInputDescriptor",
    "RuntimeBinding",
    "RuntimeTopology",
    "WorkerExecutionResult",
    "canonical_digest",
]
