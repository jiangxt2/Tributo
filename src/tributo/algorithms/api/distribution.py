"""Declarative contracts for supported distributed training strategies.

The types in this module describe algorithm facts.  Per-invocation choices
belong to :mod:`tributo.algorithms.api.execution` and resolved Ray placement
belongs to ``RuntimeBinding``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast

from tributo._common.immutable import FrozenDict
from tributo.algorithms.api.errors import AlgorithmConfigurationError
from tributo.util.annotations import PublicAPI

_REFERENCE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
    r":[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)
_NAMESPACED_ID = re.compile(r"^[a-z][a-z0-9_.-]*$")


def _positive_integer(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise AlgorithmConfigurationError(f"{field_name} must be a positive integer")
    return value


def _non_negative_integer(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AlgorithmConfigurationError(
            f"{field_name} must be a non-negative integer"
        )
    return value


def _boolean(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise AlgorithmConfigurationError(f"{field_name} must be a boolean")
    return value


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise AlgorithmConfigurationError(f"{field_name} must be non-empty")
    return value


def _require_namespaced_id(value: object, field_name: str) -> None:
    if not isinstance(value, str) or _NAMESPACED_ID.fullmatch(value) is None:
        raise AlgorithmConfigurationError(
            f"{field_name} must be a lower-case namespaced identifier"
        )


def _mapping(value: object, field_name: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise AlgorithmConfigurationError(f"{field_name} must be a mapping")
    return value


def _sequence(value: object, field_name: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise AlgorithmConfigurationError(f"{field_name} must be a sequence")
    return tuple(value)


def _number(value: object, field_name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise AlgorithmConfigurationError(f"{field_name} must be a finite number")
    return float(value)


def _qualified_reference(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _REFERENCE.fullmatch(value) is None:
        raise AlgorithmConfigurationError(
            f"{field_name} must use 'module:qualified.attribute' syntax"
        )
    return value


@PublicAPI(stability="alpha")
class ExecutionProfile(str, Enum):
    """Supported product execution targets.

    ``LOCAL`` is the Spark ``local[*]``-like owned runtime. ``CLUSTER`` is an
    attached Ray runtime whose provisioning mechanism remains outside the
    algorithm contract. Docker, Kubernetes, and VM providers are intentionally
    not product profiles.
    """

    LOCAL = "local"
    CLUSTER = "cluster"
    # Source compatibility for callers that referenced the old enum member.
    # Its serialized value is the deployment-neutral canonical value.
    KUBERNETES = "cluster"

    @classmethod
    def _missing_(cls, value: object) -> ExecutionProfile | None:
        if value == "kubernetes":
            warnings.warn(
                "execution profile 'kubernetes' is deprecated; use 'cluster'",
                DeprecationWarning,
                stacklevel=3,
            )
            return cls.CLUSTER
        return None


@PublicAPI(stability="alpha")
class DistributionStrategy(str, Enum):
    """Mathematical state-coordination strategy used by one algorithm."""

    RAY_TRAIN_COLLECTIVE = "ray_train_collective"
    FRAMEWORK_NATIVE = "framework_native"
    RAY_MAP_REDUCE = "ray_map_reduce"
    RAY_JOBLIB_ESTIMATOR = "ray_joblib_estimator"
    RAY_PARALLEL_ENSEMBLE = "ray_parallel_ensemble"
    RAY_ITERATIVE_OPTIMIZATION = "ray_iterative_optimization"
    RAY_TRAIN_TORCH = "ray_train_torch"


@PublicAPI(stability="alpha")
class InputDistribution(str, Enum):
    """How training input reaches workers."""

    SHARDED = "sharded"
    ROLE_ROUTED = "role_routed"
    FRAMEWORK_OWNED = "framework_owned"
    FULL_DATASET = "full_dataset"


@PublicAPI(stability="alpha")
class StateCoordination(str, Enum):
    """How worker-local state becomes one global model."""

    ALL_REDUCE = "all_reduce"
    TORCH_MANAGED = "torch_managed"
    FRAMEWORK_NATIVE = "framework_native"
    ASSOCIATIVE_REDUCE = "associative_reduce"
    ESTIMATOR_INTERNAL = "estimator_internal"
    ORDERED_ENSEMBLE = "ordered_ensemble"
    ITERATIVE_GLOBAL = "iterative_global"


@PublicAPI(stability="alpha")
class DistributedExactness(str, Enum):
    """Relationship between one distributed implementation and its baseline."""

    EXACT = "exact"
    CONDITIONAL = "conditional"
    APPROXIMATE = "approximate"


@PublicAPI(stability="alpha")
class ResultPolicy(str, Enum):
    """Permitted result-delivery behavior for a fit invocation."""

    BUNDLE_REQUIRED = "bundle_required"
    FIT_ONLY = "fit_only"


@PublicAPI(stability="alpha")
class MetricReduction(str, Enum):
    """Supported global metric reduction rules."""

    SUM_COUNT = "sum_count"
    WEIGHTED_MEAN = "weighted_mean"
    MIN = "min"
    MAX = "max"


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class WorkerRange:
    """Inclusive worker-count range accepted by an implementation."""

    minimum: int = 1
    maximum: int | None = None

    def __post_init__(self) -> None:
        _positive_integer(self.minimum, "worker range minimum")
        if self.maximum is not None:
            _positive_integer(self.maximum, "worker range maximum")
            if self.maximum < self.minimum:
                raise AlgorithmConfigurationError(
                    "worker range maximum must be greater than or equal to minimum"
                )

    def contains(self, worker_count: int) -> bool:
        """Return whether *worker_count* is inside the inclusive range."""
        return worker_count >= self.minimum and (
            self.maximum is None or worker_count <= self.maximum
        )


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class WorkerResources:
    """Ray resources required by each training worker."""

    num_cpus: float = 1.0
    num_gpus: float = 0.0
    custom: Mapping[str, float] = field(default_factory=dict)
    memory_bytes: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("num_cpus", "num_gpus"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise AlgorithmConfigurationError(
                    f"{field_name} must be finite and non-negative"
                )
            object.__setattr__(self, field_name, float(value))
        normalized: dict[str, float] = {}
        for name, value in self.custom.items():
            if not isinstance(name, str) or not name:
                raise AlgorithmConfigurationError(
                    "custom resource names must be non-empty strings"
                )
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise AlgorithmConfigurationError(
                    f"custom resource {name!r} must be finite and non-negative"
                )
            normalized[name] = float(value)
        object.__setattr__(self, "custom", FrozenDict(normalized))
        if self.memory_bytes is not None and (
            not isinstance(self.memory_bytes, int)
            or isinstance(self.memory_bytes, bool)
            or self.memory_bytes <= 0
        ):
            raise AlgorithmConfigurationError(
                "memory_bytes must be a positive integer when provided"
            )

    def scaled(self, worker_count: int) -> WorkerResources:
        """Return total resources for *worker_count* identical workers."""
        _positive_integer(worker_count, "worker_count")
        return WorkerResources(
            num_cpus=self.num_cpus * worker_count,
            num_gpus=self.num_gpus * worker_count,
            custom={name: value * worker_count for name, value in self.custom.items()},
            memory_bytes=(
                self.memory_bytes * worker_count
                if self.memory_bytes is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a portable representation."""
        value = {
            "num_cpus": self.num_cpus,
            "num_gpus": self.num_gpus,
            "custom": dict(sorted(self.custom.items())),
        }
        if self.memory_bytes is not None:
            value["memory_bytes"] = self.memory_bytes
        return value


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class StateField:
    """One bounded field in a MapReduce partial-state schema."""

    name: str
    dtype: str
    shape: tuple[int | None, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise AlgorithmConfigurationError("state field name must be non-empty")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise AlgorithmConfigurationError("state field dtype must be non-empty")
        shape = tuple(self.shape)
        if any(
            dimension is not None
            and (
                not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or dimension < 0
            )
            for dimension in shape
        ):
            raise AlgorithmConfigurationError(
                "state field dimensions must be non-negative integers or None"
            )
        object.__setattr__(self, "shape", shape)


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class CollectivePolicy:
    """Conditional contract for iterative collective training."""

    backend: str
    metric_reducers: Mapping[str, MetricReduction]
    checkpoint_owner_rank: int = 0
    same_world_size_resume: bool = True
    rank_seeded: bool = True

    def __post_init__(self) -> None:
        if self.backend not in {"gloo", "nccl", "auto"}:
            raise AlgorithmConfigurationError(
                "collective backend must be 'gloo', 'nccl', or 'auto'"
            )
        if (
            not isinstance(self.checkpoint_owner_rank, int)
            or isinstance(self.checkpoint_owner_rank, bool)
            or self.checkpoint_owner_rank < 0
        ):
            raise AlgorithmConfigurationError(
                "checkpoint_owner_rank must be a non-negative integer"
            )
        normalized: dict[str, MetricReduction] = {}
        for name, reduction in self.metric_reducers.items():
            if not isinstance(name, str) or not name:
                raise AlgorithmConfigurationError(
                    "metric reducer names must be non-empty strings"
                )
            try:
                normalized[name] = MetricReduction(reduction)
            except (TypeError, ValueError) as exc:
                raise AlgorithmConfigurationError(
                    f"invalid metric reducer for {name!r}: {reduction!r}"
                ) from exc
        if not normalized:
            raise AlgorithmConfigurationError(
                "collective training must declare at least one metric reducer"
            )
        object.__setattr__(self, "metric_reducers", FrozenDict(normalized))
        _boolean(self.same_world_size_resume, "same_world_size_resume")
        _boolean(self.rank_seeded, "rank_seeded")


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class MapReducePolicy:
    """Conditional contract for bounded associative model-state reduction."""

    state_schema: tuple[StateField, ...]
    max_partial_state_bytes: int
    reducer_ref: str
    finalizer_ref: str
    commutative: bool = True
    max_retries: int = 0

    def __post_init__(self) -> None:
        schema = tuple(self.state_schema)
        if not schema or any(not isinstance(item, StateField) for item in schema):
            raise AlgorithmConfigurationError(
                "MapReduce state_schema must contain StateField declarations"
            )
        if len({item.name for item in schema}) != len(schema):
            raise AlgorithmConfigurationError(
                "MapReduce state_schema field names must be unique"
            )
        object.__setattr__(self, "state_schema", schema)
        _positive_integer(self.max_partial_state_bytes, "max_partial_state_bytes")
        _qualified_reference(self.reducer_ref, "reducer_ref")
        _qualified_reference(self.finalizer_ref, "finalizer_ref")
        if (
            not isinstance(self.max_retries, int)
            or isinstance(self.max_retries, bool)
            or self.max_retries < 0
        ):
            raise AlgorithmConfigurationError(
                "MapReduce max_retries must be a non-negative integer"
            )
        _boolean(self.commutative, "MapReduce commutative")


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class JoblibEstimatorPolicy:
    """Bounded contract for estimators that expose internal Joblib tasks."""

    fit_operations: tuple[str, ...] = ("fit",)
    n_jobs_parameter: str = "n_jobs"
    max_materialized_rows: int = 100_000
    exactness: DistributedExactness = DistributedExactness.EXACT

    def __post_init__(self) -> None:
        operations = tuple(self.fit_operations)
        if not operations or any(
            not isinstance(item, str) or not item for item in operations
        ):
            raise AlgorithmConfigurationError(
                "Joblib fit_operations must contain non-empty strings"
            )
        if len(set(operations)) != len(operations):
            raise AlgorithmConfigurationError(
                "Joblib fit_operations must not contain duplicates"
            )
        object.__setattr__(self, "fit_operations", operations)
        _string(self.n_jobs_parameter, "n_jobs_parameter")
        _positive_integer(self.max_materialized_rows, "max_materialized_rows")
        try:
            exactness = DistributedExactness(self.exactness)
        except (TypeError, ValueError) as exc:
            raise AlgorithmConfigurationError("Joblib exactness is invalid") from exc
        object.__setattr__(self, "exactness", exactness)


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class ParallelEnsemblePolicy:
    """Bounded contract for deterministic independent ensemble units."""

    max_units: int
    max_unit_model_bytes: int = 64 * 1024 * 1024
    max_retries: int = 0
    checkpoint_interval: int = 1
    exactness: DistributedExactness = DistributedExactness.EXACT

    def __post_init__(self) -> None:
        _positive_integer(self.max_units, "max_units")
        _positive_integer(self.max_unit_model_bytes, "max_unit_model_bytes")
        _non_negative_integer(self.max_retries, "max_retries")
        _positive_integer(self.checkpoint_interval, "checkpoint_interval")
        try:
            exactness = DistributedExactness(self.exactness)
        except (TypeError, ValueError) as exc:
            raise AlgorithmConfigurationError(
                "Parallel Ensemble exactness is invalid"
            ) from exc
        object.__setattr__(self, "exactness", exactness)


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class IterativeOptimizationPolicy:
    """Bounded contract for synchronous partition-update training rounds."""

    max_rounds: int
    checkpoint_interval: int = 1
    max_state_bytes: int = 64 * 1024 * 1024
    max_update_bytes: int = 16 * 1024 * 1024
    max_retries: int = 0
    exactness: DistributedExactness = DistributedExactness.CONDITIONAL

    def __post_init__(self) -> None:
        _positive_integer(self.max_rounds, "max_rounds")
        _positive_integer(self.checkpoint_interval, "checkpoint_interval")
        _positive_integer(self.max_state_bytes, "max_state_bytes")
        _positive_integer(self.max_update_bytes, "max_update_bytes")
        _non_negative_integer(self.max_retries, "max_retries")
        try:
            exactness = DistributedExactness(self.exactness)
        except (TypeError, ValueError) as exc:
            raise AlgorithmConfigurationError(
                "Iterative Optimization exactness is invalid"
            ) from exc
        object.__setattr__(self, "exactness", exactness)


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class FrameworkNativePolicy:
    """Conditional contract for a framework-owned distributed trainer."""

    framework: str
    evidence_collector_ref: str
    manages_input_shards: bool = True
    manages_checkpoints: bool = True
    component_stages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.framework, str) or not self.framework:
            raise AlgorithmConfigurationError("framework must be non-empty")
        _qualified_reference(self.evidence_collector_ref, "evidence_collector_ref")
        _boolean(self.manages_input_shards, "manages_input_shards")
        _boolean(self.manages_checkpoints, "manages_checkpoints")
        stages = tuple(self.component_stages)
        if any(
            not isinstance(stage, str)
            or not stage
            or stage.strip() != stage
            or "," in stage
            for stage in stages
        ):
            raise AlgorithmConfigurationError(
                "framework-native component stages must be non-empty strings"
            )
        if len(stages) != len(set(stages)):
            raise AlgorithmConfigurationError(
                "framework-native component stages must be unique"
            )
        object.__setattr__(self, "component_stages", stages)
        if not self.manages_input_shards:
            raise AlgorithmConfigurationError(
                "framework-native training must prove framework-owned input shards"
            )
        if not self.manages_checkpoints:
            raise AlgorithmConfigurationError(
                "framework-native training must declare checkpoint ownership"
            )


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchDatasetRoute:
    """Role-specific bounded input routing for one Torch Policy."""

    role: str
    mode: str
    required: bool = True
    min_total_rows_if_present: int = 1
    min_rows_per_worker: int = 1
    empty_rank_policy: str = "reject"
    max_rows: int | None = None
    max_bytes_per_worker: int | None = None

    def __post_init__(self) -> None:
        _string(self.role, "Torch route role")
        if self.mode not in {"split_exact", "replicate", "split_framework"}:
            raise AlgorithmConfigurationError("invalid Torch route mode")
        _boolean(self.required, "Torch route required")
        _non_negative_integer(
            self.min_total_rows_if_present, "Torch route minimum total rows"
        )
        _non_negative_integer(
            self.min_rows_per_worker, "Torch route minimum rows per worker"
        )
        if self.empty_rank_policy not in {"reject", "zero_contribution"}:
            raise AlgorithmConfigurationError("invalid Torch empty rank policy")
        if (
            self.mode == "split_exact"
            and self.required
            and (
                self.min_total_rows_if_present < 1
                or self.min_rows_per_worker < 1
                or self.empty_rank_policy != "reject"
            )
        ):
            raise AlgorithmConfigurationError(
                "required split_exact training routes must reject empty ranks"
            )
        if (
            not self.required
            and self.mode == "split_exact"
            and (
                self.min_total_rows_if_present < 1
                or self.min_rows_per_worker != 0
                or self.empty_rank_policy != "zero_contribution"
            )
        ):
            raise AlgorithmConfigurationError(
                "optional split_exact routes must use zero-contribution empty ranks"
            )
        if self.empty_rank_policy == "zero_contribution" and self.required:
            raise AlgorithmConfigurationError(
                "zero-contribution routes must be optional evaluation roles"
            )
        if self.mode == "replicate":
            _positive_integer(self.max_rows, "Torch replicate max_rows")
            _positive_integer(
                self.max_bytes_per_worker,
                "Torch replicate max_bytes_per_worker",
            )
        elif self.max_rows is not None or self.max_bytes_per_worker is not None:
            raise AlgorithmConfigurationError(
                "replication budgets are only valid for replicate routes"
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": self.role,
            "mode": self.mode,
            "required": self.required,
            "min_total_rows_if_present": self.min_total_rows_if_present,
            "min_rows_per_worker": self.min_rows_per_worker,
            "empty_rank_policy": self.empty_rank_policy,
            "max_rows": self.max_rows,
            "max_bytes_per_worker": self.max_bytes_per_worker,
        }
        return payload


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchStageSpec:
    """One Core-orchestrated stage in a Torch execution plan."""

    stage_id: str
    worker_loop_ref: str
    input_roles: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    checkpoint_from_stage: str | None = None
    metric_mapping: Mapping[str, str] = field(default_factory=dict)
    checkpoint_required: bool = True
    checkpoint_interval_windows: int = 1

    def __post_init__(self) -> None:
        _string(self.stage_id, "Torch stage_id")
        _qualified_reference(self.worker_loop_ref, "Torch worker_loop_ref")
        roles = tuple(self.input_roles)
        if not roles or any(not isinstance(role, str) or not role for role in roles):
            raise AlgorithmConfigurationError("Torch stage input_roles are required")
        if len(set(roles)) != len(roles):
            raise AlgorithmConfigurationError("Torch stage input_roles must be unique")
        depends = tuple(self.depends_on)
        if any(not isinstance(item, str) or not item for item in depends):
            raise AlgorithmConfigurationError("Torch stage dependencies are invalid")
        if len(set(depends)) != len(depends):
            raise AlgorithmConfigurationError("Torch stage dependencies must be unique")
        if self.checkpoint_from_stage is not None and not isinstance(
            self.checkpoint_from_stage, str
        ):
            raise AlgorithmConfigurationError("Torch checkpoint_from_stage is invalid")
        _boolean(self.checkpoint_required, "Torch checkpoint_required")
        _positive_integer(
            self.checkpoint_interval_windows,
            "Torch checkpoint_interval_windows",
        )
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, str)
            or not value
            for name, value in self.metric_mapping.items()
        ):
            raise AlgorithmConfigurationError(
                "Torch stage metric_mapping must be named strings"
            )
        if len(set(self.metric_mapping.values())) != len(self.metric_mapping):
            raise AlgorithmConfigurationError(
                "Torch stage metric_mapping targets must be unique"
            )
        object.__setattr__(self, "input_roles", roles)
        object.__setattr__(self, "depends_on", depends)
        object.__setattr__(
            self, "metric_mapping", FrozenDict(dict(self.metric_mapping))
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stage_id": self.stage_id,
            "worker_loop_ref": self.worker_loop_ref,
            "input_roles": list(self.input_roles),
            "depends_on": list(self.depends_on),
            "checkpoint_from_stage": self.checkpoint_from_stage,
            "metric_mapping": dict(self.metric_mapping),
            "checkpoint_required": self.checkpoint_required,
            "checkpoint_interval_windows": self.checkpoint_interval_windows,
        }
        return payload


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchExecutionPlan:
    """Versioned closed union base for single and component Torch plans."""

    api_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"api_version": self.api_version}

    @property
    def stages(self) -> tuple[TorchStageSpec, ...]:
        stage = getattr(self, "stage", None)
        return (stage,) if isinstance(stage, TorchStageSpec) else ()

    @property
    def final_stage_id(self) -> str:
        explicit = getattr(self, "_final_stage_id", None)
        if isinstance(explicit, str) and explicit:
            return explicit
        stages = self.stages
        return stages[-1].stage_id if len(stages) == 1 else ""

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class SingleStageTorchPlan(TorchExecutionPlan):
    """One-stage Torch execution plan."""

    stage: TorchStageSpec = field(
        default_factory=lambda: TorchStageSpec(
            "train",
            "tributo.integrations.algorithm_runtimes.ray_train_torch:"
            "torch_recipe_train_loop_per_worker",
            ("train",),
        )
    )

    def __post_init__(self) -> None:
        if self.api_version != 1:
            raise AlgorithmConfigurationError(
                "Torch execution plan api_version must be 1"
            )
        if self.stage.depends_on or self.stage.checkpoint_from_stage is not None:
            raise AlgorithmConfigurationError(
                "single Torch stage cannot have dependencies"
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": "single",
            "api_version": self.api_version,
            "stages": [self.stage.to_dict()],
            "final_stage_id": self.stage.stage_id,
        }
        return payload


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class ComponentStageTorchPlan(TorchExecutionPlan):
    """Ordered multi-stage Torch execution plan."""

    stages: tuple[TorchStageSpec, ...] = ()
    final_stage_id: str = ""

    def __post_init__(self) -> None:
        if self.api_version != 1 or not self.stages:
            raise AlgorithmConfigurationError("component Torch plan requires stages")
        stages = tuple(self.stages)
        ids = [stage.stage_id for stage in stages]
        if len(set(ids)) != len(ids) or self.final_stage_id not in ids:
            raise AlgorithmConfigurationError("component Torch stage IDs are invalid")
        prior: set[str] = set()
        for stage in stages:
            if any(dep not in prior for dep in stage.depends_on):
                raise AlgorithmConfigurationError(
                    "Torch stage dependencies must reference earlier stages"
                )
            if (
                stage.checkpoint_from_stage is not None
                and stage.checkpoint_from_stage not in prior
            ):
                raise AlgorithmConfigurationError(
                    "Torch checkpoint_from_stage must reference an earlier stage"
                )
            if stage.checkpoint_from_stage is not None:
                source = next(
                    item
                    for item in stages
                    if item.stage_id == stage.checkpoint_from_stage
                )
                if not source.checkpoint_required:
                    raise AlgorithmConfigurationError(
                        "Torch checkpoint_from_stage requires a checkpoint-producing source"
                    )
            prior.add(stage.stage_id)
        object.__setattr__(self, "stages", stages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "component",
            "api_version": self.api_version,
            "stages": [stage.to_dict() for stage in self.stages],
            "final_stage_id": self.final_stage_id,
        }


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchPolicy:
    """Versioned policy carrying all Torch routing and execution facts."""

    torch_runtime_api_version: int
    loop_owner: str
    parallelism_id: str
    dataset_routing: tuple[TorchDatasetRoute, ...]
    execution_plan: TorchExecutionPlan
    state_layout: str
    metric_reducers: Mapping[str, MetricReduction]
    backend: str = "auto"
    checkpoint_owner_rank: int = 0
    resume_supported: bool = True
    same_world_size_resume: bool | None = True
    rank_seeded: bool = True
    checkpoint_adapter_ref: str | None = None
    evidence_adapter_ref: str | None = None
    global_loss_reducer_ref: str | None = None
    global_loss_reducer_api_version: int | None = None
    global_loss_reducer_code_digest: str | None = None
    composite_loss_schema_id: str | None = None
    capabilities: tuple[str, ...] = ()
    max_replicated_bytes_per_worker: int | None = None

    def __post_init__(self) -> None:
        if self.torch_runtime_api_version != 1:
            raise AlgorithmConfigurationError("Torch Runtime API version must be 1")
        if self.loop_owner not in {"core_recipe", "adapter"}:
            raise AlgorithmConfigurationError("Torch loop_owner is invalid")
        _require_namespaced_id(self.parallelism_id, "Torch parallelism_id")
        _qualified_reference(
            self.checkpoint_adapter_ref, "checkpoint_adapter_ref"
        ) if self.checkpoint_adapter_ref else None
        _qualified_reference(
            self.evidence_adapter_ref, "evidence_adapter_ref"
        ) if self.evidence_adapter_ref else None
        routes = tuple(self.dataset_routing)
        if not routes or any(
            not isinstance(route, TorchDatasetRoute) for route in routes
        ):
            raise AlgorithmConfigurationError("Torch Policy requires dataset routes")
        if len({route.role for route in routes}) != len(routes):
            raise AlgorithmConfigurationError("Torch Policy route roles must be unique")
        if not isinstance(self.execution_plan, TorchExecutionPlan):
            raise AlgorithmConfigurationError("Torch Policy requires an execution plan")
        declared_stages = (
            (self.execution_plan.stage,)
            if isinstance(self.execution_plan, SingleStageTorchPlan)
            else self.execution_plan.stages
        )
        stage_roles = {role for stage in declared_stages for role in stage.input_roles}
        route_roles = {route.role for route in routes}
        missing_roles = sorted(stage_roles - route_roles)
        if missing_roles:
            raise AlgorithmConfigurationError(
                f"Torch Policy is missing routes for stage role(s): {missing_roles}"
            )
        if route_roles - stage_roles:
            raise AlgorithmConfigurationError(
                "Torch Policy declares an input route unused by its execution plan"
            )
        if any(
            route.mode == "split_framework" and self.loop_owner != "adapter"
            for route in routes
        ):
            raise AlgorithmConfigurationError(
                "split_framework routing is only valid for Adapter-owned loops"
            )
        replicated_budget = sum(
            route.max_bytes_per_worker or 0
            for route in routes
            if route.mode == "replicate"
        )
        if replicated_budget and (
            self.max_replicated_bytes_per_worker is None
            or replicated_budget > self.max_replicated_bytes_per_worker
        ):
            raise AlgorithmConfigurationError(
                "Torch replicate routes exceed max_replicated_bytes_per_worker"
            )
        if self.state_layout not in {"replicated", "component", "sharded"}:
            raise AlgorithmConfigurationError("Torch Policy state_layout is invalid")
        if self.state_layout == "replicated" and not isinstance(
            self.execution_plan, SingleStageTorchPlan
        ):
            raise AlgorithmConfigurationError(
                "replicated Torch state requires a single stage plan"
            )
        if self.state_layout == "component" and not isinstance(
            self.execution_plan, ComponentStageTorchPlan
        ):
            raise AlgorithmConfigurationError(
                "component Torch state requires a component plan"
            )
        if self.backend not in {"auto", "gloo", "nccl"}:
            raise AlgorithmConfigurationError("Torch backend is invalid")
        _non_negative_integer(self.checkpoint_owner_rank, "Torch checkpoint_owner_rank")
        _boolean(self.resume_supported, "Torch resume_supported")
        _boolean(self.rank_seeded, "Torch rank_seeded")
        if self.same_world_size_resume is not None:
            _boolean(self.same_world_size_resume, "Torch same_world_size_resume")
        if self.resume_supported and self.same_world_size_resume is not True:
            raise AlgorithmConfigurationError(
                "Torch v1 recovery only supports same-world-size resume"
            )
        if not self.resume_supported and self.same_world_size_resume is not None:
            raise AlgorithmConfigurationError(
                "unsupported Torch resume must omit same_world_size_resume"
            )
        if self.max_replicated_bytes_per_worker is not None:
            _positive_integer(
                self.max_replicated_bytes_per_worker,
                "Torch max_replicated_bytes_per_worker",
            )
        reducer_fields = (
            self.global_loss_reducer_ref,
            self.global_loss_reducer_api_version,
            self.global_loss_reducer_code_digest,
            self.composite_loss_schema_id,
        )
        if any(value is not None for value in reducer_fields):
            if not all(value is not None for value in reducer_fields):
                raise AlgorithmConfigurationError(
                    "Torch composite reducer fields must be declared together"
                )
            _qualified_reference(
                cast(str, self.global_loss_reducer_ref), "global_loss_reducer_ref"
            )
            if self.global_loss_reducer_api_version != 1:
                raise AlgorithmConfigurationError(
                    "Torch global reducer API version must be 1"
                )
            _digest_value = self.global_loss_reducer_code_digest
            if (
                not isinstance(_digest_value, str)
                or len(_digest_value) != 64
                or any(char not in "0123456789abcdef" for char in _digest_value)
            ):
                raise AlgorithmConfigurationError(
                    "Torch reducer code digest is invalid"
                )
        if len(set(self.capabilities)) != len(self.capabilities):
            raise AlgorithmConfigurationError("Torch capabilities must be unique")
        for capability in self.capabilities:
            _require_namespaced_id(capability, "Torch capability")
        try:
            metric_reducers = {
                _string(name, "Torch metric name"): MetricReduction(value)
                for name, value in self.metric_reducers.items()
            }
        except (TypeError, ValueError) as exc:
            raise AlgorithmConfigurationError(
                "Torch metric reducer is invalid"
            ) from exc
        if (
            "train_loss" not in metric_reducers
            or metric_reducers["train_loss"] is not MetricReduction.SUM_COUNT
        ):
            raise AlgorithmConfigurationError(
                "Torch metric_reducers must declare train_loss=sum_count"
            )
        object.__setattr__(self, "dataset_routing", routes)
        object.__setattr__(self, "metric_reducers", FrozenDict(metric_reducers))
        object.__setattr__(self, "capabilities", tuple(sorted(set(self.capabilities))))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "torch_runtime_api_version": self.torch_runtime_api_version,
            "loop_owner": self.loop_owner,
            "parallelism_id": self.parallelism_id,
            "dataset_routing": [route.to_dict() for route in self.dataset_routing],
            "execution_plan": self.execution_plan.to_dict(),
            "state_layout": self.state_layout,
            "metric_reducers": {
                name: value.value
                for name, value in sorted(self.metric_reducers.items())
            },
            "backend": self.backend,
            "checkpoint_owner_rank": self.checkpoint_owner_rank,
            "resume_supported": self.resume_supported,
            "rank_seeded": self.rank_seeded,
            "checkpoint_adapter_ref": self.checkpoint_adapter_ref,
            "evidence_adapter_ref": self.evidence_adapter_ref,
            "global_loss_reducer_ref": self.global_loss_reducer_ref,
            "global_loss_reducer_api_version": self.global_loss_reducer_api_version,
            "global_loss_reducer_code_digest": self.global_loss_reducer_code_digest,
            "composite_loss_schema_id": self.composite_loss_schema_id,
            "capabilities": list(self.capabilities),
            "max_replicated_bytes_per_worker": self.max_replicated_bytes_per_worker,
        }
        if self.same_world_size_resume is not None:
            payload["same_world_size_resume"] = self.same_world_size_resume
        return payload

    @property
    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TorchPolicy":
        """Reconstruct a policy without importing an implementation module."""
        try:
            plan_value = _mapping(value["execution_plan"], "Torch execution_plan")
            raw_stages = _sequence(plan_value["stages"], "Torch execution stages")
            stages = tuple(
                TorchStageSpec(
                    stage_id=_string(
                        _mapping(item, "Torch stage")["stage_id"], "stage_id"
                    ),
                    worker_loop_ref=_string(
                        _mapping(item, "Torch stage")["worker_loop_ref"],
                        "worker_loop_ref",
                    ),
                    input_roles=tuple(
                        _sequence(
                            _mapping(item, "Torch stage")["input_roles"], "input_roles"
                        )
                    ),
                    depends_on=tuple(
                        _sequence(
                            _mapping(item, "Torch stage").get("depends_on", ()),
                            "depends_on",
                        )
                    ),
                    checkpoint_from_stage=_mapping(item, "Torch stage").get(
                        "checkpoint_from_stage"
                    ),
                    metric_mapping=_mapping(
                        _mapping(item, "Torch stage").get("metric_mapping", {}),
                        "metric_mapping",
                    ),
                    checkpoint_required=_boolean(
                        _mapping(item, "Torch stage").get("checkpoint_required", True),
                        "checkpoint_required",
                    ),
                )
                for item in raw_stages
            )
            if plan_value.get("kind") == "single":
                execution_plan: TorchExecutionPlan = SingleStageTorchPlan(
                    api_version=plan_value["api_version"], stage=stages[0]
                )
            elif plan_value.get("kind") == "component":
                execution_plan = ComponentStageTorchPlan(
                    api_version=plan_value["api_version"],
                    stages=stages,
                    final_stage_id=plan_value["final_stage_id"],
                )
            else:
                raise AlgorithmConfigurationError(
                    "Torch execution plan kind is invalid"
                )
            routes = tuple(
                TorchDatasetRoute(**dict(_mapping(item, "Torch route")))
                for item in _sequence(value["dataset_routing"], "dataset_routing")
            )
            metric_reducers = {
                name: MetricReduction(reducer)
                for name, reducer in _mapping(
                    value["metric_reducers"], "metric_reducers"
                ).items()
            }
            return cls(
                torch_runtime_api_version=value["torch_runtime_api_version"],
                loop_owner=value["loop_owner"],
                parallelism_id=value["parallelism_id"],
                dataset_routing=routes,
                execution_plan=execution_plan,
                state_layout=value["state_layout"],
                metric_reducers=metric_reducers,
                backend=value.get("backend", "auto"),
                checkpoint_owner_rank=value.get("checkpoint_owner_rank", 0),
                resume_supported=value.get("resume_supported", True),
                same_world_size_resume=value.get("same_world_size_resume"),
                rank_seeded=value.get("rank_seeded", True),
                checkpoint_adapter_ref=value.get("checkpoint_adapter_ref"),
                evidence_adapter_ref=value.get("evidence_adapter_ref"),
                global_loss_reducer_ref=value.get("global_loss_reducer_ref"),
                global_loss_reducer_api_version=value.get(
                    "global_loss_reducer_api_version"
                ),
                global_loss_reducer_code_digest=value.get(
                    "global_loss_reducer_code_digest"
                ),
                composite_loss_schema_id=value.get("composite_loss_schema_id"),
                capabilities=tuple(value.get("capabilities", ())),
                max_replicated_bytes_per_worker=value.get(
                    "max_replicated_bytes_per_worker"
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, AlgorithmConfigurationError):
                raise
            raise AlgorithmConfigurationError("invalid TorchPolicy payload") from exc


StrategyPolicy = (
    CollectivePolicy
    | MapReducePolicy
    | FrameworkNativePolicy
    | JoblibEstimatorPolicy
    | ParallelEnsemblePolicy
    | IterativeOptimizationPolicy
    | TorchPolicy
)


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class DistributionSpec:
    """Versioned, immutable distributed-training declaration."""

    strategy: DistributionStrategy
    supported_worker_range: WorkerRange
    supported_execution_profiles: tuple[ExecutionProfile, ...]
    resources_per_worker: WorkerResources
    input_distribution: InputDistribution
    state_coordination: StateCoordination
    policy: StrategyPolicy
    distributed_min_workers: int = 2
    result_policy: ResultPolicy = ResultPolicy.BUNDLE_REQUIRED
    api_version: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.api_version, int)
            or isinstance(self.api_version, bool)
            or self.api_version != 1
        ):
            raise AlgorithmConfigurationError(
                f"unsupported DistributionSpec api_version: {self.api_version!r}"
            )
        if not isinstance(self.supported_worker_range, WorkerRange):
            raise AlgorithmConfigurationError(
                "supported_worker_range must be WorkerRange"
            )
        if not isinstance(self.resources_per_worker, WorkerResources):
            raise AlgorithmConfigurationError(
                "resources_per_worker must be WorkerResources"
            )
        try:
            strategy = DistributionStrategy(self.strategy)
            profiles = tuple(
                ExecutionProfile(profile)
                for profile in self.supported_execution_profiles
            )
            input_distribution = InputDistribution(self.input_distribution)
            state_coordination = StateCoordination(self.state_coordination)
            result_policy = ResultPolicy(self.result_policy)
        except (TypeError, ValueError) as exc:
            raise AlgorithmConfigurationError(
                f"invalid DistributionSpec enum value: {exc}"
            ) from exc
        if not profiles or len(set(profiles)) != len(profiles):
            raise AlgorithmConfigurationError(
                "supported_execution_profiles must be non-empty and unique"
            )
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(
            self, "supported_execution_profiles", tuple(sorted(profiles, key=str))
        )
        object.__setattr__(self, "input_distribution", input_distribution)
        object.__setattr__(self, "state_coordination", state_coordination)
        object.__setattr__(self, "result_policy", result_policy)
        _positive_integer(self.distributed_min_workers, "distributed_min_workers")
        if self.distributed_min_workers < 2:
            raise AlgorithmConfigurationError(
                "distributed_min_workers must be at least two"
            )
        maximum = self.supported_worker_range.maximum
        if maximum is not None and self.distributed_min_workers > maximum:
            raise AlgorithmConfigurationError(
                "distributed_min_workers must be inside supported_worker_range"
            )

        expected: dict[
            DistributionStrategy,
            tuple[type[StrategyPolicy], InputDistribution, StateCoordination],
        ] = {
            DistributionStrategy.RAY_TRAIN_COLLECTIVE: (
                CollectivePolicy,
                InputDistribution.SHARDED,
                StateCoordination.ALL_REDUCE,
            ),
            DistributionStrategy.FRAMEWORK_NATIVE: (
                FrameworkNativePolicy,
                InputDistribution.FRAMEWORK_OWNED,
                StateCoordination.FRAMEWORK_NATIVE,
            ),
            DistributionStrategy.RAY_MAP_REDUCE: (
                MapReducePolicy,
                InputDistribution.SHARDED,
                StateCoordination.ASSOCIATIVE_REDUCE,
            ),
            DistributionStrategy.RAY_JOBLIB_ESTIMATOR: (
                JoblibEstimatorPolicy,
                InputDistribution.FULL_DATASET,
                StateCoordination.ESTIMATOR_INTERNAL,
            ),
            DistributionStrategy.RAY_PARALLEL_ENSEMBLE: (
                ParallelEnsemblePolicy,
                InputDistribution.FULL_DATASET,
                StateCoordination.ORDERED_ENSEMBLE,
            ),
            DistributionStrategy.RAY_ITERATIVE_OPTIMIZATION: (
                IterativeOptimizationPolicy,
                InputDistribution.SHARDED,
                StateCoordination.ITERATIVE_GLOBAL,
            ),
            DistributionStrategy.RAY_TRAIN_TORCH: (
                TorchPolicy,
                InputDistribution.ROLE_ROUTED,
                StateCoordination.TORCH_MANAGED,
            ),
        }
        policy_type, expected_input, expected_state = expected[strategy]
        if not isinstance(self.policy, policy_type):
            raise AlgorithmConfigurationError(
                f"{strategy.value} requires {policy_type.__name__}"
            )
        if input_distribution is not expected_input:
            raise AlgorithmConfigurationError(
                f"{strategy.value} requires input_distribution={expected_input.value!r}"
            )
        if state_coordination is not expected_state:
            raise AlgorithmConfigurationError(
                f"{strategy.value} requires state_coordination={expected_state.value!r}"
            )
        if isinstance(self.policy, CollectivePolicy):
            maximum_rank = self.supported_worker_range.maximum
            if (
                maximum_rank is not None
                and self.policy.checkpoint_owner_rank >= maximum_rank
            ):
                raise AlgorithmConfigurationError(
                    "checkpoint_owner_rank must fit every supported worker group"
                )
        if isinstance(self.policy, TorchPolicy):
            maximum_rank = self.supported_worker_range.maximum
            if (
                maximum_rank is not None
                and self.policy.checkpoint_owner_rank >= maximum_rank
            ):
                raise AlgorithmConfigurationError(
                    "Torch checkpoint_owner_rank must fit every supported worker group"
                )

    def supports(self, profile: ExecutionProfile, worker_count: int) -> bool:
        """Return whether profile and worker count are declared as executable."""
        return ExecutionProfile(
            profile
        ) in self.supported_execution_profiles and self.supported_worker_range.contains(
            worker_count
        )

    def to_dict(self) -> dict[str, Any]:
        """Return canonical JSON-compatible metadata."""
        policy: dict[str, Any]
        if isinstance(self.policy, CollectivePolicy):
            policy = {
                "kind": "collective",
                "backend": self.policy.backend,
                "metric_reducers": {
                    name: reduction.value
                    for name, reduction in sorted(self.policy.metric_reducers.items())
                },
                "checkpoint_owner_rank": self.policy.checkpoint_owner_rank,
                "same_world_size_resume": self.policy.same_world_size_resume,
                "rank_seeded": self.policy.rank_seeded,
            }
        elif isinstance(self.policy, MapReducePolicy):
            policy = {
                "kind": "map_reduce",
                "state_schema": [
                    {
                        "name": item.name,
                        "dtype": item.dtype,
                        "shape": list(item.shape),
                    }
                    for item in self.policy.state_schema
                ],
                "max_partial_state_bytes": self.policy.max_partial_state_bytes,
                "reducer_ref": self.policy.reducer_ref,
                "finalizer_ref": self.policy.finalizer_ref,
                "commutative": self.policy.commutative,
                "max_retries": self.policy.max_retries,
            }
        elif isinstance(self.policy, JoblibEstimatorPolicy):
            policy = {
                "kind": "joblib_estimator",
                "fit_operations": list(self.policy.fit_operations),
                "n_jobs_parameter": self.policy.n_jobs_parameter,
                "max_materialized_rows": self.policy.max_materialized_rows,
                "exactness": self.policy.exactness.value,
            }
        elif isinstance(self.policy, ParallelEnsemblePolicy):
            policy = {
                "kind": "parallel_ensemble",
                "max_units": self.policy.max_units,
                "max_unit_model_bytes": self.policy.max_unit_model_bytes,
                "max_retries": self.policy.max_retries,
                "checkpoint_interval": self.policy.checkpoint_interval,
                "exactness": self.policy.exactness.value,
            }
        elif isinstance(self.policy, IterativeOptimizationPolicy):
            policy = {
                "kind": "iterative_optimization",
                "max_rounds": self.policy.max_rounds,
                "checkpoint_interval": self.policy.checkpoint_interval,
                "max_state_bytes": self.policy.max_state_bytes,
                "max_update_bytes": self.policy.max_update_bytes,
                "max_retries": self.policy.max_retries,
                "exactness": self.policy.exactness.value,
            }
        elif isinstance(self.policy, TorchPolicy):
            policy = {
                "kind": "torch",
                **self.policy.to_dict(),
            }
        else:
            policy = {
                "kind": "framework_native",
                "framework": self.policy.framework,
                "evidence_collector_ref": self.policy.evidence_collector_ref,
                "manages_input_shards": self.policy.manages_input_shards,
                "manages_checkpoints": self.policy.manages_checkpoints,
            }
            if self.policy.component_stages:
                policy["component_stages"] = list(self.policy.component_stages)
        return {
            "api_version": self.api_version,
            "strategy": self.strategy.value,
            "supported_worker_range": {
                "minimum": self.supported_worker_range.minimum,
                "maximum": self.supported_worker_range.maximum,
            },
            "distributed_min_workers": self.distributed_min_workers,
            "supported_execution_profiles": [
                profile.value for profile in self.supported_execution_profiles
            ],
            "resources_per_worker": self.resources_per_worker.to_dict(),
            "input_distribution": self.input_distribution.value,
            "state_coordination": self.state_coordination.value,
            "result_policy": self.result_policy.value,
            "policy": policy,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DistributionSpec:
        """Validate and reconstruct a spec from portable metadata."""
        try:
            policy_value = _mapping(value["policy"], "policy")
            kind = policy_value["kind"]
            if kind == "collective":
                metric_reducers = _mapping(
                    policy_value["metric_reducers"], "metric_reducers"
                )
                policy: StrategyPolicy = CollectivePolicy(
                    backend=_string(policy_value["backend"], "backend"),
                    metric_reducers={
                        _string(name, "metric name"): MetricReduction(reduction)
                        for name, reduction in metric_reducers.items()
                    },
                    checkpoint_owner_rank=_non_negative_integer(
                        policy_value["checkpoint_owner_rank"],
                        "checkpoint_owner_rank",
                    ),
                    same_world_size_resume=_boolean(
                        policy_value["same_world_size_resume"],
                        "same_world_size_resume",
                    ),
                    rank_seeded=_boolean(policy_value["rank_seeded"], "rank_seeded"),
                )
            elif kind == "map_reduce":
                state_schema = _sequence(policy_value["state_schema"], "state_schema")
                policy = MapReducePolicy(
                    state_schema=tuple(
                        StateField(
                            name=_string(
                                _mapping(item, "state field")["name"],
                                "state field name",
                            ),
                            dtype=_string(
                                _mapping(item, "state field")["dtype"],
                                "state field dtype",
                            ),
                            shape=_sequence(
                                _mapping(item, "state field")["shape"],
                                "state field shape",
                            ),
                        )
                        for item in state_schema
                    ),
                    max_partial_state_bytes=_positive_integer(
                        policy_value["max_partial_state_bytes"],
                        "max_partial_state_bytes",
                    ),
                    reducer_ref=_string(policy_value["reducer_ref"], "reducer_ref"),
                    finalizer_ref=_string(
                        policy_value["finalizer_ref"], "finalizer_ref"
                    ),
                    commutative=_boolean(policy_value["commutative"], "commutative"),
                    max_retries=_non_negative_integer(
                        policy_value["max_retries"], "max_retries"
                    ),
                )
            elif kind == "joblib_estimator":
                policy = JoblibEstimatorPolicy(
                    fit_operations=tuple(
                        _string(item, "fit operation")
                        for item in _sequence(
                            policy_value["fit_operations"], "fit_operations"
                        )
                    ),
                    n_jobs_parameter=_string(
                        policy_value["n_jobs_parameter"], "n_jobs_parameter"
                    ),
                    max_materialized_rows=_positive_integer(
                        policy_value["max_materialized_rows"],
                        "max_materialized_rows",
                    ),
                    exactness=DistributedExactness(policy_value["exactness"]),
                )
            elif kind == "parallel_ensemble":
                policy = ParallelEnsemblePolicy(
                    max_units=_positive_integer(policy_value["max_units"], "max_units"),
                    max_unit_model_bytes=_positive_integer(
                        policy_value["max_unit_model_bytes"],
                        "max_unit_model_bytes",
                    ),
                    max_retries=_non_negative_integer(
                        policy_value["max_retries"], "max_retries"
                    ),
                    checkpoint_interval=_positive_integer(
                        policy_value["checkpoint_interval"],
                        "checkpoint_interval",
                    ),
                    exactness=DistributedExactness(policy_value["exactness"]),
                )
            elif kind == "iterative_optimization":
                policy = IterativeOptimizationPolicy(
                    max_rounds=_positive_integer(
                        policy_value["max_rounds"], "max_rounds"
                    ),
                    checkpoint_interval=_positive_integer(
                        policy_value["checkpoint_interval"],
                        "checkpoint_interval",
                    ),
                    max_state_bytes=_positive_integer(
                        policy_value["max_state_bytes"], "max_state_bytes"
                    ),
                    max_update_bytes=_positive_integer(
                        policy_value["max_update_bytes"], "max_update_bytes"
                    ),
                    max_retries=_non_negative_integer(
                        policy_value["max_retries"], "max_retries"
                    ),
                    exactness=DistributedExactness(policy_value["exactness"]),
                )
            elif kind == "torch":
                execution_plan = _mapping(
                    policy_value["execution_plan"], "Torch execution_plan"
                )
                stage_values = _sequence(
                    execution_plan["stages"], "Torch execution stages"
                )
                stages = tuple(
                    TorchStageSpec(
                        stage_id=_string(
                            _mapping(item, "Torch stage")["stage_id"],
                            "Torch stage_id",
                        ),
                        worker_loop_ref=_string(
                            _mapping(item, "Torch stage")["worker_loop_ref"],
                            "Torch worker_loop_ref",
                        ),
                        input_roles=tuple(
                            _string(role, "Torch input role")
                            for role in _sequence(
                                _mapping(item, "Torch stage")["input_roles"],
                                "Torch input_roles",
                            )
                        ),
                        depends_on=tuple(
                            _string(dep, "Torch dependency")
                            for dep in _sequence(
                                _mapping(item, "Torch stage").get("depends_on", ()),
                                "Torch depends_on",
                            )
                        ),
                        checkpoint_from_stage=_mapping(item, "Torch stage").get(
                            "checkpoint_from_stage"
                        ),
                        metric_mapping=_mapping(
                            _mapping(item, "Torch stage").get("metric_mapping", {}),
                            "Torch metric_mapping",
                        ),
                        checkpoint_required=_boolean(
                            _mapping(item, "Torch stage").get(
                                "checkpoint_required", True
                            ),
                            "Torch checkpoint_required",
                        ),
                    )
                    for item in stage_values
                )
                plan_value: TorchExecutionPlan
                if execution_plan.get("kind") == "single":
                    plan_value = SingleStageTorchPlan(
                        api_version=_positive_integer(
                            execution_plan["api_version"],
                            "Torch execution plan api_version",
                        ),
                        stage=stages[0],
                    )
                else:
                    plan_value = ComponentStageTorchPlan(
                        api_version=_positive_integer(
                            execution_plan["api_version"],
                            "Torch execution plan api_version",
                        ),
                        stages=stages,
                        final_stage_id=_string(
                            execution_plan["final_stage_id"],
                            "Torch final_stage_id",
                        ),
                    )
                routes = tuple(
                    TorchDatasetRoute(
                        role=_string(
                            _mapping(item, "Torch route")["role"], "Torch role"
                        ),
                        mode=_string(
                            _mapping(item, "Torch route")["mode"], "Torch mode"
                        ),
                        required=_boolean(
                            _mapping(item, "Torch route").get("required", True),
                            "Torch route required",
                        ),
                        min_total_rows_if_present=_non_negative_integer(
                            _mapping(item, "Torch route").get(
                                "min_total_rows_if_present", 1
                            ),
                            "Torch minimum total rows",
                        ),
                        min_rows_per_worker=_non_negative_integer(
                            _mapping(item, "Torch route").get("min_rows_per_worker", 1),
                            "Torch minimum rows per worker",
                        ),
                        empty_rank_policy=_string(
                            _mapping(item, "Torch route").get(
                                "empty_rank_policy", "reject"
                            ),
                            "Torch empty rank policy",
                        ),
                        max_rows=_mapping(item, "Torch route").get("max_rows"),
                        max_bytes_per_worker=_mapping(item, "Torch route").get(
                            "max_bytes_per_worker"
                        ),
                    )
                    for item in _sequence(
                        policy_value["dataset_routing"], "Torch dataset_routing"
                    )
                )
                policy = TorchPolicy(
                    torch_runtime_api_version=_positive_integer(
                        policy_value["torch_runtime_api_version"],
                        "Torch Runtime API version",
                    ),
                    loop_owner=_string(policy_value["loop_owner"], "Torch loop_owner"),
                    parallelism_id=_string(
                        policy_value["parallelism_id"], "Torch parallelism_id"
                    ),
                    dataset_routing=routes,
                    execution_plan=plan_value,
                    state_layout=_string(
                        policy_value["state_layout"], "Torch state_layout"
                    ),
                    metric_reducers={
                        _string(name, "Torch metric name"): MetricReduction(value)
                        for name, value in _mapping(
                            policy_value["metric_reducers"], "Torch metric_reducers"
                        ).items()
                    },
                    backend=_string(
                        policy_value.get("backend", "auto"), "Torch backend"
                    ),
                    checkpoint_owner_rank=_non_negative_integer(
                        policy_value.get("checkpoint_owner_rank", 0),
                        "Torch checkpoint_owner_rank",
                    ),
                    resume_supported=_boolean(
                        policy_value.get("resume_supported", True),
                        "Torch resume_supported",
                    ),
                    same_world_size_resume=policy_value.get("same_world_size_resume"),
                    rank_seeded=_boolean(
                        policy_value.get("rank_seeded", True), "Torch rank_seeded"
                    ),
                    checkpoint_adapter_ref=policy_value.get("checkpoint_adapter_ref"),
                    evidence_adapter_ref=policy_value.get("evidence_adapter_ref"),
                    global_loss_reducer_ref=policy_value.get("global_loss_reducer_ref"),
                    global_loss_reducer_api_version=policy_value.get(
                        "global_loss_reducer_api_version"
                    ),
                    global_loss_reducer_code_digest=policy_value.get(
                        "global_loss_reducer_code_digest"
                    ),
                    composite_loss_schema_id=policy_value.get(
                        "composite_loss_schema_id"
                    ),
                    capabilities=tuple(
                        _string(capability, "Torch capability")
                        for capability in _sequence(
                            policy_value.get("capabilities", ()), "Torch capabilities"
                        )
                    ),
                    max_replicated_bytes_per_worker=(
                        _positive_integer(
                            policy_value["max_replicated_bytes_per_worker"],
                            "Torch max_replicated_bytes_per_worker",
                        )
                        if policy_value.get("max_replicated_bytes_per_worker")
                        is not None
                        else None
                    ),
                )
            elif kind == "framework_native":
                policy = FrameworkNativePolicy(
                    framework=_string(policy_value["framework"], "framework"),
                    evidence_collector_ref=_string(
                        policy_value["evidence_collector_ref"],
                        "evidence_collector_ref",
                    ),
                    manages_input_shards=_boolean(
                        policy_value["manages_input_shards"],
                        "manages_input_shards",
                    ),
                    manages_checkpoints=_boolean(
                        policy_value["manages_checkpoints"],
                        "manages_checkpoints",
                    ),
                    component_stages=tuple(
                        _string(stage, "component stage")
                        for stage in _sequence(
                            policy_value.get("component_stages", ()),
                            "component_stages",
                        )
                    ),
                )
            else:
                raise ValueError(f"unknown policy kind: {kind!r}")
            worker_range = _mapping(
                value["supported_worker_range"], "supported_worker_range"
            )
            resources = _mapping(value["resources_per_worker"], "resources")
            maximum = worker_range["maximum"]
            custom_resources = _mapping(resources.get("custom", {}), "custom resources")
            return cls(
                api_version=_positive_integer(value["api_version"], "api_version"),
                strategy=DistributionStrategy(value["strategy"]),
                supported_worker_range=WorkerRange(
                    minimum=_positive_integer(
                        worker_range["minimum"], "worker range minimum"
                    ),
                    maximum=(
                        _positive_integer(maximum, "worker range maximum")
                        if maximum is not None
                        else None
                    ),
                ),
                distributed_min_workers=_positive_integer(
                    value["distributed_min_workers"], "distributed_min_workers"
                ),
                supported_execution_profiles=tuple(
                    ExecutionProfile(profile)
                    for profile in _sequence(
                        value["supported_execution_profiles"],
                        "supported_execution_profiles",
                    )
                ),
                resources_per_worker=WorkerResources(
                    num_cpus=_number(resources["num_cpus"], "num_cpus"),
                    num_gpus=_number(resources["num_gpus"], "num_gpus"),
                    memory_bytes=(
                        _positive_integer(resources["memory_bytes"], "memory_bytes")
                        if "memory_bytes" in resources
                        else None
                    ),
                    custom={
                        _string(name, "custom resource name"): _number(
                            amount, "custom resource amount"
                        )
                        for name, amount in custom_resources.items()
                    },
                ),
                input_distribution=InputDistribution(value["input_distribution"]),
                state_coordination=StateCoordination(value["state_coordination"]),
                result_policy=ResultPolicy(value["result_policy"]),
                policy=policy,
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, AlgorithmConfigurationError):
                raise
            raise AlgorithmConfigurationError(
                f"invalid DistributionSpec payload: {exc}"
            ) from exc

    @property
    def digest(self) -> str:
        """Return a deterministic SHA-256 digest of the declaration."""
        payload = json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "ComponentStageTorchPlan",
    "CollectivePolicy",
    "DistributedExactness",
    "DistributionSpec",
    "DistributionStrategy",
    "ExecutionProfile",
    "FrameworkNativePolicy",
    "InputDistribution",
    "IterativeOptimizationPolicy",
    "JoblibEstimatorPolicy",
    "MapReducePolicy",
    "MetricReduction",
    "ParallelEnsemblePolicy",
    "ResultPolicy",
    "StateCoordination",
    "StateField",
    "SingleStageTorchPlan",
    "TorchDatasetRoute",
    "TorchExecutionPlan",
    "TorchPolicy",
    "TorchStageSpec",
    "WorkerRange",
    "WorkerResources",
]
