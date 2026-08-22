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
from typing import Any

from tributo._common.immutable import FrozenDict
from tributo.algorithms.api.errors import AlgorithmConfigurationError
from tributo.util.annotations import PublicAPI

_REFERENCE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
    r":[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)


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


@PublicAPI(stability="alpha")
class InputDistribution(str, Enum):
    """How training input reaches workers."""

    SHARDED = "sharded"
    FRAMEWORK_OWNED = "framework_owned"


@PublicAPI(stability="alpha")
class StateCoordination(str, Enum):
    """How worker-local state becomes one global model."""

    ALL_REDUCE = "all_reduce"
    FRAMEWORK_NATIVE = "framework_native"
    ASSOCIATIVE_REDUCE = "associative_reduce"


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

    def scaled(self, worker_count: int) -> WorkerResources:
        """Return total resources for *worker_count* identical workers."""
        _positive_integer(worker_count, "worker_count")
        return WorkerResources(
            num_cpus=self.num_cpus * worker_count,
            num_gpus=self.num_gpus * worker_count,
            custom={name: value * worker_count for name, value in self.custom.items()},
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a portable representation."""
        return {
            "num_cpus": self.num_cpus,
            "num_gpus": self.num_gpus,
            "custom": dict(sorted(self.custom.items())),
        }


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


StrategyPolicy = CollectivePolicy | MapReducePolicy | FrameworkNativePolicy


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
    "CollectivePolicy",
    "DistributionSpec",
    "DistributionStrategy",
    "ExecutionProfile",
    "FrameworkNativePolicy",
    "InputDistribution",
    "MapReducePolicy",
    "MetricReduction",
    "ResultPolicy",
    "StateCoordination",
    "StateField",
    "WorkerRange",
    "WorkerResources",
]
