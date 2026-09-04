"""Per-invocation request and verifiable distributed execution receipt."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

from tributo._common.immutable import FrozenDict
from tributo.algorithms.api.distribution import (
    DistributionStrategy,
    ExecutionProfile,
    ResultPolicy,
    StateCoordination,
    WorkerResources,
)
from tributo.algorithms.api.errors import AlgorithmConfigurationError
from tributo.algorithms.api.models import (
    FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS,
    AlgorithmRequest,
)
from tributo.algorithms.api.torch_runtime import (
    TorchStageRunIdentity,
)
from tributo.util.annotations import PublicAPI

_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _non_empty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise AlgorithmConfigurationError(f"{field_name} must be non-empty")
    return value


def _integer(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise AlgorithmConfigurationError(f"{field_name} must be an integer")
    return value


def _positive_integer(value: object, field_name: str) -> int:
    """Validate a positive integer in worker evidence payloads."""
    result = _integer(value, field_name)
    if result < 1:
        raise AlgorithmConfigurationError(f"{field_name} must be positive")
    return result


def _number(value: object, field_name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise AlgorithmConfigurationError(f"{field_name} must be a finite number")
    return float(value)


def _mapping(value: object, field_name: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise AlgorithmConfigurationError(f"{field_name} must be a mapping")
    return value


def _boolean(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise AlgorithmConfigurationError(f"{field_name} must be a boolean")
    return value


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class ExecutionRequest:
    """Combine an algorithm request with one explicit execution choice."""

    algorithm_request: AlgorithmRequest
    profile: ExecutionProfile
    worker_count: int
    resources_per_worker: WorkerResources | None = None
    resume_from: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.algorithm_request, AlgorithmRequest):
            raise AlgorithmConfigurationError(
                "algorithm_request must be an AlgorithmRequest"
            )
        try:
            profile = ExecutionProfile(self.profile)
        except (TypeError, ValueError) as exc:
            raise AlgorithmConfigurationError(
                "execution profile must be 'local' or 'cluster'"
            ) from exc
        object.__setattr__(self, "profile", profile)
        if (
            not isinstance(self.worker_count, int)
            or isinstance(self.worker_count, bool)
            or self.worker_count < 1
        ):
            raise AlgorithmConfigurationError("worker_count must be a positive integer")
        if self.resources_per_worker is not None and not isinstance(
            self.resources_per_worker, WorkerResources
        ):
            raise AlgorithmConfigurationError(
                "resources_per_worker must be WorkerResources when provided"
            )
        if self.resume_from is not None:
            _non_empty(self.resume_from, "resume_from")


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class WorkerExecutionEvidence:
    """Observed identity, shard, and resource evidence for one worker."""

    worker_id: str
    node_id: str
    rank: int
    world_size: int
    shard_id: str
    resources: WorkerResources
    model_state_digest: str | None = None
    rows_processed: int | None = None
    input_rows: Mapping[str, int] = field(default_factory=dict)
    batch_count: int | None = None
    collective_steps: int | None = None

    def __post_init__(self) -> None:
        _non_empty(self.worker_id, "worker_id")
        _non_empty(self.node_id, "node_id")
        _non_empty(self.shard_id, "shard_id")
        if (
            not isinstance(self.rank, int)
            or isinstance(self.rank, bool)
            or self.rank < 0
        ):
            raise AlgorithmConfigurationError("rank must be non-negative")
        if (
            not isinstance(self.world_size, int)
            or isinstance(self.world_size, bool)
            or self.world_size < 1
            or self.rank >= self.world_size
        ):
            raise AlgorithmConfigurationError(
                "world_size must be positive and greater than rank"
            )
        if not isinstance(self.resources, WorkerResources):
            raise AlgorithmConfigurationError(
                "worker evidence resources must be WorkerResources"
            )
        if self.model_state_digest is not None and (
            not isinstance(self.model_state_digest, str)
            or _DIGEST.fullmatch(self.model_state_digest) is None
        ):
            raise AlgorithmConfigurationError(
                "model_state_digest must be a lower-case SHA-256 digest"
            )
        if self.rows_processed is not None and (
            not isinstance(self.rows_processed, int)
            or isinstance(self.rows_processed, bool)
            or self.rows_processed < 0
        ):
            raise AlgorithmConfigurationError(
                "rows_processed must be a non-negative integer"
            )
        normalized_input_rows: dict[str, int] = {}
        for name, count in self.input_rows.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count < 0
            ):
                raise AlgorithmConfigurationError(
                    "input_rows must map non-empty split names to non-negative integers"
                )
            normalized_input_rows[name] = count
        object.__setattr__(self, "input_rows", FrozenDict(normalized_input_rows))
        for name, value in (
            ("batch_count", self.batch_count),
            ("collective_steps", self.collective_steps),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise AlgorithmConfigurationError(
                    f"{name} must be a non-negative integer"
                )
        if (
            self.batch_count is not None
            and self.collective_steps is not None
            and self.batch_count > self.collective_steps
        ):
            raise AlgorithmConfigurationError(
                "batch_count cannot exceed collective_steps"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return portable evidence metadata."""
        return {
            "worker_id": self.worker_id,
            "node_id": self.node_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "shard_id": self.shard_id,
            "resources": self.resources.to_dict(),
            "model_state_digest": self.model_state_digest,
            "rows_processed": self.rows_processed,
            "input_rows": dict(sorted(self.input_rows.items())),
            "batch_count": self.batch_count,
            "collective_steps": self.collective_steps,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> WorkerExecutionEvidence:
        """Parse Worker evidence without coercing malformed values into facts."""
        try:
            resources = _mapping(value["resources"], "worker resources")
            custom = _mapping(resources.get("custom", {}), "custom resources")
            input_rows = _mapping(value.get("input_rows", {}), "input_rows")
            model_digest = value.get("model_state_digest")
            rows_processed = value.get("rows_processed")
            batch_count = value.get("batch_count")
            collective_steps = value.get("collective_steps")
            return cls(
                worker_id=_non_empty(value["worker_id"], "worker_id"),
                node_id=_non_empty(value["node_id"], "node_id"),
                rank=_integer(value["rank"], "rank"),
                world_size=_integer(value["world_size"], "world_size"),
                shard_id=_non_empty(value["shard_id"], "shard_id"),
                resources=WorkerResources(
                    num_cpus=_number(resources["num_cpus"], "worker num_cpus"),
                    num_gpus=_number(resources["num_gpus"], "worker num_gpus"),
                    memory_bytes=(
                        _positive_integer(
                            resources["memory_bytes"], "worker memory_bytes"
                        )
                        if "memory_bytes" in resources
                        else None
                    ),
                    custom={
                        _non_empty(name, "custom resource name"): _number(
                            amount, "custom resource amount"
                        )
                        for name, amount in custom.items()
                    },
                ),
                model_state_digest=(
                    _non_empty(model_digest, "model_state_digest")
                    if model_digest is not None
                    else None
                ),
                rows_processed=(
                    _integer(rows_processed, "rows_processed")
                    if rows_processed is not None
                    else None
                ),
                input_rows={
                    _non_empty(name, "input split name"): _integer(
                        count, "input row count"
                    )
                    for name, count in input_rows.items()
                },
                batch_count=(
                    _integer(batch_count, "batch_count")
                    if batch_count is not None
                    else None
                ),
                collective_steps=(
                    _integer(collective_steps, "collective_steps")
                    if collective_steps is not None
                    else None
                ),
            )
        except KeyError as exc:
            raise AlgorithmConfigurationError(
                f"worker evidence is missing field {exc.args[0]!r}"
            ) from exc


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class StateCoordinationEvidence:
    """Proof that worker-local state formed one bounded global model."""

    coordination: StateCoordination
    synchronized: bool
    bounded: bool
    global_model_digest: str | None = None
    details: Mapping[str, str | int | float | bool | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            coordination = StateCoordination(self.coordination)
        except (TypeError, ValueError) as exc:
            raise AlgorithmConfigurationError(
                f"invalid state coordination evidence: {self.coordination!r}"
            ) from exc
        object.__setattr__(self, "coordination", coordination)
        if not isinstance(self.synchronized, bool) or not isinstance(
            self.bounded, bool
        ):
            raise AlgorithmConfigurationError(
                "state coordination flags must be booleans"
            )
        if self.global_model_digest is not None and (
            not isinstance(self.global_model_digest, str)
            or _DIGEST.fullmatch(self.global_model_digest) is None
        ):
            raise AlgorithmConfigurationError(
                "global_model_digest must be a lower-case SHA-256 digest"
            )
        normalized: dict[str, str | int | float | bool | None] = {}
        for name, value in self.details.items():
            if not isinstance(name, str) or not name:
                raise AlgorithmConfigurationError(
                    "state evidence detail names must be non-empty strings"
                )
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise AlgorithmConfigurationError(
                    "state evidence details must contain JSON scalar values"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise AlgorithmConfigurationError(
                    "state evidence details must contain finite numbers"
                )
            normalized[name] = value
        object.__setattr__(self, "details", FrozenDict(normalized))

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> StateCoordinationEvidence:
        """Parse state evidence without truthiness or string coercion."""
        try:
            details = _mapping(value.get("details", {}), "state evidence details")
            digest = value.get("global_model_digest")
            return cls(
                coordination=cast(StateCoordination, value["coordination"]),
                synchronized=_boolean(value["synchronized"], "synchronized"),
                bounded=_boolean(value["bounded"], "bounded"),
                global_model_digest=(
                    _non_empty(digest, "global_model_digest")
                    if digest is not None
                    else None
                ),
                details=dict(details),
            )
        except KeyError as exc:
            raise AlgorithmConfigurationError(
                f"state evidence is missing field {exc.args[0]!r}"
            ) from exc


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchRoleExecutionEvidence:
    """Per-role coverage evidence for a Torch Stage."""

    role: str
    mode: str
    required: bool
    present: bool
    empty_rank_policy: str
    expected_rows: int | None
    observed_rows: int
    rows_per_rank: tuple[int, ...]
    replicated_bytes_per_worker: int | None = None
    binding_digest: str | None = None

    def __post_init__(self) -> None:
        _non_empty(self.role, "Torch role")
        if self.mode not in {"split_exact", "replicate", "split_framework"}:
            raise AlgorithmConfigurationError("invalid Torch role evidence mode")
        if not isinstance(self.required, bool) or not isinstance(self.present, bool):
            raise AlgorithmConfigurationError(
                "Torch role evidence flags must be boolean"
            )
        if self.empty_rank_policy not in {"reject", "zero_contribution"}:
            raise AlgorithmConfigurationError("invalid Torch role empty-rank policy")
        for name, value in (
            ("expected_rows", self.expected_rows),
            ("observed_rows", self.observed_rows),
            ("replicated_bytes_per_worker", self.replicated_bytes_per_worker),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise AlgorithmConfigurationError(
                    f"Torch role evidence {name} must be non-negative"
                )
        rows = tuple(self.rows_per_rank)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in rows
        ):
            raise AlgorithmConfigurationError("Torch rows_per_rank is malformed")
        if self.present:
            if self.mode == "replicate":
                if not rows or len(set(rows)) != 1 or rows[0] != self.observed_rows:
                    raise AlgorithmConfigurationError(
                        "Torch replicated role evidence is not identical"
                    )
                if self.replicated_bytes_per_worker is None:
                    raise AlgorithmConfigurationError(
                        "Torch replicated role evidence requires actual bytes"
                    )
            elif sum(rows) != self.observed_rows:
                raise AlgorithmConfigurationError("Torch role evidence rows do not sum")
        if not self.present and (self.observed_rows != 0 or any(rows)):
            raise AlgorithmConfigurationError(
                "absent Torch role must have zero evidence"
            )
        if (
            self.binding_digest is not None
            and _DIGEST.fullmatch(self.binding_digest) is None
        ):
            raise AlgorithmConfigurationError("Torch role binding_digest is invalid")
        object.__setattr__(self, "rows_per_rank", rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "mode": self.mode,
            "required": self.required,
            "present": self.present,
            "empty_rank_policy": self.empty_rank_policy,
            "expected_rows": self.expected_rows,
            "observed_rows": self.observed_rows,
            "rows_per_rank": list(self.rows_per_rank),
            "replicated_bytes_per_worker": self.replicated_bytes_per_worker,
            "binding_digest": self.binding_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TorchRoleExecutionEvidence":
        try:
            return cls(
                role=cast(str, value["role"]),
                mode=cast(str, value["mode"]),
                required=cast(bool, value["required"]),
                present=cast(bool, value["present"]),
                empty_rank_policy=cast(str, value["empty_rank_policy"]),
                expected_rows=cast(int | None, value.get("expected_rows")),
                observed_rows=cast(int, value["observed_rows"]),
                rows_per_rank=tuple(
                    cast(tuple[int, ...], value.get("rows_per_rank", ()))
                ),
                replicated_bytes_per_worker=cast(
                    int | None, value.get("replicated_bytes_per_worker")
                ),
                binding_digest=cast(str | None, value.get("binding_digest")),
            )
        except KeyError as exc:
            raise AlgorithmConfigurationError(
                "Torch role evidence is missing fields"
            ) from exc


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class ReplicatedTorchStateEvidence:
    """Evidence that all workers expose one synchronized model."""

    model_digests_by_rank: Mapping[int, str]
    global_model_digest: str

    def __post_init__(self) -> None:
        records = dict(self.model_digests_by_rank)
        if not records or any(
            not isinstance(rank, int) or rank < 0 for rank in records
        ):
            raise AlgorithmConfigurationError(
                "replicated Torch rank evidence is required"
            )
        if any(_DIGEST.fullmatch(value) is None for value in records.values()):
            raise AlgorithmConfigurationError(
                "replicated Torch model digest is invalid"
            )
        if _DIGEST.fullmatch(self.global_model_digest or "") is None:
            raise AlgorithmConfigurationError(
                "replicated Torch global digest is invalid"
            )
        if set(records.values()) != {self.global_model_digest}:
            raise AlgorithmConfigurationError(
                "replicated Torch ranks are not synchronized"
            )
        object.__setattr__(
            self,
            "model_digests_by_rank",
            MappingProxyType(dict(sorted(records.items()))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_digests_by_rank": {
                str(k): v for k, v in self.model_digests_by_rank.items()
            },
            "global_model_digest": self.global_model_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ReplicatedTorchStateEvidence":
        raw = _mapping(value.get("model_digests_by_rank", {}), "model_digests_by_rank")
        return cls(
            model_digests_by_rank={
                int(rank): cast(str, digest) for rank, digest in raw.items()
            },
            global_model_digest=cast(str, value["global_model_digest"]),
        )


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class ComponentStageEvidence:
    """Evidence for one Core-owned component Stage."""

    stage_id: str
    workers: tuple[WorkerExecutionEvidence, ...]
    roles: tuple[TorchRoleExecutionEvidence, ...]
    state_digest: str
    checkpoint_descriptor_digest: str | None = None

    def __post_init__(self) -> None:
        _non_empty(self.stage_id, "Torch stage evidence stage_id")
        workers = tuple(self.workers)
        if not workers or any(
            not isinstance(item, WorkerExecutionEvidence) for item in workers
        ):
            raise AlgorithmConfigurationError("Torch Stage evidence requires Workers")
        roles = tuple(self.roles)
        if any(not isinstance(item, TorchRoleExecutionEvidence) for item in roles):
            raise AlgorithmConfigurationError(
                "Torch Stage evidence roles are malformed"
            )
        for name, value in (("state_digest", self.state_digest),):
            if _DIGEST.fullmatch(value or "") is None:
                raise AlgorithmConfigurationError(
                    f"Torch Stage evidence {name} is invalid"
                )
        if {worker.model_state_digest for worker in workers} != {self.state_digest}:
            raise AlgorithmConfigurationError(
                "Torch Stage worker model digests do not match its state digest"
            )
        if (
            self.checkpoint_descriptor_digest is not None
            and _DIGEST.fullmatch(self.checkpoint_descriptor_digest) is None
        ):
            raise AlgorithmConfigurationError(
                "Torch Stage evidence checkpoint_descriptor_digest is invalid"
            )
        object.__setattr__(self, "workers", workers)
        object.__setattr__(self, "roles", roles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "workers": [item.to_dict() for item in self.workers],
            "roles": [item.to_dict() for item in self.roles],
            "state_digest": self.state_digest,
            "checkpoint_descriptor_digest": self.checkpoint_descriptor_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ComponentStageEvidence":
        workers_value = value.get("workers", ())
        roles_value = value.get("roles", ())
        if not isinstance(workers_value, (list, tuple)) or not isinstance(
            roles_value, (list, tuple)
        ):
            raise AlgorithmConfigurationError(
                "Torch Stage evidence worker/role fields are invalid"
            )
        return cls(
            stage_id=cast(str, value["stage_id"]),
            workers=tuple(
                WorkerExecutionEvidence.from_dict(item) for item in workers_value
            ),
            roles=tuple(
                TorchRoleExecutionEvidence.from_dict(item) for item in roles_value
            ),
            state_digest=cast(str, value["state_digest"]),
            checkpoint_descriptor_digest=cast(
                str | None, value.get("checkpoint_descriptor_digest")
            ),
        )


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchExecutionEvidence:
    """Layout-aware execution evidence attached only to Torch receipts."""

    identity: TorchStageRunIdentity
    run_config_name: str
    policy_digest: str
    parallelism_id: str
    state_layout: str
    workers: tuple[WorkerExecutionEvidence, ...]
    roles: tuple[TorchRoleExecutionEvidence, ...]
    replicated_state: ReplicatedTorchStateEvidence | None = None
    stages: tuple[ComponentStageEvidence, ...] = ()
    composition_digest: str | None = None
    final_stage_id: str | None = None
    reducer_id: str | None = None
    reducer_api_version: int | None = None
    reducer_schema_id: str | None = None
    reducer_code_digest: str | None = None
    torch_runtime_api_version: int = 1
    reducer_branch: str | None = None
    reducer_evidence: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, TorchStageRunIdentity):
            raise AlgorithmConfigurationError("Torch execution identity is required")
        if not isinstance(self.run_config_name, str) or not self.run_config_name:
            raise AlgorithmConfigurationError("Torch run_config_name is required")
        if self.run_config_name != self.identity.run_config_name:
            raise AlgorithmConfigurationError("Torch RunConfig name drifted")
        if (
            self.torch_runtime_api_version != 1
            or self.identity.torch_runtime_api_version != 1
        ):
            raise AlgorithmConfigurationError(
                "Torch runtime API version must be exactly 1"
            )
        if _DIGEST.fullmatch(self.policy_digest or "") is None:
            raise AlgorithmConfigurationError("Torch policy digest is invalid")
        if self.identity.policy_digest != self.policy_digest:
            raise AlgorithmConfigurationError("Torch execution policy digest drifted")
        if self.identity.execution_plan_digest == "":
            raise AlgorithmConfigurationError("Torch execution plan digest is required")
        if not isinstance(self.parallelism_id, str) or not self.parallelism_id:
            raise AlgorithmConfigurationError("Torch parallelism_id is required")
        if self.state_layout not in {"replicated", "component", "sharded"}:
            raise AlgorithmConfigurationError("Torch state layout is invalid")
        workers = tuple(self.workers)
        roles = tuple(self.roles)
        if not workers or any(
            not isinstance(item, WorkerExecutionEvidence) for item in workers
        ):
            raise AlgorithmConfigurationError(
                "Torch execution evidence requires workers"
            )
        if any(not isinstance(item, TorchRoleExecutionEvidence) for item in roles):
            raise AlgorithmConfigurationError(
                "Torch execution role evidence is malformed"
            )
        if self.state_layout == "replicated":
            if self.replicated_state is None or self.stages:
                raise AlgorithmConfigurationError(
                    "replicated Torch evidence has invalid state payload"
                )
        elif self.state_layout == "component":
            if not self.stages or self.replicated_state is not None:
                raise AlgorithmConfigurationError(
                    "component Torch evidence has invalid stage payload"
                )
            if _DIGEST.fullmatch(self.composition_digest or "") is None:
                raise AlgorithmConfigurationError(
                    "component Torch composition digest is required"
                )
            if len({stage.stage_id for stage in self.stages}) != len(self.stages):
                raise AlgorithmConfigurationError(
                    "component Torch Stage IDs must be unique"
                )
            expected_composition = hashlib.sha256(
                json.dumps(
                    [stage.to_dict() for stage in self.stages],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if self.composition_digest != expected_composition:
                raise AlgorithmConfigurationError(
                    "component Torch composition digest drifted"
                )
            if self.final_stage_id not in {stage.stage_id for stage in self.stages}:
                raise AlgorithmConfigurationError(
                    "component Torch final_stage_id is invalid"
                )
        else:
            if not self.stages and self.replicated_state is None:
                raise AlgorithmConfigurationError(
                    "sharded Torch evidence requires state payload"
                )
        if any(item.required and not item.present for item in roles):
            raise AlgorithmConfigurationError("required Torch role evidence is absent")
        for stage in self.stages:
            stage_workers = tuple(stage.workers)
            ranks = tuple(sorted(item.rank for item in stage_workers))
            if ranks != tuple(range(len(stage_workers))) or any(
                item.world_size != len(stage_workers) for item in stage_workers
            ):
                raise AlgorithmConfigurationError(
                    "Torch Stage worker evidence is incomplete"
                )
        reducer_fields = (
            self.reducer_id,
            self.reducer_api_version,
            self.reducer_schema_id,
            self.reducer_code_digest,
        )
        if any(value is not None for value in reducer_fields) and not all(
            value is not None for value in reducer_fields
        ):
            raise AlgorithmConfigurationError("Torch reducer evidence is incomplete")
        if self.reducer_api_version is not None and self.reducer_api_version != 1:
            raise AlgorithmConfigurationError(
                "Torch reducer API version must be exactly 1"
            )
        if self.reducer_branch is not None and (
            not isinstance(self.reducer_branch, str) or not self.reducer_branch
        ):
            raise AlgorithmConfigurationError("Torch reducer branch must be non-empty")
        evidence = dict(self.reducer_evidence)
        if any(not isinstance(name, str) or not name for name in evidence):
            raise AlgorithmConfigurationError(
                "Torch reducer evidence names are invalid"
            )
        if any(
            not isinstance(value, (str, int, float, bool, type(None)))
            for value in evidence.values()
        ):
            raise AlgorithmConfigurationError(
                "Torch reducer evidence must be JSON scalar"
            )
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in evidence.values()
        ):
            raise AlgorithmConfigurationError("Torch reducer evidence must be finite")
        object.__setattr__(self, "reducer_evidence", MappingProxyType(evidence))
        for name, value in (("reducer_code_digest", self.reducer_code_digest),):
            if value is not None and _DIGEST.fullmatch(value) is None:
                raise AlgorithmConfigurationError(f"Torch {name} is invalid")
        object.__setattr__(self, "workers", workers)
        object.__setattr__(self, "roles", roles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "run_config_name": self.run_config_name,
            "policy_digest": self.policy_digest,
            "parallelism_id": self.parallelism_id,
            "state_layout": self.state_layout,
            "workers": [item.to_dict() for item in self.workers],
            "roles": [item.to_dict() for item in self.roles],
            "replicated_state": self.replicated_state.to_dict()
            if self.replicated_state
            else None,
            "stages": [item.to_dict() for item in self.stages],
            "composition_digest": self.composition_digest,
            "final_stage_id": self.final_stage_id,
            "reducer_id": self.reducer_id,
            "reducer_api_version": self.reducer_api_version,
            "reducer_schema_id": self.reducer_schema_id,
            "reducer_code_digest": self.reducer_code_digest,
            "torch_runtime_api_version": self.torch_runtime_api_version,
            "reducer_branch": self.reducer_branch,
            "reducer_evidence": dict(self.reducer_evidence),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TorchExecutionEvidence":
        identity = TorchStageRunIdentity.from_dict(
            _mapping(value.get("identity", {}), "Torch evidence identity")
        )
        workers_value = value.get("workers", ())
        roles_value = value.get("roles", ())
        if not isinstance(workers_value, (list, tuple)) or not isinstance(
            roles_value, (list, tuple)
        ):
            raise AlgorithmConfigurationError(
                "Torch execution evidence worker/role fields are invalid"
            )
        workers = tuple(
            WorkerExecutionEvidence.from_dict(item)
            for item in workers_value
            if isinstance(item, Mapping)
        )
        if len(workers) != len(workers_value):
            raise AlgorithmConfigurationError(
                "Torch worker evidence entries are invalid"
            )
        roles = tuple(
            TorchRoleExecutionEvidence.from_dict(item)
            for item in roles_value
            if isinstance(item, Mapping)
        )
        if len(roles) != len(roles_value):
            raise AlgorithmConfigurationError("Torch role evidence entries are invalid")
        stages_value = value.get("stages", ())
        if not isinstance(stages_value, (list, tuple)):
            raise AlgorithmConfigurationError(
                "Torch Stage evidence entries are invalid"
            )
        replicated_value = value.get("replicated_state")
        replicated = (
            ReplicatedTorchStateEvidence.from_dict(replicated_value)
            if isinstance(replicated_value, Mapping)
            else None
        )
        return cls(
            identity=identity,
            run_config_name=cast(str, value["run_config_name"]),
            policy_digest=cast(str, value["policy_digest"]),
            parallelism_id=cast(str, value["parallelism_id"]),
            state_layout=cast(str, value["state_layout"]),
            workers=workers,
            roles=roles,
            replicated_state=replicated,
            stages=tuple(
                ComponentStageEvidence.from_dict(item) for item in stages_value
            ),
            composition_digest=cast(str | None, value.get("composition_digest")),
            final_stage_id=cast(str | None, value.get("final_stage_id")),
            reducer_id=cast(str | None, value.get("reducer_id")),
            reducer_api_version=cast(int | None, value.get("reducer_api_version")),
            reducer_schema_id=cast(str | None, value.get("reducer_schema_id")),
            reducer_code_digest=cast(str | None, value.get("reducer_code_digest")),
            torch_runtime_api_version=cast(
                int, value.get("torch_runtime_api_version", 1)
            ),
            reducer_branch=cast(str | None, value.get("reducer_branch")),
            reducer_evidence=cast(
                Mapping[str, object], value.get("reducer_evidence", {})
            ),
        )


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class ExecutionReceipt:
    """Immutable evidence used to classify a completed training execution."""

    run_id: str
    plan_id: str
    requested_algorithm: str
    canonical_algorithm: str
    profile: ExecutionProfile
    strategy: DistributionStrategy
    requested_worker_count: int
    distributed_min_workers: int
    requested_resources_per_worker: WorkerResources
    workers: tuple[WorkerExecutionEvidence, ...]
    input_complete: bool
    state: StateCoordinationEvidence
    result_policy: ResultPolicy = ResultPolicy.BUNDLE_REQUIRED
    driver_materialized_training_rows: int = 0
    artifact_ids: tuple[str, ...] = ()
    cluster_resources: Mapping[str, float] = field(default_factory=dict)
    runtime_owned: bool = False
    resource_preflight: str = "validated"
    api_version: int = 1
    torch_evidence: TorchExecutionEvidence | None = None

    def __post_init__(self) -> None:
        _non_empty(self.run_id, "run_id")
        _non_empty(self.requested_algorithm, "requested_algorithm")
        _non_empty(self.canonical_algorithm, "canonical_algorithm")
        if not isinstance(self.plan_id, str) or _DIGEST.fullmatch(self.plan_id) is None:
            raise AlgorithmConfigurationError(
                "plan_id must be a lower-case SHA-256 digest"
            )
        try:
            profile = ExecutionProfile(self.profile)
            strategy = DistributionStrategy(self.strategy)
            result_policy = ResultPolicy(self.result_policy)
        except (TypeError, ValueError) as exc:
            raise AlgorithmConfigurationError(
                f"invalid execution receipt enum value: {exc}"
            ) from exc
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "result_policy", result_policy)
        if strategy is DistributionStrategy.RAY_TRAIN_TORCH:
            if not isinstance(self.torch_evidence, TorchExecutionEvidence):
                raise AlgorithmConfigurationError(
                    "RAY_TRAIN_TORCH receipts require TorchExecutionEvidence"
                )
        elif self.torch_evidence is not None:
            raise AlgorithmConfigurationError(
                "non-Torch receipts must not contain TorchExecutionEvidence"
            )
        if (
            not isinstance(self.api_version, int)
            or isinstance(self.api_version, bool)
            or self.api_version != 1
        ):
            raise AlgorithmConfigurationError(
                f"unsupported ExecutionReceipt api_version: {self.api_version!r}"
            )
        if (
            not isinstance(self.requested_worker_count, int)
            or isinstance(self.requested_worker_count, bool)
            or self.requested_worker_count < 1
        ):
            raise AlgorithmConfigurationError(
                "requested_worker_count must be a positive integer"
            )
        if (
            not isinstance(self.distributed_min_workers, int)
            or isinstance(self.distributed_min_workers, bool)
            or self.distributed_min_workers < 2
        ):
            raise AlgorithmConfigurationError(
                "distributed_min_workers must be an integer of at least two"
            )
        if not isinstance(self.requested_resources_per_worker, WorkerResources):
            raise AlgorithmConfigurationError(
                "requested_resources_per_worker must be WorkerResources"
            )
        workers = tuple(self.workers)
        if not workers or any(
            not isinstance(item, WorkerExecutionEvidence) for item in workers
        ):
            raise AlgorithmConfigurationError(
                "execution receipt must contain WorkerExecutionEvidence"
            )
        if len(workers) != self.requested_worker_count:
            raise AlgorithmConfigurationError(
                "actual worker evidence count does not match the request"
            )
        expected_ranks = tuple(range(self.requested_worker_count))
        if tuple(sorted(item.rank for item in workers)) != expected_ranks:
            raise AlgorithmConfigurationError(
                "worker evidence must contain every rank exactly once"
            )
        if any(item.world_size != self.requested_worker_count for item in workers):
            raise AlgorithmConfigurationError(
                "worker world_size does not match requested_worker_count"
            )
        requested_resources = self.requested_resources_per_worker
        if any(
            worker.resources.num_cpus < requested_resources.num_cpus
            or worker.resources.num_gpus < requested_resources.num_gpus
            or (
                requested_resources.memory_bytes is not None
                and (
                    worker.resources.memory_bytes is None
                    or worker.resources.memory_bytes < requested_resources.memory_bytes
                )
            )
            or any(
                worker.resources.custom.get(name, 0.0) < amount
                for name, amount in requested_resources.custom.items()
            )
            for worker in workers
        ):
            raise AlgorithmConfigurationError(
                "worker evidence does not satisfy requested_resources_per_worker"
            )
        if len({item.worker_id for item in workers}) != len(workers):
            raise AlgorithmConfigurationError("worker IDs must be unique")
        if len({item.shard_id for item in workers}) != len(workers):
            raise AlgorithmConfigurationError("worker shard IDs must be unique")
        object.__setattr__(
            self, "workers", tuple(sorted(workers, key=lambda x: x.rank))
        )
        if not isinstance(self.input_complete, bool):
            raise AlgorithmConfigurationError("input_complete must be a boolean")
        if not isinstance(self.state, StateCoordinationEvidence):
            raise AlgorithmConfigurationError("state must be StateCoordinationEvidence")
        expected_coordination = FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS[
            strategy
        ].state_coordination
        if self.state.coordination is not expected_coordination:
            raise AlgorithmConfigurationError(
                "state coordination evidence does not match the declared strategy"
            )
        if (
            not isinstance(self.driver_materialized_training_rows, int)
            or isinstance(self.driver_materialized_training_rows, bool)
            or self.driver_materialized_training_rows < 0
        ):
            raise AlgorithmConfigurationError(
                "driver_materialized_training_rows must be non-negative"
            )
        artifacts = tuple(self.artifact_ids)
        if any(not isinstance(item, str) or not item for item in artifacts):
            raise AlgorithmConfigurationError(
                "artifact_ids must contain non-empty strings"
            )
        if len(set(artifacts)) != len(artifacts):
            raise AlgorithmConfigurationError("artifact_ids must be unique")
        object.__setattr__(self, "artifact_ids", artifacts)
        resources: dict[str, float] = {}
        for name, value in self.cluster_resources.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise AlgorithmConfigurationError(
                    "cluster_resources must map names to finite non-negative numbers"
                )
            resources[name] = float(value)
        object.__setattr__(self, "cluster_resources", FrozenDict(resources))
        if not isinstance(self.runtime_owned, bool):
            raise AlgorithmConfigurationError("runtime_owned must be a boolean")
        if self.resource_preflight not in {"validated", "deferred_to_ray"}:
            raise AlgorithmConfigurationError(
                "resource_preflight must be 'validated' or 'deferred_to_ray'"
            )
        if (
            self.profile is ExecutionProfile.LOCAL
            and self.resource_preflight != "validated"
        ):
            raise AlgorithmConfigurationError(
                "local execution requires validated resource preflight"
            )

    @property
    def node_count(self) -> int:
        """Return the number of distinct Ray nodes that executed workers."""
        return len({worker.node_id for worker in self.workers})

    @property
    def distributed(self) -> bool:
        """Return whether evidence proves one true multi-worker model."""
        return (
            len(self.workers) >= self.distributed_min_workers
            and self.input_complete
            and self.state.synchronized
            and self.state.bounded
            and self.state.global_model_digest is not None
            and self.driver_materialized_training_rows == 0
            and (self.result_policy is ResultPolicy.FIT_ONLY or bool(self.artifact_ids))
            and all(
                worker.rows_processed is not None and worker.rows_processed > 0
                for worker in self.workers
            )
            and self._strategy_state_proves_model()
        )

    def _strategy_state_proves_model(self) -> bool:
        """Validate the contribution relation for the selected strategy."""
        if self.strategy is DistributionStrategy.RAY_MAP_REDUCE:
            return True
        if self.strategy is DistributionStrategy.RAY_JOBLIB_ESTIMATOR:
            details = self.state.details
            return (
                details.get("execution_capability") == "estimator_internal_parallel"
                and isinstance(details.get("joblib_task_count"), int)
                and cast(int, details["joblib_task_count"]) >= len(self.workers)
                and details.get("observed_worker_count") == len(self.workers)
                and all(worker.model_state_digest is None for worker in self.workers)
            )
        if self.strategy is DistributionStrategy.RAY_PARALLEL_ENSEMBLE:
            details = self.state.details
            unit_count = details.get("unit_count")
            unit_digest = details.get("unit_ids_digest")
            return (
                details.get("execution_capability") == "single_model_distributed"
                and isinstance(unit_count, int)
                and not isinstance(unit_count, bool)
                and unit_count >= len(self.workers)
                and isinstance(unit_digest, str)
                and _DIGEST.fullmatch(unit_digest) is not None
                and all(
                    worker.model_state_digest is not None for worker in self.workers
                )
            )
        if self.strategy is DistributionStrategy.RAY_ITERATIVE_OPTIMIZATION:
            details = self.state.details
            rounds = details.get("rounds_completed")
            expected = details.get("expected_input_rows")
            observed = details.get("observed_input_rows")
            worker_state_digests = {
                worker.model_state_digest for worker in self.workers
            }
            return (
                details.get("execution_capability") == "single_model_distributed"
                and isinstance(rounds, int)
                and not isinstance(rounds, bool)
                and rounds >= 1
                and isinstance(expected, int)
                and not isinstance(expected, bool)
                and expected == observed
                and len(worker_state_digests) == 1
                and None not in worker_state_digests
            )
        return {worker.model_state_digest for worker in self.workers} == {
            self.state.global_model_digest
        } or self._staged_composite_matches_anchor()

    @property
    def execution_capability(self) -> str:
        """Classify observed execution without conflating trial parallelism."""
        if self.strategy is DistributionStrategy.RAY_JOBLIB_ESTIMATOR:
            return (
                "estimator_internal_parallel" if self.distributed else "single_worker"
            )
        return "single_model_distributed" if self.distributed else "single_worker"

    def _staged_composite_matches_anchor(self) -> bool:
        """Recompute staged state and bind receipt workers to its anchor model."""
        if self.strategy is not DistributionStrategy.FRAMEWORK_NATIVE:
            return False
        details = self.state.details
        stage_count = details.get("component_stage_count")
        stage_names = details.get("component_stages")
        anchor = details.get("anchor_stage")
        composition_digest = details.get("composition_digest")
        if (
            details.get("framework") != "staged_composite"
            or not isinstance(stage_count, int)
            or isinstance(stage_count, bool)
            or stage_count < 2
            or not isinstance(stage_names, str)
            or not isinstance(anchor, str)
            or not isinstance(composition_digest, str)
            or _DIGEST.fullmatch(composition_digest) is None
        ):
            return False
        stages = tuple(stage_names.split(","))
        if (
            len(stages) != stage_count
            or len(set(stages)) != stage_count
            or any(not stage for stage in stages)
            or anchor not in stages
        ):
            return False
        digest_payload: dict[str, object] = {
            "composition_digest": composition_digest,
            "stages": {},
        }
        stage_payload = cast(dict[str, object], digest_payload["stages"])
        for stage in stages:
            digest = details.get(f"stage.{stage}.digest")
            rows = details.get(f"stage.{stage}.rows")
            if (
                not isinstance(digest, str)
                or _DIGEST.fullmatch(digest) is None
                or not isinstance(rows, int)
                or isinstance(rows, bool)
                or rows < 1
            ):
                return False
            stage_payload[stage] = {"digest": digest, "rows": rows}
        anchor_digest = details.get(f"stage.{anchor}.digest")
        if {worker.model_state_digest for worker in self.workers} != {anchor_digest}:
            return False
        composite = hashlib.sha256(
            json.dumps(
                digest_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return self.state.global_model_digest == composite

    @property
    def cross_node(self) -> bool:
        """Return whether workers actually occupied multiple Ray nodes."""
        return self.node_count >= 2

    @property
    def cluster_distributed(self) -> bool:
        """Return whether an attached-cluster run proves cross-node training."""
        return (
            self.profile is ExecutionProfile.CLUSTER
            and self.distributed
            and self.cross_node
        )

    @property
    def kubernetes_distributed_supported(self) -> bool:
        """Return the deprecated KubeRay-specific evidence predicate."""
        import warnings

        warnings.warn(
            "kubernetes_distributed_supported is deprecated; use cluster_distributed",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.cluster_distributed

    def to_dict(self) -> dict[str, Any]:
        """Return portable receipt metadata."""
        payload = {
            "api_version": self.api_version,
            "run_id": self.run_id,
            "plan_id": self.plan_id,
            "requested_algorithm": self.requested_algorithm,
            "canonical_algorithm": self.canonical_algorithm,
            "execution_profile": self.profile.value,
            "strategy": self.strategy.value,
            "result_policy": self.result_policy.value,
            "requested_worker_count": self.requested_worker_count,
            "distributed_min_workers": self.distributed_min_workers,
            "requested_resources_per_worker": (
                self.requested_resources_per_worker.to_dict()
            ),
            "workers": [worker.to_dict() for worker in self.workers],
            "input_complete": self.input_complete,
            "state": {
                "coordination": self.state.coordination.value,
                "synchronized": self.state.synchronized,
                "bounded": self.state.bounded,
                "global_model_digest": self.state.global_model_digest,
                "details": dict(sorted(self.state.details.items())),
            },
            "driver_materialized_training_rows": self.driver_materialized_training_rows,
            "artifact_ids": list(self.artifact_ids),
            "cluster_resources": dict(sorted(self.cluster_resources.items())),
            "runtime_owned": self.runtime_owned,
            "resource_preflight": self.resource_preflight,
            "distributed": self.distributed,
            "cross_node": self.cross_node,
            "cluster_distributed": self.cluster_distributed,
            "execution_capability": self.execution_capability,
        }
        if self.torch_evidence is not None:
            payload["torch_evidence"] = self.torch_evidence.to_dict()
        return payload


__all__ = [
    "ComponentStageEvidence",
    "ExecutionReceipt",
    "ExecutionRequest",
    "ReplicatedTorchStateEvidence",
    "StateCoordinationEvidence",
    "TorchExecutionEvidence",
    "TorchRoleExecutionEvidence",
    "WorkerExecutionEvidence",
]
