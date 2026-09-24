"""Framework-neutral contracts for partitioned node-level graph training."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Protocol, runtime_checkable

from tributo.algorithms.api.errors import (
    AlgorithmConfigurationError,
    AlgorithmInputError,
)
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class GraphInputSpec:
    """Name graph roles consumed by the Core partitioned graph reader."""

    node_role: str
    edge_role: str
    seed_role: str
    partition_count: int | None = None
    api_version: int = 1

    def __post_init__(self) -> None:
        roles = (self.node_role, self.edge_role, self.seed_role)
        if any(not isinstance(role, str) or not role for role in roles):
            raise AlgorithmConfigurationError("GraphInputSpec roles are required")
        if len(set(roles)) != len(roles):
            raise AlgorithmConfigurationError("GraphInputSpec roles must be distinct")
        if (
            not isinstance(self.api_version, int)
            or isinstance(self.api_version, bool)
            or self.api_version != 1
        ):
            raise AlgorithmConfigurationError("GraphInputSpec api_version must be 1")
        if self.partition_count is not None and (
            not isinstance(self.partition_count, int)
            or isinstance(self.partition_count, bool)
            or self.partition_count < 2
        ):
            raise AlgorithmConfigurationError(
                "GraphInputSpec partition_count must be at least two"
            )

    def to_dict(self) -> dict[str, object]:
        """Return the portable graph role declaration."""
        return {
            "node_role": self.node_role,
            "edge_role": self.edge_role,
            "seed_role": self.seed_role,
            "partition_count": self.partition_count,
            "api_version": self.api_version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> GraphInputSpec:
        """Reconstruct one strict role declaration."""
        node_role = value.get("node_role")
        edge_role = value.get("edge_role")
        seed_role = value.get("seed_role")
        partition_count = value.get("partition_count")
        api_version = value.get("api_version", 1)
        if (
            not isinstance(node_role, str)
            or not isinstance(edge_role, str)
            or not isinstance(seed_role, str)
            or (
                partition_count is not None
                and (
                    not isinstance(partition_count, int)
                    or isinstance(partition_count, bool)
                )
            )
            or not isinstance(api_version, int)
            or isinstance(api_version, bool)
        ):
            raise AlgorithmConfigurationError("invalid GraphInputSpec payload")
        return cls(
            node_role=node_role,
            edge_role=edge_role,
            seed_role=seed_role,
            partition_count=partition_count,
            api_version=api_version,
        )


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class GraphPartitionEvidence:
    """Driver-attested row counts for disjoint in-memory graph owners."""

    graph_version: str
    seed_role: str
    partition_count: int
    total_nodes: int
    total_edges: int
    owner_node_rows: tuple[int, ...]
    owner_edge_rows: tuple[int, ...]

    def __post_init__(self) -> None:
        if not _valid_digest(self.graph_version):
            raise AlgorithmInputError("GraphPartitionEvidence graph_version is invalid")
        if not isinstance(self.seed_role, str) or not self.seed_role:
            raise AlgorithmInputError("GraphPartitionEvidence seed_role is invalid")
        node_rows = tuple(self.owner_node_rows)
        edge_rows = tuple(self.owner_edge_rows)
        if (
            not isinstance(self.partition_count, int)
            or isinstance(self.partition_count, bool)
            or self.partition_count < 2
            or len(node_rows) != self.partition_count
            or len(edge_rows) != self.partition_count
        ):
            raise AlgorithmInputError("GraphPartitionEvidence partitions are invalid")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (*node_rows, *edge_rows)
        ):
            raise AlgorithmInputError("GraphPartitionEvidence owner rows are invalid")
        if (
            not isinstance(self.total_nodes, int)
            or isinstance(self.total_nodes, bool)
            or self.total_nodes < 1
            or not isinstance(self.total_edges, int)
            or isinstance(self.total_edges, bool)
            or self.total_edges < 0
            or sum(node_rows) != self.total_nodes
            or sum(edge_rows) != self.total_edges
        ):
            raise AlgorithmInputError("GraphPartitionEvidence row totals are invalid")
        if any(nodes >= self.total_nodes for nodes in node_rows) or (
            self.total_edges > 0
            and any(edges >= self.total_edges for edges in edge_rows)
        ):
            raise AlgorithmInputError(
                "one graph owner contains a complete node or edge table"
            )
        object.__setattr__(self, "owner_node_rows", node_rows)
        object.__setattr__(self, "owner_edge_rows", edge_rows)

    def to_dict(self) -> dict[str, object]:
        """Return portable partition ownership evidence."""
        return {
            "graph_version": self.graph_version,
            "seed_role": self.seed_role,
            "partition_count": self.partition_count,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "owner_node_rows": list(self.owner_node_rows),
            "owner_edge_rows": list(self.owner_edge_rows),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> GraphPartitionEvidence:
        """Reconstruct validated partition ownership evidence."""
        graph_version = value.get("graph_version")
        seed_role = value.get("seed_role")
        partition_count = value.get("partition_count")
        total_nodes = value.get("total_nodes")
        total_edges = value.get("total_edges")
        node_rows = value.get("owner_node_rows")
        edge_rows = value.get("owner_edge_rows")
        if (
            not isinstance(graph_version, str)
            or not isinstance(seed_role, str)
            or not isinstance(partition_count, int)
            or isinstance(partition_count, bool)
            or not isinstance(total_nodes, int)
            or isinstance(total_nodes, bool)
            or not isinstance(total_edges, int)
            or isinstance(total_edges, bool)
            or not isinstance(node_rows, (list, tuple))
            or not isinstance(edge_rows, (list, tuple))
            or any(
                not isinstance(row, int) or isinstance(row, bool) for row in node_rows
            )
            or any(
                not isinstance(row, int) or isinstance(row, bool) for row in edge_rows
            )
        ):
            raise AlgorithmInputError("invalid GraphPartitionEvidence payload")
        return cls(
            graph_version=graph_version,
            seed_role=seed_role,
            partition_count=partition_count,
            total_nodes=total_nodes,
            total_edges=total_edges,
            owner_node_rows=tuple(node_rows),
            owner_edge_rows=tuple(edge_rows),
        )


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class GraphSamplingEvidence:
    """Compact per-worker evidence for one distinct sampling profile."""

    fanouts: tuple[int, ...]
    seed_batch_size: int
    direction: str
    request_count: int
    random_seed_min: int
    random_seed_max: int
    random_seed_digest: str

    def __post_init__(self) -> None:
        fanouts = tuple(self.fanouts)
        if not fanouts or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in fanouts
        ):
            raise AlgorithmInputError("GraphSamplingEvidence fanouts are invalid")
        if (
            not isinstance(self.seed_batch_size, int)
            or isinstance(self.seed_batch_size, bool)
            or self.seed_batch_size < 1
            or not isinstance(self.request_count, int)
            or isinstance(self.request_count, bool)
            or self.request_count < 1
            or not isinstance(self.random_seed_min, int)
            or isinstance(self.random_seed_min, bool)
            or self.random_seed_min < 0
            or not isinstance(self.random_seed_max, int)
            or isinstance(self.random_seed_max, bool)
            or self.random_seed_max < self.random_seed_min
        ):
            raise AlgorithmInputError("GraphSamplingEvidence bounds are invalid")
        if self.direction not in {"incoming", "outgoing"}:
            raise AlgorithmInputError("GraphSamplingEvidence direction is invalid")
        if not _valid_digest(self.random_seed_digest):
            raise AlgorithmInputError("GraphSamplingEvidence seed digest is invalid")
        object.__setattr__(self, "fanouts", fanouts)

    def to_dict(self) -> dict[str, object]:
        """Return portable sampling profile evidence."""
        return {
            "fanouts": list(self.fanouts),
            "seed_batch_size": self.seed_batch_size,
            "direction": self.direction,
            "request_count": self.request_count,
            "random_seed_min": self.random_seed_min,
            "random_seed_max": self.random_seed_max,
            "random_seed_digest": self.random_seed_digest,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> GraphSamplingEvidence:
        """Reconstruct validated sampling profile evidence."""
        fanouts = value.get("fanouts")
        seed_batch_size = value.get("seed_batch_size")
        direction = value.get("direction")
        request_count = value.get("request_count")
        random_seed_min = value.get("random_seed_min")
        random_seed_max = value.get("random_seed_max")
        random_seed_digest = value.get("random_seed_digest")
        if (
            not isinstance(fanouts, (list, tuple))
            or any(
                not isinstance(item, int) or isinstance(item, bool) for item in fanouts
            )
            or not isinstance(seed_batch_size, int)
            or isinstance(seed_batch_size, bool)
            or not isinstance(direction, str)
            or not isinstance(request_count, int)
            or isinstance(request_count, bool)
            or not isinstance(random_seed_min, int)
            or isinstance(random_seed_min, bool)
            or not isinstance(random_seed_max, int)
            or isinstance(random_seed_max, bool)
            or not isinstance(random_seed_digest, str)
        ):
            raise AlgorithmInputError("invalid GraphSamplingEvidence payload")
        return cls(
            fanouts=tuple(fanouts),
            seed_batch_size=seed_batch_size,
            direction=direction,
            request_count=request_count,
            random_seed_min=random_seed_min,
            random_seed_max=random_seed_max,
            random_seed_digest=random_seed_digest,
        )


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class GraphWorkerEvidence:
    """Worker-local sampling counts produced by a Core graph reader."""

    graph_version: str
    partition_count: int
    seed_rows: int
    sampled_node_rows: int
    sampled_edge_rows: int
    batch_count: int
    touched_partitions: tuple[int, ...]
    sampling_profiles: tuple[GraphSamplingEvidence, ...] = ()

    def __post_init__(self) -> None:
        if not _valid_digest(self.graph_version):
            raise AlgorithmInputError("GraphWorkerEvidence graph_version is invalid")
        for name in (
            "partition_count",
            "seed_rows",
            "sampled_node_rows",
            "sampled_edge_rows",
            "batch_count",
        ):
            value = getattr(self, name)
            minimum = 2 if name == "partition_count" else 0
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise AlgorithmInputError(f"GraphWorkerEvidence {name} is invalid")
        if self.sampled_node_rows < self.seed_rows:
            raise AlgorithmInputError("GraphWorkerEvidence sampled nodes omit seeds")
        if self.seed_rows > 0 and self.batch_count < 1:
            raise AlgorithmInputError("GraphWorkerEvidence batches omit seed rows")
        touched = tuple(self.touched_partitions)
        profiles = tuple(self.sampling_profiles)
        if any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < self.partition_count
            for index in touched
        ) or len(set(touched)) != len(touched):
            raise AlgorithmInputError(
                "GraphWorkerEvidence touched partitions are invalid"
            )
        if any(not isinstance(profile, GraphSamplingEvidence) for profile in profiles):
            raise AlgorithmInputError(
                "GraphWorkerEvidence sampling profiles are invalid"
            )
        if sum(profile.request_count for profile in profiles) != self.batch_count or (
            self.seed_rows > 0 and not profiles
        ):
            raise AlgorithmInputError(
                "GraphWorkerEvidence sampling profiles do not match batch count"
            )
        if self.seed_rows > 0 and set(touched) != set(range(self.partition_count)):
            raise AlgorithmInputError(
                "GraphWorkerEvidence does not prove cross-partition access"
            )
        object.__setattr__(self, "touched_partitions", touched)
        object.__setattr__(self, "sampling_profiles", profiles)

    def to_dict(self) -> dict[str, object]:
        """Return portable worker sampling evidence."""
        return {
            "graph_version": self.graph_version,
            "partition_count": self.partition_count,
            "seed_rows": self.seed_rows,
            "sampled_node_rows": self.sampled_node_rows,
            "sampled_edge_rows": self.sampled_edge_rows,
            "batch_count": self.batch_count,
            "touched_partitions": list(self.touched_partitions),
            "sampling_profiles": [
                profile.to_dict() for profile in self.sampling_profiles
            ],
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> GraphWorkerEvidence:
        """Reconstruct validated worker sampling evidence."""
        graph_version = value.get("graph_version")
        partition_count = value.get("partition_count")
        seed_rows = value.get("seed_rows")
        sampled_node_rows = value.get("sampled_node_rows")
        sampled_edge_rows = value.get("sampled_edge_rows")
        batch_count = value.get("batch_count")
        touched = value.get("touched_partitions")
        profiles = value.get("sampling_profiles", ())
        if (
            not isinstance(graph_version, str)
            or not isinstance(partition_count, int)
            or isinstance(partition_count, bool)
            or not isinstance(seed_rows, int)
            or isinstance(seed_rows, bool)
            or not isinstance(sampled_node_rows, int)
            or isinstance(sampled_node_rows, bool)
            or not isinstance(sampled_edge_rows, int)
            or isinstance(sampled_edge_rows, bool)
            or not isinstance(batch_count, int)
            or isinstance(batch_count, bool)
            or not isinstance(touched, (list, tuple))
            or any(
                not isinstance(item, int) or isinstance(item, bool) for item in touched
            )
            or not isinstance(profiles, (list, tuple))
            or any(not isinstance(item, dict) for item in profiles)
        ):
            raise AlgorithmInputError("invalid GraphWorkerEvidence payload")
        return cls(
            graph_version=graph_version,
            partition_count=partition_count,
            seed_rows=seed_rows,
            sampled_node_rows=sampled_node_rows,
            sampled_edge_rows=sampled_edge_rows,
            batch_count=batch_count,
            touched_partitions=tuple(touched),
            sampling_profiles=tuple(
                GraphSamplingEvidence.from_dict(item) for item in profiles
            ),
        )


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class GraphSamplingSpec:
    """Bound one node-seeded neighborhood request."""

    fanouts: tuple[int, ...]
    seed_batch_size: int
    random_seed: int = 0
    direction: str = "incoming"

    def __post_init__(self) -> None:
        fanouts = tuple(self.fanouts)
        if not fanouts or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in fanouts
        ):
            raise AlgorithmConfigurationError(
                "GraphSamplingSpec fanouts must be positive integers"
            )
        if (
            not isinstance(self.seed_batch_size, int)
            or isinstance(self.seed_batch_size, bool)
            or self.seed_batch_size < 1
        ):
            raise AlgorithmConfigurationError(
                "GraphSamplingSpec seed_batch_size must be positive"
            )
        if (
            not isinstance(self.random_seed, int)
            or isinstance(self.random_seed, bool)
            or self.random_seed < 0
        ):
            raise AlgorithmConfigurationError(
                "GraphSamplingSpec random_seed must be non-negative"
            )
        if self.direction not in {"incoming", "outgoing"}:
            raise AlgorithmConfigurationError(
                "GraphSamplingSpec direction must be incoming or outgoing"
            )
        object.__setattr__(self, "fanouts", fanouts)


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class GraphBatch:
    """A sampled local graph with seed nodes in the leading rows."""

    node_ids: tuple[int, ...]
    edge_index: tuple[tuple[int, int], ...]
    node_features: tuple[tuple[float, ...], ...]
    seed_count: int
    seed_labels: tuple[int, ...]
    edge_types: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        node_ids = tuple(self.node_ids)
        edges = tuple(tuple(edge) for edge in self.edge_index)
        features = tuple(tuple(row) for row in self.node_features)
        labels = tuple(self.seed_labels)
        edge_types = tuple(self.edge_types)
        if not node_ids or any(
            not isinstance(node_id, int) or isinstance(node_id, bool)
            for node_id in node_ids
        ):
            raise AlgorithmInputError("GraphBatch node_ids must be non-empty integers")
        if len(set(node_ids)) != len(node_ids):
            raise AlgorithmInputError("GraphBatch node_ids must be unique")
        if (
            not isinstance(self.seed_count, int)
            or isinstance(self.seed_count, bool)
            or not 1 <= self.seed_count <= len(node_ids)
        ):
            raise AlgorithmInputError("GraphBatch seed_count is invalid")
        if len(labels) != self.seed_count or any(
            not isinstance(label, int) or isinstance(label, bool) or label < 0
            for label in labels
        ):
            raise AlgorithmInputError(
                "GraphBatch seed_labels must match seed_count and be non-negative integers"
            )
        if len(features) != len(node_ids) or not features:
            raise AlgorithmInputError("GraphBatch node_features do not match node_ids")
        width = len(features[0])
        if width < 1 or any(len(row) != width for row in features):
            raise AlgorithmInputError(
                "GraphBatch node feature rows must have one width"
            )
        for row in features:
            for value in row:
                if not isinstance(value, Real) or isinstance(value, bool):
                    raise AlgorithmInputError("GraphBatch features must be numeric")
                if not math.isfinite(float(value)):
                    raise AlgorithmInputError("GraphBatch features must be finite")
        if any(
            len(edge) != 2
            or any(
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < len(node_ids)
                for index in edge
            )
            for edge in edges
        ):
            raise AlgorithmInputError(
                "GraphBatch edge_index contains invalid local ids"
            )
        if edge_types and (
            len(edge_types) != len(edges)
            or any(
                not isinstance(edge_type, int)
                or isinstance(edge_type, bool)
                or edge_type < 0
                for edge_type in edge_types
            )
        ):
            raise AlgorithmInputError("GraphBatch edge_types do not match edge_index")
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "edge_index", edges)
        object.__setattr__(self, "node_features", features)
        object.__setattr__(self, "seed_labels", labels)
        object.__setattr__(self, "edge_types", edge_types)

    @property
    def seed_node_ids(self) -> tuple[int, ...]:
        """Return the original IDs for the leading seed rows."""
        return self.node_ids[: self.seed_count]


@PublicAPI(stability="alpha")
@runtime_checkable
class GraphReadHandle(Protocol):
    """Read bounded, sampled neighborhoods without exposing graph partitions."""

    def sample(
        self,
        seed_node_ids: Sequence[int],
        seed_labels: Sequence[int],
        spec: GraphSamplingSpec,
    ) -> GraphBatch:
        """Return one local graph batch for the given seed nodes."""

    def worker_evidence(self) -> GraphWorkerEvidence:
        """Return Core-generated sampling counters for this worker."""


def _valid_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "GraphBatch",
    "GraphInputSpec",
    "GraphPartitionEvidence",
    "GraphReadHandle",
    "GraphSamplingEvidence",
    "GraphSamplingSpec",
    "GraphWorkerEvidence",
]
