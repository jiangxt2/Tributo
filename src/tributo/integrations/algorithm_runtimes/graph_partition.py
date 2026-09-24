"""Ray-owned, partitioned graph reads for node-level Torch training."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, NoReturn, cast

from tributo.algorithms.api.errors import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmInputError,
)
from tributo.algorithms.api.graph import (
    GraphBatch,
    GraphPartitionEvidence,
    GraphReadHandle,
    GraphSamplingEvidence,
    GraphSamplingSpec,
    GraphWorkerEvidence,
)


class _GraphPartitionOwner:
    """Hold one disjoint set of node and edge rows in a Ray actor."""

    def __init__(self, partition_id: int) -> None:
        self.partition_id = partition_id
        self._features: dict[int, tuple[float, ...]] = {}
        self._outgoing: dict[int, list[tuple[int, int | None, tuple[int, int]]]] = (
            defaultdict(list)
        )
        self._incoming: dict[int, list[tuple[int, int | None, tuple[int, int]]]] = (
            defaultdict(list)
        )

    @staticmethod
    def _columns(batch: Any) -> dict[str, list[Any]]:
        if hasattr(batch, "to_pydict"):
            return cast(dict[str, list[Any]], batch.to_pydict())
        if hasattr(batch, "to_dict"):
            try:
                return cast(dict[str, list[Any]], batch.to_dict(orient="list"))
            except TypeError:
                value = batch.to_dict()
                if isinstance(value, dict):
                    return {str(key): list(items) for key, items in value.items()}
        if isinstance(batch, dict):
            return {str(key): list(items) for key, items in batch.items()}
        raise TypeError("graph data batch has an unsupported representation")

    def load(
        self,
        node_iterator: Any,
        edge_iterator: Any,
        *,
        node_id_column: str,
        node_feature_columns: tuple[str, ...],
        edge_source_column: str,
        edge_destination_column: str,
        edge_type_column: str | None,
        batch_size: int,
    ) -> dict[str, int]:
        """Consume this actor's disjoint Ray Data iterator shards."""
        import math

        for batch in node_iterator.iter_batches(
            batch_size=batch_size,
            batch_format="pyarrow",
        ):
            columns = self._columns(batch)
            names = (node_id_column, *node_feature_columns)
            missing = [name for name in names if name not in columns]
            if missing:
                raise ValueError(f"node batch is missing required columns: {missing}")
            for values in zip(*(columns[name] for name in names), strict=True):
                raw_id, *raw_features = values
                node_id = self._integer(raw_id, "node ID")
                if node_id in self._features:
                    raise ValueError("graph node ID is duplicated within a partition")
                if any(
                    not isinstance(value, Real) or isinstance(value, bool)
                    for value in raw_features
                ):
                    raise ValueError("graph node features must be numeric")
                features = tuple(float(value) for value in raw_features)
                if not features or any(not math.isfinite(value) for value in features):
                    raise ValueError("graph node features must be finite and non-empty")
                self._features[node_id] = features

        edge_columns = [edge_source_column, edge_destination_column]
        if edge_type_column is not None:
            edge_columns.append(edge_type_column)
        edge_row_index = 0
        for batch in edge_iterator.iter_batches(
            batch_size=batch_size,
            batch_format="pyarrow",
        ):
            columns = self._columns(batch)
            missing = [name for name in edge_columns if name not in columns]
            if missing:
                raise ValueError(f"edge batch is missing required columns: {missing}")
            for values in zip(*(columns[name] for name in edge_columns), strict=True):
                source = self._integer(values[0], "edge source")
                destination = self._integer(values[1], "edge destination")
                relation = (
                    self._integer(values[2], "edge type")
                    if edge_type_column is not None
                    else None
                )
                if relation is not None and relation < 0:
                    raise ValueError("graph edge type must be non-negative")
                edge_id = (self.partition_id, edge_row_index)
                edge_row_index += 1
                self._outgoing[source].append((destination, relation, edge_id))
                self._incoming[destination].append((source, relation, edge_id))
        return {
            "node_rows": len(self._features),
            "edge_rows": sum(len(edges) for edges in self._outgoing.values()),
        }

    @staticmethod
    def _integer(value: Any, name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"graph {name} must be an integer")
        return value

    def sample_neighbors(
        self,
        node_ids: tuple[int, ...],
        *,
        fanout: int,
        direction: str,
        random_seed: int,
    ) -> tuple[tuple[int, int, int | None, tuple[int, int]], ...]:
        """Return at most ``fanout`` candidates per query from this shard."""
        adjacency = self._incoming if direction == "incoming" else self._outgoing
        sampled: list[tuple[int, int, int | None, tuple[int, int]]] = []
        for node_id in node_ids:
            neighbors = adjacency.get(node_id, ())
            ranked = sorted(
                neighbors,
                key=lambda item: self._priority(
                    random_seed,
                    node_id,
                    item[0],
                    item[1],
                    item[2],
                ),
            )
            selected = tuple(ranked[:fanout])
            for neighbor_id, relation, edge_id in selected:
                source, destination = (
                    (neighbor_id, node_id)
                    if direction == "incoming"
                    else (node_id, neighbor_id)
                )
                sampled.append((source, destination, relation, edge_id))
        return tuple(sampled)

    @staticmethod
    def _priority(
        random_seed: int,
        node_id: int,
        neighbor_id: int,
        relation: int | None,
        edge_id: tuple[int, int],
    ) -> bytes:
        payload = f"{random_seed}:{node_id}:{neighbor_id}:{relation}".encode()
        return (
            hashlib.sha256(payload).digest()
            + edge_id[0].to_bytes(4, "big")
            + edge_id[1].to_bytes(8, "big")
        )

    def features(
        self, node_ids: tuple[int, ...]
    ) -> tuple[tuple[int, tuple[float, ...]], ...]:
        """Return only requested features owned by this partition."""
        return tuple(
            (node_id, self._features[node_id])
            for node_id in node_ids
            if node_id in self._features
        )


def _balanced_row_split_indices(
    row_count: int, partition_count: int
) -> tuple[int, ...]:
    """Return sorted cut points that retain every row across balanced splits."""
    if (
        not isinstance(row_count, int)
        or isinstance(row_count, bool)
        or row_count < 0
        or not isinstance(partition_count, int)
        or isinstance(partition_count, bool)
        or partition_count < 2
    ):
        raise AlgorithmConfigurationError("graph row split settings are invalid")
    return tuple(
        row_count * split_index // partition_count
        for split_index in range(1, partition_count)
    )


def _split_ray_dataset_by_rows(dataset: Any, partition_count: int) -> tuple[Any, ...]:
    """Materialize Ray Data blocks in the cluster and split at row boundaries."""
    materialized = dataset.materialize()
    row_count = materialized.count()
    if not isinstance(row_count, int) or isinstance(row_count, bool):
        raise AlgorithmInputError("Ray Data graph row count is invalid")
    if row_count == 1:
        raise AlgorithmInputError(
            "a one-row graph input cannot be distributed across graph owners"
        )
    indices = _balanced_row_split_indices(row_count, partition_count)
    return tuple(materialized.split_at_indices(list(indices)))


def _raise_partition_error(exc: Exception, operation: str) -> NoReturn:
    """Preserve input failures and classify Ray/runtime failures as execution."""
    cause = exc
    seen: set[int] = set()
    while id(cause) not in seen:
        seen.add(id(cause))
        nested = getattr(cause, "cause", None)
        if not isinstance(nested, Exception):
            break
        cause = nested
    if isinstance(cause, AlgorithmInputError):
        raise AlgorithmInputError(
            f"graph {operation} failed with {type(cause).__name__}"
        ) from None
    if isinstance(cause, AlgorithmConfigurationError):
        raise AlgorithmConfigurationError(
            f"graph {operation} failed with {type(cause).__name__}"
        ) from None
    if isinstance(cause, AlgorithmExecutionError):
        raise AlgorithmExecutionError(
            f"graph {operation} failed with {type(cause).__name__}"
        ) from None
    error_type = (
        AlgorithmInputError
        if isinstance(
            cause,
            (
                ValueError,
                TypeError,
                OverflowError,
                FileNotFoundError,
                IsADirectoryError,
                PermissionError,
            ),
        )
        else AlgorithmExecutionError
    )
    raise error_type(f"graph {operation} failed with {type(cause).__name__}") from None


class RayGraphReadHandle(GraphReadHandle):
    """Worker-local reader that queries disjoint graph owners on demand."""

    def __init__(
        self,
        owners: tuple[Any, ...],
        stats: GraphPartitionEvidence,
    ) -> None:
        self._owners = owners
        self.partition_stats = stats
        self._seed_rows = 0
        self._sampled_node_rows = 0
        self._sampled_edge_rows = 0
        self._batch_count = 0
        self._touched_partitions: set[int] = set()
        self._sampling_profiles: dict[
            tuple[tuple[int, ...], int, str], tuple[int, int, int, str]
        ] = {}

    def sample(
        self,
        seed_node_ids: Sequence[int],
        seed_labels: Sequence[int],
        spec: GraphSamplingSpec,
    ) -> GraphBatch:
        """Fetch bounded neighborhoods and features without gathering the graph."""
        import ray

        seeds = tuple(
            self._strict_integer(value, "seed node ID") for value in seed_node_ids
        )
        labels = tuple(
            self._strict_integer(value, "seed label", non_negative=True)
            for value in seed_labels
        )
        if not seeds or len(seeds) > spec.seed_batch_size or len(labels) != len(seeds):
            raise AlgorithmInputError(
                "graph seed batch does not match its sampling spec"
            )
        if len(set(seeds)) != len(seeds):
            raise AlgorithmInputError("graph seed IDs must be unique within a batch")

        node_order = list(seeds)
        known_nodes = set(seeds)
        frontier = seeds
        sampled_edges: list[tuple[int, int, int | None, tuple[int, int]]] = []
        sampled_edge_ids: set[tuple[int, int]] = set()
        for hop, fanout in enumerate(spec.fanouts):
            candidates = self._neighbors(
                ray,
                frontier,
                fanout=fanout,
                direction=spec.direction,
                random_seed=spec.random_seed + hop,
            )
            selected = self._select_fanout(candidates, fanout, spec, hop)
            next_frontier: list[int] = []
            next_frontier_seen: set[int] = set()
            for source, destination, _relation, edge_id in selected:
                if edge_id not in sampled_edge_ids:
                    sampled_edge_ids.add(edge_id)
                    sampled_edges.append((source, destination, _relation, edge_id))
                neighbor_id = source if spec.direction == "incoming" else destination
                if neighbor_id not in known_nodes:
                    known_nodes.add(neighbor_id)
                    node_order.append(neighbor_id)
                if neighbor_id not in next_frontier_seen:
                    next_frontier_seen.add(neighbor_id)
                    next_frontier.append(neighbor_id)
            frontier = tuple(next_frontier)
            if not frontier:
                break

        feature_rows = self._features(ray, tuple(node_order))
        local_ids = {node_id: index for index, node_id in enumerate(node_order)}
        edge_index = tuple(
            (local_ids[source], local_ids[destination])
            for source, destination, _relation, _edge_id in sampled_edges
        )
        edge_types = tuple(
            self._strict_integer(relation, "edge type", non_negative=True)
            for _source, _destination, relation, _edge_id in sampled_edges
            if relation is not None
        )
        if edge_types and len(edge_types) != len(sampled_edges):
            raise AlgorithmInputError("graph edge types are incomplete")
        batch = GraphBatch(
            node_ids=tuple(node_order),
            edge_index=edge_index,
            node_features=feature_rows,
            seed_count=len(seeds),
            seed_labels=labels,
            edge_types=edge_types,
        )
        self._seed_rows += len(seeds)
        self._sampled_node_rows += len(batch.node_ids)
        self._sampled_edge_rows += len(batch.edge_index)
        self._batch_count += 1
        self._touched_partitions.update(range(self.partition_stats.partition_count))
        profile_key = (spec.fanouts, spec.seed_batch_size, spec.direction)
        profile = self._sampling_profiles.get(profile_key)
        seed = spec.random_seed
        if profile is None:
            request_count = 0
            seed_min = seed
            seed_max = seed
            seed_digest = hashlib.sha256(b"").hexdigest()
        else:
            request_count, seed_min, seed_max, seed_digest = profile
            seed_min = min(seed_min, seed)
            seed_max = max(seed_max, seed)
        seed_digest = hashlib.sha256(
            f"{seed_digest}:{seed}".encode("ascii")
        ).hexdigest()
        self._sampling_profiles[profile_key] = (
            request_count + 1,
            seed_min,
            seed_max,
            seed_digest,
        )
        return batch

    @staticmethod
    def _strict_integer(
        value: Any,
        name: str,
        *,
        non_negative: bool = False,
    ) -> int:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or (non_negative and value < 0)
        ):
            requirement = "a non-negative integer" if non_negative else "an integer"
            raise AlgorithmInputError(f"graph {name} must be {requirement}")
        return value

    def _neighbors(
        self,
        ray: Any,
        node_ids: tuple[int, ...],
        *,
        fanout: int,
        direction: str,
        random_seed: int,
    ) -> tuple[tuple[int, int, int | None, tuple[int, int]], ...]:
        refs = [
            owner.sample_neighbors.remote(
                node_ids,
                fanout=fanout,
                direction=direction,
                random_seed=random_seed,
            )
            for owner in self._owners
        ]
        try:
            candidates = tuple(edge for rows in ray.get(refs) for edge in rows)
        except Exception as exc:
            raise AlgorithmExecutionError(
                f"graph neighbor lookup failed with {type(exc).__name__}"
            ) from None
        return candidates

    @staticmethod
    def _select_fanout(
        candidates: tuple[tuple[int, int, int | None, tuple[int, int]], ...],
        fanout: int,
        spec: GraphSamplingSpec,
        hop: int,
    ) -> tuple[tuple[int, int, int | None, tuple[int, int]], ...]:
        grouped: dict[int, list[tuple[int, int, int | None, tuple[int, int]]]] = (
            defaultdict(list)
        )
        for edge in candidates:
            query_node = edge[1] if spec.direction == "incoming" else edge[0]
            grouped[query_node].append(edge)
        selected: list[tuple[int, int, int | None, tuple[int, int]]] = []
        for query_node in sorted(grouped):
            neighbors = sorted(
                grouped[query_node],
                key=lambda edge: _GraphPartitionOwner._priority(
                    # Match the owner-side priority so global top-k is
                    # independent of which owner received an edge row.
                    spec.random_seed + hop,
                    query_node,
                    edge[0] if spec.direction == "incoming" else edge[1],
                    edge[2],
                    edge[3],
                ),
            )
            selected.extend(neighbors[:fanout])
        return tuple(selected)

    def _features(
        self, ray: Any, node_ids: tuple[int, ...]
    ) -> tuple[tuple[float, ...], ...]:
        refs = [owner.features.remote(node_ids) for owner in self._owners]
        try:
            rows = [row for partition in ray.get(refs) for row in partition]
        except Exception as exc:
            raise AlgorithmExecutionError(
                f"graph feature lookup failed with {type(exc).__name__}"
            ) from None
        by_id: dict[int, tuple[float, ...]] = {}
        for node_id, features in rows:
            if node_id in by_id:
                raise AlgorithmInputError(
                    "graph node IDs are duplicated across partitions"
                )
            by_id[node_id] = tuple(features)
        missing = [node_id for node_id in node_ids if node_id not in by_id]
        if missing:
            raise AlgorithmInputError("graph edges reference nodes without features")
        widths = {len(features) for features in by_id.values()}
        if len(widths) != 1:
            raise AlgorithmInputError("graph feature widths differ across partitions")
        return tuple(by_id[node_id] for node_id in node_ids)

    def worker_evidence(self) -> GraphWorkerEvidence:
        """Return sampling counters computed by this Core reader instance."""
        return GraphWorkerEvidence(
            graph_version=self.partition_stats.graph_version,
            partition_count=self.partition_stats.partition_count,
            seed_rows=self._seed_rows,
            sampled_node_rows=self._sampled_node_rows,
            sampled_edge_rows=self._sampled_edge_rows,
            batch_count=self._batch_count,
            touched_partitions=tuple(sorted(self._touched_partitions)),
            sampling_profiles=tuple(
                GraphSamplingEvidence(
                    fanouts=fanouts,
                    seed_batch_size=seed_batch_size,
                    direction=direction,
                    request_count=profile[0],
                    random_seed_min=profile[1],
                    random_seed_max=profile[2],
                    random_seed_digest=profile[3],
                )
                for (fanouts, seed_batch_size, direction), profile in sorted(
                    self._sampling_profiles.items()
                )
            ),
        )


@dataclass
class RayGraphPartitionLease:
    """Driver-owned actor lifetime plus a serializable worker reader."""

    reader: RayGraphReadHandle
    _owners: tuple[Any, ...] = field(repr=False)
    _closed: bool = False

    def close(self) -> None:
        """Stop only graph partition actors created by this lease."""
        if self._closed:
            return
        import ray

        for owner in self._owners:
            try:
                ray.kill(owner, no_restart=True)
            except Exception:
                continue
        self._closed = True


def open_ray_graph_partitions(
    *,
    nodes_dataset: Any,
    edges_dataset: Any,
    node_id_column: str,
    node_feature_columns: tuple[str, ...],
    edge_source_column: str,
    edge_destination_column: str,
    edge_type_column: str | None,
    graph_version: str,
    seed_role: str,
    partition_count: int,
    batch_size: int = 8192,
) -> RayGraphPartitionLease:
    """Build disjoint in-memory owner shards from Ray Data without driver collection."""
    import ray

    if partition_count < 2 or batch_size < 1:
        raise AlgorithmConfigurationError("graph partition settings are invalid")
    try:
        node_shards = _split_ray_dataset_by_rows(nodes_dataset, partition_count)
        edge_shards = _split_ray_dataset_by_rows(edges_dataset, partition_count)
    except Exception as exc:
        _raise_partition_error(exc, "data split")

    # These Core service actors are outside Torch worker CPU requests. Avoid
    # unbudgeted logical CPU reservations that can make a valid worker placement
    # unschedulable; the actors still share the cluster's physical CPU capacity.
    actor_type = cast(Any, ray.remote(num_cpus=0)(_GraphPartitionOwner))
    owners: tuple[Any, ...] = tuple(
        actor_type.remote(partition_id) for partition_id in range(partition_count)
    )
    try:
        refs = [
            owner.load.remote(
                node_shards[partition_id],
                edge_shards[partition_id],
                node_id_column=node_id_column,
                node_feature_columns=node_feature_columns,
                edge_source_column=edge_source_column,
                edge_destination_column=edge_destination_column,
                edge_type_column=edge_type_column,
                batch_size=batch_size,
            )
            for partition_id, owner in enumerate(owners)
        ]
        owner_stats = tuple(ray.get(refs))
        node_rows = tuple(item["node_rows"] for item in owner_stats)
        edge_rows = tuple(item["edge_rows"] for item in owner_stats)
        total_nodes = sum(node_rows)
        total_edges = sum(edge_rows)
        if any(rows >= total_nodes for rows in node_rows) or (
            total_edges > 0 and any(rows >= total_edges for rows in edge_rows)
        ):
            raise AlgorithmInputError(
                "Ray Data did not distribute the complete graph across owners "
                f"(node rows per owner={node_rows}, edge rows per owner={edge_rows})"
            )
        stats = GraphPartitionEvidence(
            graph_version=graph_version,
            seed_role=seed_role,
            partition_count=partition_count,
            total_nodes=total_nodes,
            total_edges=total_edges,
            owner_node_rows=node_rows,
            owner_edge_rows=edge_rows,
        )
    except Exception as exc:
        for owner in owners:
            try:
                ray.kill(owner, no_restart=True)
            except Exception:
                continue
        _raise_partition_error(exc, "partition initialization")
    return RayGraphPartitionLease(
        reader=RayGraphReadHandle(owners, stats),
        _owners=owners,
    )
