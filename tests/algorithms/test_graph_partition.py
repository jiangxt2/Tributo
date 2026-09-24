"""Unit coverage for Core graph partition ownership and bounded sampling."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmInputError,
    GraphPartitionEvidence,
    GraphSamplingSpec,
)
from tributo.integrations.algorithm_runtimes.graph_partition import (
    RayGraphReadHandle,
    _balanced_row_split_indices,
    _GraphPartitionOwner,
    _raise_partition_error,
)


class _Iterator:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def iter_batches(self, *, batch_size: int, batch_format: str):
        del batch_format
        for start in range(0, len(self._rows), batch_size):
            yield {
                key: [row[key] for row in self._rows[start : start + batch_size]]
                for key in self._rows[0]
            }


class _RemoteMethod:
    def __init__(self, method: Callable[..., object]) -> None:
        self._method = method

    def remote(self, *args: object, **kwargs: object) -> object:
        return self._method(*args, **kwargs)


class _LocalActor:
    def __init__(self, actor: _GraphPartitionOwner) -> None:
        self._actor = actor

    def __getattr__(self, name: str) -> _RemoteMethod:
        return _RemoteMethod(getattr(self._actor, name))


class _LocalRay:
    @staticmethod
    def get(refs: list[object]) -> list[object]:
        return refs


def test_partitioned_graph_reader_fetches_cross_partition_neighbors(
    monkeypatch,
) -> None:
    """A worker request returns only seed and sampled context from all owners."""
    import ray

    monkeypatch.setattr(ray, "get", _LocalRay.get)
    nodes = (
        [
            {"node_id": node, "f0": float(node), "f1": float(node + 1)}
            for node in (0, 2, 4, 6)
        ],
        [
            {"node_id": node, "f0": float(node), "f1": float(node + 1)}
            for node in (1, 3, 5, 7)
        ],
    )
    edges = (
        [
            {"source": 7, "destination": 0, "relation": 1},
            {"source": 1, "destination": 0, "relation": 0},
            {"source": 2, "destination": 1, "relation": 1},
            {"source": 3, "destination": 1, "relation": 0},
        ],
        [
            {"source": 4, "destination": 5, "relation": 1},
            {"source": 6, "destination": 5, "relation": 0},
            {"source": 5, "destination": 6, "relation": 1},
            {"source": 0, "destination": 7, "relation": 0},
        ],
    )
    owners = (_GraphPartitionOwner(0), _GraphPartitionOwner(1))
    owner_rows = [
        owner.load(
            _Iterator(nodes[index]),
            _Iterator(edges[index]),
            node_id_column="node_id",
            node_feature_columns=("f0", "f1"),
            edge_source_column="source",
            edge_destination_column="destination",
            edge_type_column="relation",
            batch_size=2,
        )
        for index, owner in enumerate(owners)
    ]
    stats = GraphPartitionEvidence(
        graph_version="a" * 64,
        seed_role="train",
        partition_count=2,
        total_nodes=sum(item["node_rows"] for item in owner_rows),
        total_edges=sum(item["edge_rows"] for item in owner_rows),
        owner_node_rows=tuple(item["node_rows"] for item in owner_rows),
        owner_edge_rows=tuple(item["edge_rows"] for item in owner_rows),
    )
    reader = RayGraphReadHandle(tuple(_LocalActor(owner) for owner in owners), stats)

    batch = reader.sample(
        (0,),
        (1,),
        GraphSamplingSpec((2,), seed_batch_size=2, random_seed=3),
    )

    assert batch.seed_node_ids == (0,)
    assert set(batch.node_ids) == {0, 1, 7}
    global_edges = {
        (batch.node_ids[source], batch.node_ids[destination])
        for source, destination in batch.edge_index
    }
    assert global_edges == {(1, 0), (7, 0)}
    assert set(batch.edge_types) == {0, 1}
    assert batch.node_features[0] == (0.0, 1.0)
    evidence = reader.worker_evidence()
    assert evidence.seed_rows == 1
    assert evidence.sampled_node_rows == 3
    assert evidence.sampled_edge_rows == 2
    assert evidence.touched_partitions == (0, 1)
    assert len(evidence.sampling_profiles) == 1
    assert evidence.sampling_profiles[0].fanouts == (2,)
    assert evidence.sampling_profiles[0].direction == "incoming"
    assert evidence.sampling_profiles[0].request_count == 1
    assert evidence.sampling_profiles[0].random_seed_min == 3
    assert evidence.sampling_profiles[0].random_seed_max == 3
    assert len(evidence.sampling_profiles[0].random_seed_digest) == 64
    assert (
        reader.sample(
            (0,),
            (1,),
            GraphSamplingSpec((2,), seed_batch_size=2, random_seed=4),
        )
        == batch
    )
    sampling = reader.worker_evidence().sampling_profiles[0]
    assert sampling.random_seed_min == 3
    assert sampling.random_seed_max == 4
    assert sampling.request_count == 2


def test_multihop_sampling_expands_a_seed_again_when_it_is_also_context(
    monkeypatch,
) -> None:
    """A seed reached at an earlier hop still receives that hop's fanout."""
    import ray

    monkeypatch.setattr(ray, "get", _LocalRay.get)
    nodes = (
        [
            {"node_id": node, "f0": float(node), "f1": float(node + 1)}
            for node in (0, 2, 4)
        ],
        [
            {"node_id": node, "f0": float(node), "f1": float(node + 1)}
            for node in (1, 3)
        ],
    )
    edges = (
        [{"source": 2, "destination": 1}, {"source": 3, "destination": 2}],
        [{"source": 4, "destination": 2}],
    )
    owners = (_GraphPartitionOwner(0), _GraphPartitionOwner(1))
    rows = [
        owner.load(
            _Iterator(nodes[index]),
            _Iterator(edges[index]),
            node_id_column="node_id",
            node_feature_columns=("f0", "f1"),
            edge_source_column="source",
            edge_destination_column="destination",
            edge_type_column=None,
            batch_size=2,
        )
        for index, owner in enumerate(owners)
    ]
    stats = GraphPartitionEvidence(
        graph_version="b" * 64,
        seed_role="train",
        partition_count=2,
        total_nodes=sum(item["node_rows"] for item in rows),
        total_edges=sum(item["edge_rows"] for item in rows),
        owner_node_rows=tuple(item["node_rows"] for item in rows),
        owner_edge_rows=tuple(item["edge_rows"] for item in rows),
    )
    reader = RayGraphReadHandle(tuple(_LocalActor(owner) for owner in owners), stats)

    batch = reader.sample(
        (1, 2),
        (0, 1),
        GraphSamplingSpec((1, 2), seed_batch_size=2, random_seed=2),
    )

    assert set(batch.node_ids) == {1, 2, 3, 4}
    assert reader.worker_evidence().sampling_profiles[0].fanouts == (1, 2)


def test_graph_reader_rejects_duplicate_seeds_and_supports_empty_neighborhood(
    monkeypatch,
) -> None:
    """Repeated seeds fail and an isolated seed yields an empty edge list."""
    import ray

    monkeypatch.setattr(ray, "get", _LocalRay.get)
    owners = (_GraphPartitionOwner(0), _GraphPartitionOwner(1))
    nodes = (
        [{"node_id": 0, "f0": 1.0}],
        [{"node_id": 1, "f0": 2.0}],
    )
    edge_rows = ([], [])
    rows = [
        owner.load(
            _Iterator(nodes[index]),
            _Iterator(edge_rows[index]),
            node_id_column="node_id",
            node_feature_columns=("f0",),
            edge_source_column="source",
            edge_destination_column="destination",
            edge_type_column=None,
            batch_size=2,
        )
        for index, owner in enumerate(owners)
    ]
    stats = GraphPartitionEvidence(
        graph_version="c" * 64,
        seed_role="train",
        partition_count=2,
        total_nodes=2,
        total_edges=0,
        owner_node_rows=tuple(item["node_rows"] for item in rows),
        owner_edge_rows=tuple(item["edge_rows"] for item in rows),
    )
    reader = RayGraphReadHandle(tuple(_LocalActor(owner) for owner in owners), stats)

    with pytest.raises(AlgorithmInputError, match="unique"):
        reader.sample(
            (0, 0),
            (1, 1),
            GraphSamplingSpec((2,), seed_batch_size=2, random_seed=3),
        )

    batch = reader.sample(
        (0,),
        (1,),
        GraphSamplingSpec((2,), seed_batch_size=2, random_seed=3),
    )
    assert batch.seed_node_ids == (0,)
    assert batch.edge_index == ()
    assert batch.node_features == ((1.0,),)


def test_graph_reader_fails_when_a_sampled_edge_has_no_node_features(
    monkeypatch,
) -> None:
    """A sampled dangling endpoint must fail instead of truncating the batch."""
    import ray

    monkeypatch.setattr(ray, "get", _LocalRay.get)
    nodes = (
        [{"node_id": 0, "f0": 0.0}],
        [{"node_id": 1, "f0": 1.0}],
    )
    edges = (
        [{"source": 0, "destination": 99}],
        [{"source": 1, "destination": 0}],
    )
    owners = (_GraphPartitionOwner(0), _GraphPartitionOwner(1))
    rows = [
        owner.load(
            _Iterator(nodes[index]),
            _Iterator(edges[index]),
            node_id_column="node_id",
            node_feature_columns=("f0",),
            edge_source_column="source",
            edge_destination_column="destination",
            edge_type_column=None,
            batch_size=2,
        )
        for index, owner in enumerate(owners)
    ]
    stats = GraphPartitionEvidence(
        graph_version="d" * 64,
        seed_role="train",
        partition_count=2,
        total_nodes=2,
        total_edges=2,
        owner_node_rows=tuple(item["node_rows"] for item in rows),
        owner_edge_rows=tuple(item["edge_rows"] for item in rows),
    )
    reader = RayGraphReadHandle(tuple(_LocalActor(owner) for owner in owners), stats)

    with pytest.raises(AlgorithmInputError, match="without features"):
        reader.sample(
            (0,),
            (1,),
            GraphSamplingSpec(
                (2,),
                seed_batch_size=1,
                random_seed=3,
                direction="outgoing",
            ),
        )


def test_graph_reader_preserves_duplicate_edge_rows_as_parallel_edges(
    monkeypatch,
) -> None:
    """Duplicate input rows remain parallel edge entries in the sampled graph."""
    import ray

    monkeypatch.setattr(ray, "get", _LocalRay.get)
    nodes = (
        [
            {"node_id": 0, "f0": 0.0},
            {"node_id": 1, "f0": 1.0},
        ],
        [{"node_id": 2, "f0": 2.0}],
    )
    edges = (
        [
            {"source": 0, "destination": 1},
            {"source": 0, "destination": 1},
        ],
        [{"source": 0, "destination": 2}],
    )
    owners = (_GraphPartitionOwner(0), _GraphPartitionOwner(1))
    rows = [
        owner.load(
            _Iterator(nodes[index]),
            _Iterator(edges[index]),
            node_id_column="node_id",
            node_feature_columns=("f0",),
            edge_source_column="source",
            edge_destination_column="destination",
            edge_type_column=None,
            batch_size=2,
        )
        for index, owner in enumerate(owners)
    ]
    stats = GraphPartitionEvidence(
        graph_version="e" * 64,
        seed_role="train",
        partition_count=2,
        total_nodes=3,
        total_edges=3,
        owner_node_rows=tuple(item["node_rows"] for item in rows),
        owner_edge_rows=tuple(item["edge_rows"] for item in rows),
    )
    reader = RayGraphReadHandle(tuple(_LocalActor(owner) for owner in owners), stats)

    batch = reader.sample(
        (0,),
        (1,),
        GraphSamplingSpec(
            (3,),
            seed_batch_size=1,
            random_seed=3,
            direction="outgoing",
        ),
    )
    global_edges = [
        (batch.node_ids[source], batch.node_ids[destination])
        for source, destination in batch.edge_index
    ]
    assert global_edges.count((0, 1)) == 2
    assert global_edges.count((0, 2)) == 1


@pytest.mark.parametrize("feature", (True, "1.5"))
def test_graph_owner_rejects_non_numeric_node_features(feature: object) -> None:
    """Boolean and string columns must not be silently cast to graph features."""
    owner = _GraphPartitionOwner(0)

    with pytest.raises(ValueError, match="numeric"):
        owner.load(
            _Iterator([{"node_id": 0, "f0": feature}]),
            _Iterator([]),
            node_id_column="node_id",
            node_feature_columns=("f0",),
            edge_source_column="source",
            edge_destination_column="destination",
            edge_type_column=None,
            batch_size=1,
        )


def test_graph_owner_rejects_negative_relation_ids() -> None:
    """Relation IDs must satisfy the non-negative GraphBatch contract at load."""
    owner = _GraphPartitionOwner(0)

    with pytest.raises(ValueError, match="edge type must be non-negative"):
        owner.load(
            _Iterator([{"node_id": 0, "f0": 0.0}]),
            _Iterator([{"source": 0, "destination": 0, "relation": -1}]),
            node_id_column="node_id",
            node_feature_columns=("f0",),
            edge_source_column="source",
            edge_destination_column="destination",
            edge_type_column="relation",
            batch_size=1,
        )


@pytest.mark.parametrize(
    ("row_count", "partition_count"),
    ((0, 2), (2, 4), (8, 2), (11, 4)),
)
def test_balanced_row_split_indices_preserve_rows(
    row_count: int,
    partition_count: int,
) -> None:
    """Row boundaries retain every row and keep partition sizes balanced."""
    boundaries = _balanced_row_split_indices(row_count, partition_count)
    starts = (0, *boundaries)
    ends = (*boundaries, row_count)
    sizes = tuple(end - start for start, end in zip(starts, ends, strict=True))

    assert len(sizes) == partition_count
    assert sum(sizes) == row_count
    assert max(sizes) - min(sizes) <= 1


def test_balanced_row_split_indices_reject_invalid_settings() -> None:
    """The row splitter requires non-negative row counts and multiple owners."""
    with pytest.raises(AlgorithmConfigurationError):
        _balanced_row_split_indices(-1, 2)
    with pytest.raises(AlgorithmConfigurationError):
        _balanced_row_split_indices(4, 1)


def test_partition_failures_keep_input_and_execution_categories() -> None:
    """Remote row errors stay input failures; actor failures stay execution errors."""

    class _RemoteFailure(Exception):
        def __init__(self, cause: Exception) -> None:
            self.cause = cause

    with pytest.raises(AlgorithmInputError, match="ValueError"):
        _raise_partition_error(
            _RemoteFailure(ValueError("invalid node row")), "initialization"
        )
    with pytest.raises(AlgorithmExecutionError, match="RuntimeError"):
        _raise_partition_error(
            _RemoteFailure(RuntimeError("actor crashed")), "initialization"
        )
    with pytest.raises(AlgorithmExecutionError, match="ConnectionError"):
        _raise_partition_error(
            _RemoteFailure(ConnectionError("Ray actor transport failed")),
            "initialization",
        )
