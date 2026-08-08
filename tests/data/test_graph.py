"""Graph input composition over the canonical ingestion Gateway."""

from __future__ import annotations

from typing import Any, cast

import pytest

from tributo.data.graph import GraphDataBundle
from tributo.data.ingestion import RayDataHandle


class _Dataset:
    def __init__(self, count: int, failure: Exception | None = None) -> None:
        self._count = count
        self._failure = failure
        self.limits: list[int] = []

    def limit(self, count: int) -> "_Dataset":
        self.limits.append(count)
        return _Dataset(min(self._count, count), self._failure)

    def count(self) -> int:
        if self._failure is not None:
            raise self._failure
        return self._count


class _Result:
    def __init__(self, dataset: _Dataset) -> None:
        self.handle = RayDataHandle(cast(Any, dataset))
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_graph_roles_compose_canonical_ingestion_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = {
        "/data/nodes.parquet": _Dataset(3),
        "s3://bucket/edges.lance": _Dataset(2),
    }
    requests: list[Any] = []
    results: list[_Result] = []

    def open_ingestion(request: Any) -> _Result:
        requests.append(request)
        source = request.source
        uri = getattr(source, "path", None) or getattr(source, "uri", None)
        result = _Result(datasets[uri])
        results.append(result)
        return result

    monkeypatch.setattr(
        "tributo.data.ingestion.open_ingestion",
        open_ingestion,
    )

    bundle = GraphDataBundle.from_config(
        {
            "node_features_path": "/data/nodes.parquet",
            "edge_index_source": {
                "provider": "tributo.lance",
                "uri": "s3://bucket/edges.lance",
            },
            "schema": {"max_nodes": 3},
        }
    )

    assert bundle.graph_metadata == {"num_nodes": 3, "num_edges": 2}
    assert [request.engine for request in requests] == [
        "tributo.ray_data",
        "tributo.ray_data",
    ]
    assert all(result.closed for result in results)


def test_graph_max_nodes_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tributo.data.graph._read_graph_source",
        lambda *args: _Dataset(2),
    )

    with pytest.raises(ValueError, match=r"2 > 1"):
        GraphDataBundle.from_config(
            {
                "node_features_path": "/data/nodes.parquet",
                "edge_index_path": "/data/edges.parquet",
                "schema": {"max_nodes": 1},
            }
        )


@pytest.mark.parametrize("max_nodes", ["10", 0, True])
def test_graph_max_nodes_requires_strict_positive_integer(
    max_nodes: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tributo.data.graph._read_graph_source",
        lambda *args: pytest.fail("invalid schema must fail before ingestion"),
    )

    with pytest.raises(ValueError, match="positive integer"):
        GraphDataBundle.from_config(
            {
                "node_features_path": "/data/nodes.parquet",
                "edge_index_path": "/data/edges.parquet",
                "schema": {"max_nodes": max_nodes},
            }
        )


def test_graph_count_error_does_not_retain_native_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tributo.data.graph._read_graph_source",
        lambda *args: _Dataset(0, RuntimeError("password=top-secret")),
    )

    with pytest.raises(RuntimeError, match="Failed to count node features") as exc_info:
        GraphDataBundle.from_config(
            {
                "node_features_path": "/data/nodes.parquet",
                "edge_index_path": "/data/edges.parquet",
            }
        )

    error = exc_info.value
    assert "top-secret" not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None


@pytest.mark.parametrize("name", ["node_features", "edge_index", "node_labels"])
def test_graph_null_path_is_rejected_before_file_open(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config: dict[str, Any] = {
        "node_features_path": "/data/nodes.parquet",
        "edge_index_path": "/data/edges.parquet",
        f"{name}_path": None,
    }
    monkeypatch.setattr(
        "tributo.data.graph._read_graph_source",
        lambda *args: pytest.fail("null path must fail before ingestion"),
    )

    with pytest.raises(ValueError, match=rf"non-empty {name}_source or {name}_path"):
        GraphDataBundle.from_config(config)
