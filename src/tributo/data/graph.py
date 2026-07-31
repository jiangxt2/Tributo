"""Graph data abstraction for GNN training.

``GraphSchema`` defines the correctness contract for a graph (node/edge
column names, directionality, size limits).  ``GraphDataBundle`` wraps
node features, edge lists, and labels as ``ray.data.Dataset`` instances
so that existing S3 / Parquet / Lance pipelines are reused without
introducing PyG or DGL into the framework core.

MVP scope: homogeneous static graphs with local full-graph materialisation.
``GraphSchema.max_nodes`` provides fail-fast protection — exceeding the
threshold raises a clear error rather than silently OOM-ing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tributo.util.annotations import PublicAPI

if TYPE_CHECKING:
    import ray.data


# ── GraphSchema ──────────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
@dataclass(frozen=True)
class GraphSchema:
    """Graph correctness contract — minimal metadata for MVP.

    All fields have sensible defaults for the common case of a directed
    homogeneous graph without self-loops or multi-edges.
    """

    node_id_field: str = "node_id"
    edge_src_field: str = "src_id"
    edge_dst_field: str = "dst_id"
    is_directed: bool = True
    allows_self_loops: bool = False
    allows_multi_edges: bool = False
    split_field: str | None = None
    max_nodes: int | None = None


# ── GraphDataBundle ──────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
@dataclass
class GraphDataBundle:
    """Graph data container backed by ``ray.data.Dataset``.

    Node features, edge index, edge features, and node labels are stored
    as separate ``ray.data.Dataset`` instances.  The graph topology is
    reconstructed inside each worker (PyG ``Data`` / DGL ``DGLGraph``)
    from the edge-index dataset.

    Attributes:
        node_features: Node feature table (Parquet / Lance).
        edge_index: Edge list ``(src_id, dst_id)``.
        edge_features: Optional edge feature table.
        node_labels: Optional node label table.
        schema: Graph correctness contract.
        graph_metadata: Statistical metadata (``num_nodes``, ``num_edges``).
    """

    node_features: ray.data.Dataset
    edge_index: ray.data.Dataset
    edge_features: ray.data.Dataset | None = None
    node_labels: ray.data.Dataset | None = None
    schema: GraphSchema = field(default_factory=GraphSchema)
    graph_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> GraphDataBundle:
        """Build a ``GraphDataBundle`` from a configuration dict.

        The config must provide ``node_features_path``, ``edge_index_path``,
        and optionally ``edge_features_path`` / ``node_labels_path``.
        Each path is read via the Parquet connector.

        Example::

            bundle = GraphDataBundle.from_config({
                "node_features_path": "s3://bucket/nodes.parquet",
                "edge_index_path": "s3://bucket/edges.parquet",
                "node_labels_path": "s3://bucket/labels.parquet",
                "schema": {"is_directed": True, "max_nodes": 100_000},
            })
        """
        from tributo.data.parquet import ParquetDataConnector

        connector = ParquetDataConnector()
        node_path: str = config["node_features_path"]
        edge_path: str = config["edge_index_path"]

        node_features = connector.read(path=node_path)
        edge_index = connector.read(path=edge_path)

        edge_features = None
        if "edge_features_path" in config:
            edge_features = connector.read(path=config["edge_features_path"])

        node_labels = None
        if "node_labels_path" in config:
            node_labels = connector.read(path=config["node_labels_path"])

        schema_dict = config.get("schema", {})
        schema = GraphSchema(**schema_dict)

        # Compute statistical metadata.
        metadata: dict[str, Any] = {}
        try:
            metadata["num_nodes"] = node_features.count()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to count node features from {node_path!r}: {exc}"
            ) from exc
        try:
            metadata["num_edges"] = edge_index.count()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to count edge index from {edge_path!r}: {exc}"
            ) from exc

        return cls(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            node_labels=node_labels,
            schema=schema,
            graph_metadata=metadata,
        )
