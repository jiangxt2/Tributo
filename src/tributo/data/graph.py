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

from pydantic import TypeAdapter

from tributo.data.source_config import CanonicalSourceInput
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
        Each source is opened through the Ray Data ingestion Binding. Existing
        ``*_path`` fields remain Parquet shorthand; ``*_source`` accepts the
        canonical source model and therefore supports Lance and Iceberg too.

        Example::

            bundle = GraphDataBundle.from_config({
                "node_features_path": "s3://bucket/nodes.parquet",
                "edge_index_path": "s3://bucket/edges.parquet",
                "node_labels_path": "s3://bucket/labels.parquet",
                "schema": {"is_directed": True, "max_nodes": 100_000},
            })
        """
        node_path = str(config.get("node_features_path", "<node_features_source>"))
        edge_path = str(config.get("edge_index_path", "<edge_index_source>"))

        node_features = _read_graph_source(config, "node_features", node_path)
        edge_index = _read_graph_source(config, "edge_index", edge_path)

        edge_features = None
        if "edge_features_path" in config or "edge_features_source" in config:
            edge_features = _read_graph_source(
                config,
                "edge_features",
                str(config.get("edge_features_path", "<edge_features_source>")),
            )

        node_labels = None
        if "node_labels_path" in config or "node_labels_source" in config:
            node_labels = _read_graph_source(
                config,
                "node_labels",
                str(config.get("node_labels_path", "<node_labels_source>")),
            )

        schema_dict = config.get("schema", {})
        schema = GraphSchema(**schema_dict)

        # Compute only the small statistical metadata required by the graph
        # contract. Native Ray Data remains responsible for the count.
        metadata: dict[str, Any] = {
            "num_nodes": _count_graph_rows(node_features, "node features"),
            "num_edges": _count_graph_rows(edge_index, "edge index"),
        }
        if schema.max_nodes is not None and metadata["num_nodes"] > schema.max_nodes:
            raise ValueError(
                "Graph node count exceeds GraphSchema.max_nodes: "
                f"{metadata['num_nodes']} > {schema.max_nodes}"
            )

        return cls(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            node_labels=node_labels,
            schema=schema,
            graph_metadata=metadata,
        )


def _read_graph_source(
    config: dict[str, Any],
    name: str,
    parquet_path: str,
) -> "ray.data.Dataset":
    """Open one graph table through the explicit Ray ingestion boundary."""
    from tributo.data.ingestion import IngestionRequest, RayDataHandle, open_ingestion

    source = config.get(f"{name}_source")
    if source is None:
        if f"{name}_path" not in config:
            raise ValueError(f"Graph config requires {name}_source or {name}_path")
        source = {"type": "parquet", "path": parquet_path}
    canonical_source: CanonicalSourceInput = TypeAdapter(
        CanonicalSourceInput
    ).validate_python(source)
    result = open_ingestion(IngestionRequest(source=canonical_source, engine="ray"))
    try:
        if not isinstance(result.handle, RayDataHandle):
            raise RuntimeError("Graph ingestion requires a Ray Data handle")
        return result.handle.dataset
    finally:
        result.close()


def _count_graph_rows(dataset: "ray.data.Dataset", role: str) -> int:
    """Delegate graph row counting without exposing a native error payload."""
    failure_type: str | None = None
    try:
        count = int(dataset.count())
    except Exception as exc:
        failure_type = type(exc).__name__
    if failure_type is not None:
        # Raise outside the native handler so its message/cause cannot retain
        # a path, DSN, or credential supplied by an engine or Connector.
        raise RuntimeError(f"Failed to count {role} with {failure_type}")
    return count
