"""BaseGraphTrainer — GNN training base class.

Inherits from ``BaseTrainer`` to reuse ``registry`` / ``catalog`` /
``callback`` / ``TuneRunner`` infrastructure.  Uses
``DataLoadingMode.CANONICAL_TRAINER`` (worker-side data loading) because
GNN workers must build local sub-graph mini-batches — the graph topology
cannot be pre-partitioned on the driver.

Reference: PyG ``MessagePassing`` Template Method pattern
(``propagate → message → aggregate → update``).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from tributo.training.base import BaseTrainer
from tributo.util.annotations import PublicAPI

if TYPE_CHECKING:
    from tributo.data.graph import GraphDataBundle


@PublicAPI(stability="beta")
class BaseGraphTrainer(BaseTrainer):
    """GNN training base class.

    Subclasses must implement ``build_graph()`` and ``build_sampler()``.
    ``setup()`` has a default implementation that reads graph data from
    config and constructs the ``GraphDataBundle``.

    Uses ``DataLoadingMode.CANONICAL_TRAINER`` — each worker loads data
    locally, consistent with the PU trainer's approach.
    """

    @abstractmethod
    def build_graph(self, bundle: GraphDataBundle) -> Any:
        """Convert a ``GraphDataBundle`` into a framework-specific graph object.

        Args:
            bundle: The graph data bundle from ``setup()``.

        Returns:
            A PyG ``Data``, DGL ``DGLGraph``, or equivalent.
        """

    @abstractmethod
    def build_sampler(self, graph: Any) -> Any:
        """Build a mini-batch sampler for the given graph.

        Args:
            graph: The graph object returned by ``build_graph()``.

        Returns:
            A PyG ``NeighborLoader``, DGL ``DataLoader``, or equivalent.
        """

    def setup(self) -> None:
        """Default data-loading implementation.

        Reads ``graph_data`` from ``self.config``, constructs a
        ``GraphDataBundle`` via ``from_config()``, then calls
        ``build_graph()`` and ``build_sampler()``.

        Subclasses may override to customise data loading (e.g. to
        support heterogeneous graphs or custom preprocessing).
        """
        from tributo.data.graph import GraphDataBundle

        graph_config: dict[str, Any] = self.config.get("graph_data", {})
        if not graph_config:
            raise ValueError(
                f"{type(self).__name__}: config must contain a "
                f"'graph_data' section with node_features_path and "
                f"edge_index_path."
            )

        bundle = GraphDataBundle.from_config(graph_config)

        # Fail-fast: reject graphs exceeding the size limit declared in schema.
        max_nodes = bundle.schema.max_nodes
        if max_nodes is not None:
            num_nodes = bundle.graph_metadata.get("num_nodes", -1)
            if num_nodes > max_nodes:
                raise ValueError(
                    f"Graph has {num_nodes} nodes but schema.max_nodes={max_nodes}. "
                    f"Either increase max_nodes or use a smaller graph. "
                    f"Distributed graph sampling is not yet supported."
                )

        self._graph = self.build_graph(bundle)
        self._sampler = self.build_sampler(self._graph)
