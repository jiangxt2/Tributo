"""Shared DAG utilities — topological sort and cycle detection.

Used by ``Pipeline`` (training workflow orchestration) and ``ModelRunner``
(inference pipeline composition).  Both need Kahn's algorithm for
topological ordering with cycle detection; this module provides the
single canonical implementation.
"""

from __future__ import annotations

from collections import deque
from typing import Mapping, Sequence, TypeVar

from tributo.util.annotations import DeveloperAPI

T = TypeVar("T")


@DeveloperAPI
def topological_order(adjacency: Mapping[T, Sequence[T]]) -> list[T]:
    """Return nodes in topological order via Kahn's algorithm.

    Each key in *adjacency* is a node; its value is the sequence of
    nodes that must execute **before** it (upstream dependencies).

    Args:
        adjacency: ``{node: [upstream_node, ...]}`` — upstream nodes
            must execute before the key node.  Nodes that appear only
            as upstream values but not as keys are automatically
            included with an empty upstream list.

    Returns:
        Nodes in topological order (dependencies first).

    Raises:
        ValueError: If the graph contains a cycle.

    Example::

        >>> topological_order({"C": ["A", "B"], "B": ["A"], "A": []})
        ['A', 'B', 'C']
    """
    # Collect all nodes (keys + values that aren't keys).
    all_nodes: dict[T, int] = {}  # node → in-degree
    downstream: dict[T, list[T]] = {}  # node → nodes that depend on it

    for node, upstream in adjacency.items():
        if node not in all_nodes:
            all_nodes[node] = 0
            downstream.setdefault(node, [])
        for dep in upstream:
            if dep not in all_nodes:
                all_nodes[dep] = 0
                downstream.setdefault(dep, [])
            all_nodes[node] += 1
            downstream.setdefault(dep, [])
            downstream[dep].append(node)

    # Ensure nodes only appearing as upstream values are registered.
    for node in list(all_nodes):
        downstream.setdefault(node, [])

    # Kahn's algorithm: start with nodes of in-degree 0.
    ready: deque[T] = deque(n for n, d in all_nodes.items() if d == 0)
    order: list[T] = []

    while ready:
        node = ready.popleft()
        order.append(node)
        for dependent in downstream.get(node, []):
            all_nodes[dependent] -= 1
            if all_nodes[dependent] == 0:
                ready.append(dependent)

    if len(order) != len(all_nodes):
        remaining = {n for n, d in all_nodes.items() if d > 0}
        raise ValueError(
            f"DAG contains a cycle. Nodes still in queue: {list(remaining)}"
        )

    return order


@DeveloperAPI
def has_cycle(adjacency: Mapping[T, Sequence[T]]) -> bool:
    """Return ``True`` if the graph contains a cycle.

    A convenience wrapper around :func:`topological_order`.
    """
    try:
        topological_order(adjacency)
        return False
    except ValueError:
        return True
