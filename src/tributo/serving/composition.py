"""Composite model inference via :class:`ModelRunner` and :func:`depends`.

Allows chaining multiple models into a single inference pipeline where
the output of one runner feeds into the input of the next.

.. code-block:: python

    from tributo.serving.composition import ModelRunner, depends

    # Declare a 2-stage pipeline
    embedder = ModelRunner("embedding", load_fn=load_bge_model, predict_fn=bge_predict)
    classifier = ModelRunner(
        "classifier",
        load_fn=lambda: XGBoostFlavor.load("classifier.json"),
        predict_fn=lambda m, inp: m.predict(inp),
    )

    # classifier depends on embedder → embedder runs first
    classifier.depends(embedder)

    # Topo-sorted execution: embedder → classifier
    result = classifier.run(input_text="Hello world")
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from tributo._common.dag import topological_order
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)

# Type aliases for clarity
LoadFn = Callable[[], Any]
PredictFn = Callable[[Any, Any], Any]


@PublicAPI(stability="beta")
class ModelRunner:
    """A named model runner with optional upstream dependencies.

    Each runner has a unique name, a load function (called once on first
    use), and a predict function.  Upstream runners can be declared via
    :meth:`depends`, and :meth:`run` executes all dependencies in
    topological order before running this runner.

    Attributes:
        name: Unique identifier for this runner.
        load_fn: Zero-argument callable that returns the model object.
        predict_fn: Two-argument callable ``(model, input_data) → output``.
    """

    def __init__(
        self,
        name: str,
        load_fn: LoadFn,
        predict_fn: PredictFn,
    ) -> None:
        self.name = name
        self._load_fn = load_fn
        self._predict_fn = predict_fn
        self._upstream: list[ModelRunner] = []
        self._model: Any = None
        self._loaded = False

    # -- dependency graph ----------------------------------------------------

    def depends(self, *upstream: ModelRunner) -> ModelRunner:
        """Declare that this runner depends on *upstream* runners.

        Returns ``self`` for fluent chaining::

            final.depends(mid).depends(head)
        """
        for dep in upstream:
            if dep not in self._upstream:
                self._upstream.append(dep)
        return self

    @property
    def upstream(self) -> list[ModelRunner]:
        """Return direct upstream dependencies (read-only)."""
        return list(self._upstream)

    # -- execution -----------------------------------------------------------

    def run(self, input_data: Any = None) -> dict[str, Any]:
        """Execute the dependency graph and return per-runner results.

        Args:
            input_data: Initial input passed to the first (head) runner(s).
                If the first runner has upstream dependencies, the input
                is passed to the head runner(s) after their own upstreams
                complete.

        Returns:
            A dict mapping runner name to its output.
        """
        # Topological sort: find execution order so that every dependency
        # executes before its dependents.
        order = _topological_sort(self)

        results: dict[str, Any] = {}

        for runner in order:
            # Determine input: the result from the LAST upstream, or the
            # original input_data if there are no upstreams.
            if runner._upstream:
                # Pick the last upstream's output as input.
                inputs = [results[dep.name] for dep in runner._upstream]
                runner_input = inputs[-1] if len(inputs) == 1 else tuple(inputs)
            else:
                runner_input = input_data

            logger.info(
                "Running %r (input type=%s)", runner.name, type(runner_input).__name__
            )

            # Lazy-load model on first run
            if not runner._loaded:
                logger.debug("Loading model for %r", runner.name)
                runner._model = runner._load_fn()
                runner._loaded = True

            output = runner._predict_fn(runner._model, runner_input)
            results[runner.name] = output

        return results

    def __repr__(self) -> str:
        deps = ", ".join(d.name for d in self._upstream) if self._upstream else "none"
        return f"ModelRunner({self.name!r}, upstream=[{deps}])"


def depends(*upstream: ModelRunner) -> Callable[[ModelRunner], ModelRunner]:
    """Decorator / builder shorthand for declaring dependencies.

    Usage::

        @depends(embedder)
        def classifier_runner():
            return ModelRunner("classifier", ...)

        # Equivalent to: classifier_runner().depends(embedder)
    """

    def _decorator(runner_or_fn: Any) -> ModelRunner:
        if isinstance(runner_or_fn, ModelRunner):
            runner_or_fn.depends(*upstream)
            return runner_or_fn
        # Called as @depends(x) on a function that returns ModelRunner
        runner = runner_or_fn()  # type: ignore[no-any-return]
        runner.depends(*upstream)
        return runner

    return _decorator


# ── topological sort (delegates to shared DAG kernel) ──────────────────────


def _topological_sort(root: ModelRunner) -> list[ModelRunner]:
    """Return nodes in topological order via the shared DAG kernel."""
    # Collect all reachable nodes via BFS from *root*.
    all_nodes: dict[str, ModelRunner] = {}
    queue: list[ModelRunner] = [root]
    while queue:
        node = queue.pop()
        if node.name in all_nodes:
            continue
        all_nodes[node.name] = node
        queue.extend(node._upstream)

    # Build adjacency: {node_name: [upstream_node_names]}
    adjacency: dict[str, list[str]] = {
        name: [dep.name for dep in runner._upstream]
        for name, runner in all_nodes.items()
    }

    try:
        order = topological_order(adjacency)
    except ValueError as exc:
        raise ValueError(
            f"ModelRunner dependency cycle detected from root {root.name!r}: {exc}"
        ) from exc

    return [all_nodes[name] for name in order]
