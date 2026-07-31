"""Export planner — candidate selection, DAG construction, and topological sort.

The planner transforms a ``BundleOutputConfig`` into an ``ExportPlan``: an
ordered list of ``PlannedTarget`` nodes with implicit intermediate nodes
injected and a topological execution order.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from tributo.exceptions import JobConfigurationError
from tributo.exporting.models import (
    BundleOutputConfig,
    ExportSource,
    ExportTarget,
    PlannedTarget,
    SupportRequest,
)
from tributo.exporting.registries import (
    ExportRegistry,
    ValidatorRegistry,
    select_candidate,
)
from tributo.util.annotations import PublicAPI


def _implicit_node_id(parent_name: str, fmt: str, opts_hash: str) -> str:
    """Generate a deterministic implicit node id."""
    return f"_implicit__{parent_name}__{fmt}__{opts_hash[:8]}"


def _options_hash(opts: dict[str, Any]) -> str:
    """Deterministic short hash of options dict for implicit node naming."""
    import hashlib
    import json

    payload = json.dumps(
        dict(sorted(opts.items())), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@PublicAPI(stability="beta")
class ExportPlan:
    """Topologically sorted plan ready for execution.

    Attributes:
        nodes: Ordered ``PlannedTarget`` list (explicit + implicit), ready
            for sequential execution.
        explicit_targets: The original user-supplied targets.
    """

    def __init__(
        self,
        nodes: list[PlannedTarget],
        explicit_targets: list[ExportTarget],
    ) -> None:
        self.nodes = nodes
        self.explicit_targets = explicit_targets

    @property
    def explicit_node_map(self) -> dict[str, PlannedTarget]:
        """Target name → PlannedTarget for explicit (non-implicit) nodes."""
        return {n.target.name: n for n in self.nodes if not n.implicit}

    def __repr__(self) -> str:
        names = [n.target.name for n in self.nodes]
        return f"ExportPlan(nodes={names})"


def _topological_sort(
    all_nodes: dict[str, PlannedTarget],
) -> list[PlannedTarget]:
    """Kahn's algorithm — returns topologically ordered nodes or raises on cycle."""
    in_degree: dict[str, int] = dict.fromkeys(all_nodes, 0)
    adj: dict[str, list[str]] = {name: [] for name in all_nodes}

    for name, pnode in all_nodes.items():
        for dep in pnode.target.depends_on:
            if dep not in all_nodes:
                raise JobConfigurationError(
                    f"Target {name!r} depends on unknown target {dep!r}"
                )
            adj[dep].append(name)
            in_degree[name] += 1

    queue: deque[str] = deque(n for n, d in in_degree.items() if d == 0)
    ordered: list[PlannedTarget] = []
    while queue:
        name = queue.popleft()
        ordered.append(all_nodes[name])
        for neighbour in adj[name]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    if len(ordered) != len(all_nodes):
        remaining = [n for n, d in in_degree.items() if d > 0]
        raise JobConfigurationError(
            f"Cycle detected in export DAG. Remaining nodes: {remaining}"
        )
    return ordered


@PublicAPI(stability="beta")
class ExportPlanner:
    """Plans the export DAG from user configuration.

    Responsibilities:
    - Match each target to an exporter via candidate selection.
    - Inject implicit intermediate nodes (e.g. FP32 ONNX before INT8).
    - Detect cycles and validate dependency references.
    - Produce a topologically ordered ``ExportPlan``.
    """

    def __init__(
        self,
        export_registry: ExportRegistry,
        validator_registry: ValidatorRegistry,
    ) -> None:
        self._exports = export_registry
        self._validators = validator_registry

    def plan(
        self,
        config: BundleOutputConfig,
        source: ExportSource,
    ) -> ExportPlan:
        """Plan the export DAG.

        Args:
            config: Validated bundle output configuration.
            source: The resolved export source snapshot.

        Returns:
            An ``ExportPlan`` ready for execution.

        Raises:
            JobConfigurationError: If any target cannot be matched to an
                exporter, if cycles are detected, or if dependencies are
                unresolvable.
        """
        if config.targets is None:
            raise JobConfigurationError(
                "Cannot plan export: targets is None (legacy mode)"
            )

        # Pre-compute format of every explicit target for upstream_formats.
        _explicit_formats: dict[str, str] = {t.name: t.format for t in config.targets}

        # Phase 1: Match each explicit target to a PlannedTarget.
        planned: dict[str, PlannedTarget] = {}
        for target in config.targets:
            # Populate upstream_formats from all explicit deps (not just
            # already-planned ones) so that artifact-to-artifact exporters
            # (e.g. quantizer) can check whether their upstream
            # requirements are satisfied regardless of declaration order.
            upstream_formats = tuple(
                _explicit_formats[d]
                for d in target.depends_on
                if d in _explicit_formats
            )
            request = SupportRequest(
                source_kind=source.source_kind,
                source_metadata=source.metadata,
                upstream_formats=upstream_formats,
            )
            candidates = self._exports.list_candidates(
                source_kind=source.source_kind,
                output_format=target.format,
            )
            if not candidates:
                raise JobConfigurationError(
                    f"No candidates for format={target.format!r} "
                    f"with source_kind={source.source_kind!r}"
                )
            pt = select_candidate(candidates, target, request, self._validators)
            planned[target.name] = pt

        # Phase 2: Inject implicit nodes from upstream_requirements.
        implicit_nodes: list[PlannedTarget] = []
        for target in config.targets:
            pt = planned[target.name]
            exporter_cls = self._exports.get(pt.exporter_id)
            requirements: tuple[Any, ...] = getattr(
                exporter_cls, "upstream_requirements", ()
            )

            for dep_name in target.depends_on:
                if dep_name in planned:
                    continue  # Explicit upstream exists.

                # Find the matching upstream requirement.
                req = next((r for r in requirements if r.name == dep_name), None)
                if req is None:
                    raise JobConfigurationError(
                        f"Target {target.name!r} depends on {dep_name!r}, "
                        f"but {dep_name!r} is not an explicit target and "
                        f"exporter {pt.exporter_id!r} does not declare an "
                        f"upstream_requirements entry for it. "
                        f"Declared requirements: "
                        f"{[r.name for r in requirements]}"
                    )

                # Build implicit node options by stripping requirement keys.
                implicit_opts: dict[str, Any] = {
                    k: v for k, v in target.options.items() if k not in req.options
                }
                oh = _options_hash(implicit_opts)
                implicit_name = _implicit_node_id(target.name, req.format, oh)

                # Check if already injected (shared implicit node).
                existing = [n for n in implicit_nodes if n.target.name == implicit_name]
                # Build from the CURRENT depends_on (may have been rewritten
                # by a previous iteration in this same Phase 2 loop).
                current_deps = planned[target.name].target.depends_on
                if existing:
                    # Rewrite the dependent's depends_on to point at the
                    # shared implicit node.
                    new_target = ExportTarget(
                        name=target.name,
                        format=target.format,
                        required=target.required,
                        depends_on=tuple(
                            implicit_name if d == dep_name else d for d in current_deps
                        ),
                        options=target.options,
                        validation=target.validation,
                    )
                    planned[target.name] = PlannedTarget(
                        target=new_target,
                        exporter_id=pt.exporter_id,
                        typed_options=pt.typed_options,
                        validator_bindings=pt.validator_bindings,
                        implicit=False,
                        publish=True,
                    )
                    continue

                # Create the implicit target.
                implicit_target = ExportTarget(
                    name=implicit_name,
                    format=req.format,
                    required=target.required,
                    options=implicit_opts,
                )
                implicit_request = SupportRequest(
                    source_kind=source.source_kind,
                    source_metadata=source.metadata,
                )
                implicit_candidates = self._exports.list_candidates(
                    source_kind=source.source_kind,
                    output_format=req.format,
                )
                if not implicit_candidates:
                    raise JobConfigurationError(
                        f"Cannot create implicit {req.format!r} node for "
                        f"{target.name!r}: no candidates for "
                        f"source_kind={source.source_kind!r}"
                    )
                implicit_pt = select_candidate(
                    implicit_candidates,
                    implicit_target,
                    implicit_request,
                    self._validators,
                )
                implicit_pt = PlannedTarget(
                    target=implicit_target,
                    exporter_id=implicit_pt.exporter_id,
                    typed_options=implicit_pt.typed_options,
                    validator_bindings=implicit_pt.validator_bindings,
                    implicit=True,
                    publish=False,
                )
                implicit_nodes.append(implicit_pt)

                # Rewrite the dependent's depends_on to point at the
                # implicit node name instead of the original dep name.
                # Use current_deps (may have been rewritten by a prior
                # dep in this same loop).
                new_deps = tuple(
                    implicit_name if d == dep_name else d for d in current_deps
                )
                new_target = ExportTarget(
                    name=target.name,
                    format=target.format,
                    required=target.required,
                    depends_on=new_deps,
                    options=target.options,
                    validation=target.validation,
                )
                planned[target.name] = PlannedTarget(
                    target=new_target,
                    exporter_id=pt.exporter_id,
                    typed_options=pt.typed_options,
                    validator_bindings=pt.validator_bindings,
                    implicit=False,
                    publish=True,
                )

        # Merge explicit + implicit.
        all_nodes: dict[str, PlannedTarget] = {**planned}
        for n in implicit_nodes:
            all_nodes[n.target.name] = n

        # Phase 3: Validate role → target references.
        for role_name, target_name in config.roles.items():
            if target_name not in all_nodes:
                raise JobConfigurationError(
                    f"Role {role_name!r} references target {target_name!r} "
                    f"which does not exist. Available targets: {sorted(all_nodes)}"
                )

        # Phase 4: Topological sort (Kahn's algorithm).
        ordered = _topological_sort(all_nodes)
        return ExportPlan(nodes=ordered, explicit_targets=config.targets)
