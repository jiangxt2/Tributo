"""Export manager — DAG execution, artifact materialization, and validation.

The manager executes a planned export DAG sequentially (phase 1 — no
parallelism), materialises ``ArtifactDraft`` into verified ``LogicalArtifact``
instances, and runs the validator chain on each succeeded node.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

from tributo.exceptions import JobExecutionError
from tributo.exporting.models import (
    ArtifactDraft,
    ArtifactFile,
    ArtifactRef,
    ExportContext,
    ExportExecutionResult,
    ExportSource,
    FailureInfo,
    LogicalArtifact,
    NodeResult,
    ResolvedArtifact,
    ValidationResult,
)
from tributo.exporting.planner import ExportPlan
from tributo.exporting.protocols import ModelExporter
from tributo.exporting.registries import (
    ExportRegistry,
    ValidatorRegistry,
)
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a single file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _materialize_artifact(
    draft: ArtifactDraft,
    artifact_dir: Path,
    node_id: str,
) -> LogicalArtifact:
    """Verify draft files exist, compute hashes/sizes, return LogicalArtifact.

    The manager does NOT trust exporter-reported hashes — it re-reads every
    file from disk.  Also validates that ``draft.name`` is safe for use in
    filesystem paths (no ``/`` or ``..`` components).
    """
    # Guard against hostile draft.name (path traversal).
    if "/" in draft.name or "\\" in draft.name or draft.name in (".", ".."):
        raise JobExecutionError(
            f"Exporter {draft.producer.exporter_id!r} returned unsafe "
            f"artifact name {draft.name!r}"
        )

    materialized: list[ArtifactFile] = []
    seen_paths: set[str] = set()

    for df in draft.files:
        fp = (artifact_dir / df.relative_path).resolve()
        if not fp.is_relative_to(artifact_dir.resolve()):
            raise JobExecutionError(
                f"Exporter {draft.producer.exporter_id!r} wrote file outside "
                f"artifact_dir: {df.relative_path!r}"
            )
        if not fp.is_file():
            raise JobExecutionError(
                f"Exporter {draft.producer.exporter_id!r} declared "
                f"{df.relative_path!r} but file does not exist"
            )
        if df.relative_path in seen_paths:
            raise JobExecutionError(
                f"Duplicate relative_path in draft: {df.relative_path!r}"
            )
        seen_paths.add(df.relative_path)

        file_hash = _sha256_file(fp)
        file_size = fp.stat().st_size
        materialized.append(
            ArtifactFile(
                relative_path=df.relative_path,
                sha256=file_hash,
                size_bytes=file_size,
                role=df.role,
            )
        )

    # Check for undeclared files.
    declared = {f.relative_path for f in draft.files}
    for fp in artifact_dir.rglob("*"):
        if fp.is_file():
            rel = str(fp.relative_to(artifact_dir))
            if rel not in declared:
                raise JobExecutionError(
                    f"Exporter wrote undeclared file: {rel!r} — "
                    "all output files must appear in ArtifactDraft.files"
                )

    tree_digest = LogicalArtifact.compute_tree_digest(tuple(materialized))
    return LogicalArtifact(
        name=draft.name,
        format=draft.format,
        flavor_id=draft.flavor_id,
        variant=draft.variant,
        files=tuple(materialized),
        entrypoint=draft.entrypoint,
        tree_digest=tree_digest,
        producer=draft.producer,
        derived_from=draft.derived_from,
    )


@PublicAPI(stability="beta")
class ExportManager:
    """Executes a planned export DAG sequentially.

    Lifecycle per node:

    1. Create isolated ``artifact_dir`` under ``staging_root``.
    2. Instantiate exporter, call ``export()``.
    3. Materialise ``ArtifactDraft`` → ``LogicalArtifact`` (re-hash all files).
    4. Run validator chain.
    5. Record ``NodeResult`` with state.

    Required-node failure cancels all remaining unstarted nodes.
    Optional-node failure records ``failed``; dependents become ``blocked``.
    """

    def __init__(
        self,
        export_registry: ExportRegistry,
        validator_registry: ValidatorRegistry,
    ) -> None:
        self._exports = export_registry
        self._validators = validator_registry

    def execute(
        self,
        plan: ExportPlan,
        source: ExportSource,
        staging_root: Path,
        execution_id: str,
    ) -> ExportExecutionResult:
        """Execute *plan* and return the execution result.

        Args:
            plan: Topologically ordered ``ExportPlan``.
            source: Resolved export source (shared across all nodes).
            staging_root: Root directory for per-node artifact isolation.
            execution_id: Stable execution identifier.

        Returns:
            ``ExportExecutionResult`` with per-node status.
        """
        staging_root.mkdir(parents=True, exist_ok=True)

        node_results: dict[str, NodeResult] = {}
        resolved_artifacts: dict[str, ResolvedArtifact] = {}
        staged_descriptors: dict[str, LogicalArtifact] = {}
        execution_started = False
        terminal = False

        for node in plan.nodes:
            node_id = node.target.name
            upstream_statuses = {
                dep: node_results[dep].status for dep in node.target.depends_on
            }

            # Check if blocked by upstream failure or skip.
            is_blocked = any(
                s in ("failed", "blocked", "cancelled")
                for s in upstream_statuses.values()
            )

            if is_blocked:
                node_results[node_id] = NodeResult(
                    node_id=node_id,
                    target_name=node.target.name,
                    status="blocked",
                    required=node.target.required,
                    publish=node.publish,
                    exporter_id=node.exporter_id,
                )
                # If a required node is blocked, the run is terminal.
                if node.target.required:
                    terminal = True
                continue

            # If a previous required node failed, cancel remaining.
            if terminal:
                node_results[node_id] = NodeResult(
                    node_id=node_id,
                    target_name=node.target.name,
                    status="cancelled",
                    required=node.target.required,
                    publish=node.publish,
                    exporter_id=node.exporter_id,
                )
                continue

            execution_started = True
            start_ms = int(time.time() * 1000)

            try:
                # -- Setup staging --
                artifact_dir = staging_root / "nodes" / node_id / "artifact"
                artifact_dir.mkdir(parents=True, exist_ok=True)
                context = ExportContext(
                    execution_id=execution_id,
                    node_id=node_id,
                    artifact_dir=artifact_dir,
                )

                # -- Instantiate exporter --
                exporter_cls = self._exports.get(node.exporter_id)
                exporter: ModelExporter = exporter_cls()  # type: ignore[call-arg]

                # -- Collect upstream resolved artifacts --
                upstream: dict[str, ResolvedArtifact] = {
                    dep: resolved_artifacts[dep] for dep in node.target.depends_on
                }

                # -- Run export --
                draft = exporter.export(context, source, upstream, node)

                # -- Materialise (re-hash) --
                artifact = _materialize_artifact(draft, artifact_dir, node_id)
                ra = ResolvedArtifact(artifact, artifact_dir)
                resolved_artifacts[node_id] = ra

                # -- Validator chain --
                validation_results: list[ValidationResult] = []
                for vb in node.validator_bindings:
                    try:
                        validator_cls = self._validators.get(vb.validator_id)
                        validator = validator_cls()  # type: ignore[call-arg]
                        opts = validator_cls.options_model(**vb.default_options)
                        vr = validator.validate(source, ra, upstream, opts)
                        validation_results.append(vr)

                        if vr.status == "failed" and vb.required:
                            raise JobExecutionError(
                                f"Required validator {vb.validator_id!r} failed: "
                                f"[{vr.failure.code if vr.failure else 'UNKNOWN'}] "
                                f"{vr.failure.message if vr.failure else 'no message'}"
                            )
                    except JobExecutionError:
                        raise
                    except Exception as exc:
                        logger.warning(
                            "Validator %s failed with exception: %s",
                            vb.validator_id,
                            exc,
                            exc_info=True,
                        )
                        if vb.required:
                            raise

                # Attach validation results to artifact.
                validated = LogicalArtifact(
                    name=artifact.name,
                    format=artifact.format,
                    flavor_id=artifact.flavor_id,
                    variant=artifact.variant,
                    files=artifact.files,
                    entrypoint=artifact.entrypoint,
                    tree_digest=artifact.tree_digest,
                    producer=artifact.producer,
                    derived_from=artifact.derived_from,
                    validation=tuple(validation_results),
                )
                # Replace ra with a new ResolvedArtifact holding the validated descriptor.
                ra = ResolvedArtifact(validated, artifact_dir)
                resolved_artifacts[node_id] = ra
                staged_descriptors[node_id] = validated

                duration_ms = int(time.time() * 1000) - start_ms
                node_results[node_id] = NodeResult(
                    node_id=node_id,
                    target_name=node.target.name,
                    status="succeeded",
                    required=node.target.required,
                    publish=node.publish,
                    exporter_id=node.exporter_id,
                    output_format=validated.format,
                    flavor_id=validated.flavor_id,
                    artifact_ref=ArtifactRef(
                        node_id=node_id,
                        artifact_name=validated.name,
                        tree_digest=validated.tree_digest,
                    ),
                    duration_ms=duration_ms,
                )

            except Exception as exc:
                duration_ms = int(time.time() * 1000) - start_ms
                is_required = node.target.required

                failure = FailureInfo(
                    code=type(exc).__name__,
                    category="export",
                    message=str(exc)[:4096],
                    retryable=False,
                )
                node_results[node_id] = NodeResult(
                    node_id=node_id,
                    target_name=node.target.name,
                    status="failed",
                    required=is_required,
                    publish=node.publish,
                    exporter_id=node.exporter_id,
                    failure=failure,
                    duration_ms=duration_ms,
                )

                if is_required:
                    terminal = True

        # Compute overall status.
        all_explicit = [
            nr
            for nr in node_results.values()
            if nr.node_id in {t.name for t in plan.explicit_targets}
        ]
        if not execution_started:
            overall = "failed"
        elif any(nr.status == "failed" and nr.required for nr in all_explicit):
            overall = "failed"
        elif any(nr.status == "failed" for nr in all_explicit):
            overall = "partial"
        else:
            overall = "succeeded"

        return ExportExecutionResult(
            execution_id=execution_id,
            status=overall,
            node_results=tuple(node_results.values()),
            staged_artifacts=staged_descriptors,
            roles={},  # Populated by Service after publish.
        )
