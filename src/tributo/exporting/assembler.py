"""Pure domain assembly for immutable bundle staging."""

from __future__ import annotations

import hashlib
from pathlib import Path

from tributo.exporting.manifest import (
    ExportManifest,
    ManifestExecution,
    ManifestExecutionNode,
    ManifestSignature,
    ManifestSourceInfo,
)
from tributo.exporting.models import ExportExecutionResult, LogicalArtifact
from tributo.exporting.planner import is_implicit_node_id
from tributo.exporting.repository import StagedBundle
from tributo.util.annotations import DeveloperAPI


@DeveloperAPI
class BundleAssembler:
    """Build a canonical manifest and storage-neutral staged bundle."""

    def assemble(
        self,
        *,
        execution: ExportExecutionResult,
        staging_root: Path,
        bundle_uri: str,
        bundle_id: str,
        execution_id: str,
        tributo_version: str,
        source_info: ManifestSourceInfo,
        input_signature: ManifestSignature | None = None,
        output_signature: ManifestSignature | None = None,
        roles: dict[str, str] | None = None,
    ) -> StagedBundle:
        """Validate publication roles and assemble canonical manifest bytes."""
        if execution.status == "failed":
            raise ValueError("Cannot publish a failed execution")
        if execution.execution_id != execution_id:
            raise ValueError(
                f"Execution result ID {execution.execution_id!r} does not match "
                f"publication execution ID {execution_id!r}"
            )

        effective_roles = roles if roles is not None else execution.roles
        node_by_name = {nr.target_name: nr for nr in execution.node_results}
        for role_name, target_name in effective_roles.items():
            node = node_by_name.get(target_name)
            if node is None:
                raise ValueError(
                    f"Role {role_name!r} references unknown target {target_name!r}"
                )
            if not node.publish:
                raise ValueError(
                    f"Role {role_name!r} references non-publishable target "
                    f"{target_name!r} (implicit nodes cannot be roles)"
                )
            if node.status != "succeeded":
                raise ValueError(
                    f"Role {role_name!r} references target {target_name!r} "
                    f"with status {node.status!r}; roles must reference "
                    "succeeded artifacts"
                )

        artifacts: list[LogicalArtifact] = []
        has_failed_optional = False
        nodes: list[ManifestExecutionNode] = []
        for node in execution.node_results:
            if node.status in ("failed", "blocked") and not node.required:
                has_failed_optional = True
            nodes.append(
                ManifestExecutionNode(
                    node_id=node.node_id,
                    target_name=node.target_name,
                    exporter_id=node.exporter_id,
                    status=node.status,
                    required=node.required,
                    implicit=is_implicit_node_id(node.node_id),
                    artifact_ref=node.artifact_ref,
                    failure=node.failure,
                    duration_ms=node.duration_ms,
                )
            )
            if (
                node.status == "succeeded"
                and node.publish
                and node.artifact_ref is not None
            ):
                descriptor = execution.staged_artifacts.get(node.node_id)
                if descriptor is None:
                    raise ValueError(
                        f"Publishable node {node.node_id!r} has no staged artifact"
                    )
                if descriptor.name != node.node_id:
                    raise ValueError(
                        f"Staged artifact name {descriptor.name!r} does not match "
                        f"node ID {node.node_id!r}"
                    )
                artifacts.append(descriptor)

        manifest = ExportManifest(
            schema_version=1,
            bundle_id=bundle_id,
            status="partial" if has_failed_optional else "succeeded",
            canonical_uri=f"{bundle_uri.rstrip('/')}/{bundle_id}",
            tributo_version=tributo_version,
            source_info=source_info,
            input_signature=input_signature or ManifestSignature(),
            output_signature=output_signature or ManifestSignature(),
            artifacts=tuple(artifacts),
            roles=effective_roles,
            execution=ManifestExecution(
                execution_id=execution_id,
                nodes=tuple(nodes),
            ),
        )
        manifest_bytes = manifest.canonical_json()
        return StagedBundle(
            bundle_id=bundle_id,
            execution_id=execution_id,
            manifest=manifest,
            manifest_bytes=manifest_bytes,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            staging_root=staging_root,
        )


__all__ = ["BundleAssembler"]
