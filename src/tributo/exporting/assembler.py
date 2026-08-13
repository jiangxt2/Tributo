"""Pure domain assembly for immutable bundle staging."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from tributo.explainability.contracts import ExplainabilityConfig
from tributo.exporting.manifest import (
    ExportManifest,
    ExportManifestV2,
    ManifestExecution,
    ManifestExecutionNode,
    ManifestSignature,
    ManifestSourceInfo,
    validate_explainability_descriptor,
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
        explainability: ExplainabilityConfig | None = None,
    ) -> StagedBundle:
        """Validate publication roles and assemble canonical manifest bytes."""
        if execution.status == "failed":
            raise ValueError("Cannot publish a failed execution")
        if execution.execution_id != execution_id:
            raise ValueError(
                f"Execution result ID {execution.execution_id!r} does not match "
                f"publication execution ID {execution_id!r}"
            )

        effective_roles = dict(roles if roles is not None else execution.roles)
        node_by_name = {nr.target_name: nr for nr in execution.node_results}

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
                staged_artifact = execution.staged_artifacts.get(node.node_id)
                if staged_artifact is None:
                    raise ValueError(
                        f"Publishable node {node.node_id!r} has no staged artifact"
                    )
                if staged_artifact.name != node.node_id:
                    raise ValueError(
                        f"Staged artifact name {staged_artifact.name!r} does not "
                        f"match node ID {node.node_id!r}"
                    )
                artifacts.append(staged_artifact)

        selected_backend: str | None = None
        selected_exactness: Literal["exact", "approximate", "conditional"] | None = None
        if explainability is not None and explainability.enabled:
            native = next(
                (
                    artifact
                    for artifact in artifacts
                    if artifact.flavor_id == "xgboost-native-v1"
                    and artifact.format in {"ubj", "xgboost-json"}
                ),
                None,
            )
            if explainability.backend in {"auto", "tree"} and native is not None:
                if "explainability_model" not in effective_roles:
                    effective_roles["explainability_model"] = native.name
                selected_backend = "tree"
                selected_exactness = "exact"
            elif explainability.backend == "tree":
                raise ValueError(
                    "Tree SHAP requires a publishable xgboost-native-v1 UBJ/JSON "
                    "artifact; enable the XGBoost companion target"
                )
            elif explainability.backend in {"auto", "model_agnostic"}:
                if not explainability.allow_approximate:
                    raise ValueError(
                        "Explainability requires explicit allow_approximate=true "
                        "when no exact XGBoost artifact is available"
                    )
                if explainability.reference is None:
                    raise ValueError(
                        "ONNX model-agnostic SHAP requires an explicit reference "
                        "binding"
                    )
                candidate_name = effective_roles.get(explainability.model_role)
                candidate = next(
                    (
                        artifact
                        for artifact in artifacts
                        if artifact.name == candidate_name
                    ),
                    None,
                )
                if candidate is None:
                    raise ValueError(
                        f"Explainability model role {explainability.model_role!r} "
                        "does not resolve to a published artifact"
                    )
                if not (
                    candidate.flavor_id == "onnx-runtime-v1"
                    and candidate.format == "onnx"
                ):
                    raise ValueError(
                        "Model-agnostic SHAP requires the selected model role to "
                        "reference an onnx-runtime-v1 ONNX artifact"
                    )
                effective_roles.setdefault("explainability_model", candidate.name)
                selected_backend = "model_agnostic"
                selected_exactness = "approximate"
            else:
                raise ValueError(
                    f"Explainability backend {explainability.backend!r} is not "
                    "available for Bundle publication"
                )

        for role_name, target_name in effective_roles.items():
            role_node = node_by_name.get(target_name)
            if role_node is None:
                raise ValueError(
                    f"Role {role_name!r} references unknown target {target_name!r}"
                )
            if not role_node.publish:
                raise ValueError(
                    f"Role {role_name!r} references non-publishable target "
                    f"{target_name!r} (implicit nodes cannot be roles)"
                )
            if role_node.status != "succeeded":
                raise ValueError(
                    f"Role {role_name!r} references target {target_name!r} "
                    f"with status {role_node.status!r}; roles must reference "
                    "succeeded artifacts"
                )

        required_explainability_artifacts: tuple[str, ...] = ()
        if explainability is not None and explainability.enabled:
            companion = effective_roles.get("explainability_model")
            if companion:
                required_explainability_artifacts = (companion,)
        explainability_roles = (
            ("explainability_model",)
            if explainability is not None
            and explainability.enabled
            and "explainability_model" in effective_roles
            else None
        )
        explainability_descriptor = (
            explainability.to_descriptor(
                required_artifacts=required_explainability_artifacts,
                model_roles=explainability_roles,
                backend=selected_backend,
                exactness=selected_exactness,
            )
            if explainability is not None and explainability.enabled
            else None
        )
        manifest_kwargs: dict[str, Any] = {
            "bundle_id": bundle_id,
            "status": "partial" if has_failed_optional else "succeeded",
            "canonical_uri": f"{bundle_uri.rstrip('/')}/{bundle_id}",
            "tributo_version": tributo_version,
            "source_info": source_info,
            "input_signature": input_signature or ManifestSignature(),
            "output_signature": output_signature or ManifestSignature(),
            "artifacts": tuple(artifacts),
            "roles": effective_roles,
            "execution": ManifestExecution(
                execution_id=execution_id,
                nodes=tuple(nodes),
            ),
        }
        if explainability_descriptor is not None:
            manifest_kwargs["schema_version"] = 2
            manifest_kwargs["explainability"] = explainability_descriptor
        else:
            manifest_kwargs["schema_version"] = 1
        manifest = (
            ExportManifestV2(**manifest_kwargs)
            if explainability_descriptor is not None
            else ExportManifest(**manifest_kwargs)
        )
        validate_explainability_descriptor(manifest)
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
