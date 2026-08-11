"""Bounded helpers that normalize acquired model files into Tributo Bundles."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tributo.exporting import _get_tributo_version
from tributo.exporting.bundle_reader import BundleReader
from tributo.exporting.manifest import ManifestSignature, ManifestSourceInfo
from tributo.exporting.models import (
    ArtifactFile,
    ArtifactRef,
    BundleRef,
    ExportExecutionResult,
    LogicalArtifact,
    NodeResult,
    ProducerInfo,
)
from tributo.exporting.publisher import Publisher


def publish_model_artifact(
    *,
    files: dict[str, Path],
    entrypoint: str,
    destination_uri: str,
    destination_storage_profile: str | None,
    artifact_name: str,
    format_id: str,
    flavor_id: str,
    variant: str | None,
    producer_id: str,
    source_info: ManifestSourceInfo,
    input_signature: ManifestSignature,
    output_signature: ManifestSignature,
) -> BundleRef:
    """Publish verified local files through the canonical Bundle Publisher."""
    if entrypoint not in files:
        raise ValueError(f"entrypoint {entrypoint!r} is not present in acquired files")

    descriptor_files = tuple(
        ArtifactFile(
            relative_path=relative_path,
            sha256=_sha256(path),
            size_bytes=path.stat().st_size,
            role="model" if relative_path == entrypoint else "aux",
        )
        for relative_path, path in sorted(files.items())
    )
    artifact = LogicalArtifact(
        name=artifact_name,
        format=format_id,
        flavor_id=flavor_id,
        variant=variant,
        files=descriptor_files,
        entrypoint=entrypoint,
        tree_digest=LogicalArtifact.compute_tree_digest(descriptor_files),
        producer=ProducerInfo(
            exporter_id=producer_id,
            effective_options={
                "format_id": format_id,
                "flavor_id": flavor_id,
                "variant": variant,
            },
        ),
    )
    identity_payload = {
        "tree_digest": artifact.tree_digest,
        "artifact_name": artifact_name,
        "format_id": format_id,
        "flavor_id": flavor_id,
        "variant": variant,
        "producer_id": producer_id,
        "source_info": source_info.model_dump(mode="json"),
        "input_signature": input_signature.model_dump(mode="json"),
        "output_signature": output_signature.model_dump(mode="json"),
    }
    identity_digest = _canonical_digest(identity_payload)
    bundle_id = f"bundle-import-{identity_digest[:32]}"
    execution_id = f"import-{identity_digest[:32]}"

    with tempfile.TemporaryDirectory(prefix="tributo-model-import-") as raw_staging:
        staging = Path(raw_staging)
        artifact_root = staging / "nodes" / artifact_name / "artifact"
        for relative_path, source in files.items():
            destination = artifact_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        artifact_ref = ArtifactRef(
            node_id=artifact_name,
            artifact_name=artifact_name,
            tree_digest=artifact.tree_digest,
        )
        execution = ExportExecutionResult(
            execution_id=execution_id,
            status="succeeded",
            node_results=(
                NodeResult(
                    node_id=artifact_name,
                    target_name=artifact_name,
                    status="succeeded",
                    required=True,
                    publish=True,
                    exporter_id=producer_id,
                    output_format=format_id,
                    flavor_id=flavor_id,
                    artifact_ref=artifact_ref,
                ),
            ),
            staged_artifacts={artifact_name: artifact},
            roles={"inference": artifact_name},
        )
        published = Publisher().publish(
            execution=execution,
            staging_root=staging,
            bundle_uri=destination_uri,
            bundle_id=bundle_id,
            execution_id=execution_id,
            tributo_version=_get_tributo_version(),
            source_info=source_info,
            input_signature=input_signature,
            output_signature=output_signature,
            storage_profile=destination_storage_profile,
            roles={"inference": artifact_name},
        )
    return BundleRef(
        canonical_uri=published.result.canonical_uri,
        bundle_id=published.result.bundle_id,
        manifest_sha256=published.result.manifest_sha256,
    )


def republish_verified_bundle(
    *,
    source_bundle_uri: str,
    destination_uri: str,
    destination_storage_profile: str | None,
    source_info: ManifestSourceInfo,
) -> BundleRef:
    """Verify every declared source artifact and republish it manifest-last."""
    reader = BundleReader()
    manifest, manifest_bytes = reader.read_manifest_with_bytes(source_bundle_uri)
    source_manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    identity_digest = _canonical_digest(
        {
            "source_manifest_sha256": source_manifest_digest,
            "source_info": source_info.model_dump(mode="json"),
        }
    )
    bundle_id = f"bundle-import-{identity_digest[:32]}"
    execution_id = f"import-{identity_digest[:32]}"

    node_results: list[NodeResult] = []
    staged_artifacts: dict[str, LogicalArtifact] = {}
    with tempfile.TemporaryDirectory(prefix="tributo-bundle-import-") as raw_staging:
        staging = Path(raw_staging)
        for artifact in manifest.artifacts:
            with reader.open_artifact(
                source_bundle_uri,
                artifact_name=artifact.name,
                manifest=manifest,
                manifest_bytes=manifest_bytes,
            ) as resolved:
                for file in artifact.files:
                    source = resolved.path_for(file.relative_path)
                    destination = (
                        staging
                        / "nodes"
                        / artifact.name
                        / "artifact"
                        / file.relative_path
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
            artifact_ref = ArtifactRef(
                node_id=artifact.name,
                artifact_name=artifact.name,
                tree_digest=artifact.tree_digest,
            )
            node_results.append(
                NodeResult(
                    node_id=artifact.name,
                    target_name=artifact.name,
                    status="succeeded",
                    required=True,
                    publish=True,
                    exporter_id="external-bundle-import-v1",
                    output_format=artifact.format,
                    flavor_id=artifact.flavor_id,
                    artifact_ref=artifact_ref,
                )
            )
            staged_artifacts[artifact.name] = artifact

        execution = ExportExecutionResult(
            execution_id=execution_id,
            status="succeeded",
            node_results=tuple(node_results),
            staged_artifacts=staged_artifacts,
            roles=manifest.roles,
        )
        published = Publisher().publish(
            execution=execution,
            staging_root=staging,
            bundle_uri=destination_uri,
            bundle_id=bundle_id,
            execution_id=execution_id,
            tributo_version=_get_tributo_version(),
            source_info=source_info,
            input_signature=manifest.input_signature,
            output_signature=manifest.output_signature,
            storage_profile=destination_storage_profile,
            roles=manifest.roles,
        )
    return BundleRef(
        canonical_uri=published.result.canonical_uri,
        bundle_id=published.result.bundle_id,
        manifest_sha256=published.result.manifest_sha256,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
