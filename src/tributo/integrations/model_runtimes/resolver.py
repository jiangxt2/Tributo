"""Bundle-backed model-reference resolution outside the inference core."""

from __future__ import annotations

import hashlib
from typing import ClassVar

from tributo.exceptions import JobConfigurationError, UnsupportedArtifactFormat
from tributo.exporting.bundle_reader import BundleReader
from tributo.exporting.capabilities import (
    CapabilityRegistry,
    get_default_capability_registry,
)
from tributo.exporting.manifest import ExportManifest
from tributo.exporting.models import BundleRef, LogicalArtifact
from tributo.inference.contracts import (
    ArtifactModelReference,
    BundleModelReference,
    ModelReference,
    RegistryModelReference,
    ResolvedModelBinding,
    ResolvedModelSelection,
)
from tributo.integrations.model_importers import (
    ModelImporterRegistry,
    build_default_model_importer_registry,
)
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class BundleModelReferenceResolver:
    """Normalize model references and pin one executable Bundle artifact."""

    resolver_id: ClassVar[str] = "tributo.bundle-model-reference-v1"

    def __init__(
        self,
        *,
        bundle_reader: BundleReader | None = None,
        importers: ModelImporterRegistry | None = None,
        capabilities: CapabilityRegistry | None = None,
    ) -> None:
        self._bundles = bundle_reader or BundleReader()
        self._importers = importers or build_default_model_importer_registry()
        self._capabilities = capabilities or get_default_capability_registry()

    def resolve(self, reference: ModelReference) -> ResolvedModelBinding:
        """Return an opaque model binding with verified tensor signatures."""
        model_reference, provenance = self._normalize(reference)
        manifest, manifest_bytes = self._bundles.read_manifest_with_bytes(
            model_reference.uri,
            storage_profile=model_reference.storage_profile,
        )
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if (
            model_reference.expected_manifest_sha256 is not None
            and manifest_sha256 != model_reference.expected_manifest_sha256
        ):
            raise JobConfigurationError(
                "Bundle manifest digest mismatch: expected "
                f"{model_reference.expected_manifest_sha256[:16]}..., got "
                f"{manifest_sha256[:16]}..."
            )

        artifact = _select_artifact(
            manifest,
            role=model_reference.role,
            unsafe=model_reference.unsafe,
            capabilities=self._capabilities,
        )
        if isinstance(reference, ArtifactModelReference):
            if artifact.flavor_id != reference.flavor_id:
                raise UnsupportedArtifactFormat(
                    f"Imported artifact flavor {artifact.flavor_id!r} does not "
                    f"match requested flavor {reference.flavor_id!r}"
                )

        bundle_ref = BundleRef(
            canonical_uri=manifest.canonical_uri,
            bundle_id=manifest.bundle_id,
            manifest_sha256=manifest_sha256,
        )
        source_provenance = (
            f"{provenance};source_kind={manifest.source_info.source_kind};"
            f"source_fingerprint={manifest.source_info.source_fingerprint}"
        )
        return ResolvedModelBinding(
            selection=ResolvedModelSelection(
                bundle_ref=bundle_ref,
                role=model_reference.role,
                flavor_id=artifact.flavor_id,
                storage_profile=model_reference.storage_profile,
                source_provenance=source_provenance,
                unsafe=model_reference.unsafe,
            ),
            input_signature=manifest.input_signature,
            output_signature=manifest.output_signature,
        )

    def _normalize(self, reference: ModelReference) -> tuple[BundleModelReference, str]:
        if isinstance(reference, BundleModelReference):
            return reference, "tributo-bundle"

        bundle_ref = self._importers.import_model(reference)
        if isinstance(reference, RegistryModelReference):
            selector = reference.version or reference.alias
            provenance = (
                f"registry:{reference.provider_id}:{reference.model_name}:{selector}"
            )
        elif isinstance(reference, ArtifactModelReference):
            provenance = (
                f"artifact:{reference.provider_id}:{reference.format_id}:"
                f"{reference.expected_sha256 or 'imported'}"
            )
        else:  # pragma: no cover - discriminated union makes this unreachable.
            raise AssertionError(type(reference).__name__)
        return (
            BundleModelReference.from_bundle_ref(
                bundle_ref,
                storage_profile=reference.import_storage_profile,
            ),
            provenance,
        )


def _select_artifact(
    manifest: ExportManifest,
    *,
    role: str,
    unsafe: bool,
    capabilities: CapabilityRegistry,
) -> LogicalArtifact:
    artifact_name = manifest.roles.get(role)
    if artifact_name is None:
        raise JobConfigurationError(
            f"Role {role!r} not found in bundle. Available roles: "
            f"{sorted(manifest.roles)}"
        )
    artifact = next(
        (item for item in manifest.artifacts if item.name == artifact_name), None
    )
    if artifact is None:
        raise JobConfigurationError(
            f"Role {role!r} references missing artifact {artifact_name!r}"
        )
    try:
        capability = capabilities.for_flavor(artifact.flavor_id)
    except KeyError:
        raise UnsupportedArtifactFormat(
            f"Flavor {artifact.flavor_id!r} is not in the capability matrix"
        ) from None
    if not capability.batch:
        raise UnsupportedArtifactFormat(
            f"Flavor {artifact.flavor_id!r} is readable but does not declare "
            "batch inference capability"
        )
    if artifact.artifact_kind != capability.artifact_kind:
        raise UnsupportedArtifactFormat(
            f"Artifact {artifact.name!r} has kind {artifact.artifact_kind!r} but "
            f"flavor {artifact.flavor_id!r} requires "
            f"{capability.artifact_kind!r}"
        )
    if artifact.format not in capability.format_ids:
        raise UnsupportedArtifactFormat(
            f"Artifact {artifact.name!r} format {artifact.format!r} is not "
            f"declared by flavor {artifact.flavor_id!r}"
        )
    if capability.signature_required and not unsafe:
        if not manifest.input_signature.input_fields:
            raise UnsupportedArtifactFormat(
                f"Artifact {artifact.name!r} requires a typed input signature"
            )
        if not manifest.output_signature.output_fields:
            raise UnsupportedArtifactFormat(
                f"Artifact {artifact.name!r} requires a typed output signature"
            )
    return artifact


__all__ = ["BundleModelReferenceResolver"]
