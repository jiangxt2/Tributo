"""External model-importer protocol and exact-ID registry."""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable
from urllib.parse import urlsplit

from pydantic import BaseModel

from tributo.exporting.models import BundleRef
from tributo.inference.contracts import ArtifactModelReference, RegistryModelReference
from tributo.util.annotations import PublicAPI

ExternalModelReference = RegistryModelReference | ArtifactModelReference


@runtime_checkable
@PublicAPI(stability="alpha")
class ModelImporter(Protocol):
    """Normalize one explicit external model reference into a BundleRef."""

    api_version: ClassVar[int]
    provider_id: ClassVar[str]
    options_model: ClassVar[type[BaseModel]]
    reference_kinds: ClassVar[tuple[str, ...]]
    uri_schemes: ClassVar[tuple[str, ...]]
    credential_profile_types: ClassVar[tuple[str, ...]]
    capabilities: ClassVar[tuple[str, ...]]

    def import_model(self, reference: ExternalModelReference) -> BundleRef: ...


@PublicAPI(stability="alpha")
class ModelImporterRegistry:
    """Exact-ID registry; no priority, probing, or fallback routing."""

    def __init__(self) -> None:
        self._by_id: dict[str, type[ModelImporter]] = {}

    def register(self, importer: type[ModelImporter]) -> None:
        provider_id = importer.provider_id
        if not provider_id:
            raise ValueError("ModelImporter.provider_id must be non-empty")
        if importer.api_version != 1:
            raise ValueError(
                f"Unsupported ModelImporter api_version={importer.api_version}"
            )
        if provider_id in self._by_id:
            raise ValueError(f"ModelImporter {provider_id!r} is already registered")
        self._by_id[provider_id] = importer

    def get(self, provider_id: str) -> type[ModelImporter]:
        try:
            return self._by_id[provider_id]
        except KeyError as exc:
            raise ValueError(
                f"Unknown ModelImporter {provider_id!r}; "
                f"available: {sorted(self._by_id)}"
            ) from exc

    def import_model(self, reference: ExternalModelReference) -> BundleRef:
        """Validate capabilities, then invoke exactly one explicit importer."""
        importer_cls = self.get(reference.provider_id)
        if reference.kind not in importer_cls.reference_kinds:
            raise ValueError(
                f"ModelImporter {reference.provider_id!r} does not support "
                f"reference kind {reference.kind!r}"
            )
        if isinstance(reference, ArtifactModelReference):
            scheme = (urlsplit(reference.uri).scheme or "file").lower()
            if scheme not in importer_cls.uri_schemes:
                raise ValueError(
                    f"ModelImporter {reference.provider_id!r} does not support "
                    f"URI scheme {scheme!r}"
                )
        importer_cls.options_model.model_validate(reference.options)
        return importer_cls().import_model(reference)


@PublicAPI(stability="alpha")
def build_default_model_importer_registry() -> ModelImporterRegistry:
    """Build the first-party registry without importing optional SDKs."""
    from tributo.integrations.model_importers.artifact import ArtifactModelImporter
    from tributo.integrations.model_importers.mlflow import MLflowModelImporter

    registry = ModelImporterRegistry()
    registry.register(ArtifactModelImporter)
    registry.register(MLflowModelImporter)
    return registry


__all__ = [
    "ModelImporter",
    "ModelImporterRegistry",
    "build_default_model_importer_registry",
]
