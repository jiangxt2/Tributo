"""Tributo model export — stable public API.

``tributo.exporting`` is the user-facing entry point for model export.
It provides a minimal, stable surface::

    from tributo.exporting import export, load_bundle, ExportSpec, BundleRef

    spec = ExportSpec(
        bundle_uri="/tmp/my-model",
        targets=[ExportTarget(name="onnx-model", format="onnx")],
    )
    result = export(source, spec)
    bundle = load_bundle(result)

All concrete technology (XGBoost, PyTorch, ONNX Runtime, boto3, MLflow)
lives in ``tributo.integrations`` and is never imported by this package.
"""

from __future__ import annotations

from typing import Any

from tributo.training.exporters.models import (
    BundleOutputConfig,
    ExportTarget,
)
from tributo.training.exporters.repository import BundleRef
from tributo.util.annotations import PublicAPI

# ── Public types ───────────────────────────────────────────────────────────────


#: Stable alias for ``BundleOutputConfig``.
ExportSpec = BundleOutputConfig


@PublicAPI(stability="beta")
def export(
    source: Any,
    spec: ExportSpec,
    *,
    storage_profile: str | None = None,
) -> BundleRef:
    """Export a model source to one or more formats.

    Args:
        source: An ``ExportSource`` (from a ``SourceProvider``) or a raw
            model object that will be wrapped automatically.
        spec: An ``ExportSpec`` (alias for ``BundleOutputConfig``) defining
            targets, bundle URI, roles, and optional alias.
        storage_profile: Optional storage profile name for S3 credentials.

    Returns:
        ``BundleRef`` — an immutable reference to the committed bundle.
    """
    from tributo.training.exporters.service import BundleExportService

    service = BundleExportService()
    # If *source* is already an ExportSource, use it directly.
    from tributo.training.exporters.models import ExportSource

    if isinstance(source, ExportSource):
        export_source = source
        provider = None
    else:
        # Auto-wrap raw model objects.
        export_source = ExportSource(
            source_kind="raw",
            model_object=source,
        )
        provider = None

    result = service.export_bundle(
        source=export_source,
        config=spec,
        provider=provider,
    )
    return BundleRef(
        canonical_uri=result.canonical_uri,
        bundle_id=result.bundle_id,
        manifest_sha256=result.manifest_sha256,
    )


@PublicAPI(stability="beta")
def load_bundle(ref: BundleRef | str) -> dict[str, Any]:
    """Load and verify a bundle manifest.

    Args:
        ref: A ``BundleRef`` or a URI string (local path or ``s3://...``).

    Returns:
        The manifest as a JSON-serialisable dict.
    """
    from tributo.training.exporters.bundle_reader import BundleReader

    reader = BundleReader()
    if isinstance(ref, BundleRef):
        uri = ref.manifest_sha256  # Use manifest_sha256 to locate via alias
        # For local bundles, use canonical_uri directly.
        uri = ref.canonical_uri
    else:
        uri = ref

    raw = reader.read_manifest(uri)
    return raw.model_dump(mode="json")  # type: ignore[union-attr]


__all__ = [
    "BundleRef",
    "ExportSpec",
    "ExportTarget",
    "export",
    "load_bundle",
]
