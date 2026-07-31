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

from tributo.exporting.models import (
    BundleOutputConfig,
    ExportTarget,
)
from tributo.exporting.repository import BundleRef
from tributo.util.annotations import PublicAPI

# ── Public types ───────────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class ExportSpec(BundleOutputConfig):
    """Stable alias for ``BundleOutputConfig`` — the export configuration."""

    pass


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
    from tributo.exporting.models import ExportSource
    from tributo.exporting.service import BundleExportService

    if not isinstance(source, ExportSource):
        raise TypeError(
            f"source must be an ExportSource, got {type(source).__name__}. "
            "Use a SourceProvider to create one (e.g. RayXGBoostSourceProvider)."
        )

    service = BundleExportService()
    result = service.export_bundle(
        source=source,
        config=spec,
        tributo_version=_get_tributo_version(),
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
    from tributo.exporting.bundle_reader import BundleReader

    reader = BundleReader()
    if isinstance(ref, BundleRef):
        uri = ref.canonical_uri
        expected_sha256 = ref.manifest_sha256
    else:
        uri = ref
        expected_sha256 = None

    raw = reader.read_manifest(uri)
    manifest_dict = raw.model_dump(mode="json")  # type: ignore[union-attr]

    # Verify manifest integrity when a BundleRef was provided.
    if expected_sha256 is not None:
        import hashlib
        import json

        canonical = json.dumps(
            manifest_dict, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        actual_sha256 = hashlib.sha256(canonical).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Manifest integrity check failed: "
                f"expected sha256={expected_sha256[:16]}..., "
                f"got sha256={actual_sha256[:16]}..."
            )

    return manifest_dict


def _get_tributo_version() -> str:
    """Get the current tributo version string."""
    try:
        from importlib.metadata import version

        return version("tributo")
    except Exception:
        return "0.0.0"


__all__ = [
    "BundleRef",
    "ExportSpec",
    "ExportTarget",
    "export",
    "load_bundle",
    "generate_signing_key",
    "BundleSigner",
    "BundleVerifier",
    "ProvenanceBuilder",
    "AsyncBundleExportService",
]
