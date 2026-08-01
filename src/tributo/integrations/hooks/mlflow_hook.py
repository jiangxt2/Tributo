"""MLflow post-publish hook — log a published bundle into an MLflow run.

Invoked after a bundle has been published.  Per the export plan, the
first batch does NOT generate ``MLmodel`` files and does NOT create
MLflow Model Versions — it only records the bundle reference in a
tracking run via ``mlflow.log_artifacts(local_bundle_dir)`` (dict
fallback when no bundle-layout directory is available).  This keeps
the integration safe and reversible; Model Registry integration can
be layered on later.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from tributo.exporting.hooks import HookReceipt
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


# ── Options model ────────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class MLflowHookOptions(BaseModel):
    """Options for ``MLflowPostPublishHook``."""

    model_config = ConfigDict(extra="forbid")

    tracking_uri: str | None = Field(
        default=None,
        description="MLflow tracking URI. Defaults to MLFLOW_TRACKING_URI env var.",
    )
    run_id: str | None = Field(
        default=None,
        description=(
            "Explicit run to log into. When empty, a new run named "
            "'bundle-<bundle_id>' is created per publication."
        ),
    )
    tags: dict[str, str] = Field(default_factory=dict)
    required: bool = False


# ── Hook implementation ──────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class MLflowPostPublishHook:
    """Log a published bundle into an MLflow run.

    This hook is called after the bundle has been published.  It records
    the full local bundle directory (``mlflow.log_artifacts``) plus
    bundle tags into a tracking run — it never creates Registered
    Models or Model Versions (first-batch non-goal per the export plan).

    Failure is non-fatal by default (required=False).  Set required=True
    in options to fail the publication when logging fails.
    """

    hook_id: ClassVar[str] = "mlflow-log-artifacts-v1"

    def execute(
        self,
        canonical_uri: str,
        manifest: dict[str, Any],
        options: dict[str, Any] | None = None,
        local_bundle_dir: str | None = None,
    ) -> HookReceipt:
        """Log the bundle into MLflow.

        Args:
            canonical_uri: Bundle canonical URI.
            manifest: The committed manifest as a JSON dict.
            options: Hook-specific options (MLflowHookOptions fields).
            local_bundle_dir: Local bundle directory (valid only during
                the staging window).  When it holds a real bundle layout
                (``manifest.json`` present) the whole directory is
                recorded via ``mlflow.log_artifacts``; otherwise the
                manifest is logged as a single dict.

        Returns:
            HookReceipt indicating success/failure.
        """
        try:
            import mlflow
        except ImportError:
            return HookReceipt(
                hook_id=self.hook_id,
                status="skipped",
                error="mlflow not installed",
                retryable=False,
            )

        opts = options or {}
        tracking_uri = opts.get("tracking_uri")
        run_id = opts.get("run_id")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # ``_manifest_sha256`` is injected by PublicationRunner with the
        # Publisher's canonical digest; the fallback only covers direct
        # execute() calls that bypass the runner.
        manifest_sha = (
            manifest.get("_manifest_sha256")
            or hashlib.sha256(str(manifest).encode()).hexdigest()
        )
        bundle_id = manifest.get("bundle_id", "")
        tags = {
            "tributo_bundle_id": bundle_id,
            "tributo_manifest_sha256": manifest_sha,
            "tributo_source_kind": manifest.get("source_info", {}).get(
                "source_kind", ""
            ),
            **opts.get("tags", {}),
        }

        try:
            if run_id:
                with mlflow.start_run(run_id=run_id) as run:
                    self._log_bundle(
                        mlflow,
                        manifest,
                        manifest_sha,
                        bundle_id,
                        tags,
                        local_bundle_dir,
                    )
                    logged_run_id = run.info.run_id
            else:
                with mlflow.start_run(
                    run_name=f"bundle-{bundle_id}" if bundle_id else None
                ) as run:
                    self._log_bundle(
                        mlflow,
                        manifest,
                        manifest_sha,
                        bundle_id,
                        tags,
                        local_bundle_dir,
                    )
                    logged_run_id = run.info.run_id

            logger.info(
                "Logged bundle %s into MLflow run %s",
                bundle_id,
                logged_run_id,
            )
            return HookReceipt(
                hook_id=self.hook_id,
                status="success",
                idempotency_key=self.idempotency_key(
                    canonical_uri, manifest_sha, options
                ),
            )

        except Exception as exc:
            logger.error(
                "MLflow hook failed for bundle %r: %s",
                bundle_id,
                exc,
                exc_info=True,
            )
            return HookReceipt(
                hook_id=self.hook_id,
                status="failed",
                error=str(exc)[:4096],
                retryable=True,
                idempotency_key=self.idempotency_key(
                    canonical_uri, manifest_sha, options
                ),
            )

    @staticmethod
    def _log_bundle(
        mlflow: Any,
        manifest: dict[str, Any],
        manifest_sha: str,
        bundle_id: str,
        tags: dict[str, str],
        local_bundle_dir: str | None,
    ) -> None:
        """Record the bundle into the active run.

        Per the export plan the full local bundle directory is logged
        (``mlflow.log_artifacts``) so the layout is not flattened; the
        dict fallback only applies when no bundle-layout directory is
        available (S3-only publishes).
        """
        if local_bundle_dir and (Path(local_bundle_dir) / "manifest.json").is_file():
            mlflow.log_artifacts(local_bundle_dir, artifact_path="bundle")
        else:
            mlflow.log_dict(manifest, "bundle/manifest.json")
        mlflow.log_params(
            {
                "bundle_id": bundle_id,
                "manifest_sha256": manifest_sha,
            }
        )
        mlflow.set_tags(tags)

    def idempotency_key(
        self,
        canonical_uri: str,
        manifest_sha256: str,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Derive an idempotency key from hook_id + canonical_uri + manifest.

        Including ``hook_id`` prevents key collisions across different
        hooks operating on the same bundle.
        """
        payload = f"{self.hook_id}/{canonical_uri}/{manifest_sha256}"
        return hashlib.sha256(payload.encode()).hexdigest()
