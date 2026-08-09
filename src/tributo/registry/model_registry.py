"""MLflow Model Registry wrapper.

Provides model registration, version management, and stage transition capabilities.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from tributo.registry.schema import ModelVersion
from tributo.util.annotations import PublicAPI

try:
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException
except ImportError:
    MlflowClient = None  # type: ignore[assignment,misc]
    MlflowException = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# Regex for parsing runs:/<run_id>/<artifact_path> URIs
_RUNS_URI_RE = re.compile(r"^runs:/([^/]+)(?:/.*)?$")


@PublicAPI(stability="beta")
class ModelRegistry:
    """MLflow Model Registry wrapper.

    Provides model registration, version management, and stage transition capabilities.
    """

    def __init__(self, tracking_uri: str | None = None):
        """Initialize the model registry.

        Args:
            tracking_uri: MLflow tracking server URI.
        """
        if MlflowClient is None:
            raise ImportError(
                "mlflow is required for registry module. "
                "Install with: pip install tributo[registry]"
            )

        self._client = MlflowClient(tracking_uri=tracking_uri)

    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: dict[str, str] | None = None,
        description: str | None = None,
    ) -> ModelVersion:
        """Register a model to the Model Registry.

        Args:
            model_uri: Model URI (e.g., runs:/<run_id>/model).
            name: Registered model name.
            tags: Model tags.
            description: Model description.

        Returns:
            The registered model version info.
        """
        # Ensure the registered model exists. Different stores (HTTP/SQLite/file)
        # raise different exception subclasses; catch uniformly as MlflowException
        # and check error_code.
        model_existed = True
        try:
            self._client.get_registered_model(name)
        except MlflowException as e:
            if getattr(e, "error_code", None) == "RESOURCE_DOES_NOT_EXIST":
                self._client.create_registered_model(name)
                model_existed = False
            else:
                raise

        mv = None
        try:
            mv = self._client.create_model_version(name=name, source=model_uri)

            if tags:
                for key, value in tags.items():
                    self._client.set_model_version_tag(name, mv.version, key, value)

            if description:
                self._client.update_model_version(
                    name=name,
                    version=mv.version,
                    description=description,
                )
        except Exception:
            # Clean up on partial failure: delete version if created, delete model if newly created.
            self._cleanup_partial_registration(name, mv, model_existed)
            raise

        return ModelVersion(
            name=mv.name,
            version=int(mv.version),
            stage=mv.current_stage,
            run_id=mv.run_id,
            artifact_uri=mv.source,
            creation_timestamp=mv.creation_timestamp,
            description=description,
            tags=tags or {},
        )

    def _cleanup_partial_registration(
        self,
        name: str,
        mv: Any | None,
        model_existed: bool,
    ) -> None:
        """Clean up after a partial registration failure.

        Args:
            name: Model name.
            mv: Created ModelVersion (may be None).
            model_existed: Whether the model existed before registration.
        """
        if mv is not None:
            try:
                self._client.delete_model_version(name=name, version=mv.version)
            except Exception as e:
                logger.warning(
                    "Failed to clean up model version '%s' v%s: %s",
                    name,
                    mv.version,
                    e,
                )
        elif not model_existed:
            try:
                self._client.delete_registered_model(name=name)
            except Exception as e:
                logger.warning("Failed to clean up registered model '%s': %s", name, e)

    def get_model(
        self,
        name: str,
        version: int | None = None,
        stage: str | None = None,
    ) -> ModelVersion:
        """Get model version info.

        Args:
            name: Model name.
            version: Version number; mutually exclusive with stage.
            stage: Model stage; mutually exclusive with version.

        Returns:
            Model version info.

        Raises:
            ValueError: Neither version nor stage provided, or no model in the specified stage.
        """
        if version is not None:
            mv = self._client.get_model_version(name=name, version=str(version))
        elif stage is not None:
            # Allowlist validation on model name to defend against filter string injection.
            if not re.match(r"^[\w\-\./]+$", name):
                raise ValueError(
                    f"Invalid model name '{name}'. Only letters, digits, "
                    "underscores, hyphens, dots and slashes are allowed."
                )
            # Use search_model_versions instead of the deprecated get_latest_versions.
            # Escape single quotes in name to prevent filter string injection or parse failure.
            safe_name = name.replace("'", "''")
            results = self._client.search_model_versions(
                filter_string=f"name='{safe_name}'",
            )
            stage_versions = [mv for mv in results if mv.current_stage == stage]
            if not stage_versions:
                raise ValueError(f"No model '{name}' in stage '{stage}'")
            # Return the latest version
            mv = max(stage_versions, key=lambda m: int(m.version))
        else:
            raise ValueError("Either version or stage must be provided")

        return ModelVersion(
            name=mv.name,
            version=int(mv.version),
            stage=mv.current_stage,
            run_id=mv.run_id,
            artifact_uri=mv.source,
            creation_timestamp=mv.creation_timestamp,
            description=mv.description or None,
            tags=dict(mv.tags) if mv.tags else {},
        )

    def list_models(self) -> list[str]:
        """List all registered model names.

        Prefers ``search_registered_models`` (returns only model metadata, lightweight).
        Falls back to ``search_model_versions`` when the current MLflow server does
        not support the API (e.g., client/server version incompatibility resulting in
        empty results or errors).

        Returns:
            List of model names.
        """
        try:
            registered_models = self._client.search_registered_models()
            names = sorted(model.name for model in registered_models)
            if names:
                return names
        except MlflowException:
            logger.warning(
                "search_registered_models failed, falling back to search_model_versions"
            )

        # Fallback: extract unique model names from model versions
        model_versions = self._client.search_model_versions()
        name_set: set[str] = set()
        for mv in model_versions:
            name_set.add(mv.name)
        return sorted(name_set)

    def transition_stage(
        self,
        name: str,
        version: int,
        stage: str,
    ) -> None:
        """Transition model stage.

        Note: transition_model_version_stage is deprecated in MLflow 2.9+ and
        will be removed in a future version. This method is kept for API
        compatibility; it can be migrated to the set_model_version_tag approach later.

        Args:
            name: Model name.
            version: Version number.
            stage: Target stage (Staging/Production/Archived).
        """
        # TODO: MLflow 2.9+ deprecated; migrate to set_model_version_tag
        self._client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
        )
        logger.info("Model '%s' v%d transitioned to '%s'", name, version, stage)

    def compare_models(
        self,
        name: str,
        versions: list[int],
        metric: str = "loss",
    ) -> dict[int, float]:
        """Compare model metrics across versions.

        Queries run metrics per version. When run_id is empty (MLflow
        server/client compatibility issue), attempts to extract run_id
        from the source URI.

        Args:
            name: Model name.
            versions: List of versions to compare.
            metric: Metric name for comparison.

        Returns:
            Mapping from version to metric value.
        """
        results: dict[int, float] = {}

        for version in versions:
            try:
                mv = self._client.get_model_version(name=name, version=str(version))
                # Prefer run_id, fall back to extracting from source URI
                run_id = mv.run_id
                if not run_id:
                    match = _RUNS_URI_RE.match(mv.source)
                    if match:
                        run_id = match.group(1)
                if not run_id:
                    logger.warning("Skipping v%d: no run_id available", version)
                    continue

                run = self._client.get_run(run_id)
                if metric in run.data.metrics:
                    results[version] = run.data.metrics[metric]
            except Exception as e:
                logger.warning("Failed to get metrics for v%d: %s", version, e)

        return results

    def delete_model(self, name: str) -> None:
        """Delete a registered model (including all versions).

        Args:
            name: Model name.
        """
        self._client.delete_registered_model(name=name)
        logger.info("Model '%s' deleted.", name)

    def delete_model_version(self, name: str, version: int) -> None:
        """Delete a specific model version.

        Args:
            name: Model name.
            version: Version number.
        """
        self._client.delete_model_version(name=name, version=str(version))
        logger.info("Model '%s' v%d deleted.", name, version)
