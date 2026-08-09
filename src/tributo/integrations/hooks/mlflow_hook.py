"""MLflow adapter for committed bundle publication events."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, ClassVar
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tributo.exporting.events import OperationEvent
from tributo.exporting.hooks import ArtifactAccessor, HookOutcome
from tributo.exporting.models import HookStatus
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)

_TRACKING_URI_LOCK = threading.RLock()
_IDEMPOTENCY_TAG = "tributo.idempotency_key"
_SENSITIVE_TAG_WORDS = frozenset(
    {"accesskey", "credential", "credentials", "password", "secret", "token"}
)
_TERMINAL_MLFLOW_ERROR_CODES = frozenset(
    {
        "BAD_REQUEST",
        "CUSTOMER_UNAUTHORIZED",
        "ENDPOINT_NOT_FOUND",
        "FEATURE_DISABLED",
        "INVALID_PARAMETER_VALUE",
        "INVALID_STATE",
        "INVALID_STATE_TRANSITION",
        "MALFORMED_REQUEST",
        "NOT_IMPLEMENTED",
        "PERMISSION_DENIED",
        "UNAUTHENTICATED",
    }
)


def _is_local_bundle_uri(canonical_uri: str) -> bool:
    """Return whether MLflow can upload the committed bundle as a local tree."""
    return urlsplit(canonical_uri).scheme in ("", "file")


@contextmanager
def _tracking_uri_scope(mlflow_module: Any, tracking_uri: str) -> Iterator[None]:
    """Resolve MLflow 2.x proxy artifacts with a serialized, restored URI."""
    with _TRACKING_URI_LOCK:
        previous = mlflow_module.get_tracking_uri()
        try:
            if previous != tracking_uri:
                mlflow_module.set_tracking_uri(tracking_uri)
            yield
        finally:
            if previous != tracking_uri:
                mlflow_module.set_tracking_uri(previous)


@PublicAPI(stability="beta")
class MLflowHookOptions(BaseModel):
    """Validated target selection for the MLflow publication adapter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tracking_uri: str | None = Field(default=None, min_length=1)
    run_id: str | None = Field(default=None, min_length=1)
    experiment_name: str | None = Field(default=None, min_length=1)
    run_name: str | None = Field(default=None, min_length=1)
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("tracking_uri")
    @classmethod
    def _reject_credentials_in_uri(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("tracking_uri must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("tracking_uri must not contain a query or fragment")
        return value

    @field_validator("tags")
    @classmethod
    def _reject_sensitive_tags(cls, tags: dict[str, str]) -> dict[str, str]:
        for name in tags:
            separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
            words = [
                word for word in re.split(r"[^a-z0-9]+", separated.lower()) if word
            ]
            collapsed = "".join(words)
            if set(words) & _SENSITIVE_TAG_WORDS or collapsed in _SENSITIVE_TAG_WORDS:
                raise ValueError(f"sensitive MLflow tag name is not allowed: {name!r}")
        return tags

    @model_validator(mode="after")
    def _validate_target(self) -> MLflowHookOptions:
        if self.run_id is not None:
            if self.experiment_name is not None or self.run_name is not None:
                raise ValueError(
                    "experiment_name and run_name must be omitted when run_id is set"
                )
        elif self.experiment_name is None:
            raise ValueError("experiment_name is required when run_id is not set")
        return self


@PublicAPI(stability="beta")
class MLflowPostPublishHook:
    """Record a committed Tributo bundle in one MLflow tracking run."""

    api_version: ClassVar[int] = 1
    hook_id: ClassVar[str] = "mlflow-log-artifacts-v1"
    options_model: ClassVar[type[BaseModel]] = MLflowHookOptions

    def deliver(
        self,
        event: OperationEvent,
        artifacts: ArtifactAccessor,
        options: BaseModel,
    ) -> HookOutcome:
        """Record verified bundle provenance and available artifacts in MLflow."""
        if not isinstance(options, MLflowHookOptions):
            return HookOutcome(
                status=HookStatus.TERMINAL_FAILED,
                error_code="invalid_options_type",
                error_summary="MLflow adapter received unvalidated options",
            )
        try:
            import mlflow
            from mlflow import MlflowClient
            from mlflow.exceptions import MlflowException
        except ImportError:
            return HookOutcome(
                status=HookStatus.TERMINAL_FAILED,
                error_code="mlflow_not_installed",
                error_summary=(
                    "MLflow hook is configured but the 'registry' extra is not installed"
                ),
            )

        key = self.idempotency_key(event, options)
        effective_tracking_uri = options.tracking_uri or mlflow.get_tracking_uri()
        client = MlflowClient(tracking_uri=effective_tracking_uri)
        owns_run = options.run_id is None
        run_id: str | None = None
        selected_run: Any | None = None

        def mark_owned_run_failed() -> None:
            if not owns_run or run_id is None:
                return
            try:
                client.set_terminated(run_id, status="FAILED")
            except Exception:
                logger.debug("Failed to mark MLflow run as failed")

        try:
            manifest = artifacts.read_manifest()
            if options.run_id is not None:
                try:
                    run = client.get_run(options.run_id)
                except MlflowException as exc:
                    if getattr(exc, "error_code", None) == "RESOURCE_DOES_NOT_EXIST":
                        return HookOutcome(
                            status=HookStatus.TERMINAL_FAILED,
                            error_code="mlflow_run_not_found",
                            error_summary="The configured MLflow run_id does not exist",
                        )
                    raise
                run_id = run.info.run_id
                selected_run = run
            else:
                assert options.experiment_name is not None
                experiment = client.get_experiment_by_name(options.experiment_name)
                if experiment is None:
                    try:
                        experiment_id = client.create_experiment(
                            options.experiment_name
                        )
                    except MlflowException:
                        experiment = client.get_experiment_by_name(
                            options.experiment_name
                        )
                        if experiment is None:
                            raise
                        experiment_id = experiment.experiment_id
                else:
                    experiment_id = experiment.experiment_id

                matches = client.search_runs(
                    [experiment_id],
                    filter_string=f"tags.`{_IDEMPOTENCY_TAG}` = '{key}'",
                    max_results=3,
                )
                if len(matches) > 1:
                    return HookOutcome(
                        status=HookStatus.TERMINAL_FAILED,
                        error_code="ambiguous_mlflow_run",
                        error_summary=(
                            "Multiple MLflow runs have the publication idempotency tag"
                        ),
                    )
                if matches:
                    selected_run = matches[0]
                    run_id = selected_run.info.run_id
                else:
                    run = client.create_run(
                        experiment_id,
                        run_name=options.run_name,
                        tags={_IDEMPOTENCY_TAG: key},
                    )
                    run_id = run.info.run_id
                    selected_run = run

            assert run_id is not None
            tags = {
                **options.tags,
                _IDEMPOTENCY_TAG: key,
                "tributo.bundle_id": event.bundle_id,
                "tributo.bundle_uri": event.canonical_uri,
                "tributo.event_id": event.event_id,
                "tributo.event_schema_version": str(event.schema_version),
                "tributo.hook_id": self.hook_id,
                "tributo.manifest_sha256": event.manifest_sha256,
                "tributo.version": manifest.tributo_version,
            }
            if event.source_kind:
                tags["tributo.source_kind"] = event.source_kind
            for name, value in event.correlation_ids.items():
                tags[f"tributo.{name}"] = value

            run_data = getattr(selected_run, "data", None)
            existing_tags = getattr(run_data, "tags", {}) or {}
            existing_params = getattr(run_data, "params", {}) or {}
            identity = {
                _IDEMPOTENCY_TAG: key,
                "tributo.bundle_id": event.bundle_id,
                "tributo.manifest_sha256": event.manifest_sha256,
            }
            for name, expected in identity.items():
                existing = existing_tags.get(name)
                if existing is not None and existing != expected:
                    return HookOutcome(
                        status=HookStatus.TERMINAL_FAILED,
                        error_code="mlflow_run_identity_conflict",
                        error_summary=(
                            "The selected MLflow run already belongs to a different "
                            "bundle delivery"
                        ),
                        external_references={"mlflow_run_id": run_id},
                    )
            for name in ("tributo.bundle_id", "tributo.manifest_sha256"):
                existing = existing_params.get(name)
                if existing is not None and existing != identity[name]:
                    return HookOutcome(
                        status=HookStatus.TERMINAL_FAILED,
                        error_code="mlflow_run_identity_conflict",
                        error_summary=(
                            "The selected MLflow run already contains immutable "
                            "provenance for a different bundle"
                        ),
                        external_references={"mlflow_run_id": run_id},
                    )

            for name, value in tags.items():
                client.set_tag(run_id, name, value)
            if owns_run:
                client.log_param(run_id, "tributo.bundle_id", event.bundle_id)
                client.log_param(
                    run_id, "tributo.manifest_sha256", event.manifest_sha256
                )

            # MLflow 2.x resolves mlflow-artifacts:// repositories through the
            # process tracking URI even when MlflowClient received an explicit
            # URI. Keep that compatibility scope short, serialized, and restored.
            with _tracking_uri_scope(mlflow, effective_tracking_uri):
                if _is_local_bundle_uri(event.canonical_uri):
                    with artifacts.materialize_bundle() as bundle_root:
                        client.log_artifacts(
                            run_id, str(bundle_root), artifact_path="bundle"
                        )
                else:
                    with artifacts.materialize_manifest() as manifest_path:
                        client.log_artifact(
                            run_id, str(manifest_path), artifact_path="bundle"
                        )
            if owns_run:
                client.set_terminated(run_id, status="FINISHED")

            logger.info("Logged bundle %s into MLflow run %s", event.bundle_id, run_id)
            return HookOutcome(
                status=HookStatus.SUCCEEDED,
                external_references={"mlflow_run_id": run_id},
            )
        except MlflowException as exc:
            mark_owned_run_failed()
            logger.warning("MLflow delivery failed for event %s", event.event_id)
            mlflow_error_code = getattr(exc, "error_code", None)
            terminal = mlflow_error_code in _TERMINAL_MLFLOW_ERROR_CODES
            return HookOutcome(
                status=(
                    HookStatus.TERMINAL_FAILED
                    if terminal
                    else HookStatus.RETRYABLE_FAILED
                ),
                error_code=(
                    "mlflow_permanent_error" if terminal else "mlflow_operation_failed"
                ),
                error_summary=(
                    f"MLflow rejected the tracking operation ({mlflow_error_code})"
                    if terminal
                    else "MLflow tracking operation failed"
                ),
                external_references=(
                    {"mlflow_run_id": run_id} if run_id is not None else {}
                ),
            )
        except OSError:
            mark_owned_run_failed()
            logger.warning("Bundle verification failed for event %s", event.event_id)
            return HookOutcome(
                status=HookStatus.RETRYABLE_FAILED,
                error_code="bundle_materialization_io_failed",
                error_summary="The committed bundle could not be materialized",
                external_references=(
                    {"mlflow_run_id": run_id} if run_id is not None else {}
                ),
            )
        except ValueError:
            mark_owned_run_failed()
            logger.warning("Bundle verification failed for event %s", event.event_id)
            return HookOutcome(
                status=HookStatus.TERMINAL_FAILED,
                error_code="bundle_integrity_failed",
                error_summary="The committed bundle failed integrity verification",
                external_references=(
                    {"mlflow_run_id": run_id} if run_id is not None else {}
                ),
            )

    def idempotency_key(self, event: OperationEvent, options: BaseModel) -> str:
        """Hash the publication fact and sanitized MLflow target config."""
        if not isinstance(options, MLflowHookOptions):
            raise TypeError("MLflowHookOptions are required")
        payload: dict[str, Any] = {
            "hook_id": self.hook_id,
            "canonical_uri": event.canonical_uri,
            "manifest_sha256": event.manifest_sha256,
            "target": options.model_dump(mode="json"),
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
